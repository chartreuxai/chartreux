from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from chartreux.core.dispatch.route_models import ReplacementCapability
from chartreux.core.launch_types import LaunchConfig
from chartreux.core.recovery_models import PackageTier

#: The persisted policy-storage version written by this harness (design §6,
#: §10; package 10). Version 1 records carry no frozen policy bundle — the
#: snapshot held policy/model bindings but not resource bodies — and cannot
#: be resumed: the actionable failure tells the user to start a new session.
POLICY_STORAGE_VERSION: Literal[2] = 2


class LaunchPersonaV1(BaseModel):
    """Legacy launch persona, keyed on the retired ``system_prompt_id`` axis.

    Retained so persisted envelopes written before the identity re-key still
    PARSE (design §7 parameter-surface successors); new envelopes write
    :class:`LaunchPersonaV2`. Package 16a retires the legacy format: resume
    rejects a V1 persona with the actionable "start a new session" failure —
    there is no migration path for the old persisted format.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt_id: str
    instructions: str | None


class LaunchPersonaV2(BaseModel):
    """Launch persona re-keyed on the identity revision (design §7).

    The persona is the conversation's bound identity: ``identity`` is the
    resolved identity reference the launch committed (the same binding
    admission validated and the runtime enforces), and ``identity_revision``
    is the content revision of the identity resource at bind time. Resume
    re-reads this binding instead of re-deriving identity from the live
    profile, so a profile edited between sessions cannot re-posture a
    resumed child; a disagreement with the persisted policy bundle is
    rejected at resume admission.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[2] = 2
    identity: str
    identity_revision: str
    instructions: str | None = None


class CommittedModelIdentity(BaseModel):
    """Concrete catalog identity fixed when a conversation is assigned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str
    provider: str
    wire_name: str
    catalog_revision: str
    thinking: str | None = None


class LaunchMetadataV1(BaseModel):
    """Legacy versioned launch state (the pre-identity-revision envelope).

    Retained so persisted envelopes written before the identity re-key still
    PARSE; package 16a retires the legacy format: resume rejects a V1
    envelope with the actionable "start a new session" failure.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    profile: str
    overrides: LaunchConfig
    persona: LaunchPersonaV1

    @field_serializer("overrides")
    def _serialize_overrides(self, value: LaunchConfig) -> dict[str, Any]:
        return value.model_dump(exclude_unset=True, mode="json")


class LaunchMetadataV2(BaseModel):
    """Committed launch state with an assignment-time concrete model identity."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[2]
    overrides: LaunchConfig = Field(default_factory=LaunchConfig)
    # The persona is keyed on the identity revision from this version on
    # (LaunchPersonaV2); the V1 form parses for legacy envelopes only.
    persona: LaunchPersonaV1 | LaunchPersonaV2
    committed_model: CommittedModelIdentity
    # WP5 lineage pair + package tier: the failure-time replacement claim,
    # committed with the launch state so a resumed or restarted child keeps its
    # work-package identity (resume revalidates it rather than re-resolving).
    work_package_id: str | None = None
    predecessor_agent_id: str | None = None
    package_tier: PackageTier | None = None
    # The resolved purpose label and model-level replacement capability the
    # launch committed: resume restores them instead of re-deriving defaults,
    # so a disk-resumed child keeps describing the package's true
    # purpose/tier/capability.
    purpose: str | None = None
    replacement_capability: ReplacementCapability | None = None

    @model_validator(mode="after")
    def _lift_lineage_from_overrides(self) -> LaunchMetadataV2:
        """Lift unset lineage fields from the committed overrides.

        Both construction sites (the app server's launch metadata and the
        loop's own re-install) build from the semantic overrides; the claim
        fields ride there, so an explicitly unset first-class field is
        lifted from them instead of silently dropped.
        """
        overrides = self.overrides
        if self.work_package_id is None:
            self.work_package_id = overrides.work_package_id
        if self.predecessor_agent_id is None:
            self.predecessor_agent_id = overrides.predecessor_agent_id
        if self.package_tier is None:
            self.package_tier = overrides.package_tier
        return self

    @field_serializer("overrides")
    def _serialize_overrides(self, value: LaunchConfig) -> dict[str, Any]:
        return value.model_dump(exclude_unset=True, mode="json")


LaunchMetadata = LaunchMetadataV1 | LaunchMetadataV2


class ScheduledLoop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    interval_seconds: int
    prompt: str
    next_fire_at: float
    created_at: float


class AgentStats(BaseModel):
    steps: int = 0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_cached_tokens: int = 0
    tool_calls_agreed: int = 0
    tool_calls_rejected: int = 0
    tool_calls_hook_denied: int = 0
    tool_calls_failed: int = 0
    tool_calls_succeeded: int = 0

    known_cost_total: float = 0.0
    has_unknown_cost: bool = False

    context_tokens: int = 0

    last_turn_prompt_tokens: int = 0
    last_turn_completion_tokens: int = 0
    last_turn_cached_tokens: int = 0
    last_turn_duration: float = 0.0
    tokens_per_second: float = 0.0

    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    cached_input_price_per_million: float | None = None

    _listeners: dict[str, Callable[[AgentStats], None]] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def _mark_token_only_legacy_stats_incomplete(cls, value: Any) -> Any:
        if isinstance(value, dict) and value and "known_cost_total" not in value:
            value = dict(value)
            value.setdefault("known_cost_total", 0.0)
            value["has_unknown_cost"] = True
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in self._listeners:
            self._listeners[name](self)

    def trigger_listeners(self) -> None:
        for listener in self._listeners.values():
            listener(self)

    def add_listener(
        self, attr_name: str, listener: Callable[[AgentStats], None]
    ) -> None:
        self._listeners[attr_name] = listener

    @staticmethod
    def create_fresh(previous: AgentStats) -> AgentStats:
        fresh = AgentStats()
        fresh._listeners = previous._listeners.copy()
        return fresh

    @computed_field
    @property
    def session_total_llm_tokens(self) -> int:
        return self.session_prompt_tokens + self.session_completion_tokens

    @computed_field
    @property
    def last_turn_total_tokens(self) -> int:
        return self.last_turn_prompt_tokens + self.last_turn_completion_tokens

    @computed_field
    @property
    def session_cost(self) -> float | None:
        """Exact ledger total when complete, otherwise no complete cost is available."""
        if self.has_unknown_cost:
            return None
        return self.known_cost_total

    def update_pricing(
        self,
        input_price: float,
        output_price: float,
        cached_input_price: float | None = None,
    ) -> None:
        """Update pricing info when model changes.

        NOTE: session_cost will be recalculated using new pricing for all
        accumulated tokens. This is a known approximation when models change.
        This should not be a big issue, pricing is only used for max_price which is in
        programmatic mode, so user should not update models there.
        """
        self.input_price_per_million = input_price
        self.output_price_per_million = output_price
        self.cached_input_price_per_million = cached_input_price

    def reset_context_state(self) -> None:
        """Reset context-related fields while preserving cumulative session stats.

        Used after config reload or similar operations where the context
        changes but we want to preserve session totals.
        """
        self.context_tokens = 0
        self.last_turn_prompt_tokens = 0
        self.last_turn_completion_tokens = 0
        self.last_turn_cached_tokens = 0
        self.last_turn_duration = 0.0
        self.tokens_per_second = 0.0


class SessionInfo(BaseModel):
    session_id: str
    start_time: str
    message_count: int
    stats: AgentStats
    save_dir: str


class ChildSessionLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    tool_call_id: str
    agent: str
    relative_path: str | None = None


# Session state rather than a message, because the worktree is created before
# the session has a first turn: there is no message it could belong to, and the
# transcript is rebuilt on every resume from what was persisted here.
class WorktreeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    name: str
    branch: str
    path: str
    created_at: int


class SessionMetadata(BaseModel):
    session_id: str
    parent_session_id: str | None = None
    start_time: str
    end_time: str | None
    git_commit: str | None
    git_branch: str | None
    environment: dict[str, str | None]
    # Where the session began. ``environment.working_directory`` follows it as
    # it moves, so between them the record names both the directory the user
    # started in and the one the session is working in. Neither on its own can
    # do that, which is the whole reason this field exists.
    origin_directory: str | None = None
    username: str
    child_sessions: list[ChildSessionLink] = Field(default_factory=list)
    loops: list[ScheduledLoop] = Field(default_factory=list)
    title: str | None = None
    title_source: Literal["auto", "manual"] = "auto"
    # Session-scoped config snapshot. New sessions pin ``active_model`` to its
    # resolved alias before their first user turn; older snapshots remain valid.
    config: dict[str, JsonValue] | None = None
    import_provenance: dict[str, JsonValue] | None = None
    created_worktree: WorktreeContext | None = None
    launch_config: LaunchMetadata | None = None
    # The conversation's frozen policy record (design §6, §10; package 10):
    # the serialized package-7 policy bundle, carried inline so resume works
    # after the source resource files are edited or removed. Written together
    # with ``policy_storage_version``; a record without them is a legacy
    # (version-1) record and cannot be resumed.
    policy_bundle: dict[str, JsonValue] | None = None
    policy_storage_version: Literal[2] | None = None

    @field_validator("policy_bundle", mode="before")
    @classmethod
    def _validate_policy_bundle(cls, value: Any) -> Any:
        """Structurally validate the persisted bundle; integrity at activation.

        The bundle is the persisted policy record of the conversation, so a
        corrupt record fails at load with the actionable storage error rather
        than deep inside a resume. Hash integrity and pin verification happen
        at ACTIVATION (``activate_bundle``), before any client or resource is
        constructed.
        """
        if value is None:
            return value
        from chartreux.core.dispatch.session import (
            SessionPolicyError,
            validate_stored_policy_bundle,
        )

        if not isinstance(value, dict):
            raise SessionPolicyError(
                "The persisted policy bundle is corrupt: expected a JSON object; "
                "the conversation cannot be resumed — start a new session"
            )
        try:
            validate_stored_policy_bundle(value)
        except SessionPolicyError:
            raise
        except ValueError as exc:
            raise SessionPolicyError(
                f"The persisted policy bundle is unreadable: {exc}; the "
                "conversation cannot be resumed — start a new session"
            ) from exc
        return value

    @model_serializer(mode="wrap")
    def _omit_absent_launch_config(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.launch_config is None:
            data.pop("launch_config", None)
        return data

    @field_validator("launch_config", mode="before")
    @classmethod
    def _reject_explicit_null_launch_config(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("launch_config cannot be null when present")
        return value
