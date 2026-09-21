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

from chartreux.core.launch_types import LaunchConfig


class LaunchPersonaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt_id: str
    instructions: str | None


class CommittedModelIdentity(BaseModel):
    """Concrete catalog identity fixed when a conversation is assigned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str
    provider: str
    wire_name: str
    catalog_revision: str


class LaunchMetadataV1(BaseModel):
    """Versioned, committed semantic launch state for a child session."""

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
    profile: str | None = None
    overrides: LaunchConfig = Field(default_factory=LaunchConfig)
    persona: LaunchPersonaV1
    committed_model: CommittedModelIdentity

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
