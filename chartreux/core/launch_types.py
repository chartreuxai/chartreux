from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chartreux.core.agents.identities import AgentIdentity, ToolPosture
from chartreux.core.config.models import ThinkingLevel
from chartreux.core.dispatch.route_models import ReplacementCapability
from chartreux.core.recovery_models import PackageTier
from chartreux.core.tools.models import ToolPermission

if TYPE_CHECKING:
    # Imported lazily: session_types imports LaunchConfig from this module, and
    # policy.composition is a heavy import — both are annotation-only here.
    from chartreux.core.policy.composition import ExecutionContext
    from chartreux.core.session_types import CommittedModelIdentity


class LaunchToolOverride(BaseModel):
    """The only tool settings accepted for a subagent launch."""

    model_config = ConfigDict(extra="forbid")

    permission: ToolPermission | None = None
    allowlist: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _omit_explicit_nulls(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if item is not None}
        return value


class LaunchConfig(BaseModel):
    """Semantic launch choices, preserving supplied fields for accumulated state."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(
        default=None, description="canonical model name or `@role`"
    )
    instructions: str | None = None
    thinking: ThinkingLevel | None = None
    enabled_tools: list[str] | None = None
    disabled_tools: list[str] | None = None
    tools: dict[str, LaunchToolOverride] | None = None
    # WP5 failure-time replacement claim (handoff §2.2/Q8; package-specific
    # fallback constraints in dispatch metadata). A cross-model replacement
    # gets a NEW agent_id carrying the predecessor/package lineage; the claim
    # is validated at launch admission by the replacement policy.
    work_package_id: str | None = Field(
        default=None,
        description=(
            "Orchestrator work-package lineage for this launch; carried into "
            "the run's failure envelope so recovery counters span replacements."
        ),
    )
    predecessor_agent_id: str | None = Field(
        default=None,
        description=(
            "The failed agent this launch replaces. A replacement launch gets "
            "a NEW agent_id; the predecessor's recovery is disabled before "
            "the handoff transfers package write ownership."
        ),
    )
    package_tier: PackageTier | None = Field(
        default=None,
        description=(
            "The work package's tier for the failure-time replacement policy: "
            "mechanical-tier models are rejected as replacements for "
            "substantive packages."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _omit_explicit_nulls(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if "system_prompt_id" in value:
                raise ValueError(
                    "system_prompt_id is retired; the launch persona is the "
                    "resolved identity — supply the task tool's identity "
                    "field instead of a per-launch prompt template"
                )
            return {key: item for key, item in value.items() if item is not None}
        return value


@dataclass(frozen=True, slots=True)
class ResolvedLaunchAuthority:
    """The authority axes of a resolved launch (design §8; interfaces.md).

    Resolved BEFORE any resource construction: the child identity's personal
    and delegation posture selectors (``None`` = unrestricted on that axis — a
    custom identity carries no harness-known posture, and its delegation set
    defaults to the full granted ceiling, design decision 8) plus the parent's
    delegation ceiling the launch is authorized against. The admission
    ceilings (``_validate_tools`` and the ToolManager posture enforcement)
    consume these selectors, so resolving them here keeps launch resolution
    resource-free.
    """

    personal_posture: ToolPosture | None
    """The child identity's personal tool set (None = unrestricted)."""
    delegation_posture: ToolPosture | None
    """The child identity's delegation tool set (None = full granted ceiling)."""
    authorized_tool_names: frozenset[str]
    """The parent's delegation ceiling; granted authority never widens."""


@dataclass(frozen=True, slots=True)
class ResolvedIdentityLaunch:
    """The resource-free resolved launch (interfaces.md "Resolved launches").

    The complete resolution of one canonical task launch — identity (with its
    content revision), concrete model/deployment, effective thinking,
    purpose/seat binding, authority axes, and the effective package tier —
    computed BEFORE any backend, MCP client, child log, handle, or reservation
    is constructed. Package 11b/11c wire this contract into the task tool and
    the launch entrypoints.
    """

    identity: str
    """The validated child identity reference (bare built-in, ``builtin:``- or
    ``user:``-qualified, or bare custom prompt id)."""
    identity_revision: str
    """The content revision of the resolved identity body (manifest-compatible)."""
    identity_facts: AgentIdentity | None
    """The typed built-in identity facts; None for a custom identity."""
    execution_context: ExecutionContext
    """Always ``bounded_child``: a child never restarts the parent's ceremony."""
    workflow: str | None
    """None: children inherit NO workflow body (design §7)."""
    strategy: str | None
    """None: children inherit NO orchestration-strategy body (design §7)."""
    model_expression: str
    """The resolved model expression (explicit, or the route/session binding)."""
    committed_model: CommittedModelIdentity
    """The concrete deployment committed at launch (base, provider, wire, revision)."""
    effective_thinking: ThinkingLevel | None
    """The effective thinking level resolved for the concrete deployment."""
    purpose: str | None
    """The purpose label whose route was resolved; None for a direct launch."""
    seat: str | None
    """The selected route seat; None for single-seat and direct launches."""
    overlay: str | None
    """The route target's instruction overlay (e.g. ``review-execution``)."""
    overlay_body: str | None
    """The resolved overlay body content; None when the route names no overlay.
    The overlay rides the persona instructions (design §9: the execution seat
    is coder plus a reviewer-derived overlay) — it never mutates the identity
    binding or its revision."""
    effective_tier: PackageTier
    """The effective package tier: the route target's explicit tier, or the
    identity/purpose default for a direct unrouted launch."""
    replacement_capability: ReplacementCapability
    """The model-level replacement-capability marking — a SEPARATE axis from the
    package tier (interfaces.md settled decision; recovery_policy)."""
    authority: ResolvedLaunchAuthority
    """The resolved authority axes (postures + the parent's delegation ceiling)."""
    semantic_overrides: LaunchConfig
    """The preserved per-launch semantic config: additive instructions, tool
    overrides, thinking, and lineage fields are unchanged."""
    background: bool
    """The unchanged lifecycle field."""
    retain: bool | None
    """The unchanged lifecycle field."""
    agent_id: str | None
    """The unchanged lifecycle field (a retained-instance handle, never a new one)."""
    replace_run: bool
    """The unchanged lifecycle field."""
    task_summary: str | None
    """The unchanged lifecycle field."""

    @property
    def instructions(self) -> str | None:
        """The preserved additive per-launch instruction overlay."""
        return self.semantic_overrides.instructions

    @property
    def work_package_id(self) -> str | None:
        """The preserved lineage field (WP5 replacement-claim metadata)."""
        return self.semantic_overrides.work_package_id

    @property
    def predecessor_agent_id(self) -> str | None:
        """The preserved lineage field (WP5 replacement-claim metadata)."""
        return self.semantic_overrides.predecessor_agent_id

    @property
    def package_tier_claim(self) -> PackageTier | None:
        """The preserved lineage field (the orchestrator-declared package tier)."""
        return self.semantic_overrides.package_tier

    @property
    def base_model(self) -> str:
        """The canonical base model of the committed deployment."""
        return self.committed_model.base_model

    @property
    def provider(self) -> str:
        """The provider of the committed deployment."""
        return self.committed_model.provider

    @property
    def wire_name(self) -> str:
        """The wire name of the committed deployment."""
        return self.committed_model.wire_name
