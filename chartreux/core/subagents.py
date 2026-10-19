from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
import enum
import json
import re
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chartreux.core.agents.models import BUILTIN_SUBAGENTS
from chartreux.core.events import (
    AssistantEvent,
    BaseEvent,
    ToolResultEvent,
    ToolStreamEvent,
)
from chartreux.core.launch_types import LaunchConfig, LaunchToolOverride
from chartreux.core.recovery_models import (
    FailureEnvelope,
    RecoveryEligibility,
    RecoveryReference,
)
from chartreux.core.recovery_policy import ReplacementDecision
from chartreux.core.tools.ui import ToolUIDataAdapter

__all__ = [
    "EffectVerification",
    "LaunchConfig",
    "LaunchToolOverride",
    "ReconcileOutcome",
    "ReconcileResult",
    "RecoveryReference",
    "ResumeAgentResult",
    "ResumeOutcome",
]

if TYPE_CHECKING:
    from chartreux.core.tools.base import InvokeContext


def normalize_task_summary(
    value: str | None, *, fallback: str | None = None
) -> str | None:
    """Normalize a supplied task summary or its fallback preview."""
    if value is None:
        value = fallback
    if value is None:
        return None
    return " ".join(value.split())[:240]


class TaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(description="The task for the agent to perform")
    task_summary: str | None = Field(
        default=None,
        description="Optional short summary of the task for retained-agent status.",
    )
    agent_type: str = Field(
        default="worker",
        description=(
            "Agent type: profile name for a new agent instance (default worker). "
            "To continue a retained instance, use agent_id instead, with background: true."
        ),
    )
    background: bool = Field(
        default=True,
        description=(
            "If true (the default), launch the subagent in the background and "
            "return a launch acknowledgment with handles instead of blocking "
            "until completion. Launch in the background even when the result "
            "is a dependency, then collect it with wait_for_agent; set false "
            "only when blocking is truly required."
        ),
    )
    agent_id: str | None = Field(
        default=None,
        description=(
            "Retained agent instance handle to continue with background: true; "
            "reuse inherits the instance's committed retention policy unless "
            "retain is passed. If omitted, create a new instance using the "
            "agent_type profile name."
        ),
    )
    replace_run: bool = Field(
        default=False,
        description="Stop busy background work before reusing agent_id in the same conversation.",
    )
    retain: bool | None = Field(
        default=None,
        strict=True,
        description=(
            "Retain the agent instance after this run completes so it stays "
            "resident for reuse with agent_id; by default a background agent "
            "closes on completion and its result stays retrievable. Precedence "
            "for a new launch: this argument, then the profile's retain "
            "setting, then the [subagents].retain_agents default. On reuse, "
            "omitting retain keeps the instance's committed policy; "
            "retain=false is a final continuation that closes afterward. "
            "Only valid for background launches."
        ),
    )
    config: LaunchConfig | None = Field(
        default=None,
        description="Optional semantic launch configuration overrides for this child.",
    )

    @model_validator(mode="before")
    @classmethod
    def _omit_null_config(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        fan_out = value.get("fan_out", False)
        if fan_out is not False and fan_out is not None:
            raise ValueError(
                "Roles are single presets. Launch separate tasks with explicit "
                "presets/models for multiple agents."
            )
        value = {key: item for key, item in value.items() if key != "fan_out"}

        config = value.get("config")
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError as error:
                raise ValueError("config must be a valid JSON object") from error
            if config is not None and not isinstance(config, dict):
                raise ValueError("config JSON must decode to an object")

        if config is None:
            return {key: item for key, item in value.items() if key != "config"}
        return {**value, "config": config}

    @model_validator(mode="after")
    def _reject_instance_handle_as_type(self) -> TaskArgs:
        if re.fullmatch(r"agent-\d+", self.agent_type):
            raise ValueError(
                "agent_type takes a profile name; to continue an existing agent "
                "instance, pass its handle as agent_id (with background: true)"
            )
        return self

    @field_validator("task_summary")
    @classmethod
    def _normalize_task_summary(cls, value: str | None) -> str | None:
        return normalize_task_summary(value)


class LaunchOutcome(enum.StrEnum):
    LAUNCHED = "launched"
    ALREADY_STOPPING = "already_stopping"
    ALREADY_FINISHING = "already_finishing"
    REJECTED_RESERVATION = "rejected_reservation"


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str = Field(description="The accumulated response from the subagent")
    turns_used: int = Field(description="Number of turns the subagent used")
    completed: bool = Field(description="Whether the task completed normally")
    launch_outcome: LaunchOutcome | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    status: Literal["launched"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Launch state for a background launch acknowledgment; None for task results.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Stable agent handle. Populated for background launches; None for foreground.",
    )
    run_id: str | None = Field(
        default=None,
        description="Unique run identifier for this invocation. Populated for background launches; None for foreground.",
    )
    name: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Human-facing agent name ({profile}-{model}-{number}) for background "
            "launches; None for foreground results. Display-only: agent_id "
            "remains the only handle."
        ),
    )
    stop_reason: RunStopReason | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additive completion metadata, including provider failover visibility.",
    )
    run_status: RunStatus | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Terminal run status for background runs; None for launch "
            "acknowledgments and foreground results."
        ),
    )
    failure: FailureEnvelope | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Public typed failure envelope (bounded, redacted); set only for "
            "failed background runs. Full diagnostics stay private."
        ),
    )
    recovery: RecoveryReference | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Recovery-chain linkage for the failed run; None when there is "
            "no failure envelope."
        ),
    )
    # WP5 lineage pair: the orchestrator work-package identity and the failed
    # agent this run's agent replaced (a replacement gets a NEW agent_id with
    # predecessor/package lineage). Populated from the launch claim.
    work_package_id: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Orchestrator work-package lineage; None when not claimed.",
    )
    predecessor_agent_id: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The failed agent this run's agent replaced; None for a fresh agent.",
    )
    # WP5 failure-time replacement policy outcome for a claimed replacement
    # launch: the validated decision (with its milestone notice) or the
    # rejection/parking reason. None for unclaimed launches.
    replacement: ReplacementDecision | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "The failure-time replacement policy decision for a claimed "
            "replacement launch; None for unclaimed launches."
        ),
    )


class RunStopReason(enum.StrEnum):
    """Structured run outcome; values match the public app-server contract."""

    USER_CANCELLED = "user_cancelled"
    ORCHESTRATOR_CANCELLED = "orchestrator_cancelled"
    RETASKED = "retasked"
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_UNVERIFIABLE = "budget_unverifiable"
    ERROR = "error"


class CancelOutcome(enum.StrEnum):
    STOP_REQUESTED = "stop_requested"
    ALREADY_STOPPING = "already_stopping"
    ALREADY_FINISHING = "already_finishing"
    NOT_RUNNING = "not_running"
    UNKNOWN_RUN = "unknown_run"
    FORBIDDEN = "forbidden"


class CancelResult(BaseModel):
    """Stop-request disposition, not a claim that execution has stopped.

    run_id is the resolved target (or the explicit unknown target), and may be
    None when there is no run identity. stop_reason is the winning requested
    reason, or the retained terminal reason; refusals need not have a reason.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: CancelOutcome
    run_id: str | None = None
    stop_reason: RunStopReason | None = None


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WaitStatus(enum.StrEnum):
    """Disposition status for a wait_for_agent call (design §6)."""

    COMPLETED = "completed"
    STILL_RUNNING = "still_running"


class WaitForAgentResult(BaseModel):
    """Typed wait disposition for wait_for_agent (design §6).

    ``completed`` carries the finished run's TaskResult; ``still_running``
    carries the wait progress (agent_id, run_id, elapsed_seconds, run_status,
    turns_used) plus guidance instructing the model to yield its turn and
    re-invoke wait_for_agent in a later turn. The variants are mutually
    exclusive: ``completed`` rejects the progress fields and ``still_running``
    rejects ``result``.
    """

    model_config = ConfigDict(extra="forbid")

    status: WaitStatus
    result: TaskResult | None = Field(
        default=None,
        description="The completed run's result; set only when status is completed.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Agent handle waited on; set only when status is still_running.",
    )
    run_id: str | None = Field(
        default=None,
        description="Run handle waited on; set only when status is still_running.",
    )
    elapsed_seconds: float | None = Field(
        default=None,
        description="Seconds waited before the timeout; still_running only.",
    )
    run_status: RunStatus | None = Field(
        default=None, description="Current run status; still_running only."
    )
    turns_used: int | None = Field(
        default=None, description="Turns used by the run so far; still_running only."
    )
    guidance: str | None = Field(
        default=None, description="Guidance for the still_running outcome."
    )

    @model_validator(mode="after")
    def _require_variant_fields(self) -> WaitForAgentResult:
        progress = (
            self.agent_id,
            self.run_id,
            self.elapsed_seconds,
            self.run_status,
            self.turns_used,
            self.guidance,
        )
        if self.status is WaitStatus.COMPLETED:
            if self.result is None:
                raise ValueError("completed wait disposition requires a result")
            if any(field is not None for field in progress):
                raise ValueError(
                    "completed wait disposition cannot carry still_running fields "
                    "(agent_id, run_id, elapsed_seconds, run_status, turns_used, "
                    "guidance)"
                )
            return self
        if self.result is not None:
            raise ValueError("still_running wait disposition cannot carry a result")
        if any(field is None for field in progress):
            raise ValueError(
                "still_running wait disposition requires agent_id, run_id, "
                "elapsed_seconds, run_status, turns_used, and guidance"
            )
        return self


class AgentAvailability(enum.StrEnum):
    """Lifecycle availability of an agent record or terminal metadata.

    ``running``, ``finalizing``, and ``idle`` are resident records.
    ``closed`` is terminal metadata for a normally completed run
    (including failed or cancelled runs) of an unretained agent;
    ``evicted`` is terminal metadata for a retained runtime removed by
    the idle TTL or idle cap. Terminal metadata is bounded (a TTL and a
    count cap) and disappears without expiring retained results.

    WP2 (handoff §1.2/§1.8) adds ``suspended`` and ``restoring``: a
    recoverable failure moves the record to ``suspended`` — an explicit,
    bounded exception BEFORE closure, keeping the live conversation warm
    and the agent explicitly resumable — and an admitted resume moves it
    to ``restoring`` until the successor run starts. Suspension is not
    retention under another name and not resurrection: closed and evicted
    agents remain non-reusable.
    """

    RUNNING = "running"
    FINALIZING = "finalizing"
    IDLE = "idle"
    EVICTED = "evicted"
    CLOSED = "closed"
    SUSPENDED = "suspended"
    RESTORING = "restoring"


class ReleaseAgentOutcome(enum.StrEnum):
    """Outcome of a release request.

    ``RELEASED`` removes a live agent; ``EVICTED`` removes its terminal
    tombstone (``closed`` or ``evicted``). The guarded release path used
    by the agent browser (the ``agents/release`` RPC) additionally reports
    ``CONFLICT`` when the agent is active, reserved for replacement, or
    pinned to a different latest run, and ``UNKNOWN`` when the handle is
    already released or its terminal metadata expired. Release discards
    retained results, expiry markers, and wait leases.
    """

    RELEASED = "released"
    EVICTED = "evicted"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class ResumeOutcome(enum.StrEnum):
    """Typed disposition of an explicit ``resume_agent`` request (WP2).

    One management-port operation serves both the orchestrator's explicit
    requests and (later) the automatic controller, so they cannot create
    duplicate runs. ``scheduled`` admits exactly one successor (a NEW
    run_id on the SAME agent identity); ``already_scheduled`` is the
    single-flight deduplication of a concurrent same-key call;
    ``already_resumed`` reports that the pinned failed run already has an
    admitted successor. ``reconcile_required`` blocks replay while tool
    outcomes are unresolved/unknown; ``expired`` reports a lapsed recovery
    window (an admission stop, never an in-flight kill); ``not_resumable``
    covers non-suspended, prohibited, or budget-exhausted targets;
    ``forbidden`` covers authority failures.
    """

    SCHEDULED = "scheduled"
    ALREADY_SCHEDULED = "already_scheduled"
    ALREADY_RESUMED = "already_resumed"
    RECONCILE_REQUIRED = "reconcile_required"
    EXPIRED = "expired"
    NOT_RESUMABLE = "not_resumable"
    FORBIDDEN = "forbidden"


class ResumeAgentResult(BaseModel):
    """Result of one explicit resume admission (handoff §1.4).

    Nonblocking acceptance: ``scheduled``/``already_scheduled`` carry the
    recovery-operation id and the successor run id published on admission.
    The failed run's result and every wait pinned to it stay immutable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ResumeOutcome
    agent_id: str
    run_id: str | None = Field(
        default=None, description="The failed run the request targeted."
    )
    successor_run_id: str | None = Field(
        default=None,
        description="The admitted successor run; set for scheduled/already_*.",
    )
    recovery_operation_id: str | None = Field(
        default=None, description="Deduplicatable id of the recovery operation."
    )
    retry_at: float | None = Field(
        default=None,
        description=(
            "Epoch seconds at which the successor may start. WP2 admits "
            "immediately (no auto-scheduling), so this is the admission time."
        ),
    )
    eligibility: RecoveryEligibility | None = Field(
        default=None,
        description="The target run's current effective recovery eligibility.",
    )
    reason_codes: list[str] = Field(
        default_factory=list, description="Bounded machine-readable explanation codes."
    )
    reason: str | None = Field(
        default=None, description="Bounded human-readable explanation."
    )
    same_agent_resumes_used: int = Field(
        default=0, description="Same-agent resumes already used in this recovery chain."
    )


class EffectVerification(BaseModel):
    """One externally verified tool effect (``reconcile_result`` input).

    The orchestrator (or a worker it dispatches) verifies unknown tool
    outcomes externally — inspecting git/filesystem/job state — and submits
    the verified dispositions. ``settled`` confirms the effect completed
    consistently; ``unrepeatable_damage`` reports verification showing damage
    that cannot be undone or safely repeated, which flips the barrier to
    prohibited. An unverifiable outcome is simply not submitted: the
    barrier never auto-clears.
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(description="The tool call whose effect was verified.")
    disposition: Literal["settled", "unrepeatable_damage"] = Field(
        description=(
            "settled: the effect completed consistently. "
            "unrepeatable_damage: verification shows damage that cannot be "
            "undone or safely repeated."
        )
    )
    note: str | None = Field(
        default=None, description="Bounded (<=500 chars) redacted verification note."
    )

    @field_validator("note")
    @classmethod
    def _bound_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value[:500]


class ReconcileOutcome(enum.StrEnum):
    """Typed disposition of a ``reconcile_result`` barrier submission (WP2).

    The submission is PINNED to the failed run and its transcript
    boundary: ``stale`` evidence (an earlier run or a different boundary)
    cannot clear a newer barrier, and ``duplicate`` submissions change
    nothing. ``cleared`` flips SUSPENDED[reconcile_required] to eligible;
    ``prohibited`` flips it to prohibited with reason codes (verification
    showed unrepeatable damage). Replacement remains permitted at any
    point; the barrier never auto-clears.
    """

    CLEARED = "cleared"
    PROHIBITED = "prohibited"
    STALE = "stale"
    DUPLICATE = "duplicate"
    NOT_PENDING = "not_pending"
    FORBIDDEN = "forbidden"


class ReconcileResult(BaseModel):
    """Result of one reconcile_result submission (handoff §1.4/§1.8)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ReconcileOutcome
    agent_id: str
    run_id: str | None = Field(
        default=None, description="The failed run the submission targeted."
    )
    transcript_boundary: str | None = Field(
        default=None, description="The transcript boundary the evidence is pinned to."
    )
    eligibility: RecoveryEligibility | None = Field(
        default=None, description="The barrier's eligibility after this submission."
    )
    reason_codes: list[str] = Field(
        default_factory=list, description="Bounded machine-readable explanation codes."
    )
    reason: str | None = Field(
        default=None, description="Bounded human-readable explanation."
    )


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Per-root-generation retention snapshot for background subagents.

    Replaces the legacy ``(idle_ttl_seconds, max_idle_agents)`` 2-tuple and adds
    the generation-snapshotted global retention default. ``idle_ttl_seconds=0``
    disables TTL eviction and ``max_idle_agents=0`` disables the idle cap for
    retained instances. Terminal metadata (tombstones) expires
    ``tombstone_ttl_seconds`` after closure and is capped at ``max_tombstones``
    entries, oldest first; ``tombstone_ttl_seconds=0`` stores no terminal
    metadata.
    """

    idle_ttl_seconds: int
    max_idle_agents: int
    retain_agents: bool
    tombstone_ttl_seconds: int = 900
    max_tombstones: int = 32


@dataclass(frozen=True, slots=True)
class AgentEviction:
    agent_id: str
    run_id: str
    reason: Literal["ttl", "idle_cap"]
    idle_duration_seconds: float
    root_generation: int


@dataclass(slots=True)
class AgentSummary:
    agent_id: str
    profile: str
    availability: AgentAvailability
    current_run_id: str | None
    current_run_status: RunStatus | None
    turns_used: int | None = None
    initial_task_summary: str | None = None
    current_task_summary: str | None = None
    idle_seconds: float | None = None
    run_elapsed_seconds: float | None = None
    latest_run_id: str | None = None
    ttl_remaining_seconds: float | None = None
    effective_model: str | None = None
    """Configured model alias committed for this retained agent."""
    base_model: str | None = None
    """Canonical committed base model for this retained agent."""
    active_provider: str | None = None
    """Provider serving the retained agent's currently committed deployment."""
    effective_thinking: str | None = None
    """Semantic effective thinking level, not provider wire-level encoding."""
    result_expired: bool = False
    """Whether the retained current-run result has expired."""
    last_run_status: RunStatus | None = None
    """Terminal outcome of the most recently finalized run, if any."""
    context_tokens: int | None = None
    context_window: int | None = None
    compacting: bool = False
    stop_reason: RunStopReason | None = None
    """Current/latest run's structured stop reason, retained after finalization."""
    last_failure: FailureEnvelope | None = None
    """Public typed failure envelope of the latest failed run, if any."""
    recovery_chain_id: str | None = None
    """Stable recovery-chain identity across the resumes of one episode (WP2)."""
    same_agent_resumes_used: int | None = None
    """Same-agent resumes used in the chain; None when no chain is active."""
    resumed_from_run_id: str | None = None
    """The failed run the latest run resumed from, if it is a successor."""
    recovery_eligibility: RecoveryEligibility | None = None
    """The current effective recovery eligibility of the suspended barrier.

    WP2: the published envelope stays immutable; a reconcile_result
    submission flips this effective view (cleared -> eligible, verified
    unrepeatable damage -> prohibited). None when no barrier is active.
    """
    work_package_id: str | None = None
    """Orchestrator work-package lineage of the latest run (WP5); None when unclaimed."""
    predecessor_agent_id: str | None = None
    """The failed agent this agent replaced (WP5); None for a fresh agent."""
    name: str = ""
    """Human-facing ``{profile}-{model}-{number}`` name (design §5).

    The server always populates this when projecting summaries; the empty
    string is the legacy-constructor default for older in-process callers.
    """


_AGENT_HANDLE_PREFIX = "agent-"

_AGENT_HANDLE_PATTERN = re.compile(r"agent-\d+")
"""The allocated handle shape (``agent-{sequence}``); derived names never match."""


def _strip_model_namespace(model: str) -> str:
    """Drop a provider namespace prefix (``mistral/zai-glm-5-3`` → ``zai-glm-5-3``)."""
    stripped = model.rpartition("/")[2]
    return stripped or model


def _agent_sequence_number(agent_id: str) -> str:
    """The agent sequence digits of a handle (``agent-15`` → ``15``)."""
    if agent_id.startswith(_AGENT_HANDLE_PREFIX):
        return agent_id[len(_AGENT_HANDLE_PREFIX) :]
    return agent_id


def derive_agent_name(
    agent_id: str,
    profile: str,
    profile_display_label: str | None,
    base_model: str | None,
    effective_model: str | None,
) -> str:
    """Derive the human-facing agent name ``{profile}-{model}-{number}`` (design §5).

    The profile segment is the profile name for builtin profiles and the
    whitespace-collapsed display label for custom profiles, falling back to
    the profile name when no label was captured at admission. The model
    segment is the namespace-stripped canonical ``base_model``, falling back
    to the namespace-stripped ``effective_model``, and is omitted entirely
    when neither is known. The number is the agent sequence digits of the
    handle (``agent-15`` ↔ ``...-15``). The helper is stateless: names are
    derived from accepted task publication state, never persisted.

    The result never parses as a handle (design §5): if it would match the
    ``agent-\\d+`` handle shape — a custom profile or display label of
    exactly ``agent`` with no known model yields ``agent-7`` — the profile
    segment is prefixed to break the shape deterministically.
    """
    if profile_display_label is not None and profile not in BUILTIN_SUBAGENTS:
        profile_segment = "-".join(profile_display_label.split())
    else:
        profile_segment = profile
    model = base_model if base_model is not None else effective_model
    number = _agent_sequence_number(agent_id)
    if model is None:
        name = f"{profile_segment}-{number}"
    else:
        name = f"{profile_segment}-{_strip_model_namespace(model)}-{number}"
    if _AGENT_HANDLE_PATTERN.fullmatch(name) is not None:
        name = f"{profile_segment}-{name}"
    return name


def still_running_guidance(
    *,
    agent_id: str,
    run_id: str,
    elapsed_seconds: float,
    run_status: RunStatus,
    turns_used: int,
) -> str:
    """Turn-yield guidance carried by a ``still_running`` wait (design §6).

    A tool result does not end the turn by itself: the model must yield so
    queued user input can drain between turns. The guidance therefore
    instructs ending the turn or continuing other work now and re-invoking
    ``wait_for_agent`` in a later turn — never a same-turn re-invocation.
    """
    turn_word = "turn" if turns_used == 1 else "turns"
    return (
        f"Agent {agent_id} (run {run_id}) is still {run_status.value} after "
        f"{elapsed_seconds:g}s and {turns_used} {turn_word}. End your turn or "
        "continue other work now: queued user input is only delivered between "
        "turns. Re-invoke wait_for_agent in a later turn to keep waiting, or "
        "call check_agents to inspect progress."
    )


class EmptySubagentResponseError(ValueError):
    """A completed subagent run produced no substantive response."""


class SubagentManagementError(ValueError):
    """Base error for a retained subagent lifecycle operation."""


class AgentClosedError(SubagentManagementError):
    """The agent is closed and cannot be reused."""


class AgentEvictedError(AgentClosedError):
    """The agent was evicted and cannot be reused."""


class AgentBusyError(SubagentManagementError):
    """The agent is busy with another run."""


class AgentProfileMismatchError(SubagentManagementError):
    """The requested profile does not match the retained agent."""


class AgentResultExpiredError(SubagentManagementError):
    """The retained result has expired."""


class UnknownAgentError(SubagentManagementError):
    """The agent or run does not exist."""


class LaunchConfigError(SubagentManagementError):
    """Base error for rejected launch configuration with a safe field path."""

    def __init__(
        self, field: str, message: str = "Invalid launch configuration"
    ) -> None:
        self.field = field
        self.field_path = field
        super().__init__(f"{message} at {field}")


class InvalidLaunchConfigError(LaunchConfigError):
    """A supplied launch configuration field is invalid."""


class InvalidLaunchModelError(InvalidLaunchConfigError):
    """The selected model alias is unknown or not permitted."""


class InvalidLaunchThinkingError(InvalidLaunchConfigError):
    """The selected thinking level is not supported by the model."""


class InvalidLaunchPromptError(InvalidLaunchConfigError):
    """The selected system prompt is unavailable."""


class InvalidLaunchToolError(InvalidLaunchConfigError):
    """A launch tool selection or override is invalid."""


class ImmutableLaunchPersonaError(InvalidLaunchConfigError):
    """A retained agent's frozen persona was changed."""


class MissingAgentProfileError(LaunchConfigError):
    """The selected launch profile is unavailable."""


class UnsupportedChildForkError(LaunchConfigError):
    """Forking a child launch context is intentionally unsupported."""


class SubagentManagementPort(Protocol):
    """Port for managing background subagent lifecycle."""

    async def check_agents(self) -> list[AgentSummary]:
        """List the bounded agent inventory and current run status. Nonblocking.

        The inventory covers resident records (running, finalizing, idle)
        plus terminal metadata (closed, evicted) that has not expired.
        """
        ...

    async def get_agent_result(
        self, agent_id: str, run_id: str | None = None
    ) -> TaskResult | None:
        """Nonblocking retrieval of a completed run's result.

        If run_id is None, returns the latest run's result.
        Returns None while the run is still in progress. Raises UnknownAgentError
        if the agent or run is unknown and AgentResultExpiredError if its result expired.
        """
        ...

    async def wait_for_agent(
        self, agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> WaitForAgentResult:
        """Block until the specified run completes, returning a typed disposition.

        Returns a ``completed`` disposition carrying the finished run's
        TaskResult, or a ``still_running`` disposition when the timeout
        expires first (the timeout is a bounded checkpoint, never a failure).
        Timeout must NOT cancel the background run — it only unblocks the
        waiter. Raises ValueError for invalid agent_id, run_id mismatch, or
        non-positive timeout.
        """
        ...

    async def cancel_run(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        reason: RunStopReason,
        requester_session_id: str,
    ) -> CancelResult:
        """Request a run-pinned stop without releasing identity or wait leases."""
        ...

    async def release_agent(self, agent_id: str) -> ReleaseAgentOutcome:
        """Close and remove a resident agent or its terminal tombstone.

        Unguarded by design: the model tool may release active instances
        (design §3.1). Returns RELEASED for a live agent and EVICTED when
        removing a closed or evicted tombstone. Raises ValueError for
        unknown agent_id. Release purges retained results, expiry markers,
        and wait leases.
        """
        ...

    async def resume_agent(
        self,
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
    ) -> ResumeAgentResult:
        """Explicitly resume a suspended agent (handoff §1.4; WP2).

        Nonblocking acceptance through ONE admission operation (the later
        automatic controller must call the same operation): single-flight
        and run-pinned, so concurrent same-key and different-key calls
        admit exactly one successor with a NEW run_id on the SAME agent
        identity. The old failed result and waits pinned to it stay
        immutable; prior execution/tool tasks are joined before admission.
        NO auto-scheduling: an admitted successor starts immediately.
        """
        ...

    async def reconcile_result(
        self,
        agent_id: str,
        expected_run_id: str,
        transcript_boundary: str,
        verified_effects: Sequence[EffectVerification],
        *,
        requester_session_id: str,
    ) -> ReconcileResult:
        """Submit externally verified tool effects for a reconcile barrier.

        PINNED to the failed run and its transcript boundary: stale or
        duplicate evidence cannot clear a newer barrier. ``cleared`` flips
        SUSPENDED[reconcile_required] to eligible; verification showing
        unrepeatable damage flips it to prohibited with reason codes.
        Replacement remains permitted at any point; the barrier never
        auto-clears.
        """
        ...


class SubagentRunnerPort(Protocol):
    def run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]: ...


@dataclass(slots=True)
class SubagentRunAccumulator:
    _response: list[str] = field(default_factory=list)
    _completed: bool = True

    def observe(self, event: BaseEvent, *, tool_call_id: str) -> ToolStreamEvent | None:
        if isinstance(event, AssistantEvent):
            if event.content:
                self._response.append(event.content)
            if event.stopped_by_middleware:
                self._completed = False
            return None
        if not isinstance(event, ToolResultEvent):
            return None
        if event.skipped:
            self._completed = False
            return None
        if event.result is None or event.tool_class is None:
            return None
        if event.presentation is not None:
            display = event.presentation.display
        else:
            display = ToolUIDataAdapter(event.tool_class).get_result_display(event)
        return ToolStreamEvent(
            tool_name="task",
            message=f"{event.tool_name}: {display.text}",
            tool_call_id=tool_call_id,
        )

    def record_error(self, message: str) -> None:
        self._completed = False
        self._response.append(f"\n[Subagent error: {message}]")

    def build_result(self, *, turns_used: int, completed: bool = True) -> TaskResult:
        completed = self._completed and completed
        response = "".join(self._response)
        if completed and not response.strip():
            raise EmptySubagentResponseError(
                "Completed subagent produced an empty response"
            )
        return TaskResult(response=response, turns_used=turns_used, completed=completed)


# The worker disclosure requirement rides in every task brief, with or without
# a scratchpad, in both dispatch modes, and on agent reuse: whatever report
# format the brief or a loaded skill specifies must carry the denial record
# and the cumulative touched-path set.
_BRIEF_DISCLOSURE = (
    "Report disclosure: whatever report format this task or a loaded skill "
    "specifies, your final report must include the denial record (each denied "
    "command and the runtime's stated reason, or none) and the complete "
    "cumulative touched-path set (every repository path you create, edit, "
    "rename, or delete, including edits later reverted)."
)


def prepare_subagent_prompt(task: str, ctx: InvokeContext) -> str:
    brief = f"{_BRIEF_DISCLOSURE}\n\n{task}"
    if ctx.scratchpad_dir is None:
        return brief
    return (
        f"Scratchpad directory: {ctx.scratchpad_dir}\n"
        "You can read and write files here without permission prompts.\n\n"
        f"{brief}"
    )
