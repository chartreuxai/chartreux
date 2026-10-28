from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
import enum
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chartreux.core.agents.identity_launch import TaskIdentityArgs
from chartreux.core.completion_contract import CompletionAuditClassification
from chartreux.core.completion_evidence import (
    MAX_CHRONOLOGY_ENTRIES,
    ContentSnapshot,
    PerRunCompletionChronology,
    snapshot_bytes,
)
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
from chartreux.core.subagent_submission import (
    UNKNOWN_WRITE_FOOTPRINT,
    bash_write_paths_from_result_event,
    blocked_submission_response,
    touched_path_from_result_event,
)
from chartreux.core.tools.secret_scan import incomplete_finding, scan_paths
from chartreux.core.tools.ui import ToolUIDataAdapter
from chartreux.observability.logging import logger
from chartreux.utils.tool_presentation import ToolEffectKind

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
    from collections.abc import Callable

    from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
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


class TaskArgs(TaskIdentityArgs):
    """The task tool's argument surface (design §7; package-11b rewiring).

    The canonical fields (``model``, ``identity``, ``purpose``, ``seat``)
    and their grammar come from the package-8 schema module. The legacy
    ``agent_type`` profile bridge is retired with the profile axis (package
    16a): a call that still supplies it fails with the successor guidance.
    """

    agent_id: str | None = Field(
        default=None,
        description=(
            "Retained agent instance handle to continue with background: true; "
            "if omitted, create a new instance selected by the identity field. "
            "Reuse inherits the instance's committed retention policy unless "
            "retain is passed."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        fan_out = value.get("fan_out", False)
        if fan_out is not False and fan_out is not None:
            raise ValueError(
                "Roles are single presets. Launch separate tasks with explicit "
                "presets/models for multiple agents."
            )
        value = {key: item for key, item in value.items() if key != "fan_out"}

        if "agent_type" in value:
            raise ValueError(
                "agent_type is retired; use the identity field instead (the "
                "shipped profile names map to identities: worker -> coder, "
                "advisor -> advisor, reviewer -> reviewer; a custom profile "
                "name becomes a 'user:<name>' custom identity prompt in the "
                "user prompts directory)"
            )
        return value

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
            "Human-facing agent name ({identity}-{model}-{number}) for background "
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
    completion_audit: CompletionAuditClassification | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "The typed, versioned completion-audit stamp attached to a "
            "completed result; stamped whenever the composition has a "
            "surface (findings, receipt, check verdicts, or collected "
            "evidence) and omitted from dumps entirely only when the audit "
            "composed inertly (no surface — the stamp-mode default's "
            "behavior preservation)."
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
    """Human-facing ``{identity}-{model}-{number}`` name (design §5).

    The server always populates this when projecting summaries; the empty
    string is the legacy-constructor default for older in-process callers.
    """
    identity: str | None = None
    """Canonical identity identifier of the agent's committed launch."""
    identity_revision: str | None = None


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
    identity: str,
    identity_display_label: str | None,
    base_model: str | None,
    effective_model: str | None,
) -> str:
    """Derive the human-facing agent name ``{identity}-{model}-{number}`` (design §5).

    The identity segment is the resolved identity reference, or the display
    label of a ``user:``-qualified custom identity. The model segment is the
    namespace-stripped canonical ``base_model``, falling back to the
    namespace-stripped ``effective_model``, and is omitted when unknown.
    """
    name_segment = (
        "-".join(identity_display_label.split())
        if identity_display_label is not None and identity.startswith("user:")
        else identity
    )
    model = base_model if base_model is not None else effective_model
    number = _agent_sequence_number(agent_id)
    if model is None:
        name = f"{name_segment}-{number}"
    else:
        name = f"{name_segment}-{_strip_model_namespace(model)}-{number}"
    if _AGENT_HANDLE_PATTERN.fullmatch(name) is not None:
        name = f"{name_segment}-{name}"
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


class AgentIdentityMismatchError(SubagentManagementError):
    """The requested identity does not match the retained agent."""


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


class InvalidLaunchToolError(InvalidLaunchConfigError):
    """A launch tool selection or override is invalid."""


class ImmutableLaunchPersonaError(InvalidLaunchConfigError):
    """A retained agent's frozen persona was changed."""


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
        override_cooldown: bool = False,
    ) -> ResumeAgentResult:
        """Explicitly resume a suspended agent (handoff §1.4; WP2).

        Nonblocking acceptance through ONE admission operation (the later
        automatic controller must call the same operation): single-flight
        and run-pinned, so concurrent same-key and different-key calls
        admit exactly one successor with a NEW run_id on the SAME agent
        identity. The old failed result and waits pinned to it stay
        immutable; prior execution/tool tasks are joined before admission.
        NO auto-scheduling: an admitted successor starts immediately.
        With ``override_cooldown`` the explicit request also mints the
        endpoint-bound forced-attempt permit from the root ledger (C6b)
        and binds it to the successor's first model call; the ledger
        denies the mint when the episode is exhausted, spacing-bound, or
        has an attempt in flight, and a denied mint consumes nothing.
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


#: Mutation-roots sentinel recorded when a shell command's mutation-roots
#: footprint could not be established (a checkout/switch whose work-tree root
#: is unresolvable). Mutation-scope accounting only: it never enters the
#: submission scan's path list.
UNKNOWN_MUTATION_ROOT_FOOTPRINT = (
    "<shell command with an unestablished mutation-scope root>"
)


#: Read cap for one artifact's collection-time content snapshot — mirrors
#: the secret scan's per-file cap; an over-cap artifact records no witness
#: (it is already scan-incomplete at the submission gate).
_ARTIFACT_SNAPSHOT_READ_CAP_BYTES = 4 * 1024 * 1024


def _capture_artifact_snapshot(path: str) -> ContentSnapshot | None:
    """The bounded collection-time content snapshot of one artifact.

    Reads the artifact as the run's collection observed it — the tool just
    wrote it. A missing, unreadable, or over-cap artifact records no
    witness: an availability fact, never a fabricated one.
    """
    try:
        artifact = Path(path)
        if (
            not artifact.is_file()
            or artifact.stat().st_size > _ARTIFACT_SNAPSHOT_READ_CAP_BYTES
        ):
            return None
        return snapshot_bytes(artifact.read_bytes())
    except OSError:
        return None


def mutation_roots_from_result_event(event: ToolResultEvent) -> tuple[list[str], bool]:
    """The mutation-roots footprint a shell result reports, and its unknown flag.

    The bash tool surfaces the resolved work-tree roots of its command's
    checkout/switch invocations alongside its write paths, and bash_start
    surfaces the started command's the same way; a failed, timed-out, or
    cancelled run's footprint rides the error event's fields, copied there by
    the tool executor. The footprint is mutation-scope accounting only — the
    caller keeps it OUT of the submission scan's path list (a work-tree root
    would scan the whole repository and burn the scan's budget). Skipped
    calls never ran. Returns ``(roots, unknown)``; a result carrying the
    footprint fields without well-formed values reads as unknown, never as
    an established empty one, and a result of a tool whose model carries no
    footprint attributes at all reads as an established empty one.
    """
    if event.skipped or event.tool_class is None:
        return [], False
    result = event.result
    if result is not None:
        missing = object()
        roots = getattr(result, "mutation_roots", missing)
        unknown = bool(getattr(result, "mutation_roots_unknown", False))
        shell = getattr(event.tool_class, "effect_kind", None) is ToolEffectKind.SHELL
        if roots is missing and not shell:
            return [], False
        if not isinstance(roots, (list, tuple)):
            return [], True
    else:
        roots = getattr(event, "error_mutation_roots", None)
        unknown = bool(getattr(event, "error_mutation_roots_unknown", False))
    if not isinstance(roots, (list, tuple)):
        return [], unknown
    return [root for root in roots if isinstance(root, str) and root], unknown


@dataclass(slots=True)
class SubagentRunAccumulator:
    _response: list[str] = field(default_factory=list)
    _completed: bool = True
    # P2-i: the session's touched paths — repo paths created or edited
    # through the file tools, plus the write-classified paths shell
    # commands produced — collected from observed tool results and scanned
    # at the submission boundary (build_result). Insertion-ordered.
    _touched_paths: dict[str, None] = field(default_factory=dict)
    # A1: the session's mutation-roots footprint — the resolved work-tree
    # roots of the shell commands' checkout/switch invocations, collected
    # from observed tool results alongside the touched paths. A
    # mutation-scope footprint ONLY: it is never scanned at the submission
    # boundary (a work-tree root would scan the whole repository and burn
    # the scan's budget); the completion audit consumes it.
    _mutation_roots: dict[str, None] = field(default_factory=dict)
    # P3: the bounded per-run completion-evidence chronology — receipts
    # pair with their originating invocations via the harness-minted,
    # run-scoped receipt occurrence ID (never tool_call_id alone).
    _chronology: PerRunCompletionChronology = field(
        default_factory=PerRunCompletionChronology
    )
    # P3 (round 5): the collected artifacts' bounded content snapshots —
    # the path-keyed collection-time mutation witnesses the completion
    # audit consumes (the artifact audit reads them by path; the receipt
    # audit reads the invocation-level witness off the chronology entry).
    # Bounded like the chronology: the oldest snapshot evicts first, and
    # an evicted path's witness is an availability fact, never a
    # contradiction.
    _artifact_snapshots: dict[str, ContentSnapshot] = field(default_factory=dict)
    # P6a's run-keyed finalization cache, declared by P3 (round 6) and
    # INERT until P6a consumes it for finalization idempotency.
    _finalization_cache: dict[str, Any] = field(default_factory=dict)

    def completion_evidence_sink(self) -> Callable[[CompletionEvidenceReceipt], None]:
        """The P2 side-channel sink: registers receipts against invocations."""
        return self._chronology.observe_receipt

    def _collect_artifact_snapshots(
        self, event: ToolResultEvent, artifact_paths: Sequence[str]
    ) -> None:
        """Capture each collected artifact's collection-time snapshot (P3).

        The bounded content snapshot (hash + byte size + sampled head/tail)
        is the mutation witness the completion audit consumes: kept per
        artifact path for the artifact audit, and stored on the
        invocation's chronology entry for the receipt audit. Capture
        happens at collection, never at finalization; a missing,
        unreadable, or over-cap artifact records no witness — an
        availability fact, never a fabricated one.
        """
        witness: ContentSnapshot | None = None
        for path in artifact_paths:
            snapshot = _capture_artifact_snapshot(path)
            if snapshot is None:
                continue
            self._artifact_snapshots[path] = snapshot
            while len(self._artifact_snapshots) > MAX_CHRONOLOGY_ENTRIES:
                self._artifact_snapshots.pop(next(iter(self._artifact_snapshots)))
            if witness is None:
                witness = snapshot
        if event.occurrence_id:
            # The result event registers against its harness-minted,
            # run-scoped occurrence ID — never reminted, never keyed by
            # tool_call_id alone.
            self._chronology.observe_result_event(event, snapshot=witness)

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
        # P2-i R6/R7: bash-written artifacts join the same scan — collected
        # from a successful result, and from a failed or timed-out run's
        # error event (partial execution still wrote files); an
        # unestablished write surface records the unknown-footprint sentinel.
        # Collected before the result-less early return so error events
        # contribute their footprint.
        write_paths, write_unknown = bash_write_paths_from_result_event(event)
        for path in write_paths:
            self._touched_paths[path] = None
        if write_unknown:
            self._touched_paths[UNKNOWN_WRITE_FOOTPRINT] = None
        # A1: the mutation-roots footprint is collected alongside — into its
        # own mutation-scope accounting, never into _touched_paths.
        mutation_roots, mutation_unknown = mutation_roots_from_result_event(event)
        for root in mutation_roots:
            self._mutation_roots[root] = None
        if mutation_unknown:
            self._mutation_roots[UNKNOWN_MUTATION_ROOT_FOOTPRINT] = None
        # P3 (round 5): the collected artifacts' bounded content snapshots,
        # captured HERE at collection — the mutation witnesses the
        # completion audit consumes (never captured at finalization) — and
        # the result event's registration against its harness-minted,
        # run-scoped occurrence ID (the pairing the receipt side of the
        # chronology completes).
        touched_path = touched_path_from_result_event(event)
        artifact_paths = list(write_paths)
        if touched_path is not None:
            artifact_paths.append(touched_path)
        self._collect_artifact_snapshots(event, artifact_paths)
        if event.result is None or event.tool_class is None:
            return None
        if touched_path is not None:
            self._touched_paths[touched_path] = None
        if event.presentation is not None:
            display = event.presentation.display
        else:
            display = ToolUIDataAdapter(event.tool_class).get_result_display(event)
        return ToolStreamEvent(
            tool_name="task",
            message=f"{event.tool_name}: {display.text}",
            tool_call_id=tool_call_id,
        )

    @property
    def mutation_roots(self) -> list[str]:
        """The collected mutation-roots footprint, in collection order.

        The mutation-scope footprint the completion audit's touched-path
        comparisons consume; the submission scan never sees it.
        """
        return list(self._mutation_roots)

    def record_error(self, message: str) -> None:
        self._completed = False
        self._response.append(f"\n[Subagent error: {message}]")

    def build_result(
        self,
        *,
        turns_used: int,
        completed: bool = True,
        run_id: str | None = None,
        contract: Any = None,
        claim: Any = None,
        runner: Any = None,
        mode: Literal["stamp", "enforce"] | None = None,
    ) -> TaskResult:
        completed = self._completed and completed
        # P10: the completion-audit composition, before the submission
        # gate. The composition module is imported here, at the call:
        # subagents.py sits above it in the import graph (the composition's
        # artifact audit imports this module's mutation-roots sentinel), so
        # a module-level import would close a cycle.
        from chartreux.core.completion_audit import compose_completion_audit

        # The composition reads the once-only finalization (P6a) from the
        # run-keyed cache and runs no checks; in stamp mode the composed
        # classification rides on the result, in enforce mode a
        # worker-side disposition blocks the submission (completed=False,
        # bounded actionable rejection message).
        composition = compose_completion_audit(
            accumulator=self,
            contract=contract,
            claim=claim,
            finalization=(self._finalization_cache.get(run_id) if run_id else None),
            declared_completed=completed,
            runner=runner,
            mode=mode or getattr(contract, "audit_mode", None) or "stamp",
        )
        audit = composition.classification
        blocked_message = composition.rejection_message()
        # P1 dump-shape rule: an inert composition stamps nothing — the
        # result's completion_audit stays None (the old dump shape,
        # unchanged for runs composed without an audit surface).
        stamped_audit = audit if composition.has_surface else None
        # P2-i submission boundary: the scan runs when the session reports
        # its work complete, before the result is surfaced as final. Both
        # the foreground and the background publication paths assemble
        # their result here, so neither can bypass the gate. A finding
        # blocks the submission: the result carries the redacted block
        # message instead of the agent's response, and never surfaces as
        # final (completed=False).
        # A1: the scan scopes EXACTLY _touched_paths — the mutation-roots
        # footprint never joins it (a work-tree root would scan the whole
        # repository and burn the bounded walk's budget, blocking
        # publication); the completion audit consumes _mutation_roots.
        findings = scan_paths(self._touched_paths)
        if UNKNOWN_WRITE_FOOTPRINT in self._touched_paths:
            # Fail-closed: a shell command whose write surface could not be
            # established leaves an unscannable footprint — scan_paths skips
            # the missing sentinel path itself, so the gate adds the
            # scan-incomplete finding explicitly.
            findings.append(
                incomplete_finding(
                    UNKNOWN_WRITE_FOOTPRINT,
                    "the write surface of a shell command could not be established",
                )
            )
        if findings:
            # Secret-scan precedence: the scan's redacted block message
            # wins over any audit rejection — the scan's findings never
            # leak into another surface.
            logger.warning(
                "Subagent submission blocked by the secret scan: %d finding(s) "
                "over %d touched path(s), rules=%s",
                len(findings),
                len(self._touched_paths),
                ",".join(sorted({finding.rule_id for finding in findings})),
            )
            return TaskResult(
                response=blocked_submission_response(findings),
                turns_used=turns_used,
                completed=False,
                completion_audit=stamped_audit,
            )
        if blocked_message is not None:
            # P10 enforce mode: a worker-side disposition blocks the
            # submission — the bounded, actionable rejection message
            # replaces the response and the result never surfaces as
            # final (completed=False).
            logger.warning(
                "Subagent submission blocked by the completion audit: %s",
                blocked_message,
            )
            return TaskResult(
                response=blocked_message,
                turns_used=turns_used,
                completed=False,
                completion_audit=stamped_audit,
            )
        response = "".join(self._response)
        if completed and not response.strip():
            raise EmptySubagentResponseError(
                "Completed subagent produced an empty response"
            )
        return TaskResult(
            response=response,
            turns_used=turns_used,
            completed=completed,
            completion_audit=stamped_audit,
        )


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
