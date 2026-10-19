"""Agent cancel, release, resume, and reconcile wire models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.models import PublicRecoveryEligibility, RunStopReason


class CancelOutcome(StrEnum):
    STOP_REQUESTED = "stop_requested"
    ALREADY_STOPPING = "already_stopping"
    ALREADY_FINISHING = "already_finishing"
    NOT_RUNNING = "not_running"
    UNKNOWN_RUN = "unknown_run"
    FORBIDDEN = "forbidden"


class AgentsCancelParams(ProtocolModel):
    agent_id: str = Field(min_length=1)
    run_id: str | None = Field(default=None, min_length=1)


class AgentsCancelResponse(ProtocolModel):
    """Stop-request disposition, not confirmation of terminal cancellation."""

    outcome: CancelOutcome
    run_id: str | None = None
    stop_reason: RunStopReason | None = None


class ReleaseOutcome(StrEnum):
    RELEASED = "released"
    EVICTED = "evicted"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class AgentsReleaseParams(ProtocolModel):
    """Guarded release request with optimistic concurrency on the latest run."""

    agent_id: str = Field(min_length=1)
    expected_run_id: str | None = Field(default=None, min_length=1)


class AgentsReleaseResponse(ProtocolModel):
    """Guarded release disposition.

    ``conflict``/``unknown`` outcomes never stopped or purged anything; the
    reason explains which guard fired.
    """

    outcome: ReleaseOutcome
    reason: str | None = None


# --- WP2: explicit resume and effect reconciliation -----------------------
# Wire mirrors of the core resume/reconcile contracts
# (chartreux.core.subagents); the public app-server layer cannot import
# core, so these models mirror the field sets — pinned by a
# mirror-contract test — and the server boundary projects onto them.


class ResumeOutcome(StrEnum):
    """Typed disposition of an explicit resume_agent request (WP2)."""

    SCHEDULED = "scheduled"
    ALREADY_SCHEDULED = "already_scheduled"
    ALREADY_RESUMED = "already_resumed"
    RECONCILE_REQUIRED = "reconcile_required"
    EXPIRED = "expired"
    NOT_RESUMABLE = "not_resumable"
    FORBIDDEN = "forbidden"


class AgentsResumeParams(ProtocolModel):
    """Explicit resume request for a suspended failed run (WP2).

    ``expected_run_id`` pins the failed run to resume from;
    ``idempotency_key`` single-flights concurrent calls.
    """

    agent_id: str = Field(min_length=1)
    expected_run_id: str = Field(min_length=1)
    idempotency_key: str | None = Field(default=None, min_length=1)


class AgentsResumeResponse(ProtocolModel):
    """Resume admission disposition (WP2).

    ``scheduled``/``already_scheduled`` carry the recovery-operation id and
    the successor run id published on admission; the failed run's result
    and pinned waits stay immutable.
    """

    outcome: ResumeOutcome
    agent_id: str
    run_id: str | None = None
    successor_run_id: str | None = None
    recovery_operation_id: str | None = None
    retry_at: float | None = None
    eligibility: PublicRecoveryEligibility | None = None
    reason_codes: list[str] = Field(default_factory=list)
    reason: str | None = None
    same_agent_resumes_used: int = 0


class ReconcileOutcome(StrEnum):
    """Typed disposition of a reconcile_result barrier submission (WP2)."""

    CLEARED = "cleared"
    PROHIBITED = "prohibited"
    STALE = "stale"
    DUPLICATE = "duplicate"
    NOT_PENDING = "not_pending"
    FORBIDDEN = "forbidden"


class VerifiedEffectWire(ProtocolModel):
    """One externally verified tool effect (reconcile_result input)."""

    tool_call_id: str = Field(min_length=1)
    disposition: Literal["settled", "unrepeatable_damage"]
    note: str | None = None


class AgentsReconcileParams(ProtocolModel):
    """Reconcile request pinned to the failed run and transcript boundary.

    ``verified_effects`` carries one entry per unknown tool call; an empty
    list is the complete zero-entry accounting for a run that left no tool
    call outstanding (round-2 V4).
    """

    agent_id: str = Field(min_length=1)
    expected_run_id: str = Field(min_length=1)
    transcript_boundary: str = Field(min_length=1)
    verified_effects: list[VerifiedEffectWire] = Field(
        description=(
            "One entry per unknown tool call; an empty list is the complete "
            "zero-entry accounting for a run that left no tool call "
            "outstanding"
        )
    )


class AgentsReconcileResponse(ProtocolModel):
    """Reconcile barrier disposition (WP2).

    ``cleared`` flips the barrier to eligible; ``prohibited`` flips it with
    reason codes when verification shows unrepeatable damage; ``stale``/
    ``duplicate`` never change the barrier.
    """

    outcome: ReconcileOutcome
    agent_id: str
    run_id: str | None = None
    transcript_boundary: str | None = None
    eligibility: PublicRecoveryEligibility | None = None
    reason_codes: list[str] = Field(default_factory=list)
    reason: str | None = None
