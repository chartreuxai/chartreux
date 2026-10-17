"""Agent cancel and release wire models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.models import RunStopReason


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
