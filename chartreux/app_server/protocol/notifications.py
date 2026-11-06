"""Server notification params models and the notification method registry."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from types import MappingProxyType
from typing import ClassVar, Literal

from pydantic import AwareDatetime, Field

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.models import (
    AgentStatsSnapshot,
    JsonPatchOperation,
    PublicCompletionAudit,
    PublicFailureEnvelope,
    PublicHistoryEntry,
    PublicRecoveryEligibility,
    PublicSessionState,
    PublicTurn,
    PublicTurnQueue,
    RunStopReason,
    SessionLogSummary,
    UsageWindowSummaries,
)


class EventNotificationParams(ProtocolModel):
    event_id: int = Field(ge=0, strict=True)
    session_id: str
    emitted_at: int


class HistoryEntryAddedParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "history/entryAdded"

    turn_id: str | None = None
    entry: PublicHistoryEntry


class HistoryEntryUpdatedParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "history/entryUpdated"

    turn_id: str | None = None
    entry_id: str
    patch: list[JsonPatchOperation]


class SessionSnapshotParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "session/snapshot"

    state: PublicSessionState


class SessionHandoffParams(EventNotificationParams):
    old_session_id: str
    state: PublicSessionState
    session_log: SessionLogSummary


class SessionCompactedParams(SessionHandoffParams):
    NOTIFICATION_METHOD: ClassVar[str] = "session/compacted"

    summary_length: int = Field(ge=0)


class SessionUpdatedParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "session/updated"

    patch: list[JsonPatchOperation]


class TurnQueueUpdatedParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "turn_queue_updated"

    queue: PublicTurnQueue


class TurnStartedParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "turn/started"

    turn: PublicTurn


class TurnUpdatedParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "turn/updated"

    turn: PublicTurn


class TurnCompletedParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "turn/completed"

    turn: PublicTurn


class AgentSummaryModel(ProtocolModel):
    agent_id: str
    identity: str | None = Field(default=None, exclude_if=lambda value: value is None)
    identity_revision: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    name: str
    launch_purposes: list[str] = Field(default_factory=list)
    availability: str
    current_run_id: str | None = None
    current_run_status: str | None = None
    turns_used: int | None = None
    last_run_status: str | None = None
    # Structured current/latest run outcome, retained through finalization/eviction.
    stop_reason: RunStopReason | None = None
    context_tokens: int | None = None
    context_window: int | None = None
    compacting: bool = False
    initial_task_summary: str | None = None
    current_task_summary: str | None = None
    idle_seconds: float | None = None
    run_elapsed_seconds: float | None = None
    latest_run_id: str | None = None
    ttl_remaining_seconds: float | None = None
    effective_model: str | None = None
    base_model: str | None = None
    active_provider: str | None = None
    effective_thinking: str | None = None
    result_expired: bool = False
    # U2/B8: retrievable, not-yet-collected run results, newest first. The
    # key is ALWAYS emitted, including an empty list: its presence is part
    # of the wire contract, so no exclude_if may be added here.
    uncollected_run_ids: list[str] = Field(default_factory=list)
    # Public typed failure envelope of the latest failed run (WP1); bounded
    # and redacted, full diagnostics stay private. The public wire model
    # mirrors the core envelope (the public layer cannot import core).
    last_failure: PublicFailureEnvelope | None = None
    # WP2 recovery linkage: the stable recovery-chain identity, the
    # same-agent resume counter, the failed run the latest run resumed
    # from, and the suspended barrier's EFFECTIVE recovery eligibility
    # (a reconcile_result submission flips it; the envelope stays
    # immutable). None when no barrier is active.
    recovery_chain_id: str | None = None
    same_agent_resumes_used: int | None = None
    resumed_from_run_id: str | None = None
    recovery_eligibility: PublicRecoveryEligibility | None = None
    # WP5 lineage pair: the orchestrator work-package identity of the latest
    # run and the failed agent this agent replaced (a replacement gets a NEW
    # agent_id with predecessor/package lineage). None when unclaimed.
    work_package_id: str | None = None
    predecessor_agent_id: str | None = None
    # O1: the typed completion-audit stamp of the LATEST run, resolved against
    # that run's retained result. None when the latest run has no retained
    # result, has not finished, or its result expired — a previous run's stamp
    # never rides a new run's summary.
    completion_audit: PublicCompletionAudit | None = None


class AgentEvictionModel(ProtocolModel):
    agent_id: str
    run_id: str
    reason: Literal["ttl", "idle_cap"]
    idle_duration_seconds: float
    root_generation: int


class AgentsUpdateParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "agents/update"

    agents: list[AgentSummaryModel]
    evictions: list[AgentEvictionModel] = Field(default_factory=list)


class StatsUpdatedParams(EventNotificationParams):
    NOTIFICATION_METHOD: ClassVar[str] = "session/statsUpdated"

    stats: AgentStatsSnapshot
    context_window: int | None


class ChildCallbackClosedParams(ProtocolModel):
    """A child-session callback reached a terminal state.

    Child callbacks never enter the ROOT projection: their history entries
    stay in the child session, so the parent client cannot learn the closure
    from history patches. This notification is the only route. ``session_id``
    is the OWNING child session and ``callback_id`` is unique per session, so
    a client matches the closure against the callback it delivered earlier —
    never against a newer callback.

    Producer set note: after Q1 no shipped producer can mint a child question.
    The channel is defense-in-depth for pre-Q1 callbacks in running sessions
    and for future child-askable callback kinds; the strict-client
    compatibility caveat is paid knowingly (older clients drop the unknown
    notification method, exactly as they drop any unhandled notification).
    """

    NOTIFICATION_METHOD: ClassVar[str] = "callback/childClosed"

    session_id: str
    callback_id: str
    reason: str


class UsageUpdatedParams(ProtocolModel):
    """Coalesced global snapshot, independent of session event sequencing.

    Servers coalesce updates; clients deduplicate snapshots by revision.
    Summaries are always global, never filtered by the last usage/read.
    """

    NOTIFICATION_METHOD: ClassVar[str] = "usage/updated"

    as_of: AwareDatetime
    revision: int = Field(ge=0, strict=True)
    summaries: UsageWindowSummaries
    degraded: bool = False


@cache
def server_notification_registry() -> Mapping[str, type[ProtocolModel]]:
    """Return declared app-server notification methods and their params models.

    Notification params models are protocol declarations loaded at import time,
    so the registry is closed before its first use and needs no invalidation.
    Dynamically defining notification models after this call is unsupported.
    """
    registry: dict[str, type[ProtocolModel]] = {}
    pending = list(ProtocolModel.__subclasses__())
    while pending:
        model = pending.pop()
        pending.extend(model.__subclasses__())
        if not model.__module__.startswith("chartreux.app_server"):
            continue
        method = model.__dict__.get("NOTIFICATION_METHOD")
        if not isinstance(method, str):
            continue
        if existing := registry.get(method):
            raise RuntimeError(
                f"Duplicate notification method {method!r}: "
                f"{existing.__name__} and {model.__name__}"
            )
        registry[method] = model
    return MappingProxyType(registry)


def validate_notification_method(method: str, params: ProtocolModel) -> None:
    """Require a notification's method to be declared by its params model."""
    declared_method = getattr(type(params), "NOTIFICATION_METHOD", None)
    if not isinstance(declared_method, str):
        raise ValueError(
            f"Notification method {method!r} does not match "
            f"{type(params).__name__}.undeclared"
        )
    registered_model = server_notification_registry().get(declared_method)
    if registered_model is not type(params) or declared_method != method:
        raise ValueError(
            f"Notification method {method!r} does not match "
            f"{type(params).__name__}.{declared_method}"
        )
