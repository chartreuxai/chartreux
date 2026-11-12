"""Agent-side session state: the open reply and the snapshot payload.

The process that owns a root session also owns the state a snapshot read
answers with, and none of it can be reconstructed from a coordinator copy: the
recovery gate's availability view is computed against the committed model and
the live availability registry, the marker records exist only in this
incarnation's in-memory table, and the log projection reads the loop's own
session logger. They are therefore built here, at reply-write time, next to the
loop that owns them -- a payload can never predate an admission that preceded
its reply.

The open reply carries the same kind of state at session open: the transcript
directory this process created for itself, the jobs directory the launcher will
record into (published here and in identity-transition replies only, never on
the ready frame), and the accounting identity the open adopted so the
coordinator's usage bookkeeping stays authoritative across the boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from chartreux.app_server._projection import (
    project_config,
    project_diagnostics,
    project_history,
    project_mcp,
    project_session_log,
    project_skills,
    project_stats,
    project_tools,
)
from chartreux.app_server._state import build_public_state
from chartreux.app_server.agent_admission import (
    OP_STATE_ACTIVE,
    OP_STATE_TERMINAL,
    OperationMarker,
)
from chartreux.app_server.models import PublicSessionState, SessionLogSummary
from chartreux.app_server.protocol import RuntimeSnapshot
from chartreux.core._usage_startup import StartupAccountingContext
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.compaction import reorder_for_tool_adjacency
from chartreux.core.compaction.context import select_model_context
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.model_catalog.availability import (
    AvailabilitySnapshot,
    availability_snapshot,
)
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.usage import CoverageWarning, UsageRecord, UsageWriteResult

JOBS_DIR_NAME = "jobs"
"""The per-session directory the background-job launcher records into."""

MARKER_ABSENT = "absent"
"""No record for the request id in an admissible payload: PROVEN never admitted."""

MARKER_ACTIVE = "active"
"""The operation was admitted and had not settled when the payload was built."""

MARKER_TERMINAL = "terminal"
"""The operation settled agent-side; ``outcome`` names its terminal class."""

MARKER_STALE = "stale"
"""The payload cannot prove anything about the request id it was asked about."""

ADOPTION_ADOPTED = "adopted"
"""The open claimed and adopted a startup accounting identity."""

ADOPTION_NONE = "none"
"""The open bound an accounting owner that claims no startup identity."""

SNAPSHOT_HISTORY_LIMIT = 200
"""The bounded history window a snapshot payload carries (the in-process limit)."""


@dataclass(frozen=True, slots=True)
class RootOpenAccounting:
    """The accounting state one root open adopted, for the coordinator to replay.

    ``settlements`` are the content-free outcomes of work that settled before
    the loop existed; a cold scan of the usage ledger cannot recover their
    warnings, so they cross the boundary as records rather than as bytes.
    """

    root_session_id: str | None
    project_key: str | None
    adoption: str
    settlements: tuple[tuple[UsageRecord, UsageWriteResult], ...] = ()
    warnings: tuple[CoverageWarning, ...] = ()

    @classmethod
    def from_loop(
        cls, loop: AgentLoop, context: StartupAccountingContext | None = None
    ) -> RootOpenAccounting:
        """Describe the accounting identity one opened loop is bound to."""
        owner = getattr(loop, "accounting_owner", None)
        identity = context.identity if context is not None else None
        return cls(
            root_session_id=(
                owner.root_session_id
                if owner is not None
                else (identity.root_session_id if identity is not None else None)
            ),
            project_key=(
                owner.project_key
                if owner is not None
                else (identity.project_key if identity is not None else None)
            ),
            adoption=ADOPTION_ADOPTED if context is not None else ADOPTION_NONE,
            settlements=context.early_settlements if context is not None else (),
            warnings=context.warnings if context is not None else (),
        )


@dataclass(frozen=True, slots=True)
class AgentOpenReply:
    """What the agent process reports about the session it just opened."""

    session_id: str
    session_dir: str | None
    jobs_dir: str | None
    accounting: RootOpenAccounting


@dataclass(frozen=True, slots=True)
class AgentSessionSnapshot:
    """One agent-side session read: public state, gate view and marker records."""

    constructed_at: float
    incarnation_id: str
    state: PublicSessionState
    availability: AvailabilitySnapshot
    committed_model: CommittedModelIdentity | None
    markers: tuple[OperationMarker, ...]
    log: SessionLogSummary
    runtime: RuntimeSnapshot
    session_dir: str | None
    jobs_dir: str | None

    def marker_for(self, request_id: str) -> OperationMarker | None:
        """The marker record this payload carries for one coordinator request."""
        return next(
            (marker for marker in self.markers if marker.request_id == request_id), None
        )


@dataclass(frozen=True, slots=True)
class MarkerVerdict:
    """What one snapshot payload proves about one long-operation request."""

    state: str
    outcome: str | None = None
    op_id: str | None = None
    detail: str = ""

    @property
    def never_admitted(self) -> bool:
        """Whether the payload PROVES the request never reached admission."""
        return self.state == MARKER_ABSENT

    @property
    def provisional_in_flight(self) -> bool:
        """Whether the operation was admitted and had not settled at the read."""
        return self.state == MARKER_ACTIVE


def build_open_reply(
    loop: AgentLoop, *, accounting: RootOpenAccounting | None = None
) -> AgentOpenReply:
    """Build the open reply one opened loop answers with."""
    session_dir, jobs_dir = session_dirs(loop)
    return AgentOpenReply(
        session_id=loop.session_id,
        session_dir=str(session_dir) if session_dir is not None else None,
        jobs_dir=str(jobs_dir) if jobs_dir is not None else None,
        accounting=accounting or RootOpenAccounting.from_loop(loop),
    )


def owned_user_source(orchestrator: ConfigOrchestrator[Any]) -> SourceRestrictions:
    """The one owned, actual user source whose authorized roots this session holds.

    The layer must be a real ``UserConfigLayer`` in the stack, so an inherited
    or synthesized source never stands in for the file the user edits.
    """
    sources = [
        source
        for source in orchestrator.restrictions
        if source.kind == "source"
        and source.identity is not None
        and source.identity.owner == orchestrator.policy_owner
        and any(
            type(layer) is UserConfigLayer
            and layer.name == source.layer_name
            and layer.source_locator == source.locator
            for layer in orchestrator.layers
        )
    ]
    if len(sources) != 1:
        raise ValueError("Roots require one actual owned user source")
    return sources[0]


def session_policy_revision(loop: AgentLoop) -> str:
    """The public policy revision one loop's accepted config stands at."""
    return (
        f"{loop.session_id}:{loop._session_generation}:"
        f"{loop.config_orchestrator.accepted_token}"
    )


def session_roots_revision(loop: AgentLoop) -> str:
    """The revision a roots read answers with: policy plus source identity."""
    identity = owned_user_source(loop.config_orchestrator).identity
    if identity is None:  # pragma: no cover - owned_user_source guarantees it
        raise ValueError("Roots require one actual owned user source")
    return f"{session_policy_revision(loop)}:{identity.owner}:{identity.source}"


def session_dirs(loop: AgentLoop) -> tuple[Path | None, Path | None]:
    """The transcript directory and the jobs directory this loop publishes.

    A session whose logging is disabled (or not yet persisted) publishes
    neither: there is no directory to publish, and the job launcher's own
    fail-closed refusal covers the disabled case.
    """
    logger = getattr(loop, "session_logger", None)
    if logger is None or not getattr(logger, "persisted", False):
        return None, None
    session_dir = getattr(logger, "session_dir", None)
    if session_dir is None:
        return None, None
    return session_dir, session_dir / JOBS_DIR_NAME


def build_session_snapshot(
    loop: AgentLoop,
    *,
    markers: Iterable[OperationMarker],
    incarnation_id: str,
    history_limit: int = SNAPSHOT_HISTORY_LIMIT,
    include_history: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> AgentSessionSnapshot:
    """Build one snapshot payload from the live loop, at reply-write time."""
    session_dir, jobs_dir = session_dirs(loop)
    return AgentSessionSnapshot(
        constructed_at=clock(),
        incarnation_id=incarnation_id,
        state=build_public_state(
            loop,
            history=[],
            current_history=project_history(loop),
            callbacks=[],
            turns=[],
            retrying=None,
            history_limit=history_limit,
            include_history=include_history,
        ),
        availability=session_availability(loop),
        committed_model=loop.committed_model,
        markers=tuple(markers),
        log=project_session_log(loop),
        runtime=session_runtime_view(loop),
        session_dir=str(session_dir) if session_dir is not None else None,
        jobs_dir=str(jobs_dir) if jobs_dir is not None else None,
    )


def session_runtime_view(loop: AgentLoop) -> RuntimeSnapshot:
    """The runtime view a session read answers with, built where it lives.

    The configured model, the installed skills and tools, the live statistics
    and the MCP state are all resolved against this process's own config and
    tool manager, so the coordinator cannot assemble them from its side of the
    pipe. They travel with the snapshot payload instead, whose other fields
    are built here for the same reason.
    """
    issues, hooks_count = project_diagnostics(loop)
    return RuntimeSnapshot(
        config=project_config(loop),
        skills=project_skills(loop),
        tools=project_tools(loop),
        stats=project_stats(loop),
        context_window=_context_window(loop),
        issues=issues,
        hooks_count=hooks_count,
        mcp=project_mcp(loop, discovery_errors=loop.tool_manager.pop_mcp_errors()),
    )


def _context_window(loop: AgentLoop) -> int | None:
    try:
        return loop.config.get_active_model().context_limit
    except ValueError:
        return None


def session_availability(loop: AgentLoop) -> AvailabilitySnapshot:
    """The NON-MUTATING same-base eligibility snapshot the recovery gate reads.

    The gate's decision view is computed here rather than replicated as raw
    inputs: the coordinator receives the decision and the committed model, not
    the catalog, thinking level, compaction model and allowlist behind it.
    Missing evidence never blocks on its own: admission itself is the authority.
    """
    config = loop.config
    snapshot = config.catalog_snapshot
    committed = loop.committed_model
    if snapshot is None or committed is None:
        return AvailabilitySnapshot(
            available=True,
            permanent=False,
            endpoint_blocked=False,
            probe_in_flight=False,
            next_check_delay=0.0,
        )
    return availability_snapshot(
        snapshot=snapshot,
        committed=committed,
        registry=loop.config_orchestrator.availability_registry,
        config=config,
        history=reorder_for_tool_adjacency(select_model_context(loop.messages)),
        thinking=config.get_active_model().thinking,
        compaction_base=config.compaction_model or None,
        allowed_models=config.allowed_models,
    )


def marker_verdict(
    snapshot: AgentSessionSnapshot,
    *,
    request_id: str,
    deadline: float,
    incarnation_id: str | None = None,
) -> MarkerVerdict:
    """Judge one long-operation request against one consumed snapshot payload.

    The proof keys to CONSTRUCTION, not consumption (S8-B1): only a payload
    built after the request's deadline can prove absence, because a payload
    that left the agent earlier may simply predate the admission. A payload
    from another incarnation proves nothing either -- its table was cleared at
    the rebind -- so both cases answer ``stale`` rather than an absence claim.

    A record the payload DOES carry is always evidence, whichever incarnation
    built the payload: an identity rebind clears the marker table except the
    rebind operation's own TERMINAL record, which keeps its old incarnation's
    op id, so a replacement's re-check reads its own seed out of the new
    incarnation's table.
    """
    if snapshot.constructed_at < deadline:
        return MarkerVerdict(
            state=MARKER_STALE,
            detail="the payload was constructed before the request's deadline",
        )
    marker = snapshot.marker_for(request_id)
    if marker is not None:
        if marker.state == OP_STATE_TERMINAL:
            return MarkerVerdict(
                state=MARKER_TERMINAL, outcome=marker.outcome, op_id=marker.op_id
            )
        return MarkerVerdict(state=OP_STATE_ACTIVE, op_id=marker.op_id)
    if incarnation_id is not None and snapshot.incarnation_id != incarnation_id:
        return MarkerVerdict(
            state=MARKER_STALE,
            detail="the payload was constructed by another incarnation",
        )
    return MarkerVerdict(
        state=MARKER_ABSENT,
        detail="no record for the request id in a post-deadline payload",
    )
