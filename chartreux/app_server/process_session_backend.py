"""The production stdio topology: a spawned root agent owns the root session.

The coordinator stays the app-server in this arrangement -- it still answers the
client's protocol -- but the session the client talks to is not a local
``AgentLoop``. It is the ``python -m chartreux.app_server.root_agent`` process
this module spawns, and every session operation is answered by asking that
process: the open runs there, the snapshot is built there, the turn runs there
and streams its events back over the pipe.

What this module owns is the translation. The agent side speaks the
``agent/*`` wire; the app server speaks ``session/*`` and ``turn/*``. The
selection that put a session on this topology is resolved once, above, and
materialized into the spawn environment so the shipped env-keyed spawn guards
select the same route for the root and for every child it asks for.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Mapping,
    Sequence,
)
from contextlib import aclosing, suppress
from dataclasses import dataclass, replace as dataclass_replace
import os
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from chartreux.app_server._dispatch import DispatchResult
from chartreux.app_server._execution import cancel_tasks
from chartreux.app_server._handler import (
    session_fenced_backend_error,
    version_skew_backend_error,
)
from chartreux.app_server._model import ProtocolModel, validate_wire
from chartreux.app_server._policy_tree import (
    ManifestPolicyTreeJournal,
    PolicyChange,
    PolicyTreeError,
    PolicyTreeTransaction,
    ProcessPolicyNode,
)
from chartreux.app_server._projector import EventProjector, ProjectedUpdate
from chartreux.app_server._root_recovery import (
    RootRecoveryGate,
    RootRecoverySessionView,
)
from chartreux.app_server._session_backend_port import (
    SessionBackendError,
    SessionBackendEvent,
    SessionBackendResult,
    SessionEventSubscription,
    SessionForkResult,
    SessionLifecycleResult,
)
from chartreux.app_server._state import turns_page
from chartreux.app_server._streaming import finish_event_queue
from chartreux.app_server._tool_io import ClientToolIO
from chartreux.app_server._turn_input import vibe_content_blocks
from chartreux.app_server._turn_queue import (
    QueuedTurnRecord,
    TurnQueue,
    TurnQueueFullError,
    TurnQueueIdempotencyConflictError,
    TurnQueueItemNotFoundError,
)
from chartreux.app_server._turns import (
    CallbackClosedError,
    CallbackRejectedError,
    _record_retry_message_id,
)
from chartreux.app_server._utils import (
    DecodedInput,
    decode_content_blocks,
    decode_input,
    now_ms,
    public_error,
)
from chartreux.app_server._workspace import PromptPreparationError, WorkspaceTrustError
from chartreux.app_server.agent_snapshot import AgentSessionSnapshot
from chartreux.app_server.agent_wire import (
    AgentAuthorityChanged,
    AgentRecoveryEvidence,
    AgentRunCompletion,
    ShellRunRequest,
    decode_turn_failure,
)
from chartreux.app_server.events import (
    AppServerEvent,
    CallbackRequested,
    ClientProjection,
    EventSequenceError,
    UnknownNotificationError,
    parse_server_event,
)
from chartreux.app_server.models import (
    PublicCallbackEntry,
    PublicQueuedTurn,
    PublicSessionState,
    PublicTurn,
    PublicTurnQueue,
    PublicTurnStatus,
    PublicTurnStopReason,
    SessionTextContentBlock,
    UserDisplayContent,
    UserInputCallbackDetail,
    UserInputCallbackExpiry,
    UserInputCallbackOutput,
)
from chartreux.app_server.protocol import (
    AgentsCancelParams,
    AgentsInventoryReadParams,
    AgentsInventoryReadResponse,
    AgentsReconcileParams,
    AgentsReleaseParams,
    AgentsResumeParams,
    AgentsUpdateParams,
    AgentTranscriptGetParams,
    CallbackKeepWaitingParams,
    CallbackKeepWaitingResponse,
    CallbackOutput,
    CallbackResultError,
    CallbackResultParams,
    CallbackResultResponse,
    ConfigFieldsReadParams,
    ConfigMutationResponse,
    ConfigProxyReadParams,
    ConfigProxyWriteParams,
    ConfigReloadParams,
    ConfigWriteParams,
    ConfigWriteResponse,
    ContextInjectParams,
    ContextInjectResponse,
    DangerousCommandAuthorityReadParams,
    DangerousCommandAuthorityReadResponse,
    DangerousCommandRevokeParams,
    DangerousCommandRevokeResponse,
    DiagnosticsListParams,
    DiagnosticsLogsReadParams,
    EmptyResponse,
    IdentityReadParams,
    JsonPatchOperation,
    LoopsClearParams,
    LoopsCreateParams,
    LoopsDeleteParams,
    LoopsListParams,
    MCPAddParams,
    MCPAuthRequiredParams,
    MCPAuthUrlParams,
    MCPLoginParams,
    MCPLogoutParams,
    MCPReadParams,
    MCPRefreshParams,
    MCPRemoveParams,
    MCPToggleParams,
    Notification,
    PolicyReadParams,
    PolicyReadResponse,
    PolicyReplaceParams,
    PolicyReplaceResponse,
    ProtocolErrorCode,
    ReviewBaselineParams,
    ReviewHunksParams,
    ReviewMutationParams,
    ReviewStateParams,
    ReviewTurnDiffParams,
    RootRecoveryAdmitParams,
    RootRecoveryCancelParams,
    RootRecoveryScheduleParams,
    RootsReadParams,
    RootsReadResponse,
    RootsReplaceParams,
    RootsReplaceResponse,
    RootsSaveParams,
    RootsSaveResponse,
    RuntimeReadParams,
    RuntimeReadResponse,
    RuntimeSnapshot,
    RuntimeUpdatedParams,
    SessionCompactParams,
    SessionCompactResponse,
    SessionContinueParams,
    SessionForkParams,
    SessionHistoryClearParams,
    SessionHistoryClearResponse,
    SessionHistoryListParams,
    SessionListParams,
    SessionListResponse,
    SessionLogReadParams,
    SessionLogReadResponse,
    SessionMemorialItem,
    SessionPlanHandoffParams,
    SessionPlanHandoffResponse,
    SessionPlanRecord,
    SessionPlanRecordWriteParams,
    SessionPlanRecordWriteResponse,
    SessionReadParams,
    SessionReadResponse,
    SessionReadyReadParams,
    SessionReadyReadResponse,
    SessionReadyWaitParams,
    SessionReadyWaitResponse,
    SessionResumeParams,
    SessionSettingsUpdateParams,
    SessionShellCommandParams,
    SessionShellCommandResponse,
    SessionStartParams,
    SessionStopParams,
    SessionStopResponse,
    SessionTitleUpdateParams,
    SessionTitleUpdateResponse,
    SessionTurnsListParams,
    SessionUpdatedParams,
    SettingsReadParams,
    SkillsInstalledParams,
    SkillsListParams,
    StatsReadParams,
    ToolsListParams,
    TurnCompletedParams,
    TurnContextInputEntry,
    TurnEnqueueParams,
    TurnEnqueueResponse,
    TurnInterruptParams,
    TurnInterruptResponse,
    TurnQueueReadParams,
    TurnQueueReadResponse,
    TurnQueueRemoveParams,
    TurnQueueRemoveResponse,
    TurnQueueReplaceParams,
    TurnQueueReplaceResponse,
    TurnQueueResumeParams,
    TurnQueueResumeResponse,
    TurnQueueUpdatedParams,
    TurnShellInputEntry,
    TurnStartedParams,
    TurnStartParams,
    TurnStartResponse,
    TurnSteerParams,
    TurnSteerResponse,
    TurnUserInputEntry,
    WorkspacePromptPrepareParams,
)
from chartreux.app_server.registry_manifest import RegistryManifest
from chartreux.app_server.root_agent_process import (
    AgentProcessTurnFailed,
    RootAgentDeferredOperation,
    RootAgentExitedError,
    RootAgentProcess,
    RootAgentProcessError,
    RootAgentReconfigureConflict,
    RootAgentReplaceConflict,
    RootAgentRequestError,
    RootAgentSessionNotFound,
    RootFailureSignature,
    root_agent_command,
    spawn_root_agent_process,
)
from chartreux.app_server.subagent_ports import AgentOwnedSubagentRunner
from chartreux.app_server.transport import (
    STDIO_TRANSPORT,
    TRANSPORT_ENV_VAR,
    TransportSelection,
)
from chartreux.core.agent_loop._execution_port import (
    ExecutionTurnRequest,
    TurnOrderingConflict,
)
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.agent_loop._request_broker import (
    EXPIRED_CALLBACK_REASON,
    UserInputSettlement,
)
from chartreux.core.config.models import RootRecoveryConfig
from chartreux.core.events import (
    AssistantEvent,
    SessionTitleUpdatedEvent,
    UserInputRequestEvent,
)
from chartreux.core.job_launcher import JOB_GROUP_RECORD_SUFFIX, read_job_group_record
from chartreux.core.llm_models import ManualShellContext
from chartreux.core.model_catalog.availability import AvailabilitySnapshot
from chartreux.core.session.image_snapshot import ImageSnapshotError
from chartreux.core.subagents import AgentAvailability, RunStatus, RunStopReason
from chartreux.core.utils.async_subprocess import (
    ProcessGroupSettlement,
    RecordedJobGroup,
    settle_recorded_job_group,
)
from chartreux.observability.logging import logger
from chartreux.questions import UserQuestionRequest, UserQuestionResult

type SessionBackendServicesRef = Any
"""The app-server services this backend publishes through (duck-typed)."""

#: How long one spawned root agent gets to answer its own readiness frame.
DEFAULT_READY_TIMEOUT = 30.0
#: How long one ordinary agent request may take before it is a typed timeout.
#: The shared 30-second ordinary-request default: the stdio route answers with
#: the same bound the client and the root port already use, not a private 60s.
DEFAULT_REQUEST_TIMEOUT = 30.0

STOP_SETTLE_TIMEOUT = 60.0
"""How long the coordinator's stop may spend settling callbacks and queued work.

The stop is a long operation: the server-side receipt budget for bringing open
callbacks and the turn queue to a terminal state BEFORE the ack. A stop whose
settlement outlives this bound is an unknown outcome -- reconciled through
``session/resume``, never resent.
"""


@dataclass(frozen=True, slots=True)
class TrackedJobOutcome:
    """What cleanup established about one job group the session recorded."""

    job_id: str
    settlement: ProcessGroupSettlement


@dataclass(frozen=True, slots=True)
class _SequenceAdvance:
    """Cursor-only echo of one event the projection drained.

    The projection consumed the notification and advanced its sequence
    cursor without materializing an event, so the stream's cursor must
    advance with it or the next queued publication reads as a gap.
    """

    event_id: int
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ForwardedNotification(SessionBackendEvent):
    """A drained notification the stream forwards to its subscriber.

    A drain means this backend's projection no longer indexes the entry,
    not that the wire client lost its copy: the client can hold older
    entries via Load-More backfill, so the drained update still belongs
    on the wire. The envelope rides behind the cursor echo and yields
    unnumbered -- a plain routing delivery with no event of its own.
    """

    # The widened type IS the class's purpose: this envelope has no event of
    # its own, only the raw routed notification.
    event: AppServerEvent | None = None  # pyright: ignore[reportIncompatibleVariableOverride]


class _StreamClosed:
    """Terminal marker for one backend event stream."""


type _QueuedEvent = SessionBackendEvent | _StreamClosed | _SequenceAdvance


@dataclass(frozen=True, slots=True)
class _QueuedShellRecord:
    """A manual ``!`` record waiting for its user turn on this route.

    ``committed`` marks a record whose text already reached the agent's
    transcript: its turn then runs on that message instead of committing it
    twice. ``starts_at`` is how many turns this end had launched when the
    record was retained: a later admission that finds more means intervening
    turns ran, and the record text is then the retried turn's own input.
    """

    text: str
    context: ManualShellContext
    committed: bool = False
    starts_at: int = 0


@dataclass(frozen=True, slots=True)
class _PromotedShellRecord:
    """A promoted record turn, remembered until that turn terminalizes."""

    session_id: str
    entry_id: str
    text: str
    context: ManualShellContext
    committed: bool
    created_at: int = 0
    starts_at: int = 0


def materialized_spawn_env(
    selection: TransportSelection, *, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment one resolved selection materializes for a spawn.

    A resolved stdio selection is written into the child's environment, so the
    shipped env-keyed spawn guards (the root's own selection check and the
    child mirror) select the route the session resolved -- including a
    config-selected stdio session that named no variable. A resolved
    in-process selection materializes nothing and spawns nothing.
    """
    env = dict(os.environ if environ is None else environ)
    if selection.kind == STDIO_TRANSPORT:
        env[TRANSPORT_ENV_VAR] = STDIO_TRANSPORT
    return env


def _spawn_diagnostic(operation: str, exc: BaseException) -> SessionBackendError:
    """The fail-fast diagnostic a spawn, handshake or open failure answers with.

    There is no silent fallback to the in-process route: a topology that could
    not start says so and names the escape, so the user restarts deliberately
    instead of continuing on a route they did not choose.
    """
    return SessionBackendError(
        ProtocolErrorCode.INTERNAL_ERROR,
        f"The stdio topology could not start {operation}: {exc}. "
        "Restart with CHARTREUX_TRANSPORT=in_process to run in one process.",
    )


def _uncovered(operation: str) -> SessionBackendError:
    """The typed refusal for a session operation the stdio route does not carry.

    Explicit rather than silent: an operation this topology cannot answer must
    fail with a diagnostic that names the escape, never hand back a default
    that would leave the client projecting a session state nothing produced.
    """
    return SessionBackendError(
        ProtocolErrorCode.NOT_IMPLEMENTED,
        f"{operation} is not carried over the stdio topology; "
        "restart with CHARTREUX_TRANSPORT=in_process to use it.",
    )


# The parameter model each carried review and catalog read validates with. Both
# groups cross to the owner as typed requests, so the coordinator's parse is the
# same one the in-process route applies before the owner ever sees the request.
_REVIEW_CARRIER_PARAMS: Mapping[str, type[ProtocolModel]] = {
    "review/state": ReviewStateParams,
    "review/baseline": ReviewBaselineParams,
    "review/turnDiff": ReviewTurnDiffParams,
    "review/hunks": ReviewHunksParams,
    "review/approve": ReviewMutationParams,
    "review/revert": ReviewMutationParams,
}

_CATALOG_CARRIER_PARAMS: Mapping[str, type[ProtocolModel]] = {
    "skills/list": SkillsListParams,
    "skills/installed": SkillsInstalledParams,
    "tools/list": ToolsListParams,
    "stats/read": StatsReadParams,
    "identity/read": IdentityReadParams,
}


class _CarrierParams(Protocol):
    """The one field every carried request in these sets is scoped by."""

    session_id: str


def _carrier_session_id(params: ProtocolModel) -> str:
    """The session a validated carried request names.

    Every request in the carried sets is session-scoped and models that field,
    so the validated model is the single place the name is taken from.
    """
    return cast(_CarrierParams, params).session_id


def _retained_public_turn(retained: _PromotedShellRecord) -> PublicQueuedTurn:
    """The pending queue entry a retained record is publicly visible as."""
    return PublicQueuedTurn(
        id=retained.entry_id,
        created_at=retained.created_at,
        entries=[
            TurnUserInputEntry(
                entry_id=retained.entry_id,
                content=[SessionTextContentBlock(text=retained.text)],
            )
        ],
    )


def _shell_input_entry(record: QueuedTurnRecord) -> TurnShellInputEntry | None:
    """The manual ``!`` command a queue item carries, if it is one."""
    for entry in record.params.entries:
        if isinstance(entry, TurnShellInputEntry):
            return entry
    return None


def _queued_turn_text(turn: PublicQueuedTurn) -> str | None:
    """The user-typed text a public queue item carries, if any."""
    texts: list[str] = []
    for entry in turn.entries:
        if isinstance(entry, TurnShellInputEntry):
            texts.append(entry.command)
        else:
            texts.extend(
                block.text
                for block in entry.content
                if isinstance(block, SessionTextContentBlock) and block.text
            )
    return "\n".join(texts) or None


def _completion_status_word(completion: AgentRunCompletion) -> str:
    """The completion-notification status word for one settled run.

    Mirrors ``_subagent_execution._completion_status_word`` exactly: the stop
    reason outranks the status for a budget stop, and the user cancellation
    outranks the plain "cancelled" the remaining terminal statuses read as.
    """
    stop_reason = completion.stop_reason
    return (
        "budget-stopped (budget exceeded)"
        if stop_reason == RunStopReason.BUDGET_EXCEEDED.value
        else "budget-stopped (budget unverifiable)"
        if stop_reason == RunStopReason.BUDGET_UNVERIFIABLE.value
        else "completed"
        if completion.status == RunStatus.COMPLETED.value
        else "failed"
        if completion.status == RunStatus.FAILED.value
        else "cancelled (by user)"
        if stop_reason == RunStopReason.USER_CANCELLED.value
        else "retasked"
        if stop_reason == RunStopReason.RETASKED.value
        else "cancelled"
    )


def completion_notification_text(completion: AgentRunCompletion) -> str:
    """The model-facing text one settled background run is announced with.

    Byte-identical in wording to the in-process registry's delivered
    notification (``_subagent_execution``'s completion builder): the agent's
    display name, its id, the run id, the observed status word, and the same
    two instructions -- fetch this run by its id, and keep working or end the
    turn rather than waiting.
    """
    return (
        f"Background agent {completion.name} "
        f"({completion.agent_id}, run {completion.run_id})"
        f" {_completion_status_word(completion)}."
        " Use get_agent_result to retrieve the result."
        " Retrieve this run by its run_id; continue independent work"
        " or end your turn for other running agents rather than"
        " waiting by default."
    )


class ProcessSessionBackendHost:
    """The session-backend host of a stdio session: it spawns the root agent."""

    def __init__(
        self,
        *,
        services: SessionBackendServicesRef,
        selection: TransportSelection,
        command: Callable[[], Sequence[str]] | None = None,
        environ: Mapping[str, str] | None = None,
        ready_timeout: float = DEFAULT_READY_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        host_handler: Any = None,
    ) -> None:
        if selection.kind != STDIO_TRANSPORT:
            raise ValueError("a process session host requires a stdio selection")
        self._services = services
        self._selection = selection
        self._command = command or root_agent_command
        self._environ = environ
        self._ready_timeout = ready_timeout
        self._request_timeout = request_timeout
        self._host_handler = host_handler
        self._backends: dict[str, ProcessSessionBackend] = {}
        # Session ids with a respawn in flight: a second resume for a dead
        # generation is fenced instead of spawning a second replacement.
        self._respawns: set[str] = set()
        self._closed = False

    async def start(self, params: SessionStartParams) -> SessionLifecycleResult:
        return await self._open(params, session_id=None, continue_latest=False)

    async def resume(self, params: SessionResumeParams) -> SessionLifecycleResult:
        """Resume a stored session, respawning a confirmed-dead root once.

        After confirmed root death an explicit resume retires the dead
        backend, spawns and handshakes exactly one replacement, and restores
        only durable session state; the dead generation's coordinator-owned
        queue and shell records travel as memorial state in the result,
        never re-admitted for execution. A resume against a still-live
        generation is an idempotent re-attach (the in-process arm's same-id
        behavior), which is also the reconcile path after an unknown outcome:
        resume is not a mutation, so retrying it never replays one.
        """
        session_id = params.session_id
        existing = self._backend_for(session_id)
        if existing is not None and not existing.is_dead:
            return SessionLifecycleResult(backend=existing)
        if session_id in self._respawns:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                f"A respawn of session {session_id} is already in flight; its "
                "resume receipt settles it — reconcile by resuming again or "
                "reading the session, never by resending a mutation",
            )
        memorial: tuple[SessionMemorialItem, ...] = ()
        death_receipt: str | None = None
        if existing is not None:
            # Confirmed root death: capture the dead generation's memorial
            # state and its typed death receipt, then fence the old generation
            # so nothing still addressed to it executes.
            memorial = existing.memorial_items()
            death = existing.process.death
            death_receipt = (
                death.diagnostic
                if death is not None
                else (
                    f"{RootFailureSignature.DEATH_RECEIPT.value}: the root agent "
                    f"generation serving session {session_id} is retired"
                )
            )
            existing.retire()
        self._respawns.add(session_id)
        try:
            result = await self._open(
                params, session_id=session_id, continue_latest=False
            )
        finally:
            self._respawns.discard(session_id)
        return SessionLifecycleResult(
            backend=result.backend,
            after_response=result.after_response,
            memorial=memorial,
            death_receipt=death_receipt,
        )

    async def continue_latest(
        self, params: SessionContinueParams
    ) -> SessionLifecycleResult:
        return await self._open(params, session_id=None, continue_latest=True)

    async def _open(
        self,
        params: SessionStartParams | SessionResumeParams | SessionContinueParams,
        *,
        session_id: str | None,
        continue_latest: bool,
    ) -> SessionLifecycleResult:
        if self._closed:
            raise RuntimeError("The process session backend host is closed")
        from chartreux.app_server.process_bootstrap import (
            AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY,
        )

        try:
            process = await spawn_root_agent_process(
                session_options=params.agent_config,
                client_info=self._services.client_info(),
                client_capabilities=self._services.client_capabilities(),
                session_id=session_id,
                continue_latest=continue_latest,
                env=materialized_spawn_env(self._selection, environ=self._environ),
                cwd=params.cwd,
                command=list(self._command()),
                ready_timeout=self._ready_timeout,
                request_timeout=self._request_timeout,
            )
        except BaseException as exc:
            # spawn_root_agent_process closes the partial tree on a failed
            # handshake or open; nothing survives to be adopted here.
            raise _spawn_diagnostic("the root agent", exc) from exc
        # G5: live child authority is MANDATORY on the agent-wire peers, so a
        # root that does not announce it is refused here instead of being
        # served: on such a root a process child would run on the static
        # inherited policy with no live ceiling and no revocation, which is
        # exactly the fail-open this carrier closes. The refusal is a version
        # skew -- both ends restart on the same chartreux build.
        if (
            AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY
            not in process.advertised_capabilities
        ):
            await process.close()
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                "the root agent does not announce the mandatory live-child-authority "
                "capability; restart both ends on the same chartreux version -- no "
                "session was opened",
            )
        try:
            backend = ProcessSessionBackend(
                process, services=self._services, state=await self._open_state(process)
            )
        except BaseException:
            await process.close()
            raise
        self._backends[backend.session_id] = backend
        return SessionLifecycleResult(backend=backend)

    async def _open_state(self, process: RootAgentProcess) -> AgentSessionSnapshot:
        """The session the agent just opened, read back as the client's state."""
        return await process.snapshot()

    async def fork(self, params: SessionForkParams) -> SessionForkResult:
        raise _uncovered("session/fork")

    async def list(self, params: SessionListParams) -> SessionListResponse:
        """List the stored sessions the open agent's own config resolves.

        A list before any session is open has no agent to ask, so it is
        answered by the coordinator's host handler exactly as the in-process
        host answers it; without one the request stays a typed refusal.
        """
        backend = next(iter(self._backends.values()), None)
        if backend is not None:
            return await backend.list_sessions(params)
        if self._host_handler is None:
            raise _uncovered("session/list")
        dispatched = await self._host_handler.dispatch(
            "session/list", params.model_dump(mode="json")
        )
        return dispatched.response

    def _backend_for(self, session_id: str) -> ProcessSessionBackend | None:
        backend = self._backends.get(session_id)
        if backend is None:
            # A handoff rotates the backend's own id after it was keyed here,
            # so a read addressed to the rotated id still finds its backend.
            backend = next(
                (b for b in self._backends.values() if b.session_id == session_id), None
            )
        return backend

    async def read(self, params: SessionReadParams) -> SessionReadResponse:
        backend = self._backend_for(params.session_id)
        if backend is None:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {params.session_id}"
            )
        return await backend.read(params)

    async def rename(
        self, params: SessionTitleUpdateParams
    ) -> SessionTitleUpdateResponse:
        backend = self._backend_for(params.session_id)
        if backend is None:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {params.session_id}"
            )
        return await backend.rename(params)

    async def stop_background_tasks(self, current: Any) -> list[BaseException]:
        return []

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        backends = list(self._backends.values())
        self._backends.clear()
        errors: list[BaseException] = []
        for backend in backends:
            try:
                await backend.shutdown()
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Failed to close process session backends", errors)


class ProcessSessionBackend:  # noqa: PLR0904 - the session port surface
    """One live root agent process, seen through the app server's session port."""

    def __init__(  # noqa: PLR0915 - one install step per bound port
        self,
        process: RootAgentProcess,
        *,
        services: SessionBackendServicesRef,
        state: AgentSessionSnapshot,
    ) -> None:
        self._process = process
        self._services = services
        self._events: asyncio.Queue[_QueuedEvent] = asyncio.Queue()
        self._events_idle = asyncio.Event()
        self._events_closed = asyncio.Event()
        self._events_lock = asyncio.Lock()
        self._events_subscribed = False
        self._projection: ClientProjection | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._notices_task: asyncio.Task[None] | None = None
        self._active_turn: PublicTurn | None = None
        # The turns this end published, keyed by turn id and in start order.
        # The agent mints no turn projection of its own -- every agent-built
        # state answers an empty turn list -- so these records are the only
        # copy of the live turn view, and an adoption that replaces the
        # projection wholesale re-seeds from them instead of wiping it.
        self._turn_records: dict[str, PublicTurn] = {}
        self._closed = False
        # Set when this generation is retired by a resume respawn: every
        # request and event still addressed to the old generation then answers
        # the typed death receipt instead of executing (old-generation fence).
        self._retired = False
        # The client's own file and terminal surface travels as the turn's
        # host tool-I/O port: the agent asks for it over the same pipe, so a
        # client that declared these capabilities keeps serving them exactly
        # as it does on the in-process route.
        self._tool_io = ClientToolIO(services)
        # The subagent runner offered with every turn. This topology's children
        # belong to the root agent process -- the launch resolves against the
        # loop that only that process holds -- so the offer grants the turn its
        # child capability and the agent runs the children itself.
        self._subagent_runner = AgentOwnedSubagentRunner()
        # The callbacks the agent opened and this end published: the entry and
        # the projector that holds it, keyed by the agent's request id.
        self._open_callbacks: dict[str, tuple[PublicCallbackEntry, EventProjector]] = {}
        # The turn queue is the coordinator's: the in-process controller that
        # owns one lives with the session runtime, and a stdio root has none, so
        # the FIFO, its pause state and its idempotency receipts are held here
        # while every promoted turn still runs in the agent process.
        self._turn_queue = TurnQueue()
        self._queue_lock = asyncio.Lock()
        self._queue_tasks: set[asyncio.Task[None]] = set()
        # A manual `!` command queues as a shell-input kind beside queued
        # prompts. The item ids already executing keep a second promote pass
        # from running an effect twice, and one completion future per operation
        # lets the shell command request await its own record.
        self._shell_claimed: set[str] = set()
        self._shell_completions: dict[str, asyncio.Future[None]] = {}
        # The record each queue item will become a user turn from, keyed by
        # queue item id; the record turn each promotion started (so a failed
        # turn can retain its record), and the retained records waiting for the
        # queue's next admission — in the queue, or beside it when it is full.
        self._shell_records: dict[str, _QueuedShellRecord] = {}
        self._promoted_shell_records: dict[str, _PromotedShellRecord] = {}
        self._shell_held: set[str] = set()
        self._shell_retained: dict[str, _PromotedShellRecord] = {}
        # How many turns this end has launched. A retained record remembers the
        # count at retention, so a later admission can tell whether other turns
        # ran in between (R3-A2): the record text then becomes the retried
        # turn's own user input instead of running on the later transcript.
        self._turn_starts = 0
        # Session rotations this end adopted or is awaiting: a settled plan
        # handoff maps the superseded id to the rotated one, and an in-flight
        # handoff names its source id, so a request addressed to either is
        # fenced instead of forwarded against the pre-rotation session.
        self._rotations: dict[str, str] = {}
        self._pending_handoffs: set[str] = set()
        # R2/OD3: the root automatic-recovery gate stays coordinator-local on
        # this topology too. Its loop-owned predicate inputs (live config,
        # transcript-integrity verdict, availability) cross the agent wire as
        # one evidence bundle per decision (primed into _recovery_evidence by
        # the recovery/root/* arms below); its session view is this end's own
        # coordinator state; the typed terminal projection arrives on the
        # agent/turn/ended frames and is captured into the gate by _drive_turn.
        self._recovery_evidence: AgentRecoveryEvidence | None = None
        self._recovery_notices: EventProjector | None = None
        self._recovery_notice_tasks: set[asyncio.Task[None]] = set()
        self._recovery_gate = RootRecoveryGate(
            config=self._recovery_config,
            session_view=self._recovery_session_view,
            availability=self._recovery_availability,
            stage_notice=self._stage_recovery_notice,
            emit_projected=self._emit_recovery_notice,
            transcript_integrity_verdict=self._recovery_integrity_verdict,
        )
        # What cleanup established about the job groups this session recorded;
        # empty until the shutdown consults the jobs directory.
        self.tracked_job_outcomes: tuple[TrackedJobOutcome, ...] = ()
        self._runtime = state.runtime
        self._log = state.log
        # The shared agent-sequence cursor this end has adopted so far: the
        # coordinator's own watermark numbers its publications, not the
        # agent's events, so adoption ordering keys on the agent's cursor.
        self._adopted_agent_sequence = state.agent_sequence
        # The agent settles a user-input request where its record lives; the
        # terminal projection is this end's, so the agent asks this backend to
        # publish it before its waiter unblocks.
        self._process.bind_callback_settlement_sink(self._publish_callback_settlement)
        # The agent emits the MCP auth notifications (authUrl, authRequired)
        # owner-side; this end relays them onto the coordinator's notification
        # pump so the client-facing mcp_catalog/* notifications keep their
        # in-process shape on this topology.
        self._process.bind_mcp_notification_sink(self._relay_mcp_notification)
        # F10: the root's ready frame advertised the update capability, so
        # the owner's child-inventory snapshots relay onto the client-facing
        # agents/update exactly as the in-process registry emits its own -- a
        # peer that did not announce the capability never has the sink bound.
        from chartreux.app_server.process_bootstrap import (
            AGENT_WIRE_CAPABILITY_ROOT_AGENTS_UPDATE,
            AGENT_WIRE_CAPABILITY_ROOT_RUN_COMPLETIONS,
        )

        if (
            AGENT_WIRE_CAPABILITY_ROOT_AGENTS_UPDATE
            in self._process.advertised_capabilities
        ):
            self._process.bind_agents_update_sink(self._relay_agents_update)
        # G1b: the parent wake is coordinator-owned. The owner reports each
        # settled background run over its completion carrier and this backend
        # drives the parent's next turn (or steers the one in flight); a root
        # that did not advertise the carrier never has the sink bound, so no
        # wake is ever reconstructed from a frame it cannot vouch for.
        self._wake_capable = (
            AGENT_WIRE_CAPABILITY_ROOT_RUN_COMPLETIONS
            in self._process.advertised_capabilities
        )
        if self._wake_capable:
            self._process.bind_run_completion_sink(self._handle_run_completion)
        # G5: the owner reports each authority revision it commits. This sink is
        # OBSERVABILITY only -- no decision and no enforcement rides it, because
        # the decision itself is the owner-side ``agent/authority/check`` its
        # children ask. The root always advertises the capability (a peer that
        # does not is refused at open), so the sink is bound unconditionally.
        self.authority_incarnation: str | None = None
        self.authority_revision: dict[str, int] | None = None
        self.authority_change_count = 0
        self._process.bind_authority_changed_sink(self._observe_authority_changed)
        # The runs this generation already woke for, keyed as the owner keys
        # its result store. At-most-once per settled run: a duplicate
        # notification, a release/resume race and a recovery sweep all read
        # this set, and the sweep re-derives a baseline before it diffs.
        self._delivered_wakes: set[tuple[str, str]] = set()
        # The baseline is read once, at install, before this end can admit a
        # turn: a generation installed over a root that already holds settled
        # runs must never re-wake them.
        self._wake_baseline: asyncio.Task[None] | None = (
            asyncio.create_task(
                self._capture_wake_baseline(), name="process-wake-baseline"
            )
            if self._wake_capable
            else None
        )
        # Wake text that could not start a turn of its own (the parent was
        # busy, had queued work, or the run was user-cancelled) rides the next
        # promoted turn instead of being dropped -- the in-process pending
        # injected message, carried by this end's own promotion.
        self._pending_wake_texts: list[str] = []
        # The typed owner key every session-scoped mcp_catalog carrier sends
        # with its request: the agent files the auth bindings under a handle
        # held for exactly as long as this key lives, released explicitly on
        # shutdown and re-keyed atomically across a plan-handoff rotation.
        self._mcp_owner_key = f"mcp-owner-{uuid4().hex}"
        self._adopt_state(state.state)
        self._events_idle.set()

    @property
    def session_id(self) -> str:
        session_id = self._process.session_id
        if session_id is None:
            raise RuntimeError("The root agent process has not opened a session")
        return session_id

    @property
    def process(self) -> RootAgentProcess:
        """The spawned root agent handle this backend drives."""
        return self._process

    @property
    def is_dead(self) -> bool:
        """Whether this generation is retired or its root is confirmed dead."""
        return self._retired or self._process.death is not None

    def retire(self) -> None:
        """Fence this generation: it was retired by a resume respawn.

        Every request still addressed to the old generation answers the typed
        death receipt, and no old-generation event is published any more; the
        replacement generation serves the session from here on.
        """
        self._retired = True

    def _generation_fence(self) -> SessionBackendError | None:
        """The typed death receipt a dead or retired generation answers with."""
        if not self.is_dead:
            return None
        death = self._process.death
        if death is not None:
            return SessionBackendError(
                ProtocolErrorCode.INTERNAL_ERROR, death.diagnostic
            )
        return SessionBackendError(
            ProtocolErrorCode.INTERNAL_ERROR,
            f"{RootFailureSignature.DEATH_RECEIPT.value}: this root agent "
            "generation is retired; resume the session explicitly",
        )

    def memorial_items(self) -> tuple[SessionMemorialItem, ...]:
        """The coordinator-durable user-visible state of this generation.

        The turn queue and the shell records are coordinator-owned, so they
        survive a root death as MEMORIAL STATE ONLY: the resume receipt lists
        them here and the replacement generation never re-admits them for
        execution — the user re-submits from the receipt.
        """
        items: list[SessionMemorialItem] = []
        for turn in self._turn_queue.state.items:
            items.append(
                SessionMemorialItem(
                    kind="queued_turn",
                    id=turn.id,
                    created_at=turn.created_at,
                    text=_queued_turn_text(turn),
                )
            )
        for item_id, record in self._shell_records.items():
            items.append(
                SessionMemorialItem(
                    kind="queued_shell_record",
                    id=item_id,
                    text=record.text,
                    committed=record.committed,
                )
            )
        for record in self._promoted_shell_records.values():
            items.append(
                SessionMemorialItem(
                    kind="promoted_shell_record",
                    id=record.entry_id,
                    created_at=record.created_at,
                    text=record.text,
                    committed=True,
                )
            )
        for record in self._shell_retained.values():
            items.append(
                SessionMemorialItem(
                    kind="retained_shell_record",
                    id=record.entry_id,
                    created_at=record.created_at,
                    text=record.text,
                    committed=True,
                )
            )
        return tuple(items)

    # -- state -----------------------------------------------------------

    def _adopt_state(self, state: PublicSessionState) -> None:
        """Adopt one agent-built state as this stream's projection.

        Event ids are the app server's to mint: the agent numbers its own
        states, so the watermark the coordinator has published is stamped onto
        the adopted state and every later notification continues from it. The
        turns are this end's to retain: an agent-built state carries none, so
        adopting one as-is would wipe the live turn page, and the published
        records are re-seeded in the same stamp.
        """
        watermark = self._services.event_watermark(state.session.id)
        self._projection = ClientProjection(
            self._historyless_view(state).model_copy(
                update={
                    "event_id": watermark,
                    "turn_queue": self._public_queue_state(),
                    "turns": self._turn_records_for(state.session.id),
                }
            )
        )

    def _turn_records_for(self, session_id: str) -> list[PublicTurn]:
        """The turns published for ``session_id``, in the order they started.

        A settled turn REPLACES its record in place, so the page keeps the
        position its start took; a rotation leaves its predecessor's records
        behind and they are filtered out rather than served on the new id. An
        instance built without ``__init__`` (the stream tests build one from
        ``object.__new__``) has published nothing, so the store reads empty
        instead of raising.
        """
        records: dict[str, PublicTurn] = self.__dict__.get("_turn_records") or {}
        return [turn for turn in records.values() if turn.session_id == session_id]

    def _historyless_view(self, state: PublicSessionState) -> PublicSessionState:
        """Keep the subscribed history a history-less snapshot would drop.

        A read built with include_history=False answers a view choice, not an
        empty history: adopting it as-is would rebuild the projection's entry
        index from nothing, so every later history/entryUpdated for a
        pre-existing entry would drain (consume returns None while still
        advancing the cursor) and desynchronize the event stream. The
        retained history keeps that index alive; a snapshot that carries its
        own history still replaces it wholesale.
        """
        if state.history is not None:
            return state
        projection = self._projection
        if projection is None or projection.state.session.id != state.session.id:
            return state
        return state.model_copy(
            update={
                "history": projection.state.history,
                "history_before_cursor": projection.state.history_before_cursor,
            }
        )

    def _stamped(self, state: PublicSessionState) -> PublicSessionState:
        """Stamp the coordinator's watermark and queue onto an agent state.

        The agent process builds its own state and knows nothing of the queue:
        queued turns and their pause state are the coordinator's, so every
        state this end publishes carries its own view rather than the agent's
        empty default.
        """
        watermark = self._services.event_watermark(state.session.id)
        return state.model_copy(
            update={"event_id": watermark, "turn_queue": self._public_queue_state()}
        )

    def _public_queue_state(self) -> PublicTurnQueue:
        """The queue the client reads, with the records the queue could not take.

        A retained record is ready pending user work even while the queue has
        no room for it, so the public state names it instead of leaving it in
        backend-private state where a draining queue could strand it.
        """
        state = self._turn_queue.state
        if not self._shell_retained:
            return state
        return state.model_copy(
            update={
                "retained": [
                    _retained_public_turn(retained)
                    for retained in self._shell_retained.values()
                ]
            }
        )

    async def read(self, params: SessionReadParams) -> SessionReadResponse:
        self._require_session(params.session_id)
        snapshot = await self._process.snapshot(
            history_limit=params.history_limit, include_history=params.include_history
        )
        self._adopt_state(snapshot.state)
        self._runtime = snapshot.runtime
        self._log = snapshot.log
        state = self._stamped(snapshot.state)
        return SessionReadResponse(state=state, last_event_id=state.event_id)

    async def subscribe(self, params: SessionReadParams) -> SessionEventSubscription:
        self._require_session(params.session_id)
        self._ensure_notices_drained()
        async with self._events_lock:
            if self._events_subscribed:
                raise SessionBackendError(
                    ProtocolErrorCode.CONFLICT,
                    "The session backend already has an event subscriber",
                )
            response = await self.read(params)
            while not self._events.empty():
                self._events.get_nowait()
            self._events_closed.clear()
            self._events_subscribed = True
        return SessionEventSubscription(
            snapshot=response,
            events=self._event_stream(
                session_id=response.state.session.id,
                after_event_id=response.last_event_id,
            ),
        )

    async def _event_stream(
        self, *, session_id: str, after_event_id: int
    ) -> AsyncIterator[SessionBackendEvent]:
        try:
            if self._events.empty():
                self._events_idle.set()
            while True:
                queued = await self._events.get()
                try:
                    if isinstance(queued, _StreamClosed):
                        return
                    event_id = queued.event_id
                    if event_id is None:
                        # A cursor-only echo is always numbered, so the
                        # unnumbered yield is a plain backend event.
                        yield cast(SessionBackendEvent, queued)
                        continue
                    if queued.session_id is None:
                        raise RuntimeError("Numbered backend event has no session ID")
                    if queued.session_id != session_id:
                        session_id = queued.session_id
                        after_event_id = 0
                    if event_id <= after_event_id:
                        continue
                    expected_event_id = after_event_id + 1
                    if event_id != expected_event_id:
                        raise SessionBackendError(
                            ProtocolErrorCode.STALE_CURSOR,
                            "The session event stream has a gap: expected event"
                            f" {expected_event_id}, received event {event_id}",
                            data={
                                "expectedEventId": expected_event_id,
                                "receivedEventId": event_id,
                            },
                        )
                    after_event_id = event_id
                    if isinstance(queued, _SequenceAdvance):
                        # The projection drained the event behind this echo:
                        # the cursor moves on, nothing is yielded.
                        continue
                    yield queued
                finally:
                    if self._events.empty():
                        self._events_idle.set()
        finally:
            self._events_closed.set()
            async with self._events_lock:
                self._events_subscribed = False

    async def publish_notification(self, method: str, params: Any) -> bool:
        async with self._events_lock:
            # A retired or dead generation publishes no old-generation events;
            # the replacement generation owns the stream now.
            if (
                self.is_dead
                or self._events_closed.is_set()
                or not self._events_subscribed
                or self._projection is None
            ):
                return False
            notification = Notification(
                method=method, params=params.model_dump(mode="json", by_alias=True)
            )
            event = parse_server_event(notification)
            cursor_before: int | None = None
            if event is None:
                cursor_before = self._projection.last_event_id
                try:
                    event = self._projection.consume(
                        notification, dedupe_history_additions=False
                    )
                except UnknownNotificationError:
                    return False
                except EventSequenceError:
                    self._projection = None
                    self._events_idle.clear()
                    finish_event_queue(self._events, _StreamClosed())
                    return False
            if event is None:
                if (
                    cursor_before is None
                    or self._projection.last_event_id <= cursor_before
                ):
                    return False
                # A drained event consumed cleanly: the projection's cursor
                # advanced without materializing an event, so the stream
                # consumes the same advance instead of the coordinator
                # routing the numbered publication around a queue whose
                # subscriber is now one behind.
                advance = _SequenceAdvance(
                    event_id=self._projection.last_event_id,
                    session_id=getattr(params, "session_id", None),
                )
                self._events_idle.clear()
                if not await self._put_event(advance):
                    return False
                # The drain still swallowed a notification the wire client
                # owns a copy of -- its projection can hold the entry via
                # Load-More backfill -- so a forwarded envelope rides behind
                # the echo and the stream yields it unnumbered: the client
                # keeps the update instead of keeping a stale copy.
                queued: _QueuedEvent = _ForwardedNotification(
                    method=method,
                    params=params,
                    session_id=getattr(params, "session_id", None),
                )
            else:
                queued = SessionBackendEvent(
                    event=event,
                    method=method,
                    params=params,
                    session_id=getattr(params, "session_id", None),
                    event_id=getattr(params, "event_id", None),
                )
            self._events_idle.clear()
            return await self._put_event(queued)

    async def _put_event(self, event: _QueuedEvent) -> bool:
        if self.is_dead:
            return False
        put = asyncio.create_task(self._events.put(event))
        closed = asyncio.create_task(self._events_closed.wait())
        try:
            done, _ = await asyncio.wait(
                (put, closed), return_when=asyncio.FIRST_COMPLETED
            )
            if closed in done and self._events_closed.is_set():
                if not put.done():
                    put.cancel()
                return False
            await put
            return True
        finally:
            for task in (put, closed):
                if not task.done():
                    task.cancel()
            await asyncio.gather(put, closed, return_exceptions=True)

    async def flush_events(self) -> None:
        await self._events_idle.wait()

    def _ensure_notices_drained(self) -> None:
        """Start the one consumer of the agent's turn-less notices."""
        if self._notices_task is not None:
            return
        self._notices_task = asyncio.create_task(
            self._drain_notices(), name=f"process-notices-{self.session_id}"
        )

    async def _drain_notices(self) -> None:
        """Publish the agent's turn-less notices onto this client's stream.

        The title is the notice the coordinator can project without the tree
        state it does not own: it is session metadata, published as the single
        /title patch the in-process drain publishes.
        """
        async with aclosing(self._process.out_of_band_events()) as events:
            async for event in events:
                if isinstance(event, SessionTitleUpdatedEvent):
                    await self._publish_title(event)

    async def _publish_title(self, event: SessionTitleUpdatedEvent) -> None:
        if event.session_id != self.session_id:
            # A reset can swap in a new session while a title for the old one
            # is still queued; a stale title must not revive on the new one.
            return
        await self._services.notify(
            "session/updated",
            SessionUpdatedParams(
                event_id=0,
                session_id=event.session_id,
                patch=[
                    JsonPatchOperation(op="replace", path="/title", value=event.title)
                ],
                emitted_at=now_ms(),
            ),
        )

    def guard_request(self) -> None:
        fence = self._generation_fence()
        if fence is not None:
            # A dead or retired generation fails every in-flight request with
            # the typed death receipt instead of serving stale state.
            raise fence

    def runtime_updated_params(self) -> RuntimeUpdatedParams:
        return RuntimeUpdatedParams(session_id=self.session_id, runtime=self._runtime)

    def open_callbacks(self) -> list[PublicCallbackEntry]:
        """The callbacks this end published and the agent has not settled."""
        return [entry for entry, _ in self._open_callbacks.values()]

    async def publish_callback(self, callback: PublicCallbackEntry) -> bool:
        """Publish one opened callback onto this session's event stream.

        Answers whether the stream carried it: a client that is not subscribed
        gets the shipped direct delivery instead, exactly as on the in-process
        route. The projection is this end's, so the entry is already in it.
        A retired or dead generation publishes nothing: the old-generation
        fence refuses its callbacks with the death receipt.
        """
        if self.is_dead:
            return False
        async with self._events_lock:
            if not self._events_subscribed or self._projection is None:
                return False
            self._events_idle.clear()
            return await self._put_event(
                SessionBackendEvent(event=CallbackRequested(callback))
            )

    async def _open_callback(
        self, projector: EventProjector, event: UserInputRequestEvent
    ) -> None:
        """Publish one user-input request the agent opened for this client.

        The record and its settlement are the agent's; the prompt, the blocked
        status and the terminal patch are this end's, so the client sees the
        callback lifecycle it sees on the in-process route.
        """
        expiry = event.unattended_expiry
        if not isinstance(event.args, UserQuestionRequest):
            # The wire schema admits no other request shape, so this is a
            # driver defect: fail the waiter where the record lives rather than
            # publishing a prompt nothing can answer.
            self._process.reject_user_input(
                event.request_id,
                TypeError(f"Unsupported user-input request: {type(event.args)}"),
            )
            return
        detail = UserInputCallbackDetail(
            request=event.args,
            related_entry_id=event.tool_call_id,
            origin=event.origin.value,
            # The admission snapshot is the agent's (it owns the expiry
            # decision); this end only relays it with the prompt.
            unattended_expiry=(
                UserInputCallbackExpiry(
                    timeout_seconds=expiry.timeout_seconds, questions=expiry.questions
                )
                if expiry is not None
                else None
            ),
        )
        for update in projector.open_callback(
            event.request_id, detail, "User input required"
        ):
            await self._services.notify(update.method, update.params)
        entry = next(
            (
                candidate
                for candidate in projector.history
                if isinstance(candidate, PublicCallbackEntry)
                and candidate.callback_id == event.request_id
            ),
            None,
        )
        if entry is None:
            return
        self._open_callbacks[event.request_id] = (entry, projector)
        await self._emit_status("blocked", callback=entry)
        await self._services.publish_callback(entry)

    async def _publish_callback_settlement(
        self, request_id: str, settlement: UserInputSettlement
    ) -> bool:
        """Publish one terminal callback state the agent settled.

        Answers the agent's publication request, which it awaits BEFORE its
        waiter unblocks, so a client that answers a dead request sees the
        closed entry rather than a ghost prompt.
        """
        opened = self._open_callbacks.pop(request_id, None)
        if opened is None:
            return False
        _, projector = opened
        updates: list[ProjectedUpdate] = []
        status: str | None = None
        match settlement.kind:
            case "answered" if isinstance(settlement.result, UserQuestionResult):
                updates = projector.resolve_callback(
                    request_id, UserInputCallbackOutput(result=settlement.result)
                )
                status = "running"
            case "expired":
                updates = projector.expire_callback(request_id, EXPIRED_CALLBACK_REASON)
                status = "running"
            case _:
                updates = projector.close_callback(request_id, settlement.message or "")
        for update in updates:
            await self._services.notify(update.method, update.params)
        if status is not None:
            await self._emit_status(status)
        return bool(updates)

    async def _emit_status(
        self, status: str, *, callback: PublicCallbackEntry | None = None
    ) -> None:
        """Publish the session status the callback lifecycle moves through."""
        turn = self._active_turn
        if turn is None and status != "idle":
            return
        if status == "blocked" and callback is not None:
            value: Any = {
                "type": "blocked",
                "activeTurnId": turn.id if turn is not None else None,
                "callbackId": callback.callback_id,
                "reason": callback.detail.kind,
            }
        elif status == "running" and turn is not None:
            value = {"type": "running", "activeTurnId": turn.id}
        else:
            value = {"type": "idle"}
        await self._services.notify(
            "session/updated",
            SessionUpdatedParams(
                event_id=0,
                session_id=self.session_id,
                patch=[
                    JsonPatchOperation(op="replace", path="/status", value=value),
                    JsonPatchOperation(op="replace", path="/updatedAt", value=now_ms()),
                ],
                emitted_at=now_ms(),
            ),
        )

    async def reject_callback_delivery(
        self, session_id: str, callback_id: str, error: CallbackResultError
    ) -> None:
        """Fail one delivered callback where its request record lives."""
        self._require_session(session_id)
        try:
            outcome = await self._process.settle_user_input_rejection(
                callback_id, CallbackRejectedError(error.message)
            )
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        _settle_callback(callback_id, outcome)

    def references_child(self, session_id: str) -> bool:
        return False

    async def respond_to_callback(
        self, params: CallbackResultParams
    ) -> SessionBackendResult[CallbackResultResponse]:
        """Relay one callback answer to the agent that opened the request.

        The arbitration -- duplicate, expired, closed, wrong type, one winner
        under the record's resolution lock -- happens where the record lives,
        so this end answers with the outcome the agent already decided.
        """
        self._require_session(params.session_id)
        result = params.result
        if result.error is None and result.output is None:
            raise SessionBackendError(
                ProtocolErrorCode.INVALID_PARAMS,
                "Callback result must include output or error",
            )
        try:
            if result.error is not None:
                outcome = await self._process.settle_user_input_rejection(
                    params.callback_id, CallbackRejectedError(result.error.message)
                )
            else:
                output = TypeAdapter(CallbackOutput).validate_python(result.output)
                if not isinstance(output, UserInputCallbackOutput):
                    raise SessionBackendError(
                        ProtocolErrorCode.CONFLICT, "Callback answer has the wrong type"
                    )
                outcome = await self._process.settle_user_input_answer(
                    params.callback_id, output.result
                )
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        _settle_callback(params.callback_id, outcome)
        return SessionBackendResult(
            CallbackResultResponse(last_event_id=self._watermark())
        )

    async def list_sessions(self, params: SessionListParams) -> SessionListResponse:
        """List stored sessions from the config the agent's own loop holds."""
        return await self._process.list_sessions(params)

    async def read_plan_record(self) -> SessionPlanRecord | None:
        """Read the durable plan mode the agent's session metadata carries.

        A resume response carries this value, so it is answered before the
        session is adopted; a coordinator-side copy of the mode would be a
        second, unsynchronized record of the same fact.
        """
        return await self._process.read_plan_record()

    async def write_plan_record(
        self, params: SessionPlanRecordWriteParams
    ) -> SessionBackendResult[SessionPlanRecordWriteResponse]:
        """Persist (or remove) the plan mode where the transcript is written."""
        self._require_session(params.session_id)
        reply = await self._process.write_plan_record(params)
        return SessionBackendResult(reply)

    # -- turns -----------------------------------------------------------

    def _require_session(self, session_id: str) -> None:
        fence = self._generation_fence()
        if fence is not None:
            # Old-generation fence: a request still addressed to a retired or
            # dead generation gets the death receipt, never a stale answer.
            raise fence
        if session_id == self.session_id:
            return
        rotated_to = self._rotations.get(session_id)
        if rotated_to is not None:
            # The id was superseded by a rotation this end adopted: fence the
            # request with the rotated identity instead of a bare not-found.
            raise session_fenced_backend_error(session_id, rotated_to)
        raise SessionBackendError(
            ProtocolErrorCode.NOT_FOUND, f"Session not found: {session_id}"
        )

    async def plan_handoff(self, params: SessionPlanHandoffParams) -> DispatchResult:
        """Carry the consented plan handoff to the agent and adopt the rotation.

        The wire leg runs the full consent fence in the agent process, which
        rotates the session inside the operation; the reply's snapshot is the
        rotated identity this end adopts, and the dispatch flags are set for
        stdio explicitly (``session/fork``'s pattern) so the server resubscribes
        to the rotated session once the response has been sent.
        """
        self._require_session(params.session_id)
        source_id = params.session_id
        self._pending_handoffs.add(source_id)
        try:
            try:
                reply = await self._process.plan_handoff(params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            state = self._adopt_snapshot(reply.snapshot)
            if state is None:
                raise SessionBackendError(
                    ProtocolErrorCode.INTERNAL_ERROR,
                    "The plan handoff settled without the state it left behind",
                )
        finally:
            self._pending_handoffs.discard(source_id)
        rotated_id = self.session_id
        if rotated_id != source_id:
            self._rotations[source_id] = rotated_id
            # The rotation is a new session generation: the recovery gate's
            # episodes and projections belong to the superseded one, so they
            # are invalidated, never silently rearmed.
            self._recovery_gate.invalidate_all()
            # The typed MCP owner key atomically re-keys across the rotation:
            # the agent-side handle is unchanged, so the auth bindings survive
            # the rotation instead of being silently dropped mid-session.
            old_owner_key = self._mcp_owner_key
            new_owner_key = f"mcp-owner-{uuid4().hex}"
            try:
                await self._process.rekey_mcp_owner_key(
                    rotated_id, old_owner_key, new_owner_key
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            self._mcp_owner_key = new_owner_key
        return DispatchResult(
            SessionPlanHandoffResponse(state=state, session_log=self._log),
            session_attached=rotated_id != source_id,
        )

    async def rotated_session_id_for(self, session_id: str) -> str | None:
        """The rotated identity a superseded ``session_id`` maps to, if known.

        A settled handoff records the mapping directly. One still in flight is
        resolved from the agent's own snapshot, because the agent rotates inside
        the operation before its reply lands; a peer that cannot report a
        rotation (an older build without the fence) answers None, and the caller
        answers the version-skew error instead of a bare not-found.
        """
        rotated = self._rotations.get(session_id)
        if rotated is not None:
            return rotated
        if session_id in self._pending_handoffs:
            try:
                snapshot = await self._process.snapshot()
            except RootAgentProcessError:
                return None
            rotated = snapshot.state.session.id
            if rotated != session_id:
                return rotated
        return None

    async def _require_forwardable_session(self, session_id: str) -> None:
        """Fence a submit addressed to a session whose rotation is in flight.

        Between handoff-send and settlement the source id is still this end's
        current id, so the ordinary check passes; forwarding the submit would
        run it against the pre-rotation session, so a pending rotation fences
        it instead, with the rotated identity the agent's own snapshot reports.
        A peer that cannot report one is a version skew, never a bare
        not-found and never a silent retry.
        """
        if session_id in self._pending_handoffs:
            rotated = await self.rotated_session_id_for(session_id)
            if rotated is not None:
                raise session_fenced_backend_error(session_id, rotated)
            raise version_skew_backend_error(
                session_id, "the agent cannot report the rotated identity"
            )
        self._require_session(session_id)

    async def start_turn(
        self, params: TurnStartParams
    ) -> SessionBackendResult[TurnStartResponse]:
        await self._require_forwardable_session(params.session_id)
        if self._active_turn is not None:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                f"A turn is already running: {self._active_turn.id}",
            )
        if self._turn_queue:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                "Resume queued turns before starting a new turn",
            )
        decoded = decode_input(params, session_dir=self._session_dir())
        # G1b: a held wake rides the next turn whatever started it -- the
        # client's own start as much as a promoted queue item -- because the
        # in-process pending injected message has no other delivery point.
        await self._deliver_pending_wakes()
        turn = self._build_turn()
        self._active_turn = turn
        request = self._turn_request(
            turn,
            decoded,
            client_message_id=params.client_user_message_id,
            auto_title=params.auto_title,
            user_display_content=params.user_display_content,
            injected=params.injected,
            user_initiated_retry=params.user_initiated_retry,
        )
        await self._launch_turn(turn, request, idempotency_key=params.idempotency_key)
        return SessionBackendResult(
            TurnStartResponse(turn=turn, last_event_id=self._watermark())
        )

    def _build_turn(self, *, queue_item_id: str | None = None) -> PublicTurn:
        return PublicTurn(
            id=str(uuid4()),
            session_id=self.session_id,
            status=PublicTurnStatus.IN_PROGRESS,
            started_at=now_ms(),
            queue_item_id=queue_item_id,
        )

    def _turn_request(
        self,
        turn: PublicTurn,
        decoded: DecodedInput,
        *,
        client_message_id: str | None,
        auto_title: str | None,
        user_display_content: UserDisplayContent | None,
        injected: bool = False,
        user_initiated_retry: bool = False,
    ) -> ExecutionTurnRequest:
        return ExecutionTurnRequest(
            message=decoded.prompt,
            client_message_id=client_message_id,
            auto_title=auto_title,
            images=decoded.images or None,
            input_text=decoded.input_text,
            resources=decoded.resources or None,
            user_display_content=user_display_content,
            # The client's tool-I/O port rides the turn, and so does the
            # subagent runner: in this topology the root agent process owns its
            # children (one process each), so the offer is what grants the turn
            # its child capability and the agent runs them itself. The port is
            # never the one that runs a child here; a host request that still
            # reaches this end answers typed, naming that ownership.
            tool_io=self._tool_io,
            subagent_runner=self._subagent_runner,
            turn_options=AgentTurnOptions(
                turn_id=turn.id,
                injected=injected,
                user_initiated_retry=user_initiated_retry,
            ),
        )

    async def _launch_turn(
        self,
        turn: PublicTurn,
        request: ExecutionTurnRequest,
        *,
        idempotency_key: str | None = None,
    ) -> None:
        """Publish the start frame, then run the turn in the agent process."""
        await self._notify_turn("turn/started", turn)
        # R2: a start carrying an admitted continuation's idempotency key
        # links the successor to its episode; any other start is a user
        # takeover that cancels a pending schedule.
        self._observe_recovery_turn_started(turn.id, idempotency_key)
        self._turn_starts += 1
        self._turn_task = asyncio.create_task(
            self._drive_turn(turn, request), name=f"process-turn-{turn.id}"
        )

    async def _drive_turn(
        self, turn: PublicTurn, request: ExecutionTurnRequest
    ) -> None:
        status = PublicTurnStatus.COMPLETED
        error: Exception | None = None
        stop_reason: PublicTurnStopReason | None = None
        projector = EventProjector(self.session_id, turn.id)
        try:
            async with aclosing(self._process.run_turn(request)) as events:
                async for event in events:
                    if (
                        isinstance(event, AssistantEvent)
                        and event.stopped_by_middleware
                    ):
                        # The budget stop is the assistant event's own act; the
                        # terminal frame says only that the turn ended.
                        stop_reason = (
                            PublicTurnStopReason.BUDGET_UNVERIFIABLE
                            if event.middleware_metadata.get("code")
                            == "budget-unverifiable"
                            else PublicTurnStopReason.LIMIT
                        )
                    if isinstance(event, UserInputRequestEvent):
                        # The agent opened a user-input record; the prompt and
                        # its terminal state are this end's to publish.
                        await self._open_callback(projector, event)
                        continue
                    for update in projector.project(event):
                        await self._services.notify(update.method, update.params)
        except asyncio.CancelledError:
            status = PublicTurnStatus.INTERRUPTED
        except (AgentProcessTurnFailed, RootAgentExitedError) as exc:
            status = PublicTurnStatus.FAILED
            error = exc
        except TurnOrderingConflict as exc:
            status = PublicTurnStatus.FAILED
            error = exc
        except RootAgentProcessError as exc:
            status = PublicTurnStatus.FAILED
            error = exc
        finally:
            if status is PublicTurnStatus.FAILED:
                # R2: capture the typed terminal projection the agent attached
                # to its turn/ended frame (built agent-side BEFORE callback
                # cancellation) before the finish transition is observed.
                self._capture_recovery_turn_failure(turn.id)
            await self._observe_recovery_turn_finished(turn.id, status)
            completed = turn.model_copy(
                update={
                    "status": status,
                    "completed_at": now_ms(),
                    "error": _public_error(error),
                    "stop_reason": stop_reason,
                }
            )
            self._active_turn = None
            self._turn_task = None
            with suppress(Exception):
                await self._notify_turn("turn/completed", completed)
            if not self._closed:
                # The queue moves only after the turn's own terminal frame is
                # out: a promoted turn's start must not precede the completion
                # of the turn that made room for it.
                self._spawn_queue_task(self._after_turn_terminal(status, completed.id))

    async def _notify_turn(self, method: str, turn: PublicTurn) -> None:
        """Publish one turn frame and record the turn it projects.

        Both turn frames -- a start and its settlement -- are published here
        and nowhere else, so this is where the record store is kept: a start
        takes its place in publish order and a settlement replaces the same
        id in the same position.
        """
        self._turn_records[turn.id] = turn
        params_class = (
            TurnStartedParams if method == "turn/started" else TurnCompletedParams
        )
        await self._services.notify(
            method,
            params_class(
                event_id=0, session_id=turn.session_id, turn=turn, emitted_at=now_ms()
            ),
        )

    async def interrupt_turn(
        self, params: TurnInterruptParams
    ) -> SessionBackendResult[TurnInterruptResponse]:
        self._require_session(params.session_id)
        turn = self._active_turn
        if turn is not None:
            await self._process.interrupt_turn(turn.id)
        return SessionBackendResult(
            TurnInterruptResponse(last_event_id=self._watermark())
        )

    async def steer_turn(
        self, params: TurnSteerParams
    ) -> SessionBackendResult[TurnSteerResponse]:
        self._require_session(params.session_id)
        try:
            settlement = await self._process.steer(params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        if not settlement.committed:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT, "The steering delivery did not commit"
            )
        return SessionBackendResult(TurnSteerResponse(last_event_id=self._watermark()))

    async def inject_context(
        self, params: ContextInjectParams
    ) -> SessionBackendResult[ContextInjectResponse]:
        await self._require_forwardable_session(params.session_id)
        if self._active_turn is not None:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT, "Use turn/steer while a turn is active"
            )
        try:
            events = await self._process.inject(params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        # The agent returns the delivery's events in its reply and streams
        # nothing: projecting them here is what publishes the injection.
        projector = EventProjector(self.session_id, f"injection:{uuid4()}")
        for event in events:
            for update in projector.project(event, user_message_source="harness"):
                await self._services.notify(update.method, update.params)
        return SessionBackendResult(ContextInjectResponse(entries=projector.history))

    async def compact(
        self, params: SessionCompactParams
    ) -> SessionBackendResult[SessionCompactResponse]:
        self._require_session(params.session_id)
        try:
            record = await self._process.compact(params.extra_instructions)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        reply = record.reply
        if reply is None or reply.summary is None or reply.state is None:
            raise SessionBackendError(
                ProtocolErrorCode.COMPACTION_FAILED,
                f"Compaction settled without a published summary ({record.settlement})",
            )
        self._adopt_state(reply.state)
        return SessionBackendResult(
            SessionCompactResponse(
                summary=reply.summary,
                state=reply.state,
                session_log=await self._process.read_session_log(),
            )
        )

    # -- manual `!` commands ----------------------------------------------

    async def shell_command(
        self, params: SessionShellCommandParams
    ) -> SessionBackendResult[SessionShellCommandResponse]:
        """Run one manual ``!`` command as user input on this topology.

        The command is the user's own input, so it takes its FIFO place in this
        end's queue beside queued prompts instead of being refused while an
        epoch runs. At idle the queue asks the agent process -- which owns the
        loop, its working directory and its tool config -- for the command's
        record, and the record text becomes the ordinary user turn its queue
        item is replaced with.
        """
        self._require_session(params.session_id)
        if params.action == "interrupt":
            return await self._interrupt_shell_command(params)
        operation_id = params.operation_id or str(uuid4())
        completion = self._queue_shell_input(
            session_id=params.session_id,
            operation_id=operation_id,
            command=params.command or "",
            cwd=params.cwd,
            timeout_seconds=params.timeout_seconds or 30.0,
        )
        try:
            await completion
        except asyncio.CancelledError:
            # The request went away before its turn: a command that never ran
            # is dropped unexecuted rather than fired later for nobody. The
            # drop takes the queue lock, so it can never land inside a promote
            # pass that already peeked this queue's head.
            self._spawn_queue_task(
                self._discard_queued_shell_input_locked(operation_id)
            )
            raise
        return SessionBackendResult(
            SessionShellCommandResponse(last_event_id=self._watermark())
        )

    async def _interrupt_shell_command(
        self, params: SessionShellCommandParams
    ) -> SessionBackendResult[SessionShellCommandResponse]:
        """Drop one queued command unexecuted, or stop the one running."""
        operation_id = params.operation_id or ""
        if not operation_id:
            raise SessionBackendError(
                ProtocolErrorCode.INVALID_PARAMS,
                "operation_id is required for action='interrupt'",
            )
        async with self._queue_lock:
            discarded = self._discard_queued_shell_input(operation_id)
        if not discarded:
            try:
                await self._process.interrupt_shell(operation_id)
            except RootAgentProcessError:
                # A root that is closed or dead runs no command of the user's,
                # which is the outcome an unknown operation already has.
                pass
        return SessionBackendResult(
            SessionShellCommandResponse(last_event_id=self._watermark())
        )

    def _queue_shell_input(
        self,
        *,
        session_id: str,
        operation_id: str,
        command: str,
        cwd: str | None,
        timeout_seconds: float,
    ) -> asyncio.Future[None]:
        """Give one manual command its queue place and wait for its record."""
        self._turn_queue.enqueue(
            TurnEnqueueParams(
                session_id=session_id,
                entries=[
                    TurnShellInputEntry(
                        operation_id=operation_id,
                        command=command,
                        cwd=cwd,
                        timeout_seconds=timeout_seconds,
                    )
                ],
            )
        )
        completion: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._shell_completions[operation_id] = completion
        self._spawn_queue_task(self._after_queue_mutation(promote=True))
        return completion

    async def _discard_queued_shell_input_locked(self, operation_id: str) -> None:
        """Drop a queued command under the queue lock (cancelled request path)."""
        async with self._queue_lock:
            self._discard_queued_shell_input(operation_id)

    def _discard_queued_shell_input(self, operation_id: str) -> bool:
        """Drop a queued command that has not started; interrupt semantics.

        A command already executing is left to the agent's shell controller,
        which turns the interrupt into the interrupted record the user's turn
        carries.
        """
        record = next(
            (
                item
                for item in self._turn_queue.items
                if (entry := _shell_input_entry(item)) is not None
                and entry.operation_id == operation_id
            ),
            None,
        )
        if record is None or record.queued_turn.id in self._shell_claimed:
            return False
        self._turn_queue.remove(record.queued_turn.id)
        completion = self._shell_completions.pop(operation_id, None)
        if completion is not None and not completion.done():
            completion.set_result(None)
        self._spawn_queue_task(self._after_queue_mutation(promote=True))
        return True

    async def _run_queued_shell_input(
        self, record: QueuedTurnRecord, entry: TurnShellInputEntry
    ) -> None:
        """Ask the agent for one queued command's record, then make it input.

        The effect runs before the record is offered to admission, so a start
        that fails can only retain the record: the user's command is never
        rerun to recover it. The item keeps its own queue place as an ordinary
        user entry, which is why a retained record is visible work the next
        promote pass starts like any queued prompt.
        """
        completion = self._shell_completions.pop(entry.operation_id, None)
        try:
            try:
                text, context = await self._process.run_shell(
                    ShellRunRequest(
                        operation_id=entry.operation_id,
                        command=entry.command,
                        cwd=entry.cwd,
                        timeout_seconds=entry.timeout_seconds or 30.0,
                    )
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
        except asyncio.CancelledError:
            # The session went away mid-effect. The request awaiting this
            # record settles as cancelled here rather than waiting on a
            # promotion that will never come.
            if completion is not None and not completion.done():
                completion.cancel()
            raise
        except Exception as exc:
            self._turn_queue.remove(record.queued_turn.id)
            if completion is not None and not completion.done():
                completion.set_exception(exc)
            self._spawn_queue_task(self._after_queue_mutation(promote=True))
            return
        finally:
            self._shell_claimed.discard(record.queued_turn.id)

        # The record's text IS the user message this turn runs on. The context
        # beside it is display metadata the promotion commits with that message,
        # so the rebuilt timeline shows the shell tile the in-process route
        # shows; it never becomes input of its own.
        ready = TurnEnqueueParams(
            session_id=record.params.session_id,
            entries=[
                TurnUserInputEntry(
                    entry_id=entry.operation_id,
                    content=[SessionTextContentBlock(text=text)],
                )
            ],
        )
        async with self._queue_lock:
            queued_item_id: str | None = record.queued_turn.id
            try:
                self._turn_queue.replace(record.queued_turn.id, ready)
            except TurnQueueItemNotFoundError:
                # The user removed the queued command while its effect ran.
                # The effect is already the record's, so retain the record
                # instead of discarding an outcome that happened.
                try:
                    queued_item_id = self._turn_queue.enqueue(
                        ready
                    ).record.queued_turn.id
                except TurnQueueFullError:
                    # A full queue cannot take the place yet; the record stays
                    # retained for the next admission instead of being lost.
                    queued_item_id = None
            if queued_item_id is None:
                self._shell_retained[entry.operation_id] = _PromotedShellRecord(
                    session_id=record.params.session_id,
                    entry_id=entry.operation_id,
                    text=text,
                    context=context,
                    committed=False,
                    created_at=now_ms(),
                    starts_at=self._turn_starts,
                )
            else:
                self._shell_records[queued_item_id] = _QueuedShellRecord(
                    text=text, context=context
                )
            if completion is not None and not completion.done():
                completion.set_result(None)
            await self._emit_queue_updated()
            await self._promote_next()

    # -- turn queue -------------------------------------------------------

    def _decode_queued_entry(
        self, entry: TurnContextInputEntry | TurnUserInputEntry
    ) -> DecodedInput:
        return decode_content_blocks(
            vibe_content_blocks(entry.content), session_dir=self._session_dir()
        )

    def _validate_queued_input(self, params: TurnEnqueueParams) -> None:
        """Refuse a queued input this route cannot execute or cannot decode.

        The queue is this end's, so a queued input is checked before it takes a
        place: a shell input belongs to the shell command request that owns its
        execution and its record, and every content block is decoded here so a
        bad image or resource is refused up front instead of stranding the item
        until its promotion.
        """
        for entry in params.entries:
            if isinstance(entry, TurnShellInputEntry):
                raise SessionBackendError(
                    ProtocolErrorCode.CONFLICT,
                    "Shell inputs enter the queue through session/shellCommand",
                )
            self._decode_queued_entry(entry)

    async def enqueue_turn(
        self, params: TurnEnqueueParams
    ) -> SessionBackendResult[TurnEnqueueResponse]:
        await self._require_forwardable_session(params.session_id)
        try:
            result = self._turn_queue.enqueue(
                params, validate=self._validate_queued_input
            )
        except TurnQueueFullError as exc:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT, str(exc), {"maxItems": exc.max_items}
            ) from exc
        except TurnQueueIdempotencyConflictError as exc:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                str(exc),
                {"idempotencyKey": exc.idempotency_key},
            ) from exc
        except (ImageSnapshotError, PromptPreparationError, WorkspaceTrustError) as exc:
            raise SessionBackendError(
                ProtocolErrorCode.INVALID_PARAMS, str(exc)
            ) from exc
        return SessionBackendResult(
            TurnEnqueueResponse(queue_item_id=result.record.queued_turn.id),
            after_response=(
                None if result.duplicate else self._after_queue_response(promote=True)
            ),
        )

    async def read_turn_queue(
        self, params: TurnQueueReadParams
    ) -> SessionBackendResult[TurnQueueReadResponse]:
        self._require_session(params.session_id)
        return SessionBackendResult(
            TurnQueueReadResponse(queue=self._public_queue_state())
        )

    async def remove_queued_turn(
        self, params: TurnQueueRemoveParams
    ) -> SessionBackendResult[TurnQueueRemoveResponse]:
        self._require_session(params.session_id)
        # Serialized with promotion: a pass that peeked its head must be able
        # to pop that same item before a removal can change the FIFO order.
        async with self._queue_lock:
            removed = self._turn_queue.remove(params.queue_item_id)
        return SessionBackendResult(
            TurnQueueRemoveResponse(),
            after_response=(
                self._after_queue_response(promote=True) if removed else None
            ),
        )

    async def replace_queued_turn(
        self, params: TurnQueueReplaceParams
    ) -> SessionBackendResult[TurnQueueReplaceResponse]:
        self._require_session(params.session_id)
        try:
            async with self._queue_lock:
                result = self._turn_queue.replace(
                    params.queue_item_id,
                    params.as_enqueue_params(),
                    validate=self._validate_queued_input,
                )
        except TurnQueueItemNotFoundError as exc:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND,
                str(exc),
                {"queueItemId": exc.queue_item_id},
            ) from exc
        except TurnQueueIdempotencyConflictError as exc:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                str(exc),
                {"idempotencyKey": exc.idempotency_key},
            ) from exc
        except (ImageSnapshotError, PromptPreparationError, WorkspaceTrustError) as exc:
            raise SessionBackendError(
                ProtocolErrorCode.INVALID_PARAMS, str(exc)
            ) from exc
        return SessionBackendResult(
            TurnQueueReplaceResponse(queue_item_id=result.record.queued_turn.id),
            after_response=(
                None if result.duplicate else self._after_queue_response(promote=True)
            ),
        )

    async def resume_turn_queue(
        self, params: TurnQueueResumeParams
    ) -> SessionBackendResult[TurnQueueResumeResponse]:
        self._require_session(params.session_id)
        resumed = self._turn_queue.resume()
        # An explicit resume also offers pending work in a queue that was never
        # paused, exactly as the in-process controller does, including a
        # retained record an otherwise empty queue would never admit.
        promote = resumed or bool(self._turn_queue) or bool(self._shell_retained)
        return SessionBackendResult(
            TurnQueueResumeResponse(),
            after_response=(
                self._after_queue_response(promote=True) if promote else None
            ),
        )

    def _after_queue_response(self, *, promote: bool) -> Callable[[], None]:
        def after_response() -> None:
            self._spawn_queue_task(self._after_queue_mutation(promote=promote))

        return after_response

    def _spawn_queue_task(self, action: Coroutine[object, object, None]) -> None:
        task = asyncio.create_task(action, name="process-turn-queue")
        self._queue_tasks.add(task)
        task.add_done_callback(self._queue_task_finished)

    def _queue_task_finished(self, task: asyncio.Task[None]) -> None:
        self._queue_tasks.discard(task)
        if task.cancelled():
            return
        if error := task.exception():
            asyncio.get_running_loop().call_exception_handler({
                "message": "Turn queue task failed",
                "exception": error,
                "task": task,
            })

    async def _after_queue_mutation(self, *, promote: bool) -> None:
        async with self._queue_lock:
            # A queue mutation is the next admission the retained records were
            # waiting for.
            self._shell_held.clear()
            self._flush_retained_shell_records()
            await self._emit_queue_updated()
            if promote:
                await self._promote_next()

    async def _emit_queue_updated(self) -> None:
        await self._services.notify(
            "turn_queue_updated",
            TurnQueueUpdatedParams(
                event_id=0,
                session_id=self.session_id,
                queue=self._public_queue_state(),
                emitted_at=now_ms(),
            ),
        )

    async def _after_turn_terminal(
        self, status: PublicTurnStatus, turn_id: str
    ) -> None:
        async with self._queue_lock:
            # A terminal turn is the next admission: a record held for its own
            # failed turn's pass is promotable again, and a record retained
            # beside a full queue takes the place this turn left behind, so a
            # draining queue can never strand it.
            self._shell_held.clear()
            if status is PublicTurnStatus.INTERRUPTED:
                # An interrupted turn leaves the user's queued work waiting for
                # them: the queue pauses instead of promoting over the stop.
                self._promoted_shell_records.pop(turn_id, None)
                if self._turn_queue.pause():
                    await self._emit_queue_updated()
                return
            if status is PublicTurnStatus.FAILED:
                retained = self._retain_failed_shell_record(turn_id)
                # This terminal is the next admission. The record the failed
                # turn itself owned waits for a later one, but a record an
                # earlier failure left beside a full queue takes the place this
                # turn freed, so a failed intervening turn can never strand it.
                admitted = self._flush_retained_shell_records(
                    hold=() if retained is None else (retained,)
                )
                if retained is not None or admitted:
                    # The retained record is user-visible pending work.
                    await self._emit_queue_updated()
            else:
                self._promoted_shell_records.pop(turn_id, None)
                self._flush_retained_shell_records()
            # G1b: the terminal turn is the sweep's own admission point -- a
            # completion whose frame was lost wakes the parent here, before the
            # queue's next turn is admitted. A wake that starts a turn makes
            # the promote pass below a no-op, and a wake that cannot start one
            # is held for that promoted turn.
            await self._sweep_wake_gaps()
            await self._promote_next()

    def _retain_failed_shell_record(self, turn_id: str) -> str | None:
        """Keep a failed record turn's record as a ready user turn (B.2a).

        The effect already ran and its record is the user's message, so
        retention re-queues the record TEXT as an ordinary user entry and never
        the command: the effect is never rerun to recover its record. The record
        waits for the queue's next admission instead of retrying itself
        immediately, because the promote pass that follows this failure would
        repeat the turn that just failed.

        Returns the entry id of the record it retained, so the calling pass can
        hold exactly that record back.
        """
        pending = self._promoted_shell_records.pop(turn_id, None)
        if pending is None:
            return None
        self._shell_retained[pending.entry_id] = dataclass_replace(
            pending, starts_at=self._turn_starts
        )
        return pending.entry_id

    def _flush_retained_shell_records(
        self, *, hold: Collection[str] = frozenset()
    ) -> bool:
        """Offer every retained record the queue's next admission.

        Retention never depends on the queue having room: a record that cannot
        take a place yet stays retained here and the next admission offers it
        again, so an executed command's record is never dropped.

        ``hold`` names the records that must not run in the calling pass: they
        are admitted so they keep a visible place, but the promote pass that
        follows admits them to wait rather than repeat a failed turn. Every
        other admitted record is ready to run.

        Returns whether any record was admitted.
        """
        admitted = False
        for retained in tuple(self._shell_retained.values()):
            ready = TurnEnqueueParams(
                session_id=retained.session_id,
                entries=[
                    TurnUserInputEntry(
                        entry_id=retained.entry_id,
                        content=[SessionTextContentBlock(text=retained.text)],
                    )
                ],
            )
            try:
                queued_item_id = self._turn_queue.enqueue(ready).record.queued_turn.id
            except TurnQueueFullError:
                return admitted
            del self._shell_retained[retained.entry_id]
            self._shell_records[queued_item_id] = _QueuedShellRecord(
                text=retained.text,
                context=retained.context,
                committed=retained.committed,
                starts_at=retained.starts_at,
            )
            admitted = True
            if retained.entry_id in hold:
                self._shell_held.add(queued_item_id)
        return admitted

    async def _promote_next(self) -> None:
        """Start the next queued turn, over the pipe, as an ordinary turn.

        Promotion is an ordinary turn start: the agent process runs the turn it
        is given, and the queue item only leaves the FIFO once the agent has
        taken it, so a refused start leaves the item queued for the next pass.
        A context-only record has no prompt of its own, so its delivery is the
        promotion and the queue moves on afterwards.
        """
        if self._active_turn is not None:
            return
        record = self._turn_queue.peek_next(skip=self._shell_held)
        if record is None:
            return
        shell_entry = _shell_input_entry(record)
        if shell_entry is not None:
            # A shell input runs its effect at idle, exactly once, and only
            # then becomes the user turn its record is. The claim is what makes
            # a second promote pass (an enqueue landing mid-effect) a no-op.
            if record.queued_turn.id in self._shell_claimed:
                return
            self._shell_claimed.add(record.queued_turn.id)
            self._spawn_queue_task(self._run_queued_shell_input(record, shell_entry))
            return
        queued = record.params
        for entry in queued.entries:
            if not isinstance(entry, TurnContextInputEntry):
                continue
            # A queued context is delivered where the loop lives, before the
            # prompt it was queued with: the wire turn request carries no
            # queued-context field, so the delivery is its own request.
            await self.inject_context(
                ContextInjectParams(
                    session_id=self.session_id, input=vibe_content_blocks(entry.content)
                )
            )
        user_entry = queued.user_entry
        if user_entry is None:
            promoted = self._turn_queue.pop_next(skip=self._shell_held)
            if promoted is not record:
                # A concurrent mutation changed the head between the peek and
                # the pop. The popped item goes back to the front: a context
                # delivery must never cost another item its work.
                if promoted is not None:
                    self._turn_queue.restore(promoted)
                return
            await self._emit_queue_updated()
            await self._promote_next()
            return
        entry_id = user_entry.entry_id or record.queued_turn.id
        shell_record = self._shell_records.get(record.queued_turn.id)
        was_committed = shell_record is not None and shell_record.committed
        if shell_record is not None and not was_committed:
            # B.2 metadata carriage: the record message commits with the
            # record's context, so the timeline rebuild shows the shell tile
            # this command's outcome describes. The TEXT stays the source of
            # truth; the context is display metadata only.
            await self._commit_shell_record(entry_id, user_entry, shell_record)
            shell_record = dataclass_replace(shell_record, committed=True)
            self._shell_records[record.queued_turn.id] = shell_record
        # R3-A2: a committed record whose text already reached the transcript
        # runs on that message only while no other turn ran after its failure.
        # Once intervening turns ran, the record text itself is this turn's
        # input instead of whatever the later transcript ends with.
        redeliver = (
            was_committed
            and shell_record is not None
            and self._turn_starts > shell_record.starts_at
        )
        # G1b: the queued-turn arm of the wake. A completion that could not
        # start a turn of its own rides THIS turn, delivered where the loop
        # lives before the user's own input is launched -- the in-process
        # pending injected message's semantics.
        await self._deliver_pending_wakes()
        turn = self._build_turn(queue_item_id=record.queued_turn.id)
        self._active_turn = turn
        try:
            request = (
                self._record_turn_request(turn, entry_id, user_entry)
                if shell_record is not None and not redeliver
                else self._turn_request(
                    turn,
                    self._decode_queued_entry(user_entry),
                    client_message_id=(
                        _record_retry_message_id(turn.id) if redeliver else entry_id
                    ),
                    auto_title=None,
                    user_display_content=(
                        user_entry.annotations.chartreux_user_display_content
                    ),
                )
            )
            await self._launch_turn(
                turn, request, idempotency_key=queued.idempotency_key
            )
        except BaseException:
            self._active_turn = None
            raise
        promoted = self._turn_queue.pop_next(skip=self._shell_held)
        if promoted is not record:
            # A mutation landed between the peek and the pop (a removal is
            # serialized with this pass, so this is the last-resort path). The
            # item just popped returns to the front and the turn this pass
            # started keeps the record its failure would retain.
            if promoted is not None:
                self._turn_queue.restore(promoted)
        else:
            self._shell_held.discard(record.queued_turn.id)
            self._shell_records.pop(record.queued_turn.id, None)
        if shell_record is not None:
            # Item 5 B.2a: the promoted record turn's failure must be able to
            # retain the record it cannot publish.
            self._promoted_shell_records[turn.id] = _PromotedShellRecord(
                session_id=record.params.session_id,
                entry_id=entry_id,
                text=shell_record.text,
                context=shell_record.context,
                committed=shell_record.committed,
                created_at=now_ms(),
                starts_at=self._turn_starts,
            )
        await self._emit_queue_updated()

    async def _commit_shell_record(
        self,
        entry_id: str,
        user_entry: TurnUserInputEntry,
        shell_record: _QueuedShellRecord,
    ) -> None:
        """Commit the record's message, with its metadata, where the transcript is.

        The record text is an ordinary user message and its context is the
        timeline metadata beside it, so the delivery is the injection op the
        agent already carries a manual-shell record on. Committing before the
        turn starts is what lets the turn itself run on "no new input", which is
        also why a retained record is re-promoted without a second commit.
        """
        events = await self._process.inject(
            ContextInjectParams(
                session_id=self.session_id,
                input=vibe_content_blocks(user_entry.content),
                as_message=True,
                client_user_message_id=entry_id,
            ),
            manual_shell=shell_record.context,
        )
        projector = EventProjector(self.session_id, f"manual-shell:{entry_id}")
        for event in events:
            for update in projector.project(event):
                await self._services.notify(update.method, update.params)

    def _record_turn_request(
        self, turn: PublicTurn, entry_id: str, user_entry: TurnUserInputEntry
    ) -> ExecutionTurnRequest:
        """The turn a committed record runs on: no new input, its own message."""
        return ExecutionTurnRequest(
            message=None,
            client_message_id=entry_id,
            user_display_content=(
                user_entry.annotations.chartreux_user_display_content
            ),
            tool_io=self._tool_io,
            subagent_runner=self._subagent_runner,
            turn_options=AgentTurnOptions(turn_id=turn.id),
        )

    # -- settings and policy ---------------------------------------------

    def _adopt_snapshot(
        self, snapshot: AgentSessionSnapshot | None
    ) -> PublicSessionState | None:
        """Adopt the state one agent reply left behind, re-stamped here.

        A reconfigure reply's snapshot is built agent-side at reply time, so it
        carries the agent's own event numbering; the state this stream projects
        takes the coordinator's watermark instead, exactly as a read does. The
        adopted agent-sequence cursor only ever moves forward.
        """
        if snapshot is None:
            return None
        self._adopted_agent_sequence = max(
            self._adopted_agent_sequence, snapshot.agent_sequence
        )
        self._adopt_state(snapshot.state)
        self._runtime = snapshot.runtime
        self._log = snapshot.log
        return self._stamped(snapshot.state)

    def _adopt_read_snapshot(self, snapshot: AgentSessionSnapshot) -> bool:
        """Adopt one fresh read snapshot, monotone against relayed events.

        A read's snapshot can be built before an event the agent relays next,
        so it can be older than events already relayed; the coordinator's
        watermark cannot establish that ordering, so the shared agent-sequence
        cursor does: a snapshot older than any relayed event (or than the
        cursor already adopted) is never adopted, and the adopted state can
        never present older than events already relayed.
        """
        relayed = max(self._adopted_agent_sequence, self._process.last_agent_sequence)
        if snapshot.agent_sequence < relayed:
            return False
        self._adopt_snapshot(snapshot)
        return True

    async def update_settings(
        self, params: SessionSettingsUpdateParams
    ) -> SessionBackendResult[EmptyResponse]:
        """Apply the session bounds where the loop lives, and acknowledge.

        The mode is short: the agent sets the bounds under its own one-long-op
        fence and answers, so a refusal here is that fence answering and it
        travels back typed rather than being retried from this end.
        """
        self._require_session(params.session_id)
        try:
            await self._process.update_settings(
                max_turns=params.max_turns, max_tokens=params.max_tokens
            )
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        return SessionBackendResult(EmptyResponse(), runtime_updated=True)

    async def write_config(
        self, params: ConfigWriteParams
    ) -> SessionBackendResult[ConfigWriteResponse]:
        """Run one config transaction in the agent process as a long operation.

        The candidate crosses verbatim and the receipt is the SHIPPED result:
        the persistence x application composition, the rejected flag, the
        failure details, the revision, the saved values and the launch-metadata
        outcome all arrive as the transaction answered them.
        """
        self._require_session(params.session_id)
        try:
            reply = await self._process.write_config(params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        if reply.config_write is None:
            raise SessionBackendError(
                ProtocolErrorCode.INTERNAL_ERROR,
                "The config transaction settled without its shipped receipt",
            )
        self._adopt_snapshot(reply.snapshot)
        return SessionBackendResult(reply.config_write, runtime_updated=True)

    async def reload_config(
        self, params: ConfigReloadParams
    ) -> SessionBackendResult[ConfigMutationResponse]:
        """Reload the config inside the agent process, as the in-process leg does.

        The reload's prepared and applied halves run against the orchestrator
        and backend of the process that owns the loop; the receipt carries the
        runtime that process resolved.
        """
        self._require_session(params.session_id)
        try:
            reply = await self._process.reload_config(params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        return SessionBackendResult(reply, runtime_updated=True)

    async def read_policy(self, params: PolicyReadParams) -> PolicyReadResponse:
        """Read the policy the agent's own loop enforces.

        The revision is the tree token the agent's policy node reports, so a
        replacement compares like with like on either topology.
        """
        self._require_session(params.session_id)
        try:
            return await self._process.read_policy(params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc

    async def replace_policy(
        self, params: PolicyReplaceParams
    ) -> SessionBackendResult[PolicyReplaceResponse]:
        """Publish one policy replacement in the agent that owns the loop.

        A stdio session's tree has one node -- the agent's own policy node --
        so the transaction is journaled here and the four phases run there:
        the change is reserved, prepared, committed and retired where that
        authority lives, and this end only drives them.
        """
        self._require_session(params.session_id)
        session_dir = self._session_dir()
        if session_dir is None:
            # A session publishes its directory once it is persisted, and the
            # tree is journaled in that directory's manifest, so the agent is
            # asked which directory it now holds before anything reserves.
            try:
                snapshot = await self._process.snapshot()
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            self._adopt_snapshot(snapshot)
            if snapshot.session_dir is not None:
                session_dir = Path(snapshot.session_dir)
        if session_dir is None:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                "Policy replacement needs the session directory the tree is "
                "journaled in, and this session publishes none",
            )
        node = ProcessPolicyNode(params.session_id, self._process)
        try:
            current = await node.revision()
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        if current.tree_revision != params.expected_revision:
            # Checked before the first phase reserves: a stale request leaves
            # neither a journaled reservation nor a prepared node behind.
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT, "Stale accepted policy revision"
            )
        transaction_id = f"policy-{uuid4().hex}"
        transaction = PolicyTreeTransaction(
            transaction_id,
            nodes=[node],
            journal=ManifestPolicyTreeJournal(
                RegistryManifest(session_dir), transaction_id
            ),
            change=PolicyChange(
                source=params.source,
                tools={
                    name: value.model_dump(by_alias=False)
                    for name, value in params.tools.items()
                },
            ),
        )
        try:
            record = await transaction.run()
            self._adopt_snapshot(await self._process.snapshot())
        except PolicyTreeError as exc:
            raise SessionBackendError(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        # The accepted revision is the one the agent's committed node reports,
        # which is the token a client's own policy read answers with.
        committed = record.nodes[0].committed_revision
        return SessionBackendResult(
            response=PolicyReplaceResponse(
                revision=(
                    record.revision if committed is None else committed.tree_revision
                ),
                runtime=self._runtime,
            ),
            runtime_updated=True,
        )

    async def read_roots(self, params: RootsReadParams) -> RootsReadResponse:
        """Read the authorized roots the agent's own source composes."""
        self._require_session(params.session_id)
        try:
            return await self._process.read_roots(params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc

    async def replace_roots(
        self, params: RootsReplaceParams
    ) -> SessionBackendResult[RootsReplaceResponse]:
        """Refuse a roots replacement this topology cannot publish.

        A roots change rewrites the authorized set every node of the tree
        enforces, and a stdio session's root is its own process: the change
        has nowhere to be published from here, so it is refused typed and
        names the escape rather than half-applying.
        """
        raise SessionBackendError(
            ProtocolErrorCode.NOT_IMPLEMENTED,
            "Replacing the authorized roots across process-backed sessions is "
            "not wired in this release; run the session in-process "
            "(CHARTREUX_TRANSPORT=in_process) to change roots",
        )

    async def save_roots(self, params: RootsSaveParams) -> RootsSaveResponse:
        """Persist one approved root grant where the agent's config lives."""
        self._require_session(params.session_id)
        try:
            return await self._process.save_roots(params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc

    async def read_dangerous_command_authority(
        self, params: DangerousCommandAuthorityReadParams
    ) -> DangerousCommandAuthorityReadResponse:
        """Inspect the dangerous-command authority where the session's store lives.

        The live grants, the accepted saved grants, the fresh disk read and
        the authority generation belong to the agent's own loop, so the read
        is carried there and this end only forwards the reply — a
        coordinator copy of the authority is never read.
        """
        self._require_session(params.session_id)
        try:
            return await self._process.read_dangerous_command_authority(params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc

    async def revoke_dangerous_command_grant(
        self, params: DangerousCommandRevokeParams
    ) -> DangerousCommandRevokeResponse:
        """Revoke live and saved authority where the CAS judge lives.

        The revoke names the authority generation the read observed, and the
        agent checks it against its own store: a superseded generation is
        reported as the conflict outcome in the reply, never a false ack
        from a coordinator copy. The request is sent exactly once.
        """
        self._require_session(params.session_id)
        try:
            return await self._process.revoke_dangerous_command_grant(params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc

    # -- R2 root automatic recovery (coordinator-local gate) --------------

    def _recovery_config(self) -> RootRecoveryConfig:
        """The live [recovery.root] config from the primed wire evidence.

        Fail closed when no evidence is primed (a decision outside the
        recovery/root/* arms never reaches the gate): a disabled policy
        declines every schedule instead of guessing one.
        """
        evidence = self._recovery_evidence
        if evidence is None:
            return RootRecoveryConfig(enabled=False)
        return evidence.config

    def _recovery_availability(self) -> AvailabilitySnapshot:
        """The non-mutating availability snapshot from the primed evidence.

        Missing evidence never blocks on its own — the admission itself is
        the authority — exactly as the in-process read degrades.
        """
        evidence = self._recovery_evidence
        if evidence is None:
            return AvailabilitySnapshot(
                available=True,
                permanent=False,
                endpoint_blocked=False,
                probe_in_flight=False,
                next_check_delay=0.0,
            )
        return evidence.availability

    def _recovery_integrity_verdict(self) -> str | None:
        """The transcript-integrity decline reason carried on the wire."""
        evidence = self._recovery_evidence
        return evidence.integrity if evidence is not None else None

    def _recovery_session_view(self) -> RootRecoverySessionView:
        """The live coordinator-side interaction state the gate decides on
        (read at every decision, never snapshotted at construction).
        """
        return RootRecoverySessionView(
            session_id=self.session_id,
            active_turn_id=(
                self._active_turn.id if self._active_turn is not None else None
            ),
            queued_work=bool(self._turn_queue),
            open_callbacks=bool(self._open_callbacks),
            stop_requested=(
                self._turn_task is not None and self._turn_task.cancelling() > 0
            ),
            handoff_pending=bool(self._pending_handoffs),
            # Session lifecycle reservations (start/resume/continue) are served
            # by the backend host, not this attached backend, so none is
            # reserved here; the closing flag below covers this end's own
            # shutdown.
            lifecycle_reserved=False,
            closing=self._closed,
        )

    def _stage_recovery_notice(
        self, detail: Any, message: str
    ) -> ProjectedUpdate | None:
        """Stage one root-recovery milestone notice, exactly once.

        The deterministic entry id (``root-recovery:{milestone}:
        {episode_id}:{attempt}``) makes the emission idempotent at both the
        gate and this end's notice projector, mirroring the in-process
        staging.
        """
        if self._recovery_notices is None:
            self._recovery_notices = EventProjector(self.session_id, None)
        entry_id = (
            f"root-recovery:{detail.milestone}:{detail.episode_id}:{detail.attempt}"
        )
        if any(entry.id == entry_id for entry in self._recovery_notices.history):
            return None
        return self._recovery_notices.add_notice(
            entry_id, message=message, detail=detail
        )

    async def _emit_recovery_notice(self, update: ProjectedUpdate) -> None:
        """Deliver one staged recovery notice on this session's event stream."""
        await self._services.notify(update.method, update.params)

    def _deliver_recovery_notices(self, updates: list[ProjectedUpdate]) -> None:
        """Deliver staged notice updates (sync context: best-effort delivery,
        exactly as the in-process controller delivers them).
        """
        if not updates:
            return

        async def deliver() -> None:
            for update in updates:
                with suppress(Exception):
                    await self._emit_recovery_notice(update)

        task = asyncio.create_task(deliver())
        self._recovery_notice_tasks.add(task)
        task.add_done_callback(self._recovery_notice_tasks.discard)

    def _observe_recovery_turn_started(
        self, turn_id: str, idempotency_key: str | None
    ) -> None:
        """Feed one turn start to the gate (sync context). Total by contract:
        an observation failure never masks the turn start.
        """
        try:
            updates = self._recovery_gate.observe_turn_started(turn_id, idempotency_key)
        except Exception:
            logger.exception("root recovery turn-start observation failed")
            return
        self._deliver_recovery_notices(updates)

    async def _observe_recovery_turn_finished(
        self, turn_id: str, status: PublicTurnStatus
    ) -> None:
        """Feed one turn's terminal transition to the gate (async context).
        Total by contract: an observation failure never masks the
        terminalization it observes.
        """
        try:
            await self._recovery_gate.observe_turn_finished(turn_id, status)
        except Exception:
            logger.exception("root recovery turn-finish observation failed")

    def _capture_recovery_turn_failure(self, turn_id: str) -> None:
        """Capture the typed terminal projection the agent attached to its
        turn/ended frame (captured agent-side before callback cancellation)
        into the coordinator-local gate. Total by contract.
        """
        try:
            terminal = self._process.turn_terminal(turn_id)
            if terminal is None:
                return
            failure = decode_turn_failure(terminal)
            if failure is not None:
                self._recovery_gate.capture_terminal(failure)
        except Exception:
            logger.exception("root recovery projection capture failed")

    async def _decide_recovery(
        self, decide: Callable[[RootRecoveryGate], Awaitable[Any]]
    ) -> Any:
        """Run one gate decision with a freshly fetched evidence bundle.

        The loop-owned predicate inputs cross the agent wire once per
        decision — nothing is cached across a decision, and all predicates
        are rechecked at admission. A dead root answers the death receipt
        before any frame is sent; the request is never resent on an
        ambiguous outcome.
        """
        try:
            self._recovery_evidence = await self._process.recovery_evidence(
                self.session_id
            )
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        try:
            return await decide(self._recovery_gate)
        finally:
            self._recovery_evidence = None

    def update_mcp_projection(self, state: Any) -> None:
        """Fold one agent-resolved MCP projection into this end's runtime view.

        The coordinator holds only the projected view the owner returned: no
        orchestrator or catalog owner exists here to re-derive it from.
        """
        self._runtime = self._runtime.model_copy(update={"mcp": state})

    async def _relay_mcp_notification(
        self, method: str, payload: Mapping[str, Any]
    ) -> None:
        """Relay one agent-emitted MCP auth notification to the client.

        The agent emits ``agent/mcp/catalog/authUrl`` (the login's URL) and
        ``agent/mcp/catalog/authRequired`` (the owner-side gated handshake);
        this end publishes them under the client-facing method names on the
        coordinator's notification pump, exactly as the in-process route does.
        """
        from chartreux.app_server.agent_wire import (
            AGENT_MCP_AUTH_REQUIRED_METHOD,
            AGENT_MCP_AUTH_URL_METHOD,
        )

        if method == AGENT_MCP_AUTH_URL_METHOD:
            await self._services.notify(
                "mcp_catalog/authUrl", MCPAuthUrlParams.model_validate(dict(payload))
            )
            return
        if method == AGENT_MCP_AUTH_REQUIRED_METHOD:
            await self._services.notify(
                "mcp_catalog/authRequired",
                MCPAuthRequiredParams.model_validate(dict(payload)),
            )
            return
        logger.warning("unhandled agent MCP notification method %r", method)

    async def _relay_agents_update(
        self, method: str, payload: Mapping[str, Any]
    ) -> None:
        """Relay one owner-side child-inventory snapshot to the client (F10).

        The payload is the same ``AgentsInventoryReadResponse`` the inventory
        read serves, so the client-facing ``agents/update`` carries the
        process owner's honest subset verbatim: rows appear on spawn, states
        update with the run records, and a release or cancel removes the row
        -- while a mutation moment a process-backed handle cannot represent
        (an eviction tombstone above all) is never fabricated into one. An
        undecodable frame, or one that names anything other than this
        backend's session, is dropped as a presentation delta: the next
        mutation always carries a fresher full snapshot.
        """
        try:
            snapshot = AgentsInventoryReadResponse.model_validate(dict(payload))
        except ValidationError as exc:
            logger.warning("dropped an undecodable agent/agents/update: %s", exc)
            return
        if snapshot.session_id is None or snapshot.session_id != self.session_id:
            return
        await self._services.notify(
            "agents/update",
            AgentsUpdateParams(
                event_id=0,
                session_id=snapshot.session_id,
                emitted_at=now_ms(),
                agents=list(snapshot.agents),
                evictions=[],
            ),
        )

    # -- the parent wake for a settled child run (G1b) ----------------------
    #
    # The in-process route synthesizes the model-facing completion text and
    # either starts the parent's next turn (idle) or leaves it as a pending
    # injected message. On this topology the root process owns the children
    # and this backend owns turn admission, so the wake is driven here: the
    # owner's completion frame, the same text, and the same two arms.

    async def _observe_authority_changed(self, changed: AgentAuthorityChanged) -> None:
        """Record one owner authority-change snapshot (G5, observability).

        Nothing is enforced through this record: it states that the authority
        vector this end last saw is stale, names the owner incarnation that
        superseded it, and counts the changes so a diagnostic surface can tell
        a quiet owner from one whose policy is moving. A snapshot for another
        session is dropped -- a rebind never rewrites another session's record.
        """
        if changed.session_id != self.session_id:
            return
        self.authority_incarnation = changed.incarnation
        self.authority_revision = dict(changed.revision)
        self.authority_change_count += 1
        logger.debug(
            "root authority revision advanced: session=%s incarnation=%s vector=%s",
            changed.session_id,
            changed.incarnation,
            dict(changed.revision),
        )

    async def _handle_run_completion(self, completion: AgentRunCompletion) -> None:
        """Wake the parent for one background child run the owner settled.

        A foreign-session frame names another session and is dropped, a dead or
        retired generation wakes nothing, and a run wakes at most once per
        generation. ``USER_CANCELLED`` is not a wake -- the in-process registry
        still tells the parent the run stopped, but it never starts a turn for
        a stop the user asked for, and neither does this end.
        """
        if not self._wake_capable or self.is_dead:
            return
        if completion.session_id != self.session_id:
            return
        key = (completion.agent_id, completion.run_id)
        if key in self._delivered_wakes:
            return
        self._delivered_wakes.add(key)
        text = completion_notification_text(completion)
        if completion.stop_reason == RunStopReason.USER_CANCELLED.value:
            self._pending_wake_texts.append(text)
            return
        await self._drive_wake_text(text)

    async def _drive_wake_text(self, text: str) -> None:
        """Deliver one wake text: steer the live turn, else start an idle one.

        A turn in flight is steered, and a steer the agent refuses because its
        turn settled underneath the request falls back to the idle arm instead
        of dropping the wake. A turn started meanwhile, a non-empty queue, or a
        failed turn start leaves the text for the next promoted turn.
        """
        if self._active_turn is not None:
            if await self._steer_wake(text):
                return
        if self._active_turn is not None or self._turn_queue:
            self._pending_wake_texts.append(text)
            return
        try:
            await self._start_wake_turn(text)
        except (SessionBackendError, RootAgentProcessError) as exc:
            logger.warning("A completion wake could not start a turn: %s", exc)
            self._pending_wake_texts.append(text)

    async def _steer_wake(self, text: str) -> bool:
        """Steer one wake text into the turn in flight; False if it refused."""
        turn = self._active_turn
        if turn is None:
            return False
        try:
            settlement = await self._process.steer(
                TurnSteerParams(
                    session_id=self.session_id,
                    expected_turn_id=turn.id,
                    message=vibe_content_blocks([SessionTextContentBlock(text=text)]),
                    idempotency_key=f"run-completion-{uuid4().hex}",
                )
            )
        except RootAgentProcessError:
            return False
        return settlement.committed

    async def _start_wake_turn(self, text: str) -> None:
        """Run one wake text as an injected turn under the idle admission.

        The wake is the harness's own turn, not the user's: it carries the
        injected marker, no client message id and no idempotency key, so a
        reconnect never replays it and the client sees an ordinary turn
        (``turn/started`` then the run's own events).
        """
        turn = self._build_turn()
        self._active_turn = turn
        request = self._turn_request(
            turn,
            DecodedInput(prompt=text, input_text=text, images=[], resources=[]),
            client_message_id=None,
            auto_title=None,
            user_display_content=None,
            injected=True,
        )
        try:
            await self._launch_turn(turn, request)
        except BaseException:
            self._active_turn = None
            raise

    async def _deliver_pending_wakes(self) -> None:
        """Carry the held wake texts into the turn the queue is about to run.

        This is the queued-turn decision: a wake that could not start a turn
        of its own rides the next promoted turn as an injected user message,
        delivered before that turn launches -- the in-process pending injected
        message's own semantics. It is never a queue item the user would see
        and never a turn of its own. A failed delivery keeps the text for the
        next promotion; the contract is at-most-once plus the sweep, so a
        retry is preferred over a silently dropped wake.
        """
        pending = self._pending_wake_texts
        self._pending_wake_texts = []
        while pending:
            text = pending[0]
            try:
                await self.inject_context(
                    ContextInjectParams(
                        session_id=self.session_id,
                        input=vibe_content_blocks([SessionTextContentBlock(text=text)]),
                    )
                )
            except asyncio.CancelledError:
                self._pending_wake_texts = pending + self._pending_wake_texts
                raise
            except Exception as exc:
                logger.warning(
                    "A completion wake could not ride the promoted turn: %s", exc
                )
                self._pending_wake_texts = pending + self._pending_wake_texts
                return
            pending.pop(0)

    async def _sweep_wake_gaps(self) -> None:
        """Wake every settled run this generation has not woken yet (G1b).

        The notification carrier is at-most-once: a frame lost between the
        owner and this end would otherwise strand the parent's wake. The sweep
        re-derives the delivered set's baseline once, then diffs the owner's
        ledger of settled background runs against it, so a lost final frame
        costs at most one sweep. It runs at every turn terminal -- a point
        where this end is about to admit the next turn anyway.
        """
        if not self._wake_capable or self.is_dead:
            return
        baseline = self._wake_baseline
        if baseline is not None and not baseline.done():
            with suppress(BaseException):
                await baseline
        try:
            ledger = await self._process.read_run_completions(self.session_id)
        except (RootAgentProcessError, SessionBackendError) as exc:
            logger.warning("A completion sweep could not read the ledger: %s", exc)
            return
        for completion in ledger:
            await self._handle_run_completion(completion)

    async def _capture_wake_baseline(self) -> None:
        """Seed the delivered set from what the owner already settled (G1b).

        A generation installed over a root that already holds terminal child
        runs must not wake for them, so the terminal rows the inventory
        projection carries and the completions the ledger already holds are
        marked delivered without waking anything. The baseline is read once at
        install, before this end can admit a turn; a baseline that cannot be
        read marks nothing (waking a run twice is the safe direction, never
        waking it is not).
        """
        try:
            snapshot = await self._process.read_agent_inventory(
                self.session_id, AgentsInventoryReadParams()
            )
            ledger = await self._process.read_run_completions(self.session_id)
        except (RootAgentProcessError, SessionBackendError) as exc:
            logger.warning("A completion-wake baseline could not be read: %s", exc)
            return
        terminal = {AgentAvailability.CLOSED.value, AgentAvailability.EVICTED.value}
        for row in snapshot.agents:
            if row.availability in terminal and row.latest_run_id is not None:
                self._delivered_wakes.add((row.agent_id, row.latest_run_id))
        for completion in ledger:
            self._delivered_wakes.add((completion.agent_id, completion.run_id))

    async def prepare_mcp_reload(self) -> Any:
        """Run the config-reload prepare leg (restrictive suspensions) agent-side."""
        try:
            return await self._process.mcp_reload_prepare(
                self.session_id, self._mcp_owner_key
            )
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc

    async def finish_mcp_reload(self) -> Any:
        """Run the config-reload finish leg agent-side; adopt its runtime."""
        try:
            receipt = await self._process.mcp_reload_finish(
                self.session_id, self._mcp_owner_key
            )
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        if receipt.runtime is not None:
            self._runtime = RuntimeSnapshot.model_validate(receipt.runtime)
        return receipt

    async def fail_mcp_reload(self) -> None:
        """Run the config-reload failure-restoration leg agent-side."""
        try:
            await self._process.mcp_reload_fail(self.session_id, self._mcp_owner_key)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc

    async def read_mcp(self) -> Any:
        raise _uncovered("MCP state")

    async def reconfigure_mcp(
        self, configuration: Any, *, force_remote_discovery: bool
    ) -> Any:
        raise _uncovered("MCP reconfiguration")

    async def authorization_changed(
        self, *, name: str, descriptor_revision: str
    ) -> Any:
        raise _uncovered("MCP authorization changes")

    async def suspend_mcp(
        self, *, name: str, tool_name: str | None, reason: Any
    ) -> Any:
        raise _uncovered("MCP suspension")

    async def _mcp_catalog_carrier(
        self, forward: Callable[..., Awaitable[Any]], params: Any
    ) -> DispatchResult:
        """Forward one session-scoped mcp_catalog call to the agent owner.

        The typed owner key travels with the request, and the reply's runtime
        (when the mutation carries one) is adopted into this end's projection
        before the response is published, so the runtime/updated notification
        the server sends after the response carries the owner's own state.
        """
        addressed = getattr(params, "session_id", None)
        if addressed is not None:
            self._require_session(addressed)
        try:
            response = await forward(self.session_id, self._mcp_owner_key, params)
        except RootAgentProcessError as exc:
            raise _wire_failure(exc) from exc
        runtime = getattr(response, "runtime", None)
        if runtime is not None:
            self._runtime = runtime
            return DispatchResult(response, runtime_updated=True)
        return DispatchResult(response)

    async def dispatch_extension(  # noqa: PLR0911, PLR0912, PLR0915 - one arm per extension method
        self, method: str, raw_params: dict[str, Any]
    ) -> Any:
        if method == "session/plan/handoff":
            # The routing arm the stdio topology was missing: the consent
            # fence runs in the agent process over the existing reconfigure
            # carrier, and the reply's rotated identity is adopted here with
            # the attachment flag set, so the server resubscribes to the
            # rotated session once the response has been sent.
            return await self.plan_handoff(
                validate_wire(SessionPlanHandoffParams, raw_params)
            )
        # The runtime view is the one extension the agent's own snapshot
        # answers: it is resolved against the agent process's config and tool
        # manager, so every read asks the agent for a FRESH snapshot and the
        # reply travels in the payload rather than being assembled here from a
        # coordinator copy that would not exist -- and that a dead root would
        # otherwise keep answering from stale. The read is adopted with
        # monotone ordering against the events already relayed, and a dead
        # root answers the death receipt instead of a cached ready=True.
        if method == "runtime/read":
            params = RuntimeReadParams.model_validate(raw_params)
            self._require_session(params.session_id)
            try:
                snapshot = await self._process.snapshot(include_history=False)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            if self._adopt_read_snapshot(snapshot):
                runtime, log = snapshot.runtime, snapshot.log
            else:
                # Older than events already relayed: the adopted view stands,
                # never a stale read.
                runtime, log = self._runtime, self._log
            return DispatchResult(
                RuntimeReadResponse(runtime=runtime, session_log=log, ready=True)
            )
        if method in {"session/ready/wait", "session/ready/read"}:
            # Readiness is established by the open itself: the agent answers
            # its open reply only once the loop it owns exists, and this
            # backend is constructed from a snapshot of that loop, so while
            # the current generation is known live and open a read is
            # immediately ready and a wait has nothing left to wait for.
            # After a declared root death the read answers the death receipt
            # instead -- never a stale ready from a dead generation.
            if method == "session/ready/wait":
                params = SessionReadyWaitParams.model_validate(raw_params)
                response_class: Any = SessionReadyWaitResponse
            else:
                params = SessionReadyReadParams.model_validate(raw_params)
                response_class = SessionReadyReadResponse
            self._require_session(params.session_id)
            death = self._process.death
            if death is not None:
                raise _wire_failure(
                    RootAgentExitedError(
                        death.reason,
                        exit_code=death.exit_code,
                        signature=death.signature,
                    )
                )
            return DispatchResult(response_class(ready=True))
        if method == "session/log/read":
            # The snapshot payload carries the agent's own session-log
            # projection, so the read is answered from where the log lives
            # instead of from a coordinator copy of it.
            params = SessionLogReadParams.model_validate(raw_params)
            self._require_session(params.session_id)
            return DispatchResult(
                SessionLogReadResponse(log=await self._process.read_session_log())
            )
        if method == "session/history/clear":
            # The clear runs where the loop lives and answers with the state it
            # leaves behind: the coordinator rotates nothing itself, it adopts
            # the agent's own receipt so both ends address the same session.
            params = SessionHistoryClearParams.model_validate(raw_params)
            self._require_session(params.session_id)
            try:
                reply = await self._process.clear_history()
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            state = self._adopt_snapshot(reply.snapshot)
            if state is None:
                raise SessionBackendError(
                    ProtocolErrorCode.INTERNAL_ERROR,
                    "The history clear settled without the state it left behind",
                )
            return DispatchResult(
                SessionHistoryClearResponse(state=state, session_log=self._log)
            )
        if method == "session/history/list":
            # The attached session's live history is the agent's own
            # projection: the host reads persisted messages only, so a host
            # answer for an attached id would be stale or missing. The page
            # crosses to the loop that owns it; saved/non-attached ids still
            # reach the host through the server's narrowed not-found
            # fallback, and a dead-but-attached root answers the death
            # receipt, never the persisted projection.
            params = validate_wire(SessionHistoryListParams, raw_params)
            self._require_session(params.session_id)
            try:
                response = await self._process.history_list(self.session_id, params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "loops/list":
            # The schedule lives in the agent's own loop manager -- the one its
            # scheduler reads -- so the list is answered there. A coordinator
            # copy would be a second schedule with its own drift, and this end
            # adds no scheduler of its own.
            params = validate_wire(LoopsListParams, raw_params)
            self._require_session(params.session_id)
            try:
                response = await self._process.list_loops(self.session_id, params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "loops/create":
            # The create parses the interval, applies the per-session limit and
            # persists the new fire time where the schedule is kept; the reply
            # is the loop the owner recorded. It crosses once: a create that
            # timed out is an unknown outcome, never a blind retry.
            params = validate_wire(LoopsCreateParams, raw_params)
            self._require_session(params.session_id)
            try:
                response = await self._process.create_loop(self.session_id, params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "loops/delete":
            # An id the owner's schedule does not carry is its typed param
            # refusal, answered where the schedule is read.
            params = validate_wire(LoopsDeleteParams, raw_params)
            self._require_session(params.session_id)
            try:
                response = await self._process.delete_loop(self.session_id, params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "loops/clear":
            # The clear counts and persists where the schedule is kept, so the
            # reply is the owner's own count.
            params = validate_wire(LoopsClearParams, raw_params)
            self._require_session(params.session_id)
            try:
                response = await self._process.clear_loops(self.session_id, params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "diagnostics/list":
            # The issues and hook count are projected from the loop and hooks
            # the agent owns, so the read is answered there instead of being
            # assembled here from a coordinator copy that does not exist.
            params = validate_wire(DiagnosticsListParams, raw_params)
            self._require_session(params.session_id)
            try:
                response = await self._process.read_diagnostics(self.session_id, params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "diagnostics/logs/read":
            # The page is read through the owner's own log reader, so the
            # request's limit and offset and the shipped projection's
            # redaction apply unchanged; this end carries the page back and
            # merges nothing into it.
            params = validate_wire(DiagnosticsLogsReadParams, raw_params)
            self._require_session(params.session_id)
            try:
                response = await self._process.read_diagnostics_logs(
                    self.session_id, params
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method in _REVIEW_CARRIER_PARAMS:
            # Review checkpoints are session-local, memory-only state that the
            # loop's own recorder made, so the review manager holding them is
            # in the agent process: the request crosses and the owner's own
            # projection comes back. A mutation keeps the owner's idle
            # admission, and a session this backend does not hold is refused
            # before anything crosses.
            params = validate_wire(_REVIEW_CARRIER_PARAMS[method], raw_params)
            self._require_session(_carrier_session_id(params))
            try:
                response = await self._process.review(method, params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method in _CATALOG_CARRIER_PARAMS:
            # The skill and tool inventories, the live counters and the
            # session's identity are all resolved against the agent's own
            # config, tool manager and loop, so each read crosses and the
            # owner's projection comes back unchanged.
            params = validate_wire(_CATALOG_CARRIER_PARAMS[method], raw_params)
            self._require_session(_carrier_session_id(params))
            try:
                response = await self._process.catalog_read(method, params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "session/turns/list":
            # The live turn projection is this end's own: it is built from the
            # turn frames this end published, so the page is served from that
            # record store rather than from a projection a later adoption may
            # have replaced. A saved id is not this backend's, so it is
            # not-found here and the server's narrowed fallback serves it from
            # the host projection, exactly as a saved session read is served.
            params = validate_wire(SessionTurnsListParams, raw_params)
            self._require_session(params.session_id)
            return DispatchResult(
                turns_page(self._turn_records_for(params.session_id), params)
            )
        if method == "mcp_catalog/read":
            # The catalog projection is resolved where the orchestrator and
            # the auth bindings live -- in the agent process -- so the read
            # crosses with the typed owner key and the reply is the owner's
            # own projected view, never a coordinator re-derivation.
            return await self._mcp_catalog_carrier(
                self._process.mcp_catalog_read, validate_wire(MCPReadParams, raw_params)
            )
        if method == "mcp_catalog/refresh":
            # The refresh re-resolves servers and tools through the agent's
            # own orchestrator; this end adopts the runtime the owner returns.
            return await self._mcp_catalog_carrier(
                self._process.mcp_catalog_refresh,
                validate_wire(MCPRefreshParams, raw_params),
            )
        if method == "mcp_catalog/add":
            # A persisted add runs where the config orchestrator can reserve
            # and save; the mutation's runtime crosses back with the reply.
            return await self._mcp_catalog_carrier(
                self._process.mcp_catalog_add, validate_wire(MCPAddParams, raw_params)
            )
        if method == "mcp_catalog/toggle":
            # The enable/disable flip is revision-stamped by the owner's
            # orchestrator, so the toggle crosses instead of forking a copy.
            return await self._mcp_catalog_carrier(
                self._process.mcp_catalog_toggle,
                validate_wire(MCPToggleParams, raw_params),
            )
        if method == "mcp_catalog/remove":
            # The removal also drops the server's stored credentials, which
            # only the owner-side auth service can reach under the key.
            return await self._mcp_catalog_carrier(
                self._process.mcp_catalog_remove,
                validate_wire(MCPRemoveParams, raw_params),
            )
        if method == "mcp_catalog/login":
            # The OAuth login runs where the binding is filed under the owner
            # key; the authUrl it emits crosses as an agent notification and
            # is relayed to the client by the bound sink.
            return await self._mcp_catalog_carrier(
                self._process.mcp_catalog_login,
                validate_wire(MCPLoginParams, raw_params),
            )
        if method == "mcp_catalog/logout":
            # The logout removes the owner-side credentials under the same key.
            return await self._mcp_catalog_carrier(
                self._process.mcp_catalog_logout,
                validate_wire(MCPLogoutParams, raw_params),
            )
        if method == "workspace/prompt/prepare":
            # Prompt preparation is agent-side: it reads the live model, the
            # session's cwd and session directory and writes attachment
            # snapshots, none of which a coordinator copy could answer for.
            # The request crosses to the agent that owns the loop and the
            # reply is its own prepared prompt; a preparation that times out
            # is an unknown outcome and is never resent here.
            params = validate_wire(WorkspacePromptPrepareParams, raw_params)
            self._require_session(params.session_id)
            try:
                response = await self._process.prepare_prompt(params)
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "config/fields/read":
            # The field wires pair the agent's effective config with its
            # layered sources and their save revisions, so the read is
            # answered where the loop's orchestrator lives and this end only
            # carries the reply.
            params = validate_wire(ConfigFieldsReadParams, raw_params)
            self._require_session(params.session_id)
            try:
                return DispatchResult(await self._process.read_config_fields(params))
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
        if method == "config/settings/read":
            # The settings leaves are paired with the user revision the
            # agent's own layers carry; a coordinator copy holds neither.
            params = validate_wire(SettingsReadParams, raw_params)
            self._require_session(params.session_id)
            try:
                return DispatchResult(await self._process.read_settings(params))
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
        if method == "config/proxy/read":
            # The proxy view is read from the env file the agent process
            # resolves, so the value shown is the value that process uses.
            params = validate_wire(ConfigProxyReadParams, raw_params)
            self._require_session(params.session_id)
            try:
                return DispatchResult(await self._process.read_proxy_settings(params))
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
        if method == "config/proxy/write":
            # The write lands in the env file the agent process resolves; a
            # refused value keeps the in-process param refusal, and a busy
            # session the same conflict, because the agent's owner judged it.
            params = validate_wire(ConfigProxyWriteParams, raw_params)
            self._require_session(params.session_id)
            try:
                return DispatchResult(await self._process.write_proxy_settings(params))
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
        if method == "agents/cancel":
            # A run-pinned stop is the owner's decision: the process that
            # spawned the child holds its run handle, so the typed disposition
            # is answered there. The request is sent exactly once — a cancel
            # that times out is an unknown outcome, never a blind resend.
            params = validate_wire(AgentsCancelParams, raw_params)
            self._require_session(self.session_id)
            try:
                return DispatchResult(
                    await self._process.cancel_agent_run(self.session_id, params)
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
        if method == "agents/release":
            # The guarded release keeps the registry's rails: an active or
            # run-mismatched agent is a conflict that released nothing.
            params = validate_wire(AgentsReleaseParams, raw_params)
            self._require_session(self.session_id)
            try:
                return DispatchResult(
                    await self._process.release_agent(self.session_id, params)
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
        if method == "agents/resume":
            # Warm resume is answered by the process that keeps the child's
            # run records; a disposition it cannot honor is typed, not a
            # scheduled run this end would have to fake.
            params = validate_wire(AgentsResumeParams, raw_params)
            self._require_session(self.session_id)
            try:
                return DispatchResult(
                    await self._process.resume_agent(self.session_id, params)
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
        if method == "agents/reconcile":
            # The barrier belongs to the failed run's owner, so the evidence
            # is submitted there and its typed disposition carried back.
            params = validate_wire(AgentsReconcileParams, raw_params)
            self._require_session(self.session_id)
            try:
                return DispatchResult(
                    await self._process.reconcile_agent_result(self.session_id, params)
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
        if method == "agents/inventory/read":
            # The live inventory is read where the children live: the owner
            # answers one fresh typed request with the revision and the
            # projection it sampled together, so this end never serves a
            # coordinator copy of the inventory, and a dead root answers the
            # death receipt through the process failure instead of a stale
            # snapshot -- exactly like the runtime/read arm. The read is
            # control-class: a relay delta never sheds it.
            params = validate_wire(AgentsInventoryReadParams, raw_params)
            self._require_session(self.session_id)
            try:
                response = await self._process.read_agent_inventory(
                    self.session_id, params
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "agent/transcript/get":
            # A child transcript is scoped to the root session and lives where
            # the child runs: the owner reads the live page through the child's
            # own reader, or its saved transcript once the process is gone.
            params = validate_wire(AgentTranscriptGetParams, raw_params)
            self._require_session(self.session_id)
            try:
                response = await self._process.read_child_transcript_page(
                    self.session_id,
                    params.agent_id,
                    before=params.before,
                    limit=params.limit,
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            return DispatchResult(response)
        if method == "callback/keepWaiting":
            # The deadline decision is the agent's, taken under the request
            # record's lock, so this end only carries the answer back.
            params = CallbackKeepWaitingParams.model_validate(raw_params)
            self._require_session(params.session_id)
            try:
                outcome = await self._process.keep_waiting_user_input(
                    params.callback_id
                )
            except RootAgentProcessError as exc:
                raise _wire_failure(exc) from exc
            if outcome == "not_found":
                raise SessionBackendError(
                    ProtocolErrorCode.NOT_FOUND,
                    f"Callback not found: {params.callback_id}",
                )
            return DispatchResult(CallbackKeepWaitingResponse(status=outcome))
        if method == "session/shellCommand":
            # A manual `!` command is the user's own input, so it queues here
            # as a shell-input kind and the agent process answers its record
            # at idle: the effect, its redaction and its caps happen where the
            # loop lives, and the record text becomes the ordinary user turn.
            result = await self.shell_command(
                validate_wire(SessionShellCommandParams, raw_params)
            )
            return DispatchResult(result.response)
        if method == "recovery/root/schedule":
            # The gate stays coordinator-local: this end owns the authoritative
            # eligibility, episode accounting and single-flight admission, and
            # re-reads the loop-owned evidence (live config, transcript
            # integrity, availability) over the agent wire once for this
            # decision. The typed terminal projection arrived on the agent's
            # turn/ended frame. A dead root answers the death receipt.
            params = validate_wire(RootRecoveryScheduleParams, raw_params)
            self._require_session(params.session_id)
            response = await self._decide_recovery(lambda gate: gate.schedule(params))
            return DispatchResult(response)
        if method == "recovery/root/admit":
            # Single-flight admission with every predicate rechecked at
            # admission against a freshly fetched evidence bundle — a read-only
            # availability wait never consumes a retry, and an ambiguous
            # redelivery reconciles to the original successor, never a second
            # one. Never auto-replayed on an uncertain outcome.
            params = validate_wire(RootRecoveryAdmitParams, raw_params)
            self._require_session(params.session_id)
            response = await self._decide_recovery(lambda gate: gate.admit(params))
            return DispatchResult(response)
        if method == "recovery/root/cancel":
            # Cancelling a pending schedule reads only the coordinator-local
            # session view and episode, so no wire evidence is needed; it
            # never cancels an already-admitted successor.
            params = validate_wire(RootRecoveryCancelParams, raw_params)
            self._require_session(params.session_id)
            return DispatchResult(await self._recovery_gate.cancel(params))
        if method == "session/stop":
            # The coordinator stop lifecycle: answered here, in the
            # coordinator that owns the turn queue and the callback
            # projection, instead of the wire-declared refusal. Death is
            # checked first, callbacks and queued work settle before the ack,
            # and the root's termination and persistence run in the server's
            # close-attached-runtime step, ordered ahead of the ack frame.
            params = validate_wire(SessionStopParams, raw_params)
            self._require_session(params.session_id)
            return await self.stop_session(params)
        raise _uncovered(f"The {method} extension")

    async def rename(
        self, params: SessionTitleUpdateParams
    ) -> SessionTitleUpdateResponse:
        self._require_session(params.session_id)
        reply = await self._process.rename(params.title)
        self._projection_title(reply.title)
        return SessionTitleUpdateResponse(
            title=reply.title,
            updated_at=reply.updated_at,
            last_event_id=self._watermark(),
        )

    def _projection_title(self, title: str) -> None:
        """Adopt the title the agent just applied, so a later read agrees."""
        if self._projection is None:
            return
        state = self._projection.state
        self._projection = ClientProjection(
            state.model_copy(
                update={
                    "session": state.session.model_copy(update={"title": title}),
                    "event_id": state.event_id,
                }
            )
        )

    def _watermark(self) -> int:
        return self._services.event_watermark(self.session_id)

    def _session_dir(self) -> Path | None:
        session_dir = self._process.session_dir
        return None if session_dir is None else Path(session_dir)

    @property
    def tracked_jobs_dir(self) -> Path | None:
        """The jobs directory cleanup consults for this session, if it has one."""
        jobs_dir = self._process.jobs_dir
        return None if jobs_dir is None else Path(jobs_dir)

    async def settle_tracked_jobs(self) -> tuple[TrackedJobOutcome, ...]:
        """Settle every job group the session's jobs directory records.

        The directory consulted is the one the open reply or the most recent
        identity-transition reply published (S9-C4), so cleanup after a
        replacement consults the directory the session moved to and never the
        one it abandoned. Each group settles under the fail-closed continuity
        contract, and a record that cannot be read proves nothing: it is
        skipped rather than settled as if it named a live group.
        """
        directory = self.tracked_jobs_dir
        if directory is None or not directory.is_dir():
            return ()
        outcomes: list[TrackedJobOutcome] = []
        for path in sorted(directory.glob(f"*{JOB_GROUP_RECORD_SUFFIX}")):
            record = read_job_group_record(path)
            if record is None:
                continue
            settlement = await settle_recorded_job_group(RecordedJobGroup(record))
            outcomes.append(
                TrackedJobOutcome(job_id=record.job_id, settlement=settlement)
            )
        return tuple(outcomes)

    # -- lifecycle -------------------------------------------------------

    async def stop_session(self, params: SessionStopParams) -> DispatchResult:
        """The coordinator's stop, answered before the ack frame leaves.

        Death is checked first -- a dead-but-undetected root is re-declared
        through the liveness watcher and answers the typed death receipt
        instead of hanging to the request timeout. Open callbacks and the
        queued work then settle BEFORE the ack, under the stop's long-op
        budget; the root's termination and persistence run afterwards, in the
        server's close-attached-runtime step. A stop whose settlement
        outlives the budget is an unknown outcome: reconcile via
        ``session/resume``, never resend.
        """
        self._process.recheck_liveness()
        fence = self._generation_fence()
        if fence is not None:
            raise fence
        try:
            async with asyncio.timeout(STOP_SETTLE_TIMEOUT):
                await self._settle_open_callbacks_for_stop()
                await self._settle_turn_queue_for_stop()
        except TimeoutError as exc:
            raise SessionBackendError(
                ProtocolErrorCode.INTERNAL_ERROR,
                f"session/stop settlement did not finish within "
                f"{STOP_SETTLE_TIMEOUT:g}s; the stop outcome is unknown -- "
                "reconcile with session/resume, never resend",
            ) from exc
        return DispatchResult(SessionStopResponse())

    async def _settle_open_callbacks_for_stop(self) -> None:
        """Close every open callback before the stop ack, where its record lives."""
        for request_id in list(self._open_callbacks):
            try:
                await self._process.cancel_user_input(
                    request_id, CallbackClosedError("the session is stopping")
                )
            except RootAgentProcessError:
                # The root died under the settle: the generation fence above
                # already answers the death receipt, and the shutdown publishes
                # the terminal projection.
                return
        await self._emit_status("idle")

    async def _settle_turn_queue_for_stop(self) -> None:
        """Pause the coordinator-owned queue and stop in-flight promotions.

        The queue pauses instead of promoting over the stop; the queued items
        stay visible as memorial state. No second promote pass runs.
        """
        async with self._queue_lock:
            paused = self._turn_queue.pause()
            tasks = [*self._queue_tasks]
        if paused:
            await self._emit_queue_updated()
        if tasks:
            # A queue task mid-promotion holds an agent request; stopping it
            # keeps the pipe's teardown ordered behind its own callers.
            with suppress(Exception):
                await cancel_tasks(tasks, label="turn queue stop")

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._events_closed.set()
        # The wake baseline is a startup read nothing waits on past teardown.
        baseline = self._wake_baseline
        if baseline is not None and not baseline.done():
            baseline.cancel()
            with suppress(asyncio.CancelledError):
                await baseline
        # The session is going away: the recovery gate's references are
        # invalidated, never silently rearmed.
        self._recovery_gate.invalidate_all()
        turn = self._turn_task
        if turn is not None and not turn.done():
            turn.cancel()
            with suppress(asyncio.CancelledError):
                await turn
        notices = self._notices_task
        if notices is not None and not notices.done():
            notices.cancel()
            with suppress(asyncio.CancelledError):
                await notices
        recovery_notices = [*self._recovery_notice_tasks]
        if recovery_notices:
            with suppress(Exception):
                await cancel_tasks(recovery_notices, label="root recovery notices")
        queue_tasks = [*self._queue_tasks]
        if queue_tasks:
            # A queue task mid-promotion holds an agent request; stopping it
            # here keeps the pipe's teardown ordered behind its own callers.
            with suppress(Exception):
                await cancel_tasks(queue_tasks, label="turn queue")
        if self._events_subscribed:
            self._events_idle.clear()
            finish_event_queue(self._events, _StreamClosed())
        # The session end releases the typed MCP owner key explicitly: the
        # agent-side auth bindings filed under it are removed deterministically
        # now, never dropped mid-session by GC timing, before the pipe closes.
        with suppress(Exception):
            await self._process.release_mcp_owner_key(
                self.session_id, self._mcp_owner_key
            )
        await self._process.close()
        # The root is gone, so the groups its launcher shims started are
        # orphans: cleanup consults the jobs directory this session last
        # published and settles each recorded group, rather than trusting a
        # parent-death signal that the root's own death could never propagate
        # to a group it did not lead.
        self.tracked_job_outcomes = await self.settle_tracked_jobs()


_AGENT_REFUSAL_CODES: Mapping[str, ProtocolErrorCode] = {
    # The agent labels a typed refusal with the exception's own class name.
    # These are the delivery refusals whose in-process counterparts the app
    # server publishes as a conflict, so both routes answer the same code.
    "SteeringConflictError": ProtocolErrorCode.CONFLICT,
    "TurnConflictError": ProtocolErrorCode.CONFLICT,
    # A manual command refused before it would execute is the same param
    # refusal the in-process handler answers, whichever route runs it.
    "AgentShellRefusedError": ProtocolErrorCode.INVALID_PARAMS,
    # A prompt preparation the agent's own validation refused (too many or
    # unsupported images, an unreadable attachment) is the same param refusal
    # the in-process handler answers, whichever route runs it.
    "PromptPreparationError": ProtocolErrorCode.INVALID_PARAMS,
}


def _wire_failure(exc: RootAgentProcessError) -> SessionBackendError:
    """Translate one typed agent-side refusal into the app server's own error.

    The agent answers a request it will not honor with a typed wire failure;
    the app server publishes that refusal under the same protocol code, so a
    client reads one contract whichever route answered it.
    """
    if isinstance(exc, RootAgentDeferredOperation):
        return SessionBackendError(ProtocolErrorCode.NOT_IMPLEMENTED, str(exc))
    if isinstance(exc, RootAgentSessionNotFound):
        return SessionBackendError(ProtocolErrorCode.NOT_FOUND, str(exc))
    if isinstance(exc, RootAgentReplaceConflict | RootAgentReconfigureConflict):
        # Both refusals name an operation racing state the agent owns: the
        # in-process route answers them as a conflict, and so does this one.
        return SessionBackendError(ProtocolErrorCode.CONFLICT, str(exc))
    if not isinstance(exc, RootAgentRequestError):
        return SessionBackendError(ProtocolErrorCode.INTERNAL_ERROR, str(exc))
    code = _AGENT_REFUSAL_CODES.get(exc.code, ProtocolErrorCode.INTERNAL_ERROR)
    with suppress(ValueError):
        code = ProtocolErrorCode(exc.code)
    return SessionBackendError(code, exc.message)


def _settle_callback(callback_id: str, outcome: str | None) -> str:
    """Publish one agent-side callback arbitration outcome under its code.

    The agent's request broker decided (one winner under the record's lock);
    this end maps that decision to the same protocol codes the in-process
    controller raises, so a client reads one contract on either route.
    """
    match outcome:
        case "accepted" | "duplicate":
            return outcome
        case "not_found":
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND, f"Callback not found: {callback_id}"
            )
        case "expired":
            raise SessionBackendError(
                ProtocolErrorCode.CALLBACK_CLOSED,
                f"Callback expired unanswered: {callback_id}",
            )
        case "closed":
            raise SessionBackendError(
                ProtocolErrorCode.CALLBACK_CLOSED, f"Callback is closed: {callback_id}"
            )
        case "conflict":
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT, "Callback already has a different answer"
            )
    raise SessionBackendError(
        ProtocolErrorCode.INTERNAL_ERROR,
        f"The agent answered an unknown callback outcome: {outcome!r}",
    )


def _agent_message(text: str) -> str:
    """Strip the exception type the agent prefixes onto its terminal frame.

    The spawned agent composes its turn-failure diagnostic as
    ``f"{type(exc).__name__}: {exc}"`` so the log keeps the class; the public
    envelope carries the message alone, exactly as the in-process route does.
    """
    head, separator, rest = text.partition(": ")
    return rest if separator and head.isidentifier() else text


def _public_error(error: Exception | None) -> Any:
    if error is None:
        return None
    if isinstance(error, AgentProcessTurnFailed):
        # Strip before the public bound is applied: the bound would otherwise
        # spend the agent's type prefix out of the message's own length.
        return public_error(AgentProcessTurnFailed(_agent_message(str(error))))
    return public_error(error)


__all__ = [
    "ProcessSessionBackend",
    "ProcessSessionBackendHost",
    "TrackedJobOutcome",
    "materialized_spawn_env",
]
