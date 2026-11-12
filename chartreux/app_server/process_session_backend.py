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
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from chartreux.app_server._dispatch import DispatchResult
from chartreux.app_server._execution import cancel_tasks
from chartreux.app_server._model import validate_wire
from chartreux.app_server._policy_tree import (
    ManifestPolicyTreeJournal,
    PolicyChange,
    PolicyTreeError,
    PolicyTreeTransaction,
    ProcessPolicyNode,
)
from chartreux.app_server._projector import EventProjector, ProjectedUpdate
from chartreux.app_server._session_backend_port import (
    SessionBackendError,
    SessionBackendEvent,
    SessionBackendResult,
    SessionEventSubscription,
    SessionForkResult,
    SessionLifecycleResult,
)
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
from chartreux.app_server._turns import CallbackRejectedError, _record_retry_message_id
from chartreux.app_server._utils import (
    DecodedInput,
    decode_content_blocks,
    decode_input,
    now_ms,
    public_error,
)
from chartreux.app_server._workspace import PromptPreparationError, WorkspaceTrustError
from chartreux.app_server.agent_snapshot import AgentSessionSnapshot
from chartreux.app_server.agent_wire import ShellRunRequest
from chartreux.app_server.events import (
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
    CallbackKeepWaitingParams,
    CallbackKeepWaitingResponse,
    CallbackOutput,
    CallbackResultError,
    CallbackResultParams,
    CallbackResultResponse,
    ConfigMutationResponse,
    ConfigReloadParams,
    ConfigWriteParams,
    ConfigWriteResponse,
    ContextInjectParams,
    ContextInjectResponse,
    EmptyResponse,
    JsonPatchOperation,
    Notification,
    PolicyReadParams,
    PolicyReadResponse,
    PolicyReplaceParams,
    PolicyReplaceResponse,
    ProtocolErrorCode,
    RootsReadParams,
    RootsReadResponse,
    RootsReplaceParams,
    RootsReplaceResponse,
    RootsSaveParams,
    RootsSaveResponse,
    RuntimeReadParams,
    RuntimeReadResponse,
    RuntimeUpdatedParams,
    SessionCompactParams,
    SessionCompactResponse,
    SessionContinueParams,
    SessionForkParams,
    SessionHistoryClearParams,
    SessionHistoryClearResponse,
    SessionListParams,
    SessionListResponse,
    SessionLogReadParams,
    SessionLogReadResponse,
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
    SessionTitleUpdateParams,
    SessionTitleUpdateResponse,
    SessionUpdatedParams,
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
from chartreux.core.events import (
    AssistantEvent,
    SessionTitleUpdatedEvent,
    UserInputRequestEvent,
)
from chartreux.core.job_launcher import JOB_GROUP_RECORD_SUFFIX, read_job_group_record
from chartreux.core.llm_models import ManualShellContext
from chartreux.core.session.image_snapshot import ImageSnapshotError
from chartreux.core.utils.async_subprocess import (
    ProcessGroupSettlement,
    RecordedJobGroup,
    settle_recorded_job_group,
)
from chartreux.questions import UserQuestionRequest, UserQuestionResult

type SessionBackendServicesRef = Any
"""The app-server services this backend publishes through (duck-typed)."""

#: How long one spawned root agent gets to answer its own readiness frame.
DEFAULT_READY_TIMEOUT = 30.0
#: How long one ordinary agent request may take before it is a typed timeout.
DEFAULT_REQUEST_TIMEOUT = 60.0


@dataclass(frozen=True, slots=True)
class TrackedJobOutcome:
    """What cleanup established about one job group the session recorded."""

    job_id: str
    settlement: ProcessGroupSettlement


class _StreamClosed:
    """Terminal marker for one backend event stream."""


type _QueuedEvent = SessionBackendEvent | _StreamClosed


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
        self._closed = False

    async def start(self, params: SessionStartParams) -> SessionLifecycleResult:
        return await self._open(params, session_id=None, continue_latest=False)

    async def resume(self, params: SessionResumeParams) -> SessionLifecycleResult:
        return await self._open(
            params, session_id=params.session_id, continue_latest=False
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

    async def read(self, params: SessionReadParams) -> SessionReadResponse:
        backend = self._backends.get(params.session_id)
        if backend is None:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {params.session_id}"
            )
        return await backend.read(params)

    async def rename(
        self, params: SessionTitleUpdateParams
    ) -> SessionTitleUpdateResponse:
        backend = self._backends.get(params.session_id)
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

    def __init__(
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
        self._closed = False
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
        # What cleanup established about the job groups this session recorded;
        # empty until the shutdown consults the jobs directory.
        self.tracked_job_outcomes: tuple[TrackedJobOutcome, ...] = ()
        self._runtime = state.runtime
        self._log = state.log
        # The agent settles a user-input request where its record lives; the
        # terminal projection is this end's, so the agent asks this backend to
        # publish it before its waiter unblocks.
        self._process.bind_callback_settlement_sink(self._publish_callback_settlement)
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

    # -- state -----------------------------------------------------------

    def _adopt_state(self, state: PublicSessionState) -> None:
        """Adopt one agent-built state as this stream's projection.

        Event ids are the app server's to mint: the agent numbers its own
        states, so the watermark the coordinator has published is stamped onto
        the adopted state and every later notification continues from it.
        """
        watermark = self._services.event_watermark(state.session.id)
        self._projection = ClientProjection(
            state.model_copy(
                update={"event_id": watermark, "turn_queue": self._public_queue_state()}
            )
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
        if params.session_id != self.session_id:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {params.session_id}"
            )
        snapshot = await self._process.snapshot(
            history_limit=params.history_limit, include_history=params.include_history
        )
        self._adopt_state(snapshot.state)
        self._runtime = snapshot.runtime
        self._log = snapshot.log
        state = self._stamped(snapshot.state)
        return SessionReadResponse(state=state, last_event_id=state.event_id)

    async def subscribe(self, params: SessionReadParams) -> SessionEventSubscription:
        if params.session_id != self.session_id:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {params.session_id}"
            )
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
                        yield queued
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
                            "The session event stream has a gap",
                            data={
                                "expectedEventId": expected_event_id,
                                "receivedEventId": event_id,
                            },
                        )
                    after_event_id = event_id
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
            if self._events_closed.is_set() or not self._events_subscribed:
                return False
            if self._projection is None:
                return False
            notification = Notification(
                method=method, params=params.model_dump(mode="json", by_alias=True)
            )
            event = parse_server_event(notification)
            if event is None:
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
                return False
            self._events_idle.clear()
            return await self._put_event(
                SessionBackendEvent(
                    event=event,
                    method=method,
                    params=params,
                    session_id=getattr(params, "session_id", None),
                    event_id=getattr(params, "event_id", None),
                )
            )

    async def _put_event(self, event: _QueuedEvent) -> bool:
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
        return None

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
        """
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
        if session_id != self.session_id:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {session_id}"
            )

    async def start_turn(
        self, params: TurnStartParams
    ) -> SessionBackendResult[TurnStartResponse]:
        self._require_session(params.session_id)
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
        await self._launch_turn(turn, request)
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
        self, turn: PublicTurn, request: ExecutionTurnRequest
    ) -> None:
        """Publish the start frame, then run the turn in the agent process."""
        await self._notify_turn("turn/started", turn)
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
        self._require_session(params.session_id)
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
        self._require_session(params.session_id)
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
            await self._launch_turn(turn, request)
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
        takes the coordinator's watermark instead, exactly as a read does.
        """
        if snapshot is None:
            return None
        self._adopt_state(snapshot.state)
        self._runtime = snapshot.runtime
        self._log = snapshot.log
        return self._stamped(snapshot.state)

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

    async def read_dangerous_command_authority(self, params: Any) -> Any:
        raise _uncovered("The dangerous-command authority read")

    async def revoke_dangerous_command_grant(self, params: Any) -> Any:
        raise _uncovered("A dangerous-command grant revocation")

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

    async def dispatch_extension(self, method: str, raw_params: dict[str, Any]) -> Any:
        # The runtime view is the one extension the agent's own snapshot
        # answers: it is resolved against the agent process's config and tool
        # manager, so it travels in the payload rather than being assembled
        # here from a coordinator copy that would not exist.
        if method == "runtime/read":
            params = RuntimeReadParams.model_validate(raw_params)
            if params.session_id != self.session_id:
                raise SessionBackendError(
                    ProtocolErrorCode.NOT_FOUND,
                    f"Session not found: {params.session_id}",
                )
            return DispatchResult(
                RuntimeReadResponse(
                    runtime=self._runtime, session_log=self._log, ready=True
                )
            )
        if method in {"session/ready/wait", "session/ready/read"}:
            # Readiness is established by the open itself: the agent answers
            # its open reply only once the loop it owns exists, and this
            # backend is constructed from a snapshot of that loop. A read is
            # therefore trivially ready, and a wait has nothing left to wait
            # for -- the coordinator's own construction elapsed it.
            if method == "session/ready/wait":
                params = SessionReadyWaitParams.model_validate(raw_params)
                response_class: Any = SessionReadyWaitResponse
            else:
                params = SessionReadyReadParams.model_validate(raw_params)
                response_class = SessionReadyReadResponse
            if params.session_id != self.session_id:
                raise SessionBackendError(
                    ProtocolErrorCode.NOT_FOUND,
                    f"Session not found: {params.session_id}",
                )
            return DispatchResult(response_class())
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

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._events_closed.set()
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
        queue_tasks = [*self._queue_tasks]
        if queue_tasks:
            # A queue task mid-promotion holds an agent request; stopping it
            # here keeps the pipe's teardown ordered behind its own callers.
            with suppress(Exception):
                await cancel_tasks(queue_tasks, label="turn queue")
        if self._events_subscribed:
            self._events_idle.clear()
            finish_event_queue(self._events, _StreamClosed())
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
