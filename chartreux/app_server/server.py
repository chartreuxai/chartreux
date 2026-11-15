from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue, ValidationError

from chartreux import __version__
from chartreux.app_server._dispatch import (
    SESSION_TITLE_UPDATE_ALIAS_METHOD,
    DispatchResult,
    RequestFailure,
    method_not_found,
    unsupported_title_alias,
)
from chartreux.app_server._execution import cancel_tasks
from chartreux.app_server._handler import session_fenced_failure, version_skew_failure
from chartreux.app_server._host import HostRequestHandler
from chartreux.app_server._model import ProtocolModel, validate_wire
from chartreux.app_server._session_backend_port import (
    SessionBackend,
    SessionBackendError,
    SessionBackendEvent,
    SessionBackendHost,
    SessionBackendResult,
    SessionEventSubscription,
)
from chartreux.app_server._session_backend_services import SessionBackendServices
from chartreux.app_server.events import CallbackRequested, MCPAuthorizationRequiredEvent
from chartreux.app_server.models import OpenCallbackState, PublicCallbackEntry
from chartreux.app_server.protocol import (
    SERVER_METHODS,
    AgentConfig,
    AppServerResponseError,
    CallbackCallParams,
    CallbackCallResponse,
    CallbackResultError,
    CallbackResultParams,
    ChildCallbackClosedParams,
    ClientCapabilities,
    ClientInfo,
    ConfigReloadParams,
    ConfigWriteParams,
    ContextInjectParams,
    DangerousCommandAuthorityReadParams,
    DangerousCommandRevokeParams,
    EventBatch,
    EventNotificationParams,
    EventsReadParams,
    InitializeParams,
    InitializeResponse,
    InvalidParamsData,
    InvalidParamsIssue,
    JsonRpcErrorResponse,
    JsonRpcProtocolError,
    JsonRpcSuccessResponse,
    Notification,
    PageRequest,
    PolicyReadParams,
    PolicyReplaceParams,
    ProtocolError,
    ProtocolErrorCode,
    RootsReadParams,
    RootsReplaceParams,
    RootsSaveParams,
    RuntimeSnapshot,
    ServerInfo,
    ServerRequest,
    SessionCompactParams,
    SessionContinueParams,
    SessionContinueResponse,
    SessionDeleteParams,
    SessionForkParams,
    SessionHandoffParams,
    SessionListParams,
    SessionPlanRecordWriteParams,
    SessionReadParams,
    SessionReadResponse,
    SessionResumeParams,
    SessionResumeResponse,
    SessionSettingsUpdateParams,
    SessionSnapshotParams,
    SessionStartParams,
    SessionStartResponse,
    SessionStopParams,
    SessionStopResponse,
    SessionTitleUpdateParams,
    TransportKind,
    TurnEnqueueParams,
    TurnInterruptParams,
    TurnQueueReadParams,
    TurnQueueRemoveParams,
    TurnQueueReplaceParams,
    TurnQueueResumeParams,
    TurnStartParams,
    TurnSteerParams,
    UsageUpdatedParams,
    format_invalid_params_issues,
    redact_validation_message,
    redact_validation_path,
    validate_callback_acknowledgement,
    validate_json_rpc_envelope,
    validate_notification_method,
)
from chartreux.app_server.registry_manifest import (
    CallbackTarget,
    LiveAgentUnavailableError,
    RegistryManifest,
)
from chartreux.app_server.transport import STDIO_TRANSPORT, JsonRpcTransport
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.subagents import LaunchConfigError
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server.mcp_catalog import MCPCatalogService


class InitializationState(StrEnum):
    UNINITIALIZED = auto()
    INITIALIZE_RECEIVED = auto()
    INITIALIZED = auto()


type SessionBackendHostFactory = Callable[[SessionBackendServices], SessionBackendHost]


# Requests that can hand the connection a different session. Their answer is
# what tells the client the session changed, so notifications raised while they
# run are held back and flushed once it has been sent: an event naming the new
# session ahead of the response reaches a client still projecting the old one.
_SESSION_ATTACHMENT_CANDIDATES = frozenset({
    "session/fork",
    "session/start",
    "session/resume",
    "session/continue",
    "session/rewind",
    "session/history/clear",
    "session/plan/handoff",
})

_SESSION_BACKEND_HOST_LIFECYCLE_METHODS = frozenset({
    "session/start",
    "session/resume",
    "session/continue",
})

_SESSION_OPTIONAL_METHODS = frozenset({
    "config/read",
    "workspace/trust/decision",
    "usage/read",
})

_SESSION_BACKEND_METHODS = frozenset({
    "callback/result",
    "config/reload",
    "config/policy/read",
    "config/policy/replace",
    "policy/dangerous_commands/read",
    "policy/dangerous_commands/revoke",
    "policy/roots/read",
    "policy/roots/replace",
    "policy/roots/save",
    "config/write",
    "session/compact",
    "session/context/inject",
    "session/plan/record/write",
    "session/settings/update",
    "app_server/session/turn/enqueue",
    "app_server/session/turn/queue/read",
    "app_server/session/turn/queue/remove",
    "app_server/session/turn/queue/replace",
    "app_server/session/turn/queue/resume",
    "turn/interrupt",
    "turn/start",
    "turn/steer",
})


_STOP_RECEIPT_TIMEOUT = 30.0
"""How long close() waits for pending stop receipts before closing the transport.

Stop waiters are exempt from shutdown cancellation and their receipts are
delivered before the connection closes; this bound only guards a pathological
waiter that never settles, so the shutdown cannot hang forever.
"""


@dataclass(slots=True)
class CallbackDelivery:
    session_id: str
    callback_id: str
    answered: bool = False
    target: CallbackTarget | None = None


@dataclass(slots=True)
class PendingClientRequest:
    response_type: type[ProtocolModel]
    future: asyncio.Future[ProtocolModel]


@dataclass(frozen=True, slots=True)
class PendingNotification:
    method: str
    params: ProtocolModel


def _consume_task_result(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


@dataclass(frozen=True, slots=True)
class _ErrorDispatch:
    error: ProtocolError
    after_response: Callable[[], None] | None = None
    on_response_abandoned: Callable[[], None] | None = None


class AppServer:
    def __init__(
        self,
        transport: JsonRpcTransport,
        *,
        session_backend_host_factory: SessionBackendHostFactory,
        transport_kind: TransportKind = "in_process",
        host_handler: HostRequestHandler | None = None,
        mcp_catalog_service: MCPCatalogService | None = None,
    ) -> None:
        self._root: SessionBackend | None = None
        self._host_handler = host_handler or HostRequestHandler(
            HarnessFilesManager(sources=("user", "project"))
        )
        self._transport: JsonRpcTransport | None = transport
        self._transport_kind: TransportKind = transport_kind
        self._initialization = InitializationState.UNINITIALIZED
        self._client_info: ClientInfo | None = None
        self._client_capabilities = ClientCapabilities()
        self._next_request_id = 1
        self._event_watermarks: dict[str, int] = {}
        # Session rotations this connection adopted (superseded id -> rotated
        # id) and the handoff source ids still in flight, so a request
        # addressed to a superseded id is fenced instead of answered with a
        # bare not-found or silently retried.
        self._session_rotations: dict[str, str] = {}
        self._pending_handoffs: set[str] = set()
        self._callback_requests: dict[int, CallbackDelivery] = {}
        self._client_requests: dict[int, PendingClientRequest] = {}
        self._abandoned_client_request_ids: set[int] = set()
        self._request_tasks: set[asyncio.Task[None]] = set()
        # The request tasks currently serving session/stop. Stop waiters are
        # exempt from shutdown cancellation so every pending stop receipt --
        # coalesced ones included -- is delivered before the connection closes.
        self._stop_waiters: set[asyncio.Task[None]] = set()
        self._stop_lock = asyncio.Lock()
        # The first in-flight stop's pending receipt: a second stop arriving
        # during the first stop's persistence window coalesces into it.
        self._stop_receipt: asyncio.Future[DispatchResult] | None = None
        self._backend_event_task: asyncio.Task[None] | None = None
        self._backend_event_backend: SessionBackend | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._connection_attached = False
        self._attaching = False
        self._pending_notifications: list[PendingNotification] = []
        self._request_error: BaseException | None = None
        self._close_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._lifecycle_lock_owner: asyncio.Task[Any] | None = None
        self._connection_lock = asyncio.Lock()
        self._attachment_lock = asyncio.Lock()
        self._closed = False
        self._shutdown_complete = False
        self._mcp_catalog_service = mcp_catalog_service
        self._session_backend_host = session_backend_host_factory(self)
        if self._session_backend_host is None:
            raise TypeError(
                "session_backend_host_factory must return a SessionBackendHost"
            )

    def _require_root(self) -> SessionBackend:
        if self._root is None:
            raise RuntimeError("No app-server session runtime is active")
        return self._root

    async def _activate_backend(
        self, backend: SessionBackend, history_limit: int
    ) -> SessionReadResponse:
        params = SessionReadParams(
            session_id=backend.session_id, history=PageRequest(limit=history_limit)
        )
        if backend is self._backend_event_backend:
            return await backend.read(params)
        subscription = await backend.subscribe(params)
        previous = self._root
        self._root = backend
        await self._replace_backend_event_task(backend, subscription)
        if previous is not None and previous is not backend:
            try:
                await previous.shutdown()
            except Exception as exc:
                logger.warning(
                    "Failed to shut down previous session backend", exc_info=exc
                )
        await self._replay_pending_backend_events(
            backend, subscription.snapshot.last_event_id
        )
        await self._flush_backend_events(backend)
        return subscription.snapshot

    async def _replace_backend_event_task(
        self, backend: SessionBackend, subscription: SessionEventSubscription
    ) -> None:
        previous = self._backend_event_task
        if previous is not None:
            previous.cancel()
            with suppress(asyncio.CancelledError):
                await previous
        task = asyncio.create_task(
            self._forward_backend_events(subscription), name="vibe-backend-events"
        )
        self._backend_event_task = task
        self._backend_event_backend = backend
        task.add_done_callback(self._backend_event_finished)

    async def _forward_backend_events(
        self, subscription: SessionEventSubscription
    ) -> None:
        async for envelope in subscription.events:
            await self._forward_backend_event(envelope)

    async def _forward_backend_event(self, envelope: SessionBackendEvent) -> None:
        if isinstance(envelope.event, CallbackRequested):
            await self._deliver_callback(envelope.event.callback)
        elif isinstance(envelope.event, MCPAuthorizationRequiredEvent):
            await self._forward_mcp_authorization(envelope.event)
        else:
            if envelope.method is None or envelope.params is None:
                raise RuntimeError(
                    "Session backend event has no Chartreux notification mapping"
                )
            await self._route_notification(envelope.method, envelope.params)

    async def _forward_mcp_authorization(
        self, event: MCPAuthorizationRequiredEvent
    ) -> None:
        service = self._mcp_catalog_service
        if service is None:
            return
        root = self._root
        if root is not None:
            from chartreux.app_server.mcp_catalog import SessionMCPCatalogBinding

            if not isinstance(root, SessionMCPCatalogBinding):
                # The process-backed (stdio) root is not a catalog binding:
                # the gate ran owner-side in the agent process, so this end
                # relays the gated notification and the adopted runtime view.
                await self._route_notification("mcp_catalog/authRequired", event.params)
                await self._route_notification(
                    "runtime/updated", root.runtime_updated_params()
                )
                return
        runtime_updated = await service.accept_auth_required(event.params, root)
        if runtime_updated is None:
            return
        await self._route_notification("mcp_catalog/authRequired", event.params)
        await self._route_notification("runtime/updated", runtime_updated)

    def _backend_event_finished(self, task: asyncio.Task[None]) -> None:
        if self._backend_event_task is task:
            self._backend_event_task = None
            self._backend_event_backend = None
        if task.cancelled():
            return
        error = task.exception()
        if error is None or self._closed:
            return
        logger.error(
            "App-server backend event task failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        self._request_error = error
        if self._serve_task is not None:
            self._serve_task.cancel()

    async def _replay_pending_backend_events(
        self, backend: SessionBackend, snapshot_event_id: int
    ) -> None:
        pending = self._pending_notifications
        self._pending_notifications = []
        for notification in pending:
            params = notification.params
            if not isinstance(params, EventNotificationParams):
                self._pending_notifications.append(notification)
                continue
            if params.event_id <= snapshot_event_id:
                continue
            if await backend.publish_notification(notification.method, params):
                await self._flush_backend_events(backend)
                continue
            self._pending_notifications.append(notification)

    @staticmethod
    async def _flush_backend_events(backend: SessionBackend) -> None:
        await backend.flush_events()

    async def serve(self) -> None:
        transport = self._transport
        if transport is None:
            raise RuntimeError("App server has no transport to serve")
        await self.serve_connection(transport, close_on_disconnect=True)

    async def serve_connection(
        self, transport: JsonRpcTransport, *, close_on_disconnect: bool
    ) -> None:
        async with self._connection_lock:
            if self._closed:
                raise RuntimeError("App server is closed")
            self._transport = transport
            self._initialization = InitializationState.UNINITIALIZED
            self._client_info = None
            self._client_capabilities = ClientCapabilities()
            self._connection_attached = False
            self._attaching = False
            self._pending_notifications.clear()
            self._request_error = None
            self._serve_task = asyncio.current_task()

            def usage_updated(params: UsageUpdatedParams) -> None:
                if (
                    self._initialization != InitializationState.INITIALIZED
                    or self._closed
                ):
                    return
                task = asyncio.create_task(self._notify_host(params))
                self._request_tasks.add(task)
                task.add_done_callback(self._request_finished)

            unsubscribe_usage = self._host_handler.subscribe_usage(usage_updated)
            try:
                async for raw_message in transport.messages():
                    message = validate_json_rpc_envelope(raw_message)
                    if isinstance(message, ServerRequest):
                        if message.method == "initialize":
                            await self._handle_request(message)
                            continue
                        task = asyncio.create_task(self._handle_request(message))
                        self._request_tasks.add(task)
                        task.add_done_callback(self._request_finished)
                        continue
                    if isinstance(message, Notification):
                        self._handle_notification(message)
                        continue
                    task = asyncio.create_task(self._handle_response(message))
                    self._request_tasks.add(task)
                    task.add_done_callback(self._request_finished)
            except asyncio.CancelledError:
                if self._request_error is not None:
                    raise self._request_error
                raise
            finally:
                unsubscribe_usage()
                if close_on_disconnect:
                    if self._root is not None:
                        # Terminal-disconnect disclosure: a stdio connection
                        # drop stays terminal — in-flight work on the attached
                        # root can be lost, and the explicit session/resume is
                        # the recovery path; there is no transparent reconnect.
                        logger.warning(
                            "The stdio connection disconnected with session %s "
                            "attached: the disconnect is terminal and in-flight "
                            "work can be lost; resume explicitly with "
                            "session/resume",
                            self._root.session_id,
                        )
                    await self.close()
                else:
                    await self._detach_connection(transport)

    async def close(self) -> None:
        if self._shutdown_complete:
            return
        async with self._close_lock:
            if self._shutdown_complete:
                return
            self._closed = True
            current = asyncio.current_task()
            serve_task = self._serve_task
            if serve_task is not None and serve_task is not current:
                serve_task.cancel()
            errors = await self._stop_background_tasks(current)
            try:
                await self._close_root()
            except BaseException as exc:
                errors.append(exc)
            self._callback_requests.clear()
            self._pending_notifications.clear()
            self._connection_attached = False
            self._attaching = False
            self._reject_pending_client_requests(RuntimeError("App server closed"))
            # Mark the shutdown complete before waiting on the stop waiters,
            # so a waiter's own close() returns without blocking on this lock.
            self._shutdown_complete = True
            # Stop waiters are exempt from the cancellation above; every
            # pending stop receipt -- coalesced ones included -- is delivered
            # before the connection closes.
            waiters = [
                task
                for task in self._stop_waiters
                if task is not current and not task.done()
            ]
            if waiters:
                done, pending = await asyncio.wait(
                    waiters, timeout=_STOP_RECEIPT_TIMEOUT
                )
                if pending:
                    logger.error(
                        "Stop waiters did not deliver their receipts before close: %s",
                        [task.get_name() for task in pending],
                    )
            transport = self._transport
            if transport is not None:
                try:
                    await transport.close()
                except BaseException as exc:
                    errors.append(exc)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup("App server shutdown failed", errors)

    async def _close_root(self) -> None:
        async with self._lifecycle_transition():
            try:
                await self._session_backend_host.shutdown()
            finally:
                self._root = None

    async def _detach_connection(self, transport: JsonRpcTransport) -> None:
        if self._transport is not transport:
            return
        if self._closed:
            return
        errors = await self._stop_request_tasks(asyncio.current_task())
        self._reject_pending_client_requests(RuntimeError("Client disconnected"))
        self._callback_requests.clear()
        self._pending_notifications.clear()
        self._connection_attached = False
        self._attaching = False
        self._transport = None
        self._serve_task = None
        self._initialization = InitializationState.UNINITIALIZED
        self._client_info = None
        self._client_capabilities = ClientCapabilities()
        try:
            await transport.close()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("App-server connection cleanup failed", errors)

    async def _stop_background_tasks(
        self, current: asyncio.Task[object] | None
    ) -> list[BaseException]:
        tasks = [
            task
            for task in self._request_tasks
            if task is not current and task not in self._stop_waiters
        ]
        backend_events = self._backend_event_task
        self._backend_event_task = None
        self._backend_event_backend = None
        if backend_events is not None and backend_events is not current:
            tasks.append(backend_events)
        errors = await cancel_tasks(tasks, label="app-server background")
        errors.extend(await self._session_backend_host.stop_background_tasks(current))
        return errors

    async def _stop_request_tasks(
        self, current: asyncio.Task[object] | None
    ) -> list[BaseException]:
        tasks = [
            task
            for task in self._request_tasks
            if task is not current and task not in self._stop_waiters
        ]
        backend_events = self._backend_event_task
        self._backend_event_task = None
        self._backend_event_backend = None
        if backend_events is not None and backend_events is not current:
            tasks.append(backend_events)
        return await cancel_tasks(tasks, label="app-server request")

    def _reject_pending_client_requests(self, error: Exception) -> None:
        for pending in self._client_requests.values():
            if not pending.future.done():
                pending.future.set_exception(error)
        self._client_requests.clear()
        self._abandoned_client_request_ids.clear()

    def _request_finished(self, task: asyncio.Task[None]) -> None:
        self._request_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is None or self._closed:
            return
        logger.error(
            "App-server request task failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        self._request_error = error
        if self._serve_task is not None:
            self._serve_task.cancel()

    async def _handle_request(self, request: ServerRequest) -> None:
        current = asyncio.current_task()
        if request.method == "session/stop" and current is not None:
            # A stop waiter is exempt from shutdown cancellation, so its
            # receipt -- the first stop's or a coalesced one -- is delivered
            # before the connection closes.
            self._stop_waiters.add(current)
            try:
                await self._handle_request_message(request)
            finally:
                self._stop_waiters.discard(current)
            return
        await self._handle_request_message(request)

    async def _handle_request_message(self, request: ServerRequest) -> None:
        if request.method in _SESSION_ATTACHMENT_CANDIDATES:
            async with self._attachment_lock:
                await self._handle_request_once(request)
            return
        await self._handle_request_once(request)

    async def _handle_request_once(self, request: ServerRequest) -> None:
        was_attached = self._connection_attached
        attachment_candidate = request.method in _SESSION_ATTACHMENT_CANDIDATES
        if attachment_candidate:
            self._begin_attachment()
        outcome = await self._dispatch_or_error(request)
        if isinstance(outcome, _ErrorDispatch):
            try:
                await self._send_error(request.id, outcome.error)
            except BaseException:
                if outcome.on_response_abandoned is not None:
                    outcome.on_response_abandoned()
                if attachment_candidate:
                    await self._finish_attachment(was_attached)
                raise
            if attachment_candidate:
                await self._finish_attachment(was_attached)
            if outcome.after_response is not None:
                outcome.after_response()
            if request.method == "session/stop":
                await self.close()
            return
        try:
            await self._send({
                "jsonrpc": "2.0",
                "id": request.id,
                "result": outcome.response.model_dump(mode="json", by_alias=True),
            })
        except BaseException:
            if outcome.on_response_abandoned is not None:
                outcome.on_response_abandoned()
            if attachment_candidate:
                await self._finish_attachment(was_attached)
            raise
        if attachment_candidate:
            await self._finish_attachment(outcome.session_attached or was_attached)
        await self._after_response(request, outcome)

    def _begin_attachment(self) -> None:
        self._attaching = True
        self._connection_attached = False
        self._pending_notifications.clear()

    async def _finish_attachment(self, attached: bool) -> None:
        if not attached:
            self._pending_notifications.clear()
            self._attaching = False
            self._connection_attached = False
            return
        while self._pending_notifications:
            pending = self._pending_notifications
            self._pending_notifications = []
            for notification in pending:
                await self._send_notification(notification.method, notification.params)
        self._attaching = False
        self._connection_attached = True

    async def _dispatch_or_error(
        self, request: ServerRequest
    ) -> DispatchResult | _ErrorDispatch:
        try:
            if request.method == "initialize":
                return DispatchResult(self._initialize(request.params))
            if self._initialization is not InitializationState.INITIALIZED:
                raise RequestFailure(
                    ProtocolErrorCode.NOT_INITIALIZED,
                    "initialize must be the first request",
                )
            dispatched = await self._dispatch_request(request.method, request.params)
            try:
                if dispatched.session_attached:
                    self._validate_open_callback_capabilities()
            except BaseException:
                if dispatched.on_response_abandoned is not None:
                    dispatched.on_response_abandoned()
                raise
            return dispatched
        except (RequestFailure, SessionBackendError) as exc:
            return _ErrorDispatch(
                ProtocolError(code=exc.code, message=str(exc), data=exc.data),
                after_response=(
                    exc.after_response if isinstance(exc, SessionBackendError) else None
                ),
                on_response_abandoned=(
                    exc.on_response_abandoned
                    if isinstance(exc, SessionBackendError)
                    else None
                ),
            )
        except LaunchConfigError as exc:
            return _ErrorDispatch(
                ProtocolError(
                    code=ProtocolErrorCode.INVALID_PARAMS,
                    message="Invalid launch configuration",
                    data={"code": "launch_config_error", "fieldPath": exc.field_path},
                )
            )
        except ValidationError as exc:
            data = InvalidParamsData(
                error_count=exc.error_count(),
                issues=[
                    InvalidParamsIssue(
                        path=redact_validation_path(issue["loc"]),
                        message=redact_validation_message(issue["type"]),
                    )
                    for issue in exc.errors()
                ],
            ).model_dump(mode="json", by_alias=True)
            logger.warning(
                "Rejected %s: invalid request parameters: %s",
                request.method,
                format_invalid_params_issues(data) or "<no field detail>",
            )
            return _ErrorDispatch(
                ProtocolError(
                    code=ProtocolErrorCode.INVALID_PARAMS,
                    message="Invalid request parameters",
                    data=data,
                )
            )
        except Exception as exc:
            logger.exception("Unhandled error dispatching %s", request.method)
            return _ErrorDispatch(
                ProtocolError(code=ProtocolErrorCode.INTERNAL_ERROR, message=str(exc))
            )

    async def _after_response(
        self, request: ServerRequest, dispatched: DispatchResult
    ) -> None:
        if request.method == "callback/result":
            result = request.params.get("result")
            callback_id = result.get("callbackId") if isinstance(result, dict) else None
            if isinstance(callback_id, str):
                self._mark_callback_answered(callback_id)
        if dispatched.after_response is not None:
            dispatched.after_response()
        if dispatched.session_attached:
            await self._redeliver_open_callbacks()
        if dispatched.runtime_updated and self._root is not None:
            await self._notify("runtime/updated", self._root.runtime_updated_params())
        if request.method == "session/stop":
            await self.close()

    async def _dispatch_request(  # noqa: PLR0911 - explicit route ownership
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method not in SERVER_METHODS:
            raise method_not_found(method)
        if method == SESSION_TITLE_UPDATE_ALIAS_METHOD:
            # A retained alias with no handler in either topology: the wire
            # method for a manual title is session/rename. Refused before any
            # session routing, so both transports answer alike instead of one
            # reading as an unknown method and the other as an uncarried stdio
            # operation naming a selector that does not carry it either.
            raise unsupported_title_alias()
        if method == "usage/read":
            return await self._host_handler.read_usage(
                raw_params,
                self._root.session_id
                if self._root is not None and self._connection_attached
                else None,
            )
        if method == "events/read":
            return self._events_read(raw_params)
        if self._mcp_catalog_service is not None and self._mcp_catalog_service.handles(
            method
        ):
            agent_owned = (
                self._transport_kind == STDIO_TRANSPORT
                and self._root is not None
                and isinstance(raw_params.get("sessionId"), str)
            )
            if not agent_owned:
                return await self._mcp_catalog_service.dispatch(
                    method, raw_params, root=self._root, notify=self._notify
                )
            # A session-addressed catalog call on the stdio topology is owned
            # by the agent process: it falls through to the root's extension
            # dispatch, which carries it over the wire with the typed owner
            # key. Sessionless calls keep the coordinator's host-side service.
        if method in {
            "config/schema",
            "session/history/get",
            "workspace/git/branch",
            "workspace/git/checkouts",
            "workspace/git/worktrees/list",
            "workspace/git/worktrees/remove",
            "workspace/trust/status",
            "workspace/trust/untrustedConfig",
        }:
            return await self._host_handler.dispatch(method, raw_params)
        if method == "session/delete":
            return await self._delete_session(raw_params)
        if method == "session/stop":
            return await self._dispatch_stop(raw_params)
        if _omits_session_id(method, raw_params):
            return await self._host_handler.dispatch(method, raw_params)
        if self._root is None:
            return await self._dispatch_without_root(method, raw_params)
        try:
            result = await self._dispatch_to_root(method, raw_params)
        except (RequestFailure, SessionBackendError) as exc:
            fenced = await self._superseded_session_fence(exc, raw_params)
            if fenced is not None:
                raise fenced from exc
            raise
        return result

    async def _superseded_session_fence(
        self, exc: RequestFailure | SessionBackendError, raw_params: dict[str, Any]
    ) -> RequestFailure | None:
        """Translate a bare not-found for a superseded id into the typed fence.

        A request addressed to a session id a plan handoff rotated must never
        answer bare NOT_FOUND or be silently retried: a settled rotation maps
        the old id to the rotated one, and one still in flight between
        handoff-send and settlement is resolved through the backend. A peer
        that cannot report the rotation is a version skew, not a not-found.
        """
        if exc.code is not ProtocolErrorCode.NOT_FOUND:
            return None
        session_id = raw_params.get("sessionId")
        if not isinstance(session_id, str):
            return None
        rotated = self._session_rotations.get(session_id)
        if rotated is None and session_id in self._pending_handoffs:
            resolver = getattr(self._root, "rotated_session_id_for", None)
            if resolver is not None:
                rotated = await resolver(session_id)
        if rotated is not None and rotated != session_id:
            return session_fenced_failure(session_id, rotated)
        if session_id in self._pending_handoffs:
            return version_skew_failure(
                session_id, "the in-flight handoff's rotated identity is unavailable"
            )
        return None

    def _events_read(self, raw_params: dict[str, Any]) -> DispatchResult:
        """Reserve pull-based event replay for future clients.

        Chartreux receives live updates through notifications. Without a replay
        store, this procedure intentionally returns an empty batch.
        """
        params = validate_wire(EventsReadParams, raw_params)
        unsupported = params.filters.unsupported_v01_fields()
        if unsupported:
            fields = ", ".join(unsupported)
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                f"events/read filtering is not supported in v0.1: {fields}",
            )
        return DispatchResult(response=EventBatch())

    async def _close_attached_runtime(self) -> None:
        self._closed = True
        errors = await self._stop_background_tasks(asyncio.current_task())
        try:
            await self._close_root()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("App-server session shutdown failed", errors)

    async def _dispatch_stop(self, raw_params: dict[str, Any]) -> DispatchResult:
        """Route one ``session/stop`` on either transport.

        With a root attached the stop runs the coordinator lifecycle once
        (coalesced); without one -- a fresh connection or a saved session --
        the stop against the settled session is an ack.
        """
        params = validate_wire(SessionStopParams, raw_params)
        if self._root is None:
            return await self._ack_settled_stop(params)
        return await self._coalesced_stop(raw_params)

    async def _ack_settled_stop(self, params: SessionStopParams) -> DispatchResult:
        """Ack a stop against a settled session via a fresh connection.

        Repeat-safe applies outside the closed connection: a stop for a saved
        session answers the ack instead of a conflict, and an unknown id stays
        NOT_FOUND, exactly as the in-process arm's ``_require_session``.
        """
        probe = SessionReadParams(
            session_id=params.session_id, history=PageRequest(limit=1)
        )
        try:
            await self._host_handler.dispatch(
                "session/read", probe.model_dump(mode="json", by_alias=True)
            )
        except RequestFailure as exc:
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {params.session_id}"
            ) from exc
        return DispatchResult(SessionStopResponse())

    async def _coalesced_stop(self, raw_params: dict[str, Any]) -> DispatchResult:
        """Run the stop lifecycle once; later stops coalesce into its receipt.

        A second stop arriving during the first stop's persistence window
        answers the first stop's pending receipt: no concurrent second
        persistence runs.
        """
        async with self._stop_lock:
            pending = self._stop_receipt
            if pending is not None:
                return await pending
            receipt: asyncio.Future[DispatchResult] = (
                asyncio.get_running_loop().create_future()
            )
            self._stop_receipt = receipt
        try:
            result = await self._run_stop(raw_params)
        except BaseException as exc:
            if not receipt.done():
                receipt.set_exception(exc)
                # The direct caller already raises this; mark it retrieved so
                # a coalesced waiter is not the only consumer of it.
                receipt.exception()
            raise
        if not receipt.done():
            receipt.set_result(result)
        return result

    async def _run_stop(self, raw_params: dict[str, Any]) -> DispatchResult:
        """One stop: the backend's lifecycle, then the attached runtime closes.

        The root's termination and persistence are ordered ahead of the ack
        frame: dispatch returns only after ``_close_attached_runtime`` ran, and
        the response is sent afterwards.
        """
        try:
            result = await self._dispatch_to_root("session/stop", raw_params)
        except (RequestFailure, SessionBackendError) as exc:
            fenced = await self._superseded_session_fence(exc, raw_params)
            if fenced is not None:
                raise fenced from exc
            raise
        await self._close_attached_runtime()
        return result

    async def _delete_session(self, raw_params: dict[str, Any]) -> DispatchResult:
        params = validate_wire(SessionDeleteParams, raw_params)
        root = self._root
        is_root = root is not None and params.session_id == root.session_id
        references_child = root is not None and root.references_child(params.session_id)
        if is_root or references_child:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT, "Deleting a live session is not supported"
            )
        return await self._host_handler.dispatch("session/delete", raw_params)

    async def _dispatch_without_root(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method in {"session/start", "session/resume", "session/continue"}:
            return await self._open_initial_session(method, raw_params)
        if result := await self._dispatch_backend_host_operation(method, raw_params):
            return result
        if self._host_handler.handles(method):
            return await self._host_handler.dispatch(method, raw_params)
        return await self._open_initial_session(method, raw_params)

    async def _dispatch_to_root(  # noqa: PLR0912 - the rotation fence adds a branch
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        root = self._root
        if root is None:
            raise RuntimeError("Root session closed while dispatching a request")
        host_result = await self._dispatch_backend_host_operation(method, raw_params)
        if host_result is not None:
            return host_result
        # These run-pinned stops remain available during root lifecycle work;
        # their routes retain attachment and shutdown checks. Steering delegates
        # its guard to the backend for receipt replay and pre-commit rejection.
        if method not in {"turn/interrupt", "agents/cancel", "turn/steer"}:
            root.guard_request()
        backend_result = await self._dispatch_backend_operation(
            root, method, raw_params
        )
        if backend_result is not None:
            try:
                await self._flush_backend_events(root)
            except BaseException:
                if backend_result.on_response_abandoned is not None:
                    backend_result.on_response_abandoned()
                raise
            return backend_result
        if method in _SESSION_BACKEND_METHODS:
            raise RuntimeError(f"Session backend method was not routed: {method}")
        previous_session_id = root.session_id
        handoff_source = (
            raw_params.get("sessionId") if method == "session/plan/handoff" else None
        )
        if isinstance(handoff_source, str):
            # Between handoff-send and settlement the source id is pending:
            # a submit addressed to it is fenced, never forwarded against the
            # pre-rotation session.
            self._pending_handoffs.add(handoff_source)
        try:
            try:
                result = await root.dispatch_extension(method, raw_params)
            except SessionBackendError as exc:
                if exc.code is ProtocolErrorCode.NOT_FOUND and method in {
                    "session/read",
                    "session/history/list",
                    "session/turns/list",
                }:
                    if (
                        method == "session/history/list"
                        and raw_params.get("sessionId") == root.session_id
                    ):
                        # The attached session's live history is agent-owned:
                        # a not-found from its carrier is a typed mismatch,
                        # never the persisted host projection served for an
                        # attached id.
                        raise SessionBackendError(
                            ProtocolErrorCode.CONFLICT,
                            f"Session {root.session_id} is attached; its live "
                            "history is served by the root agent, not the "
                            "persisted host projection",
                        ) from exc
                    return await self._host_handler.dispatch(method, raw_params)
                raise
            try:
                if result.session_attached and root.session_id != previous_session_id:
                    # The dispatch rotated the session: remember old -> new so a
                    # late request addressed to the old id is fenced, then
                    # resubscribe the event stream to the rotated session.
                    self._session_rotations[previous_session_id] = root.session_id
                    # Release the old event consumer before subscribing: the
                    # backend admits exactly one subscriber, and the rotation
                    # replaces it.
                    previous_task = self._backend_event_task
                    if previous_task is not None:
                        previous_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await previous_task
                    subscription = await root.subscribe(
                        SessionReadParams(
                            session_id=root.session_id, history=PageRequest(limit=200)
                        )
                    )
                    await self._replace_backend_event_task(root, subscription)
                    await self._replay_pending_backend_events(
                        root, subscription.snapshot.last_event_id
                    )
                await self._flush_backend_events(root)
            except BaseException:
                if result.on_response_abandoned is not None:
                    result.on_response_abandoned()
                raise
            return result
        finally:
            if isinstance(handoff_source, str):
                self._pending_handoffs.discard(handoff_source)

    async def _dispatch_backend_host_operation(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult | None:
        if method == "session/list":
            response = await self._session_backend_host.list(
                validate_wire(SessionListParams, raw_params)
            )
            return DispatchResult(response)
        if method == "session/read":
            response = await self._session_backend_host.read(
                validate_wire(SessionReadParams, raw_params)
            )
            return DispatchResult(response)
        if method == "session/rename":
            params = validate_wire(SessionTitleUpdateParams, raw_params)
            response = await self._session_backend_host.rename(params)
            if self._root is not None and self._root.session_id == params.session_id:
                await self._flush_backend_events(self._root)
            return DispatchResult(response)
        if method == "session/fork":
            params = validate_wire(SessionForkParams, raw_params)
            if params.after_turn_id is not None:
                raise RequestFailure(
                    ProtocolErrorCode.INVALID_PARAMS,
                    "session/fork afterTurnId is not supported in v0.1",
                )
            self._validate_agent_config(params.agent_config)
            result = await self._session_backend_host.fork(params)
            response = result.response
            if result.backend is not None:
                snapshot = await self._activate_backend(
                    result.backend, params.history_limit
                )
                response = response.model_copy(
                    update={
                        "state": snapshot.state,
                        "last_event_id": snapshot.last_event_id,
                    }
                )
            return DispatchResult(
                response,
                after_response=result.after_response,
                session_attached=result.backend is not None,
            )
        return await self._dispatch_backend_host_lifecycle(method, raw_params)

    @staticmethod
    def _validate_agent_config(config: AgentConfig | None) -> None:
        if config is None:
            return
        unsupported = config.unsupported_v01_fields()
        if unsupported:
            fields = ", ".join(unsupported)
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                f"AgentConfig fields are not supported in v0.1: {fields}",
            )

    async def _dispatch_backend_host_lifecycle(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult | None:
        if method not in _SESSION_BACKEND_HOST_LIFECYCLE_METHODS:
            return None
        async with self._lifecycle_transition():
            return await self._dispatch_backend_host_lifecycle_locked(
                method, raw_params
            )

    async def _dispatch_backend_host_lifecycle_locked(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method == "session/start":
            params = validate_wire(SessionStartParams, raw_params)
            self._validate_agent_config(params.agent_config)
            lifecycle = await self._session_backend_host.start(params)
            self._record_session_created(lifecycle.backend)
            response = await self._activate_backend(
                lifecycle.backend, params.history_limit
            )
            return DispatchResult(
                SessionStartResponse(
                    state=response.state, last_event_id=response.last_event_id
                ),
                after_response=lifecycle.after_response,
                session_attached=True,
            )
        if method == "session/resume":
            params = validate_wire(SessionResumeParams, raw_params)
            self._validate_agent_config(params.agent_config)
            lifecycle = await self._session_backend_host.resume(params)
            response = await self._activate_backend(
                lifecycle.backend, params.history_limit
            )
            return DispatchResult(
                SessionResumeResponse(
                    state=response.state,
                    last_event_id=response.last_event_id,
                    plan_record=await lifecycle.backend.read_plan_record(),
                    # A resume that respawned a confirmed-dead root surfaces the
                    # retired generation's memorial queue/shell state and its
                    # typed death receipt; the memorial is never re-admitted
                    # for execution on the new generation.
                    memorial=list(lifecycle.memorial),
                    death_receipt=lifecycle.death_receipt,
                ),
                after_response=lifecycle.after_response,
                session_attached=True,
                runtime_updated=True,
            )
        if method == "session/continue":
            params = validate_wire(SessionContinueParams, raw_params)
            self._validate_agent_config(params.agent_config)
            lifecycle = await self._session_backend_host.continue_latest(params)
            response = await self._activate_backend(
                lifecycle.backend, params.history_limit
            )
            return DispatchResult(
                SessionContinueResponse(
                    state=response.state, last_event_id=response.last_event_id
                ),
                after_response=lifecycle.after_response,
                session_attached=True,
                runtime_updated=True,
            )
        raise RuntimeError(f"Unsupported session lifecycle method: {method}")

    def _record_session_created(self, backend: SessionBackend) -> None:
        logger.debug("Session created: session_id=%s", backend.session_id)

    async def _dispatch_backend_operation(
        self, root: SessionBackend, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult | None:
        if method.startswith("app_server/session/turn/"):
            return await self._dispatch_backend_turn(root, method, raw_params)
        match method.partition("/")[0]:
            case "session":
                return await self._dispatch_backend_session(root, method, raw_params)
            case "turn":
                return await self._dispatch_backend_turn(root, method, raw_params)
            case "config" | "policy":
                return await self._dispatch_backend_config(root, method, raw_params)
            case "callback":
                return await self._dispatch_backend_callback(root, method, raw_params)
        return None

    @staticmethod
    async def _dispatch_backend_session(
        root: SessionBackend, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult | None:
        if method == "session/settings/update":
            result = await root.update_settings(
                validate_wire(SessionSettingsUpdateParams, raw_params)
            )
            return DispatchResult(result.response, after_response=result.after_response)
        if method == "session/context/inject":
            result = await root.inject_context(
                validate_wire(ContextInjectParams, raw_params)
            )
            return DispatchResult(result.response, after_response=result.after_response)
        if method == "session/compact":
            result = await root.compact(validate_wire(SessionCompactParams, raw_params))
            return DispatchResult(
                result.response,
                after_response=result.after_response,
                on_response_abandoned=result.on_response_abandoned,
            )
        if method == "session/plan/record/write":
            result = await root.write_plan_record(
                validate_wire(SessionPlanRecordWriteParams, raw_params)
            )
            return DispatchResult(result.response, after_response=result.after_response)
        return None

    @staticmethod
    async def _dispatch_backend_turn(
        root: SessionBackend, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult | None:
        if method == "app_server/session/turn/enqueue":
            result = await root.enqueue_turn(
                validate_wire(TurnEnqueueParams, raw_params)
            )
        elif method == "app_server/session/turn/queue/read":
            result = await root.read_turn_queue(
                validate_wire(TurnQueueReadParams, raw_params)
            )
        elif method == "app_server/session/turn/queue/remove":
            result = await root.remove_queued_turn(
                validate_wire(TurnQueueRemoveParams, raw_params)
            )
        elif method == "app_server/session/turn/queue/replace":
            result = await root.replace_queued_turn(
                validate_wire(TurnQueueReplaceParams, raw_params)
            )
        elif method == "app_server/session/turn/queue/resume":
            result = await root.resume_turn_queue(
                validate_wire(TurnQueueResumeParams, raw_params)
            )
        elif method == "turn/start":
            result = await root.start_turn(validate_wire(TurnStartParams, raw_params))
        elif method == "turn/steer":
            result = await root.steer_turn(validate_wire(TurnSteerParams, raw_params))
        elif method == "turn/interrupt":
            result = await root.interrupt_turn(
                validate_wire(TurnInterruptParams, raw_params)
            )
        else:
            return None
        return DispatchResult(
            result.response,
            after_response=result.after_response,
            on_response_abandoned=result.on_response_abandoned,
            runtime_updated=result.runtime_updated,
        )

    async def _dispatch_backend_config(  # noqa: PLR0911, PLR0912, PLR0915 - explicit route ownership
        self, root: SessionBackend, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult | None:
        if method in {"policy/roots/read", "policy/roots/replace"}:
            if method == "policy/roots/read":
                return DispatchResult(
                    await root.read_roots(validate_wire(RootsReadParams, raw_params))
                )
            try:
                roots_params = validate_wire(RootsReplaceParams, raw_params)
            except ValidationError:
                raise SessionBackendError(
                    ProtocolErrorCode.INVALID_PARAMS,
                    "Invalid root replacement parameters",
                ) from None
            roots_result = await root.replace_roots(roots_params)
            return DispatchResult(roots_result.response, runtime_updated=True)
        if method == "policy/roots/save":
            try:
                save_params = validate_wire(RootsSaveParams, raw_params)
            except ValidationError:
                raise SessionBackendError(
                    ProtocolErrorCode.INVALID_PARAMS, "Invalid root save parameters"
                ) from None
            return DispatchResult(await root.save_roots(save_params))
        if method == "policy/dangerous_commands/read":
            return DispatchResult(
                await root.read_dangerous_command_authority(
                    validate_wire(DangerousCommandAuthorityReadParams, raw_params)
                )
            )
        if method == "policy/dangerous_commands/revoke":
            try:
                revoke_params = validate_wire(DangerousCommandRevokeParams, raw_params)
            except ValidationError:
                raise SessionBackendError(
                    ProtocolErrorCode.INVALID_PARAMS,
                    "Invalid dangerous-command revocation parameters",
                ) from None
            return DispatchResult(
                await root.revoke_dangerous_command_grant(revoke_params)
            )
        if method in {"config/policy/read", "config/policy/replace"}:
            if method == "config/policy/read":
                return DispatchResult(
                    await root.read_policy(validate_wire(PolicyReadParams, raw_params))
                )
            policy_result = await root.replace_policy(
                validate_wire(PolicyReplaceParams, raw_params)
            )
            return DispatchResult(policy_result.response, runtime_updated=True)
        if method == "config/write":
            result = await root.write_config(
                validate_wire(ConfigWriteParams, raw_params)
            )
            runtime_updated = (
                not result.response.rejected and not result.response.failures
            )
        elif method == "config/reload":
            params = validate_wire(ConfigReloadParams, raw_params)
            if params.session_id != root.session_id:
                raise RequestFailure(
                    ProtocolErrorCode.NOT_FOUND,
                    f"Session not found: {params.session_id}",
                )
            prepare_port = getattr(root, "prepare_mcp_reload", None)
            finish_port = getattr(root, "finish_mcp_reload", None)
            fail_port = getattr(root, "fail_mcp_reload", None)
            if (
                prepare_port is not None
                and finish_port is not None
                and fail_port is not None
            ):
                # The process-backed (stdio) root carries the MCP convergence
                # legs to the agent process that owns the loop; this end only
                # relays them and adopts the receipt's runtime view.
                await prepare_port()
                try:
                    result = await root.reload_config(params)
                    receipt = await finish_port()
                except Exception:
                    await fail_port()
                    raise
                mcp_runtime = (
                    RuntimeSnapshot.model_validate(receipt.runtime)
                    if receipt.runtime is not None
                    else None
                )
            else:
                plan = (
                    await self._mcp_catalog_service.prepare_config_reload(root)
                    if self._mcp_catalog_service is not None
                    else None
                )
                try:
                    result = await root.reload_config(params)
                    mcp_runtime = (
                        await self._mcp_catalog_service.finish_config_reload(root, plan)
                        if self._mcp_catalog_service is not None
                        else None
                    )
                except Exception:
                    if self._mcp_catalog_service is not None:
                        await self._mcp_catalog_service.fail_config_reload(root, plan)
                    raise
            if mcp_runtime is not None:
                result = SessionBackendResult(
                    response=result.response.model_copy(
                        update={"runtime": mcp_runtime}
                    ),
                    after_response=result.after_response,
                )
            runtime_updated = True
        else:
            return None
        return DispatchResult(
            result.response,
            after_response=result.after_response,
            runtime_updated=runtime_updated,
        )

    @staticmethod
    async def _dispatch_backend_callback(
        root: SessionBackend, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult | None:
        if method != "callback/result":
            return None
        result = await root.respond_to_callback(
            validate_wire(CallbackResultParams, raw_params)
        )
        return DispatchResult(result.response, after_response=result.after_response)

    async def _open_initial_session(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        async with self._lifecycle_transition():
            root = self._root
            if root is None:
                return await self._open_initial_session_locked(method, raw_params)
        return await self._dispatch_to_root(method, raw_params)

    @asynccontextmanager
    async def _lifecycle_transition(self) -> AsyncIterator[None]:
        task = asyncio.current_task()
        if task is not None and task is self._lifecycle_lock_owner:
            yield
            return
        async with self._lifecycle_lock:
            self._lifecycle_lock_owner = task
            try:
                yield
            finally:
                self._lifecycle_lock_owner = None

    async def _open_initial_session_locked(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        result = await self._dispatch_backend_host_operation(method, raw_params)
        if result is not None:
            return result
        raise RequestFailure(
            ProtocolErrorCode.CONFLICT,
            "Start, resume, or continue a session before using this method",
        )

    def _initialize(self, raw_params: dict[str, Any]) -> InitializeResponse:
        if self._initialization is not InitializationState.UNINITIALIZED:
            raise RequestFailure(
                ProtocolErrorCode.INVALID_REQUEST, "initialize may only be called once"
            )
        params = validate_wire(InitializeParams, raw_params)
        self._client_info = params.client_info
        self._client_capabilities = params.capabilities
        self._initialization = InitializationState.INITIALIZE_RECEIVED
        return InitializeResponse(
            server_info=ServerInfo(name="chartreux-app-server", version=__version__)
        )

    def _initialized_client_info(self) -> ClientInfo:
        if self._client_info is None:
            raise RuntimeError("App-server client metadata is unavailable")
        return self._client_info

    def _handle_notification(self, notification: Notification) -> None:
        if (
            notification.method == "initialized"
            and self._initialization is InitializationState.INITIALIZE_RECEIVED
        ):
            self._initialization = InitializationState.INITIALIZED
            return
        raise JsonRpcProtocolError(
            f"Unknown or unexpected client notification: {notification.method}"
        )

    async def _handle_response(
        self, response: JsonRpcErrorResponse | JsonRpcSuccessResponse
    ) -> None:
        request_id = response.id
        if not isinstance(request_id, int):
            raise JsonRpcProtocolError("Server request responses require integer IDs")
        pending = self._client_requests.pop(request_id, None)
        if pending is not None:
            self._resolve_client_request(request_id, pending, response)
            return
        if request_id in self._abandoned_client_request_ids:
            self._abandoned_client_request_ids.remove(request_id)
            return
        callback_route = self._callback_requests.pop(request_id, None)
        if callback_route is None:
            raise JsonRpcProtocolError(
                f"Response does not match a pending server request: {request_id}"
            )
        session_id = callback_route.session_id
        callback_id = callback_route.callback_id
        if isinstance(response, JsonRpcErrorResponse):
            if callback_route.answered:
                return
            await self._reject_callback(session_id, callback_id, response.error.message)
            return
        acknowledgement = validate_wire(CallbackCallResponse, response.result)
        validate_callback_acknowledgement(callback_id, acknowledgement)
        if not acknowledgement.accepted:
            if callback_route.answered:
                return
            await self._reject_callback(
                session_id, callback_id, "Client did not accept callback delivery"
            )

    def _resolve_client_request(
        self,
        request_id: int,
        pending: PendingClientRequest,
        response: JsonRpcErrorResponse | JsonRpcSuccessResponse,
    ) -> None:
        if pending.future.cancelled():
            self._abandoned_client_request_ids.discard(request_id)
            return
        if isinstance(response, JsonRpcErrorResponse):
            pending.future.set_exception(AppServerResponseError(response.error))
            return
        try:
            result = validate_wire(pending.response_type, response.result)
        except Exception as exc:
            pending.future.set_exception(exc)
        else:
            pending.future.set_result(result)

    async def _notify_host(self, params: UsageUpdatedParams) -> None:
        """Global notifications bypass root publication and event sequencing."""
        method = params.NOTIFICATION_METHOD
        validate_notification_method(method, params)
        if (
            self._closed
            or self._initialization != InitializationState.INITIALIZED
            or method in self._client_capabilities.disabled_notifications
        ):
            return
        await self._send_notification(method, params)

    async def _notify(self, method: str, params: ProtocolModel) -> None:
        validate_notification_method(method, params)
        params = self._sequence_notification(params)
        root = self._root
        if root is not None and await root.publish_notification(method, params):
            return
        await self._route_notification(method, params)

    async def _route_notification(self, method: str, params: ProtocolModel) -> None:
        validate_notification_method(method, params)
        if (
            method in self._client_capabilities.disabled_notifications
            and not isinstance(params, EventNotificationParams)
        ):
            return
        if self._attaching:
            self._pending_notifications.append(PendingNotification(method, params))
            return
        if not self._connection_attached:
            return
        await self._send_notification(method, params)

    async def _send_notification(self, method: str, params: ProtocolModel) -> None:
        transport = self._transport
        if transport is None:
            return
        try:
            await transport.send({
                "jsonrpc": "2.0",
                "method": method,
                "params": params.model_dump(mode="json", by_alias=True),
            })
        except Exception:
            self._disconnect_current_client()

    async def _deliver_callback(self, callback: PublicCallbackEntry) -> None:
        transport = self._transport
        if transport is None or not self._connection_attached or self._attaching:
            return
        try:
            target = self._callback_delivery_target(callback.callback_id)
        except LiveAgentUnavailableError:
            # Bidirectional invalidation reaches prompts, not just frames: the
            # routing row went away with its agent's incarnation, so no answer
            # could be routed back. Showing the prompt would strand the client.
            return
        callback_kind = callback.detail.kind
        if callback_kind not in self._client_capabilities.callback_kinds:
            raise RuntimeError(f"Client does not support {callback_kind} callbacks")
        request_id = self._next_request_id
        self._next_request_id += 1
        self._callback_requests[request_id] = CallbackDelivery(
            session_id=callback.session_id,
            callback_id=callback.callback_id,
            target=target,
        )
        params = CallbackCallParams(callback=callback)
        try:
            await transport.send({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "callback/call",
                "params": params.model_dump(mode="json", by_alias=True),
            })
        except Exception:
            self._callback_requests.pop(request_id, None)
            self._disconnect_current_client()

    async def _publish_callback(self, callback: PublicCallbackEntry) -> None:
        root = self._root
        if root is not None and await root.publish_callback(callback):
            return
        await self._deliver_callback(callback)

    def _registry_manifest(self) -> RegistryManifest | None:
        """The registry manifest of the root session, when it has one.

        The manifest belongs to the runtime factory keyed by the root agent
        loop, so the app server reaches it through the root backend's request
        handler — the only surface that names that loop. A backend without a
        handler, or without a registered manifest, is an in-process runtime
        whose callbacks stay in memory.
        """
        handler = getattr(self._root, "handler", None)
        factory = getattr(handler, "_runtime_factory", None)
        loop = getattr(handler, "_agent_loop", None)
        if factory is None or loop is None:
            return None
        manifest: RegistryManifest | None = factory.registry_manifest(loop)
        return manifest

    def _callback_delivery_target(self, callback_id: str) -> CallbackTarget | None:
        """The manifest route a callback prompt is delivered for, if routed."""
        manifest = self._registry_manifest()
        if manifest is None:
            return None
        return manifest.delivery_target(callback_id)

    async def _request_client_result[ResultT: ProtocolModel](
        self, method: str, params: ProtocolModel, response_type: type[ResultT]
    ) -> ResultT:
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[ProtocolModel] = (
            asyncio.get_running_loop().create_future()
        )
        self._client_requests[request_id] = PendingClientRequest(
            response_type=response_type, future=future
        )
        transport = self._transport
        if transport is None or not self._connection_attached or self._attaching:
            self._client_requests.pop(request_id, None)
            raise RuntimeError("No app-server client is attached")
        sent = False
        try:
            await transport.send({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params.model_dump(mode="json", by_alias=True),
            })
            sent = True
            return cast(ResultT, await future)
        except asyncio.CancelledError:
            if sent and self._client_requests.pop(request_id, None) is not None:
                self._abandoned_client_request_ids.add(request_id)
            raise
        except Exception:
            self._disconnect_current_client()
            raise
        finally:
            self._client_requests.pop(request_id, None)

    def _mark_callback_answered(self, callback_id: str) -> None:
        for delivery in self._callback_requests.values():
            if delivery.callback_id == callback_id:
                delivery.answered = True

    async def _redeliver_open_callbacks(self) -> None:
        root = self._root
        if root is None:
            return
        for callback in root.open_callbacks():
            if not isinstance(callback.state, OpenCallbackState):
                continue
            await self._deliver_callback(callback)

    def _validate_open_callback_capabilities(self) -> None:
        root = self._root
        if root is None:
            return
        required = {
            callback.detail.kind
            for callback in root.open_callbacks()
            if isinstance(callback.state, OpenCallbackState)
        }
        missing = sorted(required - set(self._client_capabilities.callback_kinds))
        if not missing:
            return
        raise RequestFailure(
            ProtocolErrorCode.INVALID_PARAMS,
            f"Client does not support open callback kinds: {', '.join(missing)}",
            data=cast(JsonValue, {"missingCallbackKinds": missing}),
        )

    async def _record_child_notification(
        self, method: str, params: ProtocolModel
    ) -> None:
        if method == ChildCallbackClosedParams.NOTIFICATION_METHOD:
            # Child history never enters the ROOT projection, and every other
            # child notification is only re-sequenced (its state lives in the
            # child session). This one must reach the attached client: without
            # it a parent-side widget for a torn-down child callback strands.
            # _notify validates the declared method and routes it on the normal
            # notification path — through the root backend stream when one is
            # subscribed, straight to the transport otherwise.
            await self._notify(method, params)
            return
        self._sequence_notification(params)

    def client_info(self) -> ClientInfo:
        return self._initialized_client_info()

    def client_capabilities(self) -> ClientCapabilities:
        return self._client_capabilities

    def current_session_id(self) -> str:
        return self._require_root().session_id

    def event_watermark(self, session_id: str) -> int:
        return self._event_watermark(session_id)

    def lifecycle_transition(self) -> AbstractAsyncContextManager[None]:
        return self._lifecycle_transition()

    def task_finished(self, task: asyncio.Task[None]) -> None:
        self._request_finished(task)

    async def notify(self, method: str, params: ProtocolModel) -> None:
        await self._notify(method, params)

    async def publish_callback(self, callback: PublicCallbackEntry) -> None:
        await self._publish_callback(callback)

    async def record_child_notification(
        self, method: str, params: ProtocolModel
    ) -> None:
        await self._record_child_notification(method, params)

    async def request_client_result[ResultT: ProtocolModel](
        self, method: str, params: ProtocolModel, response_type: type[ResultT]
    ) -> ResultT:
        return await self._request_client_result(method, params, response_type)

    def _sequence_notification(self, params: ProtocolModel) -> ProtocolModel:
        if not isinstance(params, EventNotificationParams):
            return params
        session_id = _notification_session_id(params)
        event_id = self._event_watermark(session_id) + 1
        self._event_watermarks[session_id] = event_id
        update: dict[str, Any] = {"event_id": event_id}
        if isinstance(params, SessionSnapshotParams | SessionHandoffParams):
            update["state"] = params.state.model_copy(update={"event_id": event_id})
        return params.model_copy(update=update)

    async def _reject_callback(
        self, session_id: str, callback_id: str, message: str
    ) -> None:
        error = CallbackResultError(message=message)
        root = self._root
        if root is None:
            raise RuntimeError("The selected backend cannot reject callback delivery")
        await root.reject_callback_delivery(session_id, callback_id, error)

    async def _send_error(self, request_id: Any, error: ProtocolError) -> None:
        await self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error.model_dump(mode="json", by_alias=True),
        })

    async def _send(self, message: dict[str, Any]) -> None:
        transport = self._transport
        if transport is None:
            raise RuntimeError("No app-server client is attached")
        await transport.send(message)

    def _disconnect_current_client(self) -> None:
        task = self._serve_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _event_watermark(self, session_id: str) -> int:
        return self._event_watermarks.get(session_id, 0)


def _omits_session_id(method: str, raw_params: dict[str, Any]) -> bool:
    """Whether a method that can answer from either scope named no session.

    `_SESSION_OPTIONAL_METHODS` take an optional session id. When it is absent
    the caller is asking the host, so they must not be routed to an attached
    root: the answer would otherwise depend on whether a session happens to be
    attached, and a `cwd` the host would have honoured would be silently
    dropped. The handler still validates the params it was given.
    """
    return method in _SESSION_OPTIONAL_METHODS and raw_params.get("sessionId") is None


def _notification_session_id(params: EventNotificationParams) -> str:
    return params.session_id
