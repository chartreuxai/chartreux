from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import JsonValue

from chartreux.app_server._dispatch import DispatchResult
from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.events import AppServerEvent
from chartreux.app_server.models import PublicCallbackEntry
from chartreux.app_server.protocol import (
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
    PolicyReadParams,
    PolicyReadResponse,
    PolicyReplaceParams,
    PolicyReplaceResponse,
    ProtocolErrorCode,
    RootsReadParams,
    RootsReadResponse,
    RootsReplaceParams,
    RootsReplaceResponse,
    RuntimeUpdatedParams,
    SessionCompactParams,
    SessionCompactResponse,
    SessionContinueParams,
    SessionForkParams,
    SessionForkResponse,
    SessionListParams,
    SessionListResponse,
    SessionReadParams,
    SessionReadResponse,
    SessionResumeParams,
    SessionSettingsUpdateParams,
    SessionStartParams,
    SessionTitleUpdateParams,
    SessionTitleUpdateResponse,
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
    TurnStartParams,
    TurnStartResponse,
    TurnSteerParams,
    TurnSteerResponse,
)

type MCPAuthorizationReason = Literal["missing", "expired", "rejected", "invalid"]


@dataclass(frozen=True, slots=True)
class MCPAuthorizationRef:
    server_name: str
    server_fingerprint: str
    kind: Literal["none", "static", "oauth"]
    descriptor_revision: str
    binding_id: str = ""


@dataclass(frozen=True, slots=True)
class MCPAuthorizationSnapshot:
    headers: Mapping[str, str] = field(repr=False)
    connection_revision: str
    descriptor_revision: str
    expires_at: datetime | None = None
    # Process-private descriptor context; never projected onto the wire.
    _descriptor_context: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class MCPAuthorizationRequired:
    reason: MCPAuthorizationReason
    descriptor_revision: str
    observed_connection_revision: str | None = None


type MCPAuthorizationResult = MCPAuthorizationSnapshot | MCPAuthorizationRequired


class MCPAuthorizationProvider(Protocol):
    async def resolve(
        self, reference: MCPAuthorizationRef
    ) -> MCPAuthorizationResult: ...

    async def reject(
        self,
        reference: MCPAuthorizationRef,
        *,
        observed_connection_revision: str,
        reason: Literal["http_unauthorized", "mcp_unauthorized"],
    ) -> MCPAuthorizationResult: ...


@dataclass(frozen=True, slots=True)
class ResolvedMCPServerConfig:
    name: str
    transport: Literal["streamable-http", "stdio"]
    url: str | None
    command: str | None
    args: tuple[str, ...]
    cwd: Path | None
    env: Mapping[str, str]
    authorization: MCPAuthorizationRef
    prompt: str | None
    startup_timeout_s: float
    tool_timeout_s: float
    disabled: bool
    disabled_tools: frozenset[str]


@dataclass(frozen=True, slots=True)
class ResolvedMCPCatalog:
    revision: str
    servers: tuple[ResolvedMCPServerConfig, ...]


@dataclass(frozen=True, slots=True)
class SessionMCPToolDescriptor:
    remote_name: str
    description: str
    enabled: bool
    display_name: str


@dataclass(frozen=True, slots=True)
class SessionMCPSourceState:
    name: str
    transport: Literal["streamable-http", "stdio"]
    status: Literal["disabled", "enabled", "connected", "needs_auth", "unavailable"]
    tools: tuple[SessionMCPToolDescriptor, ...] = ()
    descriptor_revision: str = ""
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SessionMCPState:
    catalog_revision: str
    route_revision: str
    sources: tuple[SessionMCPSourceState, ...]
    discovery_errors: Mapping[str, str]


class SessionBackendError(Exception):
    def __init__(
        self,
        code: ProtocolErrorCode,
        message: str,
        data: JsonValue = None,
        *,
        after_response: Callable[[], None] | None = None,
        on_response_abandoned: Callable[[], None] | None = None,
    ) -> None:
        self.code = code
        self.data = data
        self.after_response = after_response
        self.on_response_abandoned = on_response_abandoned
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SessionBackendEvent:
    event: AppServerEvent
    method: str | None = None
    params: ProtocolModel | None = None
    session_id: str | None = None
    event_id: int | None = None


@dataclass(frozen=True, slots=True)
class SessionEventSubscription:
    snapshot: SessionReadResponse
    events: AsyncIterator[SessionBackendEvent]


@dataclass(frozen=True, slots=True)
class SessionForkResult:
    response: SessionForkResponse
    backend: SessionBackend | None
    after_response: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class SessionLifecycleResult:
    """Result of a session lifecycle operation (start, resume, continue).

    Carries the activated backend plus an optional deferred action to run
    after the RPC response is sent. The server calls ``after_response()``
    once the client has received the result, so long-running post-activation
    work (e.g. awaiting deferred init) does not block the response.
    """

    backend: SessionBackend
    after_response: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class SessionBackendResult[ResponseT: ProtocolModel]:
    response: ResponseT
    after_response: Callable[[], None] | None = None
    on_response_abandoned: Callable[[], None] | None = None
    runtime_updated: bool = False


class SessionBackend(Protocol):  # noqa: PLR0904 - unified backend contract
    @property
    def session_id(self) -> str: ...

    async def read(self, params: SessionReadParams) -> SessionReadResponse: ...

    async def subscribe(
        self, params: SessionReadParams
    ) -> SessionEventSubscription: ...

    def guard_request(self) -> None: ...

    async def update_settings(
        self, params: SessionSettingsUpdateParams
    ) -> SessionBackendResult[EmptyResponse]: ...

    async def write_config(
        self, params: ConfigWriteParams
    ) -> SessionBackendResult[ConfigWriteResponse]: ...

    async def reload_config(
        self, params: ConfigReloadParams
    ) -> SessionBackendResult[ConfigMutationResponse]: ...

    async def read_policy(self, params: PolicyReadParams) -> PolicyReadResponse: ...

    async def replace_policy(
        self, params: PolicyReplaceParams
    ) -> SessionBackendResult[PolicyReplaceResponse]: ...

    async def read_roots(self, params: RootsReadParams) -> RootsReadResponse: ...

    async def replace_roots(
        self, params: RootsReplaceParams
    ) -> SessionBackendResult[RootsReplaceResponse]: ...

    async def read_mcp(self) -> SessionMCPState: ...

    async def reconfigure_mcp(
        self, configuration: ResolvedMCPCatalog, *, force_remote_discovery: bool
    ) -> SessionMCPState: ...

    async def authorization_changed(
        self, *, name: str, descriptor_revision: str
    ) -> SessionMCPState: ...

    async def suspend_mcp(
        self,
        *,
        name: str,
        tool_name: str | None,
        reason: Literal["logout", "remove", "disable", "replace"],
    ) -> SessionMCPState: ...

    async def start_turn(
        self, params: TurnStartParams
    ) -> SessionBackendResult[TurnStartResponse]: ...

    async def enqueue_turn(
        self, params: TurnEnqueueParams
    ) -> SessionBackendResult[TurnEnqueueResponse]: ...

    async def read_turn_queue(
        self, params: TurnQueueReadParams
    ) -> SessionBackendResult[TurnQueueReadResponse]: ...

    async def remove_queued_turn(
        self, params: TurnQueueRemoveParams
    ) -> SessionBackendResult[TurnQueueRemoveResponse]: ...

    async def replace_queued_turn(
        self, params: TurnQueueReplaceParams
    ) -> SessionBackendResult[TurnQueueReplaceResponse]: ...

    async def resume_turn_queue(
        self, params: TurnQueueResumeParams
    ) -> SessionBackendResult[TurnQueueResumeResponse]: ...

    async def steer_turn(
        self, params: TurnSteerParams
    ) -> SessionBackendResult[TurnSteerResponse]: ...

    async def interrupt_turn(
        self, params: TurnInterruptParams
    ) -> SessionBackendResult[TurnInterruptResponse]: ...

    async def inject_context(
        self, params: ContextInjectParams
    ) -> SessionBackendResult[ContextInjectResponse]: ...

    async def respond_to_callback(
        self, params: CallbackResultParams
    ) -> SessionBackendResult[CallbackResultResponse]: ...

    async def compact(
        self, params: SessionCompactParams
    ) -> SessionBackendResult[SessionCompactResponse]: ...

    async def dispatch_extension(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult: ...

    async def publish_notification(
        self, method: str, params: ProtocolModel
    ) -> bool: ...

    async def publish_callback(self, callback: PublicCallbackEntry) -> bool: ...

    async def flush_events(self) -> None: ...

    def runtime_updated_params(self) -> RuntimeUpdatedParams: ...

    def open_callbacks(self) -> list[PublicCallbackEntry]: ...

    async def reject_callback_delivery(
        self, session_id: str, callback_id: str, error: CallbackResultError
    ) -> None: ...

    def references_child(self, session_id: str) -> bool: ...

    async def shutdown(self) -> None: ...


class SessionBackendHost(Protocol):
    async def start(self, params: SessionStartParams) -> SessionLifecycleResult: ...

    async def resume(self, params: SessionResumeParams) -> SessionLifecycleResult: ...

    async def continue_latest(
        self, params: SessionContinueParams
    ) -> SessionLifecycleResult: ...

    async def fork(self, params: SessionForkParams) -> SessionForkResult: ...

    async def list(self, params: SessionListParams) -> SessionListResponse: ...

    async def read(self, params: SessionReadParams) -> SessionReadResponse: ...

    async def rename(
        self, params: SessionTitleUpdateParams
    ) -> SessionTitleUpdateResponse: ...

    async def stop_background_tasks(self, current: Any) -> list[BaseException]: ...

    async def shutdown(self) -> None: ...
