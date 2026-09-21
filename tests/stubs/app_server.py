from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from chartreux.app_server._host import HostRequestHandler
from chartreux.app_server._mcp_auth import MCPAuthenticationService
from chartreux.app_server._projector import EventProjector
from chartreux.app_server._runtime import AgentRuntimeFactory, RootOpenRequest
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._session_backend_port import SessionBackendHost
from chartreux.app_server._session_backend_services import SessionBackendServices
from chartreux.app_server._session_runtime_impl import (
    OpenRoot,
    StageRoot,
    create_session_backend_host_impl,
)
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import AppServerEvent, ClientProjection
from chartreux.app_server.mcp_catalog import MCPCatalogService
from chartreux.app_server.models import (
    IdleSessionStatus,
    PublicHistoryEntry,
    PublicSession,
    PublicSessionState,
)
from chartreux.app_server.protocol import (
    ClientCapabilities,
    ClientInfo,
    Notification,
    SessionOptions,
    TransportKind,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import JsonRpcTransport, memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.events import BaseEvent, ToolCallEvent, ToolResultEvent
from chartreux.core.tools.ui import ToolUIDataAdapter


def create_legacy_app_server(
    transport: JsonRpcTransport,
    *,
    open_root: OpenRoot,
    transport_kind: TransportKind = "in_process",
    runtime_factory: AgentRuntimeFactory | None = None,
    host_handler: HostRequestHandler | None = None,
    stage_root: StageRoot | None = None,
    mcp_catalog_service: MCPCatalogService | None = None,
) -> AppServer:
    effective_host_handler = host_handler or HostRequestHandler(
        HarnessFilesManager(sources=("user", "project"))
    )
    runtime_factory = runtime_factory or AgentRuntimeFactory()
    effective_mcp_catalog = mcp_catalog_service or MCPCatalogService(
        MCPAuthenticationService()
    )

    def create_session_backend_host(
        services: SessionBackendServices,
    ) -> SessionBackendHost:
        return create_session_backend_host_impl(
            open_root=open_root,
            runtime_factory=runtime_factory,
            host_handler=effective_host_handler,
            stage_root=stage_root,
            services=services,
            mcp_catalog_service=effective_mcp_catalog,
        )

    return AppServer(
        transport,
        session_backend_host_factory=create_session_backend_host,
        transport_kind=transport_kind,
        host_handler=effective_host_handler,
        mcp_catalog_service=effective_mcp_catalog,
    )


class CoreEventProjection:
    def __init__(self) -> None:
        session_id = "session-1"
        self._projector = EventProjector(session_id, "turn-1")
        self._projection = ClientProjection(
            PublicSessionState(
                event_id=0,
                session=PublicSession(
                    id=session_id,
                    status=IdleSessionStatus(),
                    created_at=1,
                    updated_at=1,
                ),
                history=[],
                active_callbacks=[],
                turns=[],
            )
        )
        self._event_id = 0

    def project(self, event: BaseEvent) -> list[AppServerEvent]:
        match event:
            case ToolCallEvent(presentation=None):
                event = event.model_copy(
                    update={
                        "presentation": ToolUIDataAdapter(
                            event.tool_class
                        ).get_call_presentation(event)
                    }
                )
            case ToolResultEvent(presentation=None):
                event = event.model_copy(
                    update={
                        "presentation": ToolUIDataAdapter(
                            event.tool_class
                        ).get_result_presentation(event)
                    }
                )
        projected: list[AppServerEvent] = []
        for update in self._projector.project(event):
            self._event_id += 1
            params = update.params.model_copy(update={"event_id": self._event_id})
            notification = Notification(
                method=update.method,
                params=params.model_dump(mode="json", by_alias=True),
            )
            if client_event := self._projection.consume(notification):
                projected.append(client_event)
        return projected

    @property
    def history(self) -> list[PublicHistoryEntry]:
        return self._projection.history

    async def dispatch[ResultT](
        self, event: BaseEvent, consumer: Callable[[AppServerEvent], Awaitable[ResultT]]
    ) -> list[ResultT]:
        return [await consumer(projected) for projected in self.project(event)]


def start_test_app_server(agent_loop: AgentLoop) -> AppServerClient:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    return AppServerClient(client_transport, run_peer=server.serve)


def build_test_app_server(
    agent_loop: AgentLoop, transport: JsonRpcTransport
) -> AppServer:
    runtime_factory = AgentRuntimeFactory()

    async def open_root(request: RootOpenRequest) -> AgentLoop:
        session_id = request.session_id
        if request.continue_latest:
            session_id = runtime_factory.resolve_latest(
                agent_loop, Path(request.options.cwd or agent_loop.cwd)
            )
        if session_id is not None:
            await runtime_factory.resume_root(agent_loop, session_id)
        return agent_loop

    return create_legacy_app_server(
        transport, open_root=open_root, runtime_factory=runtime_factory
    )


def legacy_backend(server: AppServer) -> SessionBackendImpl:
    root = server._root
    assert isinstance(root, SessionBackendImpl)
    return root


async def create_test_app_server_session(agent_loop: AgentLoop) -> AppServerSession:
    return await attach_test_app_server_session(start_test_app_server(agent_loop))


async def attach_test_app_server_session(
    client: AppServerClient,
    *,
    resume_session_id: str | None = None,
    session_options: SessionOptions | None = None,
) -> AppServerSession:
    return await AppServerSession.start(
        client,
        client_info=ClientInfo(name="vibe_test", version="0"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
        resume_session_id=resume_session_id,
        session_options=session_options,
    )
