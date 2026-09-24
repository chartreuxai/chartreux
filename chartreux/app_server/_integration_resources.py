from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from chartreux.app_server._model import validate_wire
from chartreux.app_server._streaming import BoundedEventQueue, stream_request
from chartreux.app_server.client_state import ClientSessionState
from chartreux.app_server.connection import AppServerResourceConnection
from chartreux.app_server.models import MCPState, SkillSummary
from chartreux.app_server.protocol import (
    MCPAddParams,
    MCPAddResponse,
    MCPAuthUrlParams,
    MCPCatalogMutationResponse,
    MCPLoginParams,
    MCPLogoutParams,
    MCPReadParams,
    MCPReadResponse,
    MCPRefreshParams,
    MCPToggleParams,
    Notification,
    RuntimeSnapshot,
    SkillsInstalledParams,
    SkillsInstalledResponse,
)
from chartreux.utils.mcp import MCPAddTransport


def _required_mcp_runtime(runtime: RuntimeSnapshot | None) -> RuntimeSnapshot:
    if runtime is None:
        raise RuntimeError("A session-targeted MCP mutation returned no runtime")
    return runtime


class SkillsResource:
    """Read the session's discovered local skills through the app server."""

    def __init__(
        self, connection: AppServerResourceConnection, state: ClientSessionState
    ) -> None:
        self._connection = connection
        self._state = state

    @property
    def installed(self) -> list[SkillSummary]:
        return self._state.skills

    async def read_installed(self) -> list[SkillSummary]:
        client = await self._connection.connect()
        response = validate_wire(
            SkillsInstalledResponse,
            await client.request(
                "skills/installed",
                SkillsInstalledParams(session_id=self._state.session_id),
            ),
        )
        return response.skills


class MCPResource:
    _NOTIFICATION_METHODS = frozenset({"mcp_catalog/authUrl"})

    @classmethod
    def notification_methods(cls) -> frozenset[str]:
        return cls._NOTIFICATION_METHODS

    def __init__(
        self, connection: AppServerResourceConnection, state: ClientSessionState
    ) -> None:
        self._connection = connection
        self._state = state
        self._login_events: dict[str, asyncio.Queue[MCPAuthUrlParams]] = {}

    @property
    def state(self) -> MCPState:
        return self._state.mcp

    async def read(self) -> MCPState:
        client = await self._connection.connect()
        mcp_response = validate_wire(
            MCPReadResponse,
            await client.request(
                "mcp_catalog/read", MCPReadParams(session_id=self._state.session_id)
            ),
        )
        state = mcp_response.mcp
        self._state.mcp = state
        return state

    async def refresh(self) -> MCPState:
        client = await self._connection.connect()
        response = validate_wire(
            MCPCatalogMutationResponse,
            await client.request(
                "mcp_catalog/refresh",
                MCPRefreshParams(session_id=self._state.session_id),
            ),
        )
        runtime = _required_mcp_runtime(response.runtime)
        self._state.apply_runtime(runtime)
        return runtime.mcp

    async def toggle(
        self, name: str, *, disabled: bool, tool_name: str | None = None
    ) -> MCPState:
        client = await self._connection.connect()
        response = validate_wire(
            MCPCatalogMutationResponse,
            await client.request(
                "mcp_catalog/toggle",
                MCPToggleParams(
                    session_id=self._state.session_id,
                    name=name,
                    disabled=disabled,
                    tool_name=tool_name,
                ),
            ),
        )
        runtime = _required_mcp_runtime(response.runtime)
        self._state.apply_runtime(runtime)
        return runtime.mcp

    async def add(
        self,
        *,
        url: str,
        name: str | None,
        scopes: list[str],
        transport: MCPAddTransport,
    ) -> MCPAddResponse:
        client = await self._connection.connect()
        response = validate_wire(
            MCPAddResponse,
            await client.request(
                "mcp_catalog/add",
                MCPAddParams(
                    session_id=self._state.session_id,
                    url=url,
                    name=name,
                    scopes=scopes,
                    transport=transport,
                ),
            ),
        )
        self._state.apply_runtime(_required_mcp_runtime(response.runtime))
        return response

    async def logout(self, name: str) -> MCPState:
        client = await self._connection.connect()
        response = validate_wire(
            MCPCatalogMutationResponse,
            await client.request(
                "mcp_catalog/logout",
                MCPLogoutParams(session_id=self._state.session_id, name=name),
            ),
        )
        runtime = _required_mcp_runtime(response.runtime)
        self._state.apply_runtime(runtime)
        return runtime.mcp

    async def login(self, name: str) -> AsyncGenerator[MCPAuthUrlParams, None]:
        if name in self._login_events:
            raise RuntimeError(f"MCP login already in progress: {name}")
        client = await self._connection.connect()
        events = BoundedEventQueue[MCPAuthUrlParams]()
        self._login_events[name] = events
        try:
            async for event in stream_request(
                client,
                "mcp_catalog/login",
                MCPLoginParams(session_id=self._state.session_id, name=name),
                events,
                MCPCatalogMutationResponse,
            ):
                if isinstance(event, MCPCatalogMutationResponse):
                    self._state.apply_runtime(_required_mcp_runtime(event.runtime))
                else:
                    yield event
        finally:
            self._login_events.pop(name, None)

    async def consume_notification(self, notification: Notification) -> bool:
        if notification.method not in self.notification_methods():
            return False
        params = validate_wire(MCPAuthUrlParams, notification.params)
        if events := self._login_events.get(params.name):
            await events.put(params)
        return True
