from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._mcp_auth import MCPAuthenticationService
from chartreux.app_server._mcp_authorization_bridge import (
    app_authorization_ref,
    registry_authorization_ref,
)
from chartreux.app_server._session_backend_impl import configure_mcp_registry_impl
from chartreux.app_server._session_backend_port import (
    MCPAuthorizationRequired,
    MCPAuthorizationSnapshot,
    SessionMCPSourceState,
    SessionMCPState,
    SessionMCPToolDescriptor,
)
from chartreux.app_server.mcp_catalog import MCPCatalogService, project_mcp_sources
from chartreux.app_server.models import MCPSourceStatus
from chartreux.app_server.protocol import (
    AppServerResponseError,
    Notification,
    ProtocolErrorCode,
)
from chartreux.core.config import MCPHttp, MCPOAuth, MCPStaticAuth, MCPStdio
from chartreux.core.config.types import ConcurrencyConflictError
from chartreux.core.tools.base import BaseToolConfig, ToolError
from chartreux.core.tools.mcp import MCPToolResult, RemoteTool
from chartreux.core.tools.mcp.authorization import (
    MCPAuthorizationRequired as RegistryAuthorizationRequired,
)
from chartreux.core.tools.mcp.registry import MCPRegistry
from chartreux.core.tools.mcp.tools import MCPHttpAuthorizationRuntime, _OpenArgs
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


class FailingMCPRegistry(FakeMCPRegistry):
    """Registry that always fails discovery for a named server."""

    def __init__(self, failing_server: str) -> None:
        super().__init__()
        self._failing_server = failing_server

    async def get_tools_async(self, servers):
        working = [s for s in servers if s.name != self._failing_server]
        for s in servers:
            if s.name == self._failing_server:
                self._failed[s.name] = "connection refused"
        return await super().get_tools_async(working)


def test_project_mcp_sources_includes_disabled_server_without_discovery() -> None:
    config = build_test_vibe_config(
        mcp_servers=[
            MCPStdio(name="local", transport="stdio", command="fake-mcp", disabled=True)
        ]
    )

    sources, discovery_errors = project_mcp_sources(
        orchestrator=FakeConfigOrchestrator(config),
        state=SessionMCPState(
            catalog_revision="catalog",
            route_revision="route",
            sources=(),
            discovery_errors={},
        ),
    )

    assert [(source.name, source.status) for source in sources] == [
        ("local", MCPSourceStatus.DISABLED)
    ]
    assert discovery_errors == {}


def test_project_mcp_sources_projects_direct_server_and_tool_description() -> None:
    config = build_test_vibe_config(
        mcp_servers=[
            MCPHttp(
                name="linear",
                transport="streamable-http",
                url="https://mcp.example.test",
            )
        ]
    )
    sources, errors = project_mcp_sources(
        orchestrator=FakeConfigOrchestrator(config),
        state=SessionMCPState(
            catalog_revision="catalog",
            route_revision="route",
            discovery_errors={},
            sources=(
                SessionMCPSourceState(
                    name="linear",
                    transport="streamable-http",
                    status="connected",
                    tools=(
                        SessionMCPToolDescriptor(
                            remote_name="create_issue",
                            display_name="linear_create_issue",
                            enabled=True,
                            description="Create an issue.\n\nAccepts a title and a team id.\n- title\n- team",
                        ),
                    ),
                ),
            ),
        ),
    )
    assert [(source.name, source.status) for source in sources] == [
        ("linear", MCPSourceStatus.CONNECTED)
    ]
    assert [tool.description for tool in sources[0].tools] == ["Create an issue."]
    assert "pluginName" not in sources[0].model_dump(mode="json")
    assert errors == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "toggle"])
async def test_mcp_config_conflicts_are_public_conflicts(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """*Prepare*: The authoritative catalog persistence helper reports a conflict.
    *Do*: Submit add and toggle through the public MCP resource.
    *Assert*: The app server returns the stable public conflict code.
    """
    # Prepare
    conflict = ConcurrencyConflictError("expected", "actual")
    target = "persist_oauth_mcp_server" if operation == "add" else "persist_mcp_toggle"
    monkeypatch.setattr(
        f"chartreux.app_server.mcp_catalog.{target}", AsyncMock(side_effect=conflict)
    )
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        with pytest.raises(AppServerResponseError) as exc_info:
            if operation == "add":
                await session.resources.mcp.add(
                    url="https://mcp.example.com/mcp",
                    name=None,
                    scopes=[],
                    transport="streamable-http",
                )
            else:
                await session.resources.mcp.toggle("search", disabled=True)
    finally:
        await session.close()

    # Assert
    assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT


@pytest.mark.asyncio
async def test_mcp_login_streams_typed_auth_url_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: The process authentication service begins an interactive login.
    *Do*: Consume the public login stream.
    *Assert*: The typed catalog auth URL reaches the MCP resource.
    """
    # Prepare
    login_calls: list[str] = []

    async def login(
        _service: MCPAuthenticationService,
        name: str,
        *,
        on_url: Callable[[str], Awaitable[None]],
        owner: object | None = None,
    ) -> str:
        login_calls.append(name)
        await on_url("https://auth.example.com/oauth")
        return _service.descriptor_revision(name)

    monkeypatch.setattr(MCPAuthenticationService, "login", login)
    config = build_test_vibe_config(
        mcp_servers=[
            MCPHttp(
                name="search",
                transport="streamable-http",
                url="https://mcp.example.com",
                auth=MCPOAuth(type="oauth", scopes=[]),
            )
        ]
    )
    agent_loop = build_test_agent_loop(config=config, mcp_registry=FakeMCPRegistry())
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        events = [event async for event in session.resources.mcp.login("search")]
    finally:
        await session.close()

    # Assert
    assert login_calls == ["search"]
    assert [(event.name, event.url) for event in events] == [
        ("search", "https://auth.example.com/oauth")
    ]


@pytest.mark.asyncio
async def test_unknown_notifications_do_not_enter_mcp_login_stream() -> None:
    """*Prepare*: A connected MCP resource has no matching notification stream.
    *Do*: Deliver an unknown future notification.
    *Assert*: The MCP resource leaves it unconsumed.
    """
    # Prepare
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        consumed = await session.resources.consume_notification(
            Notification(method="future/event", params={})
        )
    finally:
        await session.close()

    # Assert
    assert consumed is False


@pytest.mark.asyncio
async def test_mcp_toggle_enable_reuses_valid_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: An enabled server and observable legacy reconfiguration paths.
    *Do*: Enable the configured server through the catalog facade.
    *Assert*: Ordinary convergence retains valid descriptors instead of forcing refresh.
    """
    # Prepare
    monkeypatch.setattr(
        "chartreux.app_server.mcp_catalog.persist_mcp_toggle", AsyncMock()
    )
    server = MCPStdio(name="search", transport="stdio", command="fake-cmd")
    config = build_test_vibe_config(mcp_servers=[server])
    agent_loop = build_test_agent_loop(config=config)
    refresh_mock = AsyncMock()
    reconfigure_mock = AsyncMock()
    agent_loop.tool_manager.refresh_remote_tools_async = refresh_mock
    agent_loop.tool_manager.reconfigure_mcp_async = reconfigure_mock
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        await session.resources.mcp.toggle("search", disabled=False)
    finally:
        await session.close()

    # Assert
    reconfigure_mock.assert_awaited_once()
    refresh_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_toggle_disable_does_not_rediscover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: A configured server and observable legacy remote refresh.
    *Do*: Disable the server through the catalog facade.
    *Assert*: Restrictive convergence withdraws routes without rediscovery.
    """
    # Prepare
    monkeypatch.setattr(
        "chartreux.app_server.mcp_catalog.persist_mcp_toggle", AsyncMock()
    )
    server = MCPStdio(name="search", transport="stdio", command="fake-cmd")
    config = build_test_vibe_config(mcp_servers=[server])
    agent_loop = build_test_agent_loop(config=config)
    refresh_mock = AsyncMock()
    agent_loop.tool_manager.refresh_remote_tools_async = refresh_mock
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        await session.resources.mcp.toggle("search", disabled=True)
    finally:
        await session.close()

    # Assert
    refresh_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_toggle_tool_does_not_rediscover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: One server tool and observable legacy remote refresh.
    *Do*: Disable only that tool through the catalog facade.
    *Assert*: Tool-level withdrawal does not rediscover the source.
    """
    # Prepare
    monkeypatch.setattr(
        "chartreux.app_server.mcp_catalog.persist_mcp_toggle", AsyncMock()
    )
    server = MCPStdio(name="search", transport="stdio", command="fake-cmd")
    config = build_test_vibe_config(mcp_servers=[server])
    agent_loop = build_test_agent_loop(config=config)
    refresh_mock = AsyncMock()
    agent_loop.tool_manager.refresh_remote_tools_async = refresh_mock
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        await session.resources.mcp.toggle(
            "search", disabled=True, tool_name="search_web"
        )
    finally:
        await session.close()

    # Assert
    refresh_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_login_rediscovers_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """*Prepare*: A successful app-server login and observable legacy discovery.
    *Do*: Complete the public login stream.
    *Assert*: The affected legacy descriptors are rediscovered once.
    """

    # Prepare
    async def login(
        _service: MCPAuthenticationService,
        name: str,
        *,
        on_url: Callable[[str], Awaitable[None]],
        owner: object | None = None,
    ) -> str:
        await on_url("https://auth.example.com/oauth")
        return _service.descriptor_revision(name)

    monkeypatch.setattr(MCPAuthenticationService, "login", login)

    async def resolve(_service, reference):
        return MCPAuthorizationSnapshot(
            headers={"Authorization": "Bearer token"},
            connection_revision="connection-1",
            descriptor_revision=reference.descriptor_revision,
        )

    monkeypatch.setattr(MCPAuthenticationService, "resolve", resolve)
    config = build_test_vibe_config(
        mcp_servers=[
            MCPHttp(
                name="search",
                transport="streamable-http",
                url="https://mcp.example.com",
                auth=MCPOAuth(type="oauth", scopes=[]),
            )
        ]
    )
    agent_loop = build_test_agent_loop(config=config, mcp_registry=FakeMCPRegistry())
    reconfigure_mock = AsyncMock()
    agent_loop.tool_manager.reconfigure_mcp_async = reconfigure_mock
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        async for _event in session.resources.mcp.login("search"):
            pass
    finally:
        await session.close()

    # Assert
    reconfigure_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_mcp_login_keeps_disabled_source_transport_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: A disabled OAuth source completes app-owned interactive login.
    *Do*: Complete the public login stream with a live legacy session target.
    *Assert*: Runtime convergence does not resolve credentials or rediscover the source.
    """

    # Prepare
    async def login(
        _service: MCPAuthenticationService,
        name: str,
        *,
        on_url: Callable[[str], Awaitable[None]],
        owner: object | None = None,
    ) -> str:
        await on_url("https://auth.example.com/oauth")
        return "descriptor-2"

    resolve = AsyncMock(side_effect=AssertionError("disabled provider resolution"))
    monkeypatch.setattr(MCPAuthenticationService, "login", login)
    monkeypatch.setattr(MCPAuthenticationService, "resolve", resolve)
    config = build_test_vibe_config(
        mcp_servers=[
            MCPHttp(
                name="search",
                transport="streamable-http",
                url="https://mcp.example.com",
                auth=MCPOAuth(type="oauth", scopes=[]),
                disabled=True,
            )
        ]
    )
    agent_loop = build_test_agent_loop(config=config, mcp_registry=FakeMCPRegistry())
    refresh_mock = AsyncMock()
    reconfigure_mock = AsyncMock()
    agent_loop.tool_manager.refresh_remote_tools_async = refresh_mock
    agent_loop.tool_manager.reconfigure_mcp_async = reconfigure_mock
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        async for _event in session.resources.mcp.login("search"):
            pass
    finally:
        await session.close()

    # Assert
    resolve.assert_not_awaited()
    refresh_mock.assert_not_awaited()
    reconfigure_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_logout_authorization_required_does_not_restore_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: Logout withdraws an enabled OAuth source before deleting credentials.
    *Do*: Runtime convergence observes the new authorization-required revision.
    *Assert*: The source remains withdrawn without any remote rediscovery path.
    """

    # Prepare
    async def logout(
        _service: MCPAuthenticationService, name: str, *, owner: object | None = None
    ) -> str:
        return "descriptor-2"

    async def resolve(_service, reference):
        return MCPAuthorizationRequired(
            reason="missing", descriptor_revision=reference.descriptor_revision
        )

    monkeypatch.setattr(MCPAuthenticationService, "logout", logout)
    monkeypatch.setattr(MCPAuthenticationService, "resolve", resolve)
    config = build_test_vibe_config(
        mcp_servers=[
            MCPHttp(
                name="search",
                transport="streamable-http",
                url="https://mcp.example.com",
                auth=MCPOAuth(type="oauth", scopes=[]),
            )
        ]
    )
    registry = FakeMCPRegistry()
    agent_loop = build_test_agent_loop(config=config, mcp_registry=registry)
    refresh_mock = AsyncMock()
    reconfigure_mock = AsyncMock()
    agent_loop.tool_manager.refresh_remote_tools_async = refresh_mock
    agent_loop.tool_manager.reconfigure_mcp_async = reconfigure_mock
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        state = await session.resources.mcp.logout("search")
    finally:
        await session.close()

    # Assert
    source = next(item for item in state.sources if item.name == "search")
    assert source.status is MCPSourceStatus.NEEDS_AUTH
    assert registry.needs_auth == {"search"}
    refresh_mock.assert_not_awaited()
    reconfigure_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_toggle_enable_broken_server_stays_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: One configured source whose isolated discovery fails.
    *Do*: Enable the source through the catalog facade.
    *Assert*: Its public state remains unavailable without failing the request.
    """
    # Prepare
    monkeypatch.setattr(
        "chartreux.app_server.mcp_catalog.persist_mcp_toggle", AsyncMock()
    )
    server = MCPStdio(name="search", transport="stdio", command="fake-cmd")
    config = build_test_vibe_config(mcp_servers=[server])
    registry = FailingMCPRegistry(failing_server="search")
    agent_loop = build_test_agent_loop(config=config, mcp_registry=registry)
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.connect()
        # Do
        mcp_state = await session.resources.mcp.toggle("search", disabled=False)
    finally:
        await session.close()

    # Assert
    broken = next(s for s in mcp_state.sources if s.name == "search")
    assert broken.status == MCPSourceStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_catalog_legacy_registry_roundtrip_preserves_owner_binding() -> None:
    authentication = MCPAuthenticationService()
    catalog = MCPCatalogService(authentication)
    first = MCPHttp(
        name="shared",
        transport="streamable-http",
        url="https://mcp.example.test",
        auth=MCPStaticAuth(headers={"Authorization": "Bearer synthetic-a"}),
    )
    second = first.model_copy(
        update={"auth": MCPStaticAuth(headers={"Authorization": "Bearer synthetic-b"})}
    )
    first_owner = FakeConfigOrchestrator(build_test_vibe_config(mcp_servers=[first]))
    second_owner = FakeConfigOrchestrator(build_test_vibe_config(mcp_servers=[second]))
    first_catalog = await catalog.resolve_catalog(first_owner)
    second_catalog = await catalog.resolve_catalog(second_owner)
    first_ref = first_catalog.servers[0].authorization
    second_ref = second_catalog.servers[0].authorization
    assert first_ref.server_fingerprint == second_ref.server_fingerprint
    assert first_catalog.revision == second_catalog.revision
    assert first_ref.binding_id != second_ref.binding_id
    assert app_authorization_ref(registry_authorization_ref(first_ref)) == first_ref
    registry = MCPRegistry()
    other_registry = MCPRegistry()
    configure_mcp_registry_impl(registry, first_catalog, authentication)
    configure_mcp_registry_impl(other_registry, second_catalog, authentication)
    for target, server, reference in (
        (registry, first, first_ref),
        (other_registry, second, second_ref),
    ):
        result = await target._resolve_authorization(server)
        assert isinstance(result, tuple)
        snapshot, core_reference = result
        assert snapshot.headers == server.http_headers()
        assert core_reference.binding_id == reference.binding_id
    await authentication.bind_catalog([], owner=first_owner)
    stale = await registry._resolve_authorization(first)
    assert isinstance(stale, RegistryAuthorizationRequired)
    assert stale.reason == "invalid"
    surviving = await other_registry._resolve_authorization(second)
    assert isinstance(surviving, tuple)
    assert surviving[0].headers == second.http_headers()


@pytest.mark.asyncio
async def test_cached_proxy_uses_replacement_binding_with_rediscovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authentication = MCPAuthenticationService()
    catalog = MCPCatalogService(authentication)
    registry = MCPRegistry()
    discover = AsyncMock(return_value=[RemoteTool(name="read", input_schema={})])
    call = AsyncMock(return_value=MCPToolResult(server="shared", tool="read"))
    monkeypatch.setattr("chartreux.core.tools.mcp.registry.list_tools_http", discover)
    monkeypatch.setattr("chartreux.core.tools.mcp.tools.call_tool_http", call)
    retained = []
    last_server: MCPHttp | None = None
    for token in ("synthetic-a", "synthetic-b"):
        server = MCPHttp(
            name="shared",
            transport="streamable-http",
            url="https://mcp.example.test",
            auth=MCPStaticAuth(headers={"Authorization": f"Bearer {token}"}),
        )
        orchestrator = FakeConfigOrchestrator(
            build_test_vibe_config(mcp_servers=[server])
        )
        configuration = await catalog.resolve_catalog(orchestrator, owner=registry)
        configure_mcp_registry_impl(registry, configuration, authentication)
        tools = await registry.get_tools_async([server])
        runtime = getattr(tools["shared_read"], "_authorization_runtime", None)
        assert isinstance(runtime, MCPHttpAuthorizationRuntime)
        assert (
            runtime.reference.binding_id
            == configuration.servers[0].authorization.binding_id
        )
        tool = tools["shared_read"].from_config(lambda: BaseToolConfig())
        retained.append((tool, runtime))
        assert [result async for result in tool.run(_OpenArgs())] == [call.return_value]
        assert call.await_args is not None
        assert call.await_args.kwargs["headers"] == server.http_headers()
        last_server = server
    old_tool, old_runtime = retained[0]
    new_tool, new_runtime = retained[1]
    assert type(old_tool) is not type(new_tool)
    assert old_runtime.reference.binding_id != new_runtime.reference.binding_id
    revoked = await old_runtime.provider.resolve(old_runtime.reference)
    assert isinstance(revoked, RegistryAuthorizationRequired)
    assert revoked.reason == "invalid"
    with pytest.raises(ToolError, match="needs re-authentication"):
        async for _ in old_tool.run(_OpenArgs()):
            pass
    assert call.await_count == 2
    assert registry.needs_auth == set()
    assert last_server is not None
    assert (await registry.get_tools_async([last_server]))["shared_read"] is type(
        new_tool
    )
    assert [result async for result in new_tool.run(_OpenArgs())] == [call.return_value]
    assert call.await_args is not None
    assert call.await_args.kwargs["headers"] == {"Authorization": "Bearer synthetic-b"}
    assert discover.await_count == 2
