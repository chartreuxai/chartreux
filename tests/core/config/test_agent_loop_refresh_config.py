from __future__ import annotations

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import MCPHttp, MCPOAuth
from chartreux.core.tools.mcp import AuthStatus
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    stub_config_reload,
)
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


@pytest.mark.asyncio
async def test_refresh_config_reconciles_mcp_registry_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = MCPHttp(name="kept", transport="streamable-http", url="http://kept:1")
    removed = MCPHttp(
        name="removed", transport="streamable-http", url="http://removed:1"
    )
    registry = FakeMCPRegistry()
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(mcp_servers=[kept, removed]),
        mcp_registry=registry,
    )
    refreshed_config = build_test_vibe_config(mcp_servers=[kept])

    stub_config_reload(monkeypatch, refreshed_config)
    await agent_loop.refresh_config()

    assert registry.status() == {"kept": AuthStatus.STATIC}


@pytest.mark.asyncio
async def test_refresh_config_does_not_guess_oauth_authorization_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oauth = MCPHttp(
        name="linear",
        transport="streamable-http",
        url="https://mcp.example.com/mcp",
        auth=MCPOAuth(type="oauth", scopes=["read"]),
    )
    registry = FakeMCPRegistry()
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(mcp_servers=[]), mcp_registry=registry
    )
    refreshed_config = build_test_vibe_config(mcp_servers=[oauth])

    stub_config_reload(monkeypatch, refreshed_config)
    await agent_loop.refresh_config()

    # Config synchronization owns active membership only. Discovery resolves the
    # server through the authorization provider and records NEEDS_AUTH when needed.
    assert registry.status() == {"linear": AuthStatus.OK}
    assert registry.needs_auth == set()


@pytest.mark.asyncio
async def test_refresh_config_creates_mcp_registry_when_first_server_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Startup with no MCP servers leaves the registry uninitialised. Adding the
    # first server via `/mcp add` calls refresh_config, which must materialise the
    # registry so the follow-up `/mcp login` can find it.
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(mcp_servers=[]),
        mcp_registry=None,
        defer_heavy_init=True,
    )
    await agent_loop.wait_until_ready()
    assert agent_loop.mcp_registry is None

    registry = FakeMCPRegistry()
    monkeypatch.setattr(
        AgentLoop, "_create_mcp_registry", staticmethod(lambda: registry)
    )
    oauth = MCPHttp(
        name="linear",
        transport="streamable-http",
        url="https://mcp.example.com/mcp",
        auth=MCPOAuth(type="oauth", scopes=["read"]),
    )
    refreshed_config = build_test_vibe_config(mcp_servers=[oauth])

    stub_config_reload(monkeypatch, refreshed_config)
    await agent_loop.refresh_config()

    assert agent_loop.mcp_registry is registry
    assert registry.status() == {"linear": AuthStatus.OK}
    assert registry.needs_auth == set()
