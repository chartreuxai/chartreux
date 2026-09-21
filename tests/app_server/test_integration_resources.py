from __future__ import annotations

import pytest

from chartreux.app_server import _integration_resources, models, protocol, resources
from chartreux.app_server.protocol import AppServerResponseError, ProtocolErrorCode
from chartreux.core.config import MCPStdio
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import attach_test_app_server_session, start_test_app_server
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


def test_plugin_resource_declarations_are_absent() -> None:
    for module in (_integration_resources, models, protocol, resources):
        assert not any(name.startswith("Plugin") for name in vars(module))
    assert "mcp_catalog/read" in protocol.SERVER_METHODS


@pytest.mark.asyncio
async def test_removed_plugin_resource_leaves_direct_mcp_runtime_intact() -> None:
    config = build_test_vibe_config(
        mcp_servers=[
            MCPStdio(name="local", transport="stdio", command="fake-mcp", disabled=True)
        ]
    )
    client = start_test_app_server(
        build_test_agent_loop(config=config, mcp_registry=FakeMCPRegistry())
    )
    session = await attach_test_app_server_session(client)
    try:
        assert not hasattr(session.resources, "plugins")
        before = await session.resources.mcp.read()
        assert [(source.name, source.status) for source in before.sources] == [
            ("local", models.MCPSourceStatus.DISABLED)
        ]
        for method in ("plugin/info", "plugin/reload", "plugin_catalog/read"):
            with pytest.raises(AppServerResponseError) as excinfo:
                await client.request(method, {"sessionId": session.session_id})
            assert excinfo.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND
        await session.resources.runtime.refresh()
        after = await session.resources.mcp.read()
        assert after.sources == before.sources
    finally:
        await session.close()
