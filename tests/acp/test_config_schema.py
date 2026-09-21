from __future__ import annotations

import pytest

from chartreux.acp.agent import ChartreuxAcpAgent


@pytest.mark.asyncio
async def test_config_schema_returns_runtime_schema(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    response = await acp_agent_loop.ext_method("config/schema", {})

    assert response["version"].startswith("sha256:")
    assert acp_agent_loop.sessions == {}
    schema = response["schema"]
    assert schema["title"] == "ChartreuxConfigSchema"
    assert {"active_model", "disabled_tools", "mcp_servers"} <= schema[
        "properties"
    ].keys()


@pytest.mark.asyncio
async def test_config_schema_preserves_mcp_transport_discriminator(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    response = await acp_agent_loop.ext_method("config/schema", {})

    discriminator = response["schema"]["properties"]["mcp_servers"]["items"][
        "discriminator"
    ]
    assert discriminator == {
        "mapping": {"stdio": "#/$defs/MCPStdio", "streamable-http": "#/$defs/MCPHttp"},
        "propertyName": "transport",
    }
