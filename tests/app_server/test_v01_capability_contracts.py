from __future__ import annotations

import pytest

from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server.protocol import (
    AgentConfig,
    AppServerResponseError,
    ClientInfo,
    CompletionConfig,
    EventBatch,
    HookDefinition,
    ProtocolErrorCode,
    ToolDefinition,
)
from chartreux.app_server.server import AppServer
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import start_test_app_server


@pytest.mark.parametrize(
    "config",
    [
        AgentConfig(completion=CompletionConfig(model="unsupported")),
        AgentConfig(sandbox={}),
        AgentConfig(tools=[ToolDefinition(name="client")]),
        AgentConfig(hooks=[HookDefinition(type="pre", name="hook")]),
    ],
)
def test_non_default_unsupported_agent_config_is_rejected(config: AgentConfig) -> None:
    with pytest.raises(RequestFailure, match="not supported") as error:
        AppServer._validate_agent_config(config)

    assert error.value.code is ProtocolErrorCode.INVALID_PARAMS


@pytest.mark.parametrize(
    "config",
    [
        None,
        AgentConfig(),
        AgentConfig(completion=CompletionConfig()),
        AgentConfig(tools=[], hooks=[]),
    ],
)
def test_default_or_omitted_agent_config_is_accepted(
    config: AgentConfig | None,
) -> None:
    AppServer._validate_agent_config(config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filters",
    [
        {"sessionIds": ["session-1"]},
        {"rootSessionIds": ["root-1"]},
        {"parentSessionIds": ["parent-1"]},
        {"eventTypes": ["turn/completed"]},
    ],
)
async def test_events_read_rejects_non_default_filters(
    filters: dict[str, list[str]],
) -> None:
    agent_loop = build_test_agent_loop()
    client = start_test_app_server(agent_loop)
    try:
        await client.initialize(
            ClientInfo(name="capability-contract-test", version="1")
        )
        await client.notify("initialized")
        with pytest.raises(
            AppServerResponseError, match="filtering is not supported"
        ) as error:
            await client.request("events/read", {"filters": filters})
    finally:
        await client.close()
        await agent_loop.aclose()

    assert error.value.error.code is ProtocolErrorCode.INVALID_PARAMS


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{}, {"filters": {}}])
async def test_events_read_accepts_default_or_omitted_filters(
    params: dict[str, object],
) -> None:
    agent_loop = build_test_agent_loop()
    client = start_test_app_server(agent_loop)
    try:
        await client.initialize(
            ClientInfo(name="capability-contract-test", version="1")
        )
        await client.notify("initialized")
        result = await client.request("events/read", params)
    finally:
        await client.close()
        await agent_loop.aclose()

    assert EventBatch.model_validate(result).events == []
