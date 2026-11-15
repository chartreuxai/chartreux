from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chartreux.acp.agent import ChartreuxAcpAgent
from chartreux.acp.exceptions import InvalidRequestError, NotImplementedMethodError
from chartreux.app_server.protocol import SessionOptions
from chartreux.app_server.transport import IN_PROCESS_TRANSPORT, STDIO_TRANSPORT
from chartreux.observability.logging import (
    get_log_level_chain,
    get_session_override,
    set_config_log_level,
    set_session_override,
)

ACP_TRANSPORTS = (IN_PROCESS_TRANSPORT, STDIO_TRANSPORT)


@pytest.fixture(autouse=True)
def reset_process_global_log_level() -> Iterator[None]:
    """Logging overrides are process-global, so isolate ACP tests."""
    set_session_override(None)
    set_config_log_level(None)
    yield
    set_session_override(None)
    set_config_log_level(None)


async def _new_session(agent: ChartreuxAcpAgent) -> str:
    return (await agent.new_session(cwd=".", mcp_servers=[])).session_id


@pytest.mark.asyncio
async def test_log_level_read_reports_the_complete_precedence_chain(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    set_config_log_level("WARNING")
    set_session_override("DEBUG")

    response = await acp_agent_loop.ext_method("logLevel/read", {})

    assert response == {
        "session": "DEBUG",
        "env": "INFO",
        "config": "WARNING",
        "effective": "DEBUG",
    }


@pytest.mark.asyncio
async def test_log_level_write_preserves_omitted_fields_and_clears_explicit_null(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _new_session(acp_agent_loop)
    update = AsyncMock()
    monkeypatch.setattr(
        acp_agent_loop.sessions[session_id].app_server.resources.config,
        "update",
        update,
    )

    written = await acp_agent_loop.ext_method(
        "logLevel/write", {"sessionId": session_id, "sessionOverride": "debug"}
    )
    assert written["session"] == "DEBUG"
    assert written["effective"] == "DEBUG"
    update.assert_not_awaited()

    written = await acp_agent_loop.ext_method(
        "logLevel/write", {"sessionId": session_id, "configLevel": "info"}
    )
    update.assert_awaited_once_with({"log_level": "INFO"}, reload_runtime=True)
    assert written["session"] == "DEBUG"
    assert written["config"] == "INFO"
    assert written["effective"] == "DEBUG"

    cleared = await acp_agent_loop.ext_method(
        "logLevel/write", {"sessionId": session_id, "sessionOverride": None}
    )
    assert cleared["session"] is None
    assert cleared["effective"] == "INFO"


@pytest.mark.asyncio
async def test_log_level_write_validates_all_requested_levels_before_mutating(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _new_session(acp_agent_loop)
    set_session_override("WARNING")
    update = AsyncMock()
    monkeypatch.setattr(
        acp_agent_loop.sessions[session_id].app_server.resources.config,
        "update",
        update,
    )

    with pytest.raises(InvalidRequestError, match="Invalid log level"):
        await acp_agent_loop.ext_method(
            "logLevel/write",
            {
                "sessionId": session_id,
                "sessionOverride": "DEBUG",
                "configLevel": "invalid",
            },
        )

    assert get_log_level_chain().session == "WARNING"
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_log_level_write_restores_override_when_config_update_fails(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _new_session(acp_agent_loop)
    set_session_override("WARNING")

    async def fail_update(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("configuration update failed")

    monkeypatch.setattr(
        acp_agent_loop.sessions[session_id].app_server.resources.config,
        "update",
        fail_update,
    )

    with pytest.raises(RuntimeError, match="configuration update failed"):
        await acp_agent_loop.ext_method(
            "logLevel/write",
            {
                "sessionId": session_id,
                "sessionOverride": "DEBUG",
                "configLevel": "INFO",
            },
        )

    assert get_session_override() == "WARNING"


# -- ACP mapping pins ---------------------------------------------------------
#
# logLevel/* is an ACP-only surface: a read is answered from this process's own
# level chain and a session override never crosses the wire, while the config
# level is the config/write the ACP adapter already routes. These pins record
# the requests that actually cross instead of trusting the adapter's shape, and
# the ids in ``LOG_LEVEL_MAPPING_TEST_IDS`` are the ones the capability manifest
# (C5a) cites for the ACP logLevel classifications -- each node asserts it still
# runs under a listed id, so a rename or a move drifts the manifest loudly.

LOG_LEVEL_APP_SERVER_METHODS = {"logLevel/read": None, "logLevel/write": "config/write"}
LOG_LEVEL_MAPPING_TEST_IDS = {
    method: (
        f"tests/acp/test_log_level.py::test_log_level_mapping_pins"
        f"[{method}-{IN_PROCESS_TRANSPORT}]",
        f"tests/acp/test_log_level.py::test_log_level_mapping_pins"
        f"[{method}-{STDIO_TRANSPORT}]",
    )
    for method in LOG_LEVEL_APP_SERVER_METHODS
}


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
@pytest.mark.parametrize("method", sorted(LOG_LEVEL_APP_SERVER_METHODS))
async def test_log_level_mapping_pins(
    transport: str,
    method: str,
    request: pytest.FixtureRequest,
    acp_sessions,
    acp_request_recorder,
) -> None:
    """Each ACP logLevel call reaches exactly the app-server method it maps to."""
    assert request.node.nodeid in LOG_LEVEL_MAPPING_TEST_IDS[method]

    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd()))
    )
    session_id = await _new_session(harness.agent)
    # The config leg is asserted but never persisted: the pin answers the write
    # with a typed protocol error at the client, so no configuration changes.
    recorded = acp_request_recorder(harness.server, blocked={"config/write"})

    if method == "logLevel/read":
        response = await harness.agent.ext_method("logLevel/read", {})
        assert set(response) == {"session", "env", "config", "effective"}
        assert recorded.methods == []
    else:
        with pytest.raises(InvalidRequestError):
            await harness.agent.ext_method(
                "logLevel/write", {"sessionId": session_id, "configLevel": "info"}
            )
        assert recorded.issued("config/write") == 1
        assert get_session_override() is None

    with pytest.raises(NotImplementedMethodError):
        await harness.agent.ext_method("logLevel/unknown", {"sessionId": session_id})
