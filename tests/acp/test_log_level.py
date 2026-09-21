from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest

from chartreux.acp.agent import ChartreuxAcpAgent
from chartreux.acp.exceptions import InvalidRequestError
from chartreux.observability.logging import (
    get_log_level_chain,
    get_session_override,
    set_config_log_level,
    set_session_override,
)


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
