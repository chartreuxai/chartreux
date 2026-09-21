from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

from acp.schema import (
    AgentMessageChunk,
    AvailableCommandsUpdate,
    TextContentBlock,
    ToolCallProgress,
)
import pytest

from chartreux.acp.agent import ChartreuxAcpAgent
from chartreux.acp.exceptions import COMPACTION_FAILED, CompactionError
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolError,
    ProtocolErrorCode,
)
from chartreux.utils.paths import get_chartreux_home
from chartreux.utils.retry_prompt import build_retry_prompt
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_client import FakeClient


def _texts(agent: ChartreuxAcpAgent) -> list[str]:
    client = agent.client
    assert isinstance(client, FakeClient)
    return [
        update.update.content.text
        for update in client._session_updates
        if isinstance(update.update, AgentMessageChunk)
    ]


@pytest.mark.asyncio
async def test_help_command_uses_the_acp_adapter_without_starting_a_turn(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])

    response = await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="/help")],
    )

    assert response.stop_reason == "end_turn"
    assert any(
        "/compact" in text and "/reload" in text for text in _texts(acp_agent_loop)
    )


@pytest.mark.asyncio
async def test_compact_command_maps_app_server_failure(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="Hello")],
    )
    session = acp_agent_loop.sessions[created.session_id]

    async def fail_compact(_instructions: str = "") -> str:
        raise AppServerResponseError(
            ProtocolError(
                code=ProtocolErrorCode.COMPACTION_FAILED,
                message="Compaction failed",
                data={"reason": "tool_call"},
            )
        )

    monkeypatch.setattr(session.app_server, "compact", fail_compact)

    with pytest.raises(CompactionError) as exc_info:
        await acp_agent_loop.prompt(
            session_id=created.session_id,
            prompt=[TextContentBlock(type="text", text="/compact")],
        )

    assert exc_info.value.code == COMPACTION_FAILED
    assert exc_info.value.data == {"reason": "tool_call"}
    client = acp_agent_loop.client
    assert isinstance(client, FakeClient)
    failed = [
        notification.update
        for notification in client._session_updates
        if isinstance(notification.update, ToolCallProgress)
    ][-1]
    assert failed.status == "failed"
    assert failed.title == "Compaction failed"
    assert failed.raw_output == "Compaction failed"


@pytest.mark.asyncio
async def test_reload_failure_is_reported_as_a_command_reply(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    monkeypatch.setattr(
        acp_agent_loop.sessions[created.session_id].app_server.resources.config,
        "reload",
        AsyncMock(side_effect=RuntimeError("invalid config")),
    )

    response = await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="/reload")],
    )

    assert response.stop_reason == "end_turn"
    assert _texts(acp_agent_loop)[-1] == "Failed to reload config: invalid config"


@pytest.mark.asyncio
async def test_mcp_status_rejects_extra_arguments_like_main(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])

    response = await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="/mcp status extra")],
    )

    assert response.stop_reason == "end_turn"
    assert _texts(acp_agent_loop)[-1] == "Usage: `/mcp status`"


@pytest.mark.asyncio
async def test_proxy_setup_normalizes_key_and_matches_main_success_message(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    config = acp_agent_loop.sessions[created.session_id].app_server.resources.config
    update_proxy = AsyncMock()
    monkeypatch.setattr(config, "update_proxy", update_proxy)

    response = await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[
            TextContentBlock(
                type="text", text="/proxy-setup https_proxy https://proxy.test"
            )
        ],
    )

    assert response.stop_reason == "end_turn"
    update_proxy.assert_awaited_once_with({"HTTPS_PROXY": "https://proxy.test"})
    assert _texts(acp_agent_loop)[-1] == (
        f"Set `HTTPS_PROXY=https://proxy.test` in {get_chartreux_home() / '.env'}\n\n"
        "Please start a new chat for changes to take effect."
    )


@pytest.mark.asyncio
async def test_retry_command_is_advertised_to_clients(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    client = acp_agent_loop.client
    assert isinstance(client, FakeClient)

    await acp_agent_loop._command_controller.send_commands(
        acp_agent_loop.sessions[created.session_id]
    )

    advertised = [
        command.name
        for update in client._session_updates
        if isinstance(update.update, AvailableCommandsUpdate)
        for command in update.update.available_commands
    ]

    assert "retry" in advertised


@pytest.mark.asyncio
async def test_retry_command_continues_the_last_response_as_an_injected_turn(
    acp_agent_loop: ChartreuxAcpAgent, backend: FakeBackend
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="Hello")],
    )

    response = await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="/retry stay brief")],
    )

    assert response.stop_reason == "end_turn"
    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content == build_retry_prompt("stay brief")


@pytest.mark.asyncio
async def test_retry_instructions_never_attach_workspace_files(
    acp_agent_loop: ChartreuxAcpAgent, backend: FakeBackend, tmp_working_directory: Path
) -> None:
    (tmp_working_directory / "shot.png").write_bytes(b"not really an image")
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="Hello")],
    )

    response = await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="/retry compare @shot.png closely")],
    )

    assert response.stop_reason == "end_turn"
    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.content == build_retry_prompt("compare @shot.png closely")
    assert retry_message.images is None
    assert retry_message.resources is None


@pytest.mark.asyncio
async def test_retry_command_without_history_does_not_start_a_turn(
    acp_agent_loop: ChartreuxAcpAgent, backend: FakeBackend
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])

    response = await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="/retry")],
    )

    assert response.stop_reason == "end_turn"
    assert backend.requests_messages == []
    assert _texts(acp_agent_loop)[-1] == "No interrupted response to continue."


@pytest.mark.asyncio
async def test_command_arguments_survive_any_whitespace_separator(
    acp_agent_loop: ChartreuxAcpAgent, backend: FakeBackend
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="Hello")],
    )

    await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="/retry\tstay brief")],
    )

    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.content == build_retry_prompt("stay brief")


@pytest.mark.parametrize(
    ("prompt_text", "expected_reply"),
    [
        ("/mcp", "No MCP servers configured."),
        ("/mcp status", "No MCP servers configured."),
        ("/mcp STATUS", "No MCP servers configured."),
        ("/mcp   status", "No MCP servers configured."),
        ("/mcp\tstatus", "No MCP servers configured."),
        ("/mcp status extra", "Usage: `/mcp status`"),
        ("/mcp login", "Usage: `/mcp login <alias>`"),
        ("/mcp logout", "Usage: `/mcp logout <alias>`"),
        ("/mcp login srv", "Unknown MCP server: `srv`"),
        ("/mcp login   srv", "Unknown MCP server: `srv`"),
        ("/mcp login\tsrv", "Unknown MCP server: `srv`"),
        ("/mcp\tlogin srv", "Unknown MCP server: `srv`"),
        ("/mcp Login SRV", "Unknown MCP server: `SRV`"),
        ("  /mcp   login   srv  ", "Unknown MCP server: `srv`"),
        (
            "/mcp bogus",
            "Usage: `/mcp status`, `/mcp login <alias>`, or `/mcp logout <alias>`",
        ),
    ],
)
@pytest.mark.asyncio
async def test_mcp_command_argument_parsing(
    acp_agent_loop: ChartreuxAcpAgent, prompt_text: str, expected_reply: str
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])

    await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text=prompt_text)],
    )

    assert _texts(acp_agent_loop)[-1] == expected_reply


@pytest.mark.parametrize(
    ("prompt_text", "expected_update"),
    [
        (
            "/proxy-setup https_proxy https://proxy.test",
            {"HTTPS_PROXY": "https://proxy.test"},
        ),
        (
            "/proxy-setup HTTPS_PROXY https://proxy.test",
            {"HTTPS_PROXY": "https://proxy.test"},
        ),
        (
            "/proxy-setup https_proxy   https://proxy.test",
            {"HTTPS_PROXY": "https://proxy.test"},
        ),
        (
            "/proxy-setup https_proxy\thttps://proxy.test",
            {"HTTPS_PROXY": "https://proxy.test"},
        ),
        (
            "/proxy-setup\thttps_proxy https://proxy.test",
            {"HTTPS_PROXY": "https://proxy.test"},
        ),
        (
            "  /proxy-setup   https_proxy   https://proxy.test  ",
            {"HTTPS_PROXY": "https://proxy.test"},
        ),
        ("/proxy-setup https_proxy", {"HTTPS_PROXY": None}),
    ],
)
@pytest.mark.asyncio
async def test_proxy_setup_command_argument_parsing(
    acp_agent_loop: ChartreuxAcpAgent,
    monkeypatch: pytest.MonkeyPatch,
    prompt_text: str,
    expected_update: dict[str, str | None],
) -> None:
    created = await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    config = acp_agent_loop.sessions[created.session_id].app_server.resources.config
    update_proxy = AsyncMock()
    monkeypatch.setattr(config, "update_proxy", update_proxy)

    await acp_agent_loop.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text=prompt_text)],
    )

    update_proxy.assert_awaited_once_with(expected_update)
