from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.events import (
    AssistantEvent,
    BaseEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from chartreux.core.llm_models import Role
from chartreux.core.tools.builtins.read_file import ReadFileArgs, ReadFileResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.core.agent_loop.test_accepted_source_snapshot import make_orchestrator
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


async def _act_and_collect(agent_loop: AgentLoop, prompt: str) -> list[BaseEvent]:
    return [ev async for ev in agent_loop.act(prompt)]


def _make_loop(turns: int = 1, *, cwd: Path | None = None) -> AgentLoop:
    config = build_test_vibe_config(enabled_tools=["read_file"])
    return build_test_agent_loop(
        config=config,
        backend=FakeBackend([
            [mock_llm_chunk(content="Done reading the file.")] for _ in range(turns)
        ]),
        cwd=cwd,
    )


def _read_file_tool_messages(agent_loop: AgentLoop) -> list[str]:
    return [
        m.content or ""
        for m in agent_loop.messages
        if m.role == Role.tool and m.name == "read_file"
    ]


@pytest.mark.asyncio
async def test_mentioning_file_injects_read_file_call_and_result(
    tmp_working_directory: Path,
) -> None:
    (tmp_working_directory / "notes.md").write_text("hello from notes")
    agent_loop = _make_loop()

    events = await _act_and_collect(agent_loop, "look at @notes.md")

    types = [type(e) for e in events]
    assert types[0] is UserMessageEvent
    assert types[-1] is AssistantEvent
    call_events = [e for e in events if isinstance(e, ToolCallEvent)]
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(call_events) == 1
    assert len(result_events) == 1
    assert call_events[0].tool_name == "read_file"
    assert result_events[0].tool_name == "read_file"
    assert call_events[0].tool_call_id == result_events[0].tool_call_id


@pytest.mark.asyncio
async def test_user_turn_keeps_literal_mention(tmp_working_directory: Path) -> None:
    (tmp_working_directory / "notes.md").write_text("hello")
    agent_loop = _make_loop()

    await _act_and_collect(agent_loop, "look at @notes.md please")

    user_msgs = [m for m in agent_loop.messages if m.role == Role.user]
    assert user_msgs[-1].content == "look at @notes.md please"


@pytest.mark.asyncio
async def test_file_content_lands_in_tool_message(tmp_working_directory: Path) -> None:
    (tmp_working_directory / "notes.md").write_text("secret marker line")
    agent_loop = _make_loop()

    await _act_and_collect(agent_loop, "read @notes.md")

    assistant_with_call = next(
        m for m in agent_loop.messages if m.role == Role.assistant and m.tool_calls
    )
    tool_call = (
        assistant_with_call.tool_calls[0] if assistant_with_call.tool_calls else None
    )
    assert tool_call is not None
    assert tool_call.function.name == "read_file"

    tool_msg = next(m for m in agent_loop.messages if m.role == Role.tool)
    assert tool_msg.tool_call_id == tool_call.id
    assert tool_msg.name == "read_file"
    assert "secret marker line" in (tool_msg.content or "")


@pytest.mark.asyncio
async def test_file_reinjected_every_turn_without_dedup(
    tmp_working_directory: Path,
) -> None:
    file_path = tmp_working_directory / "notes.md"
    file_path.write_text("first content")
    agent_loop = _make_loop(turns=2)

    await _act_and_collect(agent_loop, "read @notes.md")
    file_path.write_text("second content")
    await _act_and_collect(agent_loop, "read @notes.md")

    tool_messages = _read_file_tool_messages(agent_loop)
    assert len(tool_messages) == 2
    assert "first content" in tool_messages[0]
    assert "second content" in tool_messages[1]


@pytest.mark.asyncio
async def test_multiple_files_inject_multiple_calls(
    tmp_working_directory: Path,
) -> None:
    (tmp_working_directory / "a.txt").write_text("alpha content")
    (tmp_working_directory / "b.txt").write_text("beta content")
    agent_loop = _make_loop()

    await _act_and_collect(agent_loop, "compare @a.txt and @b.txt")

    tool_messages = _read_file_tool_messages(agent_loop)
    assert len(tool_messages) == 2
    joined = "\n".join(tool_messages)
    assert "alpha content" in joined
    assert "beta content" in joined


@pytest.mark.asyncio
async def test_plain_prompt_does_not_inject_file(tmp_working_directory: Path) -> None:
    agent_loop = _make_loop()

    events = await _act_and_collect(agent_loop, "just a normal question")

    assert not any(isinstance(e, ToolCallEvent) for e in events)
    assert not any(m.role == Role.tool for m in agent_loop.messages)


@pytest.mark.asyncio
async def test_inject_user_context_returns_injected_file_events(
    tmp_working_directory: Path,
) -> None:
    (tmp_working_directory / "notes.md").write_text("queued file body")
    agent_loop = _make_loop()

    events = await agent_loop.inject_user_context(
        "read @notes.md", as_message=True, inject_implicit=True
    )

    assert [type(e) for e in events] == [
        UserMessageEvent,
        ToolCallEvent,
        ToolResultEvent,
    ]
    tool_msg = next(m for m in agent_loop.messages if m.role == Role.tool)
    assert tool_msg.name == "read_file"
    assert "queued file body" in (tool_msg.content or "")


async def _make_loop_with_accepted_root(
    session_cwd: Path, accepted_root: Path
) -> AgentLoop:
    settings = session_cwd.parent / "settings.toml"
    settings.write_text(
        "[authorized_roots_by_project]\n"
        f"{json.dumps(str(session_cwd))} = {json.dumps([str(accepted_root)])}\n",
        encoding="utf-8",
    )
    return AgentLoop(
        config_orchestrator=await make_orchestrator(settings),
        backend=FakeBackend([[mock_llm_chunk(content="Done reading the file.")]]),
        cwd=session_cwd,
    )


@pytest.mark.asyncio
async def test_absolute_mention_in_accepted_root_uses_read_file(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    accepted_root = tmp_path / "accepted"
    session_cwd.mkdir()
    accepted_root.mkdir()
    target = accepted_root / "accepted.txt"
    target.write_text("accepted root content", encoding="utf-8")
    agent_loop = await _make_loop_with_accepted_root(session_cwd, accepted_root)

    events = await _act_and_collect(agent_loop, f"read @{target}")

    call = next(event for event in events if isinstance(event, ToolCallEvent))
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert call.args is not None
    assert cast(ReadFileArgs, call.args).file_path == str(target.resolve())
    assert not result.skipped
    assert result.result is not None
    assert "accepted root content" in cast(ReadFileResult, result.result).content


@pytest.mark.asyncio
async def test_relative_mention_resolves_against_session_cwd(
    tmp_working_directory: Path,
) -> None:
    session_cwd = tmp_working_directory / "session"
    session_cwd.mkdir()
    (tmp_working_directory / "notes.md").write_text(
        "process cwd content", encoding="utf-8"
    )
    session_note = session_cwd / "notes.md"
    session_note.write_text("session cwd content", encoding="utf-8")
    agent_loop = _make_loop(cwd=session_cwd)

    events = await _act_and_collect(agent_loop, "read @notes.md")

    call = next(event for event in events if isinstance(event, ToolCallEvent))
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert call.args is not None
    assert cast(ReadFileArgs, call.args).file_path == str(session_note.resolve())
    assert not result.skipped
    assert result.result is not None
    content = cast(ReadFileResult, result.result).content
    assert "session cwd content" in content
    assert "process cwd content" not in content


@pytest.mark.asyncio
async def test_outside_root_mention_keeps_read_file_permission_gate(
    tmp_path: Path,
) -> None:
    session_cwd = tmp_path / "session"
    outside_root = tmp_path / "outside"
    session_cwd.mkdir()
    outside_root.mkdir()
    target = outside_root / "outside.txt"
    target.write_text("outside root content", encoding="utf-8")
    agent_loop = _make_loop(cwd=session_cwd)

    events = await _act_and_collect(agent_loop, f"read @{target}")

    assert any(isinstance(event, ToolCallEvent) for event in events)
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.skipped
    assert result.result is None
    assert result.skip_reason == (
        "File access outside authorized project and session scratch roots; "
        "only an explicit user scope change can authorize it"
    )
