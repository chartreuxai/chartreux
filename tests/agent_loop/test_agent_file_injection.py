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
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins.read_file import ReadFileArgs, ReadFileResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.core.agent_loop.test_accepted_source_snapshot import make_orchestrator
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _read_permission(loop: AgentLoop, path: Path) -> ToolPermission:
    decision = loop.tool_manager.get("read_file").resolve_permission(
        ReadFileArgs(file_path=str(path))
    )
    assert decision is not None
    return decision.permission


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
        "File read is outside authorized workspace and session scratch roots "
        "and is not an instruction file injected into this agent's context. "
        "Other paths require an explicit user scope change."
    )


@pytest.mark.asyncio
async def test_injected_user_instruction_mention_and_refresh_manifest(
    tmp_path, config_dir
):
    from chartreux.core.config.harness_files import HarnessFilesManager

    cwd = tmp_path / "workspace"
    cwd.mkdir()
    doc = config_dir / "AGENTS.md"
    doc.write_text("user instruction marker")
    loop = build_test_agent_loop(
        config=build_test_vibe_config(include_project_context=True),
        cwd=cwd,
        harness_files=HarnessFilesManager(sources=("user",), cwd=cwd),
    )
    try:
        await loop.wait_until_ready()
        assert loop.tool_manager.instruction_read_files == frozenset([doc.resolve()])
        events = await loop.inject_user_context(
            f"read @{doc}", as_message=True, inject_implicit=True
        )
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert not result.skipped
        assert isinstance(result.result, ReadFileResult)
        assert "user instruction marker" in result.result.content
        doc.write_text("  \n")
        await loop.refresh_system_prompt()
        assert loop.tool_manager.instruction_read_files == frozenset()
        assert _read_permission(loop, doc) == ToolPermission.NEVER
    finally:
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_tools", [False, True])
async def test_staged_instruction_manifest_installs_only_on_publish_and_rolls_back(
    tmp_path, config_dir, replace_tools
):
    from chartreux.core.agents.launch import (
        FrozenPersona,
        LaunchAuthorityInputs,
        LaunchCandidate,
    )
    from chartreux.core.agents.models import BUILTIN_SUBAGENTS
    from chartreux.core.config.harness_files import HarnessFilesManager
    from chartreux.core.subagents import LaunchConfig

    cwd = tmp_path / "workspace"
    cwd.mkdir()
    original = tmp_path / "original.md"
    replacement = tmp_path / "replacement.md"
    original.write_text("original instructions")
    replacement.write_text("replacement instructions")
    link = config_dir / "AGENTS.md"
    link.symlink_to(original)
    loop = build_test_agent_loop(
        config=build_test_vibe_config(include_project_context=True),
        cwd=cwd,
        harness_files=HarnessFilesManager(sources=("user",), cwd=cwd),
        parent_authority_revision_getter=lambda: 0,
    )
    try:
        await loop.wait_until_ready()
        old_manager = loop.tool_manager
        old_prompt = loop.messages[0].content
        link.unlink()
        link.symlink_to(replacement)
        overrides = (
            LaunchConfig(enabled_tools=["read_file", "grep"])
            if replace_tools
            else LaunchConfig()
        )
        target = loop.config_orchestrator.copy()
        assert loop.committed_model is not None
        candidate = LaunchCandidate(
            profile=next(iter(BUILTIN_SUBAGENTS.values())),
            config_inputs={},
            orchestrator=target,
            semantic_overrides=overrides,
            persona=FrozenPersona(loop.config.system_prompt_id, None),
            effective_model=target.config.get_active_model(),
            committed_model=loop.committed_model,
            effective_thinking=target.config.get_active_model().thinking,
            authority_inputs=LaunchAuthorityInputs(frozenset()),
        )
        prepared = await loop.prepare_launch_reconfiguration(
            candidate,
            expected_session_generation=loop._session_generation,
            expected_parent_authority_revision=0,
        )
        assert loop.tool_manager is old_manager
        assert loop.tool_manager.instruction_read_files == frozenset([original])
        assert loop.messages[0].content == old_prompt
        assert "replacement instructions" in prepared.consumers.system_prompt
        assert _read_permission(loop, replacement) == ToolPermission.NEVER
        publication = loop.publish_launch_reconfiguration(prepared)
        assert loop.tool_manager.instruction_read_files == frozenset([replacement])
        assert "replacement instructions" in str(loop.messages[0].content)
        loop.rollback_launch_reconfiguration(prepared, publication)
        assert loop.tool_manager is old_manager
        assert loop.tool_manager.instruction_read_files == frozenset([original])
        assert loop.messages[0].content == old_prompt
        assert _read_permission(loop, replacement) == ToolPermission.NEVER
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_reload_instruction_manifest_refresh_and_failed_commit_preserves_previous(
    tmp_path, config_dir, monkeypatch
):
    from chartreux.core.config.harness_files import HarnessFilesManager

    cwd = tmp_path / "workspace"
    cwd.mkdir()
    doc = config_dir / "AGENTS.md"
    doc.write_text("original instructions")
    loop = build_test_agent_loop(
        config=build_test_vibe_config(include_project_context=True),
        cwd=cwd,
        harness_files=HarnessFilesManager(sources=("user",), cwd=cwd),
    )
    try:
        await loop.wait_until_ready()
        previous = loop.tool_manager
        previous_prompt = loop.messages[0].content
        target = loop.config.model_copy(update={"include_project_context": False})
        prepared = loop._prepare_reload(target, False)
        assert loop.tool_manager.instruction_read_files == frozenset([doc.resolve()])
        with monkeypatch.context() as patch:

            def fail():
                raise RuntimeError("failed publication")

            patch.setattr(loop, "install_launch_metadata", fail)
            with pytest.raises(RuntimeError, match="failed publication"):
                loop._commit_reload(prepared, reset_middleware=False)
        assert loop.tool_manager is previous
        assert loop.messages[0].content == previous_prompt
        assert loop.tool_manager.instruction_read_files == frozenset([doc.resolve()])
        loop._commit_reload(prepared, reset_middleware=False)
        assert loop.tool_manager.instruction_read_files == frozenset()
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_trusted_ancestor_read_and_dynamic_subdirectory_excluded(tmp_path):
    from chartreux.core.config.harness_files import HarnessFilesManager
    from chartreux.core.trusted_folders import TrustedFoldersManager

    cwd = tmp_path / "workspace"
    subdir = cwd / "subdir"
    subdir.mkdir(parents=True)
    ancestor = tmp_path / "AGENTS.md"
    ancestor.write_text("ancestor instructions")
    dynamic = subdir / "AGENTS.md"
    dynamic.write_text("dynamic instructions")
    target = subdir / "file.py"
    target.write_text("source")
    trust = TrustedFoldersManager()
    trust.trust_for_session(tmp_path)
    loop = build_test_agent_loop(
        config=build_test_vibe_config(include_project_context=True),
        cwd=cwd,
        harness_files=HarnessFilesManager(
            sources=("project",), cwd=cwd, trust_store=trust
        ),
    )
    try:
        await loop.wait_until_ready()
        tool = loop.tool_manager.get("read_file")
        assert _read_permission(loop, ancestor) == ToolPermission.ALWAYS
        assert loop.tool_manager.instruction_read_files == frozenset([ancestor])
        result = [item async for item in tool.run(ReadFileArgs(file_path=str(target)))][
            -1
        ]
        assert "dynamic instructions" in (tool.get_result_extra(result) or "")
        assert str(subdir.resolve()) in tool.state.injected_agents_md
        assert dynamic not in loop.tool_manager.instruction_read_files
    finally:
        await loop.aclose()
