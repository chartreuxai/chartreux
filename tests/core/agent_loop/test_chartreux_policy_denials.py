"""Bounded deny enforcement, not source-aware policy or completed P6 guards."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import ToolExecutionResponse
import chartreux.core.events as event_module
from chartreux.core.events import BaseEvent, ToolResultEvent
from chartreux.core.llm_models import FunctionCall, Role, ToolCall
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
)
from chartreux.core.tools.builtins.bash import BashArgs, BashToolConfig
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.builtins.todo import TodoArgs
from chartreux.core.tools.permissions import PermissionContext
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    set_agent_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


async def _collect(agent: AgentLoop) -> list[BaseEvent]:
    events: list[BaseEvent] = []
    async with asyncio.timeout(5):
        async for event in agent.act("Exercise synthetic policy fixtures"):
            events.append(event)
    return events


def _call(name: str, args: dict[str, str], call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name=name, arguments=json.dumps(args)),
    )


@pytest.mark.parametrize("configured", list(ToolPermission))
@pytest.mark.parametrize("resolved", [None, *ToolPermission])
async def test_decision_denials_respect_configured_permission(
    configured: ToolPermission,
    resolved: ToolPermission | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = build_test_agent_loop(
        config=build_test_vibe_config(tools={"todo": {"permission": configured.value}})
    )
    ctx = (
        None
        if resolved is None
        else PermissionContext(permission=resolved, reason="fixture policy")
    )
    tool = agent.tool_manager.get("todo")
    resolver = MagicMock(return_value=ctx)
    monkeypatch.setattr(tool, "resolve_permission", resolver)
    try:
        decision = await agent._should_execute_tool(tool, TodoArgs(action="read"))
        effective = resolved or configured
        denied = configured == ToolPermission.NEVER or effective == ToolPermission.NEVER
        assert decision.approval_type == (
            ToolPermission.NEVER if denied else ToolPermission.ALWAYS
        )
        assert decision.verdict == (
            ToolExecutionResponse.SKIP if denied else ToolExecutionResponse.EXECUTE
        )
        assert not hasattr(agent._request_broker, "request_approval")
        if denied:
            assert decision.feedback
        resolver.assert_called_once()
    finally:
        await agent.aclose()


@pytest.mark.parametrize("denial", ["tool", "path", "scratch"])
async def test_direct_denial_no_approval_and_continued_work(
    tmp_path: Path, denial: str
) -> None:
    target = tmp_path / "synthetic.private"
    target.write_text("harmless fixture")
    overrides = {
        "permission": "never" if denial == "tool" else "always",
        "sensitive_patterns": ["*.private"],
        "denylist": [str(target)] if denial != "tool" else [],
        "allowlist": [str(target)] if denial != "tool" else [],
    }
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(target)}, "denied")]
            )
        ],
        [mock_llm_chunk(tool_calls=[_call("todo", {"action": "read"}, "continued")])],
        [mock_llm_chunk(content="Continued safely")],
    ])
    agent = build_test_agent_loop(
        config=build_test_vibe_config(
            tools={"read_file": overrides, "todo": {"permission": "always"}}
        ),
        backend=backend,
    )
    if denial == "scratch":
        agent.tool_manager.get("read_file").scratchpad_dir = tmp_path
    try:
        events = await _collect(agent)
        assert not hasattr(event_module, "ApprovalRequestEvent")
        assert not any("approval" in type(event).__name__.lower() for event in events)
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(results) == 2
        assert results[0].skipped and results[0].skip_reason
        assert results[0].result is None
        assert results[1].result is not None and not results[1].skipped
        assert agent.stats.tool_calls_rejected == 1
        assert agent.stats.tool_calls_succeeded == 1
        messages = [m for m in agent.messages if m.role == Role.tool]
        assert results[0].skip_reason in (messages[0].content or "")
        assert target.read_text() == "harmless fixture"
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "command, expected",
    [
        ("python", ToolPermission.NEVER),
        ("python script.py", ToolPermission.ALWAYS),
        ("sudo printf fixture", ToolPermission.NEVER),
        ("find . -exec printf fixture \\;", ToolPermission.NEVER),
        ("printf fixture", ToolPermission.ALWAYS),
    ],
)
async def test_posix_resolver_characterization_without_invocation(
    command: str, expected: ToolPermission
) -> None:
    agent = build_test_agent_loop(config=build_test_vibe_config())
    try:
        context = agent.tool_manager.get("bash").resolve_permission(
            BashArgs(command=command)
        )
        assert context is not None and context.permission == expected
    finally:
        await agent.aclose()


async def test_actual_command_deny_without_running_shell() -> None:
    agent = build_test_agent_loop(
        config=build_test_vibe_config(tools={"bash": {"denylist": ["printf"]}})
    )
    try:
        decision = await agent._should_execute_tool(
            agent.tool_manager.get("bash"), BashArgs(command="printf fixture")
        )
        assert decision.verdict == ToolExecutionResponse.SKIP
        assert decision.approval_type == ToolPermission.NEVER
        assert decision.feedback and "denylist" in decision.feedback
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "problem", ["unknown", "disabled", "invalid_args", "invalid_config"]
)
async def test_invalid_tools_or_config_are_reported(problem: str) -> None:
    name = "not_registered" if problem == "unknown" else "todo"
    args = {"action": "invalid" if problem == "invalid_args" else "read"}
    agent = build_test_agent_loop(
        config=build_test_vibe_config(
            disabled_tools=["todo"] if problem == "disabled" else []
        ),
        backend=FakeBackend([
            [mock_llm_chunk(tool_calls=[_call(name, args, "invalid")])],
            [mock_llm_chunk(content="Failure handled")],
        ]),
    )
    if problem == "invalid_config":
        # Cached instances must still validate the current snapshot under bypass.
        agent.tool_manager.get("todo")
        set_agent_config(
            agent,
            agent.config.model_copy(
                update={"tools": {"todo": {"permission": "invalid"}}}
            ),
        )
    try:
        events = await _collect(agent)
        assert not hasattr(event_module, "ApprovalRequestEvent")
        assert not any("approval" in type(event).__name__.lower() for event in events)
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.error and result.result is None
        assert agent.stats.tool_calls_succeeded == 0
    finally:
        await agent.aclose()


async def test_child_current_snapshot_deny_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # This tests factory construction/current snapshots, not non-weakening policy
    # across arbitrary child overrides or subsequent public profile switches.
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: FakeBackend()
    )
    parent = build_test_agent_loop(
        config=build_test_vibe_config(tools={"read_file": {"permission": "never"}})
    )
    child = await AgentRuntimeFactory().create_child(parent, "worker")
    try:
        assert child.config_orchestrator is not parent.config_orchestrator
        assert (
            child.tool_manager.get_tool_config("read_file").permission
            == ToolPermission.NEVER
        )
        for agent in (parent, child):
            decision = await agent._should_execute_tool(
                agent.tool_manager.get("read_file"),
                ReadFileArgs(file_path=str(tmp_path / "fixture.txt")),
            )
            assert decision.verdict == ToolExecutionResponse.SKIP
            assert decision.approval_type == ToolPermission.NEVER
    finally:
        await child.aclose()
        await parent.aclose()


@pytest.mark.parametrize("pattern", ["*", "npm *", "/tmp/*"])
@pytest.mark.parametrize(
    "command",
    [
        "cat {target}",
        "npm install {target}",
        "npm install file:{target}",
        "npm install file:../../outside-package",
        "pip install file://{target}",
        "uv pip install file://{target}",
        "printf x > {target}",
    ],
)
async def test_outside_shell_denial_survives_stale_allowlist_and_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pattern: str, command: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("unchanged")
    command = command.format(target=target)
    agent = build_test_agent_loop(
        cwd=project,
        config=build_test_vibe_config(tools={"bash": {"permission": "always"}}),
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", {"command": command}, "outside")]
                )
            ],
            [mock_llm_chunk(content="Denied safely")],
        ]),
    )
    # Simulate old, already-deserialized state, bypassing current validators.
    # There is no live grant store in this fork; these legacy fields are inert.
    stale = BashToolConfig(permission=ToolPermission.ALWAYS).model_copy(
        update={"allowlist": [pattern]}
    )
    monkeypatch.setattr(agent.tool_manager, "get_tool_config", lambda _: stale)
    monkeypatch.setattr(agent, "_permission_store", MagicMock(), raising=False)
    monkeypatch.setattr(agent, "bypass_tool_permissions", True, raising=False)
    spawn = AsyncMock(side_effect=AssertionError("Denied command reached executor"))
    monkeypatch.setattr("chartreux.core.tools.builtins.bash.spawn_shell_command", spawn)
    try:
        tool = agent.tool_manager.get("bash")
        decision = await agent._should_execute_tool(tool, BashArgs(command=command))
        assert decision.approval_type is ToolPermission.NEVER
        assert decision.verdict is ToolExecutionResponse.SKIP
        assert (
            decision.feedback
            and "outside the authorized workspace" in decision.feedback
        )
        events = await _collect(agent)
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert result.skipped and result.result is None
        spawn.assert_not_called()
        assert target.read_text() == "unchanged"
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "command",
    [
        "sed -i s/a/b/ input.txt",
        "touch input.txt",
        "npm install file:./package",
        "pip install file:./package",
        "uv pip install file:./package",
    ],
)
async def test_in_workspace_mutators_are_allowed_not_readers(
    command: str, tmp_path: Path
) -> None:
    agent = build_test_agent_loop(
        cwd=tmp_path,
        config=build_test_vibe_config(tools={"bash": {"permission": "always"}}),
    )
    try:
        if command.startswith("sed "):
            assert analyze_shell_command_policy(command.split()).mutates_files
        decision = await agent._should_execute_tool(
            agent.tool_manager.get("bash"), BashArgs(command=command)
        )
        assert decision.approval_type is ToolPermission.ALWAYS
        assert decision.verdict is ToolExecutionResponse.EXECUTE
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "command",
    [
        "sort --output=output.txt input.txt",
        "sed -i s/a/b/ ../outside.txt",
        "find . -exec printf fixture \\;",
        "git diff --ext-diff",
        "git diff --output=output.txt",
        "cat input.txt > ../outside.txt",
    ],
)
async def test_read_only_command_side_effect_guards_survive_loop(
    command: str, tmp_path: Path
) -> None:
    agent = build_test_agent_loop(
        cwd=tmp_path,
        config=build_test_vibe_config(tools={"bash": {"permission": "always"}}),
    )
    try:
        decision = await agent._should_execute_tool(
            agent.tool_manager.get("bash"), BashArgs(command=command)
        )
        assert decision.approval_type is ToolPermission.NEVER
        assert decision.verdict is ToolExecutionResponse.SKIP
        assert decision.feedback
    finally:
        await agent.aclose()
