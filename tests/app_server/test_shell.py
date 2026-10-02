from __future__ import annotations

import asyncio
from pathlib import Path
import time
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server import _shell
from chartreux.app_server._projection import project_history
from chartreux.app_server._session_resources import ShellTimelineEvent
from chartreux.app_server._shell_requests import _manual_shell_context
from chartreux.app_server.events import HistoryEntryAdded, HistoryEntryUpdated
from chartreux.app_server.models import (
    CompletedEffectState,
    FailedEffectState,
    PublicEffectEntry,
)
from chartreux.app_server.protocol import ShellRunParams, ShellRunResponse
from chartreux.core.agent_loop._loop import ToolExecutionResponse
from chartreux.core.llm_models import Role
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins.bash import BashArgs
from chartreux.utils.tool_presentation import ToolEffectKind
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import create_test_app_server_session


def _final_effect(events: list[ShellTimelineEvent]) -> PublicEffectEntry:
    entries = [event.entry for event in events]
    assert entries, "no timeline events were emitted"
    final = entries[-1]
    assert isinstance(final, PublicEffectEntry)
    return final


def test_manual_shell_context_caps_stdout_and_stderr_independently() -> None:
    result = ShellRunResponse(
        operation_id="shell-1",
        command="build",
        cwd="/workspace",
        stdout="o" * 10,
        stderr="e" * 10,
        exit_code=1,
    )

    context = _manual_shell_context(result, max_output_bytes=5)

    assert context.count("[truncated]") == 2
    assert "oooooo" not in context
    assert "eeeeee" not in context


@pytest.mark.asyncio
async def test_wp2_manual_shell_bypasses_model_outside_operand_denial(
    tmp_path: Path,
) -> None:
    """Characterize WP2 GAP: manual shell is not the guarded model-tool path."""
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("unchanged")
    agent_loop = build_test_agent_loop(cwd=project)
    command = f"printf changed > {target}"
    decision = await agent_loop._should_execute_tool(
        agent_loop.tool_manager.get("bash"), BashArgs(command=command)
    )
    assert decision.verdict is ToolExecutionResponse.SKIP
    assert decision.approval_type is ToolPermission.NEVER
    session = await create_test_app_server_session(agent_loop)
    try:
        events = [event async for event in session.resources.shell.run(command)]
        assert isinstance(_final_effect(events).state, CompletedEffectState)
        assert target.read_text() == "changed"
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_shell_is_one_public_effect_and_injects_model_context() -> None:
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)

    try:
        events = [
            event
            async for event in session.resources.shell.run(
                "printf 'hello'; printf 'warning' >&2"
            )
        ]
    finally:
        await session.close()

    added = next(event for event in events if isinstance(event, HistoryEntryAdded))
    assert isinstance(added.entry, PublicEffectEntry)
    assert added.entry.detail.kind is ToolEffectKind.SHELL
    updates = [event for event in events if isinstance(event, HistoryEntryUpdated)]
    assert updates
    assert all(update.entry.id == added.entry.id for update in updates)
    final = updates[-1].entry
    assert isinstance(final, PublicEffectEntry)
    assert isinstance(final.state, CompletedEffectState)
    assert final.state.output == {
        "stdout": "hello",
        "stderr": "warning",
        "output": final.state.output_text,
    }
    assert any(entry.id == final.id for entry in session.history)

    injected = agent_loop.messages[-1]
    assert injected.role is Role.user
    assert injected.injected is True
    assert injected.content is not None
    assert "Manual `!` command result from the user." in injected.content
    assert "Command: `printf 'hello'; printf 'warning' >&2`" in injected.content
    assert "Stdout:\n```text\nhello\n```" in injected.content
    assert "Stderr:\n```text\nwarning\n```" in injected.content

    restored = next(
        entry
        for entry in project_history(agent_loop)
        if isinstance(entry, PublicEffectEntry) and entry.id == final.id
    )
    assert restored.detail == final.detail
    assert restored.state == final.state


@pytest.mark.asyncio
async def test_interleaved_stderr_is_recorded_in_arrival_order() -> None:
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)

    try:
        events = [
            event
            async for event in session.resources.shell.run(
                "printf 'E' >&2; sleep 0.1; printf 'o'"
            )
        ]
    finally:
        await session.close()

    final = _final_effect(events)
    assert isinstance(final.state, CompletedEffectState)
    # Pipe order would have reported "oE"; the split is kept for the model context.
    assert final.state.output == {"stdout": "o", "stderr": "E", "output": "Eo"}

    restored = next(
        entry
        for entry in project_history(agent_loop)
        if isinstance(entry, PublicEffectEntry) and entry.id == final.id
    )
    assert restored.state == final.state


@pytest.mark.asyncio
async def test_shell_timeout_terminates_process() -> None:
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)
    started_at = time.monotonic()

    try:
        events = [
            event
            async for event in session.resources.shell.run(
                "sleep 10", timeout_seconds=0.01
            )
        ]
    finally:
        await session.close()

    final = _final_effect(events)
    assert isinstance(final.state, FailedEffectState)
    assert "timed out" in final.state.error.message
    assert time.monotonic() - started_at < 2


@pytest.mark.asyncio
async def test_closing_shell_stream_interrupts_process_and_allows_next_command() -> (
    None
):
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)
    stream = session.resources.shell.run("printf 'started'; sleep 10")

    try:
        events: list[ShellTimelineEvent] = []
        while not any(isinstance(event, HistoryEntryUpdated) for event in events):
            events.append(await anext(stream))
        await stream.aclose()

        events = [
            event async for event in session.resources.shell.run("printf 'finished'")
        ]
    finally:
        await stream.aclose()
        await session.close()

    final = _final_effect(events)
    assert isinstance(final.state, CompletedEffectState)
    assert final.state.output == {
        "stdout": "finished",
        "stderr": "",
        "output": "finished",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["timeout", "interrupt"])
async def test_shell_bounds_readers_even_when_group_cleanup_cannot_close_pipe(
    action, tmp_path, monkeypatch
) -> None:
    controller = _shell.ShellController(tmp_path)
    process = Mock(
        pid=12345,
        returncode=0,
        stdout=asyncio.StreamReader(),
        stderr=asyncio.StreamReader(),
        wait=AsyncMock(return_value=0),
    )
    monkeypatch.setattr(_shell, "spawn_shell_command", AsyncMock(return_value=process))
    monkeypatch.setattr(_shell, "kill_async_subprocess", AsyncMock())
    readers = []
    original_read = controller._read_stream

    async def read(*args):
        readers.append(asyncio.current_task())
        await original_read(*args)

    monkeypatch.setattr(controller, "_read_stream", read)
    task = asyncio.create_task(
        controller.run(
            ShellRunParams(
                session_id="test",
                operation_id="open-pipe",
                command="unused",
                timeout_seconds=0.05 if action == "timeout" else 2,
            )
        )
    )
    try:
        async with asyncio.timeout(1):
            while len(readers) < 2:
                await asyncio.sleep(0)
        if action == "interrupt":
            assert await controller.interrupt("open-pipe")
        result = await asyncio.wait_for(task, 0.5)
        assert result.timed_out is (action == "timeout")
        assert result.interrupted is (action == "interrupt")
        assert all(reader.done() for reader in readers)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["timeout", "interrupt", "cancel"])
async def test_exited_leader_cleanup_terminates_descendant_and_readers(
    action, tmp_path, monkeypatch
) -> None:
    controller = _shell.ShellController(tmp_path)
    processes = []
    readers = []
    output = []
    original_spawn = _shell.spawn_shell_command
    original_read = controller._read_stream

    async def spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        processes.append(process)
        # Deliberately return after the leader exits: Process.wait() now
        # returns immediately, whereas its inherited pipes are still open.
        async with asyncio.timeout(1):
            while process.returncode is None:
                await asyncio.sleep(0.005)
        return process

    async def read(*args):
        readers.append(asyncio.current_task())
        await original_read(*args)

    async def observe(text):
        output.append(text)

    monkeypatch.setattr(_shell, "spawn_shell_command", spawn)
    monkeypatch.setattr(controller, "_read_stream", read)
    task = asyncio.create_task(
        controller.run(
            ShellRunParams(
                session_id="test",
                operation_id="exited-leader",
                command="sleep 2.5 & echo $!; exit 0",
                timeout_seconds=0.2 if action == "timeout" else 2,
            ),
            observe,
        )
    )
    try:
        async with asyncio.timeout(1):
            while not processes or processes[0].returncode is None or not output:
                await asyncio.sleep(0.005)
        assert processes[0].returncode == 0
        child_pid = int("".join(output).strip())
        started = time.monotonic()
        if action == "interrupt":
            assert await controller.interrupt("exited-leader")
        elif action == "cancel":
            task.cancel()
        if action == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            result = await asyncio.wait_for(task, 1)
            assert result.timed_out is (action == "timeout")
            assert result.interrupted is (action == "interrupt")
        assert time.monotonic() - started < 1
        # EOF alone is insufficient: verify the descendant is no longer alive,
        # allowing an orphan zombie awaiting the host's reaper.
        status = Path(f"/proc/{child_pid}/stat")
        async with asyncio.timeout(0.5):
            while status.exists():
                try:
                    if status.read_text().split(")", 1)[1].split()[0] == "Z":
                        break
                except FileNotFoundError:
                    break
                await asyncio.sleep(0.005)
        assert all(reader.done() for reader in readers)
        assert not controller._processes
        assert not controller._operations
        followup = await asyncio.wait_for(
            controller.run(
                ShellRunParams(
                    session_id="test", operation_id="next", command="printf finished"
                )
            ),
            1,
        )
        assert followup.stdout == "finished"
        assert followup.exit_code == 0
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await controller.close()
