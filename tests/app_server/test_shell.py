from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server import _shell
from chartreux.app_server._projection import project_history
from chartreux.app_server._session_resources import ShellTimelineEvent
from chartreux.app_server._shell_record import (
    ManualShellOutcome,
    build_manual_shell_record,
)
from chartreux.app_server.events import HistoryEntryAdded, HistoryEntryUpdated
from chartreux.app_server.models import (
    CompletedEffectState,
    FailedEffectState,
    PublicEffectEntry,
    PublicMessageEntry,
)
from chartreux.app_server.protocol import ShellRunParams
from chartreux.core.agent_loop._loop import ToolExecutionResponse
from chartreux.core.llm_models import ManualShellStatus, Role
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins.bash import BashArgs
from chartreux.core.utils import shell as shell_utils
from chartreux.core.utils.shell import shell_executor
from chartreux.utils.tool_presentation import ToolEffectKind
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import create_test_app_server_session


def _final_effect(events: list[ShellTimelineEvent]) -> PublicEffectEntry:
    entries = [event.entry for event in events]
    assert entries, "no timeline events were emitted"
    final = entries[-1]
    assert isinstance(final, PublicEffectEntry)
    return final


def test_manual_shell_record_caps_stdout_and_stderr_independently() -> None:
    record = build_manual_shell_record(
        ManualShellOutcome(
            operation_id="shell-1",
            command="build",
            cwd="/workspace",
            status="failed",
            exit_code=1,
            stdout="o" * 10,
            stderr="e" * 10,
        ),
        max_output_bytes=5,
    )

    assert record.text.count("[truncated]") == 2
    assert "oooooo" not in record.text
    assert "eeeeee" not in record.text


def test_failed_manual_shell_record_keeps_partial_output() -> None:
    record = build_manual_shell_record(
        ManualShellOutcome(
            operation_id="shell-1",
            command="build",
            cwd="/workspace",
            status="failed",
            output_text="partial line",
            error="spawn failed",
        ),
        max_output_bytes=16_000,
    )

    assert "Status: failed" in record.text
    assert "Error:\n```text\nspawn failed\n```" in record.text
    assert "Output:\n```text\npartial line\n```" in record.text


@pytest.mark.parametrize(
    ("status", "exit_code", "expected"),
    [
        ("ok", 0, "Exit code: 0"),
        ("failed", 7, "Exit code: 7"),
        ("failed", None, "Status: failed"),
        ("timed_out", 1, "Status: timed out"),
        ("interrupted", 1, "Status: interrupted by user"),
        ("refused", None, "Status: refused"),
    ],
)
def test_every_manual_shell_status_renders_the_one_record(
    status: ManualShellStatus, exit_code: int | None, expected: str
) -> None:
    record = build_manual_shell_record(
        ManualShellOutcome(
            operation_id="shell-1",
            command="build",
            cwd="/workspace",
            status=status,
            exit_code=exit_code,
            timed_out=status == "timed_out",
            interrupted=status == "interrupted",
            error="spawn refused" if status in {"failed", "refused"} else None,
        ),
        max_output_bytes=16_000,
    )

    assert record.text.startswith("Manual `!` command result from the user.")
    assert "Command: `build`" in record.text
    assert "Working directory: `/workspace`" in record.text
    assert expected in record.text
    if status in {"failed", "refused"}:
        assert "Error:\n```text\nspawn refused\n```" in record.text
    assert record.context.status == status
    assert record.context.exit_code == exit_code


def test_manual_shell_record_redacts_output_and_error_representations() -> None:
    synthetic_value = "fixture-only-redaction-value"
    policy = secret_redaction.ScrubPolicy(
        redaction_credentials=(("CHARTREUX_TEST_CREDENTIAL_KEY", synthetic_value),)
    )
    with secret_redaction.bind_policy(policy):
        succeeded = build_manual_shell_record(
            ManualShellOutcome(
                operation_id="shell-1",
                command="env",
                cwd="/workspace",
                status="ok",
                exit_code=0,
                stdout=synthetic_value,
            ),
            max_output_bytes=16_000,
        )
        failed = build_manual_shell_record(
            ManualShellOutcome(
                operation_id="shell-2",
                command="env",
                cwd="/workspace",
                status="failed",
                error=f"spawn failed with {synthetic_value}",
            ),
            max_output_bytes=16_000,
        )

    for record in (succeeded, failed):
        assert synthetic_value not in record.text
        assert synthetic_value not in record.context.model_dump_json()


@pytest.mark.asyncio
async def test_wp2_manual_shell_bypasses_model_outside_operand_denial(
    tmp_path: Path,
) -> None:
    """Characterize WP2 GAP: manual shell is not the guarded model-tool path.

    F1pkg1(a) elected the design's REFUSAL lane, which is glob-bearing only:
    this command carries no glob metacharacter, so the characterization below
    stays exactly as the gap describes it.
    """
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
async def test_glob_bearing_manual_shell_request_executes(tmp_path: Path) -> None:
    """A manual `!` command is the user's own typing.

    The terminal-transport expansion refusal belongs to the model-tool path;
    on the manual path the command runs as typed and the shell expands it.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / "kept.log").write_text("kept")
    agent_loop = build_test_agent_loop(cwd=project)
    session = await create_test_app_server_session(agent_loop)
    try:
        events = [event async for event in session.resources.shell.run("ls *.log")]
    finally:
        await session.close()
        await agent_loop.aclose()
    state = _final_effect(events).state
    assert isinstance(state, CompletedEffectState)
    assert isinstance(state.output, dict)
    assert state.output["stdout"] == "kept.log\n"


@pytest.mark.asyncio
async def test_unknown_executor_is_surfaced_as_a_failed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1pkg1(c): the spawn-side executor contract still fails closed.

    An unknown ambient executor hard-denies at the spawn — no
    ``CompletedEffectState``, no side effect — and B.1 turns that spawn
    failure into the one record, so the agent sees it rather than a lost
    request error.
    """
    project = tmp_path / "project"
    project.mkdir()
    marker = project / "spawned.log"
    agent_loop = build_test_agent_loop(cwd=project)
    session = await create_test_app_server_session(agent_loop)
    # The contract is consumed at the spawn, so the executor is read when the
    # request runs, not at session setup. The ambient chain resolves from
    # $SHELL only when the host has no bash, so pin that here.
    monkeypatch.setattr(shell_utils, "_locate_bash", lambda: None)
    shell_utils.reset_executor_selection()
    monkeypatch.setenv("SHELL", "/nonexistent/chartreux-unknown-executor")
    events: list[ShellTimelineEvent] = []
    try:
        events = [
            event async for event in session.resources.shell.run(f"touch {marker}")
        ]
    finally:
        await session.close()
        await agent_loop.aclose()
    final = _final_effect(events)
    assert isinstance(final.state, FailedEffectState)
    assert not marker.exists()
    injected = agent_loop.messages[-1]
    assert injected.role is Role.user
    assert injected.content is not None
    assert "Status: failed" in injected.content
    assert "no pinned executor option contract" in injected.content


@pytest.mark.asyncio
async def test_shell_text_executor_identity_is_the_spawned_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Executor identification reads the binary the spawn actually runs."""
    spawned: list[object] = []

    async def create_subprocess_shell(*args: Any, **kwargs: Any) -> Any:
        del args
        spawned.append(kwargs["executable"])
        return object()

    async def spawn_registered_process(spawn: Any) -> Any:
        await spawn
        return object()

    monkeypatch.setattr(
        shell_utils.asyncio, "create_subprocess_shell", create_subprocess_shell
    )
    monkeypatch.setattr(
        shell_utils, "spawn_registered_process", spawn_registered_process
    )

    monkeypatch.setattr(shell_utils, "_locate_bash", lambda: None)
    shell_utils.reset_executor_selection()
    monkeypatch.setenv("SHELL", "/bin/sh")
    assert shell_executor() == "/bin/sh"
    await shell_utils.spawn_shell_command("printf hello", cwd=tmp_path)
    assert spawned == ["/bin/sh"]

    # The process-level resolution (config value, else the shipped default)
    # outranks the ambient $SHELL and is the binary the spawn receives.
    shell_utils.apply_executor_selection("/usr/bin/zsh")
    assert shell_executor() == "/usr/bin/zsh"
    await shell_utils.spawn_shell_command("printf hello", cwd=tmp_path)
    assert spawned == ["/bin/sh", "/usr/bin/zsh"]

    monkeypatch.delenv("SHELL", raising=False)
    shell_utils.reset_executor_selection()
    assert shell_executor() is None
    await shell_utils.spawn_shell_command("printf hello", cwd=tmp_path)
    assert spawned == ["/bin/sh", "/usr/bin/zsh", None]


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

    record = next(
        message for message in agent_loop.messages if message.role is Role.user
    )
    assert record.injected is False
    assert record.manual_shell is not None
    assert record.content is not None
    assert "Manual `!` command result from the user." in record.content
    assert "Command: `printf 'hello'; printf 'warning' >&2`" in record.content
    assert "Stdout:\n```text\nhello\n```" in record.content
    assert "Stderr:\n```text\nwarning\n```" in record.content

    restored = next(
        entry
        for entry in project_history(agent_loop)
        if isinstance(entry, PublicEffectEntry) and entry.id == final.id
    )
    assert restored.detail == final.detail
    assert restored.state == final.state

    # Item-5 B.2: rebuild projects the record on the normal user-message branch.
    assert any(
        isinstance(entry, PublicMessageEntry)
        and entry.role == "user"
        and entry.id == record.message_id
        for entry in project_history(agent_loop)
    )


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
    injected = agent_loop.messages[-1]
    assert injected.content is not None
    assert "Status: timed out" in injected.content


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
                except (FileNotFoundError, ProcessLookupError):
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


# --- A.2: the manual path carries no remote-exec grammar ---------------------
#
# A manual `!` command is the user's own typing, so `! ssh host cmd` reaches
# the spawn exactly as typed. The model-tool path keeps its destination
# grants, pinned argv, and refusals (test_bash_execution.py, test_remote_exec_
# grants.py); only the manual path is ungated.


class _ManualShellStream:
    async def read(self, n: int = -1) -> bytes:
        return b""


class _ManualShellProc:
    def __init__(self) -> None:
        self.stdout = _ManualShellStream()
        self.stderr = _ManualShellStream()
        self.returncode = 0

    async def wait(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_empty_manual_command_is_the_refused_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 5 B.1: the empty-input no-op is the one refusal and still a record."""
    project = tmp_path / "project"
    project.mkdir()
    agent_loop = build_test_agent_loop(cwd=project)
    spawned: list[str] = []

    async def _record_spawn(command: str, **kwargs: Any) -> Any:
        spawned.append(command)
        return _ManualShellProc()

    monkeypatch.setattr(_shell, "spawn_shell_command", _record_spawn)
    session = await create_test_app_server_session(agent_loop)
    try:
        events = [event async for event in session.resources.shell.run("   ")]
        async with asyncio.timeout(5):
            while not any(
                message.role is Role.user and message.manual_shell is not None
                for message in agent_loop.messages
            ):
                await asyncio.sleep(0.01)
    finally:
        await session.close()
        await agent_loop.aclose()

    assert spawned == []
    final = _final_effect(events)
    assert isinstance(final.state, FailedEffectState)
    assert final.state.error.message == "Command refused"
    record = next(
        message for message in agent_loop.messages if message.role is Role.user
    )
    assert record.manual_shell is not None
    assert record.manual_shell.status == "refused"
    assert record.manual_shell.exit_code is None
    assert record.content is not None
    assert record.content.startswith("Manual `!` command result from the user.")
    assert "Status: refused" in record.content
    assert "Shell command cannot be empty" in record.content


@pytest.mark.asyncio
async def test_manual_shell_runs_a_remote_command_as_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spawn observes the user's text, not a permission-derived plan."""
    project = tmp_path / "project"
    project.mkdir()
    agent_loop = build_test_agent_loop(cwd=project)
    spawned: list[str] = []

    async def _record_spawn(command: str, **kwargs: Any) -> Any:
        spawned.append(command)
        return _ManualShellProc()

    monkeypatch.setattr(_shell, "spawn_shell_command", _record_spawn)
    session = await create_test_app_server_session(agent_loop)
    try:
        events = [
            event
            async for event in session.resources.shell.run(
                "ssh deploy@example.com uptime"
            )
        ]
        assert isinstance(_final_effect(events).state, CompletedEffectState)
    finally:
        await session.close()
        await agent_loop.aclose()
    assert spawned == ["ssh deploy@example.com uptime"]
