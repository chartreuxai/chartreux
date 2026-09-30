from __future__ import annotations

import asyncio
import shlex
import signal
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.core.hooks.executor import HookExecutor
from chartreux.core.hooks.models import HookConfig, HookType, PostAgentInvocation
from chartreux.core.utils import async_subprocess
from chartreux.core.utils.shell import spawn_shell_command


@pytest.mark.asyncio
async def test_kill_exited_leader_uses_captured_group(monkeypatch) -> None:
    proc = Mock(pid=12345, returncode=0, wait=AsyncMock(return_value=0))
    async_subprocess.register_process_group(proc)
    getpgid = Mock(side_effect=ProcessLookupError)
    killpg = Mock()
    monkeypatch.setattr(async_subprocess.os, "getpgid", getpgid)
    monkeypatch.setattr(async_subprocess.os, "killpg", killpg)

    await async_subprocess.kill_async_subprocess(proc)

    getpgid.assert_not_called()
    killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
    proc.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_reaped_process_cannot_signal_reused_pid(monkeypatch) -> None:
    proc = Mock(pid=12345, returncode=0, wait=AsyncMock(return_value=0))
    async_subprocess.register_process_group(proc)
    killpg = Mock()
    getpgid = Mock(return_value=54321)
    monkeypatch.setattr(async_subprocess.os, "killpg", killpg)
    monkeypatch.setattr(async_subprocess.os, "getpgid", getpgid)

    await async_subprocess.kill_async_subprocess(proc)

    assert proc not in async_subprocess._PROCESS_GROUPS
    await async_subprocess.kill_async_subprocess(proc)
    getpgid.assert_not_called()
    killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
    proc.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_normal_completion_preserves_group(monkeypatch) -> None:
    proc = Mock(pid=12345, returncode=0, wait=AsyncMock(return_value=0))
    async_subprocess.register_process_group(proc)
    killpg = Mock()
    monkeypatch.setattr(async_subprocess.os, "killpg", killpg)

    await async_subprocess.kill_async_subprocess(proc, kill_exited_process_group=False)

    assert proc not in async_subprocess._PROCESS_GROUPS
    await async_subprocess.kill_async_subprocess(proc)
    killpg.assert_not_called()


@pytest.mark.asyncio
async def test_kill_missing_group_is_harmless(monkeypatch) -> None:
    proc = Mock(pid=12345, returncode=0, wait=AsyncMock(return_value=0))
    async_subprocess.register_process_group(proc)
    monkeypatch.setattr(
        async_subprocess.os, "killpg", Mock(side_effect=ProcessLookupError)
    )

    await async_subprocess.kill_async_subprocess(proc)

    proc.wait.assert_awaited_once()
    assert proc not in async_subprocess._PROCESS_GROUPS


@pytest.mark.asyncio
async def test_kill_group_falls_back_when_getpgid_fails(monkeypatch) -> None:
    proc = Mock(pid=12345, returncode=None, wait=AsyncMock(return_value=0))
    monkeypatch.setattr(
        async_subprocess.os, "getpgid", Mock(side_effect=ProcessLookupError)
    )
    killpg = Mock()
    monkeypatch.setattr(async_subprocess.os, "killpg", killpg)

    await async_subprocess.kill_async_subprocess(proc)

    killpg.assert_called_once_with(proc.pid, signal.SIGKILL)


@pytest.mark.asyncio
async def test_kill_bounds_wait_with_inherited_pipe(monkeypatch) -> None:
    cancelled = asyncio.Event()

    async def wait() -> int:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return 0

    proc = Mock(pid=12345, returncode=None, wait=wait)
    async_subprocess.register_process_group(proc)
    monkeypatch.setattr(async_subprocess.os, "killpg", Mock())
    monkeypatch.setattr(async_subprocess, "_CLEANUP_TIMEOUT", 0.01)

    await asyncio.wait_for(async_subprocess.kill_async_subprocess(proc), 0.2)

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_kill_child_only_does_not_signal_group(monkeypatch) -> None:
    proc = Mock(pid=12345, returncode=None, wait=AsyncMock(return_value=-9))
    killpg = Mock()
    monkeypatch.setattr(async_subprocess.os, "killpg", killpg)

    await async_subprocess.kill_async_subprocess(proc, kill_process_group=False)

    proc.kill.assert_called_once()
    killpg.assert_not_called()
    proc.returncode = 0
    await async_subprocess.kill_async_subprocess(proc, kill_process_group=False)
    assert proc.kill.call_count == 1


@pytest.mark.asyncio
async def test_spawn_registers_and_returns_process() -> None:
    proc = Mock(pid=12345)
    factory = AsyncMock(return_value=proc)

    assert await async_subprocess.spawn_registered_process(factory()) is proc

    factory.assert_awaited_once()
    assert async_subprocess._PROCESS_GROUPS[proc] == proc.pid
    proc.kill.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_failure_propagates() -> None:
    factory = AsyncMock(side_effect=OSError("spawn failed"))

    with pytest.raises(OSError, match="spawn failed"):
        await async_subprocess.spawn_registered_process(factory())


@pytest.mark.asyncio
async def test_cancelled_spawn_failure_preserves_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def factory() -> asyncio.subprocess.Process:
        started.set()
        await release.wait()
        raise OSError("spawn failed")

    task = asyncio.create_task(async_subprocess.spawn_registered_process(factory()))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["shell", "hook"])
@pytest.mark.parametrize("cancel_again", [False, True])
async def test_spawn_window_cancellation_kills_descendant_group(
    monkeypatch, entrypoint: str, cancel_again: bool
) -> None:
    spawned = asyncio.Event()
    release = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []
    factory_name = (
        "create_subprocess_shell" if entrypoint == "shell" else "create_subprocess_exec"
    )
    real_factory = getattr(asyncio, factory_name)

    async def delayed_factory(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_factory(*args, **kwargs)
        processes.append(proc)
        assert proc.stdout is not None
        # The short-lived background child inherits stdout. Wait for proof that
        # it was spawned, then pause before returning the process handle.
        assert await proc.stdout.readline() == b"ready\n"
        spawned.set()
        await release.wait()
        return proc

    monkeypatch.setattr(asyncio, factory_name, delayed_factory)
    killpg = Mock(wraps=async_subprocess.os.killpg)
    monkeypatch.setattr(async_subprocess.os, "killpg", killpg)
    script = "sleep 2 & echo ready; exit 0"
    task: asyncio.Task[Any]
    if entrypoint == "shell":
        task = asyncio.create_task(spawn_shell_command(script))
    else:
        hook = HookConfig(
            name="spawn-window",
            type=HookType.POST_AGENT,
            command=f"sh -c {shlex.quote(script)}",
        )
        invocation = PostAgentInvocation(
            session_id="test-session", transcript_path="", cwd="."
        )
        task = asyncio.create_task(HookExecutor().run(hook, invocation))

    try:
        await asyncio.wait_for(spawned.wait(), 1)
        proc = processes[0]
        # Exercise the harder case: the leader exits while the child holds pipes.
        async with asyncio.timeout(1):
            while proc.returncode is None:
                await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        if cancel_again:
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)

        killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
        assert proc.stdout is not None
        # EOF well before the child's two-second lifetime proves group cleanup,
        # not merely leader termination, finished before cancellation propagated.
        assert await asyncio.wait_for(proc.stdout.read(), 0.3) == b""
        assert proc in async_subprocess._REAPED_PROCESSES
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for proc in processes:
            await async_subprocess.kill_async_subprocess(proc)


@pytest.mark.asyncio
async def test_repeated_cancellation_during_group_cleanup_wait(monkeypatch) -> None:
    spawned = asyncio.Event()
    release = asyncio.Event()
    waiting = asyncio.Event()
    release_wait = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []
    real_factory = asyncio.create_subprocess_shell

    async def delayed_factory(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_factory(*args, **kwargs)
        processes.append(proc)
        assert proc.stdout is not None
        assert await proc.stdout.readline() == b"ready\n"
        real_wait = proc.wait

        async def delayed_wait() -> int:
            waiting.set()
            await release_wait.wait()
            return await real_wait()

        monkeypatch.setattr(proc, "wait", delayed_wait)
        spawned.set()
        await release.wait()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", delayed_factory)
    killpg = Mock(wraps=async_subprocess.os.killpg)
    monkeypatch.setattr(async_subprocess.os, "killpg", killpg)
    task = asyncio.create_task(spawn_shell_command("sleep 2 & echo ready; exit 0"))

    try:
        await asyncio.wait_for(spawned.wait(), 1)
        proc = processes[0]
        async with asyncio.timeout(1):
            while proc.returncode is None:
                await asyncio.sleep(0)
        task.cancel()
        release.set()
        # Pause inside Process.wait(), after cleanup has signalled the group.
        await asyncio.wait_for(waiting.wait(), 1)
        killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_wait.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)

        assert proc.stdout is not None
        assert await asyncio.wait_for(proc.stdout.read(), 0.3) == b""
        assert proc in async_subprocess._REAPED_PROCESSES
    finally:
        release.set()
        release_wait.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for proc in processes:
            await async_subprocess.kill_async_subprocess(proc)
