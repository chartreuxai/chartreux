from __future__ import annotations

import asyncio
from collections.abc import Coroutine
import logging
import os
import signal
from weakref import WeakKeyDictionary, WeakSet

logger = logging.getLogger(__name__)

_PROCESS_GROUPS: WeakKeyDictionary[asyncio.subprocess.Process, int] = (
    WeakKeyDictionary()
)
_REAPED_PROCESSES: WeakSet[asyncio.subprocess.Process] = WeakSet()
_CLEANUP_TIMEOUT = 0.5


def register_process_group(proc: asyncio.subprocess.Process) -> None:
    """Record ownership immediately after spawning with start_new_session=True."""
    # A new session's group ID is its leader PID, even if it has already exited.
    _PROCESS_GROUPS[proc] = proc.pid


async def spawn_registered_process(
    spawn: Coroutine[object, object, asyncio.subprocess.Process],
) -> asyncio.subprocess.Process:
    """Finish an isolated spawn and reclaim its group if the caller cancels.

    Shield only the factory and registration, not subsequent process execution.
    Cancelling the factory itself can lose the handle after the OS spawn, leaving
    asyncio to kill just the leader rather than its descendants.
    """

    async def spawn_and_register() -> asyncio.subprocess.Process:
        proc = await spawn
        register_process_group(proc)
        return proc

    spawn_task = asyncio.create_task(spawn_and_register())
    try:
        return await asyncio.shield(spawn_task)
    except asyncio.CancelledError:

        async def cleanup() -> None:
            try:
                proc = await spawn_task
            except Exception:
                # Preserve the caller's cancellation if the spawn also failed.
                logger.debug("Cancelled subprocess spawn failed", exc_info=True)
                return
            await kill_async_subprocess(proc)

        cleanup_task = asyncio.create_task(cleanup())
        # Repeated cancellation must not interrupt handle recovery or group kill.
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        cleanup_task.result()
        raise


async def kill_async_subprocess(
    proc: asyncio.subprocess.Process,
    *,
    kill_process_group: bool = True,
    kill_exited_process_group: bool = True,
) -> None:
    """Force-terminate an asyncio child, with a bounded wait for pipe closure.

    With ``kill_process_group=True`` (default), the child is expected to be
    isolated in its own process group (for example with
    ``start_new_session=True``). Register that group's identity at spawn with
    ``register_process_group`` so cleanup also works after its leader exits.
    Pass ``kill_process_group=False`` to kill only the child process; group
    isolation is not detected automatically. Pass ``kill_exited_process_group=False``
    on normal completion to preserve detached background jobs.
    """
    if proc in _REAPED_PROCESSES:
        return
    if proc.returncode is not None and (
        not kill_process_group or not kill_exited_process_group
    ):
        _PROCESS_GROUPS.pop(proc, None)
        _REAPED_PROCESSES.add(proc)
        return

    try:
        if not kill_process_group:
            proc.kill()
        else:
            group_id = _PROCESS_GROUPS.get(proc)
            if group_id is None:
                try:
                    group_id = os.getpgid(proc.pid)
                except ProcessLookupError:
                    # The caller owns an isolated group; its leader may be gone.
                    group_id = proc.pid
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except (ProcessLookupError, PermissionError, OSError):
        pass
    except Exception:
        logger.debug("Unexpected error killing pid %s", proc.pid, exc_info=True)

    # Process.wait() can still await pipe disconnection after the leader exits.
    # An escaped descendant must not make cleanup itself wait indefinitely.
    try:
        await asyncio.wait_for(proc.wait(), timeout=_CLEANUP_TIMEOUT)
    except (TimeoutError, ProcessLookupError, PermissionError, OSError):
        pass
    else:
        _PROCESS_GROUPS.pop(proc, None)
        _REAPED_PROCESSES.add(proc)
    finally:
        if proc.returncode is not None:
            # The leader can be reaped even while inherited pipes remain open.
            _PROCESS_GROUPS.pop(proc, None)
            _REAPED_PROCESSES.add(proc)
