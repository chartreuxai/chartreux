from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
import logging
import os
import signal
import threading
import time
from weakref import WeakKeyDictionary, WeakSet

logger = logging.getLogger(__name__)

_PROCESS_GROUPS: WeakKeyDictionary[asyncio.subprocess.Process, int] = (
    WeakKeyDictionary()
)
_REAPED_PROCESSES: WeakSet[asyncio.subprocess.Process] = WeakSet()
_CLEANUP_TIMEOUT = 0.5


def _process_identity(pid: int) -> tuple[int, int, int, str] | None:
    """Linux start-time, process-group, session, state (not a PID-only identity)."""
    try:
        with open(f"/proc/{pid}/stat") as stream:
            fields = stream.read().rsplit(")", 1)[1].split()
        return int(fields[19]), int(fields[2]), int(fields[3]), fields[0]
    except (FileNotFoundError, ProcessLookupError):
        return None


class ProcessGroupIdentityChanged(RuntimeError):
    """Ownership continuity was lost; no historical group may be signaled."""


@dataclass
class ProcessGroupOwnership:
    pgid: int
    leader: tuple[int, int, int, str] | None
    valid: bool = True
    witnessed: dict[int, tuple[int, int, int]] = field(default_factory=dict, repr=False)
    spawn_started: int | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def _identity_changed(self) -> None:
        self.valid = False
        raise ProcessGroupIdentityChanged("Process group identity changed")

    def members(self) -> bool:
        with self._lock:
            return self._members()

    def _members(self) -> bool:
        if not self.valid:
            return False
        current = _process_identity(self.pgid)
        if current is not None:
            if (
                self.leader is None
                or current[:3] != self.leader[:3]
                or current[1:3] != (self.pgid, self.pgid)
            ):
                self._identity_changed()
            if current[3] != "Z" and self.spawn_started is not None:
                # The live leader pins PGID/SID without a full /proc scan.
                self.witnessed[self.pgid] = current[:3]
                return True
        earliest = self.spawn_started
        if earliest is None and self.leader is not None:
            earliest = self.leader[0]
        members: dict[int, tuple[int, int, int, str]] = {}
        # /proc is not an atomic snapshot: retry empty scans before settlement,
        # and retain previous witnesses across them.
        for _ in range(3):
            for name in os.listdir("/proc"):
                if not name.isdecimal():
                    continue
                pid = int(name)
                identity = _process_identity(pid)
                if identity is None or identity[1:3] != (self.pgid, self.pgid):
                    continue
                previous = self.witnessed.get(pid)
                if (earliest is not None and identity[0] < earliest) or (
                    previous is not None and previous != identity[:3]
                ):
                    self._identity_changed()
                if pid == self.pgid and (
                    self.leader is None or identity[:3] != self.leader[:3]
                ):
                    self._identity_changed()
                members[pid] = identity
            if members:
                break
        if not members:
            # Only transient empty snapshots may be retried. Once all retries
            # are empty, PGID reuse must never restore this owner's authority.
            self.valid = False
            return False
        if not any(
            self.witnessed.get(pid) == identity[:3]
            or (
                earliest is not None
                and earliest <= identity[0]
                and (self.spawn_started is not None or identity[0] == earliest)
            )
            for pid, identity in members.items()
        ):
            # Registered groups may fork at any age while membership remains
            # continuous. Owners without a spawn boundary require a witness.
            self._identity_changed()
        self.witnessed.update({pid: identity[:3] for pid, identity in members.items()})
        if any(identity[3] != "Z" for identity in members.values()):
            return True
        self.valid = False
        return False

    def signal(self, sig: signal.Signals) -> None:
        with self._lock:
            if self.members():
                try:
                    os.killpg(self.pgid, sig)
                except ProcessLookupError:
                    self.valid = False


_GROUP_OWNERS: WeakKeyDictionary[asyncio.subprocess.Process, ProcessGroupOwnership] = (
    WeakKeyDictionary()
)


def registered_process_group(proc: asyncio.subprocess.Process) -> ProcessGroupOwnership:
    """No PID/PGID fallback: only a still-owned registered spawn may signal."""
    owner = _GROUP_OWNERS.get(proc)
    if owner is None or not owner.valid or proc in _REAPED_PROCESSES:
        raise RuntimeError("Process group is not owned")
    return owner


def release_process_group(proc: asyncio.subprocess.Process) -> None:
    owner = _GROUP_OWNERS.pop(proc, None)
    if owner is not None:
        owner.valid = False
    _PROCESS_GROUPS.pop(proc, None)
    _REAPED_PROCESSES.add(proc)


async def terminate_process_group(
    owner: ProcessGroupOwnership, *, grace: float = 2.0, kill_timeout: float = 0.5
) -> None:
    """Settle an owned group without blocking the loop on leaderless /proc scans."""
    await asyncio.to_thread(owner.signal, signal.SIGTERM)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace
    while await asyncio.to_thread(owner.members):
        if loop.time() >= deadline:
            break
        await asyncio.sleep(0.05)
    if not await asyncio.to_thread(owner.members):
        return
    await asyncio.to_thread(owner.signal, signal.SIGKILL)
    deadline = loop.time() + kill_timeout
    while await asyncio.to_thread(owner.members):
        if loop.time() >= deadline:
            raise RuntimeError("Process group did not settle")
        await asyncio.sleep(0.05)


def _start_ticks() -> int | None:
    """Linux /proc start times use boot-time ticks, including suspended time."""
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    if clock is None:
        return None
    return int(time.clock_gettime(clock) * os.sysconf("SC_CLK_TCK"))


def register_process_group(
    proc: asyncio.subprocess.Process, *, spawn_started: int | None = None
) -> None:
    """Record ownership immediately after spawning with start_new_session=True."""
    # A new session's group ID is its leader PID, even if it has already exited.
    _PROCESS_GROUPS[proc] = proc.pid
    leader = _process_identity(proc.pid)
    if leader is not None:
        spawn_started = leader[0]
    _GROUP_OWNERS[proc] = ProcessGroupOwnership(
        proc.pid, leader, spawn_started=spawn_started
    )


async def spawn_registered_process(
    spawn: Coroutine[object, object, asyncio.subprocess.Process],
) -> asyncio.subprocess.Process:
    """Finish an isolated spawn and reclaim its group if the caller cancels.

    Shield only the factory and registration, not subsequent process execution.
    Cancelling the factory itself can lose the handle after the OS spawn, leaving
    asyncio to kill just the leader rather than its descendants.
    """

    async def spawn_and_register() -> asyncio.subprocess.Process:
        spawn_started = _start_ticks()
        proc = await spawn
        register_process_group(proc, spawn_started=spawn_started)
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
