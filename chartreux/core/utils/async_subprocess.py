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

from chartreux.core.job_launcher import JOB_OWNER_MARKER_ENV, JobGroupRecord

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


PROCESS_GROUP_REAPED = "reaped"
"""The group holds no live member: the settlement course is complete."""

PROCESS_GROUP_STALE_REFUSED = "stale-refused"
"""The recorded owner no longer holds the group, so no signal was sent."""

PROCESS_GROUP_INCONCLUSIVE = "inconclusive"
"""Ownership (or membership) could not be established: no signal was sent."""


@dataclass(frozen=True, slots=True)
class ProcessGroupSettlement:
    """What one TERM-to-KILL settlement ESTABLISHED about a recorded job group.

    ``reaped`` is the only outcome that claims the group is gone; a stale
    refusal and an inconclusive settlement are typed warnings, never a pass.
    """

    outcome: str
    detail: str = ""

    @property
    def reaped(self) -> bool:
        """Whether the group is settled with no live member left."""
        return self.outcome == PROCESS_GROUP_REAPED


def _process_environ(pid: int) -> dict[str, str] | None:
    """One process's environment, or ``None`` when /proc cannot be read."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as stream:
            raw = stream.read(1 << 20)
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    entries: dict[str, str] = {}
    for item in raw.split(b"\0"):
        name, separator, value = item.partition(b"=")
        if separator:
            entries[name.decode("utf-8", "surrogateescape")] = value.decode(
                "utf-8", "surrogateescape"
            )
    return entries


class RecordedJobGroup:
    """One job group's DURABLE identity, revalidated before every signal.

    The identity comes from the pgid file the launcher wrote before it armed
    itself, so this ownership survives the process that spawned the job: it is
    reconstructed from disk rather than from an in-process spawn handle.
    Numeric pgid reuse is not excluded by the shipped member validation, so a
    group is signaled only while the RECORDED leader identity is live (pid plus
    start time) or every live member carries the session job-owner marker read
    from /proc. A group that proves neither is never signaled.

    A verdict of ``None`` means the predicate could not be ESTABLISHED at all:
    it is a different answer from a mismatch, and neither one is permission.
    """

    def __init__(self, record: JobGroupRecord) -> None:
        self.record = record

    @property
    def pgid(self) -> int:
        """The group id: the launcher's pid, which survives its exec."""
        return self.record.pid

    def members(self) -> bool | None:
        """Whether a live (non-zombie) member remains; ``None`` if unreadable."""
        try:
            names = os.listdir("/proc")
        except OSError:
            return None
        for name in names:
            if not name.isdecimal():
                continue
            identity = _process_identity(int(name))
            if identity is None or identity[1:3] != (self.pgid, self.pgid):
                continue
            if identity[3] != "Z":
                return True
        return False

    def validated(self) -> bool | None:
        """Whether this record still owns the group it names; ``None`` if unproven.

        The leader path is the recorded pid plus its start time, so a reused pid
        can never pass. When the leader is gone (the kernel may have killed it
        while cooperative descendants live on), coverage comes from the owner
        marker every member inherited.
        """
        leader = _process_identity(self.pgid)
        if leader is not None:
            if leader[0] != self.record.start_time or leader[1:3] != (
                self.pgid,
                self.pgid,
            ):
                return False
            return leader[3] != "Z"
        if not self.record.session_id:
            return None
        return self._members_carry_owner_marker()

    def _members_carry_owner_marker(self) -> bool | None:
        """Whether every live member carries this session's job-owner marker."""
        try:
            names = os.listdir("/proc")
        except OSError:
            return None
        unreadable = False
        for name in names:
            if not name.isdecimal():
                continue
            pid = int(name)
            identity = _process_identity(pid)
            if (
                identity is None
                or identity[1:3] != (self.pgid, self.pgid)
                or identity[3] == "Z"
            ):
                continue
            environ = _process_environ(pid)
            if environ is None:
                unreadable = True
                continue
            if environ.get(JOB_OWNER_MARKER_ENV) != self.record.session_id:
                return False
        return None if unreadable else True

    def signal_if_owned(self, sig: signal.Signals) -> bool | None:
        """Signal the group only once its recorded ownership is re-established.

        The predicate is read fresh here, immediately before the signal, which
        is what makes the escalation fail closed at both ends: ``None`` when it
        cannot be established, ``False`` on a mismatch, and neither one signals.
        """
        verdict = self.validated()
        if verdict is not True:
            return verdict
        try:
            os.killpg(self.pgid, sig)
        except ProcessLookupError:
            return True
        return True


async def settle_recorded_job_group(
    group: RecordedJobGroup,
    *,
    grace: float = 2.0,
    kill_timeout: float = 0.5,
    poll_interval: float = 0.05,
) -> ProcessGroupSettlement:
    """Settle one recorded job group under the fail-closed continuity contract.

    Ownership is reconstructed from the durable record BEFORE any signal and
    re-established independently before EACH signal in the TERM-to-KILL
    escalation, so a group whose identity changed in between is left alone: an
    un-re-establishable predicate escalates to the typed INCONCLUSIVE outcome,
    never a signal. Leader coverage is the launcher's (PDEATHSIG is SIGKILL), so
    this course never claims a graceful stop.
    """
    loop = asyncio.get_running_loop()
    live = await asyncio.to_thread(group.members)
    if live is None:
        return ProcessGroupSettlement(
            PROCESS_GROUP_INCONCLUSIVE, "the group's members could not be read"
        )
    if not live:
        return ProcessGroupSettlement(
            PROCESS_GROUP_REAPED, "the group had no live member"
        )
    for sig, bound, stage in (
        (signal.SIGTERM, grace, "term"),
        (signal.SIGKILL, kill_timeout, "kill"),
    ):
        refusal = await _signal_if_owned(group, sig, stage)
        if refusal is not None:
            return refusal
        live = await _await_members(group, loop.time() + max(bound, 0.0), poll_interval)
        if live is None:
            return ProcessGroupSettlement(
                PROCESS_GROUP_INCONCLUSIVE, "the group's members could not be read"
            )
        if not live:
            return ProcessGroupSettlement(
                PROCESS_GROUP_REAPED, f"the group settled after the {stage} signal"
            )
    return ProcessGroupSettlement(
        PROCESS_GROUP_INCONCLUSIVE,
        "the group was still live inside the kill bound; it is a typed "
        "residual, not a settled group",
    )


async def _signal_if_owned(
    group: RecordedJobGroup, sig: signal.Signals, stage: str
) -> ProcessGroupSettlement | None:
    """Send one signal, or the typed settlement that refuses to.

    The predicate is read fresh inside :meth:`RecordedJobGroup.signal_if_owned`,
    immediately before the signal, so an escalation whose ownership cannot be
    re-established stops here instead of signaling a group it may not own.
    """
    issued = await asyncio.to_thread(group.signal_if_owned, sig)
    if issued is None:
        return ProcessGroupSettlement(
            PROCESS_GROUP_INCONCLUSIVE,
            f"the recorded ownership could not be established before the {stage} "
            "signal; nothing was signaled",
        )
    if not issued:
        return ProcessGroupSettlement(
            PROCESS_GROUP_STALE_REFUSED,
            f"the recorded leader identity no longer owns the group before the "
            f"{stage} signal; nothing was signaled",
        )
    return None


async def _await_members(
    group: RecordedJobGroup, deadline: float, poll_interval: float
) -> bool | None:
    """Poll the group's membership until it empties or the bound expires."""
    loop = asyncio.get_running_loop()
    while True:
        live = await asyncio.to_thread(group.members)
        if live is None or not live:
            return live
        if loop.time() >= deadline:
            return True
        await asyncio.sleep(poll_interval)


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
