"""Ordered tree shutdown: admission → interrupt → saves → kill → IPC → reap (WP7 [7B]).

The bounds under test are the plan's validation candidates: ONE 10 s shutdown
bound covering the whole ordered sequence including the save drain and the
terminate/reap tail. Expired bounds report a typed timeout and never read as a
successful drain, and no new tree starts before the old tree is reaped.
"""

from __future__ import annotations

import asyncio
from inspect import signature
import subprocess
import sys
from typing import IO
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._child_runtime_lifecycle import TreeShutdownStatus
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.process_bootstrap import (
    DEFAULT_SHUTDOWN_TIMEOUT,
    ProcessTreeBudget,
    ProcessTreeClosingError,
    ProcessTreeShutdown,
    ShutdownPhase,
    ShutdownStatus,
    spawn_process,
)
from chartreux.app_server.subagent_ports import SubagentAdmissionClosedError
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    _ManualClock,
    _reaper_registry,
    _record,
)


class _JumpClock:
    """Every read after the first advances by ``step`` (an expired bound)."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.value = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls > 1:
            self.value += self.step
        return self.value


class _GateSleep:
    """Park the first bounded wait, then spend each delay on the fake clock."""

    def __init__(self, clock: _ManualClock) -> None:
        self._clock = clock
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, delay: float) -> None:
        self.entered.set()
        await self.release.wait()
        # A real SIGKILL needs real time to land: spend the delay for real so
        # the fake clock cannot race past the reap before the child dies.
        await asyncio.sleep(delay)
        self._clock.advance(delay)


class _AdvancingSleep:
    """Spend each delay on the fake clock at once."""

    def __init__(self, clock: _ManualClock) -> None:
        self._clock = clock

    async def __call__(self, delay: float) -> None:
        self._clock.advance(delay)


class _StubbornProcess:
    """A process double that ignores SIGTERM and never exits."""

    stdin: IO[bytes] | None = None
    stdout: IO[bytes] | None = None

    def __init__(self) -> None:
        self.signals: list[str] = []

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.signals.append("terminate")

    def kill(self) -> None:
        self.signals.append("kill")


class _KillableProcess:
    """A process double that exits only once killed."""

    stdin: IO[bytes] | None = None
    stdout: IO[bytes] | None = None

    def __init__(self) -> None:
        self.alive = True
        self.signals: list[str] = []

    def poll(self) -> int | None:
        return None if self.alive else 0

    def terminate(self) -> None:
        self.signals.append("terminate")

    def kill(self) -> None:
        self.signals.append("kill")
        self.alive = False


def _cooperative_child() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time\ntime.sleep(60)\n"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )


def _sigterm_ignoring_child() -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "sys.stdout.write('ready\\n')\n"
                "sys.stdout.flush()\n"
                "time.sleep(60)\n"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == b"ready"
    return process


def _reap(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)


def test_the_shutdown_bound_is_ten_seconds_on_every_surface() -> None:
    assert DEFAULT_SHUTDOWN_TIMEOUT == 10.0
    assert (
        signature(SessionRuntimeRegistry.__init__)
        .parameters["shutdown_timeout"]
        .default
        == DEFAULT_SHUTDOWN_TIMEOUT
    )
    assert (
        signature(ProcessTreeShutdown.__init__).parameters["timeout"].default
        == DEFAULT_SHUTDOWN_TIMEOUT
    )


@pytest.mark.asyncio
async def test_ordered_shutdown_terminates_closes_ipc_and_reaps_a_child() -> None:
    child = _cooperative_child()
    try:
        tree = ProcessTreeShutdown(
            processes=lambda: [child], grace=2.0, timeout=DEFAULT_SHUTDOWN_TIMEOUT
        )
        report = await tree.shutdown()
        assert report.status is ShutdownStatus.COMPLETE
        assert report.complete
        assert report.phases == (
            ShutdownPhase.TERMINATE,
            ShutdownPhase.CLOSE_IPC,
            ShutdownPhase.REAP,
        )
        assert report.terminated == 1
        assert report.reaped == 1
        assert report.remaining == 0
        assert child.poll() is not None
    finally:
        _reap(child)


@pytest.mark.asyncio
async def test_a_child_that_ignores_sigterm_is_killed_then_reaped() -> None:
    child = _sigterm_ignoring_child()
    try:
        tree = ProcessTreeShutdown(processes=lambda: [child], grace=0.1)
        report = await tree.shutdown()
        assert report.complete
        assert report.terminated == 1
        assert report.killed == 1
        assert report.reaped == 1
        assert report.remaining == 0
        assert child.poll() is not None
    finally:
        _reap(child)


@pytest.mark.asyncio
async def test_an_expired_bound_is_typed_and_never_a_clean_stop() -> None:
    process = _StubbornProcess()
    tree = ProcessTreeShutdown(
        processes=lambda: [process], timeout=0.001, grace=1.0, clock=_JumpClock(100.0)
    )

    report = await tree.shutdown()

    assert report.status is ShutdownStatus.TIMED_OUT
    assert report.timed_out is True
    assert report.complete is False
    assert report.reaped == 0
    assert report.remaining == 1
    # The escalation still runs: SIGTERM then SIGKILL, and the bound's expiry
    # is reported instead of being mistaken for a stop that succeeded.
    assert process.signals == ["terminate", "kill"]


@pytest.mark.asyncio
async def test_no_new_tree_starts_before_the_old_tree_is_reaped() -> None:
    child = _sigterm_ignoring_child()
    budget = ProcessTreeBudget(max_processes=4, reserved_members=1)
    budget.attach(budget.reserve(), child)
    clock = _ManualClock(0.0)
    sleeper = _GateSleep(clock)
    tree = ProcessTreeShutdown(
        budget, grace=1.0, timeout=DEFAULT_SHUTDOWN_TIMEOUT, clock=clock, sleep=sleeper
    )
    shutdown = asyncio.create_task(tree.shutdown())
    try:
        await asyncio.wait_for(sleeper.entered.wait(), timeout=5)
        # The old tree is still being stopped: admission is shut and a spawn
        # through the same budget is denied typed, not started.
        assert budget.shutting_down
        with pytest.raises(ProcessTreeClosingError, match="shutting down"):
            budget.reserve()
        with pytest.raises(ProcessTreeClosingError):
            await spawn_process([sys.executable, "-c", ""], budget=budget)
    finally:
        sleeper.release.set()
        report = await asyncio.wait_for(shutdown, timeout=10)
        _reap(child)

    assert report.complete
    # Only after the old tree is reaped does the budget reopen.
    assert not budget.shutting_down
    budget.release(budget.reserve())


@pytest.mark.asyncio
async def test_a_timed_out_drain_is_never_a_successful_tree_shutdown() -> None:
    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        clock=_JumpClock(100.0),
        shutdown_timeout=DEFAULT_SHUTDOWN_TIMEOUT,
    )

    report = await registry.shutdown_tree()

    assert report.status is TreeShutdownStatus.TIMED_OUT
    assert report.timed_out is True
    assert report.complete is False
    assert report.drain is not None
    assert report.drain.timed_out is True
    assert report.phases == (
        ShutdownPhase.STOP_ADMISSION,
        ShutdownPhase.INTERRUPT,
        ShutdownPhase.DRAIN_SAVES,
    )


@pytest.mark.asyncio
async def test_shutdown_stops_admission_before_it_drains() -> None:
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock)
    record = _record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def gated_close() -> None:
        entered.set()
        await gate.wait()

    record.runtime.close = AsyncMock(side_effect=gated_close)
    shutdown = asyncio.create_task(registry.shutdown_tree())
    try:
        # Admission closed before the drain reached its close phase.
        await asyncio.wait_for(entered.wait(), timeout=1)
        with pytest.raises(SubagentAdmissionClosedError):
            registry._reserve_admission(
                TaskArgs(task="late work", identity="coder", background=True),
                InvokeContext(tool_call_id="late", session_id="root"),
            )
    finally:
        gate.set()
        report = await asyncio.wait_for(shutdown, timeout=2)

    assert report.complete
    assert report.phases == (
        ShutdownPhase.STOP_ADMISSION,
        ShutdownPhase.INTERRUPT,
        ShutdownPhase.DRAIN_SAVES,
    )


@pytest.mark.asyncio
async def test_the_process_tail_runs_after_the_drain_inside_the_same_bound() -> None:
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock)
    process = _KillableProcess()
    tree = ProcessTreeShutdown(
        processes=lambda: [process],
        grace=2.0,
        timeout=DEFAULT_SHUTDOWN_TIMEOUT,
        clock=clock,
        sleep=_AdvancingSleep(clock),
    )
    registry._tree_shutdown.attach_process_tree(tree)

    report = await registry.shutdown_tree()

    assert report.phases == (
        ShutdownPhase.STOP_ADMISSION,
        ShutdownPhase.INTERRUPT,
        ShutdownPhase.DRAIN_SAVES,
        ShutdownPhase.TERMINATE,
        ShutdownPhase.CLOSE_IPC,
        ShutdownPhase.REAP,
    )
    assert report.process_tree is not None
    assert report.process_tree.complete
    assert report.complete
    assert process.signals == ["terminate", "kill"]
    assert report.elapsed <= DEFAULT_SHUTDOWN_TIMEOUT


@pytest.mark.asyncio
async def test_close_runs_the_ordered_tree_shutdown() -> None:
    registry, _root = _reaper_registry(_ManualClock(0))
    shutdown = AsyncMock(
        return_value=AsyncMock(status=TreeShutdownStatus.COMPLETE, elapsed=0.0)
    )
    registry._tree_shutdown.shutdown = shutdown

    await registry.close()

    shutdown.assert_awaited_once()
