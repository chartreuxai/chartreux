"""Ready/version/deadline/spawn-cap lifecycle primitives (WP7 [7A]).

The bounds under test are the plan's validation candidates: ONE 15 s
spawn-ready deadline per process, enforced on a monotonic clock (fake clocks
here, no waiting out the bound), and the hard process-tree spawn cap whose
default is the existing concurrent-children cap.
"""

from __future__ import annotations

import asyncio
from inspect import signature
from pathlib import Path
import sys
import time

import pytest

from chartreux.app_server.child_agent_process import ProcessSubagentSpawn
from chartreux.app_server.process_bootstrap import (
    DEFAULT_MAX_CHILD_PROCESSES,
    DEFAULT_MAX_PROCESS_TREE_SIZE,
    DEFAULT_SPAWN_READY_TIMEOUT,
    PROCESS_PROTOCOL_VERSION,
    PROCESS_TREE_FIXED_MEMBERS,
    ProcessBootstrapError,
    ProcessTreeBudget,
    ProtocolVersionMismatchError,
    ReadyDeadline,
    ReadyFrameMissingError,
    ReadyTimeoutError,
    SpawnCapExceededError,
    encode_ready_frame,
    spawn_process,
)
from chartreux.app_server.root_agent_process import (
    RootAgentProcess,
    spawn_root_agent_process,
)
from chartreux.core.config.models import SubagentsConfig


class _FakeClock:
    """A monotonic clock a test spends explicitly, never the wall clock."""

    def __init__(self, started: float = 1000.0) -> None:
        self.value = started

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeSleep:
    """A sleeper that spends the requested delay on the fake clock at once."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        self._clock.advance(delay)


def _child_script(
    body: str, *, protocol_version: int = PROCESS_PROTOCOL_VERSION
) -> list[str]:
    ready_literal = encode_ready_frame(protocol_version).decode().strip()
    announcement = f"sys.stdout.write({ready_literal!r} + '\\n')\nsys.stdout.flush()\n"
    return [sys.executable, "-c", "import sys\n" + announcement + body]


def _echo_child() -> list[str]:
    return _child_script(
        "for line in sys.stdin:\n    sys.stdout.write(line)\n    sys.stdout.flush()\n"
    )


def _silent_child() -> list[str]:
    return [sys.executable, "-c", "import time\ntime.sleep(5)\n"]


def test_the_spawn_ready_bound_is_fifteen_seconds_on_every_spawn_surface() -> None:
    assert DEFAULT_SPAWN_READY_TIMEOUT == 15.0
    for surface in (spawn_process, spawn_root_agent_process):
        default = signature(surface).parameters["ready_timeout"].default
        assert default == DEFAULT_SPAWN_READY_TIMEOUT, surface.__name__
    adapter_default = (
        signature(ProcessSubagentSpawn.__init__).parameters["ready_timeout"].default
    )
    assert adapter_default == DEFAULT_SPAWN_READY_TIMEOUT


def test_the_ready_deadline_reads_the_monotonic_clock_by_default() -> None:
    deadline = ReadyDeadline(DEFAULT_SPAWN_READY_TIMEOUT)

    assert deadline.clock is time.monotonic
    assert deadline.timeout == DEFAULT_SPAWN_READY_TIMEOUT
    assert deadline.expires_at - deadline.started_at == DEFAULT_SPAWN_READY_TIMEOUT


def test_the_ready_deadline_counts_down_on_the_injected_clock() -> None:
    clock = _FakeClock()
    deadline = ReadyDeadline(15.0, clock=clock)

    assert deadline.remaining() == pytest.approx(15.0)
    assert deadline.expired() is False

    clock.advance(14.0)
    assert deadline.remaining() == pytest.approx(1.0)
    assert deadline.expired() is False

    clock.advance(1.0)
    assert deadline.remaining() == pytest.approx(0.0)
    assert deadline.expired() is True


def test_the_ready_deadline_sleeps_exactly_its_remaining_time() -> None:
    clock = _FakeClock()
    sleep = _FakeSleep(clock)
    deadline = ReadyDeadline(15.0, clock=clock)

    asyncio.run(deadline.expire(sleep))

    assert sleep.delays == [pytest.approx(15.0)]


def test_an_expired_ready_deadline_never_sleeps() -> None:
    clock = _FakeClock()
    sleep = _FakeSleep(clock)
    deadline = ReadyDeadline(15.0, clock=clock)
    clock.advance(20.0)

    asyncio.run(deadline.expire(sleep))

    assert sleep.delays == []


def test_the_ready_deadline_rejects_a_non_positive_bound() -> None:
    for timeout in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite positive"):
            ReadyDeadline(timeout)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_no_ready_frame_is_a_typed_timeout_on_the_injected_clock() -> None:
    clock = _FakeClock()
    sleep = _FakeSleep(clock)
    started = time.monotonic()

    with pytest.raises(ReadyTimeoutError, match="within 15.0s"):
        await spawn_process(_silent_child(), clock=clock, sleep=sleep)

    # The bound was spent on the fake clock, not by sleeping the real one.
    assert sleep.delays == [pytest.approx(DEFAULT_SPAWN_READY_TIMEOUT)]
    assert time.monotonic() - started < DEFAULT_SPAWN_READY_TIMEOUT


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_late_ready_frame_is_a_typed_timeout() -> None:
    ready_literal = encode_ready_frame().decode().strip()
    late_child = [
        sys.executable,
        "-c",
        "import sys, time\ntime.sleep(0.5)\n"
        f"sys.stdout.write({ready_literal!r} + '\\n')\nsys.stdout.flush()\n",
    ]

    with pytest.raises(ReadyTimeoutError, match="within 0.05s"):
        await spawn_process(late_child, ready_timeout=0.05)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_version_mismatch_is_typed_and_never_negotiated() -> None:
    # The child announces a wrong version, then a correct one: a handshake
    # that negotiated would accept the second frame.
    negotiated = _child_script(
        "sys.stdout.write("
        + repr(encode_ready_frame().decode().strip())
        + " + '\\n')\n"
        "sys.stdout.flush()\n"
        "for line in sys.stdin:\n    sys.stdout.write(line)\n    sys.stdout.flush()\n",
        protocol_version=PROCESS_PROTOCOL_VERSION + 1,
    )

    with pytest.raises(ProtocolVersionMismatchError) as caught:
        await spawn_process(negotiated, ready_timeout=15)

    assert isinstance(caught.value, ProcessBootstrapError)
    assert caught.value.expected == PROCESS_PROTOCOL_VERSION
    assert caught.value.received == PROCESS_PROTOCOL_VERSION + 1


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_mismatched_child_is_terminated_and_reaped(tmp_path: Path) -> None:
    marker = tmp_path / "exited.txt"
    script = _child_script(
        f"for line in sys.stdin:\n    pass\n"
        f"with open({str(marker)!r}, 'w', encoding='utf-8') as handle:\n"
        "    handle.write('exited')\n",
        protocol_version=PROCESS_PROTOCOL_VERSION + 1,
    )

    with pytest.raises(ProtocolVersionMismatchError):
        await spawn_process(script, ready_timeout=15)

    # The failed handshake closes the child's stdin and reaps it, so the
    # child observes EOF and records its own exit.
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert marker.read_text(encoding="utf-8") == "exited"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_failed_handshake_returns_its_tree_slot() -> None:
    budget = ProcessTreeBudget(max_processes=2, reserved_members=1)

    with pytest.raises(ReadyFrameMissingError):
        await spawn_process(
            [sys.executable, "-c", "pass"], ready_timeout=15, budget=budget
        )

    assert budget.live_processes == 0
    spawned = await spawn_process(_echo_child(), ready_timeout=15, budget=budget)
    try:
        assert budget.live_processes == 1
    finally:
        await spawned.transport.close()
        spawned.retire()


def test_the_default_tree_cap_mirrors_the_concurrent_child_cap() -> None:
    assert DEFAULT_MAX_CHILD_PROCESSES == SubagentsConfig().max_running_subagents
    assert PROCESS_TREE_FIXED_MEMBERS == 2
    budget = ProcessTreeBudget()
    assert budget.max_processes == DEFAULT_MAX_PROCESS_TREE_SIZE
    assert budget.capacity == DEFAULT_MAX_CHILD_PROCESSES


def test_the_tree_budget_rejects_bounds_it_cannot_defend() -> None:
    for max_processes, reserved in ((0, 0), (1, 1), (2, 3), (True, 0), (2, -1)):
        with pytest.raises(ValueError):
            ProcessTreeBudget(max_processes, reserved_members=reserved)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_spawn_cap_denies_typed_and_starts_nothing() -> None:
    budget = ProcessTreeBudget(max_processes=3, reserved_members=1)
    spawned = [
        await spawn_process(_echo_child(), ready_timeout=15, budget=budget)
        for _ in range(budget.capacity)
    ]
    try:
        assert budget.live_processes == 2

        # A command that cannot exist: had the cap admitted the spawn, Popen
        # would have raised instead of the typed denial.
        with pytest.raises(SpawnCapExceededError) as caught:
            await spawn_process(
                ["/nonexistent/chartreux-cap-child"], ready_timeout=15, budget=budget
            )

        denial = caught.value
        assert (denial.live, denial.cap) == (2, 2)
        assert isinstance(denial, ProcessBootstrapError)
        assert "close a process before spawning another" in str(denial)
        assert budget.live_processes == 2
    finally:
        for spawned_process in spawned:
            await spawned_process.transport.close()
            spawned_process.retire()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_retiring_a_handle_frees_its_tree_slot_once() -> None:
    budget = ProcessTreeBudget(max_processes=2, reserved_members=1)
    spawned = await spawn_process(_echo_child(), ready_timeout=15, budget=budget)
    try:
        assert budget.live_processes == 1
        spawned.retire()
        spawned.retire()
        assert budget.live_processes == 0

        replacement = await spawn_process(
            _echo_child(), ready_timeout=15, budget=budget
        )
        replacement.retire()
        await replacement.transport.close()
    finally:
        await spawned.transport.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_crashed_process_is_pruned_at_the_next_reservation() -> None:
    budget = ProcessTreeBudget(max_processes=2, reserved_members=1)
    spawned = await spawn_process(_echo_child(), ready_timeout=15, budget=budget)
    try:
        spawned.process.kill()
        spawned.process.wait(timeout=10)
        assert spawned.process.poll() is not None

        # The crashed handle was never retired; the slot must not leak.
        replacement = await spawn_process(
            _echo_child(), ready_timeout=15, budget=budget
        )
        replacement.retire()
        await replacement.transport.close()
        assert budget.live_processes == 0
    finally:
        await spawned.transport.close()
        spawned.retire()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_closing_the_root_handle_returns_its_tree_slot() -> None:
    budget = ProcessTreeBudget(max_processes=2, reserved_members=1)
    spawned = await spawn_process(_echo_child(), ready_timeout=15, budget=budget)
    handle = RootAgentProcess(spawned)

    assert budget.live_processes == 1
    await handle.close()

    assert budget.live_processes == 0
    assert handle.process.returncode is not None


def test_the_child_spawn_adapter_owns_a_default_tree_budget() -> None:
    port = ProcessSubagentSpawn()

    assert port._budget.capacity == DEFAULT_MAX_CHILD_PROCESSES
