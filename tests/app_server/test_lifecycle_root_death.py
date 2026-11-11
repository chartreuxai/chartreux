"""Root death, session switch, orphan reaping, and fresh respawn (WP7 [7C]).

The lifecycle rules under test: root-process death is typed (pipe EOF or a
missed liveness heartbeat on the root pipe), a session switch closes the old
tree completely before it opens the new one, a child whose parent session is
gone is reaped and reconciled from its durable markers but NEVER replayed, and
a respawned child is a fresh incarnation that never resumes the earlier one.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._child_runtime_lifecycle import TreeNotReapedError
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._subagent_transcript import (
    ChildStartMarker,
    ChildTranscriptOutcome,
    ChildTranscriptStore,
    child_transcript_dir_name,
    child_transcripts_root,
)
from chartreux.app_server.child_agent_process import ProcessSubagentSpawn
from chartreux.app_server.process_bootstrap import (
    DEFAULT_SHUTDOWN_TIMEOUT,
    ReadyFrame,
    SpawnedProcess,
    encode_ready_frame,
    spawn_process,
)
from chartreux.app_server.root_agent_process import (
    DEFAULT_ROOT_HEARTBEAT_INTERVAL,
    RootAgentExitedError,
    RootAgentProcess,
)
from chartreux.app_server.subagent_ports import (
    SubagentAdmissionClosedError,
    SubagentSpawnDeniedError,
    child_spawn_spec,
)
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    _ManualClock,
    _reaper_registry,
    _record,
)

_READY_LITERAL = encode_ready_frame().decode().strip()

_SLEEPING_CHILD = (
    "import sys, time\n"
    f"sys.stdout.write({_READY_LITERAL!r} + chr(10))\n"
    "sys.stdout.flush()\n"
    "time.sleep(60)\n"
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


class _ProcessDouble:
    """A process handle double whose liveness the test controls."""

    stdin = None
    stdout = None

    def __init__(self, exit_code: int | None) -> None:
        self.exit_code = exit_code
        self.signals: list[str] = []

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.signals.append("terminate")

    def kill(self) -> None:
        self.signals.append("kill")


class _SilentTransport:
    """A transport whose message stream never ends: only the heartbeat can tell."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._never = asyncio.Event()

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        await self._never.wait()
        if False:  # pragma: no cover - an async generator must contain a yield
            yield {}

    async def close(self) -> None:
        self.closed = True


def _dead_root(
    *, exit_code: int | None, heartbeat_interval: float = 0.01
) -> RootAgentProcess:
    transport = _SilentTransport()
    spawned = SpawnedProcess(
        process=cast("subprocess.Popen[bytes]", _ProcessDouble(exit_code)),
        transport=transport,  # type: ignore[arg-type] - transport double
        ready=ReadyFrame(protocol_version=1),
    )
    return RootAgentProcess(spawned, heartbeat_interval=heartbeat_interval)


def _reap(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)


class _ClosingHandle:
    """A spawned-subagent double that records when it is reaped."""

    agent_id = "agent-1"
    completion_source: Any = None

    def __init__(self, order: list[str]) -> None:
        self._order = order

    async def close(self) -> None:
        self._order.append("close")


def _orphan_child_dir(root_session_dir: Path) -> Path:
    child_dir = child_transcripts_root(root_session_dir) / child_transcript_dir_name(
        "coder", "session-agent-1", timestamp="20260101_000000"
    )
    child_dir.mkdir(parents=True)
    ChildTranscriptStore(child_dir, root_session_dir=root_session_dir).record_start(
        ChildStartMarker(
            agent_id="agent-1",
            session_id="session-agent-1",
            run_id="run-1",
            started_at="2026-01-01T00:00:00+00:00",
        )
    )
    return child_dir


def test_the_root_heartbeat_is_the_planned_two_seconds() -> None:
    assert DEFAULT_ROOT_HEARTBEAT_INTERVAL == 2.0


@pytest.mark.asyncio
async def test_a_root_pipe_eof_is_a_typed_root_death() -> None:
    spawned = await spawn_process([sys.executable, "-c", _SLEEPING_CHILD])
    process = RootAgentProcess(spawned, heartbeat_interval=0.05)
    try:
        assert process.alive
        spawned.process.kill()
        spawned.process.wait(timeout=5)
        death = await asyncio.wait_for(process.wait_for_death(), timeout=10)
        assert process.alive is False
        assert death.exit_code is not None
        with pytest.raises(RootAgentExitedError):
            await process.read_page("agent-1")
    finally:
        await process.close()
        _reap(spawned.process)


@pytest.mark.asyncio
async def test_a_missed_heartbeat_detects_a_dead_root_whose_pipe_stays_open() -> None:
    process = _dead_root(exit_code=9)
    try:
        death = await asyncio.wait_for(process.wait_for_death(), timeout=5)
        assert death.exit_code == 9
        assert death.reason == "the root agent process exited"
        assert process.alive is False
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_a_request_in_flight_fails_typed_when_the_root_dies() -> None:
    process = _dead_root(exit_code=None)
    try:
        request = asyncio.create_task(process.read_page("agent-1"))
        await asyncio.sleep(0)
        # The root process is gone while its pipe stays open: the heartbeat
        # (not EOF) is what declares death here.
        cast("_ProcessDouble", process.process).exit_code = 3
        await asyncio.wait_for(process.wait_for_death(), timeout=5)
        with pytest.raises(RootAgentExitedError):
            await request
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_a_deliberate_close_is_never_reported_as_root_death() -> None:
    spawned = await spawn_process([sys.executable, "-c", _SLEEPING_CHILD])
    process = RootAgentProcess(spawned, heartbeat_interval=0.05)
    try:
        await process.close()
        assert process.death is None
        assert process.alive is False
    finally:
        _reap(spawned.process)


@pytest.mark.asyncio
async def test_root_death_disposes_of_the_whole_child_tree() -> None:
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock)
    record = _record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    report = await registry.dispose_tree_after_root_death()

    assert report.complete
    assert report.drain is not None
    assert report.drain.complete
    cast("AsyncMock", record.runtime.close).assert_awaited()
    assert not registry._children


@pytest.mark.asyncio
async def test_a_session_switch_closes_the_old_tree_before_it_reopens_admission() -> (
    None
):
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
    switch = asyncio.create_task(registry.close_tree_for_switch())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        # The old tree is still closing: no new child may start beside it.
        with pytest.raises(SubagentAdmissionClosedError):
            registry._reserve_admission(
                TaskArgs(task="late work", identity="coder", background=True),
                InvokeContext(tool_call_id="late", session_id="root"),
            )
    finally:
        gate.set()
        report = await asyncio.wait_for(switch, timeout=2)

    assert report.complete
    cast("AsyncMock", record.runtime.close).assert_awaited()
    assert registry._admission_closed


@pytest.mark.asyncio
async def test_a_tree_that_could_not_be_reaped_refuses_a_switch() -> None:
    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        clock=_JumpClock(100.0),
        shutdown_timeout=DEFAULT_SHUTDOWN_TIMEOUT,
    )

    with pytest.raises(TreeNotReapedError) as switch_error:
        await registry.close_tree_for_switch()
    with pytest.raises(TreeNotReapedError) as death_error:
        await registry.dispose_tree_after_root_death()

    assert switch_error.value.report.timed_out
    assert death_error.value.report.timed_out
    assert "no new tree starts before the old one is reaped" in str(switch_error.value)


@pytest.mark.asyncio
async def test_an_orphaned_child_is_reaped_and_never_replayed(tmp_path: Path) -> None:
    clock = _ManualClock(0)
    registry, root = _reaper_registry(clock)
    root_session_dir = tmp_path / "session-root"
    root_session_dir.mkdir()
    root.agent_loop.session_logger.session_dir = root_session_dir
    child_dir = _orphan_child_dir(root_session_dir)
    record = _record("agent-1", clock=clock)
    record.runtime.agent_loop.session_id = "session-agent-1"
    record.runtime.agent_loop.parent_session_id = "old-root"
    record.runtime.agent_loop.session_logger.session_dir = child_dir
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    report = await registry.reap_orphaned_children()

    assert report.complete
    assert report.reaped == 1
    assert report.reconciled == 1
    orphan = report.orphans[0]
    assert orphan.agent_id == "agent-1"
    assert orphan.reconciliation is not None
    assert orphan.reconciliation.child_dir == child_dir
    assert (
        orphan.reconciliation.outcome is ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN
    )
    assert orphan.reconciliation.denial is not None
    assert "will not be replayed" in orphan.reconciliation.denial
    cast("AsyncMock", record.runtime.close).assert_awaited_once()
    assert not registry._children
    assert not registry._agent_records


@pytest.mark.asyncio
async def test_a_child_of_the_live_session_is_not_an_orphan() -> None:
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock)
    record = _record("agent-1", clock=clock)
    record.runtime.agent_loop.parent_session_id = "root"
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    report = await registry.reap_orphaned_children()

    assert report.orphans == ()
    assert registry._children[record.session_id] is record.runtime
    cast("AsyncMock", record.runtime.close).assert_not_awaited()


@pytest.mark.asyncio
async def test_an_orphan_is_reconciled_under_the_session_that_owned_it(
    tmp_path: Path,
) -> None:
    """A switch leaves the orphan's transcript under the OLD root session."""
    clock = _ManualClock(0)
    registry, root = _reaper_registry(clock)
    old_root_session_dir = tmp_path / "session-old"
    old_root_session_dir.mkdir()
    new_root_session_dir = tmp_path / "session-new"
    new_root_session_dir.mkdir()
    root.agent_loop.session_logger.session_dir = new_root_session_dir
    child_dir = _orphan_child_dir(old_root_session_dir)
    record = _record("agent-1", clock=clock)
    record.runtime.agent_loop.session_id = "session-agent-1"
    record.runtime.agent_loop.parent_session_id = "old-root"
    record.runtime.agent_loop.session_logger.session_dir = child_dir
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    report = await registry.reap_orphaned_children()

    assert report.complete
    assert report.reconciled == 1
    orphan = report.orphans[0]
    assert orphan.reconciliation is not None
    assert orphan.reconciliation.child_dir == child_dir
    assert (
        orphan.reconciliation.outcome is ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN
    )


@pytest.mark.asyncio
async def test_a_respawn_reaps_the_previous_child_before_it_spawns(
    tmp_path: Path,
) -> None:
    port = ProcessSubagentSpawn(env={})
    order: list[str] = []
    spawned = object()

    async def fake_spawn(spec: object) -> object:
        order.append("spawn")
        return spawned

    port.spawn = fake_spawn  # type: ignore[method-assign]
    spec = child_spawn_spec(
        agent_id="agent-1", parent_session_id="root", root_session_dir=tmp_path
    )

    result = await port.respawn(spec, previous=_ClosingHandle(order))

    assert result is spawned
    assert order == ["close", "spawn"]


@pytest.mark.asyncio
async def test_a_respawn_never_resumes_a_named_child_session(tmp_path: Path) -> None:
    port = ProcessSubagentSpawn(env={})
    port.spawn = AsyncMock()  # type: ignore[method-assign]
    spec = replace(
        child_spawn_spec(
            agent_id="agent-1", parent_session_id="root", root_session_dir=tmp_path
        ),
        session_dir=str(tmp_path / "child-session"),
    )

    with pytest.raises(SubagentSpawnDeniedError, match="never resumed"):
        await port.respawn(spec, previous=_ClosingHandle([]))

    port.spawn.assert_not_awaited()
