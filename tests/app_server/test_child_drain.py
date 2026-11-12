"""WP5 [5E] child drain: quiescence, durable terminal publication, bounded drain.

The terminal completion frame of every child run is published durably — the
child's terminal result record ([5C]) exists first, and the run's capacity slot
is released only after the publication settles. A drain is bounded by the
registry's shutdown timeout on the injected monotonic clock; a drain bounded
out reports TIMED_OUT and never reads as a successful drain.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._child_runtime_lifecycle import (
    ChildDrainReport,
    ChildDrainStatus,
)
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._subagent_recovery import ChildTranscriptReconciler
from chartreux.app_server._subagent_transcript import (
    ChildStartMarker,
    ChildTranscriptError,
    ChildTranscriptOutcome,
    ChildTranscriptStore,
)
from chartreux.app_server.agent_wire import (
    AGENT_TURN_ENDED_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_INTERRUPTED,
)
from chartreux.app_server.child_agent_process import teardown_spawned_attempt
from chartreux.app_server.subagent_ports import (
    AdmittingSubagentSpawn,
    ChildTeardownOutcome,
    SpawnedSubagent,
    SubagentAdmission,
    SubagentCompletion,
    SubagentCompletionSource,
    SubagentEventFeed,
    SubagentFrame,
    SubagentSpawnSpec,
)
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    _background_result,
    _config,
    _ManualClock,
    _reaper_registry,
    _record,
    _retention_registry,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _turn_request() -> ExecutionTurnRequest:
    return ExecutionTurnRequest(
        message="hello", turn_options=AgentTurnOptions(turn_id="turn-1")
    )


class _JumpClock:
    """A monotonic-clock double: every call past the first advances the clock.

    The drain reads the clock once at entry and once per bound check, so a
    step larger than the shutdown timeout expires the bound deterministically
    — no wall-clock waiting, and the measured bound is the injected one.
    """

    def __init__(self, step: float) -> None:
        self.step = step
        self.value = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls > 1:
            self.value += self.step
        return self.value


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


def _logging_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SessionRuntimeRegistry, Any]:
    """A registry over a parent whose children own transcripts under it."""
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    child_backend = FakeBackend([mock_llm_chunk(content="child done")])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config().model_copy(update={"session_logging": logging}),
        backend=FakeBackend([mock_llm_chunk(content="Notification received")]),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


# ---------------------------------------------------------------------------
# Terminal publication is exactly once and precedes the slot release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminal_publication_is_exactly_once() -> None:
    frames: list[dict[str, Any]] = []

    async def sink(frame: Mapping[str, Any]) -> None:
        frames.append(dict(frame))

    feed = SubagentEventFeed("agent-1", sink, incarnation=0)
    completion = SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED)
    assert await feed.publish_terminal(completion) is True
    # The run's terminal frame is published once: a second publication is a
    # no-op, so a cancelled consumer and its teardown never emit two frames.
    assert await feed.publish_terminal(completion) is False
    assert [frame["method"] for frame in frames] == [AGENT_TURN_ENDED_METHOD]
    assert feed.terminal_published is True


class _BlockingSource:
    def __init__(self, agent_id: str, started: asyncio.Event) -> None:
        self._agent_id = agent_id
        self._started = started

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def stream(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[SubagentFrame, None]:
        return self._stream()

    async def _stream(self) -> AsyncGenerator[SubagentFrame, None]:
        self._started.set()
        await asyncio.Event().wait()
        yield SubagentCompletion(outcome=TURN_OUTCOME_INTERRUPTED)  # pragma: no cover


class _BlockingSpawned:
    def __init__(self, agent_id: str, started: asyncio.Event) -> None:
        self._agent_id = agent_id
        self.completion_source: SubagentCompletionSource = _BlockingSource(
            agent_id, started
        )

    @property
    def agent_id(self) -> str:
        return self._agent_id

    async def close(self) -> None:
        return None


class _BlockingPort:
    def __init__(self, started: asyncio.Event) -> None:
        self._started = started

    async def spawn(self, spec: SubagentSpawnSpec) -> SpawnedSubagent:
        return _BlockingSpawned(spec.agent_id, self._started)


@pytest.mark.asyncio
async def test_cancelled_run_publishes_its_terminal_frame_before_releasing_the_slot() -> (
    None
):
    admission = SubagentAdmission()
    frames: list[dict[str, Any]] = []

    async def sink(frame: Mapping[str, Any]) -> None:
        frames.append(dict(frame))

    started = asyncio.Event()
    port = AdmittingSubagentSpawn(_BlockingPort(started), admission, sink=sink)
    admitted = await port.spawn(SubagentSpawnSpec(agent_id="agent-1"), cap=1)
    run = asyncio.create_task(admitted.run(_turn_request()))
    await asyncio.wait_for(started.wait(), timeout=1)
    assert admission.slots
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    # The interrupted run published its single terminal frame, and the slot
    # was released only after that publication.
    assert [frame["method"] for frame in frames] == [AGENT_TURN_ENDED_METHOD]
    assert frames[-1]["params"]["outcome"] == TURN_OUTCOME_INTERRUPTED
    assert not admission.slots


@pytest.mark.asyncio
async def test_terminal_frame_is_durable_before_the_slot_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent = _logging_registry(tmp_path, monkeypatch)
    entered = asyncio.Event()
    gate = asyncio.Event()
    frames: list[dict[str, Any]] = []

    async def sink(agent_id: str, frame: Any) -> None:
        if frame["method"] == AGENT_TURN_ENDED_METHOD:
            # Hold the terminal frame inside the feed: everything the
            # publication promises must already be true here.
            entered.set()
            await gate.wait()
        frames.append(dict(frame))

    registry._subagent_event_sink = sink
    try:
        ctx = InvokeContext(tool_call_id="run", session_id=parent.session_id)
        result = await _background_result(
            registry,
            TaskArgs(task="durable", identity="coder", background=True, retain=True),
            ctx,
        )
        assert result.agent_id is not None and result.run_id is not None
        await asyncio.wait_for(entered.wait(), timeout=5)
        # Durability first: the child's terminal result record exists BEFORE
        # the terminal frame is emitted.
        record = registry._agent_records[result.agent_id]
        child_runtime = registry._children[record.session_id]
        child_dir = child_runtime.agent_loop.session_logger.session_dir
        root_dir = parent.session_logger.session_dir
        assert child_dir is not None and root_dir is not None
        adjudication = ChildTranscriptStore(
            Path(child_dir), root_session_dir=Path(root_dir)
        ).adjudicate()
        assert adjudication.outcome is ChildTranscriptOutcome.COMPLETED
        assert adjudication.result_id == f"{result.agent_id}:{result.run_id}"
        # And the capacity slot is still held while the publication is
        # outstanding: completion never frees capacity before its durable
        # terminal publication.
        assert registry._active_work_slots
        gate.set()
        await _wait_for(
            lambda: (
                not registry._active_work_slots
                and any(frame["method"] == AGENT_TURN_ENDED_METHOD for frame in frames)
            )
        )
    finally:
        gate.set()
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_reconciliation_consumes_the_published_result_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent = _logging_registry(tmp_path, monkeypatch)
    try:
        ctx = InvokeContext(tool_call_id="run", session_id=parent.session_id)
        result = await _background_result(
            registry,
            TaskArgs(task="reconcile", identity="coder", background=True, retain=True),
            ctx,
        )
        assert result.agent_id is not None and result.run_id is not None
        await _wait_for(lambda: not registry._active_work_slots)
        # The published result is discoverable by the coordinator's
        # reconciliation, once.
        scanned = await registry.scan_child_transcripts()
        reported = [item for item in scanned if item.agent_id == result.agent_id]
        assert len(reported) == 1
        assert reported[0].outcome is ChildTranscriptOutcome.COMPLETED
        assert reported[0].result_id == f"{result.agent_id}:{result.run_id}"
        assert reported[0].unconsumed is True
        # The receipt makes the handoff idempotent: adopting it once consumes
        # it, and a repeated scan reports nothing left to adopt.
        assert reported[0].result_id is not None
        assert await registry.record_child_result_receipt(
            result.agent_id, reported[0].result_id
        )
        assert not await registry.record_child_result_receipt(
            result.agent_id, reported[0].result_id
        )
        rescanned = await registry.scan_child_transcripts()
        assert [
            item.unconsumed for item in rescanned if item.agent_id == result.agent_id
        ] == [False]
    finally:
        await registry.close()
        await parent.aclose()


# ---------------------------------------------------------------------------
# The drain is bounded and reports its disposition, typed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_within_the_bound_reports_drained() -> None:
    registry, _root = _reaper_registry(_ManualClock(0))
    record = _record("agent-1", clock=_ManualClock(0))
    close = AsyncMock()
    record.runtime.close = close
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    try:
        report = await registry.drain_children()
        assert isinstance(report, ChildDrainReport)
        assert report.status is ChildDrainStatus.DRAINED
        assert report.timed_out is False
        assert report.complete is True
        assert report.drained_children == 1
        assert close.await_count == 1
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_timed_out_drain_is_typed_and_never_reports_success() -> None:
    # A clock that jumps past the shutdown bound between the drain's entry
    # reading and its bound check: the bound is the injected monotonic one,
    # and expiring it must never read as a successful drain.
    clock = _JumpClock(step=100.0)
    registry, _root = _reaper_registry(cast("Any", clock))
    registry._shutdown_timeout = 10.0
    record = _record("agent-1", clock=_ManualClock(0))
    close = AsyncMock()
    record.runtime.close = close
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    try:
        report = await registry.drain_children()
        assert report.status is ChildDrainStatus.TIMED_OUT
        assert report.timed_out is True
        assert report.complete is False
        assert report.remaining_children == 1
        # The remaining child was terminated, and the inventory cleared: the
        # timed-out drain claims neither a drained child nor a live one.
        assert report.terminated_children == 1
        assert close.await_count == 1
        assert not registry._agent_records
        assert not registry._children
        # The final inventory is the empty one, never a ghost row.
        notify = cast("AsyncMock", registry._notify_agents)
        notify.assert_awaited_with([], [])
    finally:
        await registry.close()


# ---------------------------------------------------------------------------
# Item 1 child seam: the conservative publication teardown
# ---------------------------------------------------------------------------


class _TeardownAttempt:
    """A spawned publication attempt whose liveness the test decides."""

    def __init__(
        self,
        root: Path,
        *,
        alive_after_close: bool = False,
        reconciliation: Any = None,
        reconcile_error: Exception | None = None,
    ) -> None:
        self.agent_id = "coder-py"
        self.child_session_dir = str(
            root / "agents" / "coder-py_20260101_000000_abcdef12"
        )
        self.root_session_dir = str(root)
        self._alive = True
        self._alive_after_close = alive_after_close
        self._reconciliation = reconciliation
        self._reconcile_error = reconcile_error
        self.closed = 0

    @property
    def alive(self) -> bool:
        return self._alive

    async def close(self) -> None:
        self.closed += 1
        if not self._alive_after_close:
            self._alive = False

    def reconcile_death(self) -> Any:
        if self._reconcile_error is not None:
            raise self._reconcile_error
        return self._reconciliation


@pytest.mark.asyncio
async def test_teardown_of_a_still_live_attempt_reports_inconclusive(
    tmp_path: Path,
) -> None:
    attempt = _TeardownAttempt(tmp_path, alive_after_close=True)
    report = await teardown_spawned_attempt(
        attempt, attempt_id="attempt-live", wait_seconds=0.1, sleep=AsyncMock()
    )
    # The bounded wait expired with the child still alive: the teardown claims
    # no rollback, and nothing was retracted.
    assert report.outcome is ChildTeardownOutcome.INCONCLUSIVE
    assert report.settled is False
    assert report.retracted is False
    assert attempt.closed == 1
    assert "still alive" in (report.detail or "")


@pytest.mark.asyncio
async def test_teardown_retracts_the_records_a_dead_attempt_left(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sessions" / "root-session-1"
    child_dir = root / "agents" / "coder-py_20260101_000000_abcdef12"
    child_dir.mkdir(parents=True)
    store = ChildTranscriptStore(child_dir, root_session_dir=root)
    store.record_start(
        ChildStartMarker(
            agent_id="coder:py",
            session_id="child-session-1",
            run_id="turn-1",
            started_at="2026-01-01T00:00:00+00:00",
            attempt_id="attempt-dead",
        )
    )
    reconciliation = ChildTranscriptReconciler(root).reconcile(
        child_dir, agent_id="coder:py"
    )
    attempt = _TeardownAttempt(tmp_path, reconciliation=reconciliation)
    attempt.root_session_dir = str(root)
    attempt.child_session_dir = str(child_dir)
    report = await teardown_spawned_attempt(
        attempt, attempt_id="attempt-dead", sleep=AsyncMock()
    )
    assert report.outcome is ChildTeardownOutcome.TOMBSTONED
    assert report.settled is True
    assert report.retracted is True
    assert report.session_id == "child-session-1"
    # The retraction is durable and gates the reconciliation scan.
    assert store.adjudicate().outcome is ChildTranscriptOutcome.TOMBSTONED
    assert ChildTranscriptReconciler(root).scan() == []


@pytest.mark.asyncio
async def test_teardown_of_an_attempt_without_records_is_reaped(tmp_path: Path) -> None:
    attempt = _TeardownAttempt(
        tmp_path, reconcile_error=ChildTranscriptError("no transcript directory")
    )
    report = await teardown_spawned_attempt(
        attempt, attempt_id="attempt-empty", sleep=AsyncMock()
    )
    # Nothing was ever persisted under an attempt key, so the removal is the
    # whole settlement and no record can resurrect.
    assert report.outcome is ChildTeardownOutcome.REAPED
    assert report.settled is True
    assert report.retracted is False
