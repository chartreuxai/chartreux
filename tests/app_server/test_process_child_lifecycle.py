"""Process-child terminal lifecycle: settled results, closed rows, retention.

G1a pins the process-backed child host's lifecycle parity with the in-process
registry. A background child that finishes is reaped as a PROCESS, but its run
settles into three independent lifetimes: the host-observed terminal outcome
(a closed row), the result captured BEFORE the settle (available to the latest
and run-pinned reads, and to waits), and the retention policy in force
(``[subagents]`` metadata TTL/cap, stored-result count bound, retained-idle
TTL/cap, ``retain_agents`` default). On the pre-G1a base a non-retained
completion reaps the handle outright, so the row vanishes, the result is lost
to reads and waits, and an explicit ``retain`` argument is the only way to
retain a child.

Behavioral discrimination: no pin here touches the production files' internals
unless that internal IS the feature under test, so nothing fails to import on
the base. Sixteen of the seventeen pins fail there. Nine fail with the observed
lost-result/vanished-row behavior (``UnknownAgentError``, a missing row, or the
ignored retention default), four fail at the injected retention clock (the
time seam the TTL/cap/idle retirement features introduce), two fail on the
retention reaper/result store the settle itself creates, and one fails on the
release outcome of a row that no longer exists. The concurrent
release/settle pin is a robustness guard: its post-condition holds on both
trees, so it does not discriminate and is disclosed as such.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server._subagent_transcript import child_transcript_dir_name
from chartreux.app_server.agent_wire import (
    AGENT_TURN_ENDED_METHOD,
    AGENT_TURN_EVENT_METHOD,
    SUBAGENT_OPEN_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
)
from chartreux.app_server.child_agent import ChildAgentServer
from chartreux.app_server.child_agent_process import ProcessSubagentCompletionSource
from chartreux.app_server.process_bootstrap import (
    AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY,
)
from chartreux.app_server.root_agent import AgentChildHost
from chartreux.app_server.root_agent_process import AgentProcessTurnFailed
from chartreux.app_server.subagent_ports import (
    SubagentCompletion,
    child_spawn_spec,
    spawn_spec_to_wire,
)
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.models import SubagentsConfig
from chartreux.core.events import AssistantEvent, WaitingForInputEvent
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.subagents import (
    AgentAvailability,
    AgentClosedError,
    AgentEvictedError,
    AgentResultExpiredError,
    CancelOutcome,
    ReleaseAgentOutcome,
    RunStatus,
    RunStopReason,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    WaitStatus,
)
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_COMPLETED = "completed"
_FAILED = "failed"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _ManualClock:
    """A monotone clock the retention TTL/cap pins advance by hand."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ScriptedSource:
    """A child completion source serving one scripted frame list per run.

    A retained child's process serves every re-task through the same source,
    so each ``stream`` call consumes the next scripted run.
    """

    def __init__(self, *runs: list[Any]) -> None:
        self._runs = list(runs)
        self._index = 0

    def stream(self, request: Any) -> Any:
        index = self._index
        self._index += 1
        frames = self._runs[index] if index < len(self._runs) else []
        return self._iter(frames)

    async def _iter(self, frames: list[Any]) -> AsyncGenerator[Any, None]:
        for frame in frames:
            yield frame


class _RaisingSource:
    """A child completion source whose transport fails mid-run."""

    def stream(self, request: Any) -> Any:
        return self._iter()

    async def _iter(self) -> AsyncGenerator[Any, None]:
        raise RuntimeError("child transport died")
        yield  # pragma: no cover - unreachable, keeps this a generator


class _BlockingSource:
    """A child completion source whose run blocks until its frames are posted."""

    def __init__(self) -> None:
        self._release = asyncio.Event()
        self._frames: list[Any] = []

    def complete(self, frames: list[Any]) -> None:
        self._frames = frames
        self._release.set()

    def stream(self, request: Any) -> Any:
        return self._iter()

    async def _iter(self) -> AsyncGenerator[Any, None]:
        await self._release.wait()
        for frame in self._frames:
            yield frame


class _SpawnedChild:
    """One spawned child double: a completion source plus a close counter."""

    def __init__(
        self, agent_id: str, session_id: str, source: Any, wire_client: Any = None
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.child_session_dir = None
        self.completion_source = source
        self.wire_client = wire_client
        self.closes = 0
        self.alive = True

    async def close(self) -> None:
        self.closes += 1
        self.alive = False


class _TelemetryClient:
    """A spawned child's wire client answering one session snapshot (G3).

    Only the pieces the host reads are present, so the pin holds the host to
    the values the CHILD reports: the snapshot carries the committed model
    and the runtime stats exactly as the child's own projection would, and --
    when the pin asks for it -- the config view whose effective-thinking
    estimate the child's own ``/status`` renders (H1).
    """

    def __init__(
        self,
        *,
        model_alias: str | None = None,
        committed: Any = None,
        context_tokens: int = 0,
        context_window: int | None = None,
        context_threshold: Any = None,
        raises: BaseException | None = None,
        thinking: str | None = None,
        thinking_known: bool = True,
    ) -> None:
        self._model_alias = model_alias
        self._committed = committed
        self._context_tokens = context_tokens
        self._context_window = context_window
        self._context_threshold = context_threshold
        self._raises = raises
        self._thinking = thinking
        self._thinking_known = thinking_known
        self.reads = 0
        # G5: a process child is published only when its ready frame announced
        # the mandatory live-child-authority capability.
        self.advertised_capabilities = frozenset({
            AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY
        })

    async def snapshot(self, *, include_history: bool = True) -> Any:
        self.reads += 1
        assert include_history is False
        if self._raises is not None:
            raise self._raises
        return self.payload()

    def payload(self) -> Any:
        """The snapshot the child answers with; unread, so a gate can hold it."""
        config = SimpleNamespace(
            effective_thinking=(
                SimpleNamespace(
                    requested=self._thinking,
                    semantic=self._thinking,
                    known=self._thinking_known,
                )
                if self._thinking is not None
                else None
            ),
            active_model=SimpleNamespace(
                thinking=self._thinking, auto_compact_threshold=self._context_threshold
            ),
        )
        return SimpleNamespace(
            committed_model=self._committed,
            state=SimpleNamespace(session=SimpleNamespace(model=self._model_alias)),
            runtime=SimpleNamespace(
                stats=SimpleNamespace(context_tokens=self._context_tokens),
                context_window=self._context_window,
                config=config,
            ),
        )


class _GatedTelemetryClient(_TelemetryClient):
    """A child client whose snapshot read stays in flight until released (H1).

    The gate is what makes the observation fence observable: the run id can
    move (a re-task) or the handle can close while the read suspends, which is
    exactly the window a late snapshot must not write through.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reading = asyncio.Event()
        self.release = asyncio.Event()

    async def snapshot(self, *, include_history: bool = True) -> Any:
        self.reads += 1
        self.reading.set()
        await self.release.wait()
        if self._raises is not None:
            raise self._raises
        return self.payload()


class _HeldSource:
    """A run that relays its first frames, then holds until released.

    The hold is the window a RUNNING row is inspected in: the frames already
    relayed are the whole of what the host may know at that moment, and the
    tail (a terminal carrier) arrives only when the pin lets it. A retained
    child serves every re-task through the SAME source, so the source takes
    one ``(frames, tail)`` pair per run, in order.
    """

    def __init__(self, *runs: tuple[Sequence[Any], Sequence[Any]]) -> None:
        self._runs = [([*frames], [*tail]) for frames, tail in runs]
        self.releases = [asyncio.Event() for _ in self._runs]
        self._index = 0

    @property
    def release(self) -> asyncio.Event:
        """The gate of the first (or only) run."""
        return self.releases[0]

    def stream(self, request: Any) -> Any:
        return self._iter()

    async def _iter(self) -> AsyncGenerator[Any, None]:
        index = self._index
        self._index += 1
        frames, tail = self._runs[index] if index < len(self._runs) else ([], [])
        for frame in frames:
            yield frame
        await self.releases[index].wait()
        for frame in tail:
            yield frame


class _RecordingSpawn:
    """The spawn adapter these pins launch children through."""

    def __init__(self, *sources: Any, clients: Sequence[Any] = ()) -> None:
        self._sources = list(sources)
        self._clients = list(clients)
        self.spawned: list[_SpawnedChild] = []

    async def spawn(self, spec: Any) -> _SpawnedChild:
        source = self._sources.pop(0) if self._sources else _ScriptedSource([])
        client = self._clients.pop(0) if self._clients else None
        child = _SpawnedChild(
            agent_id=spec.agent_id,
            session_id=f"child-session-{len(self.spawned) + 1}",
            source=source,
            wire_client=client,
        )
        self.spawned.append(child)
        return child


def _root(tmp_path: Path, **subagents: Any) -> Any:
    """The loop shape a process-owned root opens, with retention overrides."""
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
        )
    )
    if subagents:
        config = config.model_copy(update={"subagents": SubagentsConfig(**subagents)})
    return build_test_agent_loop(
        config=config, backend=FakeBackend([[mock_llm_chunk(content="ack")]])
    )


def _ctx(loop: Any, call_id: str) -> InvokeContext:
    return InvokeContext(tool_call_id=call_id, session_id=loop.session_id)


def _completed_frames(response: str = "work", *, turns: int | None = 1) -> list[Any]:
    """One completed run: its text frame and its terminal carrier.

    ``turns`` rides the carrier because that is the only turn source the host
    accepts (G3); ``None`` models a child that reports no count.
    """
    return [
        AssistantEvent(content=response),
        SubagentCompletion(outcome=_COMPLETED, turns_used=turns),
    ]


def _rows(host: AgentChildHost) -> dict[str, Any]:
    return {row.agent_id: row for row in host.inventory_snapshot().agents}


async def _start_background(
    host: AgentChildHost, loop: Any, *, call_id: str, retained: bool = False
) -> tuple[Any, TaskResult]:
    """Publish and launch one background child without awaiting its settle."""
    args = TaskArgs(task="Inspect the project", identity="coder")
    ctx = _ctx(loop, call_id)
    handle = await host.publish(
        args, ctx, attempt_id=f"attempt-{call_id}", retained=retained
    )
    ack = host.admit_background(handle, args, ctx)
    assert handle.task is not None
    return handle, ack


async def _settle_background(
    host: AgentChildHost,
    loop: Any,
    *,
    call_id: str,
    retained: bool = False,
    source: Any = None,
) -> tuple[Any, TaskResult]:
    """Publish, launch, and settle one background child; answer (handle, ack)."""
    handle, ack = await _start_background(
        host, loop, call_id=call_id, retained=retained
    )
    if source is not None:
        source.complete(_completed_frames())
    await asyncio.gather(handle.task, return_exceptions=True)
    return handle, ack


async def _consume_runner(
    host: AgentChildHost, args: TaskArgs, ctx: InvokeContext
) -> list[Any]:
    return [item async for item in host.runner.run(args, ctx)]


async def _launch(
    host: AgentChildHost, loop: Any, call_id: str, *, retain: bool | None
) -> Any:
    """Drive one background launch through the runner and answer its handle."""
    args = TaskArgs(
        task="Background work", identity="coder", background=True, retain=retain
    )
    events = await _consume_runner(host, args, _ctx(loop, call_id))
    handle = host._manageable(events[-1].agent_id)
    assert handle is not None
    return handle


# ---------------------------------------------------------------------------
# Non-retained completion: process reaped, row kept, result retrievable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_non_retained_completion_keeps_the_result_and_a_closed_row(
    tmp_path: Path,
) -> None:
    """A finished non-retained child is reaped but keeps its result and row."""
    loop = _root(tmp_path)
    spawn = _RecordingSpawn(_ScriptedSource(_completed_frames("finished work")))
    host = AgentChildHost(loop, spawn=spawn)
    try:
        handle, ack = await _settle_background(
            host, loop, call_id="task-1", retained=False
        )
        snapshot = host.inventory_snapshot()
        row = _rows(host)[handle.agent_id]
        slots_after = set(host._admission.slots)
        latest = await host.get_agent_result(handle.agent_id)
        pinned = await host.get_agent_result(handle.agent_id, ack.run_id)
        waited = await host.wait_for_agent(handle.agent_id, ack.run_id, timeout=1)
    finally:
        await host.close()
        await loop.aclose()

    # The process is gone and its admission slot is free.
    assert spawn.spawned[0].closes == 1
    assert handle.closed is True
    assert slots_after == set()
    # The closed row is the honest host-observed outcome.
    assert row.availability == AgentAvailability.CLOSED.value
    assert row.last_run_status == RunStatus.COMPLETED.value
    assert row.turns_used == 1
    assert row.latest_run_id == ack.run_id
    assert row.result_expired is False
    assert snapshot.revision >= 1
    # The exact result survives the process: latest, pinned and waited agree.
    assert latest is not None and pinned is not None
    assert latest == pinned
    assert latest.response == "finished work"
    assert latest.completed is True
    assert latest.agent_id == handle.agent_id
    assert latest.run_id == ack.run_id
    assert waited.status is WaitStatus.COMPLETED
    assert waited.result == latest


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_failure_and_cancel_dispositions_project_and_keep_partial_results(
    tmp_path: Path,
) -> None:
    """A failed and a cancelled child each close with their honest outcome."""
    loop = _root(tmp_path)
    failed_source = _ScriptedSource([
        AssistantEvent(content="partial work"),
        SubagentCompletion(outcome=_FAILED, error="child exploded"),
    ])
    cancelled_source = _BlockingSource()
    spawn = _RecordingSpawn(failed_source, cancelled_source, _RaisingSource())
    host = AgentChildHost(loop, spawn=spawn)
    try:
        failed, failed_ack = await _settle_background(host, loop, call_id="task-failed")
        cancelled, cancelled_ack = await _start_background(
            host, loop, call_id="task-cancelled"
        )
        await asyncio.sleep(0.05)
        stopped = await host.cancel_run(
            cancelled.agent_id,
            cancelled_ack.run_id,
            reason=RunStopReason.ORCHESTRATOR_CANCELLED,
            requester_session_id=loop.session_id,
        )
        await asyncio.gather(cancelled.task, return_exceptions=True)
        raised, raised_ack = await _settle_background(
            host, loop, call_id="task-transport"
        )
        rows = _rows(host)
        failed_result = await host.get_agent_result(failed.agent_id, failed_ack.run_id)
        cancelled_result = await host.get_agent_result(
            cancelled.agent_id, cancelled_ack.run_id
        )
        raised_result = await host.get_agent_result(raised.agent_id, raised_ack.run_id)
    finally:
        await host.close()
        await loop.aclose()

    assert stopped.outcome is CancelOutcome.STOP_REQUESTED
    assert rows[failed.agent_id].availability == AgentAvailability.CLOSED.value
    assert rows[failed.agent_id].last_run_status == RunStatus.FAILED.value
    assert failed_result is not None
    assert failed_result.completed is False
    assert "child exploded" in failed_result.response
    assert rows[cancelled.agent_id].last_run_status == RunStatus.CANCELLED.value
    # The wire projection carries the public enum, so the two are compared by
    # value rather than identity across the two enum classes.
    assert (
        rows[cancelled.agent_id].stop_reason
        == RunStopReason.ORCHESTRATOR_CANCELLED.value
    )
    assert cancelled_result is not None
    assert cancelled_result.completed is False
    assert cancelled_result.stop_reason is RunStopReason.ORCHESTRATOR_CANCELLED
    assert rows[raised.agent_id].last_run_status == RunStatus.FAILED.value
    assert raised_result is not None
    assert raised_result.completed is False


# ---------------------------------------------------------------------------
# Retention: metadata TTL/cap, stored-result bound, expired vs unknown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_zero_metadata_ttl_keeps_no_row_but_the_result_survives(
    tmp_path: Path,
) -> None:
    """``tombstone_ttl_seconds=0`` stores no row; the result stays retrievable."""
    loop = _root(tmp_path, tombstone_ttl_seconds=0)
    spawn = _RecordingSpawn(_ScriptedSource(_completed_frames("kept")))
    host = AgentChildHost(loop, spawn=spawn)
    try:
        handle, ack = await _settle_background(host, loop, call_id="task-1")
        inventory = host.inventory_snapshot().agents
        latest = await host.get_agent_result(handle.agent_id)
        pinned = await host.get_agent_result(handle.agent_id, ack.run_id)
        released = await host.release_agent(handle.agent_id)
        results_after_release = dict(host._results)
        with pytest.raises(UnknownAgentError):
            await host.get_agent_result(handle.agent_id, ack.run_id)
    finally:
        await host.close()
        await loop.aclose()

    # No metadata at all, but both reads resolve from the stored result.
    assert inventory == []
    assert latest is not None and pinned is not None
    assert latest.response == "kept"
    # Releasing a metadata-less agent whose result outlived the row still
    # purges that result (the in-process result purge keeps this shape).
    assert released is ReleaseAgentOutcome.RELEASED
    assert results_after_release == {}


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_terminal_metadata_expiry_removes_the_row_but_not_the_result(
    tmp_path: Path,
) -> None:
    """Expired metadata drops the row; the result outlives its own row."""
    clock = _ManualClock()
    loop = _root(tmp_path, tombstone_ttl_seconds=10)
    spawn = _RecordingSpawn(_ScriptedSource(_completed_frames("outlives")))
    host = AgentChildHost(loop, spawn=spawn, clock=clock)
    try:
        handle, ack = await _settle_background(host, loop, call_id="task-1")
        settled_revision = host.inventory_snapshot().revision
        assert handle.agent_id in _rows(host)
        clock.advance(11)
        pruned = host._expire_terminal_metadata()
        after = host.inventory_snapshot()
        pinned = await host.get_agent_result(handle.agent_id, ack.run_id)
        latest = await host.get_agent_result(handle.agent_id)
    finally:
        await host.close()
        await loop.aclose()

    assert pruned == [handle.agent_id]
    assert after.agents == []
    assert after.revision > settled_revision
    assert pinned is not None and latest is not None
    assert pinned.response == "outlives"
    assert latest.run_id == ack.run_id


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_terminal_metadata_cap_evicts_oldest_first(tmp_path: Path) -> None:
    """The metadata cap drops the oldest closed row, keeping its result."""
    clock = _ManualClock()
    loop = _root(tmp_path, tombstone_ttl_seconds=100, max_tombstones=2)
    spawn = _RecordingSpawn(
        _ScriptedSource(_completed_frames("first")),
        _ScriptedSource(_completed_frames("second")),
        _ScriptedSource(_completed_frames("third")),
    )
    host = AgentChildHost(loop, spawn=spawn, clock=clock)
    try:
        first, first_ack = await _settle_background(host, loop, call_id="task-1")
        clock.advance(1)
        second, _ = await _settle_background(host, loop, call_id="task-2")
        clock.advance(1)
        third, _ = await _settle_background(host, loop, call_id="task-3")
        rows = _rows(host)
        first_result = await host.get_agent_result(first.agent_id, first_ack.run_id)
    finally:
        await host.close()
        await loop.aclose()

    assert set(rows) == {second.agent_id, third.agent_id}
    # The capped row's result is untouched by metadata eviction.
    assert first_result is not None
    assert first_result.response == "first"


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_stored_results_are_count_bounded_and_expired_differs_from_unknown(
    tmp_path: Path,
) -> None:
    """The 33rd result expires the oldest; expired answers are not unknown."""
    loop = _root(tmp_path)
    spawn = _RecordingSpawn(*[
        _ScriptedSource(_completed_frames(f"result-{index}")) for index in range(33)
    ])
    host = AgentChildHost(loop, spawn=spawn)
    handles: list[tuple[Any, TaskResult]] = []
    try:
        for index in range(33):
            handles.append(
                await _settle_background(host, loop, call_id=f"task-{index}")
            )
        oldest, oldest_ack = handles[0]
        newest, newest_ack = handles[-1]
        stored_count = len(host._results)
        expired_markers = set(host._expired_results)
        with pytest.raises(AgentResultExpiredError):
            await host.get_agent_result(oldest.agent_id, oldest_ack.run_id)
        with pytest.raises(UnknownAgentError):
            await host.get_agent_result("agent-never-seen", "run-never")
        newest_result = await host.get_agent_result(newest.agent_id, newest_ack.run_id)
    finally:
        await host.close()
        await loop.aclose()

    assert len(handles) == 33
    # The oldest result left the store under the count bound ...
    assert stored_count == 32
    assert (oldest.agent_id, oldest_ack.run_id) in expired_markers
    # ... while every newer one stays retrievable.
    assert newest_result is not None
    assert newest_result.response == "result-32"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reuse_retrieves_the_earlier_run_after_a_re_task(tmp_path: Path) -> None:
    """A retained handle's earlier result stays retrievable after re-task."""
    loop = _root(tmp_path)
    spawn = _RecordingSpawn(
        _ScriptedSource(_completed_frames("run one"), _completed_frames("run two"))
    )
    host = AgentChildHost(loop, spawn=spawn)
    try:
        handle, first_ack = await _settle_background(
            host, loop, call_id="task-1", retained=True
        )
        idle_row = _rows(host)[handle.agent_id]
        args = TaskArgs(
            task="Continue the work", agent_id=handle.agent_id, identity="coder"
        )
        ctx = _ctx(loop, "task-2")
        reused = await host.publish(
            args, ctx, attempt_id="attempt-task-2", retained=True, reuse=handle
        )
        second_ack = host.admit_background(reused, args, ctx)
        await asyncio.gather(handle.task, return_exceptions=True)
        first = await host.get_agent_result(handle.agent_id, first_ack.run_id)
        second = await host.get_agent_result(handle.agent_id, second_ack.run_id)
        latest = await host.get_agent_result(handle.agent_id)
    finally:
        await host.close()
        await loop.aclose()

    assert reused is handle
    assert idle_row.availability == "idle"
    assert first is not None and second is not None and latest is not None
    assert first.response == "run one"
    assert first.run_id == first_ack.run_id
    assert second.response == "run two"
    assert second.run_id == second_ack.run_id
    assert latest == second


# ---------------------------------------------------------------------------
# Explicit release purges both lifetimes; closed handles refuse re-task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_explicit_release_purges_metadata_and_result(tmp_path: Path) -> None:
    """Releasing a closed row purges its metadata and its stored result."""
    loop = _root(tmp_path)
    spawn = _RecordingSpawn(_ScriptedSource(_completed_frames("purge me")))
    host = AgentChildHost(loop, spawn=spawn)
    try:
        handle, ack = await _settle_background(host, loop, call_id="task-1")
        before = host.inventory_snapshot().revision
        guarded_stale = await host.release_guarded(
            handle.agent_id, expected_run_id="run-never"
        )
        released = await host.release_guarded(handle.agent_id, ack.run_id)
        after = host.inventory_snapshot()
        with pytest.raises(UnknownAgentError):
            await host.release_agent(handle.agent_id)
        # A sibling of ::class:`AgentResultExpiredError`, so this read proves
        # the release dropped the expiry marker too, not just the row.
        with pytest.raises(UnknownAgentError):
            await host.get_agent_result(handle.agent_id, ack.run_id)
    finally:
        await host.close()
        await loop.aclose()

    assert guarded_stale is ReleaseAgentOutcome.CONFLICT
    assert released is ReleaseAgentOutcome.EVICTED
    assert after.revision > before
    assert after.agents == []


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_closed_handle_refuses_a_re_task_with_the_typed_closed_error(
    tmp_path: Path,
) -> None:
    """Re-tasking a closed handle raises the registry's typed closed error."""
    loop = _root(tmp_path)
    spawn = _RecordingSpawn(_ScriptedSource(_completed_frames()))
    host = AgentChildHost(loop, spawn=spawn)
    try:
        handle, _ack = await _settle_background(host, loop, call_id="task-1")
        with pytest.raises(AgentClosedError) as closed:
            await _consume_runner(
                host,
                TaskArgs(
                    task="Continue",
                    agent_id=handle.agent_id,
                    identity="coder",
                    background=True,
                ),
                _ctx(loop, "task-2"),
            )
        with pytest.raises(UnknownAgentError):
            await _consume_runner(
                host,
                TaskArgs(
                    task="Continue",
                    agent_id="agent-never-seen",
                    identity="coder",
                    background=True,
                ),
                _ctx(loop, "task-3"),
            )
        spawned_count = len(spawn.spawned)
    finally:
        await host.close()
        await loop.aclose()

    assert "closed" in str(closed.value)
    assert spawned_count == 1


# ---------------------------------------------------------------------------
# Retention configuration: retain_agents default on both spawn routes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_runner_takes_the_retain_agents_default_for_a_fresh_spawn(
    tmp_path: Path,
) -> None:
    """A fresh process child takes the config default unless retain overrides."""
    loop = _root(tmp_path, retain_agents=True)
    spawn = _RecordingSpawn(_BlockingSource(), _BlockingSource(), _BlockingSource())
    host = AgentChildHost(loop, spawn=spawn)
    try:
        default = await _launch(host, loop, "task-default", retain=None)
        explicit_false = await _launch(host, loop, "task-false", retain=False)
        explicit_true = await _launch(host, loop, "task-true", retain=True)
    finally:
        await host.close()
        await loop.aclose()

    assert default.retained is True
    assert explicit_false.retained is False
    assert explicit_true.retained is True


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_runner_default_is_false_when_the_config_default_is_false(
    tmp_path: Path,
) -> None:
    """With ``retain_agents=false`` a fresh launch is not retained.

    The default-false control for the pin above: the child is not retained,
    and its completion still settles into a closed row with a retrievable
    result -- the pre-G1a base loses both.
    """
    loop = _root(tmp_path, retain_agents=False)
    source = _BlockingSource()
    spawn = _RecordingSpawn(source)
    host = AgentChildHost(loop, spawn=spawn)
    try:
        default = await _launch(host, loop, "task-default", retain=None)
        assert default.task is not None
        run_id = default.run_id
        assert run_id is not None
        source.complete(_completed_frames("not retained, still retrievable"))
        await asyncio.gather(default.task, return_exceptions=True)
        row = _rows(host)[default.agent_id]
        result = await host.get_agent_result(default.agent_id, run_id)
    finally:
        await host.close()
        await loop.aclose()

    assert default.retained is False
    assert row.availability == AgentAvailability.CLOSED.value
    assert result is not None
    assert result.response == "not retained, still retrievable"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_retain_false_on_reuse_closes_the_retained_child(tmp_path: Path) -> None:
    """An explicit ``retain=false`` on reuse makes the continuation final."""
    loop = _root(tmp_path)
    spawn = _RecordingSpawn(
        _ScriptedSource(_completed_frames("run one"), _completed_frames("run two"))
    )
    host = AgentChildHost(loop, spawn=spawn)
    try:
        handle, first_ack = await _settle_background(
            host, loop, call_id="task-1", retained=True
        )
        was_retained = handle.retained
        events = await _consume_runner(
            host,
            TaskArgs(
                task="Final continuation",
                agent_id=handle.agent_id,
                identity="coder",
                background=True,
                retain=False,
            ),
            _ctx(loop, "task-2"),
        )
        second_ack = events[-1]
        await asyncio.gather(handle.task, return_exceptions=True)
        row = _rows(host)[handle.agent_id]
        first = await host.get_agent_result(handle.agent_id, first_ack.run_id)
        second = await host.get_agent_result(handle.agent_id, second_ack.run_id)
        children = host.children()
    finally:
        await host.close()
        await loop.aclose()

    assert was_retained is True
    assert handle.retained is False
    assert handle.closed is True
    assert spawn.spawned[0].closes == 1
    assert len(spawn.spawned) == 1
    assert children == ()
    assert row.availability == AgentAvailability.CLOSED.value
    assert row.last_run_status == RunStatus.COMPLETED.value
    assert first is not None and second is not None
    assert first.response == "run one"
    assert second.response == "run two"


# ---------------------------------------------------------------------------
# Retained-idle retirement: idle TTL and idle cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_retained_idle_cap_retires_oldest_first_and_zero_ttl_disables_ttl(
    tmp_path: Path,
) -> None:
    """The idle cap retires oldest first; ``idle_ttl_seconds=0`` disables TTL."""
    clock = _ManualClock()
    loop = _root(tmp_path, idle_ttl_seconds=0, max_idle_agents=2)
    spawn = _RecordingSpawn(*[
        _ScriptedSource(_completed_frames(f"run-{i}")) for i in range(3)
    ])
    host = AgentChildHost(loop, spawn=spawn, clock=clock)
    try:
        first, _ = await _settle_background(host, loop, call_id="task-1", retained=True)
        clock.advance(1)
        second, _ = await _settle_background(
            host, loop, call_id="task-2", retained=True
        )
        clock.advance(1)
        third, _ = await _settle_background(host, loop, call_id="task-3", retained=True)
        rows = _rows(host)
        # A far future clock cannot evict another child under a zero TTL.
        clock.advance(10_000)
        await host._retire_idle_retained()
        after_ttl_sweep = _rows(host)
        evicted_retask = None
        try:
            await _consume_runner(
                host,
                TaskArgs(
                    task="Continue",
                    agent_id=first.agent_id,
                    identity="coder",
                    background=True,
                ),
                _ctx(loop, "task-4"),
            )
        except AgentEvictedError as exc:
            evicted_retask = exc
    finally:
        await host.close()
        await loop.aclose()

    assert rows[first.agent_id].availability == AgentAvailability.EVICTED.value
    assert rows[second.agent_id].availability == "idle"
    assert rows[third.agent_id].availability == "idle"
    assert set(after_ttl_sweep) == set(rows)
    assert spawn.spawned[0].closes == 1
    assert evicted_retask is not None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_retained_idle_ttl_retires_and_arms_the_reaper(tmp_path: Path) -> None:
    """The idle TTL retires an idle retained child and the reaper gets armed."""
    clock = _ManualClock()
    loop = _root(tmp_path, idle_ttl_seconds=5, max_idle_agents=0)
    spawn = _RecordingSpawn(_ScriptedSource(_completed_frames()))
    host = AgentChildHost(loop, spawn=spawn, clock=clock)
    try:
        handle, _ = await _settle_background(
            host, loop, call_id="task-1", retained=True
        )
        reaper_armed = host._retention_reaper is not None
        idle_row = _rows(host)[handle.agent_id]
        clock.advance(6)
        await host._retire_idle_retained()
        row = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert reaper_armed
    assert idle_row.availability == "idle"
    assert row.availability == AgentAvailability.EVICTED.value
    assert handle.closed is True
    assert spawn.spawned[0].closes == 1


# ---------------------------------------------------------------------------
# Revision ordering, concurrency, and teardown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_closed_row_appears_and_leaves_with_strictly_increasing_revisions(
    tmp_path: Path,
) -> None:
    """Only the settle, then release/expiry, change the row -- each on a new revision."""
    clock = _ManualClock()
    loop = _root(tmp_path, tombstone_ttl_seconds=10)
    spawn = _RecordingSpawn(
        _ScriptedSource(_completed_frames()), _ScriptedSource(_completed_frames())
    )
    host = AgentChildHost(loop, spawn=spawn, clock=clock)
    try:
        empty = host.inventory_snapshot()
        released_handle, _ = await _settle_background(host, loop, call_id="task-1")
        settled = host.inventory_snapshot()
        expired_handle, _ = await _settle_background(host, loop, call_id="task-2")
        settled_two = host.inventory_snapshot()
        await host.release_agent(released_handle.agent_id)
        after_release = host.inventory_snapshot()
        clock.advance(11)
        host._expire_terminal_metadata()
        after_expiry = host.inventory_snapshot()
    finally:
        await host.close()
        await loop.aclose()

    assert empty.agents == []
    assert released_handle.agent_id in {row.agent_id for row in settled.agents}
    assert settled.revision > empty.revision
    assert settled_two.revision > settled.revision
    assert {row.agent_id for row in after_release.agents} == {expired_handle.agent_id}
    assert after_release.revision > settled_two.revision
    assert after_expiry.agents == []
    assert after_expiry.revision > after_release.revision


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_concurrent_release_and_settle_close_the_process_once(
    tmp_path: Path,
) -> None:
    """A release racing a settle reaps once and leaves nothing behind."""
    loop = _root(tmp_path)
    source = _BlockingSource()
    spawn = _RecordingSpawn(source)
    host = AgentChildHost(loop, spawn=spawn)
    try:
        handle, _ack = await _start_background(host, loop, call_id="task-1")
        await asyncio.sleep(0.05)
        release = asyncio.create_task(host.release_agent(handle.agent_id))
        source.complete(_completed_frames())
        released = await release
        await asyncio.gather(handle.task, return_exceptions=True)
        with pytest.raises(UnknownAgentError):
            await host.get_agent_result(handle.agent_id)
        rows = _rows(host)
    finally:
        await host.close()
        await loop.aclose()

    assert released in {ReleaseAgentOutcome.RELEASED, ReleaseAgentOutcome.EVICTED}
    assert rows == {}
    assert spawn.spawned[0].closes == 1
    assert handle.closed is True


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_host_close_tears_down_live_closed_and_retained_children(
    tmp_path: Path,
) -> None:
    """Teardown reaps every owned process and drops every retained lifetime."""
    loop = _root(tmp_path)
    spawn = _RecordingSpawn(
        _ScriptedSource(_completed_frames()),
        _ScriptedSource(_completed_frames()),
        _BlockingSource(),
    )
    host = AgentChildHost(loop, spawn=spawn)
    try:
        closed, _ = await _settle_background(host, loop, call_id="task-closed")
        retained, _ = await _settle_background(
            host, loop, call_id="task-retained", retained=True
        )
        running, _ = await _start_background(host, loop, call_id="task-running")
        assert running.task is not None
        await asyncio.sleep(0.05)
        await host.close()
        remaining = host.inventory_snapshot().agents
        assert host._tasks == set()
        assert host._retention_reaper is None
        assert all(child.closes == 1 for child in spawn.spawned)
        assert closed.closed and retained.closed and running.closed
    finally:
        await loop.aclose()

    assert remaining == []


# ---------------------------------------------------------------------------
# G1b: the settled-run completion ledger the parent wake is fed from
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_background_settle_records_one_completion_and_a_foreground_none(
    tmp_path: Path,
) -> None:
    """G1b: only a background run's settle is a completion.

    A foreground run's result leaves with its own tool response, so nobody has
    to be told it finished; a background run's outcome is only visible here, so
    its settle records one typed completion and publishes it. The ledger holds
    exactly what was published -- the recovery read replays it verbatim.
    """
    loop = _root(tmp_path)
    published: list[Any] = []
    spawn = _RecordingSpawn(
        _ScriptedSource(_completed_frames("background work")),
        _ScriptedSource(_completed_frames("foreground work")),
    )
    host = AgentChildHost(loop, spawn=spawn)
    host.bind_completion_publisher(published.append)
    try:
        handle, _ = await _settle_background(host, loop, call_id="task-bg")
        assert [item.run_id for item in published] == [handle.run_id]
        completion = published[0]
        assert completion.session_id == loop.session_id
        assert completion.agent_id == handle.agent_id
        assert completion.status == RunStatus.COMPLETED.value
        assert completion.stop_reason is None
        assert completion.result_available is True
        assert (
            completion.name
            == {row.agent_id: row.name for row in host.inventory_snapshot().agents}[
                handle.agent_id
            ]
        )
        assert host._run_completion_ledger() == tuple(published)

        await _consume_runner(
            host,
            TaskArgs(task="Foreground work", identity="coder"),
            _ctx(loop, "task-fg"),
        )
        assert [item.run_id for item in published] == [handle.run_id]
        assert host._run_completion_ledger() == tuple(published)
    finally:
        await host.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_explicit_release_purges_the_completion_ledger(tmp_path: Path) -> None:
    """G1b: a release removes the resolution, so it removes the wake source too.

    The ledger describes exactly the results a release purges; keeping an entry
    would let a later sweep wake the parent for a run that is already unknown.
    """
    loop = _root(tmp_path)
    host = AgentChildHost(
        loop, spawn=_RecordingSpawn(_ScriptedSource(_completed_frames()))
    )
    try:
        handle, _ = await _settle_background(
            host, loop, call_id="task-release", retained=True
        )
        assert [item.run_id for item in host._run_completion_ledger()] == [
            handle.run_id
        ]
        outcome = await host.release_agent(handle.agent_id)
        assert outcome is ReleaseAgentOutcome.RELEASED
        assert host._run_completion_ledger() == ()
        with pytest.raises(UnknownAgentError):
            await host.get_agent_result(handle.agent_id, handle.run_id)
    finally:
        await host.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_completion_ledger_is_count_bounded_oldest_first(
    tmp_path: Path,
) -> None:
    """G1b: the ledger is the carrier's backstop, not a history.

    It keeps a bounded number of settled runs, oldest evicted first, so a long
    session's ledger cannot grow without bound; an evicted run's result remains
    answerable by the result store until that bound evicts it too.
    """
    loop = _root(tmp_path)
    spawn = _RecordingSpawn(*[_ScriptedSource([]) for _ in range(35)])
    host = AgentChildHost(loop, spawn=spawn)
    try:
        runs: list[str] = []
        for index in range(35):
            handle, _ = await _settle_background(host, loop, call_id=f"task-{index}")
            assert handle.run_id is not None
            runs.append(handle.run_id)
        ledger = host._run_completion_ledger()
        assert len(ledger) == 32
        kept = {item.run_id for item in ledger}
        assert kept == set(runs[-32:])
        assert runs[0] not in kept
        assert all(item.status == RunStatus.FAILED.value for item in ledger)
    finally:
        await host.close()
        await loop.aclose()


# ---------------------------------------------------------------------------
# G3: host-observed telemetry -- timing, task summaries, turns, model/context
#
# Every pin here fails on the pre-G3 base (4c062176): the handle recorded no
# run stamps, no task summary, no child-reported turn count and no child
# snapshot, so the row's run_elapsed_seconds / idle_seconds /
# initial_task_summary / current_task_summary / effective_model /
# base_model / active_provider / context_tokens / context_window were all
# unset and turns_used counted AssistantEvent frames.
# ---------------------------------------------------------------------------


def _committed_alpha() -> CommittedModelIdentity:
    return CommittedModelIdentity(
        base_model="alpha-model",
        provider="alpha",
        wire_name="alpha-wire",
        catalog_revision="test-fixture",
        thinking="medium",
    )


async def _await_run_start(handle: Any) -> None:
    """Let the background run reach its first suspension (stamps recorded)."""
    for _ in range(200):
        if handle.run_started_at is not None:
            return
        await asyncio.sleep(0)
    raise AssertionError("the background run never started")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_an_active_run_reports_its_elapsed_time_and_task_summary(
    tmp_path: Path,
) -> None:
    """An active row measures its own run and names the task it is running.

    The elapsed value is the host's monotonic stamp difference, so successive
    snapshots of the SAME live run report a growing duration; the task summary
    is the launch's own task, normalized exactly as the in-process record
    normalizes it. This fixture's child answers no session snapshot and relays
    no interim count, so its live row honestly reports no turn count: H1's own
    pins cover the running row that DOES report one, and the terminal carrier
    remains the settle's answer.
    """
    clock = _ManualClock()
    loop = _root(tmp_path)
    source = _BlockingSource()
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source), clock=clock)
    args = TaskArgs(task="  Inspect   the project\nlayout  ", identity="coder")
    ctx = _ctx(loop, "task-1")
    try:
        handle = await host.publish(
            args, ctx, attempt_id="attempt-telemetry-1", retained=False
        )
        host.admit_background(handle, args, ctx)
        await _await_run_start(handle)
        started_row = _rows(host)[handle.agent_id]
        clock.advance(3.5)
        later_row = _rows(host)[handle.agent_id]
        source.complete(_completed_frames())
        assert handle.task is not None
        await asyncio.gather(handle.task, return_exceptions=True)
        closed_row = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert started_row.availability == RunStatus.RUNNING.value
    assert started_row.run_elapsed_seconds == 0.0
    assert started_row.idle_seconds is None
    assert later_row.run_elapsed_seconds == 3.5
    assert started_row.initial_task_summary == "Inspect the project layout"
    assert started_row.current_task_summary == "Inspect the project layout"
    assert started_row.turns_used is None
    # The settle freezes the same measure, so a closed row cannot drift.
    assert closed_row.availability == AgentAvailability.CLOSED.value
    assert closed_row.run_elapsed_seconds == 3.5
    assert closed_row.turns_used == 1


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_retained_idle_row_measures_idle_and_an_evicted_row_freezes_it(
    tmp_path: Path,
) -> None:
    """A retained-idle row reports its idle segment; a retirement keeps it."""
    clock = _ManualClock()
    loop = _root(tmp_path, idle_ttl_seconds=60, max_idle_agents=0)
    host = AgentChildHost(
        loop, spawn=_RecordingSpawn(_ScriptedSource(_completed_frames())), clock=clock
    )
    try:
        handle, _ack = await _settle_background(
            host, loop, call_id="task-1", retained=True
        )
        idle_row = _rows(host)[handle.agent_id]
        clock.advance(7)
        idle_later = _rows(host)[handle.agent_id]
        await host._close_terminal(handle, reason="evicted")
        evicted_row = _rows(host)[handle.agent_id]
        clock.advance(100)
        evicted_later = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert idle_row.availability == AgentAvailability.IDLE.value
    assert idle_row.idle_seconds == 0.0
    assert idle_row.run_elapsed_seconds == 0.0
    assert idle_later.idle_seconds == 7.0
    assert evicted_row.availability == AgentAvailability.EVICTED.value
    assert evicted_row.idle_seconds == 7.0
    # A closed tombstone keeps no growing segment: the frozen value stands.
    assert evicted_later.idle_seconds == 7.0


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_re_task_resets_the_run_telemetry_and_keeps_the_first_summary(
    tmp_path: Path,
) -> None:
    """A continuation measures its own run and its own task, not the first's."""
    clock = _ManualClock()
    loop = _root(tmp_path)
    # A retained child re-tasks through the SAME source, so it serves both runs.
    spawn = _RecordingSpawn(
        _ScriptedSource(
            _completed_frames("first", turns=2), _completed_frames("second", turns=3)
        )
    )
    host = AgentChildHost(loop, spawn=spawn, clock=clock)
    try:
        handle, _first_ack = await _settle_background(
            host, loop, call_id="task-1", retained=True
        )
        first_row = _rows(host)[handle.agent_id]
        clock.advance(5)
        args = TaskArgs(
            task="Continue the work", agent_id=handle.agent_id, identity="coder"
        )
        ctx = _ctx(loop, "task-2")
        reused = await host.publish(
            args, ctx, attempt_id="attempt-task-2", retained=True, reuse=handle
        )
        host.admit_background(reused, args, ctx)
        await asyncio.gather(handle.task, return_exceptions=True)
        second_row = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert first_row.turns_used == 2
    assert first_row.current_task_summary == "Inspect the project"
    # The instance's first task survives the continuation ...
    assert second_row.initial_task_summary == "Inspect the project"
    # ... while the run-level telemetry follows the new run.
    assert second_row.current_task_summary == "Continue the work"
    assert second_row.turns_used == 3
    assert second_row.run_elapsed_seconds == 0.0


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_turn_count_comes_from_the_carrier_not_the_frame_stream(
    tmp_path: Path,
) -> None:
    """Frames are not turns: a count must be REPORTED on the carrier.

    A streaming child emits one ``AssistantEvent`` per chunk (three here, one
    message id) and a tool-only step emits none, so counting the frames the
    host iterates answers 3 -- and 0 for the tool-only run -- while the child
    reports one logical turn. A carrier that reports no count leaves the row
    unknown rather than falling back to the frame count. H1 carries the same
    rule onto the live path: the interim count is a value the child stamps on
    the frame, never a frame tally.
    """
    loop = _root(tmp_path)
    chunked = [
        AssistantEvent(content="po", message_id="message-1"),
        AssistantEvent(content="rtio", message_id="message-1"),
        AssistantEvent(content="n", message_id="message-1"),
        SubagentCompletion(outcome=_COMPLETED, turns_used=1),
    ]
    silent = [
        AssistantEvent(content="one", message_id="message-2"),
        AssistantEvent(content="two", message_id="message-2"),
        AssistantEvent(content="three", message_id="message-3"),
        SubagentCompletion(outcome=_COMPLETED),
    ]
    spawn = _RecordingSpawn(_ScriptedSource(chunked), _ScriptedSource(silent))
    host = AgentChildHost(loop, spawn=spawn)
    try:
        chunked_handle, chunked_ack = await _settle_background(
            host, loop, call_id="task-1"
        )
        chunked_row = _rows(host)[chunked_handle.agent_id]
        silent_handle, silent_ack = await _settle_background(
            host, loop, call_id="task-2"
        )
        silent_row = _rows(host)[silent_handle.agent_id]
        chunked_result = await host.get_agent_result(
            chunked_handle.agent_id, chunked_ack.run_id
        )
        silent_result = await host.get_agent_result(
            silent_handle.agent_id, silent_ack.run_id
        )
    finally:
        await host.close()
        await loop.aclose()

    # Three frames, one logical turn: the carrier wins.
    assert chunked_row.turns_used == 1
    assert chunked_result is not None and chunked_result.turns_used == 1
    # Three frames and NO reported count: unknown, never 3.
    assert silent_row.turns_used is None
    assert silent_result is not None and silent_result.turns_used == 0


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_child_snapshot_supplies_the_model_and_context(
    tmp_path: Path,
) -> None:
    """The model and context are the child's own reported values, or unknown.

    The host asks the child (its session snapshot) and projects the answer:
    the committed identity when the child committed one, its active-model
    alias otherwise, and the context the child measured. A child whose read
    fails leaves those fields unset -- a failed observation is not a zero.
    """
    loop = _root(tmp_path)
    answering = _TelemetryClient(
        committed=_committed_alpha(),
        context_tokens=321,
        context_window=100_000,
        context_threshold=250_000,
        thinking="high",
    )
    uncommitted = _TelemetryClient(model_alias="alias-model", context_tokens=5)
    failing = _TelemetryClient(raises=RuntimeError("child transport died"))
    spawn = _RecordingSpawn(
        _ScriptedSource(_completed_frames()),
        _ScriptedSource(_completed_frames()),
        _ScriptedSource(_completed_frames()),
        clients=[answering, uncommitted, failing],
    )
    host = AgentChildHost(loop, spawn=spawn)
    try:
        reported, _ = await _settle_background(host, loop, call_id="task-1")
        reported_row = _rows(host)[reported.agent_id]
        alias_handle, _ = await _settle_background(host, loop, call_id="task-2")
        alias_row = _rows(host)[alias_handle.agent_id]
        failed_handle, _ = await _settle_background(host, loop, call_id="task-3")
        failed_row = _rows(host)[failed_handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert answering.reads == 1
    assert reported_row.effective_model == "alpha/alpha-wire"
    assert reported_row.base_model == "alpha-model"
    assert reported_row.active_provider == "alpha"
    assert reported_row.context_tokens == 321
    assert reported_row.context_window == 100_000
    # H3: the row's denominator is the child's OWN stated trigger, which is a
    # different number from the window beside it.
    assert reported_row.context_threshold == 250_000
    assert reported_row.effective_thinking == "high"
    # No committed identity: the child's own active-model alias is the answer.
    assert alias_row.effective_model == "alias-model"
    assert alias_row.base_model is None
    assert alias_row.context_tokens == 5
    assert alias_row.context_window is None
    assert alias_row.context_threshold is None
    # The read failed: unknown stays unknown.
    assert failed_row.effective_model is None
    assert failed_row.base_model is None
    assert failed_row.context_tokens is None
    assert failed_row.context_window is None
    assert failed_row.context_threshold is None
    assert failed_row.effective_thinking is None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_telemetry_observation_advances_the_inventory_revision(
    tmp_path: Path,
) -> None:
    """The child's reported telemetry is a mutation like any other.

    The F10 emission carries whatever the revision names, so the observation
    that lands the model and context must advance it: a reader that saw the
    pre-observation revision can prove its next snapshot is newer. This pin
    holds the settle read; H1's cadence pins hold the live reads, including
    that a discarded or failed read advances nothing.
    """
    loop = _root(tmp_path)
    client = _TelemetryClient(committed=_committed_alpha(), context_tokens=12)
    host = AgentChildHost(
        loop,
        spawn=_RecordingSpawn(_ScriptedSource(_completed_frames()), clients=[client]),
    )
    try:
        handle = await host.publish(
            TaskArgs(task="Telemetry run", identity="coder"),
            _ctx(loop, "task-1"),
            attempt_id="attempt-telemetry-revision",
            retained=True,
        )
        before = host.inventory_snapshot()
        assert before.agents[0].effective_model is None
        host.admit_background(
            handle,
            TaskArgs(task="Telemetry run", identity="coder"),
            _ctx(loop, "task-1"),
        )
        assert handle.task is not None
        await asyncio.gather(handle.task, return_exceptions=True)
        after = host.inventory_snapshot()
    finally:
        await host.close()
        await loop.aclose()

    assert client.reads == 1
    assert after.revision > before.revision
    assert after.agents[0].effective_model == "alpha/alpha-wire"


# ---------------------------------------------------------------------------
# H1: a RUNNING child's row carries live telemetry
#
# G3 observed the child's own facts exactly once, at settle, so a browser row
# rendered model/context/turns as unknown for the whole run -- the live-log
# defect (R1). H1 adds the two sources a live row needs: a bounded periodic
# observation of the child's snapshot (fenced by run id, advancing the
# inventory revision only when the read lands) and the child's own running
# turn count, stamped beside each relayed event and carried through the real
# wire codec. Every pin here fails on the pre-H1 base tree this branch was cut
# from: there is no observation tick to reach a live row with, and no interim
# count to read off a relayed frame.
# ---------------------------------------------------------------------------

#: Bounded patience for one live observation, in ticks (set below at 0.02s).
_LIVE_WAIT = 5.0


def _live_wire() -> Any:
    """The H1 wire additions, imported lazily (the module must collect on base)."""
    from chartreux.app_server import agent_wire

    return agent_wire


def _relayed(event: Any, turns: int | None) -> Any:
    """One event as it crosses the pipe: encoded with the child's count, decoded."""
    wire = _live_wire()
    return wire.decode_event(wire.encode_event(event, live_turns_used=turns))


def _telemetry_tasks() -> list[asyncio.Task[Any]]:
    """The observation ticks still alive in this event loop."""
    return [
        task
        for task in asyncio.all_tasks()
        if (task.get_name() or "").startswith("child-telemetry-")
    ]


async def _wait_for_row(host: AgentChildHost, agent_id: str, check: Any) -> Any:
    """Poll the inventory until ``check`` accepts a row (bounded, never a sleep)."""
    deadline = asyncio.get_running_loop().time() + _LIVE_WAIT
    row = None
    while asyncio.get_running_loop().time() < deadline:
        row = _rows(host).get(agent_id)
        if row is not None and check(row):
            return row
        await asyncio.sleep(0.01)
    raise AssertionError(f"the live row never satisfied the check: {row}")


@pytest.fixture
def fast_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the observation cadence fast, without weakening its bound.

    The pins assert the bound itself (``test_the_observation_cadence_is_bounded``);
    shortening the tick here only decides how long a pin waits for a tick that
    the production code still schedules, cancels and fences identically.
    """
    from chartreux.app_server import root_agent

    monkeypatch.setattr(root_agent, "_CHILD_TELEMETRY_TICK_SECONDS", 0.02)


def test_the_observation_cadence_is_bounded() -> None:
    """H1: the live cadence and its read bound are explicit and sane.

    A tick slower than a couple of seconds would let a running browser row
    drift (and freeze its elapsed measure between emissions); a read bound
    longer than several seconds would let a wedged child hold the tick. Both
    bounds are production constants this pin holds in place.
    """
    from chartreux.app_server import root_agent

    tick = getattr(root_agent, "_CHILD_TELEMETRY_TICK_SECONDS", None)
    assert tick is not None, "the live observation tick does not exist"
    assert 0.25 <= tick <= 2.0, f"the observation tick {tick} is outside the bound"
    assert root_agent._CHILD_TELEMETRY_TIMEOUT == 5.0


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_quiet_running_run_is_observed_before_it_settles(
    tmp_path: Path, fast_tick: None
) -> None:
    """H1: a run that relays NO frames still shows its model and context live.

    The quiet window is the R1 defect exactly: the child is mid-run and sends
    nothing, so only the periodic read can say what model it committed and how
    much context it measures. The row must carry them while the run is still
    in flight, and the revision must have advanced -- which is what makes the
    F10 emission deliver the fresh row to the browser.
    """
    loop = _root(tmp_path)
    client = _TelemetryClient(
        committed=_committed_alpha(), context_tokens=311, thinking="high"
    )
    source = _BlockingSource()
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source, clients=[client]))
    args = TaskArgs(task="Quiet inspection", identity="coder")
    try:
        handle = await host.publish(
            args, _ctx(loop, "task-1"), attempt_id="attempt-live-1", retained=True
        )
        admitted = host.inventory_snapshot().revision
        host.admit_background(handle, args, _ctx(loop, "task-1"))
        assert handle.task is not None
        assert not handle.task.done(), "the run settled before the tick could observe"
        row = await _wait_for_row(
            host, handle.agent_id, lambda live: live.effective_model is not None
        )
        live_revision = host.inventory_snapshot().revision
        assert not handle.task.done(), "the observation raced the settle"
        source.complete(_completed_frames())
        await asyncio.gather(handle.task, return_exceptions=True)
        closed_row = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert client.reads >= 1
    assert row.availability == RunStatus.RUNNING.value
    assert row.effective_model == "alpha/alpha-wire"
    assert row.base_model == "alpha-model"
    assert row.active_provider == "alpha"
    assert row.context_tokens == 311
    assert row.effective_thinking == "high"
    assert live_revision > admitted
    # A retained child settles into its resident idle row, not a closed one.
    assert closed_row.availability == AgentAvailability.IDLE.value
    assert _telemetry_tasks() == []


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_observation_read_is_bounded_and_failure_tolerant(
    tmp_path: Path, fast_tick: None
) -> None:
    """H1: a child that cannot answer the read costs the row, never the run.

    The read is bounded per tick and its failure leaves the fields unset: a
    dead or wedged child must still reach its own settle, and the row renders
    unknown rather than a fabricated model.
    """
    loop = _root(tmp_path)
    client = _TelemetryClient(raises=RuntimeError("child transport died"))
    source = _BlockingSource()
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source, clients=[client]))
    args = TaskArgs(task="Unreadable run", identity="coder")
    try:
        handle = await host.publish(
            args, _ctx(loop, "task-1"), attempt_id="attempt-live-gone", retained=True
        )
        host.admit_background(handle, args, _ctx(loop, "task-1"))
        assert handle.task is not None
        for _ in range(100):
            if client.reads > 0:
                break
            await asyncio.sleep(0.01)
        assert client.reads > 0, "the tick never attempted the read"
        assert not handle.task.done()
        source.complete(_completed_frames())
        await asyncio.gather(handle.task, return_exceptions=True)
        row = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert row.effective_model is None
    assert row.context_tokens is None
    assert row.effective_thinking is None
    assert row.last_run_status == RunStatus.COMPLETED.value


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_wire_codec_carries_the_child_run_progress(tmp_path: Path) -> None:
    """H1: the interim count rides the event payload and back, or reads unknown.

    The count is stamped beside the encoded event (never inside its own
    schema), so an older peer's frame -- which simply has no such key -- still
    decodes, and its count is unknown rather than an error. A malformed value
    is treated the same way, mirroring the terminal carrier's strictness.
    """
    wire = _live_wire()
    key = wire.CHILD_RUN_TURNS_FRAME_KEY
    event = AssistantEvent(content="chunk")
    plain = wire.encode_event(event)
    stamped = wire.encode_event(event, live_turns_used=2)

    assert key not in plain
    assert stamped[key] == 2
    assert wire.event_live_turns_used(wire.decode_event(plain)) is None
    assert wire.event_live_turns_used(wire.decode_event(stamped)) == 2
    malformed = [
        wire.event_live_turns_used(wire.decode_event({**plain, key: value}))
        for value in ("2", True, -1, 1.5, None)
    ]
    assert malformed == [None] * 5
    assert wire.encode_event(event, live_turns_used=-1) == plain


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_live_turn_count_comes_from_the_carrier_not_the_frames(
    tmp_path: Path, fast_tick: None
) -> None:
    """H1: three relayed chunks report ONE turn while the run is still live.

    The child stamps its own committed-message count beside the events it
    relays, so a running row reports the count the run has actually reached:
    here one committed message streamed as three chunks. A frame count would
    answer three, and a count of the frames is never used.
    """
    loop = _root(tmp_path)
    source = _HeldSource((
        [
            _relayed(AssistantEvent(content="po", message_id="message-1"), 1),
            _relayed(AssistantEvent(content="rtio", message_id="message-1"), 1),
            _relayed(AssistantEvent(content="n", message_id="message-1"), 1),
        ],
        [SubagentCompletion(outcome=_COMPLETED, turns_used=1)],
    ))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    try:
        handle, _ack = await _start_background(host, loop, call_id="task-1")
        row = await _wait_for_row(
            host, handle.agent_id, lambda live: live.turns_used == 1
        )
        assert handle.task is not None and not handle.task.done()
        source.release.set()
        await asyncio.gather(handle.task, return_exceptions=True)
        closed_row = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert row.turns_used == 1
    assert closed_row.turns_used == 1


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_tool_only_step_still_reports_its_live_turn_count(
    tmp_path: Path, fast_tick: None
) -> None:
    """H1: a step that streams no assistant text still carries the count.

    A tool-only step relays no ``AssistantEvent`` at all, which is exactly why
    the frames cannot be counted -- and it still reports the run's progress,
    because the count is the child's, stamped on whatever it relays.
    """
    loop = _root(tmp_path)
    source = _HeldSource((
        [_relayed(WaitingForInputEvent(task_id="tool-step"), 1)],
        [SubagentCompletion(outcome=_COMPLETED, turns_used=1)],
    ))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    try:
        handle, _ack = await _start_background(host, loop, call_id="task-1")
        row = await _wait_for_row(
            host, handle.agent_id, lambda live: live.turns_used == 1
        )
        source.release.set()
        await asyncio.gather(handle.task, return_exceptions=True)
    finally:
        await host.close()
        await loop.aclose()

    assert row.turns_used == 1


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_terminal_carrier_settles_over_the_interim_count(
    tmp_path: Path, fast_tick: None
) -> None:
    """H1: the interim count is progress; the terminal carrier is the outcome.

    A run that reported one turn mid-flight and three at its end settles at
    three: the live count never outranks the frame the run ends with.
    """
    loop = _root(tmp_path)
    source = _HeldSource((
        [_relayed(AssistantEvent(content="first", message_id="message-1"), 1)],
        [SubagentCompletion(outcome=_COMPLETED, turns_used=3)],
    ))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    try:
        handle, _ack = await _start_background(
            host, loop, call_id="task-1", retained=True
        )
        live = await _wait_for_row(
            host, handle.agent_id, lambda row: row.turns_used == 1
        )
        source.release.set()
        await asyncio.gather(handle.task, return_exceptions=True)
        settled = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert live.turns_used == 1
    assert settled.turns_used == 3


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_frame_without_a_usable_count_leaves_the_row_unknown(
    tmp_path: Path, fast_tick: None
) -> None:
    """H1: an older peer's frame, or a malformed count, is not a turn count."""
    loop = _root(tmp_path)
    wire = _live_wire()
    unstamped = _relayed(AssistantEvent(content="text", message_id="message-1"), None)
    malformed = wire.decode_event({
        **wire.encode_event(AssistantEvent(content="text", message_id="message-2")),
        wire.CHILD_RUN_TURNS_FRAME_KEY: "2",
    })
    source = _HeldSource((
        [unstamped, malformed],
        [SubagentCompletion(outcome=_COMPLETED, turns_used=1)],
    ))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    try:
        handle, _ack = await _start_background(host, loop, call_id="task-1")
        row = await _wait_for_row(
            host, handle.agent_id, lambda live: live.availability == "running"
        )
        await asyncio.sleep(0.05)
        live_row = _rows(host)[handle.agent_id]
        source.release.set()
        await asyncio.gather(handle.task, return_exceptions=True)
    finally:
        await host.close()
        await loop.aclose()

    assert row.availability == "running"
    assert live_row.turns_used is None
    assert live_row.last_run_status is None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_re_task_resets_the_live_count_to_its_own_run(
    tmp_path: Path, fast_tick: None
) -> None:
    """H1: a retained child's second run starts at unknown, not at the first's.

    The count is per run: the handle's own number is cleared at admission, and
    only the new run's frames can raise it, exactly as the terminal carrier is
    per run.
    """
    loop = _root(tmp_path)
    source = _HeldSource(
        (
            [_relayed(AssistantEvent(content="one", message_id="message-1"), 2)],
            [SubagentCompletion(outcome=_COMPLETED, turns_used=2)],
        ),
        (
            [],
            [
                _relayed(AssistantEvent(content="two", message_id="message-2"), 1),
                SubagentCompletion(outcome=_COMPLETED, turns_used=1),
            ],
        ),
    )
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    try:
        handle, _ack = await _start_background(
            host, loop, call_id="task-1", retained=True
        )
        await _wait_for_row(host, handle.agent_id, lambda row: row.turns_used == 2)
        source.releases[0].set()
        await asyncio.gather(handle.task, return_exceptions=True)
        args = TaskArgs(
            task="Continue the work", agent_id=handle.agent_id, identity="coder"
        )
        ctx = _ctx(loop, "task-2")
        reused = await host.publish(
            args, ctx, attempt_id="attempt-task-2", retained=True, reuse=handle
        )
        host.admit_background(reused, args, ctx)
        # The second run's OWN pin, before it relays anything: the count the
        # first run left on the handle is already gone (the admission clears
        # the prior status, and the count with it).
        reset_row = await _wait_for_row(
            host,
            handle.agent_id,
            lambda row: (
                row.current_run_status == "running" and row.last_run_status is None
            ),
        )
        source.releases[1].set()
        await asyncio.gather(handle.task, return_exceptions=True)
        final_row = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert reset_row.turns_used is None
    assert final_row.turns_used == 1


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_late_observation_never_writes_over_a_newer_run(tmp_path: Path) -> None:
    """H1: the fence is taken AFTER the read suspends, before any mutation.

    The read is in flight when the handle is re-tasked, so the snapshot that
    arrives describes a run the handle no longer holds: it must be discarded
    entirely -- no field written, no revision advanced, so no emission claims
    the stale values for the new run.
    """
    loop = _root(tmp_path)
    client = _GatedTelemetryClient(
        committed=_committed_alpha(), context_tokens=999, thinking="low"
    )
    host = AgentChildHost(
        loop, spawn=_RecordingSpawn(_ScriptedSource([]), clients=[client])
    )
    try:
        handle = await host.publish(
            TaskArgs(task="Stale read", identity="coder"),
            _ctx(loop, "task-1"),
            attempt_id="attempt-stale",
            retained=True,
        )
        handle.run_id = "run-2"
        before = host.inventory_snapshot().revision
        reader = asyncio.create_task(
            host._observe_child_telemetry(handle, run_id="run-1")
        )
        await asyncio.wait_for(client.reading.wait(), _LIVE_WAIT)
        client.release.set()
        await asyncio.wait_for(reader, _LIVE_WAIT)
        after = host.inventory_snapshot().revision
        row = _rows(host)[handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert client.reads == 1
    assert after == before
    assert handle.effective_model is None
    assert handle.context_tokens is None
    assert handle.effective_thinking is None
    assert row.effective_model is None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_closed_handle_discards_the_observation_and_its_revision(
    tmp_path: Path,
) -> None:
    """H1: a terminal handle is never overwritten by a read that lands late."""
    loop = _root(tmp_path)
    client = _TelemetryClient(committed=_committed_alpha(), context_tokens=5)
    host = AgentChildHost(
        loop, spawn=_RecordingSpawn(_ScriptedSource([]), clients=[client])
    )
    try:
        handle = await host.publish(
            TaskArgs(task="Closed read", identity="coder"),
            _ctx(loop, "task-1"),
            attempt_id="attempt-closed-read",
            retained=False,
        )
        handle.run_id = "run-1"
        await host._close_terminal(handle)
        before = host.inventory_snapshot().revision
        await host._observe_child_telemetry(handle, run_id="run-1")
        after = host.inventory_snapshot().revision
    finally:
        await host.close()
        await loop.aclose()

    assert after == before
    assert handle.effective_model is None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_failed_read_does_not_advance_the_inventory_revision(
    tmp_path: Path,
) -> None:
    """H1: only a read that lands is a mutation.

    A failed observation leaves the revision where it was, so the F10 emitter
    is never asked to carry a snapshot that did not change.
    """
    loop = _root(tmp_path)
    client = _TelemetryClient(raises=RuntimeError("child transport died"))
    host = AgentChildHost(
        loop, spawn=_RecordingSpawn(_ScriptedSource([]), clients=[client])
    )
    try:
        handle = await host.publish(
            TaskArgs(task="Failed read", identity="coder"),
            _ctx(loop, "task-1"),
            attempt_id="attempt-failed-read",
            retained=True,
        )
        handle.run_id = "run-1"
        before = host.inventory_snapshot().revision
        await host._observe_child_telemetry(handle, run_id="run-1")
        after = host.inventory_snapshot().revision
    finally:
        await host.close()
        await loop.aclose()

    assert client.reads == 1
    assert after == before


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_observation_tick_is_cancelled_on_every_exit_path(
    tmp_path: Path, fast_tick: None
) -> None:
    """H1: no observation task outlives its run -- completed, failed or cancelled.

    A cancelled run reaches its settle through the cancellation path and a
    failed one through the failure path; both must stop the tick, because a
    task that survived would observe a handle the run no longer owns.
    """
    loop = _root(tmp_path)
    client = _TelemetryClient(committed=_committed_alpha(), context_tokens=1)
    cancelled_source = _BlockingSource()
    failed_source = _RaisingSource()
    host = AgentChildHost(
        loop,
        spawn=_RecordingSpawn(
            cancelled_source,
            failed_source,
            clients=[client, _TelemetryClient(committed=_committed_alpha())],
        ),
    )
    try:
        cancelled_handle, _ = await _start_background(host, loop, call_id="task-1")
        assert cancelled_handle.task is not None
        await asyncio.sleep(0.05)
        assert _telemetry_tasks(), "the tick was never started"
        cancelled_handle.task.cancel()
        await asyncio.gather(cancelled_handle.task, return_exceptions=True)
        await host.release(cancelled_handle)
        assert _telemetry_tasks() == []

        failed_handle, _ = await _start_background(host, loop, call_id="task-2")
        assert failed_handle.task is not None
        await asyncio.gather(failed_handle.task, return_exceptions=True)
        assert _telemetry_tasks() == []
    finally:
        await host.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_child_effective_thinking_follows_its_own_estimate(
    tmp_path: Path,
) -> None:
    """H1: the thinking level the child resolves is projected, or stays unknown.

    The value is the child's own: the semantic level its estimate resolved to
    when the deployment is encodable, and the level it requested when the
    estimate cannot be resolved. A child that reports no config leaves the row
    unknown rather than implying the parent's own level.
    """
    loop = _root(tmp_path)
    resolved = _TelemetryClient(committed=_committed_alpha(), thinking="high")
    unresolved = _TelemetryClient(
        committed=_committed_alpha(), thinking="medium", thinking_known=False
    )
    silent = _TelemetryClient(committed=_committed_alpha())
    host = AgentChildHost(
        loop,
        spawn=_RecordingSpawn(
            _ScriptedSource(_completed_frames()),
            _ScriptedSource(_completed_frames()),
            _ScriptedSource(_completed_frames()),
            clients=[resolved, unresolved, silent],
        ),
    )
    try:
        resolved_handle, _ = await _settle_background(host, loop, call_id="task-1")
        resolved_row = _rows(host)[resolved_handle.agent_id]
        unresolved_handle, _ = await _settle_background(host, loop, call_id="task-2")
        unresolved_row = _rows(host)[unresolved_handle.agent_id]
        silent_handle, _ = await _settle_background(host, loop, call_id="task-3")
        silent_row = _rows(host)[silent_handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert resolved_row.effective_thinking == "high"
    assert unresolved_row.effective_thinking == "medium"
    assert silent_row.effective_thinking is None


# ---------------------------------------------------------------------------
# G3b: the run's owner produces the turn count it reports
#
# G3's carrier field is only honest when the child fills it: the child counts
# the assistant messages its run committed and stamps that count on its
# terminal ``agent/turn/ended`` frame, which the coordinator relays into the
# completion frame unchanged. Every pin here fails on the pre-G3b base
# (d08bf219), where the carrier field exists but nothing produces it and the
# row renders unknown.
# ---------------------------------------------------------------------------


class _QuietTransport:
    """A child transport that only records the frames the server sends."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        await asyncio.Event().wait()
        yield {}  # pragma: no cover - the peer never speaks

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        pass


class _ChildTurnScript:
    """One driven child run: the messages its loop commits and its frames.

    A real loop appends one assistant message per LLM step and relays the
    events that step streamed, so the two counts are independent here: a pin
    can prove which one the child reports.
    """

    def __init__(
        self, loop: Any, *, messages: Sequence[LLMMessage], frames: Sequence[Any] = ()
    ) -> None:
        self._loop = loop
        self._messages = list(messages)
        self._frames = list(frames)

    def run_turn(self, request: Any) -> AsyncIterator[Any]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        for message in self._messages:
            self._loop.messages.append(message)
        for frame in self._frames:
            yield frame


async def _open_child(
    root: Path, *, session_id: str = "child-session-1"
) -> tuple[ChildAgentServer, _QuietTransport, Any]:
    """Open a real child server over a loop these pins drive turn by turn."""
    session_dir = root / "agents" / child_transcript_dir_name("coder-py", session_id)
    session_dir.mkdir(parents=True)
    loop = SimpleNamespace(
        session_id=session_id,
        messages=[],
        session_logger=SimpleNamespace(session_dir=session_dir),
    )

    class _Factory:
        async def open(self, params: Any) -> Any:
            return loop

        async def close(self) -> None:
            pass

    transport = _QuietTransport()
    server = ChildAgentServer(cast("Any", transport), factory=cast("Any", _Factory()))
    spec = child_spawn_spec(
        agent_id="coder:py", parent_session_id="root-session", root_session_dir=root
    )
    await server._handle_request(SUBAGENT_OPEN_METHOD, spawn_spec_to_wire(spec))
    return server, transport, loop


def _ended_params(transport: _QuietTransport) -> list[dict[str, Any]]:
    return [
        message["params"]
        for message in transport.sent
        if message.get("method") == AGENT_TURN_ENDED_METHOD
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_child_reports_its_own_committed_turns_on_the_terminal_frame(
    tmp_path: Path,
) -> None:
    """The child counts the assistant messages its run committed, not frames.

    The first run commits two assistant messages -- a tool-only step that
    relays nothing and a text step that streams three chunks -- so the frames
    the child relayed number three while its own count is two. A second run on
    the same child counts only its own message, so the value is the run's
    difference, the measure the in-process route derives from its loop.
    """
    server, transport, loop = await _open_child(tmp_path / "root-session")
    loop.messages.append(LLMMessage(role=Role.assistant, content="an earlier run"))
    loop.execution_port = _ChildTurnScript(
        loop,
        messages=[
            LLMMessage(role=Role.assistant, content=""),
            LLMMessage(role=Role.assistant, content="portio"),
        ],
        frames=[
            AssistantEvent(content="po", message_id="message-1"),
            AssistantEvent(content="rtio", message_id="message-1"),
            AssistantEvent(content="n", message_id="message-1"),
        ],
    )
    await server._run_turn(
        "run-1", {"message": "hi", "turn_options": {"turn_id": "run-1"}}
    )
    first = _ended_params(transport)
    event_frames = [
        message
        for message in transport.sent
        if message.get("method") == AGENT_TURN_EVENT_METHOD
    ]
    loop.execution_port = _ChildTurnScript(
        loop, messages=[LLMMessage(role=Role.assistant, content="continued")]
    )
    await server._run_turn(
        "run-2", {"message": "again", "turn_options": {"turn_id": "run-2"}}
    )
    second = _ended_params(transport)

    assert first[-1]["outcome"] == TURN_OUTCOME_COMPLETED
    # Three relayed frames, two committed messages: the child's count is two.
    assert len(event_frames) == 3
    assert first[-1]["turns_used"] == 2
    # The re-tasked child reports its own run, not the four messages it holds.
    assert second[-1]["turns_used"] == 1
    assert sum(message.role is Role.assistant for message in loop.messages) == 4


class _WireChild:
    """A child process double: relays frames and retains its terminal frame."""

    def __init__(
        self,
        *,
        frames: Sequence[Any] = (),
        terminal: dict[str, Any] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._frames = list(frames)
        self._terminal = terminal
        self._raises = raises

    async def run_turn(self, request: Any) -> AsyncIterator[Any]:
        for frame in self._frames:
            yield frame
        if self._raises is not None:
            raise self._raises

    def turn_terminal(self, turn_id: str) -> dict[str, Any] | None:
        if self._terminal is None:
            return None
        return {"turn_id": turn_id, **self._terminal}


async def _completions(process: Any, *, turn_id: str = "run-1") -> list[Any]:
    source = ProcessSubagentCompletionSource("coder:py", cast("Any", process))
    request = ExecutionTurnRequest(
        message="hi", turn_options=AgentTurnOptions(turn_id=turn_id)
    )
    return [frame async for frame in source.stream(request)]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_completion_carries_the_child_reported_count_or_unknown() -> None:
    """Only a usable count crosses; a malformed or missing one stays unknown.

    The count is the child's, relayed from its terminal frame. A frame that
    carries none -- or carries anything but a non-negative integer -- leaves
    the field unset, so the reader renders unknown instead of a guessed
    number. A failed run still keeps the count its owner published.
    """
    reported = await _completions(_WireChild(terminal={"turns_used": 2}))
    missing = await _completions(_WireChild(terminal={"outcome": _COMPLETED}))
    no_terminal = await _completions(_WireChild())
    malformed = [
        await _completions(_WireChild(terminal={"turns_used": value}))
        for value in ("2", True, -1, 1.5, None)
    ]
    failed = await _completions(
        _WireChild(
            terminal={"outcome": _FAILED, "turns_used": 1},
            raises=AgentProcessTurnFailed("the child turn failed"),
        )
    )

    assert reported[-1].outcome == TURN_OUTCOME_COMPLETED
    assert reported[-1].turns_used == 2
    assert missing[-1].turns_used is None
    assert no_terminal[-1].turns_used is None
    assert all(frames[-1].turns_used is None for frames in malformed)
    assert failed[-1].outcome == TURN_OUTCOME_FAILED
    assert failed[-1].turns_used == 1


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_host_projects_the_child_reported_count_end_to_end(
    tmp_path: Path,
) -> None:
    """The producer's count reaches the row and the stored result unchanged.

    The run streams three chunks of one message and its child reports two
    committed turns, so both surfaces carry 2 -- the child's number, never the
    three frames the same run relayed.
    """
    loop = _root(tmp_path)
    process = _WireChild(
        frames=[
            AssistantEvent(content="po", message_id="message-1"),
            AssistantEvent(content="rtio", message_id="message-1"),
            AssistantEvent(content="n", message_id="message-1"),
        ],
        terminal={"outcome": _COMPLETED, "turns_used": 2},
    )
    host = AgentChildHost(
        loop,
        spawn=_RecordingSpawn(
            ProcessSubagentCompletionSource("coder:py", cast("Any", process))
        ),
    )
    try:
        handle, ack = await _settle_background(host, loop, call_id="task-1")
        row = _rows(host)[handle.agent_id]
        result = await host.get_agent_result(handle.agent_id, ack.run_id)
    finally:
        await host.close()
        await loop.aclose()

    assert row.turns_used == 2
    assert result is not None and result.turns_used == 2


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_child_compaction_threshold_is_projected_or_unknown(
    tmp_path: Path,
) -> None:
    """H3: the row's denominator is the child's own stated trigger, or unknown.

    The trigger rides the child's config view, so the host projects the value
    that child's own auto-compaction middleware would compare tokens with. A
    peer that states nothing usable -- a bool, a string, a nonpositive count,
    no config at all -- leaves the denominator unknown instead of being
    coerced, and the window beside it is never substituted.
    """
    loop = _root(tmp_path)
    stated = _TelemetryClient(committed=_committed_alpha(), context_threshold=500_000)
    silent = _TelemetryClient(committed=_committed_alpha())
    garbage = _TelemetryClient(committed=_committed_alpha(), context_threshold="500000")
    disabled = _TelemetryClient(committed=_committed_alpha(), context_threshold=0)
    host = AgentChildHost(
        loop,
        spawn=_RecordingSpawn(
            _ScriptedSource(_completed_frames()),
            _ScriptedSource(_completed_frames()),
            _ScriptedSource(_completed_frames()),
            _ScriptedSource(_completed_frames()),
            clients=[stated, silent, garbage, disabled],
        ),
    )
    try:
        stated_handle, _ = await _settle_background(host, loop, call_id="task-1")
        stated_row = _rows(host)[stated_handle.agent_id]
        silent_handle, _ = await _settle_background(host, loop, call_id="task-2")
        silent_row = _rows(host)[silent_handle.agent_id]
        garbage_handle, _ = await _settle_background(host, loop, call_id="task-3")
        garbage_row = _rows(host)[garbage_handle.agent_id]
        disabled_handle, _ = await _settle_background(host, loop, call_id="task-4")
        disabled_row = _rows(host)[disabled_handle.agent_id]
    finally:
        await host.close()
        await loop.aclose()

    assert stated_row.context_threshold == 500_000
    assert silent_row.context_threshold is None
    assert garbage_row.context_threshold is None
    # A nonpositive trigger states "compaction off", which is not a count a row
    # may render a percentage against, so it reads as unknown.
    assert disabled_row.context_threshold is None
