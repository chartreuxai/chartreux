"""P2 chunk 2a: registry-owned synchronization + run-state writer inventory.

The terminal-state registry acquisition
(``SessionRuntimeRegistry.acquire_run_state``) is the serialization point:
every enumerated run-state writer site in ``_subagent_execution.py`` acquires
it around its write, and the final quiescence check + termination declaration
execute while HOLDING it, mutually exclusive with every run-state writer. The
three quiescence conditions read as ONE consistent snapshot under that
acquisition; a transition-triggered re-check alone never declares
termination.

Harness: ``tests/app_server/_subagent_test_support.py`` backends.
"""

from __future__ import annotations

import ast
import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry, _AgentState
from chartreux.core.completion_contract import CompletionRejectionReason
from chartreux.core.recovery_models import build_completion_rejection_envelope
from chartreux.core.subagents import RunStatus, TaskArgs, WaitStatus
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _ManualClock,
    _record,
    _root_registry,
    _run,
    _stored_result,
    _wait_result,
    _zero_retention_registry,
)

_EXECUTION_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "chartreux"
    / "app_server"
    / "_subagent_execution.py"
)


# ---------------------------------------------------------------------------
# Writer-site inventory (static): every enumerated run-state writer statement
# sits inside an `async with ...acquire_run_state()` block.
# ---------------------------------------------------------------------------


def _is_runrecord_running_ctor(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "RunRecord"
        and any(kw.arg == "status" for kw in stmt.value.keywords)
    )


def _is_run_state_write(stmt: ast.AST) -> bool:
    if not isinstance(stmt, ast.Assign):
        return False
    value = stmt.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "_resolve_run_outcome"
    ):
        return True
    if len(stmt.targets) != 1:
        return False
    target = stmt.targets[0]
    return (
        isinstance(target, ast.Attribute)
        and target.attr in ("status", "stop_reason")
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "run_record"
    )


def _acquire_blocks(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AsyncWith]:
    return [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.AsyncWith)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "acquire_run_state"
            for item in node.items
        )
    ]


def test_every_enumerated_writer_site_acquires_the_registry() -> None:
    tree = ast.parse(_EXECUTION_SOURCE.read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    execution_cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "SubagentExecution"
    )
    sites: list[tuple[str, ast.AST]] = []
    acquire_blocks: set[int] = set()
    acquire_counts: dict[str, int] = {}
    for func in execution_cls.body:
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if func.name not in ("prepare_background", "monitor"):
            continue
        blocks = _acquire_blocks(func)
        acquire_counts[func.name] = len(blocks)
        acquire_blocks.update(id(block) for block in blocks)
        for node in ast.walk(func):
            if _is_runrecord_running_ctor(node) or _is_run_state_write(node):
                sites.append((func.name, node))

    # The pinned inventory: one admission RUNNING constructor in
    # prepare_background and the four monitor writer blocks (normal terminal
    # resolution, publication-rejection FAILED, CancelledError resolution,
    # monitor-failure FAILED).
    ctors = [stmt for name, stmt in sites if _is_runrecord_running_ctor(stmt)]
    assert len(ctors) == 1
    assert (
        next(name for name, stmt in sites if stmt is ctors[0]) == "prepare_background"
    )
    monitor_writes = [stmt for name, stmt in sites if name == "monitor"]
    assert len(monitor_writes) == 6  # 2 resolution tuples + 4 FAILED writes
    assert acquire_counts == {"prepare_background": 1, "monitor": 4}

    def inside_acquire(stmt: ast.AST) -> bool:
        node: ast.AST = stmt
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.AsyncWith) and id(node) in acquire_blocks:
                return True
        return False

    for _name, stmt in sites:
        assert inside_acquire(stmt), (
            f"writer site at line {getattr(stmt, 'lineno', '?')} is not inside an "
            "acquire_run_state() block"
        )


# ---------------------------------------------------------------------------
# Dynamic writer discipline: an instrumented acquisition records one span per
# acquisition with the observable run-state at each span boundary; every
# writer-site transition must land inside a span.
# ---------------------------------------------------------------------------


@dataclass
class _Span:
    enter: tuple[Any, ...]
    exit: tuple[Any, ...] | None = None


class _RunStateRecorder:
    """Wraps the registry's acquire_run_state, recording span boundaries."""

    def __init__(self, registry: SessionRuntimeRegistry) -> None:
        self._registry = registry
        self._original = registry.acquire_run_state
        self.spans: list[_Span] = []

    @staticmethod
    def _observe(registry: SessionRuntimeRegistry) -> tuple[Any, ...]:
        runs = tuple(
            sorted(
                (
                    record.agent_id,
                    record.current_run.run_id if record.current_run else None,
                    record.current_run.status.value if record.current_run else None,
                )
                for record in registry._agent_records.values()
            )
        )
        return (
            runs,
            tuple(sorted(registry._pending_notifications)),
            tuple(sorted(registry._result_store)),
            tuple(sorted(registry._collected_results)),
        )

    @asynccontextmanager
    async def wrapped(self):
        span = _Span(enter=self._observe(self._registry))
        self.spans.append(span)
        async with self._original():
            yield
        span.exit = self._observe(self._registry)

    def install(self) -> None:
        self._registry.acquire_run_state = self.wrapped  # type: ignore[method-assign]

    def status_transition_inside_span(
        self, agent_id: str, before: str | None, after: str
    ) -> bool:
        def pick(obs: tuple[Any, ...]) -> str | None:
            for aid, _rid, status in obs[0]:
                if aid == agent_id:
                    return status
            return None

        return any(
            span.exit is not None
            and pick(span.enter) == before
            and pick(span.exit) == after
            for span in self.spans
        )


async def _launch_blocked(
    registry: SessionRuntimeRegistry, parent: Any, backend: BlockingBackend
) -> Any:
    launch = await _background_result(
        registry,
        TaskArgs(task="work"),
        InvokeContext(tool_call_id="child", session_id=parent.session_id),
    )
    assert launch.agent_id is not None and launch.run_id is not None
    await asyncio.wait_for(backend.started.wait(), 2)
    return launch


@pytest.mark.asyncio
async def test_admission_and_terminal_writes_acquire_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    recorder = _RunStateRecorder(registry)
    recorder.install()
    try:
        launch = await _launch_blocked(registry, parent, backend)
        # Site 1 (admission RUNNING write): None -> running inside a span.
        assert recorder.status_transition_inside_span(launch.agent_id, None, "running")
        backend.release.set()
        result = await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), 2
        )
        assert result.completed
        # Site 2 (normal terminal resolution): running -> completed in a span.
        assert recorder.status_transition_inside_span(
            launch.agent_id, "running", "completed"
        )
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_cancelled_monitor_write_acquires_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    recorder = _RunStateRecorder(registry)
    recorder.install()
    try:
        launch = await _launch_blocked(registry, parent, backend)
        monitor = next(iter(registry._monitor_tasks))
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor
        # Site 4 (CancelledError resolution): running -> cancelled in a span.
        assert recorder.status_transition_inside_span(
            launch.agent_id, "running", "cancelled"
        )
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_monitor_failure_write_acquires_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    recorder = _RunStateRecorder(registry)
    recorder.install()

    async def boom(**_kwargs: Any) -> None:
        raise RuntimeError("finalize blew up")

    monkeypatch.setattr(
        "chartreux.app_server._subagent_execution.finalize_run_with_drain", boom
    )
    try:
        launch = await _launch_blocked(registry, parent, backend)
        run_record = registry._agent_records[launch.agent_id].current_run
        assert run_record is not None
        monitor = next(iter(registry._monitor_tasks))
        backend.release.set()
        await asyncio.wait_for(asyncio.shield(monitor), 5)
        # Site 2 resolved the run completed first; site 5 (monitor failure)
        # flipped it to failed inside its own span.
        assert recorder.status_transition_inside_span(
            launch.agent_id, "running", "completed"
        )
        assert recorder.status_transition_inside_span(
            launch.agent_id, "completed", "failed"
        )
        assert run_record.status is RunStatus.FAILED
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_publication_rejection_write_acquires_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    recorder = _RunStateRecorder(registry)
    recorder.install()

    def fake_rejection(context: Any, result: Any) -> Any:
        return build_completion_rejection_envelope(
            agent_id=context.agent_id,
            run_id=context.run_id,
            rejection=CompletionRejectionReason.CLAIM_CONTRADICTED,
        )

    monkeypatch.setattr(
        "chartreux.app_server._subagent_execution.rejection_envelope", fake_rejection
    )
    try:
        launch = await _launch_blocked(registry, parent, backend)
        monitor = next(iter(registry._monitor_tasks))
        backend.release.set()
        await asyncio.wait_for(asyncio.shield(monitor), 5)
        # Site 3 (publication rejection): completed -> failed in a span.
        assert recorder.status_transition_inside_span(
            launch.agent_id, "completed", "failed"
        )
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# Quiescence snapshot + the serialization invariant.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quiescence_snapshot_reads_the_three_conditions_as_one_snapshot() -> None:
    clock = _ManualClock()
    registry, _root = _root_registry(clock)
    record = _record("agent-1", clock=clock, state=_AgentState.RUNNING)
    run = _run(record, "run-1", started_at=0, status=RunStatus.RUNNING, done=False)
    record.current_run = run
    registry._agent_records[record.agent_id] = record
    registry._result_store[("agent-1", "run-0")] = _stored_result("agent-1", "run-0")
    registry._pending_notifications.add(("agent-1", "run-0"))
    registry._result_store[("agent-1", "run-x")] = _stored_result("agent-1", "run-x")

    snapshot = await registry.quiescence_snapshot()
    assert snapshot.active_dependency_runs == (("agent-1", "run-1"),)
    assert snapshot.pending_notifications == (("agent-1", "run-0"),)
    # run-0 is pending-notified, so only run-x is stranded.
    assert snapshot.unnotified_unfetched_results == (("agent-1", "run-x"),)
    assert not snapshot.quiescent
    # A bare snapshot never declares termination.
    assert not registry.termination_declared

    async with registry.acquire_run_state():
        record.current_run = None
        registry._pending_notifications.clear()
        registry._collected_results.update(registry._result_store.keys())
    snapshot = await registry.quiescence_snapshot()
    assert snapshot.quiescent
    assert not registry.termination_declared


@pytest.mark.asyncio
async def test_check_and_declare_termination_only_under_the_held_acquisition() -> None:
    registry, _root = _root_registry(_ManualClock())
    snapshot = await registry.check_quiescence_and_declare_termination()
    assert snapshot.quiescent
    assert registry.termination_declared

    # A transition-triggered re-check alone does NOT satisfy the invariant:
    # quiescence_snapshot never declares.
    other, _root2 = _root_registry(_ManualClock())
    snapshot = await other.quiescence_snapshot()
    assert snapshot.quiescent
    assert not other.termination_declared


class _ParkingLock(asyncio.Lock):
    """Holds the real lock while the designated task's first acquisition parks."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.park_task: asyncio.Task[Any] | None = None
        self.parked = asyncio.Event()
        self.unpark = asyncio.Event()
        self._parked_once = False

    async def acquire(self) -> Literal[True]:
        await super().acquire()
        if (
            self.armed
            and not self._parked_once
            and asyncio.current_task() is self.park_task
        ):
            self._parked_once = True
            self.parked.set()
            await self.unpark.wait()  # still holding the real lock
        return True


@pytest.mark.asyncio
async def test_quiescence_check_is_mutually_exclusive_with_a_run_state_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    lock = _ParkingLock()
    registry._run_state_lock = lock  # type: ignore[assignment]
    try:
        launch = await _launch_blocked(registry, parent, backend)
        monitor = next(iter(registry._monitor_tasks))
        lock.park_task = monitor
        lock.armed = True
        backend.release.set()
        await asyncio.wait_for(lock.parked.wait(), 2)
        run_record = registry._agent_records[launch.agent_id].current_run
        assert run_record is not None
        # The writer parked INSIDE its acquisition before the write.
        assert run_record.status is RunStatus.RUNNING
        check = asyncio.create_task(registry.check_quiescence_and_declare_termination())
        await asyncio.sleep(0.05)
        # The final quiescence check blocks on the writer's held acquisition.
        assert not check.done()
        lock.unpark.set()
        snapshot = await asyncio.wait_for(check, 2)
        # The writer's write landed first; the check then read one consistent
        # post-write snapshot (the run is still active until publication).
        assert run_record.status is RunStatus.COMPLETED
        assert not snapshot.quiescent
        assert not registry.termination_declared
        await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), 2
        )
    finally:
        lock.armed = False
        lock.unpark.set()
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_held_quiescence_check_blocks_a_run_state_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    lock = _ParkingLock()
    registry._run_state_lock = lock  # type: ignore[assignment]
    try:
        launch = await _launch_blocked(registry, parent, backend)
        check = asyncio.create_task(registry.check_quiescence_and_declare_termination())
        lock.park_task = check
        lock.armed = True
        await asyncio.wait_for(lock.parked.wait(), 2)
        # The check holds the acquisition across check + declaration; a writer
        # transitioning the run must wait.
        backend.release.set()
        await asyncio.sleep(0.1)
        run_record = registry._agent_records[launch.agent_id].current_run
        assert run_record is not None
        assert run_record.status is RunStatus.RUNNING
        assert not check.done()
        lock.unpark.set()
        snapshot = await asyncio.wait_for(check, 2)
        # The check read ONE consistent pre-write snapshot under the held
        # acquisition: the active run is visible, no declaration.
        assert snapshot.active_dependency_runs == ((launch.agent_id, launch.run_id),)
        assert not snapshot.quiescent
        assert not registry.termination_declared
        await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), 2
        )
        assert run_record.status is RunStatus.COMPLETED
    finally:
        lock.armed = False
        lock.unpark.set()
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_wait_for_agent_perf_call_shape_remains_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # tests/perf callers: registry.wait_for_agent(agent_id, run_id, timeout=N)
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    try:
        launch = await _launch_blocked(registry, parent, backend)
        pending = await registry.wait_for_agent(
            launch.agent_id, launch.run_id, timeout=0.01
        )
        assert pending.status is WaitStatus.STILL_RUNNING
        backend.release.set()
        done = await registry.wait_for_agent(launch.agent_id, launch.run_id, timeout=2)
        assert done.status is WaitStatus.COMPLETED
        assert done.result is not None and done.result.completed
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()
