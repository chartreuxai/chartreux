"""P2 chunk 2b: the reconciliation sweep + the interactive notification backstop.

The sweep (``SubagentExecution.sweep_unnotified``) is the guaranteed backstop
behind the conditional delivered-notification path: at continuation start,
at drain completion, on every run-state terminal transition, on a bounded
periodic timer while the continuation is active, and at the wall-clock cap
boundary, it synthesizes a notification CHANNEL-EQUIVALENT to a delivered
completion notification for any completed-but-unnotified run. It writes NO
delivery marking — ``_collected_results`` (fetch markings) and
``_pending_notifications`` are only ever read by the sweep — and the
delivered path records its own surfaced keys so the channel is exactly-once
per completion event (one channel, no double-messaging).

Harness: ``tests/app_server/_subagent_test_support.py`` backends.
"""

from __future__ import annotations

import ast
import asyncio
from contextlib import suppress
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest

from chartreux.app_server._projection import project_message_history
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._subagent_execution import (
    SWEEP_TRIGGER_CAP_BOUNDARY,
    SWEEP_TRIGGER_CONTINUATION_START,
    SWEEP_TRIGGER_DRAIN_COMPLETE,
    SWEEP_TRIGGER_TERMINAL_TRANSITION,
    SWEEP_TRIGGER_TIMER,
    SWEEP_TRIGGERS,
    _completion_status_word,
)
from chartreux.app_server.models import TextContentBlock
from chartreux.app_server.protocol import TurnStartParams
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.subagents import RunStatus, RunStopReason, TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _ManualClock,
    _record,
    _root_registry,
    _run,
    _stored_result,
    _zero_retention_registry,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_EXECUTION_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "chartreux"
    / "app_server"
    / "_subagent_execution.py"
)

_TRIGGER_CONSTANT_NAMES = {
    "SWEEP_TRIGGER_CONTINUATION_START",
    "SWEEP_TRIGGER_DRAIN_COMPLETE",
    "SWEEP_TRIGGER_TERMINAL_TRANSITION",
    "SWEEP_TRIGGER_TIMER",
    "SWEEP_TRIGGER_CAP_BOUNDARY",
}


async def _wait_for(predicate, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met within timeout")


async def _cancel_sweep_task(registry: SessionRuntimeRegistry) -> None:
    task = registry._execution._sweep_task
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


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


class _StubTurns:
    def __init__(self) -> None:
        self.active_turn = None
        self._active_task = None
        self.queue_state = SimpleNamespace(items=[])
        self.started: list[str] = []

    def start(self, params: TurnStartParams) -> tuple[MagicMock, Any]:
        block = params.message[0]
        assert isinstance(block, TextContentBlock)
        self.started.append(block.text)
        return MagicMock(), lambda: None


class _StubParent:
    """Minimal parent double for direct sweep calls (idle, open)."""

    def __init__(self) -> None:
        self.turns = _StubTurns()
        self.agent_loop = SimpleNamespace(
            session_id="root", _pending_injected_messages=[]
        )
        self._closed = False


def _fabricate_gap(
    registry: SessionRuntimeRegistry, *, run_id: str = "run-1"
) -> tuple[str, Any, Any]:
    clock = _ManualClock()
    record = _record("agent-1", clock=clock)
    run = _run(record, run_id, started_at=0, status=RunStatus.COMPLETED, done=True)
    record.run_history.append(run)
    registry._agent_records[record.agent_id] = record
    registry._result_store[(record.agent_id, run_id)] = _stored_result(
        record.agent_id, run_id
    )
    return record.agent_id, record, run


# ---------------------------------------------------------------------------
# Static: every sweep trigger label has a call site; the status word mirrors
# the delivered builder's wording.
# ---------------------------------------------------------------------------


def test_every_trigger_label_has_a_sweep_call_site() -> None:
    tree = ast.parse(_EXECUTION_SOURCE.read_text(encoding="utf-8"))
    call_labels: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sweep_unnotified"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
        ):
            call_labels.add(node.args[1].id)
    assert call_labels == _TRIGGER_CONSTANT_NAMES
    # The constant values are the design's five trigger labels.
    assert SWEEP_TRIGGERS == {
        SWEEP_TRIGGER_CONTINUATION_START,
        SWEEP_TRIGGER_DRAIN_COMPLETE,
        SWEEP_TRIGGER_TERMINAL_TRANSITION,
        SWEEP_TRIGGER_TIMER,
        SWEEP_TRIGGER_CAP_BOUNDARY,
    }


def test_status_word_mirrors_the_delivered_builder() -> None:
    cases = [
        (
            RunStatus.FAILED,
            RunStopReason.BUDGET_EXCEEDED,
            "budget-stopped (budget exceeded)",
        ),
        (
            RunStatus.FAILED,
            RunStopReason.BUDGET_UNVERIFIABLE,
            "budget-stopped (budget unverifiable)",
        ),
        (RunStatus.COMPLETED, None, "completed"),
        (RunStatus.FAILED, None, "failed"),
        (RunStatus.CANCELLED, RunStopReason.USER_CANCELLED, "cancelled (by user)"),
        (RunStatus.CANCELLED, RunStopReason.RETASKED, "retasked"),
        (RunStatus.CANCELLED, None, "cancelled"),
    ]
    for status, reason, expected in cases:
        assert _completion_status_word(status, reason) == expected


def test_synthesized_notifications_project_like_any_injected_notice() -> None:
    # Recovery-notice projection consistency: a sweep-synthesized backstop
    # notification is an injected text message and projects exactly like any
    # other injected notice (no history entry); a wake-started notification
    # projects as an ordinary user message — identical to the delivered path.
    synthesized = (
        "Background agent coder-1 (agent-1, run run-1) completed."
        " Use get_agent_result to retrieve the result."
    )
    messages = [
        LLMMessage(role=Role.user, content="user question"),
        LLMMessage(role=Role.user, content=synthesized, injected=True),
        LLMMessage(role=Role.user, content="recovery notice", injected=True),
        LLMMessage(role=Role.user, content=synthesized),
    ]
    entries = project_message_history("session-1", messages, None)
    texts = [
        block.text
        for entry in entries
        for block in (getattr(entry, "content", None) or [])
        if isinstance(block, TextContentBlock)
    ]
    assert texts == ["user question", synthesized]


# ---------------------------------------------------------------------------
# Gap semantics: pending / collected / delivered / already-surfaced keys are
# not gaps; a real gap is surfaced exactly once; no delivery marking is
# written; delivery is skipped at the session boundary.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sweep_gap_filters_exactly_once_and_no_marking() -> None:
    registry, _root = _root_registry(_ManualClock())
    execution = registry._execution
    agent_id, record, _run_record = _fabricate_gap(registry)
    key = (agent_id, "run-1")
    parent: Any = _StubParent()
    try:
        # Pending a delivered notification -> not a gap.
        registry._pending_notifications.add(key)
        assert await execution.sweep_unnotified(parent, SWEEP_TRIGGER_TIMER) == ()
        assert not parent.turns.started
        registry._pending_notifications.clear()
        # Surfaced by the delivered path -> not a gap.
        execution._delivered_notification_keys.add(key)
        assert await execution.sweep_unnotified(parent, SWEEP_TRIGGER_TIMER) == ()
        execution._delivered_notification_keys.clear()
        # Collected (fetch marking) -> not a gap.
        registry._collected_results.add(key)
        assert await execution.sweep_unnotified(parent, SWEEP_TRIGGER_TIMER) == ()
        registry._collected_results.clear()
        # A real gap: surfaced once, channel-equivalent text.
        surfaced = await execution.sweep_unnotified(parent, SWEEP_TRIGGER_TIMER)
        assert surfaced == (key,)
        assert len(parent.turns.started) == 1
        expected = (
            f"Background agent {registry._record_name(record)} "
            f"(agent-1, run run-1) completed."
            " Use get_agent_result to retrieve the result."
        )
        assert parent.turns.started[0] == expected
        assert key in execution._sweep_surfaced
        # Exactly-once: a further sweep does not re-surface.
        assert (
            await execution.sweep_unnotified(parent, SWEEP_TRIGGER_DRAIN_COMPLETE) == ()
        )
        assert len(parent.turns.started) == 1
        # NO delivery marking: fetch markings and pending notifications are
        # untouched by the sweep.
        assert key not in registry._collected_results
        assert not registry._pending_notifications
    finally:
        await _cancel_sweep_task(registry)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_timer_surfaces_a_gap_within_one_interval() -> None:
    registry, _root = _root_registry(_ManualClock())
    execution = registry._execution
    execution._sweep_interval = 0.02
    clock = _ManualClock()
    record = _record("agent-1", clock=clock)
    run = _run(record, "run-1", started_at=0, status=RunStatus.RUNNING, done=False)
    record.current_run = run  # active dependency run keeps the timer alive
    registry._agent_records[record.agent_id] = record
    parent: Any = _StubParent()
    try:
        # Arm the timer with no gaps present.
        assert (
            await execution.sweep_unnotified(parent, SWEEP_TRIGGER_CONTINUATION_START)
            == ()
        )
        assert execution._sweep_task is not None and not execution._sweep_task.done()
        # A dropped notification leaves a stored result with no further
        # trigger: only the bounded periodic timer can surface it.
        run.status = RunStatus.COMPLETED
        record.run_history.append(run)
        record.current_run = None
        registry._result_store[("agent-1", "run-1")] = _stored_result(
            "agent-1", "run-1"
        )
        inserted_at = time.monotonic()
        await _wait_for(lambda: parent.turns.started)
        elapsed = time.monotonic() - inserted_at
        assert elapsed <= execution._sweep_interval + 0.5
        assert SWEEP_TRIGGER_TIMER in execution._sweep_triggers_seen
        assert len(parent.turns.started) == 1
        assert "run run-1" in parent.turns.started[0]
        assert "completed" in parent.turns.started[0]
        # Still exactly once after further intervals.
        await asyncio.sleep(0.1)
        assert len(parent.turns.started) == 1
        assert ("agent-1", "run-1") not in registry._collected_results
        assert not registry._pending_notifications
    finally:
        await _cancel_sweep_task(registry)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sweep_skips_delivery_at_the_session_boundary() -> None:
    registry, _root = _root_registry(_ManualClock())
    execution = registry._execution
    agent_id, _record_obj, _run_record = _fabricate_gap(registry)
    parent: Any = _StubParent()
    try:
        registry._draining_children = True
        await execution.sweep_unnotified(parent, SWEEP_TRIGGER_CAP_BOUNDARY)
        # The scan ran (trigger recorded, gap processed) but nothing is
        # injected into a draining parent.
        assert SWEEP_TRIGGER_CAP_BOUNDARY in execution._sweep_triggers_seen
        assert not parent.turns.started
        assert not parent.agent_loop._pending_injected_messages
        assert (agent_id, "run-1") not in registry._collected_results
    finally:
        registry._draining_children = False
        await _cancel_sweep_task(registry)


# ---------------------------------------------------------------------------
# Real launches: delivered exactly-once vs. suppressed-surfaced-by-the-sweep,
# trigger coverage, and the quiescence serialization of the terminal
# transition sweep.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_delivered_notification_is_surfaced_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    registry._execution._sweep_interval = 0.02
    try:
        launch = await _launch_blocked(registry, parent, backend)
        key = (launch.agent_id, launch.run_id)
        backend.release.set()
        await _wait_for(lambda: key in registry._result_store)
        await _wait_for(lambda: key not in registry._pending_notifications)
        await _wait_for(lambda: any(f"run {launch.run_id}" in t for t in notifications))
        delivered = [t for t in notifications if f"run {launch.run_id}" in t]
        assert len(delivered) == 1
        assert "completed" in delivered[0]
        # The delivered path recorded the surfaced key...
        assert key in registry._execution._delivered_notification_keys
        # ...so several timer intervals with the result still uncollected do
        # not produce a second channel.
        await asyncio.sleep(0.15)
        assert len([t for t in notifications if f"run {launch.run_id}" in t]) == 1
        assert key not in registry._execution._sweep_surfaced
        assert key not in registry._collected_results
        # Lifecycle trigger coverage (the cap-boundary label needs a budget
        # stop — covered by the dedicated test below).
        seen = registry._execution._sweep_triggers_seen
        assert {
            SWEEP_TRIGGER_CONTINUATION_START,
            SWEEP_TRIGGER_TERMINAL_TRANSITION,
            SWEEP_TRIGGER_DRAIN_COMPLETE,
            SWEEP_TRIGGER_TIMER,
        } <= seen
    finally:
        backend.release.set()
        await _cancel_sweep_task(registry)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_suppressed_notification_surfaced_once_within_one_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    registry._execution._sweep_interval = 0.02
    try:
        launch = await _launch_blocked(registry, parent, backend)
        key = (launch.agent_id, launch.run_id)
        # Deliberately drop the delivered notification at publish time: the
        # suppression gate (:1845-1887) stays silent and leaks no second
        # channel — the sweep is the one channel.
        registry._suppressed_notifications.add(launch.agent_id)
        backend.release.set()
        await _wait_for(lambda: key in registry._result_store)
        stored_at = time.monotonic()
        # The 0.02s sweep may already have surfaced it here; the delivered
        # channel's silence is asserted by the key check below.
        assert key not in registry._execution._delivered_notification_keys
        # The sweep (idle-window timer) surfaces it within one timer interval.
        await _wait_for(lambda: any(f"run {launch.run_id}" in t for t in notifications))
        surfaced_at = time.monotonic()
        hits = [t for t in notifications if f"run {launch.run_id}" in t]
        assert len(hits) == 1
        assert "completed" in hits[0]
        assert surfaced_at - stored_at <= registry._execution._sweep_interval + 0.5
        # Exactly once across further intervals; no delivery marking written.
        await asyncio.sleep(0.1)
        assert len([t for t in notifications if f"run {launch.run_id}" in t]) == 1
        assert key in registry._execution._sweep_surfaced
        assert key not in registry._collected_results
        assert not registry._pending_notifications
    finally:
        backend.release.set()
        registry._suppressed_notifications.clear()
        await _cancel_sweep_task(registry)
        await registry.drain_children()
        await parent.aclose()


class _UnresolvedToolCallBackend(FakeBackend):
    """Two-turn child drive: an unresolved tool call, then a denied response."""

    def __init__(self) -> None:
        super().__init__([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            index=0,
                            function=FunctionCall(name="no_such_tool", arguments="{}"),
                        )
                    ],
                )
            ],
            [mock_llm_chunk(content="unreachable after the budget stop")],
        ])


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_budget_stop_fires_the_cap_boundary_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _UnresolvedToolCallBackend()
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    registry._execution._sweep_interval = 0.02
    parent.config.subagents.run_max_price = 0.00001
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", background=True),
            InvokeContext(tool_call_id="budget", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        run = record.current_run
        assert run is not None
        # The never-completing child is stopped by its run budget.
        await asyncio.wait_for(run.completion_task, 10)
        assert run.stop_reason is RunStopReason.BUDGET_EXCEEDED
        assert SWEEP_TRIGGER_CAP_BOUNDARY in registry._execution._sweep_triggers_seen
        hits = [t for t in notifications if f"run {launch.run_id}" in t]
        assert len(hits) == 1
        assert "budget-stopped (budget exceeded)" in hits[0]
        # Delivered exactly once: no sweep duplicate for the same event.
        await asyncio.sleep(0.1)
        assert len([t for t in notifications if f"run {launch.run_id}" in t]) == 1
        key = (launch.agent_id, launch.run_id)
        assert key in registry._execution._delivered_notification_keys
        assert key not in registry._execution._sweep_surfaced
    finally:
        await _cancel_sweep_task(registry)
        await registry.drain_children()
        await parent.aclose()


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
@pytest.mark.timeout(30)
async def test_terminal_transition_during_final_quiescence_check_serializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    registry._execution._sweep_interval = 0.05
    lock = _ParkingLock()
    registry._run_state_lock = lock  # type: ignore[assignment]
    seen: list[tuple[str, RunStatus | None]] = []
    try:
        launch = await _launch_blocked(registry, parent, backend)
        execution = registry._execution
        original = execution.sweep_unnotified

        async def recording(parent_arg: Any, trigger: str):
            record = registry._agent_records.get(launch.agent_id)
            run = record.current_run if record is not None else None
            seen.append((trigger, run.status if run is not None else None))
            return await original(parent_arg, trigger)

        monkeypatch.setattr(execution, "sweep_unnotified", recording)
        monitor = next(iter(registry._monitor_tasks))
        lock.park_task = monitor
        lock.armed = True
        backend.release.set()
        # The writer parks INSIDE its terminal-write acquisition.
        await asyncio.wait_for(lock.parked.wait(), 2)
        run_record = registry._agent_records[launch.agent_id].current_run
        assert run_record is not None
        assert run_record.status is RunStatus.RUNNING
        check = asyncio.create_task(registry.check_quiescence_and_declare_termination())
        await asyncio.sleep(0.05)
        # The final quiescence check blocks on the writer's held acquisition.
        assert not check.done()
        lock.unpark.set()
        snapshot = await asyncio.wait_for(check, 2)
        # The writer completed before the check read: one consistent
        # post-write snapshot; the run stays a dependency until publication,
        # so no termination is declared.
        assert run_record.status is RunStatus.COMPLETED
        assert not snapshot.quiescent
        assert not registry.termination_declared
        # The terminal-transition sweep ran AFTER the write (serialized, not
        # concurrent with the held check).
        await _wait_for(
            lambda: any(
                trigger == SWEEP_TRIGGER_TERMINAL_TRANSITION
                and status is RunStatus.COMPLETED
                for trigger, status in seen
            )
        )
    finally:
        lock.armed = False
        lock.unpark.set()
        backend.release.set()
        await _cancel_sweep_task(registry)
        await registry.drain_children()
        await parent.aclose()
