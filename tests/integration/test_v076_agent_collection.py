"""v0.7.6 P2 integration: notification-first agent collection.

Co-owned by chunks 2a-2e as NAMED TEST CLASSES per the plan's applicability
mapping (plan P2 section): every design-matrix case is owned in every mode
where its mechanism exists.

- ``TestLifecycleQuiescence`` (chunk 2a): the three-condition
  atomic-quiescence snapshot in INTERACTIVE mode — the final quiescence
  check + termination declaration execute while HOLDING the terminal-state
  registry acquisition, mutually exclusive with every run-state writer, and
  the three conditions read as ONE consistent snapshot. Programmatic and
  post-restart quiescence coverage is 2c's classes.
- ``TestLifecycleSweep`` (chunk 2b): the reconciliation sweep + the
  interactive notification backstop in INTERACTIVE mode — a deliberately
  dropped notification in an idle window surfaced within one timer
  interval; sweep trigger coverage (all five triggers); a terminal
  transition during the final quiescence check; completion during the
  final drain; idle transition; cancellation; queued user input
  interleaving; a never-completing child's budget stop surfaced by the
  sweep — exactly one surfaced notification per event, and no delivery
  marking written by the sweep.
- ``TestLifecycleProgrammatic`` + ``TestLifecycleRestart`` (chunk 2c): the
  harness-owned continuation lifecycle in PROGRAMMATIC mode (one act()
  stream, then ``run_continuation``) and POST-RESTART mode (a resumed
  session over the persisted transcript) — the three-condition atomic
  quiescence in both modes; sweep triggers including the dropped-idle-
  notification and the transition-during-final-check in both modes;
  completion during the final drain; idle transition; cancellation; queued
  input; the never-completing child's budget stop; cap/STOP with unfetched
  results force-surfaced + named cancelled runs; approval-callback
  auto-deny; the pre-subscription completion race; the sweep across the
  act() boundary; restart with owned runs (transcript-derived orphan
  recovery notice); and the post-restart compaction-between-completion-
  and-delivery variant.
- ``TestDeliveryStates`` (chunk 2d): the digest/full delivery states in
  INTERACTIVE and PROGRAMMATIC routes — the marking committed atomically
  with the transcript write (per-message ``result_delivery`` envelope key),
  cross-route marking equality, the mechanical refusal on a repeat
  same-mode fetch, the refusal lift + re-arm after a compaction boundary,
  and compaction-between-completion-and-delivery in both routes.
- ``TestDigestSufficiency`` (chunk 2e): the worker-digest contract's
  decision-sufficiency in the fixture class, per route (INTERACTIVE and
  PROGRAMMATIC) — the digest assembled from the retained run via the
  shared ``build_worker_digest`` suffices for a reviewer to adjudicate
  (run identity, disposition, the always-contains disclosure list) without
  any full-body fetch; no ``full`` marking is ever written.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from io import StringIO
import json
from pathlib import Path
import time
from typing import Any, Literal, cast

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._subagent_execution import (
    SWEEP_TRIGGER_CONTINUATION_START,
    SWEEP_TRIGGER_DRAIN_COMPLETE,
    SWEEP_TRIGGER_TERMINAL_TRANSITION,
    SWEEP_TRIGGER_TIMER,
)
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    OpenCallbackState,
    PublicCallbackEntry,
    PublicEntryGenerationStatus,
    QuestionChoice,
    SessionTextContentBlock,
    TurnUserInputEntry,
    UserInputCallbackDetail,
    UserQuestion,
    UserQuestionRequest,
)
from chartreux.app_server.protocol import TurnEnqueueParams
from chartreux.app_server.server import AppServer
from chartreux.cli.programmatic import (
    ContinuationStopReason,
    OutputFormat,
    ProgrammaticOutput,
    _detect_restart_orphans,
    _render_terminal_notice,
    run_continuation,
)
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.subagents import (
    WORKER_DIGEST_ALWAYS_CONTAINS,
    AgentAvailability,
    RunStatus,
    RunStopReason,
    WorkerDigest,
    WorkerDigestDisclosures,
    build_worker_digest,
)
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _config,
    _task_call,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


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


async def _interactive_launch(
    monkeypatch: pytest.MonkeyPatch, child_backend: BlockingBackend
) -> tuple[Any, Any, SessionRuntimeRegistry, str, str]:
    """Drive one interactive parent turn that launches a blocked background child."""
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call(background=True, retain=True)]
                )
            ],
            [mock_llm_chunk(content="Parent completed")],
            [mock_llm_chunk(content="Notification received")],
        ]),
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    _ = [event async for event in session.act("Delegate this")]
    await asyncio.wait_for(child_backend.started.wait(), 2)
    registry = legacy_backend(server).children
    summaries = await registry.check_agents()
    assert len(summaries) == 1
    agent_id = summaries[0].agent_id
    run_id = summaries[0].current_run_id
    assert run_id is not None
    return session, parent, registry, agent_id, run_id


class TestLifecycleQuiescence:
    """2a: the three-condition atomic-quiescence snapshot, INTERACTIVE mode."""

    @pytest.mark.asyncio
    async def test_interactive_three_condition_atomic_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch(
            monkeypatch, child_backend
        )
        try:
            # (i) while the child run is active, the snapshot reads the active
            # dependency run and is not quiescent; no termination declared.
            snapshot = await registry.quiescence_snapshot()
            assert snapshot.active_dependency_runs == ((agent_id, run_id),)
            assert not snapshot.quiescent
            assert not registry.termination_declared
            final = await registry.check_quiescence_and_declare_termination()
            assert not final.quiescent
            assert not registry.termination_declared

            child_backend.release.set()
            result = await asyncio.wait_for(_wait_result(registry, agent_id, run_id), 2)
            assert result.completed
            # Fetch the result so condition (iii) clears once notified.
            fetched = await registry.get_agent_result(agent_id, run_id)
            assert fetched == result

            async def settle() -> Any:
                while True:
                    snap = await registry.quiescence_snapshot()
                    if snap.quiescent:
                        return snap
                    await asyncio.sleep(0.01)

            snapshot = await asyncio.wait_for(settle(), 5)
            assert snapshot.quiescent
            assert snapshot.active_dependency_runs == ()
            assert snapshot.pending_notifications == ()
            assert snapshot.unnotified_unfetched_results == ()
            # The final check + declaration under the held acquisition.
            final = await registry.check_quiescence_and_declare_termination()
            assert final.quiescent
            assert registry.termination_declared
        finally:
            child_backend.release.set()
            await session.close()
            await parent.aclose()

    @pytest.mark.asyncio
    async def test_interactive_check_holds_the_acquisition_against_writers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch(
            monkeypatch, child_backend
        )
        lock = _ParkingLock()
        registry._run_state_lock = lock  # type: ignore[assignment]
        try:
            monitor = next(iter(registry._monitor_tasks))
            lock.park_task = monitor
            lock.armed = True
            child_backend.release.set()
            # The writer parks INSIDE its terminal-write acquisition, holding
            # the serialization point before the write lands.
            await asyncio.wait_for(lock.parked.wait(), 2)
            run_record = registry._agent_records[agent_id].current_run
            assert run_record is not None
            assert run_record.status is RunStatus.RUNNING
            check = asyncio.create_task(
                registry.check_quiescence_and_declare_termination()
            )
            await asyncio.sleep(0.05)
            # The final quiescence check + declaration block on the writer's
            # held acquisition — mutual exclusion, not a re-check.
            assert not check.done()
            lock.unpark.set()
            snapshot = await asyncio.wait_for(check, 2)
            assert run_record.status is RunStatus.COMPLETED
            # One consistent post-write snapshot; the run stays a dependency
            # until publication, so no termination is declared.
            assert not snapshot.quiescent
            assert not registry.termination_declared
            await asyncio.wait_for(_wait_result(registry, agent_id, run_id), 2)
        finally:
            lock.armed = False
            lock.unpark.set()
            child_backend.release.set()
            await session.close()
            await parent.aclose()


# ---------------------------------------------------------------------------
# Chunk 2b helpers: notification observation, polling, and a launch variant
# with explicit parent streams / run-budget price.
# ---------------------------------------------------------------------------


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(getattr(block, "text", ""))
            for block in content
            if hasattr(block, "text")
        )
    return ""


def _notification_texts(parent: Any, run_id: str) -> list[str]:
    """Every surfaced completion notification for one run (any channel)."""
    texts: list[str] = []
    for message in parent.messages:
        text = _message_text(message)
        if text.startswith("Background agent") and f"run {run_id}" in text:
            texts.append(text)
    for message in parent._pending_injected_messages:
        text = _message_text(message)
        if text.startswith("Background agent") and f"run {run_id}" in text:
            texts.append(text)
    return texts


async def _wait_for(predicate, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met within timeout")


async def _drain_act(session: Any, text: str) -> None:
    _ = [event async for event in session.act(text)]


async def _cancel_sweep_task(registry: SessionRuntimeRegistry) -> None:
    task = registry._execution._sweep_task
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def _root_turns(registry: SessionRuntimeRegistry) -> Any:
    root = registry._root
    assert root is not None
    return root.turns


async def _interactive_launch_ex(
    monkeypatch: pytest.MonkeyPatch,
    child_backend: Any,
    *,
    parent_streams: list[list[Any]] | None = None,
    parent_backend: FakeBackend | None = None,
    run_max_price: float | None = None,
    sweep_interval: float = 0.05,
    config: Any = None,
) -> tuple[Any, Any, SessionRuntimeRegistry, str, str]:
    """Drive one interactive parent turn that launches a background child."""
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    if parent_backend is None:
        parent_backend = FakeBackend(
            parent_streams
            or [
                [
                    mock_llm_chunk(
                        content="",
                        tool_calls=[_task_call(background=True, retain=True)],
                    )
                ],
                [mock_llm_chunk(content="Parent completed")],
                [mock_llm_chunk(content="Notification received")],
            ]
        )
    parent = build_test_agent_loop(config=config or _config(), backend=parent_backend)
    if run_max_price is not None:
        # Live-config mutation: children created after this copy the budget.
        parent.config.subagents.run_max_price = run_max_price
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry = legacy_backend(server).children
    # Set the sweep interval BEFORE the launch so the armed timer's very
    # first sleep already uses the test interval.
    registry._execution._sweep_interval = sweep_interval
    _ = [event async for event in session.act("Delegate this")]
    if hasattr(child_backend, "started"):
        await asyncio.wait_for(child_backend.started.wait(), 2)
    summaries = await registry.check_agents()
    assert len(summaries) == 1
    agent_id = summaries[0].agent_id
    run_id = summaries[0].current_run_id
    assert run_id is not None
    return session, parent, registry, agent_id, run_id


class _GatedParentBackend(FakeBackend):
    """Gates the LLM request with 1-based number ``gate_request`` until released."""

    def __init__(
        self, streams: list[list[Any]], *, gate_request: int, release: asyncio.Event
    ) -> None:
        super().__init__(streams)
        self._gate_request = gate_request
        self._release = release
        self.gate_arrived = asyncio.Event()

    async def _maybe_gate(self) -> None:
        # The request about to be recorded is number len(requests) + 1.
        if len(self._requests_messages) + 1 == self._gate_request:
            self.gate_arrived.set()
            await self._release.wait()

    async def complete(self, **kwargs: Any):
        await self._maybe_gate()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any):
        await self._maybe_gate()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


class _GatedBudgetChildBackend(FakeBackend):
    """Unresolved-tool-call child drive, gated before the first response.

    The run budget (tiny ``run_max_price``) stops the child's second turn, so
    the child never completes; the gate keeps the suppression window
    deterministic (added after ``started``, before any publication).
    """

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
        self.started = asyncio.Event()
        self.tool_release = asyncio.Event()

    async def complete(self, **kwargs: Any):
        self.started.set()
        await self.tool_release.wait()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any):
        self.started.set()
        await self.tool_release.wait()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


class TestLifecycleSweep:
    """2b: the reconciliation sweep + the interactive notification backstop."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_dropped_notification_in_idle_window_surfaced_within_one_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch(
            monkeypatch, child_backend
        )
        registry._execution._sweep_interval = 0.05
        key = (agent_id, run_id)
        try:
            # Deliberately drop the delivered notification at publish time:
            # the suppression gate stays silent and leaks no second channel.
            registry._suppressed_notifications.add(agent_id)
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            stored_at = time.monotonic()
            assert _notification_texts(parent, run_id) == []
            assert key not in registry._execution._delivered_notification_keys
            # The sweep surfaces it within one timer interval (idle window).
            await _wait_for(lambda: _notification_texts(parent, run_id))
            surfaced_at = time.monotonic()
            texts = _notification_texts(parent, run_id)
            assert len(texts) == 1
            assert "completed" in texts[0]
            assert surfaced_at - stored_at <= 0.05 + 0.5
            # Exactly once across further intervals; no delivery marking.
            await asyncio.sleep(0.2)
            assert len(_notification_texts(parent, run_id)) == 1
            assert key in registry._execution._sweep_surfaced
            assert key not in registry._collected_results
            assert not registry._pending_notifications
        finally:
            child_backend.release.set()
            registry._suppressed_notifications.clear()
            await _cancel_sweep_task(registry)
            await session.close()
            await parent.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_delivered_notification_is_the_single_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch(
            monkeypatch, child_backend
        )
        registry._execution._sweep_interval = 0.05
        key = (agent_id, run_id)
        try:
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            await _wait_for(lambda: key not in registry._pending_notifications)
            await _wait_for(lambda: _notification_texts(parent, run_id))
            assert len(_notification_texts(parent, run_id)) == 1
            # The delivered path recorded the event; the sweep never
            # double-messages it, even with the result uncollected.
            assert key in registry._execution._delivered_notification_keys
            # Longer than the timer's first (default-interval) tick.
            await asyncio.sleep(0.4)
            assert len(_notification_texts(parent, run_id)) == 1
            assert key not in registry._execution._sweep_surfaced
            assert key not in registry._collected_results
            # Sweep trigger coverage across the interactive lifecycle.
            seen = registry._execution._sweep_triggers_seen
            assert {
                SWEEP_TRIGGER_CONTINUATION_START,
                SWEEP_TRIGGER_TERMINAL_TRANSITION,
                SWEEP_TRIGGER_DRAIN_COMPLETE,
                SWEEP_TRIGGER_TIMER,
            } <= seen
        finally:
            child_backend.release.set()
            await _cancel_sweep_task(registry)
            await session.close()
            await parent.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_terminal_transition_during_final_quiescence_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch(
            monkeypatch, child_backend
        )
        lock = _ParkingLock()
        registry._run_state_lock = lock  # type: ignore[assignment]
        try:
            monitor = next(iter(registry._monitor_tasks))
            lock.park_task = monitor
            lock.armed = True
            child_backend.release.set()
            # The writer parks INSIDE its terminal-write acquisition.
            await asyncio.wait_for(lock.parked.wait(), 2)
            run_record = registry._agent_records[agent_id].current_run
            assert run_record is not None
            assert run_record.status is RunStatus.RUNNING
            check = asyncio.create_task(
                registry.check_quiescence_and_declare_termination()
            )
            await asyncio.sleep(0.05)
            # The final quiescence check blocks on the writer's held
            # acquisition — the writer either completes before the read or
            # waits; serialization holds.
            assert not check.done()
            lock.unpark.set()
            snapshot = await asyncio.wait_for(check, 2)
            assert run_record.status is RunStatus.COMPLETED
            assert not snapshot.quiescent
            assert not registry.termination_declared
            # The terminal-transition sweep ran (serialized after the write).
            assert SWEEP_TRIGGER_TERMINAL_TRANSITION in (
                registry._execution._sweep_triggers_seen
            )
            await asyncio.wait_for(_wait_result(registry, agent_id, run_id), 2)
        finally:
            lock.armed = False
            lock.unpark.set()
            child_backend.release.set()
            await _cancel_sweep_task(registry)
            await session.close()
            await parent.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_completion_during_final_drain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        # The parent's second-turn LLM request (request 3) blocks until the
        # child has completed, so the completion lands mid-turn and the
        # notification queues as a pending injection for the final drain.
        parent_release = asyncio.Event()
        parent_backend = _GatedParentBackend(
            [
                [
                    mock_llm_chunk(
                        content="",
                        tool_calls=[_task_call(background=True, retain=True)],
                    )
                ],
                [mock_llm_chunk(content="Parent completed")],
                [mock_llm_chunk(content="Second turn")],
                [mock_llm_chunk(content="Notification received")],
            ],
            gate_request=3,
            release=parent_release,
        )
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch, child_backend, parent_backend=parent_backend
        )
        key = (agent_id, run_id)
        try:
            act2 = asyncio.create_task(_drain_act(session, "again"))
            await _wait_for(lambda: parent_backend.gate_arrived.is_set())
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            parent_release.set()
            await asyncio.wait_for(act2, 10)
            await _wait_for(lambda: _notification_texts(parent, run_id))
            texts = _notification_texts(parent, run_id)
            assert len(texts) == 1
            assert "completed" in texts[0]
            await asyncio.sleep(0.3)
            assert len(_notification_texts(parent, run_id)) == 1
            assert key not in registry._collected_results
        finally:
            child_backend.release.set()
            parent_release.set()
            await _cancel_sweep_task(registry)
            await session.close()
            await parent.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_queued_user_input_interleaving(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        parent_release = asyncio.Event()
        parent_backend = _GatedParentBackend(
            [
                [
                    mock_llm_chunk(
                        content="",
                        tool_calls=[_task_call(background=True, retain=True)],
                    )
                ],
                [mock_llm_chunk(content="Parent completed")],
                [mock_llm_chunk(content="Second turn")],
                [mock_llm_chunk(content="Notification received")],
                [mock_llm_chunk(content="Queued response")],
            ],
            gate_request=3,
            release=parent_release,
        )
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch, child_backend, parent_backend=parent_backend
        )
        key = (agent_id, run_id)
        try:
            act2 = asyncio.create_task(_drain_act(session, "again"))
            await _wait_for(lambda: parent_backend.gate_arrived.is_set())
            # Queue user input while the turn runs: the completion
            # notification must not start a competing turn.
            _root_turns(registry).enqueue(
                TurnEnqueueParams(
                    session_id=parent.session_id,
                    entries=[
                        TurnUserInputEntry(
                            content=[SessionTextContentBlock(text="queued input")]
                        )
                    ],
                )
            )
            await _wait_for(lambda: _root_turns(registry).queue_state.items)
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            parent_release.set()
            await asyncio.wait_for(act2, 10)
            await _wait_for(lambda: _notification_texts(parent, run_id))
            texts = _notification_texts(parent, run_id)
            assert len(texts) == 1
            assert "completed" in texts[0]
            await asyncio.sleep(0.3)
            assert len(_notification_texts(parent, run_id)) == 1
        finally:
            child_backend.release.set()
            parent_release.set()
            await _cancel_sweep_task(registry)
            await session.close()
            await parent.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_idle_transition_after_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch(
            monkeypatch, child_backend
        )
        registry._execution._sweep_interval = 0.05
        key = (agent_id, run_id)
        try:
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            await _wait_for(lambda: _notification_texts(parent, run_id))
            # The retained agent transitions to idle with the result still
            # uncollected — visible in the inventory, notified exactly once.
            summaries = await registry.check_agents()
            assert summaries[0].availability is AgentAvailability.IDLE
            assert run_id in summaries[0].uncollected_run_ids
            assert len(_notification_texts(parent, run_id)) == 1
            assert key not in registry._collected_results
        finally:
            child_backend.release.set()
            await _cancel_sweep_task(registry)
            await session.close()
            await parent.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_cancellation_surfaced_by_the_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch(
            monkeypatch, child_backend
        )
        registry._execution._sweep_interval = 0.05
        key = (agent_id, run_id)
        try:
            # Drop the delivered notification, then cancel the run.
            registry._suppressed_notifications.add(agent_id)
            outcome = await registry.cancel_run(
                agent_id,
                reason=RunStopReason.USER_CANCELLED,
                requester_session_id=parent.session_id,
            )
            assert outcome.outcome is not None
            await _wait_for(lambda: key in registry._result_store)
            stored_at = time.monotonic()
            record = registry._agent_records[agent_id]
            run = next(
                (r for r in record.run_history if r.run_id == run_id),
                record.current_run,
            )
            assert run is not None
            assert run.stop_reason is RunStopReason.USER_CANCELLED
            assert _notification_texts(parent, run_id) == []
            # The sweep surfaces the cancellation within one timer interval.
            await _wait_for(lambda: _notification_texts(parent, run_id))
            texts = _notification_texts(parent, run_id)
            assert len(texts) == 1
            assert "cancelled (by user)" in texts[0]
            assert time.monotonic() - stored_at <= 0.05 + 0.5
            await asyncio.sleep(0.15)
            assert len(_notification_texts(parent, run_id)) == 1
            assert key in registry._execution._sweep_surfaced
            assert key not in registry._collected_results
            assert not registry._pending_notifications
        finally:
            child_backend.release.set()
            registry._suppressed_notifications.clear()
            await _cancel_sweep_task(registry)
            await session.close()
            await parent.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_never_completing_child_budget_stop_surfaced_by_the_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = _GatedBudgetChildBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch, child_backend, run_max_price=0.00001
        )
        registry._execution._sweep_interval = 0.05
        key = (agent_id, run_id)
        try:
            # Drop the delivered notification (deterministic: the child's
            # first response is still gated, so no publication has run).
            registry._suppressed_notifications.add(agent_id)
            child_backend.tool_release.set()
            await _wait_for(lambda: key in registry._result_store)
            stored_at = time.monotonic()
            record = registry._agent_records[agent_id]
            run = next(
                (r for r in record.run_history if r.run_id == run_id),
                record.current_run,
            )
            assert run is not None
            # The never-completing child is stopped by its run budget.
            assert run.stop_reason is RunStopReason.BUDGET_EXCEEDED
            assert _notification_texts(parent, run_id) == []
            # The sweep surfaces the budget stop within one timer interval.
            await _wait_for(lambda: _notification_texts(parent, run_id))
            texts = _notification_texts(parent, run_id)
            assert len(texts) == 1
            assert "budget-stopped (budget exceeded)" in texts[0]
            assert time.monotonic() - stored_at <= 0.05 + 0.5
            await asyncio.sleep(0.15)
            assert len(_notification_texts(parent, run_id)) == 1
            assert key in registry._execution._sweep_surfaced
            assert key not in registry._collected_results
            assert not registry._pending_notifications
        finally:
            child_backend.tool_release.set()
            registry._suppressed_notifications.clear()
            await _cancel_sweep_task(registry)
            await session.close()
            await parent.aclose()


# ---------------------------------------------------------------------------
# Chunk 2c helpers: the programmatic-mode continuation (one act() stream,
# then run_continuation) and the post-restart harness (crash + resume over
# the persisted transcript in a fresh in-memory registry).
# ---------------------------------------------------------------------------


def _fetch_call(agent_id: str = "agent-1", call_id: str = "call-fetch") -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="get_agent_result", arguments=json.dumps({"agent_id": agent_id})
        ),
    )


def _fetch_config() -> Any:
    """Parent config with the fetch tool enabled (the notification-driven
    turn calls get_agent_result).
    """
    return _config(enabled_tools=["task", "get_agent_result"])


def _task_stream(tool_call_id: str) -> list[Any]:
    return [
        mock_llm_chunk(
            content="",
            tool_calls=[
                _task_call(background=True, retain=True, tool_call_id=tool_call_id)
            ],
        )
    ]


def _fetch_streams(tool_call_id: str = "task-1") -> list[list[Any]]:
    """Parent streams: launch, complete, fetch on the notification-driven
    turn (across the act() boundary), then finish.
    """
    return [
        _task_stream(tool_call_id),
        [mock_llm_chunk(content="Parent completed")],
        [mock_llm_chunk(content="", tool_calls=[_fetch_call("agent-1")])],
        [mock_llm_chunk(content="Fetched.")],
    ]


def _noted_streams(tool_call_id: str = "task-1") -> list[list[Any]]:
    """Parent streams where the notification-driven turn does NOT fetch."""
    return [
        _task_stream(tool_call_id),
        [mock_llm_chunk(content="Parent completed")],
        [mock_llm_chunk(content="Noted.")],
    ]


async def _run_continuation(
    session: Any, registry: SessionRuntimeRegistry, *, cap_seconds: float = 5.0
) -> tuple[Any, ProgrammaticOutput]:
    """Run the harness-owned continuation after the single act() stream."""
    output = ProgrammaticOutput(OutputFormat.TEXT, stream=StringIO())
    outcome = await run_continuation(
        session, output, registry=registry, cap_seconds=cap_seconds
    )
    return outcome, output


def _callback_entry(session_id: str) -> PublicCallbackEntry:
    return PublicCallbackEntry(
        id="callback:callback-1",
        session_id=session_id,
        turn_id=None,
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        callback_id="callback-1",
        title="Approval required",
        detail=UserInputCallbackDetail(
            request=UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="Approve?",
                        options=[
                            QuestionChoice(label="Yes"),
                            QuestionChoice(label="No"),
                        ],
                    )
                ]
            )
        ),
        state=OpenCallbackState(),
    )


async def _teardown(
    session: Any, parent: Any, registry: SessionRuntimeRegistry, child_backend: Any
) -> None:
    if hasattr(child_backend, "release"):
        child_backend.release.set()
    if hasattr(child_backend, "tool_release"):
        child_backend.tool_release.set()
    registry._suppressed_notifications.clear()
    await _cancel_sweep_task(registry)
    await session.close()
    await parent.aclose()


class TestLifecycleProgrammatic:
    """2c: the harness-owned continuation lifecycle, PROGRAMMATIC mode."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_three_condition_atomic_quiescence_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch,
            child_backend,
            parent_streams=_fetch_streams(),
            config=_fetch_config(),
        )
        try:
            # (i) while the dependency run is active, the snapshot reads it
            # and no termination is declared.
            snapshot = await registry.quiescence_snapshot()
            assert snapshot.active_dependency_runs == ((agent_id, run_id),)
            assert not snapshot.quiescent
            assert not registry.termination_declared
            child_backend.release.set()
            outcome, _output = await _run_continuation(session, registry)
            # Termination only at 2a's atomic quiescence check.
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert registry.termination_declared
            snapshot = await registry.quiescence_snapshot()
            assert snapshot.quiescent
            assert snapshot.active_dependency_runs == ()
            assert snapshot.pending_notifications == ()
            assert snapshot.unnotified_unfetched_results == ()
            assert (agent_id, run_id) in registry._collected_results
            # The notification-driven turn ran ACROSS the act() boundary.
            assert len(session.state.turns or []) >= 2
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_dropped_idle_notification_swept_within_one_interval_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch,
            child_backend,
            parent_streams=_fetch_streams(),
            config=_fetch_config(),
        )
        key = (agent_id, run_id)
        try:
            # Drop the delivered notification: the continuation-lifecycle
            # timer sweep surfaces it within one interval (idle window).
            registry._suppressed_notifications.add(agent_id)
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            stored_at = time.monotonic()
            assert _notification_texts(parent, run_id) == []
            outcome, _output = await _run_continuation(session, registry)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert SWEEP_TRIGGER_TIMER in registry._execution._sweep_triggers_seen
            assert key in registry._execution._sweep_surfaced
            texts = _notification_texts(parent, run_id)
            assert len(texts) == 1
            assert "completed" in texts[0]
            assert time.monotonic() - stored_at <= 5.0
            assert key not in registry._collected_results or True
            assert key in registry._collected_results  # fetched on the turn
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_transition_during_final_quiescence_check_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch,
            child_backend,
            parent_streams=_fetch_streams(),
            config=_fetch_config(),
        )
        lock = _ParkingLock()
        registry._run_state_lock = lock  # type: ignore[assignment]
        try:
            monitor = next(iter(registry._monitor_tasks))
            lock.park_task = monitor
            lock.armed = True
            child_backend.release.set()
            # The writer parks INSIDE its terminal-write acquisition.
            await asyncio.wait_for(lock.parked.wait(), 2)
            task = asyncio.create_task(_run_continuation(session, registry))
            await asyncio.sleep(0.05)
            # The continuation's final quiescence check blocks on the
            # writer's held acquisition — mutual exclusion, not a re-check.
            assert not task.done()
            lock.unpark.set()
            outcome, _output = await asyncio.wait_for(task, 10)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert SWEEP_TRIGGER_TERMINAL_TRANSITION in (
                registry._execution._sweep_triggers_seen
            )
            assert registry.termination_declared
        finally:
            lock.armed = False
            lock.unpark.set()
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_completion_during_final_drain_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        parent_release = asyncio.Event()
        parent_backend = _GatedParentBackend(
            _fetch_streams(), gate_request=2, release=parent_release
        )
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
        )
        parent = build_test_agent_loop(config=_fetch_config(), backend=parent_backend)
        client = start_test_app_server(parent)
        run_peer = client._run_peer
        assert run_peer is not None
        server = cast(AppServer, cast(Any, run_peer).__self__)
        session = await attach_test_app_server_session(client)
        registry = legacy_backend(server).children
        registry._execution._sweep_interval = 0.05
        # The act() stream itself is gated: the child completes while the
        # single act() turn is still draining, and the notification is
        # dropped in the drain window.
        registry._suppressed_notifications.add("agent-1")
        act_task = asyncio.create_task(_drain_act(session, "Delegate this"))
        try:
            await _wait_for(lambda: parent_backend.gate_arrived.is_set())
            await asyncio.wait_for(child_backend.started.wait(), 2)
            summaries = await registry.check_agents()
            assert len(summaries) == 1
            agent_id = summaries[0].agent_id
            run_id = summaries[0].current_run_id
            assert run_id is not None
            key = (agent_id, run_id)
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            parent_release.set()
            await asyncio.wait_for(act_task, 10)
            # The completion landed during the final drain, before the
            # continuation even started.
            outcome, _output = await _run_continuation(session, registry)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert key in registry._execution._sweep_surfaced
            assert key in registry._collected_results
        finally:
            parent_release.set()
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_idle_transition_and_cap_force_surface_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch, child_backend, parent_streams=_noted_streams()
        )
        key = (agent_id, run_id)
        try:
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)

            async def idle_with_uncollected() -> None:
                while True:
                    summaries = await registry.check_agents()
                    if (
                        summaries[0].availability is AgentAvailability.IDLE
                        and run_id in summaries[0].uncollected_run_ids
                    ):
                        return
                    await asyncio.sleep(0.01)

            # The retained agent transitions to idle with the result still
            # uncollected (visible in the inventory before the cap).
            await asyncio.wait_for(idle_with_uncollected(), 5)
            # The continuation holds the session open until the wall-clock
            # cap, then force-surfaces the full body.
            outcome, _output = await _run_continuation(
                session, registry, cap_seconds=1.0
            )
            assert outcome.stop_reason is ContinuationStopReason.WALL_CLOCK_CAP
            assert len(outcome.surfaced) == 1
            assert outcome.surfaced[0].disposition == "full"
            assert outcome.surfaced[0].body == "Child completed"
            notice = _render_terminal_notice(outcome)
            assert notice is not None and "wall-clock-cap" in notice
            assert "Child completed" in notice
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_cancellation_during_continuation_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch, child_backend, parent_streams=_noted_streams()
        )
        key = (agent_id, run_id)
        try:
            task = asyncio.create_task(_run_continuation(session, registry))
            await asyncio.sleep(0.05)
            outcome = await registry.cancel_run(
                agent_id,
                run_id,
                reason=RunStopReason.USER_CANCELLED,
                requester_session_id=parent.session_id,
            )
            assert outcome.outcome is not None
            await _wait_for(lambda: key in registry._result_store)
            # The collection (the parent's fetch) clears condition (iii).
            await registry.get_agent_result(agent_id, run_id)
            result, _output = await asyncio.wait_for(task, 10)
            assert result.stop_reason is ContinuationStopReason.QUIESCENT
            record = registry._agent_records[agent_id]
            run = next(
                (r for r in record.run_history if r.run_id == run_id),
                record.current_run,
            )
            assert run is not None and run.stop_reason is RunStopReason.USER_CANCELLED
            texts = _notification_texts(parent, run_id)
            assert len(texts) == 1 and "cancelled (by user)" in texts[0]
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_queued_input_interleaving_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch,
            child_backend,
            parent_streams=[
                *_noted_streams(),
                [mock_llm_chunk(content="", tool_calls=[_fetch_call("agent-1")])],
                [mock_llm_chunk(content="Fetched.")],
            ],
            config=_fetch_config(),
        )
        key = (agent_id, run_id)
        try:
            task = asyncio.create_task(_run_continuation(session, registry))
            await asyncio.sleep(0.05)
            # Queue user input while the continuation waits: the queued turn
            # runs inside the continuation lifecycle.
            _result, after_enqueue = _root_turns(registry).enqueue(
                TurnEnqueueParams(
                    session_id=parent.session_id,
                    entries=[
                        TurnUserInputEntry(
                            content=[SessionTextContentBlock(text="queued input")]
                        )
                    ],
                )
            )
            assert after_enqueue is not None
            await _wait_for(lambda: _root_turns(registry).queue_state.items)
            # The server invokes the after-response callback (DispatchResult)
            # once the enqueue response is sent; that is what promotes the
            # queued turn when no act() is in flight.
            after_enqueue()
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            outcome, _output = await asyncio.wait_for(task, 10)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            # act() turn + the queued turn, which ran inside the continuation
            # (the completion notification steered into it rather than
            # starting a competing turn, and fetched the result there).
            assert len(session.state.turns or []) >= 2
            assert not _root_turns(registry).queue_state.items
            assert key in registry._collected_results
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_never_completing_child_budget_stop_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = _GatedBudgetChildBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch,
            child_backend,
            parent_streams=_fetch_streams(),
            config=_fetch_config(),
            run_max_price=0.00001,
        )
        key = (agent_id, run_id)
        try:
            registry._suppressed_notifications.add(agent_id)
            child_backend.tool_release.set()
            await _wait_for(lambda: key in registry._result_store)
            outcome, _output = await _run_continuation(session, registry)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            record = registry._agent_records[agent_id]
            run = next(
                (r for r in record.run_history if r.run_id == run_id),
                record.current_run,
            )
            assert run is not None
            # The never-completing child stops at its run budget.
            assert run.stop_reason is RunStopReason.BUDGET_EXCEEDED
            texts = _notification_texts(parent, run_id)
            assert len(texts) == 1 and "budget-stopped (budget exceeded)" in texts[0]
            assert key in registry._collected_results
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_cap_names_cancelled_runs_separately_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()  # never released
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch, child_backend, parent_streams=_noted_streams()
        )
        try:
            outcome, _output = await _run_continuation(
                session, registry, cap_seconds=0.5
            )
            assert outcome.stop_reason is ContinuationStopReason.WALL_CLOCK_CAP
            # The active dependency run was cancelled via the registry cancel
            # path and is named separately from the surfaced results.
            assert outcome.cancelled == [(agent_id, run_id, "orchestrator_cancelled")]
            snapshot = await registry.quiescence_snapshot()
            assert snapshot.active_dependency_runs == ()
            record = registry._agent_records[agent_id]
            run = next(
                (r for r in record.run_history if r.run_id == run_id),
                record.current_run,
            )
            assert (
                run is not None
                and run.stop_reason is RunStopReason.ORCHESTRATOR_CANCELLED
            )
            notice = _render_terminal_notice(outcome)
            assert notice is not None
            assert "Cancelled dependency runs:" in notice
            assert "orchestrator_cancelled" in notice
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_approval_callback_during_continuation_auto_denied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch,
            child_backend,
            parent_streams=_fetch_streams(),
            config=_fetch_config(),
        )
        try:
            synthetic = CallbackRequested(_callback_entry(session.session_id))
            real_events = session.events
            denied: list[str] = []
            original_deny = session.deny_callback

            async def fake_events() -> Any:
                yield synthetic
                async for event in real_events():
                    yield event

            async def spy(callback: Any) -> None:
                denied.append(callback.callback_id)
                with suppress(Exception):
                    await original_deny(callback)

            session.events = fake_events  # type: ignore[assignment]
            session.deny_callback = spy  # type: ignore[assignment]
            child_backend.release.set()
            outcome, _output = await _run_continuation(session, registry)
            # The approval callback raised during the continuation was
            # auto-denied (documented: a programmatic session has no user).
            assert denied == ["callback-1"]
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_pre_subscription_completion_race_at_continuation_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch,
            child_backend,
            parent_streams=_fetch_streams(),
            config=_fetch_config(),
        )
        key = (agent_id, run_id)
        try:
            # The child completes BEFORE the continuation subscribes: the
            # first quiescence check sees the stranded result, and the sweep
            # at continuation start / on the timer surfaces it.
            registry._suppressed_notifications.add(agent_id)
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            outcome, _output = await _run_continuation(session, registry)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert key in registry._execution._sweep_surfaced
            assert SWEEP_TRIGGER_TIMER in registry._execution._sweep_triggers_seen
            assert key in registry._collected_results
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_sweep_executes_across_the_act_boundary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch,
            child_backend,
            parent_streams=_fetch_streams(),
            config=_fetch_config(),
        )
        key = (agent_id, run_id)
        try:
            child_backend.release.set()
            outcome, _output = await _run_continuation(session, registry)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            # The sweep's site is the registry, executing across turn
            # boundaries — not inside the single act() stream: the
            # continuation-start sweep ran at admission (inside act), and
            # the drain/timer sweeps ran after it returned.
            seen = registry._execution._sweep_triggers_seen
            assert SWEEP_TRIGGER_CONTINUATION_START in seen
            assert SWEEP_TRIGGER_DRAIN_COMPLETE in seen
            assert SWEEP_TRIGGER_TERMINAL_TRANSITION in seen
            # The notification-driven turn ran after act() returned.
            assert len(session.state.turns or []) >= 2
            assert key in registry._collected_results
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_mid_continuation_limit_stop_programmatic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
            monkeypatch, child_backend, parent_streams=_noted_streams()
        )
        key = (agent_id, run_id)
        try:
            # Any turn started after this STOPs at the turn limit (the
            # token/price-style STOP path): the notification-driven turn
            # hits it mid-continuation.
            parent.set_max_turns(0)
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            outcome, _output = await _run_continuation(session, registry)
            assert outcome.stop_reason is ContinuationStopReason.LIMIT_STOP
            # The completed-but-unfetched result is force-surfaced in the
            # terminal outcome before finalization.
            assert any(item.agent_id == agent_id for item in outcome.surfaced)
            notice = _render_terminal_notice(outcome)
            assert notice is not None and "limit-stop" in notice
            assert "Force-surfaced completed-but-unfetched results:" in notice
        finally:
            await _teardown(session, parent, registry, child_backend)


async def _restart_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pre_child_backend: Any,
    post_child_backend: Any,
    pre_streams: list[list[Any]],
    post_streams: list[list[Any]],
    post_run_max_price: float | None = None,
) -> tuple[Any, Any, SessionRuntimeRegistry, str, str]:
    """Crash + restart: one act() launches a background child over a
    PERSISTED session; the harness closes; a FRESH server/registry resumes
    the same persisted session (post-restart mode).
    """
    logging_cfg = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    config = _config(logging_cfg)
    holder = {"backend": pre_child_backend}
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: holder["backend"]
    )
    parent = build_test_agent_loop(config=config, backend=FakeBackend(pre_streams))
    client = start_test_app_server(parent)
    server = cast(AppServer, cast(Any, client._run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry1 = legacy_backend(server).children
    registry1._execution._sweep_interval = 0.05
    _ = [event async for event in session.act("Delegate this")]
    if hasattr(pre_child_backend, "started"):
        await asyncio.wait_for(pre_child_backend.started.wait(), 2)
    summaries = await registry1.check_agents()
    assert len(summaries) == 1
    pre_agent_id = summaries[0].agent_id
    session_id = session.session_id
    await session.close()
    await parent.aclose()
    # Restart into a fresh in-memory registry over the persisted transcript.
    holder["backend"] = post_child_backend
    resumed_config = config
    if post_run_max_price is not None:
        # The run budget is copied at child creation, so the resumed loop's
        # config must carry the lowered budget from the start.
        resumed_config = config.model_copy(
            update={
                "subagents": config.subagents.model_copy(
                    update={"run_max_price": post_run_max_price}
                )
            }
        )
    resumed_loop = build_test_agent_loop(
        config=resumed_config, backend=FakeBackend(post_streams)
    )
    resumed_client = start_test_app_server(resumed_loop)
    resumed_server = cast(AppServer, cast(Any, resumed_client._run_peer).__self__)
    resumed = await attach_test_app_server_session(
        resumed_client, resume_session_id=session_id
    )
    registry2 = legacy_backend(resumed_server).children
    registry2._execution._sweep_interval = 0.05
    return resumed, resumed_loop, registry2, session_id, pre_agent_id


class TestLifecycleRestart:
    """2c: the continuation lifecycle in POST-RESTART mode (a resumed session
    over the persisted transcript, fresh in-memory registry).
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_restart_with_owned_runs_recovery_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pre_child = BlockingBackend()
        post_child = BlockingBackend()
        (
            resumed,
            resumed_loop,
            registry,
            session_id,
            pre_agent_id,
        ) = await _restart_harness(
            monkeypatch,
            tmp_path,
            pre_child,
            post_child,
            _noted_streams(),
            [[mock_llm_chunk(content="Resumed reply")]],
        )
        try:
            # Transcript-derived orphan detection: the pre-crash background
            # run has no live record in the fresh registry.
            notice = _detect_restart_orphans(resumed, registry)
            assert notice is not None
            assert "TERMINAL-UNKNOWN" in notice
            assert "task-1" in notice
            assert "unrecoverable" in notice
            _ = [event async for event in resumed.act("Continue post-restart")]
            outcome, _output = await _run_continuation(resumed, registry)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            outcome.recovery_notice = notice
            rendered = _render_terminal_notice(outcome)
            assert rendered is not None
            assert "Recovery notice" in rendered and "TERMINAL-UNKNOWN" in rendered
        finally:
            await _teardown(resumed, resumed_loop, registry, post_child)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_post_restart_three_condition_quiescence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pre_child = BlockingBackend()
        post_child = BlockingBackend()
        (
            resumed,
            resumed_loop,
            registry,
            session_id,
            pre_agent_id,
        ) = await _restart_harness(
            monkeypatch,
            tmp_path,
            pre_child,
            post_child,
            _noted_streams(),
            _noted_streams("task-2"),
        )
        try:
            _ = [event async for event in resumed.act("Delegate post-restart")]
            summaries = await registry.check_agents()
            assert len(summaries) == 1
            agent_id = summaries[0].agent_id
            run_id = summaries[0].current_run_id
            assert run_id is not None
            snapshot = await registry.quiescence_snapshot()
            assert snapshot.active_dependency_runs == ((agent_id, run_id),)
            assert not snapshot.quiescent and not registry.termination_declared
            post_child.release.set()
            key = (agent_id, run_id)
            await _wait_for(lambda: key in registry._result_store)
            task = asyncio.create_task(_run_continuation(resumed, registry))
            await _wait_for(lambda: key in registry._result_store)
            await registry.get_agent_result(agent_id, run_id)  # the fetch
            outcome, _output = await asyncio.wait_for(task, 10)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert registry.termination_declared
            snapshot = await registry.quiescence_snapshot()
            assert snapshot.quiescent
            # The pre-restart orphan did not block quiescence (not a
            # dependency: no live registry record).
            assert (pre_agent_id, run_id) not in snapshot.active_dependency_runs
        finally:
            await _teardown(resumed, resumed_loop, registry, post_child)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_post_restart_dropped_idle_notification_swept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pre_child = BlockingBackend()
        post_child = BlockingBackend()
        resumed, resumed_loop, registry, session_id, _pre = await _restart_harness(
            monkeypatch,
            tmp_path,
            pre_child,
            post_child,
            _noted_streams(),
            _noted_streams("task-2"),
        )
        try:
            _ = [event async for event in resumed.act("Delegate post-restart")]
            summaries = await registry.check_agents()
            agent_id = summaries[0].agent_id
            run_id = summaries[0].current_run_id
            assert run_id is not None
            key = (agent_id, run_id)
            registry._suppressed_notifications.add(agent_id)
            post_child.release.set()
            await _wait_for(lambda: key in registry._result_store)
            task = asyncio.create_task(_run_continuation(resumed, registry))
            await _wait_for(lambda: key in registry._execution._sweep_surfaced)
            await registry.get_agent_result(agent_id, run_id)
            outcome, _output = await asyncio.wait_for(task, 10)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert SWEEP_TRIGGER_TIMER in registry._execution._sweep_triggers_seen
            texts = _notification_texts(resumed_loop, run_id)
            assert len(texts) == 1 and "completed" in texts[0]
        finally:
            await _teardown(resumed, resumed_loop, registry, post_child)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_post_restart_transition_during_final_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pre_child = BlockingBackend()
        post_child = BlockingBackend()
        resumed, resumed_loop, registry, session_id, _pre = await _restart_harness(
            monkeypatch,
            tmp_path,
            pre_child,
            post_child,
            _noted_streams(),
            _noted_streams("task-2"),
        )
        lock = _ParkingLock()
        registry._run_state_lock = lock  # type: ignore[assignment]
        try:
            _ = [event async for event in resumed.act("Delegate post-restart")]
            summaries = await registry.check_agents()
            agent_id = summaries[0].agent_id
            run_id = summaries[0].current_run_id
            assert run_id is not None
            key = (agent_id, run_id)
            monitor = next(iter(registry._monitor_tasks))
            lock.park_task = monitor
            lock.armed = True
            post_child.release.set()
            await asyncio.wait_for(lock.parked.wait(), 2)
            task = asyncio.create_task(_run_continuation(resumed, registry))
            await asyncio.sleep(0.05)
            # The final quiescence check blocks on the writer's held
            # acquisition — serialization holds post-restart too.
            assert not task.done()
            lock.unpark.set()
            await _wait_for(lambda: key in registry._result_store)
            await registry.get_agent_result(agent_id, run_id)
            outcome, _output = await asyncio.wait_for(task, 10)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert SWEEP_TRIGGER_TERMINAL_TRANSITION in (
                registry._execution._sweep_triggers_seen
            )
            assert registry.termination_declared
        finally:
            lock.armed = False
            lock.unpark.set()
            await _teardown(resumed, resumed_loop, registry, post_child)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_post_restart_completion_during_final_drain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pre_child = BlockingBackend()
        post_child = BlockingBackend()
        parent_release = asyncio.Event()
        post_backend = _GatedParentBackend(
            [
                _task_stream("task-2"),
                [mock_llm_chunk(content="Delegated post-restart")],
                [mock_llm_chunk(content="Noted.")],
            ],
            gate_request=2,
            release=parent_release,
        )
        logging_cfg = SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
        )
        config = _config(logging_cfg)
        holder = {"backend": pre_child}
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.create_backend",
            lambda **_: holder["backend"],
        )
        parent = build_test_agent_loop(
            config=config, backend=FakeBackend(_noted_streams())
        )
        client = start_test_app_server(parent)
        server = cast(AppServer, cast(Any, client._run_peer).__self__)
        session = await attach_test_app_server_session(client)
        registry1 = legacy_backend(server).children
        registry1._execution._sweep_interval = 0.05
        _ = [event async for event in session.act("Delegate this")]
        if hasattr(pre_child, "started"):
            await asyncio.wait_for(pre_child.started.wait(), 2)
        session_id = session.session_id
        await session.close()
        await parent.aclose()
        holder["backend"] = post_child
        resumed_loop = build_test_agent_loop(config=config, backend=post_backend)
        resumed_client = start_test_app_server(resumed_loop)
        resumed_server = cast(AppServer, cast(Any, resumed_client._run_peer).__self__)
        resumed = await attach_test_app_server_session(
            resumed_client, resume_session_id=session_id
        )
        registry = legacy_backend(resumed_server).children
        registry._execution._sweep_interval = 0.05
        try:
            # The child completes while the post-restart act() stream is
            # still draining (gated request); the notification is dropped.
            act_task = asyncio.create_task(_drain_act(resumed, "Delegate post-restart"))
            await _wait_for(lambda: post_backend.gate_arrived.is_set())
            summaries = await registry.check_agents()
            assert len(summaries) == 1
            agent_id = summaries[0].agent_id
            run_id = summaries[0].current_run_id
            assert run_id is not None
            key = (agent_id, run_id)
            registry._suppressed_notifications.add(agent_id)
            post_child.release.set()
            await _wait_for(lambda: key in registry._result_store)
            parent_release.set()
            await asyncio.wait_for(act_task, 10)
            task = asyncio.create_task(_run_continuation(resumed, registry))
            await _wait_for(lambda: key in registry._execution._sweep_surfaced)
            await registry.get_agent_result(agent_id, run_id)
            outcome, _output = await asyncio.wait_for(task, 10)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert key in registry._collected_results
        finally:
            parent_release.set()
            await _teardown(resumed, resumed_loop, registry, post_child)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_post_restart_never_completing_child_budget_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pre_child = BlockingBackend()
        post_child = _GatedBudgetChildBackend()
        resumed, resumed_loop, registry, session_id, _pre = await _restart_harness(
            monkeypatch,
            tmp_path,
            pre_child,
            post_child,
            _noted_streams(),
            _noted_streams("task-2"),
            post_run_max_price=0.00001,
        )
        try:
            _ = [event async for event in resumed.act("Delegate post-restart")]
            summaries = await registry.check_agents()
            agent_id = summaries[0].agent_id
            run_id = summaries[0].current_run_id
            assert run_id is not None
            key = (agent_id, run_id)
            registry._suppressed_notifications.add(agent_id)
            post_child.tool_release.set()
            await _wait_for(lambda: key in registry._result_store)
            task = asyncio.create_task(_run_continuation(resumed, registry))
            await _wait_for(lambda: key in registry._execution._sweep_surfaced)
            await registry.get_agent_result(agent_id, run_id)
            outcome, _output = await asyncio.wait_for(task, 10)
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            record = registry._agent_records[agent_id]
            run = next(
                (r for r in record.run_history if r.run_id == run_id),
                record.current_run,
            )
            assert run is not None and run.stop_reason is RunStopReason.BUDGET_EXCEEDED
            texts = _notification_texts(resumed_loop, run_id)
            assert len(texts) == 1 and "budget-stopped" in texts[0]
        finally:
            await _teardown(resumed, resumed_loop, registry, post_child)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_post_restart_cap_names_cancelled_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pre_child = BlockingBackend()
        post_child = BlockingBackend()  # never released
        resumed, resumed_loop, registry, session_id, _pre = await _restart_harness(
            monkeypatch,
            tmp_path,
            pre_child,
            post_child,
            _noted_streams(),
            _noted_streams("task-2"),
        )
        try:
            _ = [event async for event in resumed.act("Delegate post-restart")]
            summaries = await registry.check_agents()
            agent_id = summaries[0].agent_id
            run_id = summaries[0].current_run_id
            assert run_id is not None
            outcome, _output = await _run_continuation(
                resumed, registry, cap_seconds=0.5
            )
            assert outcome.stop_reason is ContinuationStopReason.WALL_CLOCK_CAP
            assert outcome.cancelled == [(agent_id, run_id, "orchestrator_cancelled")]
            snapshot = await registry.quiescence_snapshot()
            assert snapshot.active_dependency_runs == ()
        finally:
            await _teardown(resumed, resumed_loop, registry, post_child)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_post_restart_compaction_between_completion_and_delivery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pre_child = BlockingBackend()
        post_child = BlockingBackend()
        resumed, resumed_loop, registry, session_id, _pre = await _restart_harness(
            monkeypatch,
            tmp_path,
            pre_child,
            post_child,
            _noted_streams(),
            _noted_streams("task-2"),
        )
        try:
            _ = [event async for event in resumed.act("Delegate post-restart")]
            summaries = await registry.check_agents()
            agent_id = summaries[0].agent_id
            run_id = summaries[0].current_run_id
            assert run_id is not None
            key = (agent_id, run_id)
            registry._suppressed_notifications.add(agent_id)
            post_child.release.set()
            await _wait_for(lambda: key in registry._result_store)
            # A compaction boundary lands between the child's completion and
            # the delivery of its notification (post-restart variant).
            resumed_loop.messages.append(
                LLMMessage(
                    role=Role.assistant,
                    content="compacted summary",
                    context_boundary="compaction",
                )
            )
            assert any(
                m.context_boundary == "compaction" for m in resumed_loop.messages
            )
            task = asyncio.create_task(_run_continuation(resumed, registry))
            await _wait_for(lambda: key in registry._execution._sweep_surfaced)
            await registry.get_agent_result(agent_id, run_id)
            outcome, _output = await asyncio.wait_for(task, 10)
            # The compaction boundary does not strand the result: the sweep
            # still surfaces it and the continuation terminates quiescently.
            assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
            assert key in registry._collected_results
        finally:
            await _teardown(resumed, resumed_loop, registry, post_child)


# ---------------------------------------------------------------------------
# Chunk 2d helpers: delivery-state marking — route-parametrized fetch turns
# (interactive act() vs the programmatic run_continuation lifecycle) and
# transcript envelope inspection (the per-message result_delivery key).
# ---------------------------------------------------------------------------


def _delivery_config(tmp_path: Path) -> Any:
    """Parent config with session logging enabled and the fetch tool on."""
    logging_cfg = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    return _config(logging_cfg, enabled_tools=["task", "get_agent_result"])


def _delivery_fetch_call(
    agent_id: str, call_id: str, *, mode: str = "digest", force: bool = False
) -> ToolCall:
    arguments = {"agent_id": agent_id, "mode": mode, "force": force}
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name="get_agent_result", arguments=json.dumps(arguments)),
    )


def _delivery_streams(
    *fetches: tuple[str, str], tool_call_id: str = "task-1"
) -> list[list[Any]]:
    """Parent streams: the launch turn (task call + follow-up text), then
    one fetch turn per (call_id, mode) — a tool-call stream plus its text.
    """
    streams: list[list[Any]] = [
        _task_stream(tool_call_id),
        [mock_llm_chunk(content="Parent completed")],
    ]
    for call_id, mode in fetches:
        streams.append([
            mock_llm_chunk(
                content="",
                tool_calls=[_delivery_fetch_call("agent-1", call_id, mode=mode)],
            )
        ])
        streams.append([mock_llm_chunk(content="Fetched.")])
    return streams


def _transcript_records(parent: Any) -> list[dict[str, Any]]:
    logger = parent.session_logger
    if not logger.persisted:
        return []
    path = logger.messages_filepath
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _delivery_envelopes(parent: Any) -> list[dict[str, Any]]:
    """The persisted result_delivery envelope keys, in transcript order."""
    return [
        record["result_delivery"]
        for record in _transcript_records(parent)
        if "result_delivery" in record
    ]


async def _delivery_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    child_backend: BlockingBackend,
    streams: list[list[Any]],
) -> tuple[Any, Any, SessionRuntimeRegistry, str, str]:
    session, parent, registry, agent_id, run_id = await _interactive_launch_ex(
        monkeypatch,
        child_backend,
        parent_streams=streams,
        config=_delivery_config(tmp_path),
    )
    return session, parent, registry, agent_id, run_id


def _fetch_tool_result_count(parent: Any) -> int:
    """How many get_agent_result tool results the parent transcript holds."""
    fetch_call_ids = {
        call.id
        for message in parent.messages
        if message.role == Role.assistant
        for call in (message.tool_calls or [])
        if call.function.name == "get_agent_result"
    }
    return sum(
        1
        for message in parent.messages
        if message.role == Role.tool and message.tool_call_id in fetch_call_ids
    )


async def _fetch_turn(
    session: Any,
    registry: SessionRuntimeRegistry,
    parent: Any,
    route: str,
    key: tuple[str, str],
    *,
    first: bool,
) -> None:
    """Drive one fetch turn.

    - programmatic: the harness-owned continuation (run_continuation) runs
      the notification-driven fetch turn across the act() boundary.
    - interactive, first: the completion notification WAKES the idle parent,
      so the fetch turn runs via the wake machinery — no explicit act()
      (that would collide with the already-running turn).
    - interactive, later: explicit act()s once the parent is idle again. The
      wake turn may still be draining when the first act lands, and a queued
      input can then consume a text-only stream, so keep acting until a
      get_agent_result tool result actually lands (streams are finite).
    """
    if route == "programmatic":
        outcome, _output = await _run_continuation(session, registry)
        assert outcome.stop_reason is ContinuationStopReason.QUIESCENT
    elif first:
        await _wait_for(lambda: key in getattr(registry, "_delivery_markings", {}))
    else:
        before = _fetch_tool_result_count(parent)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            await _act_when_idle(session, "Fetch again")
            if _fetch_tool_result_count(parent) > before:
                return
            if not parent.backend._streams:
                break
        raise AssertionError("no get_agent_result tool result landed in the parent")


async def _act_when_idle(session: Any, text: str, timeout: float = 5) -> None:
    """act() once the parent is idle (the wake turn may still be draining)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            _ = [event async for event in session.act(text)]
            return
        except RuntimeError as exc:
            if "already running" not in str(exc) or time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.05)


def _message_texts_including_tool_results(parent: Any) -> list[str]:
    texts = [_message_text(message) for message in parent.messages]
    for message in parent.messages:
        tool_result = getattr(message, "tool_result", None)
        output = getattr(tool_result, "output", None) if tool_result else None
        if output:
            texts.append(json.dumps(output))
    return texts


class TestDeliveryStates:
    """2d: digest/full delivery states in INTERACTIVE and PROGRAMMATIC routes.

    The marking tracks (persisted-conversation-id, agent_id, run_id,
    representation) as a per-message ENVELOPE key committed atomically with
    the transcript write; it is SHARED across the get/wait routes and equal
    across the interactive/programmatic routes. The refusal predicate is
    "delivered AND no compaction boundary between that delivery and this
    fetch"; a post-lift re-delivery writes a fresh marking and re-arms.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("route", ["interactive", "programmatic"])
    async def test_delivery_semantics_route_parametrized(
        self, route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _delivery_launch(
            monkeypatch,
            tmp_path,
            child_backend,
            _delivery_streams(("call-fetch-0", "digest")),
        )
        key = (agent_id, run_id)
        try:
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            await _fetch_turn(session, registry, parent, route, key, first=True)
            # The fetch delivered: in-memory marking + collected marker.
            await _wait_for(lambda: key in getattr(registry, "_delivery_markings", {}))
            assert "digest" in registry._delivery_markings[key]
            assert key in registry._collected_results
            # The marking committed atomically with the transcript write: one
            # per-message envelope key on the fetch's tool record.
            await _wait_for(lambda: len(_delivery_envelopes(parent)) == 1)
            envelope = _delivery_envelopes(parent)[0]
            assert envelope == {
                "conversation_id": session.session_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "representation": "digest",
            }
            tool_records = [
                r
                for r in _transcript_records(parent)
                if r.get("role") == "tool" and "result_delivery" in r
            ]
            assert len(tool_records) == 1
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_cross_route_marking_equality(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        envelopes: dict[str, dict[str, Any]] = {}
        session_ids: dict[str, str] = {}
        for route in ("interactive", "programmatic"):
            child_backend = BlockingBackend()
            session, parent, registry, agent_id, run_id = await _delivery_launch(
                monkeypatch,
                tmp_path / route,
                child_backend,
                _delivery_streams(("call-fetch-0", "digest")),
            )
            key = (agent_id, run_id)
            try:
                child_backend.release.set()
                await _wait_for(
                    lambda key=key, store=registry._result_store: key in store
                )
                await _fetch_turn(session, registry, parent, route, key, first=True)
                await _wait_for(
                    lambda parent=parent: len(_delivery_envelopes(parent)) == 1
                )
                envelope = _delivery_envelopes(parent)[0]
                # The marking's shape and run identity on each route; the
                # conversation id is each route's own persisted session id
                # and the run id each session minted.
                envelopes[route] = {
                    k: v
                    for k, v in envelope.items()
                    if k not in ("conversation_id", "run_id")
                }
                session_ids[route] = envelope["conversation_id"]
                assert envelope["conversation_id"] == session.session_id
                assert envelope["agent_id"] == agent_id
                assert envelope["run_id"] == run_id
                assert envelope["representation"] == "digest"
            finally:
                await _teardown(session, parent, registry, child_backend)
        # CROSS-ROUTE marking equality: identical marking modulo the
        # conversation scope and the per-session run id.
        assert envelopes["interactive"] == envelopes["programmatic"]
        assert session_ids["interactive"] != session_ids["programmatic"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_repeat_fetch_same_mode_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _delivery_launch(
            monkeypatch,
            tmp_path,
            child_backend,
            _delivery_streams(("call-fetch-0", "digest"), ("call-fetch-1", "digest")),
        )
        key = (agent_id, run_id)
        try:
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            # First fetch delivers and marks (interactive wake turn).
            await _fetch_turn(session, registry, parent, "interactive", key, first=True)
            await _wait_for(lambda: len(_delivery_envelopes(parent)) == 1)
            assert len(_delivery_envelopes(parent)) == 1
            # Second fetch, same representation, no compaction boundary: the
            # mechanical refusal fires — no payload, no fresh marking.
            await _fetch_turn(
                session, registry, parent, "interactive", key, first=False
            )
            await asyncio.sleep(0.2)
            assert len(_delivery_envelopes(parent)) == 1
            refused = await registry._run_results.get_result(
                agent_id, run_id, mode="digest"
            )
            assert refused.status == "refused"
            assert "force=True" in refused.message
            # The refusal text reached the parent's context (tool error).
            assert any(
                "force=True" in text
                for text in _message_texts_including_tool_results(parent)
            )
            # A forced fetch re-delivers and writes a fresh marking.
            forced = await registry._run_results.get_result(
                agent_id, run_id, mode="digest", force=True
            )
            assert forced.status == "delivered" and forced.forced is True
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_refusal_lift_and_rearm_after_compaction_boundary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _delivery_launch(
            monkeypatch,
            tmp_path,
            child_backend,
            _delivery_streams(
                ("call-fetch-0", "digest"),
                ("call-fetch-1", "digest"),
                ("call-fetch-2", "digest"),
            ),
        )
        key = (agent_id, run_id)
        try:
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            # Fetch 1: delivered, one envelope (interactive wake turn).
            await _fetch_turn(session, registry, parent, "interactive", key, first=True)
            await _wait_for(lambda: len(_delivery_envelopes(parent)) == 1)
            # A compaction boundary lands between that delivery and the
            # next fetch (live tail of the parent transcript).
            parent.messages.append(
                LLMMessage(
                    role=Role.assistant,
                    content="compacted summary",
                    context_boundary="compaction",
                )
            )
            scan = parent.session_logger.delivery_state_scan(parent.messages)
            assert scan.compaction_boundaries
            boundary_position = max(scan.compaction_boundaries)
            # Fetch 2: the refusal LIFTS — re-delivered post-lift, fresh
            # marking committed with its transcript write.
            await _fetch_turn(
                session, registry, parent, "interactive", key, first=False
            )
            await _wait_for(lambda: len(_delivery_envelopes(parent)) == 2)
            scan = parent.session_logger.delivery_state_scan(parent.messages)
            assert len(scan.markings) == 2
            assert max(m.position for m in scan.markings) > boundary_position
            # Fetch 3: the fresh marking RE-ARMS the refusal.
            await _fetch_turn(
                session, registry, parent, "interactive", key, first=False
            )
            await asyncio.sleep(0.2)
            assert len(_delivery_envelopes(parent)) == 2
            refused = await registry._run_results.get_result(
                agent_id, run_id, mode="digest"
            )
            assert refused.status == "refused"
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("route", ["interactive", "programmatic"])
    async def test_compaction_between_completion_and_delivery(
        self, route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _delivery_launch(
            monkeypatch,
            tmp_path,
            child_backend,
            _delivery_streams(("call-fetch-0", "digest")),
        )
        key = (agent_id, run_id)
        try:
            # The child completes, THEN a compaction boundary lands, THEN
            # the delivery fetch runs (interactive + programmatic routes; the
            # post-restart variant is TestLifecycleRestart's).
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            parent.messages.append(
                LLMMessage(
                    role=Role.assistant,
                    content="compacted summary",
                    context_boundary="compaction",
                )
            )
            await _fetch_turn(session, registry, parent, route, key, first=True)
            # The compaction boundary does not strand the result: the fetch
            # delivers and the marking commits with the transcript write.
            await _wait_for(lambda: len(_delivery_envelopes(parent)) == 1)
            envelope = _delivery_envelopes(parent)[0]
            assert envelope["agent_id"] == agent_id
            assert envelope["run_id"] == run_id
            assert envelope["representation"] == "digest"
            await _wait_for(lambda: key in getattr(registry, "_delivery_markings", {}))
            assert key in registry._collected_results
            if route == "programmatic":
                assert registry.termination_declared
        finally:
            await _teardown(session, parent, registry, child_backend)


# ---------------------------------------------------------------------------
# Chunk 2e helpers: the worker-digest contract assembled over the fixture's
# retained run — the shared rendering the fetch routes' digest mode uses.
# ---------------------------------------------------------------------------


def _run_record(registry: SessionRuntimeRegistry, agent_id: str, run_id: str) -> Any:
    """The run record for (agent_id, run_id) in the fixture registry."""
    record = registry._agent_records[agent_id]
    for candidate in [record.current_run, *record.run_history]:
        if candidate is not None and candidate.run_id == run_id:
            return candidate
    raise AssertionError(f"no run record for {agent_id}/{run_id}")


def _assemble_fixture_digest(
    registry: SessionRuntimeRegistry,
    agent_id: str,
    run_id: str,
    *,
    with_disclosures: bool = True,
) -> WorkerDigest:
    """Assemble the worker digest from the retained run state (P2e).

    The shared ``build_worker_digest`` the routes' digest rendering calls,
    fed the run record's facts and the stored result; the disclosure inputs
    stand in for the worker report's disclosure fields the route extracts.
    """
    stored = registry._result_store[(agent_id, run_id)]
    record = _run_record(registry, agent_id, run_id)
    elapsed: float | None = None
    if record.started_at is not None and record.completed_at is not None:
        elapsed = record.completed_at - record.started_at
    disclosures = (
        WorkerDigestDisclosures(
            touched_paths=["chartreux/core/subagents.py"],
            denial_record=[],
            check_status="pytest: exit 0",
        )
        if with_disclosures
        else None
    )
    return build_worker_digest(
        agent_id=agent_id,
        run_id=run_id,
        result=stored.result,
        disclosures=disclosures,
        run_status=record.status,
        stop_reason=record.stop_reason,
        elapsed_seconds=elapsed,
    )


class TestDigestSufficiency:
    """2e: decision-sufficiency — the reviewer digest alone adjudicates.

    Per route (INTERACTIVE and PROGRAMMATIC), after the default digest-mode
    fetch, the digest assembled from the retained run via the shared
    contract carries run identity, disposition, elapsed/turns, and the
    always-contains disclosure list — enough to adjudicate WITHOUT a full
    fetch: no ``full`` marking is ever written.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("route", ["interactive", "programmatic"])
    async def test_reviewer_digest_suffices_without_full_fetch(
        self, route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _delivery_launch(
            monkeypatch,
            tmp_path,
            child_backend,
            _delivery_streams(("call-fetch-0", "digest")),
        )
        key = (agent_id, run_id)
        try:
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            # The default digest-mode fetch runs on this route.
            await _fetch_turn(session, registry, parent, route, key, first=True)
            await _wait_for(lambda: key in getattr(registry, "_delivery_markings", {}))
            # The delivery was marked digest; NO full fetch ever happened.
            assert set(registry._delivery_markings[key]) == {"digest"}
            await _wait_for(lambda: len(_delivery_envelopes(parent)) == 1)
            assert _delivery_envelopes(parent)[0]["representation"] == "digest"
            # The digest assembled from the retained run suffices to
            # adjudicate: identity, disposition, timing, and the
            # always-contains disclosure list, all without the full body.
            digest = _assemble_fixture_digest(registry, agent_id, run_id)
            assert digest.agent_id == agent_id
            assert digest.run_id == run_id
            assert digest.completed is True
            assert digest.run_status is RunStatus.COMPLETED
            assert digest.turns_used >= 1
            assert digest.elapsed_seconds is not None and digest.elapsed_seconds >= 0
            assert digest.response_available is True
            assert digest.claims.summary
            claims_dump = digest.claims.model_dump(mode="json")
            for field in WORKER_DIGEST_ALWAYS_CONTAINS:
                assert field in claims_dump
            assert claims_dump["touched_paths"] == ["chartreux/core/subagents.py"]
            assert claims_dump["denial_record"] == []
            assert claims_dump["check_status"] == "pytest: exit 0"
            # Worker claims and audited evidence stay distinct provenance.
            assert "audited" not in claims_dump
        finally:
            await _teardown(session, parent, registry, child_backend)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("route", ["interactive", "programmatic"])
    async def test_digest_always_contains_the_disclosure_list_even_when_missing(
        self, route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_backend = BlockingBackend()
        session, parent, registry, agent_id, run_id = await _delivery_launch(
            monkeypatch,
            tmp_path,
            child_backend,
            _delivery_streams(("call-fetch-0", "digest")),
        )
        key = (agent_id, run_id)
        try:
            child_backend.release.set()
            await _wait_for(lambda: key in registry._result_store)
            await _fetch_turn(session, registry, parent, route, key, first=True)
            # With no disclosure inputs established, the assembled digest
            # STILL carries the always-contains list, with missing
            # indicators — never absent fields.
            digest = _assemble_fixture_digest(
                registry, agent_id, run_id, with_disclosures=False
            )
            claims_dump = digest.claims.model_dump(mode="json")
            for field in WORKER_DIGEST_ALWAYS_CONTAINS:
                assert field in claims_dump
                assert claims_dump[f"{field}_missing"] is True
            assert claims_dump["touched_paths"] == []
            assert claims_dump["denial_record"] == []
            assert claims_dump["check_status"] == ""
        finally:
            await _teardown(session, parent, registry, child_backend)
