from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
import threading
from typing import Any

import pytest

from chartreux.app_server._turns import CallbackRejectedError, TurnController
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import (
    CallbackRequested,
    HistoryEntryAdded,
    SessionSnapshot,
    SessionUpdated,
    StatsUpdated,
    TurnRetrying,
)
from chartreux.app_server.models import (
    AnsweredCallbackState,
    CancelledCallbackState,
    OpenCallbackState,
    PublicRetryCategory,
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import CallbackResultError, HistoryEntryUpdatedParams
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.events import AssistantEvent, UserMessageEvent
from chartreux.core.utils import RetryReason
from tests.app_server._session_runtime_support import (
    _isolate_recovery_registry,  # noqa: F401 - autouse fixture registered via module namespace
    _question_tool_call,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["prepared_cancel", "scheduled_cancel", "rollback"])
async def test_unstarted_turn_cancel_settles_but_rollback_does_not(mode: str) -> None:
    from unittest.mock import AsyncMock

    from chartreux.app_server._sessions import SessionRuntimeRegistry
    from chartreux.app_server.models import PublicTurnStatus
    from chartreux.app_server.protocol import TurnStartParams

    loop = build_test_agent_loop()
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    runtime = registry._build_child_runtime(loop)
    try:
        response, action = runtime.turns.start(
            TurnStartParams(session_id=loop.session_id, message=[])
        )
        if mode == "rollback":
            action.abort()
            assert runtime.turns.completed_turns == []
            with pytest.raises(RuntimeError, match="Turn did not complete"):
                await runtime.turns.wait_for_operation(response.turn.id)
        else:
            if mode == "scheduled_cancel":
                action()
            waiter = asyncio.create_task(
                runtime.turns.wait_for_operation(response.turn.id)
            )
            assert runtime.turns.interrupt_operation(response.turn.id)
            action()  # A cancelled prepared action cannot resurrect execution.
            terminal = await asyncio.wait_for(waiter, 2)
            assert terminal.status is PublicTurnStatus.INTERRUPTED
            assert runtime.turns.operation_pending_turn_id(response.turn.id) is None
        assert runtime.execution.active is None
        assert runtime.turns.active_turn is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_background_title_keeps_runtime_non_quiescent_until_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_started = asyncio.Event()
    finish_generation = asyncio.Event()

    async def generate_title(*_args, **_kwargs) -> str:
        generation_started.set()
        await finish_generation.wait()
        return "Generated title"

    monkeypatch.setattr(
        "chartreux.core.session.title_model.generate_session_title", generate_title
    )
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            save_dir=str(tmp_path / "sessions"), session_prefix="session", enabled=True
        )
    )
    agent_loop = build_test_agent_loop(
        config=config,
        backend=FakeBackend([[mock_llm_chunk(content="done")]]),
        auto_title_enabled=True,
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )
    turn_events = session.act("hello", auto_title="Deterministic")
    background_events = session.events()

    try:
        during_turn = await asyncio.wait_for(_consume(turn_events), timeout=1)
        await asyncio.wait_for(generation_started.wait(), timeout=1)

        pending = [
            event
            for event in during_turn
            if isinstance(event, SessionSnapshot) and event.state.is_quiescent is False
        ]
        assert len(pending) == 1
        assert session.state.is_quiescent is False

        finish_generation.set()
        delivered: list[object] = []
        while True:
            event = await asyncio.wait_for(background_events.__anext__(), timeout=1)
            delivered.append(event)
            if isinstance(event, SessionSnapshot) and event.state.is_quiescent is True:
                break

        title_update = next(
            event for event in delivered if isinstance(event, SessionUpdated)
        )
        assert title_update.session.title == "Generated title"
        assert isinstance(delivered[-1], SessionSnapshot)
        assert session.state.is_quiescent is True
    finally:
        finish_generation.set()
        await background_events.aclose()
        await turn_events.aclose()
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_retry_state_survives_unrelated_updates_until_generation_resumes() -> (
    None
):
    retry_started = asyncio.Event()
    resume_generation = asyncio.Event()
    finish_turn = asyncio.Event()
    agent_loop = build_test_agent_loop()

    async def retrying_act(message: str, *, turn_options, **_kwargs):
        assert turn_options.retry_sink is not None
        await turn_options.retry_sink(RetryReason.from_http_status(429))
        retry_started.set()
        await resume_generation.wait()
        yield UserMessageEvent(content=message, message_id="user-1")
        await finish_turn.wait()
        yield AssistantEvent(content="done", message_id="assistant-1")

    agent_loop.act = retrying_act
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )
    events = session.act("hello")
    retry_snapshot_task = asyncio.create_task(_next_event(events, SessionSnapshot))
    turns = legacy_backend(server).session.turns

    try:
        await asyncio.wait_for(retry_started.wait(), timeout=1)
        retry_snapshot = await asyncio.wait_for(retry_snapshot_task, timeout=1)
        legacy_retry = await _next_event(events, TurnRetrying)
        retrying = turns.retrying
        active_turn = turns.active_turn
        assert retrying is not None
        assert active_turn is not None
        assert retrying.turn_id == active_turn.id
        assert retrying.category is PublicRetryCategory.RATE_LIMITED
        assert retrying.detail == "HTTP 429"
        assert retry_snapshot.state.retrying == retrying
        assert session.state.retrying == retrying
        assert legacy_retry.params.category is PublicRetryCategory.RATE_LIMITED
        assert legacy_retry.params.detail == "HTTP 429"

        await turns._emit_stats()
        await turns._emit_status("running")
        stats = await _next_event(events, StatsUpdated)
        assert stats.params.event_id == retry_snapshot.state.event_id + 1
        assert turns.retrying == retrying
        assert session.state.retrying == retrying

        resume_generation.set()
        resumed = await _next_event(events, SessionSnapshot)
        assert resumed.state.retrying is None
        assert resumed.state.event_id > stats.params.event_id
        assert turns.retrying is None
        await _next_event(events, HistoryEntryAdded)

        finish_turn.set()
        await _consume(events)
    finally:
        resume_generation.set()
        finish_turn.set()
        if not retry_snapshot_task.done():
            retry_snapshot_task.cancel()
            with suppress(asyncio.CancelledError):
                await retry_snapshot_task
        await events.aclose()
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_interrupt_clears_retry_state() -> None:
    retry_started = asyncio.Event()
    agent_loop = build_test_agent_loop()

    async def retrying_act(*_args, turn_options, **_kwargs):
        retry_sink = turn_options.retry_sink
        assert retry_sink is not None
        loop = asyncio.get_running_loop()

        def emit_retry() -> None:
            async def report_retry() -> None:
                await retry_sink(RetryReason.from_http_status(503))

            future = asyncio.run_coroutine_threadsafe(report_retry(), loop)
            future.result(timeout=1)

        await asyncio.to_thread(emit_retry)
        retry_started.set()
        await asyncio.Event().wait()
        yield AssistantEvent(content="unreachable", message_id="assistant-1")

    agent_loop.act = retrying_act
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )
    events = session.act("hello")
    retry_snapshot_task = asyncio.create_task(_next_event(events, SessionSnapshot))
    turns = legacy_backend(server).session.turns

    try:
        await asyncio.wait_for(retry_started.wait(), timeout=1)
        retry_snapshot = await asyncio.wait_for(retry_snapshot_task, timeout=1)
        await _next_event(events, TurnRetrying)
        assert turns.retrying is not None
        assert retry_snapshot.state.retrying == turns.retrying

        await session.interrupt()
        cleared = await _next_event(events, SessionSnapshot)
        assert cleared.state.retrying is None
        assert turns.retrying is None
        await _consume(events)
    finally:
        if not retry_snapshot_task.done():
            retry_snapshot_task.cancel()
            with suppress(asyncio.CancelledError):
                await retry_snapshot_task
        await events.aclose()
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_retry_from_interrupted_turn_is_not_applied_to_next_turn() -> None:
    first_turn_started = asyncio.Event()
    second_turn_started = asyncio.Event()
    release_late_retry = threading.Event()
    late_retry_finished = asyncio.Event()
    finish_second_turn = asyncio.Event()

    class DelayedRetryBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0
            self.late_retry_task: asyncio.Task[None] | None = None

        async def complete(self, **_kwargs):
            self.call_count += 1
            if self.call_count == 1:
                loop = asyncio.get_running_loop()

                def emit_late_retry() -> None:
                    release_late_retry.wait()

                    async def report_retry() -> None:
                        assert self.on_retry is not None
                        await self.on_retry(RetryReason.from_http_status(503))

                    future = asyncio.run_coroutine_threadsafe(report_retry(), loop)
                    future.result(timeout=1)
                    loop.call_soon_threadsafe(late_retry_finished.set)

                self.late_retry_task = asyncio.create_task(
                    asyncio.to_thread(emit_late_retry)
                )
                first_turn_started.set()
                await asyncio.Event().wait()
                return mock_llm_chunk(content="unreachable")

            second_turn_started.set()
            await finish_second_turn.wait()
            return mock_llm_chunk(content="done")

    backend = DelayedRetryBackend()
    agent_loop = build_test_agent_loop(backend=backend)
    backend.on_retry = agent_loop.notice_retry
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )
    turns = legacy_backend(server).session.turns
    first_events = session.act("first")
    first_consumer = asyncio.create_task(_consume(first_events))
    second_events: AsyncIterator[object] | None = None
    second_consumer: asyncio.Task[list[object]] | None = None

    try:
        await asyncio.wait_for(first_turn_started.wait(), timeout=1)
        await session.interrupt()
        await asyncio.wait_for(first_consumer, timeout=1)
        assert turns.retrying is None

        second_events = session.act("second")
        second_consumer = asyncio.create_task(_consume(second_events))
        await asyncio.wait_for(second_turn_started.wait(), timeout=1)

        release_late_retry.set()
        await asyncio.wait_for(late_retry_finished.wait(), timeout=1)

        assert turns.retrying is None
        assert session.state.retrying is None

        finish_second_turn.set()
        await asyncio.wait_for(second_consumer, timeout=1)
    finally:
        release_late_retry.set()
        finish_second_turn.set()
        if not first_consumer.done():
            first_consumer.cancel()
            with suppress(asyncio.CancelledError):
                await first_consumer
        if second_consumer is not None and not second_consumer.done():
            second_consumer.cancel()
            with suppress(asyncio.CancelledError):
                await second_consumer
        if backend.late_retry_task is not None:
            if not backend.late_retry_task.done():
                backend.late_retry_task.cancel()
            with suppress(asyncio.CancelledError):
                await backend.late_retry_task
        await first_events.aclose()
        if second_events is not None:
            await second_events.aclose()
        await session.close()
        await agent_loop.aclose()


async def _consume(events: AsyncIterator[object]) -> list[object]:
    return [event async for event in events]


async def _next_event[EventT](
    events: AsyncIterator[object], event_type: type[EventT]
) -> EventT:
    async for event in events:
        if isinstance(event, event_type):
            return event
    raise AssertionError(f"Event stream ended before {event_type.__name__}")


def _question_answer() -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(
            answers=[UserAnswer(question="Ship it?", answer="Yes")]
        )
    )


async def _open_question_session() -> tuple[Any, TurnController, Any]:
    """Attach a session whose first turn opens one ``ask_user_question``."""
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_question_tool_call()])],
        [mock_llm_chunk(content="Done")],
    ])
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["ask_user_question"]),
        backend=backend,
        enable_streaming=True,
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )
    return session, legacy_backend(server).session.turns, agent_loop


def _record_callback_patches(
    turns: TurnController, callback_id: str, record: Any
) -> tuple[list[tuple[bool, object]], list[str]]:
    """Record each callback patch as it is emitted, before delivery.

    Returns ``(observed, future_states)``: for every patch, whether the
    record already carried a terminal disposition (a resolution or the
    core-side rejection) and the patched entry's state, plus the raw asyncio
    future state at that moment.
    """
    observed: list[tuple[bool, object]] = []
    future_states: list[str] = []
    original = turns._emit_projected

    async def recording(update: Any) -> None:
        params = update.params
        if (
            isinstance(params, HistoryEntryUpdatedParams)
            and params.entry_id == f"callback:{callback_id}"
        ):
            entry = next(
                item for item in turns.callbacks if item.callback_id == callback_id
            )
            terminal = record.core_resolved or record.resolution is not None
            observed.append((terminal, entry.state))
            future = record.future
            future_states.append(
                "cancelled"
                if future.cancelled()
                else "done"
                if future.done()
                else "pending"
            )
        await original(update)

    turns._emit_projected = recording  # type: ignore[method-assign]
    return observed, future_states


async def _drain(events: AsyncIterator[object]) -> None:
    with suppress(Exception):
        await asyncio.wait_for(_consume(events), timeout=2)


@pytest.mark.asyncio
async def test_cancel_callbacks_terminalizes_before_the_waiter_resolves() -> None:
    session, turns, agent_loop = await _open_question_session()
    events = session.act("read todos")
    try:
        requested = await _next_event(events, CallbackRequested)
        callback_id = requested.callback.callback_id
        record = turns._callbacks[callback_id]
        observed, future_states = _record_callback_patches(turns, callback_id, record)

        await turns._cancel_callbacks("App server closed")

        assert len(observed) == 1
        terminal_before_publish, state = observed[0]
        assert terminal_before_publish is False
        assert future_states == ["pending"]
        assert isinstance(state, CancelledCallbackState)
        assert state.reason == "App server closed"
        assert isinstance(record.future.exception(), CallbackRejectedError)

        # A repeat terminalization publishes nothing and re-rejects nothing.
        await turns._cancel_callbacks("App server closed")
        assert len(observed) == 1
        assert isinstance(record.future.exception(), CallbackRejectedError)
    finally:
        await _drain(events)
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_reset_terminalizes_open_callbacks_before_dropping_state() -> None:
    session, turns, agent_loop = await _open_question_session()
    events = session.act("read todos")
    try:
        requested = await _next_event(events, CallbackRequested)
        callback_id = requested.callback.callback_id
        record = turns._callbacks[callback_id]
        observed, _ = _record_callback_patches(turns, callback_id, record)

        await turns.reset()

        # Reset funnels through close(), which cancels the turn task before
        # dropping the projector: the terminal patch is published while the
        # callback record still has no terminal disposition.
        assert observed
        terminal_before_publish, state = observed[0]
        assert terminal_before_publish is False
        assert isinstance(state, CancelledCallbackState)
        assert turns.callbacks == []
        assert turns._callbacks == {}
        # The waiter itself was already cancelled with the turn task; the
        # record carries the core-side rejection.
        assert record.core_resolved is True
    finally:
        await _drain(events)
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_close_never_overwrites_an_answered_callback() -> None:
    session, turns, agent_loop = await _open_question_session()
    events = session.act("read todos")
    try:
        requested = await _next_event(events, CallbackRequested)
        callback_id = requested.callback.callback_id
        record = turns._callbacks[callback_id]
        answer = _question_answer()

        await session.respond_to_callback(callback_id, answer)
        answered = next(
            entry for entry in turns.callbacks if entry.callback_id == callback_id
        )
        assert isinstance(answered.state, AnsweredCallbackState)
        observed, _ = _record_callback_patches(turns, callback_id, record)

        await turns._cancel_callbacks("App server closed")

        # The terminal cancelled patch belongs to open entries only: the
        # answered record keeps exactly one terminal result.
        assert observed == []
        assert record.future.result() == answer
    finally:
        await _drain(events)
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_answer_records_the_settlement_before_publishing_the_terminal_patch() -> (
    None
):
    """Server-authoritative settlement under the resolution lock (plan U).

    The record carries the terminal disposition while the answered patch goes
    out, and the core waiter resolves only afterwards. A concurrent expiry,
    cancellation, or reconnect therefore observes a settled callback, never a
    ghost open one.
    """
    session, turns, agent_loop = await _open_question_session()
    events = session.act("read todos")
    try:
        requested = await _next_event(events, CallbackRequested)
        callback_id = requested.callback.callback_id
        record = turns._callbacks[callback_id]
        observed, future_states = _record_callback_patches(turns, callback_id, record)
        answer = _question_answer()

        assert await turns.answer_callback(callback_id, answer) == "accepted"

        assert len(observed) == 1
        terminal_before_publish, state = observed[0]
        assert terminal_before_publish is True
        assert isinstance(state, AnsweredCallbackState)
        assert future_states == ["pending"]
        assert record.future.result() == answer
        assert record.core_resolved is True
    finally:
        await _drain(events)
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_reject_callback_publishes_the_terminal_entry_before_settling() -> None:
    """A rejected delivery is a server settlement with a visible terminal state."""
    session, turns, agent_loop = await _open_question_session()
    events = session.act("read todos")
    try:
        requested = await _next_event(events, CallbackRequested)
        callback_id = requested.callback.callback_id
        record = turns._callbacks[callback_id]
        observed, future_states = _record_callback_patches(turns, callback_id, record)

        status = await turns.reject_callback(
            callback_id, CallbackResultError(message="client refused")
        )

        assert status == "accepted"
        assert len(observed) == 1
        terminal_before_publish, state = observed[0]
        assert terminal_before_publish is True
        assert isinstance(state, CancelledCallbackState)
        assert state.reason == "client refused"
        assert future_states == ["pending"]
        assert isinstance(record.future.exception(), CallbackRejectedError)
        assert record.core_resolved is True
        entry = next(
            item for item in turns.callbacks if item.callback_id == callback_id
        )
        assert isinstance(entry.state, CancelledCallbackState)
    finally:
        await _drain(events)
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_answer_and_cancellation_race_publishes_one_terminal_state() -> None:
    """Answer/expiry/cancel arbitration has exactly one winner and no ghost."""
    session, turns, agent_loop = await _open_question_session()
    events = session.act("read todos")
    try:
        requested = await _next_event(events, CallbackRequested)
        callback_id = requested.callback.callback_id
        record = turns._callbacks[callback_id]
        observed, _ = _record_callback_patches(turns, callback_id, record)

        await asyncio.gather(
            turns.answer_callback(callback_id, _question_answer()),
            turns._cancel_callbacks("Turn ended"),
            return_exceptions=True,
        )

        entry = next(
            item for item in turns.callbacks if item.callback_id == callback_id
        )
        assert not isinstance(entry.state, OpenCallbackState)
        assert len(observed) == 1
        assert record.core_resolved is True
    finally:
        await _drain(events)
        await session.close()
        await agent_loop.aclose()
