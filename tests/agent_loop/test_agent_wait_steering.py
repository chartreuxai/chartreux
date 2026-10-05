from __future__ import annotations

import asyncio
import contextlib
from typing import cast

import pytest

from chartreux.core.agent_loop import AgentLoop, AgentTurnOptions
from chartreux.core.agent_loop._loop import ToolDecision, ToolExecutionResponse
from chartreux.core.events import (
    BaseEvent,
    ToolCallEvent,
    ToolCancellationOrigin,
    ToolResultEvent,
    ToolWaitStateChangedEvent,
    UserInputRequestEvent,
)
from chartreux.core.llm_models import FunctionCall, Role, ToolCall
from chartreux.core.subagents import SubagentRunnerPort, TaskResult
from chartreux.core.tools.base import ToolPermission
from chartreux.core.usage import UsageOutcome, UsagePurpose, UsageRecord
from chartreux.core.utils.tags import is_user_cancellation_event
from chartreux.questions import UserAnswer, UserQuestionResult
from tests.agent_loop.test_agent_steer_tool_call import (
    _assert_tool_calls_immediately_paired,
)
from tests.agent_loop.test_agent_tool_call import make_question_tool_call
from tests.agent_loop.test_usage_accounting import ATTRIBUTION
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class WaitManager:
    def __init__(self) -> None:
        self.results: dict[str, asyncio.Future[TaskResult]] = {}
        self.released: list[str] = []
        self.entered = asyncio.Event()

    async def wait_for_agent(
        self, agent_id: str, run_id: str | None, *, timeout: float | None
    ) -> TaskResult:
        future = self.results.setdefault(
            agent_id, asyncio.get_running_loop().create_future()
        )
        self.entered.set()
        try:
            return await asyncio.shield(future)
        finally:
            self.released.append(agent_id)


def wait_call(call_id: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(
            name="wait_for_agent", arguments=f'{{"agent_id": "{call_id}"}}'
        ),
    )


def make_loop(calls: list[ToolCall]) -> tuple[AgentLoop, FakeBackend]:
    backend = FakeBackend([
        [mock_llm_chunk(content="Waiting", tool_calls=calls)],
        [mock_llm_chunk(content="Continued")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["wait_for_agent", "ask_user_question"]
        ),
        backend=backend,
    )
    return loop, backend


def responses(loop: AgentLoop, call_id: str) -> list:
    return [
        m for m in loop.messages if m.role == Role.tool and m.tool_call_id == call_id
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 3])
async def test_steering_cancels_waits_not_turn_and_pairs_payload(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = [wait_call(f"wait_{i}", i) for i in range(count)]
    loop, backend = make_loop(calls)
    records: list[UsageRecord] = []

    async def sink(record: UsageRecord) -> None:
        records.append(record)

    monkeypatch.setattr(loop, "_accounting_sink", sink, raising=False)
    monkeypatch.setattr(loop, "_usage_attribution", ATTRIBUTION, raising=False)
    original_stream = backend.complete_streaming

    async def stream(**kwargs):
        from chartreux.core.llm.backend.generic import notify_request_started

        notify_request_started()
        async for chunk in original_stream(**kwargs):
            yield chunk

    monkeypatch.setattr(backend, "complete_streaming", stream)
    original_complete = backend.complete

    async def complete(**kwargs):
        from chartreux.core.llm.backend.generic import notify_request_started

        notify_request_started()
        return await original_complete(**kwargs)

    monkeypatch.setattr(backend, "complete", complete)
    preceding_record = None
    manager = WaitManager()
    events = []
    async for event in loop.act(
        "start",
        subagent_runner=cast(SubagentRunnerPort, manager),
        turn_options=AgentTurnOptions(turn_id="turn"),
    ):
        events.append(event)
        if isinstance(event, ToolCallEvent):
            # Before blocking-await registration, cancellation is ineligible.
            assert loop.cancel_outstanding_waits_for_steering("turn") == ()
        if isinstance(event, ToolWaitStateChangedEvent) and loop.is_waiting_only(
            "turn"
        ):
            assert len(records) == 1
            preceding_record = records[0]
            assert preceding_record.outcome is UsageOutcome.COMPLETED
            assert loop.outstanding_wait_call_ids("turn") == tuple(c.id for c in calls)
            assert loop.cancel_outstanding_waits_for_steering("retired") == ()
            await loop.inject_user_context(
                "steering",
                as_message=True,
                before_commit=lambda: None,
                on_commit=lambda _: loop.cancel_outstanding_waits_for_steering("turn"),
            )
            assert not loop.is_waiting_only("turn")
            assert loop.cancel_outstanding_waits_for_steering("turn") == ()
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    states = [e for e in events if isinstance(e, ToolWaitStateChangedEvent)]
    assert [(e.turn_id, e.waiting_only) for e in states] == [
        ("turn", True),
        ("turn", False),
    ]
    assert len(results) == count
    assert all(
        e.cancellation_origin is ToolCancellationOrigin.STEERING for e in results
    )
    assert all(not is_user_cancellation_event(e) for e in results)
    assert len(backend.requests_messages) == 2
    payload = backend.requests_messages[-1]
    _assert_tool_calls_immediately_paired(payload)
    assert [m.tool_call_id for m in payload if m.role == Role.tool] == [
        c.id for c in calls
    ]
    assert next(i for i, m in enumerate(payload) if m.content == "steering") > max(
        i for i, m in enumerate(payload) if m.role == Role.tool
    )
    for call in calls:
        assert len(responses(loop, str(call.id))) == 1
        assert responses(loop, str(call.id))[0].tool_result.cancelled
    assert sorted(manager.released) == sorted(str(c.id) for c in calls)
    assert all(not future.cancelled() for future in manager.results.values())
    assert not loop._tool_invocations
    assert not loop.is_waiting_only("turn")
    assert len(records) == 2
    assert records[0] is preceding_record
    assert all(record.outcome is UsageOutcome.COMPLETED for record in records)
    assert all(record.purpose is UsagePurpose.CONVERSATION for record in records)
    assert all(record.session_id == ATTRIBUTION.session_id for record in records)
    await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "timeout", "failure"])
async def test_completion_first_cleanup_and_call_id_reuse(outcome: str) -> None:
    loop, backend = make_loop([wait_call("same")])
    manager = WaitManager()
    events = []
    backend._streams.extend([
        [mock_llm_chunk(content="Again", tool_calls=[wait_call("same")])],
        [mock_llm_chunk(content="Done")],
    ])
    for turn_id in ("first", "second"):
        manager.results.clear()
        async for event in loop.act(
            "start",
            subagent_runner=cast(SubagentRunnerPort, manager),
            turn_options=AgentTurnOptions(turn_id=turn_id),
        ):
            events.append(event)
            if isinstance(event, ToolWaitStateChangedEvent) and loop.is_waiting_only(
                turn_id
            ):
                invocation = loop._tool_invocations[(turn_id, "same")]
                assert invocation.wait_task is not None
                if outcome == "success":
                    manager.results["same"].set_result(
                        TaskResult(response="done", turns_used=1, completed=True)
                    )
                else:
                    manager.results["same"].set_exception(
                        TimeoutError()
                        if outcome == "timeout"
                        else RuntimeError("failed")
                    )
                await asyncio.gather(invocation.wait_task, return_exceptions=True)
                assert loop.cancel_outstanding_waits_for_steering(turn_id) == ()
        assert not loop._tool_invocations
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert not result.cancelled
    assert len(responses(loop, "same")) == 2
    await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["escape", "shutdown", "stream_close"])
async def test_outer_cancellation_preserves_stop_and_joins_finalization(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, backend = make_loop([wait_call("wait")])
    manager = WaitManager()
    ready = asyncio.Event()
    collector_entered = asyncio.Event()
    finish = asyncio.Event()
    events: list[BaseEvent] = []
    collectors = 0
    original = loop._collect_post_tool_events

    async def collect(*args, **kwargs):
        nonlocal collectors
        collectors += 1
        collector_entered.set()
        await finish.wait()
        text, hook_events = await original(*args, **kwargs)
        return text + " audited-finalization", hook_events

    monkeypatch.setattr(loop, "_collect_post_tool_events", collect)

    async def consume() -> None:
        async with contextlib.aclosing(
            loop.act(
                "start",
                subagent_runner=cast(SubagentRunnerPort, manager),
                turn_options=AgentTurnOptions(turn_id="turn"),
            )
        ) as stream:
            async for event in stream:
                events.append(event)
                if isinstance(
                    event, ToolWaitStateChangedEvent
                ) and loop.is_waiting_only("turn"):
                    ready.set()
                    if source == "stream_close":
                        return

    parent = asyncio.create_task(consume())
    await ready.wait()
    if source != "stream_close":
        parent.cancel()
    await collector_entered.wait()
    invocation = loop._tool_invocations[("turn", "wait")]
    assert not loop.is_waiting_only("turn")
    assert loop.cancel_outstanding_waits_for_steering("turn") == ()
    invocation.task.cancel()  # A second cancellation must join the SAME collector.
    assert not parent.done()
    finish.set()
    await asyncio.gather(parent, return_exceptions=True)
    assert collectors == 1
    assert len(responses(loop, "wait")) == 1
    assert "audited-finalization" in responses(loop, "wait")[0].content
    assert not loop._tool_invocations
    assert len(backend.requests_messages) == 1
    assert all(
        e.cancellation_origin is None for e in events if isinstance(e, ToolResultEvent)
    )
    await loop.aclose()


@pytest.mark.asyncio
async def test_double_steer_then_escape_keeps_emitted_origin_but_stops_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, backend = make_loop([wait_call("wait")])
    manager = WaitManager()
    entered = asyncio.Event()
    finish = asyncio.Event()
    results = []
    original = loop._collect_post_tool_events

    async def collect(*args, **kwargs):
        entered.set()
        await finish.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(loop, "_collect_post_tool_events", collect)

    async def consume() -> None:
        async for event in loop.act(
            "start",
            subagent_runner=cast(SubagentRunnerPort, manager),
            turn_options=AgentTurnOptions(turn_id="turn"),
        ):
            if isinstance(event, ToolWaitStateChangedEvent) and loop.is_waiting_only(
                "turn"
            ):
                assert loop.cancel_outstanding_waits_for_steering("turn") == ("wait",)
                assert loop.cancel_outstanding_waits_for_steering("turn") == ()
            if isinstance(event, ToolResultEvent):
                results.append(event)

    parent = asyncio.create_task(consume())
    await entered.wait()
    parent.cancel()
    finish.set()
    await asyncio.gather(parent, return_exceptions=True)
    assert len(results) == 1
    assert results[0].cancellation_origin is ToolCancellationOrigin.STEERING
    assert len(responses(loop, "wait")) == 1
    assert not loop._tool_invocations
    assert len(backend.requests_messages) == 1
    await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_unrelated_first", [False, True])
async def test_mixed_batch_keeps_unrelated_tool_running(
    finish_unrelated_first: bool,
) -> None:
    loop, backend = make_loop([
        wait_call("wait"),
        make_question_tool_call("question", 1),
    ])
    manager = WaitManager()
    request_id = None
    results = []
    async for event in loop.act(
        "start",
        subagent_runner=cast(SubagentRunnerPort, manager),
        turn_options=AgentTurnOptions(turn_id="turn"),
    ):
        if isinstance(event, UserInputRequestEvent):
            request_id = event.request_id
            assert not loop.is_waiting_only("turn")
            if finish_unrelated_first:
                loop.resolve_user_input_request(
                    request_id,
                    UserQuestionResult(
                        answers=[UserAnswer(question="Which runtime?", answer="Python")]
                    ),
                )
            else:
                await manager.entered.wait()
                assert loop.cancel_outstanding_waits_for_steering("turn") == ("wait",)
        if isinstance(event, ToolWaitStateChangedEvent) and loop.is_waiting_only(
            "turn"
        ):
            assert finish_unrelated_first
            assert loop.cancel_outstanding_waits_for_steering("turn") == ("wait",)
        if isinstance(event, ToolResultEvent):
            results.append(event)
            if event.tool_call_id == "wait" and not finish_unrelated_first:
                assert request_id is not None
                unrelated = loop._tool_invocations[("turn", "question")]
                assert not unrelated.task.done() and not unrelated.task.cancelling()
                loop.resolve_user_input_request(
                    request_id,
                    UserQuestionResult(
                        answers=[UserAnswer(question="Which runtime?", answer="Python")]
                    ),
                )
    assert len(results) == 2
    assert next(e for e in results if e.tool_call_id == "wait").cancelled
    assert not next(e for e in results if e.tool_call_id == "question").cancelled
    assert not loop._tool_invocations
    assert len(backend.requests_messages) == 2
    _assert_tool_calls_immediately_paired(backend.requests_messages[-1])
    await loop.aclose()


@pytest.mark.asyncio
async def test_steering_at_invocation_startup_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, backend = make_loop([wait_call("wait")])
    manager = WaitManager()
    entered = asyncio.Event()
    proceed = asyncio.Event()
    original = loop._should_execute_tool

    async def approve(*args):
        entered.set()
        await proceed.wait()
        return await original(*args)

    monkeypatch.setattr(loop, "_should_execute_tool", approve)

    async def consume():
        events = []
        async for event in loop.act(
            "start",
            subagent_runner=cast(SubagentRunnerPort, manager),
            turn_options=AgentTurnOptions(turn_id="turn"),
        ):
            events.append(event)
            if isinstance(event, ToolWaitStateChangedEvent) and loop.is_waiting_only(
                "turn"
            ):
                assert loop.cancel_outstanding_waits_for_steering("turn") == ("wait",)
        return events

    parent = asyncio.create_task(consume())
    await entered.wait()
    assert loop.outstanding_wait_call_ids("turn") == ("wait",)
    assert not loop.is_waiting_only("turn")
    assert loop.cancel_outstanding_waits_for_steering("turn") == ()
    proceed.set()
    events = await parent
    assert len([e for e in events if isinstance(e, ToolResultEvent)]) == 1
    assert len(responses(loop, "wait")) == 1
    _assert_tool_calls_immediately_paired(backend.requests_messages[-1])
    assert not loop._tool_invocations
    await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["escape", "shutdown"])
async def test_outer_cancel_before_result_capture_takes_precedence(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, backend = make_loop([wait_call("wait")])
    manager = WaitManager()
    releasing = asyncio.Event()
    finish = asyncio.Event()
    original = manager.wait_for_agent

    async def wait(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        finally:
            releasing.set()
            await finish.wait()

    monkeypatch.setattr(manager, "wait_for_agent", wait)
    captured = []

    # Inspect the queued result even when parent interruption closes the stream.
    original_sanitize = loop._is_steering_cancellation

    def origin(call_id):
        value = original_sanitize(call_id)
        captured.append(value)
        return value

    monkeypatch.setattr(loop, "_is_steering_cancellation", origin)

    async def consume():
        async for event in loop.act(
            "start",
            subagent_runner=cast(SubagentRunnerPort, manager),
            turn_options=AgentTurnOptions(turn_id="turn"),
        ):
            if isinstance(event, ToolWaitStateChangedEvent) and loop.is_waiting_only(
                "turn"
            ):
                assert loop.cancel_outstanding_waits_for_steering("turn") == ("wait",)

    parent = asyncio.create_task(consume())
    await releasing.wait()
    parent.cancel()  # Escape and shutdown use the same core task-cancel boundary.
    finish.set()
    await asyncio.gather(parent, return_exceptions=True)
    assert captured == [False]
    assert len(responses(loop, "wait")) == 1
    assert not loop._tool_invocations
    assert len(backend.requests_messages) == 1
    await loop.aclose()


@pytest.mark.asyncio
async def test_nonsteering_cancelled_tool_still_stops_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, backend = make_loop([wait_call("permission")])

    async def skip(*args):
        return ToolDecision(
            verdict=ToolExecutionResponse.SKIP,
            approval_type=ToolPermission.NEVER,
            feedback="<user_cancellation>Permission cancelled</user_cancellation>",
        )

    monkeypatch.setattr(loop, "_should_execute_tool", skip)
    results = []
    async for event in loop.act("start"):
        if isinstance(event, ToolResultEvent):
            results.append(event)
    assert len(results) == 1
    assert results[0].cancelled and results[0].cancellation_origin is None
    assert is_user_cancellation_event(results[0])
    assert len(backend.requests_messages) == 1
    assert not loop._tool_invocations
    await loop.aclose()
