"""Agent-side execution port: ordering, streaming, and callback contract.

The in-process adapter must run the same turn the loop runs directly, refuse a
second turn while one is in flight, release its ordering guard when the stream
ends or is closed, and settle user-input requests through the loop's broker.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
import json

import pytest

from chartreux.core.agent_loop import AgentLoop, AgentTurnOptions
from chartreux.core.agent_loop._execution_port import (
    ExecutionTurnRequest,
    TurnOrderingConflict,
)
from chartreux.core.events import AssistantEvent, BaseEvent, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools.base import ToolError
from chartreux.questions import UserAnswer, UserQuestionResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


async def _consume(stream: AsyncIterator[BaseEvent]) -> list[BaseEvent]:
    return [event async for event in stream]


def _assistant_contents(events: list[BaseEvent]) -> list[str]:
    return [event.content for event in events if isinstance(event, AssistantEvent)]


def _question_call() -> ToolCall:
    return ToolCall(
        id="question-1",
        index=0,
        function=FunctionCall(
            name="ask_user_question",
            arguments=json.dumps({
                "questions": [
                    {
                        "question": "Ship it?",
                        "options": [{"label": "Yes"}, {"label": "No"}],
                    }
                ]
            }),
        ),
    )


@pytest.mark.asyncio
async def test_port_replays_the_loop_turn_stream() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="direct")],
        [mock_llm_chunk(content="ported")],
    ])
    loop = build_test_agent_loop(backend=backend)
    try:
        direct = await _consume(loop.act("first"))
        ported = await _consume(
            loop.execution_port.run_turn(ExecutionTurnRequest(message="second"))
        )
    finally:
        await loop.aclose()

    assert [type(event) for event in ported] == [type(event) for event in direct]
    assert _assistant_contents(direct) == ["direct"]
    assert _assistant_contents(ported) == ["ported"]


@pytest.mark.asyncio
async def test_port_refuses_a_second_turn_while_one_is_in_flight() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def gated_act(*_args, **_kwargs):
        started.set()
        await release.wait()
        yield AssistantEvent(content="done", message_id="assistant-1")

    loop = build_test_agent_loop()
    loop.act = gated_act
    port = loop.execution_port
    try:
        consumer = asyncio.create_task(
            _consume(
                port.run_turn(
                    ExecutionTurnRequest(
                        message="first", turn_options=AgentTurnOptions(turn_id="turn-1")
                    )
                )
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        with pytest.raises(TurnOrderingConflict) as conflict:
            await _consume(port.run_turn(ExecutionTurnRequest(message="second")))
        assert conflict.value.active_turn_id == "turn-1"

        release.set()
        events = await asyncio.wait_for(consumer, timeout=1)
        assert _assistant_contents(events) == ["done"]

        finishing = await _consume(
            port.run_turn(
                ExecutionTurnRequest(
                    message="third", turn_options=AgentTurnOptions(turn_id="turn-2")
                )
            )
        )
        assert _assistant_contents(finishing) == ["done"]
    finally:
        release.set()
        await loop.aclose()


@pytest.mark.asyncio
async def test_port_releases_ordering_when_the_consumer_closes_the_stream() -> None:
    started = asyncio.Event()
    hold = asyncio.Event()

    async def hanging_act(*_args, **_kwargs):
        started.set()
        await hold.wait()
        yield AssistantEvent(content="late", message_id="assistant-1")

    loop = build_test_agent_loop()
    loop.act = hanging_act
    port = loop.execution_port
    try:
        stream = port.run_turn(ExecutionTurnRequest(message="first"))
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.wait_for(started.wait(), timeout=1)
        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        await stream.aclose()

        async def quick_act(*_args, **_kwargs):
            yield AssistantEvent(content="after", message_id="assistant-2")

        loop.act = quick_act
        events = await _consume(port.run_turn(ExecutionTurnRequest(message="second")))
        assert _assistant_contents(events) == ["after"]
    finally:
        hold.set()
        await loop.aclose()


async def _answer_question(
    loop: AgentLoop, answer: Callable[[AgentLoop, str], None]
) -> tuple[list[BaseEvent], int]:
    """Run one question turn, settling each request through the port."""
    events: list[BaseEvent] = []
    requests = 0
    async for event in loop.execution_port.run_turn(
        ExecutionTurnRequest(message="ask")
    ):
        events.append(event)
        if isinstance(event, UserInputRequestEvent):
            requests += 1
            answer(loop, event.request_id)
    return events, requests


@pytest.mark.asyncio
async def test_port_resolves_a_user_input_request_mid_turn() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_question_call()])],
        [mock_llm_chunk(content="done")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["ask_user_question"]),
        backend=backend,
    )

    def resolve(loop: AgentLoop, request_id: str) -> None:
        loop.execution_port.resolve_user_input(
            request_id,
            UserQuestionResult(answers=[UserAnswer(question="Ship it?", answer="Yes")]),
        )

    try:
        events, requests = await _answer_question(loop, resolve)
    finally:
        await loop.aclose()

    assert requests == 1
    assert _assistant_contents(events)[-1] == "done"


@pytest.mark.asyncio
async def test_port_rejects_a_user_input_request_mid_turn() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_question_call()])],
        [mock_llm_chunk(content="done")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["ask_user_question"]),
        backend=backend,
    )

    def reject(loop: AgentLoop, request_id: str) -> None:
        loop.execution_port.reject_user_input(request_id, ToolError("no user"))

    try:
        events, requests = await _answer_question(loop, reject)
    finally:
        await loop.aclose()

    assert requests == 1
    assert _assistant_contents(events)[-1] == "done"
