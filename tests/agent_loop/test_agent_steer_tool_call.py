from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.events import BaseEvent, UserInputRequestEvent, UserMessageEvent
from chartreux.core.llm_models import (
    FunctionCall,
    LLMMessage,
    Role,
    ToolCall,
    use_posting_clock,
)
from chartreux.questions import UserAnswer, UserQuestionResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_STEER_TEXT = "sorry i meant the python runtime inside it"


def _assert_tool_calls_immediately_paired(messages: list[LLMMessage]) -> None:
    # Anthropic requires every assistant `tool_use` to be followed immediately by
    # the matching `tool_result`, with no other message in between.
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == Role.assistant and message.tool_calls:
            pending = {tc.id for tc in message.tool_calls if tc.id is not None}
            cursor = index + 1
            while pending and cursor < len(messages):
                following = messages[cursor]
                assert following.role == Role.tool, (
                    f"message #{cursor} ({following.role}) landed between a "
                    f"tool_use and its tool_result: {following.content!r}"
                )
                if following.tool_call_id is not None:
                    pending.discard(following.tool_call_id)
                cursor += 1
            assert not pending, f"tool_use without a tool_result: {pending}"
        index += 1


def _make_steering_loop(backend: FakeBackend) -> AgentLoop:
    config = build_test_vibe_config(enabled_tools=["ask_user_question"])
    return build_test_agent_loop(config=config, backend=backend)


@pytest.mark.asyncio
async def test_steer_during_tool_call_keeps_backend_payload_tool_paired() -> None:
    tool_call = ToolCall(
        id="call_1",
        index=0,
        function=FunctionCall(
            name="ask_user_question",
            arguments='{"questions": [{"question": "Which runtime?", "options": [{"label": "Python"}, {"label": "Node"}]}]}',
        ),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Let me check your todos.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Done.")],
    ])
    agent_loop = _make_steering_loop(backend)

    injected_events: list[BaseEvent] = []

    async for event in agent_loop.act("please check the archi of vibe_sdk"):
        if isinstance(event, UserInputRequestEvent):
            # The assistant tool call is recorded but its result is still pending.
            with use_posting_clock(lambda: datetime(2026, 1, 2, tzinfo=UTC)):
                injected_events.extend(
                    await agent_loop.inject_user_context(_STEER_TEXT, as_message=True)
                )
            agent_loop.resolve_user_input_request(
                event.request_id,
                UserQuestionResult(
                    answers=[UserAnswer(question="Which runtime?", answer="Python")]
                ),
            )

    # Every request that reached the backend must satisfy tool_use/tool_result
    # adjacency -- this is exactly what a real provider would reject otherwise.
    assert backend.requests_messages, "no backend request was made"
    for request in backend.requests_messages:
        _assert_tool_calls_immediately_paired(request)

    # The steered follow-up must reach the model, after the tool result, on the
    # second call.
    final_request = backend.requests_messages[-1]
    steered = [
        m for m in final_request if m.role == Role.user and m.content == _STEER_TEXT
    ]
    assert len(steered) == 1
    steered_index = final_request.index(steered[0])
    last_tool_index = max(i for i, m in enumerate(final_request) if m.role == Role.tool)
    assert steered_index > last_tool_index

    # The UI still learns about the steered message immediately.
    assert steered[0].posted_at == datetime(2026, 1, 2, tzinfo=UTC)
    assert any(
        isinstance(e, UserMessageEvent)
        and e.content == _STEER_TEXT
        and e.posted_at == steered[0].posted_at
        for e in injected_events
    )
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_injection_commit_callbacks_bracket_append_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _make_steering_loop(FakeBackend())
    phases = []

    def validate() -> None:
        assert not any(m.content == _STEER_TEXT for m in loop.messages)
        phases.append("before")

    def commit(event: UserMessageEvent) -> None:
        assert loop.messages[-1].message_id == event.message_id
        assert loop.messages[-1].content == _STEER_TEXT
        phases.append("commit")

    async def fail_save() -> None:
        assert phases == ["before", "commit"]
        raise RuntimeError("persistence failed after acceptance")

    monkeypatch.setattr(loop, "_save_messages", fail_save)
    with pytest.raises(RuntimeError, match="after acceptance"):
        await loop.inject_user_context(
            _STEER_TEXT, as_message=True, before_commit=validate, on_commit=commit
        )
    assert phases == ["before", "commit"]
    assert sum(m.content == _STEER_TEXT for m in loop.messages) == 1
    await loop.aclose()


@pytest.mark.asyncio
async def test_injection_precommit_rejection_never_appends() -> None:
    loop = _make_steering_loop(FakeBackend())

    def reject() -> None:
        raise ValueError("ineligible")

    with pytest.raises(ValueError, match="ineligible"):
        await loop.inject_user_context(
            _STEER_TEXT,
            as_message=True,
            before_commit=reject,
            on_commit=lambda _: pytest.fail("rejected input committed"),
        )
    assert not any(m.content == _STEER_TEXT for m in loop.messages)
    await loop.aclose()
