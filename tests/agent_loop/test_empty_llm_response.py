from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from unittest.mock import Mock

import pytest

from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    IncompleteLLMResponseError,
    LLMGateway,
    TranscriptAppend,
    _map_call_error,
)
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.errors import RefusalError, ResponseTooLongError
from chartreux.core.llm.backend.anthropic import AnthropicAdapter
from chartreux.core.llm.backend.generic import OpenAIAdapter
from chartreux.core.llm.backend.openai_responses import OpenAIResponsesAdapter
from chartreux.core.llm.failures import RequestRetryBudget
from chartreux.core.llm_models import (
    FunctionCall,
    LLMChunk,
    LLMMessage,
    Role,
    StopInfo,
    ToolCall,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.utils import retry
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class RecordingBackend(FakeBackend):
    def __init__(self, streams):
        super().__init__(streams)
        self.closed = 0

    async def complete_streaming(self, **kwargs) -> AsyncGenerator[LLMChunk]:
        try:
            async for chunk in super().complete_streaming(**kwargs):
                yield chunk
        finally:
            self.closed += 1


def setup_call(backend):
    model = ModelConfig(
        name="test", provider="test", alias="test", input_price=1, output_price=2
    )
    inputs = CompletionInputs(
        model=model,
        provider_name="test",
        emits_finish_reason=True,
        messages=(LLMMessage(role=Role.user, content="hello"),),
        tools=None,
        tool_choice=None,
        extra_headers={"test": "same"},
        metadata={"message_id": "same"},
        max_tokens=42,
    )
    resources = CallResources(
        backend=backend, stats=AgentStats(), process_message=lambda m: m
    )
    return inputs, resources


async def run_call(streaming, inputs, resources, outcomes):
    gateway = LLMGateway()
    if streaming:
        return [
            chunk
            async for chunk in gateway.chat_streaming(
                inputs, resources, transcript=outcomes.append
            )
        ]
    return [await gateway.chat(inputs, resources, transcript=outcomes.append)]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("kind", ["empty", "whitespace", "reasoning", "payload"])
@pytest.mark.parametrize("has_usage", [False, True])
async def test_empty_exhaustion(streaming, kind, has_usage, monkeypatch):
    message = LLMMessage(
        role=Role.assistant,
        content=" \n\t" if kind == "whitespace" else "",
        reasoning_content="thinking" if kind == "reasoning" else None,
        reasoning_payloads=[{"type": "opaque"}] if kind == "payload" else None,
    )
    chunk = mock_llm_chunk(content="").model_copy(update={"message": message})
    if not has_usage:
        chunk = chunk.model_copy(update={"usage": None})
    backend = RecordingBackend([[chunk], [chunk]])
    inputs, resources = setup_call(backend)
    outcomes: list[TranscriptAppend] = []
    success = Mock()
    monkeypatch.setattr(
        "chartreux.core.agent_loop.llm_gateway.log_model_call_success", success
    )
    with pytest.raises(EmptyLLMResponseError):
        await run_call(streaming, inputs, resources, outcomes)
    published_reasoning = streaming and kind == "reasoning"
    calls = 1 if published_reasoning else 2
    assert len(backend.requests_messages) == calls
    assert len(outcomes) == (1 if published_reasoning else 0)
    if outcomes:
        assert outcomes[0].kind == "interrupted"
        assert outcomes[0].message.reasoning_content == "thinking"
    assert resources.stats.session_prompt_tokens == (10 * calls if has_usage else 0)
    assert resources.stats.has_unknown_cost is (not has_usage)
    assert not success.called
    if streaming:
        assert backend.closed == calls


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_then_valid_replays_identical_inputs(streaming, monkeypatch):
    backend = RecordingBackend([
        [mock_llm_chunk(content=" \n")],
        [mock_llm_chunk(content="answer")],
    ])
    inputs, resources = setup_call(backend)
    notices = []

    async def observe(reason):
        if streaming:
            assert backend.closed == 1
        notices.append(reason)

    resources = replace(resources, on_retry=observe)
    outcomes = []
    success = Mock()
    monkeypatch.setattr(
        "chartreux.core.agent_loop.llm_gateway.log_model_call_success", success
    )
    await run_call(streaming, inputs, resources, outcomes)
    assert backend.requests_messages[0] == backend.requests_messages[1]
    assert backend.requests_metadata == [inputs.metadata, inputs.metadata]
    assert backend.requests_extra_headers == [
        inputs.extra_headers,
        inputs.extra_headers,
    ]
    assert backend.requests_max_tokens == [42, 42]
    assert backend.requests_tools == [None, None]
    assert backend.requests_tool_choices == [None, None]
    assert len(outcomes) == 1
    assert outcomes[0].kind == "complete"
    assert outcomes[0].message.content == "answer"
    assert success.call_count == 1
    assert notices[0].detail == "Empty assistant response; retrying once"
    assert notices[0].category == retry.RetryCategory.UNKNOWN
    assert resources.stats.session_prompt_tokens == 20
    assert resources.stats.session_completion_tokens == 10
    assert resources.stats.known_cost_total == pytest.approx(0.00004)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_tool_only_is_valid(streaming):
    chunk = mock_llm_chunk(
        content="",
        tool_calls=[
            ToolCall(
                id="t", index=0, function=FunctionCall(name="tool", arguments="{}")
            )
        ],
    )
    backend = RecordingBackend([[chunk]])
    inputs, resources = setup_call(backend)
    outcomes = []
    await run_call(streaming, inputs, resources, outcomes)
    assert len(backend.requests_messages) == 1
    assert outcomes[0].message.tool_calls


@pytest.mark.asyncio
async def test_buffered_prefix_flushes_in_order():
    chunks = [
        LLMChunk(message=LLMMessage(role=Role.assistant)),
        LLMChunk(message=LLMMessage(role=Role.assistant, content=" \n")),
        LLMChunk(
            message=LLMMessage(
                role=Role.assistant, reasoning_payloads=[{"opaque": "x"}]
            )
        ),
        mock_llm_chunk(content="answer"),
    ]
    backend = RecordingBackend([chunks])
    inputs, resources = setup_call(backend)
    outcomes = []
    output = await run_call(True, inputs, resources, outcomes)
    assert [c.message.content for c in output] == [c.message.content for c in chunks]
    assert outcomes[0].message.content == " \nanswer"
    assert resources.stats.session_prompt_tokens == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "category,error_type",
    [(None, IncompleteLLMResponseError), ("max_output_tokens", ResponseTooLongError)],
)
@pytest.mark.parametrize("content", ["", " \n\t"])
async def test_known_empty_incomplete_is_terminal(
    streaming, category, error_type, content
):
    chunk = mock_llm_chunk(content=content).model_copy(
        update={"stop": StopInfo(reason="incomplete", category=category)}
    )
    backend = RecordingBackend([[chunk]])
    inputs, resources = setup_call(backend)
    with pytest.raises(error_type) as exc:
        await run_call(streaming, inputs, resources, [])
    assert type(exc.value) is error_type
    assert len(backend.requests_messages) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("category", [None, "max_output_tokens"])
@pytest.mark.parametrize("tool_only", [False, True])
async def test_known_nonempty_incomplete_is_accepted(streaming, category, tool_only):
    chunk = mock_llm_chunk(
        content="" if tool_only else "partial answer",
        tool_calls=[
            ToolCall(
                id="t", index=0, function=FunctionCall(name="tool", arguments="{}")
            )
        ]
        if tool_only
        else None,
    ).model_copy(update={"stop": StopInfo(reason="incomplete", category=category)})
    backend = RecordingBackend([[chunk]])
    inputs, resources = setup_call(backend)
    outcomes = []
    await run_call(streaming, inputs, resources, outcomes)
    assert len(backend.requests_messages) == 1
    assert len(outcomes) == 1
    assert outcomes[0].kind == "complete"
    assert (outcomes[0].message.content or "") == (chunk.message.content or "")
    assert outcomes[0].message.tool_calls == chunk.message.tool_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_refusal_is_not_empty(streaming):
    chunk = mock_llm_chunk(content="", stop_reason="refusal")
    backend = RecordingBackend([[chunk]])
    inputs, resources = setup_call(backend)
    with pytest.raises(RefusalError):
        await run_call(streaming, inputs, resources, [])
    assert len(backend.requests_messages) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", " \n", "answer"])
async def test_finish_marker_opt_out_still_validates_output(content):
    backend = RecordingBackend([[mock_llm_chunk(content=content, stop_reason=None)]])
    inputs, resources = setup_call(backend)
    inputs = replace(inputs, emits_finish_reason=False)
    if content.strip():
        await run_call(True, inputs, resources, [])
        assert len(backend.requests_messages) == 1
    else:
        with pytest.raises(EmptyLLMResponseError):
            await run_call(True, inputs, resources, [])
        assert len(backend.requests_messages) == 2


class WaitingBackend(FakeBackend):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = 0
        self.calls = 0

    async def complete_streaming(self, **kwargs):
        self.calls += 1
        try:
            if self.prefix is not None:
                yield LLMChunk(message=self.prefix)
            self.waiting.set()
            await self.release.wait()
            yield mock_llm_chunk(content="")
        finally:
            self.closed += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [None, " \n", "prose", "reasoning"])
async def test_cancellation_publication_and_closure(prefix):
    message = (
        None
        if prefix is None
        else LLMMessage(
            role=Role.assistant,
            content=prefix if prefix != "reasoning" else None,
            reasoning_content="thinking" if prefix == "reasoning" else None,
        )
    )
    backend = WaitingBackend(message)
    inputs, resources = setup_call(backend)
    resources = replace(resources, retry_budget=RequestRetryBudget(10))
    outcomes = []
    stream = LLMGateway().chat_streaming(inputs, resources, transcript=outcomes.append)
    published = prefix in {"prose", "reasoning"}
    if published:
        assert message is not None
        first = await anext(stream)
        assert first.message == message.model_copy(
            update={"posted_at": first.message.posted_at}
        )
        assert retry._LOGICAL_RETRY_BUDGET.get() is None
    task = asyncio.create_task(anext(stream))
    await backend.waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()
    assert backend.closed == 1
    assert backend.calls == 1
    assert len(outcomes) == int(published)
    if outcomes:
        assert outcomes[0].kind == "interrupted"
    assert retry._LOGICAL_RETRY_BUDGET.get() is None


@pytest.mark.asyncio
async def test_cancellation_during_discard_cleanup_does_not_replay(monkeypatch):
    backend = RecordingBackend([[mock_llm_chunk(content=" \n")]])
    inputs, resources = setup_call(backend)
    budget = RequestRetryBudget(10)
    resources = replace(resources, retry_budget=budget)
    backend_stream = backend.complete_streaming
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    class ClosingStream:
        def __init__(self, stream):
            self.stream = stream
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await anext(self.stream)

        async def aclose(self):
            assert retry._LOGICAL_RETRY_BUDGET.get() is budget
            cleanup_started.set()
            try:
                await cleanup_release.wait()
            finally:
                await self.stream.aclose()
                self.closed = True

    streams = []

    def complete_streaming(**kwargs):
        stream = ClosingStream(backend_stream(**kwargs))
        streams.append(stream)
        return stream

    monkeypatch.setattr(backend, "complete_streaming", complete_streaming)
    outcomes = []
    budgets_after_failure = []

    async def consume():
        try:
            await run_call(True, inputs, resources, outcomes)
        finally:
            # Probe in the consumer's task, not the parent's copied context.
            budgets_after_failure.append(retry._LOGICAL_RETRY_BUDGET.get())

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(cleanup_started.wait(), 1)
        assert not streams[0].closed
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        cleanup_release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert len(backend.requests_messages) == len(streams) == backend.closed == 1
    assert streams[0].closed
    assert outcomes == []
    assert budgets_after_failure == [None]
    assert retry._LOGICAL_RETRY_BUDGET.get() is None


@pytest.mark.asyncio
async def test_reasoning_liveness_and_explicit_post_publication_failure():
    backend = WaitingBackend(
        LLMMessage(role=Role.assistant, reasoning_content="thinking")
    )
    inputs, resources = setup_call(backend)
    outcomes = []
    stream = LLMGateway().chat_streaming(inputs, resources, transcript=outcomes.append)
    chunk = await asyncio.wait_for(anext(stream), 1)
    assert chunk.message.reasoning_content == "thinking"
    assert not backend.release.is_set()
    backend.release.set()
    with pytest.raises(EmptyLLMResponseError):
        [c async for c in stream]
    assert backend.calls == backend.closed == 1
    assert outcomes[0].kind == "interrupted"


@pytest.mark.asyncio
async def test_aclose_after_output_closes_backend():
    backend = WaitingBackend(LLMMessage(role=Role.assistant, content="partial"))
    inputs, resources = setup_call(backend)
    resources = replace(resources, retry_budget=RequestRetryBudget(10))
    outcomes = []
    stream = LLMGateway().chat_streaming(inputs, resources, transcript=outcomes.append)
    await anext(stream)
    assert retry._LOGICAL_RETRY_BUDGET.get() is None
    await stream.aclose()
    assert backend.closed == backend.calls == 1
    assert outcomes[0].kind == "interrupted"
    assert outcomes[0].message.content == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_cancellation_during_backoff_does_not_replay(streaming):
    backend = RecordingBackend([[mock_llm_chunk(content="")]])
    inputs, resources = setup_call(backend)
    notice = asyncio.Event()

    async def observe(reason):
        notice.set()

    resources = replace(resources, on_retry=observe)
    outcomes = []
    task = asyncio.create_task(run_call(streaming, inputs, resources, outcomes))
    await notice.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(backend.requests_messages) == 1
    assert outcomes == []
    if streaming:
        assert backend.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "family", ["openai", "anthropic-empty", "anthropic-redacted", "responses"]
)
async def test_real_adapter_empty_output_reaches_gateway(family):
    provider = ProviderConfig(
        name="test", api_base="https://test.invalid", api_key_env_var="TEST_KEY"
    )
    if family == "openai":
        chunk = OpenAIAdapter().parse_response({"choices": []}, provider)
    elif family.startswith("anthropic"):
        content = (
            [{"type": "redacted_thinking", "data": "opaque"}]
            if family.endswith("redacted")
            else []
        )
        chunk = AnthropicAdapter().parse_response({
            "content": content,
            "stop_reason": "end_turn",
        })
    else:
        chunk = OpenAIResponsesAdapter().parse_response(
            {"type": "response.completed", "response": {"output": []}}, provider
        )
    backend = RecordingBackend([[chunk], [chunk]])
    inputs, resources = setup_call(backend)
    with pytest.raises(EmptyLLMResponseError):
        await run_call(family == "responses", inputs, resources, [])
    assert len(backend.requests_messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_shared_deadline_blocks_empty_replay(streaming, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("chartreux.core.llm.failures.time.monotonic", lambda: clock[0])
    budget = RequestRetryBudget(1)

    class SlowBackend(RecordingBackend):
        async def complete(self, **kwargs):
            result = await super().complete(**kwargs)
            clock[0] = 2.0
            return result

        async def complete_streaming(self, **kwargs):
            async for chunk in super().complete_streaming(**kwargs):
                clock[0] = 2.0
                yield chunk

    backend = SlowBackend([[mock_llm_chunk(content="")]])
    inputs, resources = setup_call(backend)
    resources = replace(resources, retry_budget=budget)
    with pytest.raises(EmptyLLMResponseError):
        await run_call(streaming, inputs, resources, [])
    assert len(backend.requests_messages) == 1


@pytest.mark.asyncio
async def test_close_before_start_does_not_call_backend():
    backend = WaitingBackend(None)
    inputs, resources = setup_call(backend)
    outcomes = []
    stream = LLMGateway().chat_streaming(inputs, resources, transcript=outcomes.append)
    await stream.aclose()
    assert backend.calls == 0
    assert outcomes == []


@pytest.mark.asyncio
async def test_nonstreaming_cancellation_before_output_does_not_replay():
    waiting = asyncio.Event()
    calls = []

    class BlockedBackend(FakeBackend):
        async def complete(self, **kwargs):
            calls.append(kwargs)
            waiting.set()
            await asyncio.Event().wait()
            return mock_llm_chunk(content="unreachable")

    inputs, resources = setup_call(BlockedBackend())
    outcomes = []
    task = asyncio.create_task(run_call(False, inputs, resources, outcomes))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) == 1
    assert outcomes == []


@pytest.mark.asyncio
async def test_zero_chunk_stream_is_empty_not_incomplete():
    backend = RecordingBackend([[], []])
    inputs, resources = setup_call(backend)
    outcomes = []
    with pytest.raises(EmptyLLMResponseError):
        await run_call(True, inputs, resources, outcomes)
    assert len(backend.requests_messages) == backend.closed == 2
    assert resources.stats.has_unknown_cost
    assert outcomes == []


@pytest.mark.asyncio
async def test_responses_adapter_incomplete_bookkeeping_is_terminal():
    provider = ProviderConfig(
        name="test", api_base="https://test.invalid", api_key_env_var="TEST_KEY"
    )
    chunk = OpenAIResponsesAdapter().parse_response(
        {
            "type": "response.incomplete",
            "response": {
                "output": [],
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        },
        provider,
    )
    backend = RecordingBackend([[chunk]])
    inputs, resources = setup_call(backend)
    with pytest.raises(IncompleteLLMResponseError) as exc:
        await run_call(True, inputs, resources, [])
    assert type(exc.value) is IncompleteLLMResponseError
    assert len(backend.requests_messages) == 1


def test_error_mapper_preserves_empty_response_identity():
    inputs, _ = setup_call(FakeBackend())
    error = EmptyLLMResponseError(inputs.provider_name, inputs.model.name)
    assert _map_call_error(error, inputs) is error


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_mapped_empty_error_triggers_exactly_one_replay(streaming):
    backend = RecordingBackend([[mock_llm_chunk(content="")]] * 3)
    inputs, resources = setup_call(backend)
    notices = []

    async def observe(reason):
        notices.append(reason)

    resources = replace(resources, on_retry=observe)
    with pytest.raises(EmptyLLMResponseError) as caught:
        await run_call(streaming, inputs, resources, [])
    assert type(caught.value) is EmptyLLMResponseError
    assert len(backend.requests_messages) == 2
    assert len(notices) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("cap", ["tokens", "price"])
async def test_empty_replay_admission_reads_post_attempt_stats(streaming, cap):
    from tests.conftest import build_test_agent_loop

    backend = RecordingBackend([
        [mock_llm_chunk(content="")],
        [mock_llm_chunk(content="should not be called")],
    ])
    inputs, resources = setup_call(backend)
    agent = build_test_agent_loop(backend=backend)
    agent.stats = resources.stats
    if cap == "tokens":
        agent._max_session_tokens = 14
    else:
        agent._max_price = 0.000019
    assert agent._admit_empty_replay()
    observed = []

    def admit():
        observed.append((
            agent.stats.session_total_llm_tokens,
            agent.stats.known_cost_total,
        ))
        return agent._admit_empty_replay()

    resources = replace(resources, admit_replay=admit)
    with pytest.raises(EmptyLLMResponseError):
        await run_call(streaming, inputs, resources, [])
    assert len(backend.requests_messages) == 1
    assert len(observed) == 1
    assert observed[0][0] == 15
    assert observed[0][1] == pytest.approx(0.00002)
    await agent.aclose()
