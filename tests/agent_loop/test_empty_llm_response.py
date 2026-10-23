from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from types import MethodType
from typing import Any
from unittest.mock import Mock

import pytest

from chartreux.core.agent_loop import AgentTurnOptions
from chartreux.core.agent_loop.errors import (
    EmptyLLMResponseError,
    EmptyReplayBudgetStop,
)
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    IncompleteLLMResponseError,
    LLMGateway,
    TranscriptAppend,
    _map_call_error,
)
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig, ProviderConfig
from chartreux.core.config.models import RecoveryEligibleClass
from chartreux.core.errors import RefusalError, ResponseTooLongError
from chartreux.core.events import AssistantEvent, ReasoningEvent
from chartreux.core.llm.backend.anthropic import AnthropicAdapter
from chartreux.core.llm.backend.generic import OpenAIAdapter, notify_request_started
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
from chartreux.core.middleware import MiddlewareAction, RunBudgetMiddleware
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.session_types import AgentStats, CommittedModelIdentity
from chartreux.core.usage import UsageAttribution, UsageOutcome, UsageRecord
from chartreux.core.utils import VIBE_STOP_EVENT_TAG, retry
from tests.conftest import build_test_agent_loop, build_test_vibe_config
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
    with pytest.raises(EmptyReplayBudgetStop) as caught:
        await run_call(streaming, inputs, resources, [])
    # The terminal translation consults the same synchronous admission after
    # the retry predicate already denied the replay: both observations read
    # the post-attempt stats, and the signal carries the original error.
    assert len(backend.requests_messages) == 1
    assert len(observed) == 2
    assert observed[0][0] == 15
    assert observed[0][1] == pytest.approx(0.00002)
    assert observed[1] == observed[0]
    assert isinstance(caught.value.empty_error, EmptyLLMResponseError)
    assert caught.value.__cause__ is caught.value.empty_error
    await agent.aclose()


# --- B1: replay exhaustion propagates through the middleware budget-STOP path.
#
# Loop-level helpers shared by the frozen budget-stop tests. Every scenario
# drives a real AgentLoop through act() with a fake backend; the STOP event is
# the outward middleware event the loop emits after re-evaluating the full
# pipeline at the EmptyReplayBudgetStop catch boundary.


def _budget_config(*, unpriced: bool = False) -> ChartreuxConfigSchema:
    """A one-model test config with explicit $1/$2 per-million prices."""
    config = build_test_vibe_config(
        providers=[{"name": "test", "api_base": "https://test.invalid"}],
        models=[
            ModelConfig(
                name="test",
                provider="test",
                alias="test",
                input_price=1,
                output_price=2,
            )
        ],
        active_model="test",
    )
    if unpriced:
        assert config.catalog_snapshot is not None
        raw = config.catalog_snapshot.catalog.model_dump(mode="json")
        raw["models"]["test"]["deployments"][0]["prices"]["input"] = None
        raw["models"]["test"]["deployments"][0]["prices"]["output"] = None
        config.attach_catalog_snapshot(
            CatalogSnapshot(ModelCatalog.model_validate(raw), "unpriced")
        )
    return config


def _empty_chunk(
    prompt: int = 10,
    completion: int = 5,
    *,
    reasoning: str | None = None,
    no_usage: bool = False,
) -> LLMChunk:
    chunk = mock_llm_chunk(
        content=" \n",
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_content=reasoning,
    )
    if no_usage:
        chunk = chunk.model_copy(update={"usage": None})
    return chunk


async def _drive(
    agent: Any, message: str = "hello", *, turn_options: AgentTurnOptions | None = None
) -> list[Any]:
    events = []
    async for event in agent.act(message, turn_options=turn_options):
        events.append(event)
    return events


def _stop_events(events: list[Any]) -> list[AssistantEvent]:
    return [
        event
        for event in events
        if isinstance(event, AssistantEvent) and event.stopped_by_middleware
    ]


def _stop_content(reason: str) -> str:
    return f"<{VIBE_STOP_EVENT_TAG}>{reason}</{VIBE_STOP_EVENT_TAG}>"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
async def test_empty_response_price_exhaustion_emits_budget_stop(streaming) -> None:
    # Scenario table: the run-price denial and the absolute-price denial are
    # covered separately. One empty transport attempt; the replay admission
    # denies the replay; the loop terminates through the middleware STOP path
    # with the exact pipeline reason and no surfaced provider error.
    scenarios = [
        ({"run_max_price": 0.1}, "Run price limit exceeded: $0.2000 > $0.10"),
        ({"max_price": 0.1}, "Price limit exceeded: $0.2000 > $0.10"),
    ]
    for budget_kwargs, expected_reason in scenarios:
        backend = RecordingBackend([[_empty_chunk(prompt=100_000, completion=50_000)]])
        agent = build_test_agent_loop(
            config=_budget_config(),
            backend=backend,
            enable_streaming=streaming,
            **budget_kwargs,
        )
        try:
            events = await _drive(agent)
        finally:
            await agent.aclose()
        stops = _stop_events(events)
        assert len(stops) == 1, (budget_kwargs, events)
        assert stops[0].content == _stop_content(expected_reason)
        assert stops[0].middleware_metadata == {}
        assert len(backend.requests_messages) == 1
        assert len([e for e in events if isinstance(e, AssistantEvent)]) == 1
        assert agent.stats.session_prompt_tokens == 100_000
        assert agent.stats.known_cost_total == pytest.approx(0.2)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
async def test_empty_response_token_exhaustion_emits_budget_stop(streaming) -> None:
    # Run-token denial with the session tokens still affordable: the emitted
    # reason is the anchor-relative run-token one, not the session total.
    backend = RecordingBackend([[_empty_chunk(prompt=80, completion=30)]])
    agent = build_test_agent_loop(
        config=_budget_config(),
        backend=backend,
        enable_streaming=streaming,
        max_session_tokens=1_000_000,
        run_max_session_tokens=100,
    )
    # Pre-run consumption sits in the run anchor: only the run delta counts.
    agent.stats.session_prompt_tokens = 5_000
    try:
        events = await _drive(agent)
    finally:
        await agent.aclose()
    stops = _stop_events(events)
    assert len(stops) == 1, events
    assert stops[0].content == _stop_content("Run token limit exceeded: 110 > 100")
    assert stops[0].middleware_metadata == {}
    assert len(backend.requests_messages) == 1
    # The session total carries the pre-run baseline; the reason reports the
    # anchor-relative run delta (110), and the session cap stayed affordable.
    assert agent.stats.session_total_llm_tokens == 5_110
    assert agent.stats.session_total_llm_tokens <= 1_000_000


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
async def test_empty_response_unknown_cost_preserves_unverifiable_code(
    streaming,
) -> None:
    # Scenario table: the absolute-price and run-budget unknown-cost paths,
    # including absent usage. The STOP keeps the exact budget-unverifiable
    # metadata the public/run stop mappings project from.
    scenarios = [
        # (label, config kwargs, budget kwargs, chunk kwargs)
        ("absolute price, absent usage", {}, {"max_price": 1.0}, {"no_usage": True}),
        ("run budget, absent usage", {}, {"run_max_price": 1.0}, {"no_usage": True}),
        ("absolute price, unpriced model", {"unpriced": True}, {"max_price": 1.0}, {}),
    ]
    for label, config_kwargs, budget_kwargs, chunk_kwargs in scenarios:
        backend = RecordingBackend([[_empty_chunk(**chunk_kwargs)]])
        agent = build_test_agent_loop(
            config=_budget_config(**config_kwargs),
            backend=backend,
            enable_streaming=streaming,
            **budget_kwargs,
        )
        try:
            events = await _drive(agent)
        finally:
            await agent.aclose()
        stops = _stop_events(events)
        assert len(stops) == 1, (label, events)
        if "run budget" in label:
            expected_reason = (
                "Budget unverifiable: usage with unknown pricing was recorded "
                "during this run"
            )
        else:
            expected_reason = (
                "Budget unverifiable: usage with unknown pricing was recorded "
                "(known-cost lower bound $0.0000)"
            )
        assert stops[0].content == _stop_content(expected_reason), label
        assert stops[0].middleware_metadata == {"code": "budget-unverifiable"}, label
        assert len(backend.requests_messages) == 1, label
        assert agent.stats.has_unknown_cost, label


def _failover_snapshot() -> CatalogSnapshot:
    """Two providers serving one base model, with explicit $1/$2 prices."""
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": "https://first.invalid"},
                "test-second": {"api_base": "https://second.invalid"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "prices": {"input": 1, "output": 2},
                        },
                        {
                            "provider": "test-second",
                            "name": "second",
                            "prices": {"input": 1, "output": 2},
                        },
                    ]
                }
            },
            "roles": {},
        }),
        "test-revision",
    )


def _failover_backends(agent: Any, first: FakeBackend, second: FakeBackend) -> None:
    def backend_for_attempt(_self: Any, model: ModelConfig, _budget: object) -> Any:
        return {"test-first": first, "test-second": second}[model.provider]

    agent._backend_for_attempt = MethodType(backend_for_attempt, agent)  # type: ignore[method-assign]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
async def test_budget_stop_prevents_replay_and_provider_failover(streaming) -> None:
    # An AVAILABLE alternative deployment (test-second) and recovery
    # eligibility for an ordinary EMPTY_RESPONSE are both configured, and the
    # request deadline is exhausted from the start (api_retry_max_elapsed_time
    # 0.0), so the in-call retry decorator short-circuits on the deadline
    # without ever consulting the replay predicate. The budget stop still
    # propagates through the terminal translation: no second provider request,
    # no retry notification, no provider switch, no self-heal, and no
    # post-STOP recovery admission (act() returns normally — the recovery
    # gate's only input is a raised terminal failure). The streaming variant
    # covers already-published reasoning.
    snapshot = _failover_snapshot()
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base", "api_retry_max_elapsed_time": 0.0},
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)
    # An ordinary empty response is recovery-eligible under this policy.
    assert RecoveryEligibleClass.EMPTY_RESPONSE in config.recovery.root.eligible_classes

    first = RecordingBackend(
        [[_empty_chunk(prompt=100_000, completion=50_000, reasoning="thinking")]]
        if streaming
        else [[_empty_chunk(prompt=100_000, completion=50_000)]]
    )
    second = RecordingBackend([[mock_llm_chunk(content="must not be called")]])
    agent = build_test_agent_loop(
        config=config,
        backend=FakeBackend(),
        enable_streaming=streaming,
        run_max_price=0.1,
    )
    agent.committed_model = CommittedModelIdentity(
        base_model="base",
        provider="test-first",
        wire_name="first",
        catalog_revision="test-revision",
    )
    _failover_backends(agent, first, second)
    notices: list[Any] = []

    async def observe(reason: Any) -> None:
        notices.append(reason)

    outcomes: list[TranscriptAppend] = []
    append_transcript = agent._append_transcript

    def capture(outcome: TranscriptAppend) -> None:
        outcomes.append(outcome)
        append_transcript(outcome)

    agent._append_transcript = capture  # type: ignore[method-assign]
    try:
        events = await _drive(agent, turn_options=AgentTurnOptions(retry_sink=observe))
    finally:
        await agent.aclose()

    stops = _stop_events(events)
    assert len(stops) == 1, events
    assert stops[0].content == _stop_content(
        "Run price limit exceeded: $0.2000 > $0.10"
    )
    # No replay and no failover: one transport attempt on the first provider.
    assert len(first.requests_messages) == 1
    assert len(second.requests_messages) == 0
    # No retry notification, no provider switch, no self-heal.
    assert notices == []
    assert agent.committed_model == CommittedModelIdentity(
        base_model="base",
        provider="test-first",
        wire_name="first",
        catalog_revision="test-revision",
    )
    metadata = agent.completion_metadata_since((0, 0))
    assert metadata["providers_used"] == [["test-first"]]
    assert metadata["switch_notices"] == []
    assert agent._reactive_recovery_used is False
    if streaming:
        # Already-published reasoning: the outward reasoning event fired and
        # the interrupted transcript append carried it before the stop.
        assert any(
            isinstance(event, ReasoningEvent) and event.content == "thinking"
            for event in events
        )
        assert len(outcomes) == 1
        assert outcomes[0].kind == "interrupted"
        assert outcomes[0].message.reasoning_content == "thinking"
        assert first.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
async def test_unexhausted_empty_response_preserves_existing_replay(streaming) -> None:
    # Empty then valid: two identical requests, one retry notice, one
    # successful transcript append. The exact-limit affordability (strict >)
    # and the pre-run-baseline anchoring are preserved.
    backend = RecordingBackend([[_empty_chunk()], [mock_llm_chunk(content="answer")]])
    agent = build_test_agent_loop(
        config=_budget_config(),
        backend=backend,
        enable_streaming=streaming,
        run_max_session_tokens=30,
    )
    # Pre-run consumption sits in the run anchor: only the run delta counts.
    agent.stats.session_prompt_tokens = 1_000
    notices: list[Any] = []

    async def observe(reason: Any) -> None:
        notices.append(reason)

    outcomes: list[TranscriptAppend] = []
    append_transcript = agent._append_transcript

    def capture(outcome: TranscriptAppend) -> None:
        outcomes.append(outcome)
        append_transcript(outcome)

    agent._append_transcript = capture  # type: ignore[method-assign]
    try:
        events = await _drive(agent, turn_options=AgentTurnOptions(retry_sink=observe))
    finally:
        await agent.aclose()

    assert len(backend.requests_messages) == 2
    assert backend.requests_messages[0] == backend.requests_messages[1]
    assert backend.requests_metadata[0] == backend.requests_metadata[1]
    assert len(notices) == 1
    assert notices[0].detail == "Empty assistant response; retrying once"
    assert len(outcomes) == 1
    assert outcomes[0].kind == "complete"
    assert outcomes[0].message.content == "answer"
    assistants = [e for e in events if isinstance(e, AssistantEvent)]
    assert len(assistants) == 1
    assert assistants[0].content == "answer"
    assert not assistants[0].stopped_by_middleware
    assert _stop_events(events) == []
    # Pre-run baseline plus two 15-token attempts; the run delta is exactly
    # the 30-token limit, which the strict > comparison still affords.
    assert agent.stats.session_total_llm_tokens == 1_030
    assert (
        RunBudgetMiddleware(run_max_session_tokens=30).evaluate(agent.stats).action
        is MiddlewareAction.CONTINUE
    )
    assert agent._admit_empty_replay() is True


class AccountingBackend(RecordingBackend):
    """A recording backend that trips the transport-start observer (WP1)."""

    async def complete(self, **kwargs: Any) -> LLMChunk:
        notify_request_started()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        notify_request_started()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


async def _usage_scenario(
    *, streaming: bool, label: str, budget_kwargs: dict[str, Any], attempts: int
) -> None:
    """One exactly-once scenario: one record per transport attempt, finalized
    before the middleware re-evaluation, with no extra record from the STOP.
    """
    backend = AccountingBackend([
        [_empty_chunk(prompt=100_000, completion=50_000)] for _ in range(attempts)
    ])
    agent = build_test_agent_loop(
        config=_budget_config(),
        backend=backend,
        enable_streaming=streaming,
        **budget_kwargs,
    )
    records: list[UsageRecord] = []

    async def sink(record: UsageRecord) -> None:
        records.append(record)

    agent._accounting_sink = sink  # type: ignore[attr-defined]
    agent._usage_attribution = UsageAttribution(  # type: ignore[attr-defined]
        root_session_id="root",
        session_id="session",
        agent_role="root",
        model="test",
        provider="test",
        wire_name="test",
        project_key="project",
    )
    # The pipeline re-evaluation must observe the finalized records: the
    # first (pre-turn) call sees none, the STOP re-evaluation sees all.
    snapshots: list[int] = []
    run_before_turn = agent.middleware_pipeline.run_before_turn

    async def spy(context: Any) -> Any:
        snapshots.append(len(records))
        return await run_before_turn(context)

    agent.middleware_pipeline.run_before_turn = spy  # type: ignore[method-assign]
    try:
        events = await _drive(agent)
    finally:
        await agent.aclose()

    stops = _stop_events(events)
    assert len(stops) == 1, (label, events)
    assert len(backend.requests_messages) == attempts, label
    if streaming:
        assert backend.closed == attempts, label
    # Exactly one usage record per transport attempt; the STOP adds none.
    assert len(records) == attempts, label
    assert [record.attempt_ordinal for record in records] == list(
        range(1, attempts + 1)
    ), label
    assert all(record.input_tokens == 100_000 for record in records), label
    assert all(record.output_tokens == 50_000 for record in records), label
    assert all(record.outcome == UsageOutcome.FAILED for record in records), label
    # Exact accumulated counters and cost reconcile with the records.
    assert agent.stats.session_prompt_tokens == 100_000 * attempts, label
    assert agent.stats.session_completion_tokens == 50_000 * attempts, label
    assert agent.stats.known_cost_total == pytest.approx(0.2 * attempts), label
    assert sum(record.input_tokens or 0 for record in records) == (
        agent.stats.session_prompt_tokens
    ), label
    assert sum(record.known_cost_usd for record in records) == pytest.approx(
        agent.stats.known_cost_total
    ), label
    # Finalization completed BEFORE the middleware re-evaluation.
    assert snapshots == [0, attempts], label


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
async def test_budget_stop_preserves_attempt_usage_exactly_once(streaming) -> None:
    # Scenario table: first-attempt exhaustion and affordable-first /
    # exhausting-second-attempt.
    scenarios = [
        # (label, budget kwargs, attempts)
        ("first-attempt exhaustion", {"run_max_price": 0.1}, 1),
        ("affordable first, exhausting second", {"run_max_price": 0.3}, 2),
    ]
    for label, budget_kwargs, attempts in scenarios:
        await _usage_scenario(
            streaming=streaming,
            label=label,
            budget_kwargs=budget_kwargs,
            attempts=attempts,
        )
