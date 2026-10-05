from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC
from typing import Any, Literal
from unittest.mock import AsyncMock

import httpx
import pytest

from chartreux.core.agent_loop.errors import AgentLoopLLMResponseError
from chartreux.core.agent_loop.llm_gateway import (
    CallFinalizer,
    CallResources,
    CompletionInputs,
    LLMGateway,
)
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.errors import RefusalError
from chartreux.core.llm.backend.generic import (
    _ADAPTERS,
    GenericBackend,
    notify_request_started,
    request_start_observer,
)
from chartreux.core.llm.failures import classify_failure
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role, StopInfo
from chartreux.core.session.title_model import generate_session_title
from chartreux.core.session_types import AgentStats
from chartreux.core.usage import (
    UsageAttribution,
    UsageOutcome,
    UsagePurpose,
    UsageRecord,
    UsageState,
    aggregate_usage,
)
from tests.stubs.fake_backend import FakeBackend

ATTRIBUTION = UsageAttribution(
    root_session_id="root",
    session_id="session",
    agent_role="root",
    model="stale-model",
    provider="stale-provider",
    wire_name="stale-wire",
    project_key="project",
)
USAGE = LLMUsage(prompt_tokens=100, completion_tokens=20, cached_tokens=10)


def inputs() -> CompletionInputs:
    return CompletionInputs(
        model=ModelConfig(
            name="wire",
            alias="model",
            provider="provider",
            input_price=1,
            output_price=2,
            cached_input_price=0.5,
        ),
        provider_name="provider",
        emits_finish_reason=True,
        messages=(LLMMessage(role=Role.user, content="hello"),),
        tools=None,
        tool_choice=None,
        extra_headers={},
        metadata={},
        max_tokens=None,
    )


def chunk(*, usage: LLMUsage | None = USAGE, refusal: bool = False) -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.assistant, content="answer"),
        usage=usage,
        stop=StopInfo(reason="refusal" if refusal else "stop"),
    )


class AttemptBackend(FakeBackend):
    def __init__(
        self, result: LLMChunk | None = None, error: BaseException | None = None
    ) -> None:
        super().__init__()
        self.result = result if result is not None else chunk()
        self.error = error
        self.calls = 0

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self.calls += 1
        notify_request_started()
        if self.error is not None:
            raise self.error
        return self.result


def resources(backend: Any, records: list[UsageRecord]) -> CallResources:
    async def sink(record: UsageRecord) -> None:
        records.append(record)

    return CallResources(
        backend=backend,
        stats=AgentStats(),
        process_message=lambda message: message,
        accounting_sink=sink,
        usage_attribution=ATTRIBUTION,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("second_empty", [False, True])
async def test_empty_replay_settles_two_distinct_attempt_records(
    streaming: bool, second_empty: bool
) -> None:
    from chartreux.core.agent_loop.errors import EmptyLLMResponseError

    empty = chunk().model_copy(
        update={"message": LLMMessage(role=Role.assistant, content=" \n")}
    )
    records: list[UsageRecord] = []

    class ReplayBackend(FakeBackend):
        async def complete(self, **kwargs: Any) -> LLMChunk:
            notify_request_started()
            return await super().complete(**kwargs)

        async def complete_streaming(self, **kwargs: Any):
            notify_request_started()
            async for part in super().complete_streaming(**kwargs):
                yield part

    backend = ReplayBackend([[empty], [empty if second_empty else chunk()]])
    call_resources = resources(backend, records)

    async def run() -> None:
        if streaming:
            _ = [
                part
                async for part in LLMGateway().chat_streaming(
                    inputs(), call_resources, transcript=lambda _: None
                )
            ]
        else:
            await LLMGateway().complete(inputs(), call_resources)

    if second_empty:
        with pytest.raises(EmptyLLMResponseError):
            await run()
    else:
        await run()
    assert len(records) == len(backend.requests_messages) == 2
    assert records[0].record_id != records[1].record_id
    assert [r.outcome for r in records] == [
        UsageOutcome.FAILED,
        UsageOutcome.FAILED if second_empty else UsageOutcome.COMPLETED,
    ]
    assert all(r.usage_state == UsageState.COMPLETE for r in records)
    assert all(r.input_tokens == 100 and r.output_tokens == 20 for r in records)
    assert call_resources.stats.session_prompt_tokens == 200
    assert call_resources.stats.known_cost_total == pytest.approx(
        sum(r.known_cost_usd for r in records)
    )


def test_title_scheduler_supplies_retry_budget_and_preserves_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    from tests.conftest import build_test_agent_loop

    agent = build_test_agent_loop(backend=AttemptBackend())
    records: list[UsageRecord] = []
    call_resources = resources(agent.backend, records)
    monkeypatch.setattr(
        agent, "_accounting_sink", call_resources.accounting_sink, raising=False
    )
    monkeypatch.setattr(agent, "_usage_attribution", ATTRIBUTION, raising=False)
    original = agent._call_resources
    budgets = []

    def capture(backend, budget):
        budgets.append(budget)
        return original(backend, budget)

    monkeypatch.setattr(agent, "_call_resources", capture)
    schedule = Mock(return_value=None)
    monkeypatch.setattr(agent._title_controller, "schedule", schedule)
    assert agent._maybe_schedule_title_generation(turn_completing=True) is None
    assert len(budgets) == 1 and not budgets[0].exhausted
    title_inputs = schedule.call_args.args[0]
    assert title_inputs.accounting_sink is call_resources.accounting_sink
    assert title_inputs.usage_attribution is ATTRIBUTION


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_success_exactly_once_and_stats_unchanged(chat: bool) -> None:
    records: list[UsageRecord] = []
    backend = AttemptBackend()
    call_resources = resources(backend, records)
    transcript = []
    gateway = LLMGateway()
    result = (
        await gateway.chat(inputs(), call_resources, transcript=transcript.append)
        if chat
        else await gateway.complete(inputs(), call_resources)
    )
    assert result.message.content == "answer"
    assert len(records) == backend.calls == 1
    record = records[0]
    assert record.outcome == UsageOutcome.COMPLETED
    assert record.usage_state == UsageState.COMPLETE
    assert (record.input_tokens, record.output_tokens, record.cached_input_tokens) == (
        100,
        20,
        10,
    )
    assert record.known_cost_usd == pytest.approx(0.000135)
    assert not record.has_unknown_cost
    assert len(transcript) == int(chat)
    assert call_resources.stats.session_prompt_tokens == 100
    assert call_resources.stats.session_completion_tokens == 20
    assert call_resources.stats.session_cached_tokens == 10
    assert call_resources.stats.context_tokens == 120
    assert call_resources.stats.known_cost_total == record.known_cost_usd


@pytest.mark.asyncio
async def test_refusal_accounted_before_chat_raises() -> None:
    records: list[UsageRecord] = []
    transcript = []
    with pytest.raises(RefusalError):
        await LLMGateway().chat(
            inputs(),
            resources(AttemptBackend(chunk(refusal=True)), records),
            transcript=transcript.append,
        )
    assert len(records) == len(transcript) == 1
    assert records[0].outcome == UsageOutcome.REFUSED
    assert records[0].input_tokens == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("refusal", [False, True])
@pytest.mark.parametrize("is_final", [False, True])
async def test_completed_and_refused_usage_classification_follows_finality(
    streaming: bool, refusal: bool, is_final: bool
) -> None:
    records: list[UsageRecord] = []
    result = chunk(
        usage=USAGE.model_copy(update={"is_final": is_final}), refusal=refusal
    )
    backend = StreamingAttemptBackend([result]) if streaming else AttemptBackend(result)
    call_resources = resources(backend, records)
    gateway = LLMGateway()

    async def run() -> None:
        if streaming:
            _ = [
                part
                async for part in gateway.chat_streaming(
                    inputs(), call_resources, transcript=lambda _: None
                )
            ]
        else:
            await gateway.chat(inputs(), call_resources, transcript=lambda _: None)

    if refusal:
        with pytest.raises(RefusalError):
            await run()
    else:
        await run()
    assert len(records) == 1
    record = records[0]
    assert record.outcome == (
        UsageOutcome.REFUSED if refusal else UsageOutcome.COMPLETED
    )
    assert record.usage_state == (
        UsageState.COMPLETE if is_final else UsageState.PARTIAL
    )
    assert record.has_unknown_cost is (not is_final)
    assert (record.input_tokens, record.output_tokens, record.cached_input_tokens) == (
        100,
        20,
        10,
    )
    assert record.known_cost_usd == pytest.approx(0.000135)


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, LLMUsage()])
async def test_attempted_missing_usage_is_unknown(usage: LLMUsage | None) -> None:
    records: list[UsageRecord] = []
    call = LLMGateway().complete(
        inputs(), resources(AttemptBackend(chunk(usage=usage)), records)
    )
    if usage is None:
        with pytest.raises(RuntimeError) as error:
            await call
        assert isinstance(error.value.__cause__, AgentLoopLLMResponseError)
    else:
        await call
    assert len(records) == 1
    assert records[0].usage_state == UsageState.MISSING
    assert records[0].input_tokens is records[0].output_tokens is None
    assert records[0].cached_input_tokens is None
    assert records[0].known_cost_usd == 0
    assert records[0].has_unknown_cost


@pytest.mark.asyncio
@pytest.mark.parametrize("with_usage", [False, True])
async def test_failure_preserves_classification_and_reported_usage(
    with_usage: bool,
) -> None:
    records: list[UsageRecord] = []
    error = httpx.ConnectError("offline")
    if with_usage:
        error.usage = LLMUsage.from_reported(completion_tokens=5)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError) as caught:
        await LLMGateway().complete(
            inputs(), resources(AttemptBackend(error=error), records)
        )
    assert caught.value.__cause__ is error
    assert classify_failure(caught.value) == classify_failure(error)
    assert len(records) == 1
    assert records[0].outcome == UsageOutcome.FAILED
    assert records[0].usage_state == (
        UsageState.PARTIAL if with_usage else UsageState.MISSING
    )
    assert records[0].output_tokens == (5 if with_usage else None)
    assert records[0].has_unknown_cost


@pytest.mark.asyncio
async def test_processing_failure_retains_raw_usage_and_stats() -> None:
    records: list[UsageRecord] = []
    error = ValueError("processing")

    def process(_message: LLMMessage) -> LLMMessage:
        raise error

    call_resources = replace(
        resources(AttemptBackend(), records), process_message=process
    )
    with pytest.raises(RuntimeError) as caught:
        await LLMGateway().complete(inputs(), call_resources)
    assert caught.value.__cause__ is error
    assert len(records) == 1
    assert records[0].outcome == UsageOutcome.FAILED
    assert records[0].input_tokens == 100
    assert call_resources.stats.session_prompt_tokens == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_cancellation_identity_and_attempt_boundary(started: bool) -> None:
    records: list[UsageRecord] = []
    error = asyncio.CancelledError("original cancellation")
    backend = AttemptBackend(error=error)
    if not started:
        backend.complete = AsyncMock(side_effect=error)
    with pytest.raises(asyncio.CancelledError) as caught:
        await LLMGateway().complete(inputs(), resources(backend, records))
    assert caught.value is error
    assert len(records) == int(started)
    if started:
        assert records[0].outcome == UsageOutcome.INTERRUPTED
        assert records[0].usage_state == UsageState.MISSING


@pytest.mark.asyncio
async def test_timeout_after_attempt_records_missing() -> None:
    records: list[UsageRecord] = []
    entered = asyncio.Event()

    async def complete(**_kwargs: Any) -> LLMChunk:
        notify_request_started()
        entered.set()
        await asyncio.Event().wait()
        return chunk()

    backend = AttemptBackend()
    backend.complete = complete

    async def run_with_timeout() -> None:
        async with asyncio.timeout(0.01):
            await LLMGateway().complete(inputs(), resources(backend, records))

    task = asyncio.create_task(run_with_timeout())
    await entered.wait()
    with pytest.raises(TimeoutError):
        await task
    assert len(records) == 1
    assert records[0].outcome == UsageOutcome.INTERRUPTED


@pytest.mark.asyncio
@pytest.mark.parametrize("api_style", ["openai", "anthropic", "openai-responses"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_adapter_preparation_failure_never_signals_attempt(
    monkeypatch: pytest.MonkeyPatch,
    api_style: Literal["openai", "anthropic", "openai-responses"],
    streaming: bool,
) -> None:
    records: list[UsageRecord] = []
    backend = GenericBackend(
        provider=ProviderConfig(
            name="provider", api_base="https://provider.invalid", api_style=api_style
        ),
        resolved_credential=None,
    )

    def fail_preparation(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("preparation")

    adapter = _ADAPTERS[api_style](None)
    monkeypatch.setattr(adapter, "prepare_request", fail_preparation)
    monkeypatch.setitem(_ADAPTERS, api_style, lambda _call: adapter)
    if streaming:
        with pytest.raises(RuntimeError):
            _ = [
                part
                async for part in LLMGateway().chat_streaming(
                    inputs(), resources(backend, records), transcript=lambda _: None
                )
            ]
        assert not records
    else:
        with pytest.raises(RuntimeError):
            await LLMGateway().complete(inputs(), resources(backend, records))
        assert not records


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_error", [False, True])
async def test_sink_failure_never_changes_inference_result(
    caplog: pytest.LogCaptureFixture, backend_error: bool
) -> None:
    records: list[UsageRecord] = []
    error = httpx.ConnectError("offline") if backend_error else None
    backend = AttemptBackend(error=error)

    async def sink(_record: UsageRecord) -> None:
        raise OSError("sensitive error text")

    call_resources = replace(resources(backend, records), accounting_sink=sink)
    if error is not None:
        with pytest.raises(RuntimeError) as caught:
            await LLMGateway().complete(inputs(), call_resources)
        assert caught.value.__cause__ is error
        assert classify_failure(caught.value) == classify_failure(error)
    else:
        assert (
            await LLMGateway().complete(inputs(), call_resources)
        ).message.content == "answer"
    assert backend.calls == 1
    assert "coverage is degraded" in caplog.text
    assert "sensitive error text" not in caplog.text


@pytest.mark.asyncio
async def test_frozen_candidate_identity_prices_and_purpose() -> None:
    records: list[UsageRecord] = []
    first = inputs()
    second = replace(
        first,
        model=ModelConfig(
            name="other-wire",
            alias="other-model",
            provider="other-provider",
            input_price=3,
            output_price=4,
            cached_input_price=1,
        ),
        provider_name="other-provider",
        purpose=UsagePurpose.COMPACTION,
    )
    for candidate in (first, second):
        backend = AttemptBackend()

        async def complete(
            *, attempt: CompletionInputs = candidate, **_kwargs: Any
        ) -> LLMChunk:
            notify_request_started()
            attempt.model.input_price = 99
            return chunk()

        backend.complete = complete
        await LLMGateway().complete(candidate, resources(backend, records))
    assert [(r.model, r.provider, r.wire_name) for r in records] == [
        ("model", "provider", "wire"),
        ("other-model", "other-provider", "other-wire"),
    ]
    assert [r.prices_usd_per_million.input for r in records] == [1, 3]
    assert records[1].purpose == UsagePurpose.COMPACTION
    assert all(r.root_session_id == "root" for r in records)


@pytest.mark.asyncio
async def test_shared_finalizer_exactly_once_under_repeated_cancellation() -> None:
    records: list[UsageRecord] = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def sink(record: UsageRecord) -> None:
        entered.set()
        await release.wait()
        records.append(record)

    finalizer = CallFinalizer(
        inputs(), replace(resources(AttemptBackend(), records), accounting_sink=sink)
    )
    finalizer.start()
    finalizer.usage = USAGE
    finalizer.outcome = UsageOutcome.COMPLETED
    task = asyncio.create_task(finalizer.finalize())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await finalizer.finalize()
    assert len(records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("sink_raises", [False, True])
async def test_real_failover_accounts_candidate_inputs_not_session_state(
    monkeypatch: pytest.MonkeyPatch, sink_raises: bool
) -> None:
    from tests.agent_loop.test_failover_coordinator import _agent

    agent = _agent()
    records: list[UsageRecord] = []
    first = AttemptBackend(error=httpx.ConnectError("offline"))
    second = AttemptBackend()
    original_attempt_model = agent._attempt_model

    def attempt_model(*args: Any, **kwargs: Any) -> ModelConfig:
        model = original_attempt_model(*args, **kwargs)
        model.input_price = 1 if model.provider == "test-first" else 3
        model.input_price_known = True
        return model

    def backend_for_attempt(model: ModelConfig, _budget: Any) -> AttemptBackend:
        return first if model.provider == "test-first" else second

    async def sink(record: UsageRecord) -> None:
        records.append(record)
        if sink_raises:
            raise OSError("ledger unavailable")

    monkeypatch.setattr(agent, "_attempt_model", attempt_model)
    monkeypatch.setattr(agent, "_backend_for_attempt", backend_for_attempt)
    monkeypatch.setattr(agent, "_accounting_sink", sink, raising=False)
    monkeypatch.setattr(agent, "_usage_attribution", ATTRIBUTION, raising=False)
    assert (await agent._chat()).message.content == "answer"
    assert first.calls == second.calls == 1
    assert len(records) == 2
    assert [(record.provider, record.wire_name) for record in records] == [
        ("test-first", "first"),
        ("test-second", "second"),
    ]
    assert [record.prices_usd_per_million.input for record in records] == [1, 3]
    assert [record.outcome for record in records] == [
        UsageOutcome.FAILED,
        UsageOutcome.COMPLETED,
    ]
    assert records[0].usage_state == UsageState.MISSING
    assert all(record.purpose == UsagePurpose.CONVERSATION for record in records)
    assert agent.committed_model is not None
    assert agent.committed_model.provider == "test-second"


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_untagged_loop_completion_defaults_to_conversation(
    monkeypatch: pytest.MonkeyPatch, chat: bool
) -> None:
    from tests.conftest import build_test_agent_loop

    agent = build_test_agent_loop(backend=AttemptBackend())
    records: list[UsageRecord] = []

    async def sink(record: UsageRecord) -> None:
        records.append(record)

    monkeypatch.setattr(agent, "_accounting_sink", sink, raising=False)
    monkeypatch.setattr(agent, "_usage_attribution", ATTRIBUTION, raising=False)
    if chat:
        await agent._chat()
    else:
        await agent._complete(
            model=agent.config.get_active_model(),
            messages=agent.messages,
            tools=None,
            tool_choice=None,
            call_type=None,
        )
    assert len(records) == 1
    assert records[0].purpose == UsagePurpose.CONVERSATION
    assert agent.stats.session_prompt_tokens == USAGE.prompt_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "failover", "retry", "fallback"])
async def test_compaction_accounts_each_attempt_with_frozen_deployment(
    monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    from tests.agent_loop.test_agent_auto_compact import _ctx_too_long_error
    from tests.agent_loop.test_failover_coordinator import _agent

    agent = _agent(compaction=True)
    agent.messages.extend([
        LLMMessage(role=Role.user, content="old ask"),
        LLMMessage(role=Role.assistant, content="old answer"),
        LLMMessage(role=Role.user, content="new ask"),
    ])
    records: list[UsageRecord] = []
    attempted_models: list[ModelConfig] = []
    original_attempt_model = agent._attempt_model
    stats_before = agent.stats.model_dump()

    def attempt_model(*args: Any, **kwargs: Any) -> ModelConfig:
        model = original_attempt_model(*args, **kwargs)
        model.input_price = 1 if model.provider == "test-first" else 3
        model.input_price_known = True
        return model

    async def complete(*, model: ModelConfig, **_kwargs: Any) -> LLMChunk:
        notify_request_started()
        attempted_models.append(model)
        # Accounting must use prices frozen before the backend was invoked.
        model.input_price = 99
        if len(attempted_models) == 1:
            if scenario == "failover":
                raise httpx.ConnectError("offline")
            if scenario == "retry":
                raise _ctx_too_long_error()
        if scenario == "fallback" and len(attempted_models) <= 2:
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=""), usage=USAGE
            )
        return LLMChunk(
            message=LLMMessage(role=Role.assistant, content="<summary>done</summary>"),
            usage=USAGE,
        )

    def backend_for_attempt(_model: ModelConfig, _budget: Any) -> AttemptBackend:
        backend = AttemptBackend()
        backend.complete = complete
        return backend

    async def sink(record: UsageRecord) -> None:
        records.append(record)

    monkeypatch.setattr(agent, "_attempt_model", attempt_model)
    monkeypatch.setattr(agent, "_backend_for_attempt", backend_for_attempt)
    monkeypatch.setattr(agent, "_accounting_sink", sink, raising=False)
    monkeypatch.setattr(agent, "_usage_attribution", ATTRIBUTION, raising=False)
    assert await agent.compaction_manager.compact() == "done"
    expected_calls = 1 if scenario == "success" else 3 if scenario == "fallback" else 2
    assert len(records) == len(attempted_models) == expected_calls
    assert all(record.purpose == UsagePurpose.COMPACTION for record in records)
    expected_providers = ["test-first"]
    if scenario != "success":
        expected_providers.append(
            "test-second" if scenario == "failover" else "test-first"
        )
    if scenario == "fallback":
        expected_providers.append("test-first")
    assert [
        (record.model, record.provider, record.wire_name) for record in records
    ] == [
        (
            "compact",
            provider,
            "compact-first" if provider == "test-first" else "compact-second",
        )
        for provider in expected_providers
    ]
    expected_prices = [
        1 if provider == "test-first" else 3 for provider in expected_providers
    ]
    if scenario == "fallback":
        # The identical empty replay freezes the now-mutated model anew; the
        # subsequent compaction fallback constructs a fresh deployment model.
        expected_prices[1] = 99
    assert [
        record.prices_usd_per_million.input for record in records
    ] == expected_prices
    assert [record.outcome for record in records] == (
        [UsageOutcome.FAILED] * (len(records) - 1) + [UsageOutcome.COMPLETED]
    )
    # Compaction still adds session spend/tokens, but not conversation/context usage.
    assert agent.stats.session_prompt_tokens == sum(
        record.input_tokens or 0 for record in records
    )
    assert agent.stats.session_completion_tokens == sum(
        record.output_tokens or 0 for record in records
    )
    for field in (
        "last_turn_prompt_tokens",
        "last_turn_completion_tokens",
        "last_turn_cached_tokens",
        "last_turn_duration",
        "tokens_per_second",
    ):
        assert getattr(agent.stats, field) == stats_before[field]
    assert agent.stats.context_tokens == -1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["generic", "mistral", "anthropic", "openai-responses"]
)
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_transport_signals_started_after_preparation(
    monkeypatch: pytest.MonkeyPatch, kind: str, streaming: bool
) -> None:
    from chartreux.core.llm.backend.mistral import MistralBackend
    from chartreux.utils.http import ChartreuxAsyncHTTPClient
    from tests.core.llm.test_retry_visibility import _patch_client_transport

    requests = []
    records: list[UsageRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, json={"error": {"message": "invalid request"}})

    transport = httpx.MockTransport(handler)
    api_style: Literal["openai", "anthropic", "openai-responses"] = (
        "anthropic"
        if kind == "anthropic"
        else "openai-responses"
        if kind == "openai-responses"
        else "openai"
    )
    provider = ProviderConfig(
        name="provider", api_base="https://provider.invalid/v1", api_style=api_style
    )
    if kind == "mistral":
        _patch_client_transport(monkeypatch, transport)
        backend = MistralBackend(provider=provider, resolved_credential=None)
    else:
        backend = GenericBackend(
            provider=provider,
            resolved_credential=None,
            client=ChartreuxAsyncHTTPClient(transport=transport),
        )
    try:
        with pytest.raises(RuntimeError):
            if streaming:
                _ = [
                    part
                    async for part in LLMGateway().chat_streaming(
                        inputs(), resources(backend, records), transcript=lambda _: None
                    )
                ]
            else:
                await LLMGateway().complete(inputs(), resources(backend, records))
    finally:
        if kind == "mistral":
            await backend.__aexit__(None, None, None)
        else:
            await backend._get_client().aclose()
    assert len(requests) == len(records) == 1
    assert records[0].usage_state == UsageState.MISSING
    assert records[0].has_unknown_cost


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_mistral_local_preparation_failure_never_signals_attempt(
    monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    from chartreux.core.llm.backend.mistral import MistralBackend

    backend = MistralBackend(
        provider=ProviderConfig(
            name="provider", api_base="https://provider.invalid/v1"
        ),
        resolved_credential=None,
    )

    def fail_preparation(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("preparation")

    monkeypatch.setattr(backend._mapper, "prepare_message", fail_preparation)
    attempts = []
    token = request_start_observer.set(lambda: attempts.append(True))
    kwargs: dict[str, Any] = dict(
        model=inputs().model,
        messages=inputs().messages,
        temperature=0.2,
        tools=None,
        max_tokens=None,
        tool_choice=None,
        extra_headers=None,
    )
    try:
        with pytest.raises(ValueError, match="preparation"):
            if streaming:
                _ = [part async for part in backend.complete_streaming(**kwargs)]
            else:
                await backend.complete(**kwargs)
    finally:
        request_start_observer.reset(token)
        await backend.__aexit__(None, None, None)
    assert not attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "failure", "timeout", "cancel"])
@pytest.mark.parametrize(
    "usage", [None, LLMUsage.from_reported(completion_tokens=5), USAGE]
)
async def test_title_accounts_attempt_with_presence_and_title_model_identity(
    monkeypatch: pytest.MonkeyPatch, scenario: str, usage: LLMUsage | None
) -> None:
    from chartreux.core.llm import utility_completion
    from chartreux.core.session.title_policy import TitlePolicy
    from tests.conftest import build_test_vibe_config

    config = build_test_vibe_config()
    model = config.get_active_model()
    model.input_price = 1
    model.input_price_known = True
    model.alias = "title-model"
    model.name = "title-wire"
    provider = config.get_active_provider()
    monkeypatch.setattr(
        utility_completion, "select_utility_model", lambda _: (model, provider)
    )
    records: list[UsageRecord] = []
    backend = AttemptBackend()
    entered = asyncio.Event()

    async def complete(**_kwargs: Any) -> LLMChunk:
        notify_request_started()
        entered.set()
        model.input_price = 99
        if scenario in {"timeout", "cancel"}:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                error.usage = usage  # type: ignore[attr-defined]
                raise
        if scenario == "failure":
            error = RuntimeError("title failed")
            error.usage = usage  # type: ignore[attr-defined]
            raise error
        return chunk(usage=usage)

    backend.complete = complete
    monkeypatch.setattr(utility_completion, "create_backend", lambda **_: backend)
    task = asyncio.create_task(
        generate_session_title(
            [LLMMessage(role=Role.user, content="title this")],
            config=config,
            policy=TitlePolicy(
                total_timeout_seconds=0.03 if scenario == "timeout" else 5
            ),
            accounting_sink=resources(backend, records).accounting_sink,
            usage_attribution=ATTRIBUTION,
        )
    )
    await entered.wait()
    if scenario == "cancel":
        task.cancel()
    if scenario == "success":
        assert await task == "answer"
    else:
        expected = {
            "failure": RuntimeError,
            "timeout": TimeoutError,
            "cancel": asyncio.CancelledError,
        }[scenario]
        with pytest.raises(expected):
            await task
    assert len(records) == 1
    record = records[0]
    assert record.purpose == UsagePurpose.TITLE
    assert (record.model, record.provider, record.wire_name) == (
        model.alias,
        config.get_active_provider().name,
        model.name,
    )
    assert record.prices_usd_per_million.input == 1
    assert record.session_id == ATTRIBUTION.session_id
    assert record.outcome == (
        UsageOutcome.COMPLETED
        if scenario == "success"
        else UsageOutcome.FAILED
        if scenario == "failure"
        else UsageOutcome.INTERRUPTED
    )
    assert record.usage_state == (
        UsageState.MISSING
        if usage is None
        else UsageState.COMPLETE
        if usage is USAGE
        else UsageState.PARTIAL
    )
    assert record.output_tokens == (
        usage.completion_tokens if usage is not None else None
    )


class StreamingAttemptBackend(AttemptBackend):
    def __init__(
        self,
        chunks: list[LLMChunk],
        error: BaseException | None = None,
        *,
        block: bool = False,
    ) -> None:
        super().__init__(error=error)
        self.chunks = chunks
        self.block = block
        self.waiting = asyncio.Event()
        self.closed = False

    async def complete_streaming(self, **kwargs: Any):
        self.calls += 1
        notify_request_started()
        try:
            for part in self.chunks:
                yield part
            if self.block:
                self.waiting.set()
                await asyncio.Event().wait()
            if self.error is not None:
                raise self.error
        finally:
            self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [False, True])
async def test_completed_stream_accounts_once(refusal: bool) -> None:
    records: list[UsageRecord] = []
    backend = StreamingAttemptBackend([chunk(refusal=refusal)])
    call_resources = resources(backend, records)
    transcript = []
    stream = LLMGateway().chat_streaming(
        inputs(), call_resources, transcript=transcript.append
    )
    if refusal:
        with pytest.raises(RefusalError):
            _ = [part async for part in stream]
    else:
        assert len([part async for part in stream]) == 1
    await stream.aclose()
    assert len(records) == backend.calls == len(transcript) == 1
    assert backend.closed
    assert records[0].outcome == (
        UsageOutcome.REFUSED if refusal else UsageOutcome.COMPLETED
    )
    assert records[0].usage_state == UsageState.COMPLETE
    assert records[0].input_tokens == 100
    assert call_resources.stats.session_prompt_tokens == 100
    assert call_resources.stats.known_cost_total == records[0].known_cost_usd


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage", [None, LLMUsage(), LLMUsage.from_reported(completion_tokens=5)]
)
async def test_backend_stream_error_preserves_presence(usage: LLMUsage | None) -> None:
    records: list[UsageRecord] = []
    error = httpx.ConnectError("offline")
    backend = StreamingAttemptBackend([chunk(usage=usage)], error)
    with pytest.raises(RuntimeError) as caught:
        _ = [
            part
            async for part in LLMGateway().chat_streaming(
                inputs(), resources(backend, records), transcript=lambda _: None
            )
        ]
    assert caught.value.__cause__ is error
    assert classify_failure(caught.value) == classify_failure(error)
    assert len(records) == 1
    assert records[0].outcome == UsageOutcome.FAILED
    assert records[0].usage_state == (
        UsageState.PARTIAL
        if usage and usage.completion_tokens_reported
        else UsageState.MISSING
    )
    assert records[0].input_tokens is None
    assert records[0].output_tokens == (
        5 if usage and usage.completion_tokens_reported else None
    )
    assert records[0].has_unknown_cost


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, LLMUsage()])
async def test_stream_without_usage_is_missing(usage: LLMUsage | None) -> None:
    records: list[UsageRecord] = []
    stream = LLMGateway().chat_streaming(
        inputs(),
        resources(StreamingAttemptBackend([chunk(usage=usage)]), records),
        transcript=lambda _: None,
    )
    if usage is None:
        with pytest.raises(RuntimeError):
            _ = [part async for part in stream]
    else:
        _ = [part async for part in stream]
    assert len(records) == 1
    assert records[0].usage_state == UsageState.MISSING
    assert records[0].input_tokens is records[0].output_tokens is None
    assert records[0].cached_input_tokens is None
    assert records[0].has_unknown_cost


@pytest.mark.asyncio
async def test_stream_processing_failure_retains_raw_usage() -> None:
    records: list[UsageRecord] = []
    backend = StreamingAttemptBackend([chunk(usage=None), chunk()])
    error = ValueError("processing")
    processed = 0

    def process(message: LLMMessage) -> LLMMessage:
        nonlocal processed
        processed += 1
        if processed == 2:
            raise error
        return message

    call_resources = replace(resources(backend, records), process_message=process)
    with pytest.raises(RuntimeError) as caught:
        _ = [
            part
            async for part in LLMGateway().chat_streaming(
                inputs(), call_resources, transcript=lambda _: None
            )
        ]
    assert caught.value.__cause__ is error
    assert len(records) == 1
    assert records[0].outcome == UsageOutcome.FAILED
    assert records[0].usage_state == UsageState.COMPLETE
    assert records[0].input_tokens == 100
    # AgentStats still accounts only chunks whose processing succeeded.
    assert call_resources.stats.session_prompt_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["cancel", "close", "generator-exit"])
@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_stream_interruption_settles_once(
    exit_kind: str, repeat_cancel: bool
) -> None:
    records: list[UsageRecord] = []
    entered, release = asyncio.Event(), asyncio.Event()
    backend = StreamingAttemptBackend(
        [chunk(usage=LLMUsage.from_reported(completion_tokens=5))], block=True
    )

    async def sink(record: UsageRecord) -> None:
        entered.set()
        await release.wait()
        records.append(record)

    call_resources = replace(resources(backend, records), accounting_sink=sink)
    stream = LLMGateway().chat_streaming(
        inputs(), call_resources, transcript=lambda _: None
    )
    await anext(stream)
    assert request_start_observer.get() is None
    original = GeneratorExit("original closure")
    if exit_kind == "cancel":
        task = asyncio.create_task(anext(stream))
        await backend.waiting.wait()
        task.cancel("original cancellation")
    elif exit_kind == "close":
        task = asyncio.create_task(stream.aclose())
    else:
        task = asyncio.create_task(stream.athrow(original))
    await entered.wait()
    if repeat_cancel:
        task.cancel("later cancellation")
        await asyncio.sleep(0)
        task.cancel("repeated cancellation")
    release.set()
    if exit_kind == "cancel":
        with pytest.raises(asyncio.CancelledError, match="original cancellation"):
            await task
    elif exit_kind == "generator-exit":
        with pytest.raises(GeneratorExit) as caught:
            await task
        assert caught.value is original
    else:
        await task
    await stream.aclose()
    assert backend.closed
    assert len(records) == backend.calls == 1
    assert records[0].outcome == UsageOutcome.INTERRUPTED
    assert records[0].usage_state == UsageState.PARTIAL
    assert records[0].input_tokens is None
    assert records[0].output_tokens == 5
    assert records[0].has_unknown_cost


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["cancel", "close", "complete", "final-close"])
async def test_anthropic_preliminary_usage_is_not_complete(exit_kind: str) -> None:
    from chartreux.core.llm.backend.anthropic import AnthropicAdapter

    adapter = AnthropicAdapter()
    start = adapter.parse_response({
        "type": "message_start",
        "message": {
            "usage": {
                "input_tokens": 90,
                "output_tokens": 1,
                "cache_read_input_tokens": 10,
            }
        },
    })
    final = adapter.parse_response({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 20},
    })
    chunks = [start, chunk(usage=None)]
    if exit_kind in {"complete", "final-close"}:
        chunks.append(final)
    backend = StreamingAttemptBackend(chunks, block=exit_kind == "cancel")
    records: list[UsageRecord] = []
    stream = LLMGateway().chat_streaming(
        inputs(), resources(backend, records), transcript=lambda _: None
    )
    if exit_kind == "complete":
        _ = [part async for part in stream]
    else:
        for _ in chunks:
            await anext(stream)
        if exit_kind == "cancel":
            task = asyncio.create_task(anext(stream))
            await backend.waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await stream.aclose()
    assert len(records) == 1
    assert records[0].usage_state == (
        UsageState.COMPLETE
        if exit_kind in {"complete", "final-close"}
        else UsageState.PARTIAL
    )
    assert records[0].input_tokens == 100
    assert records[0].cached_input_tokens == 10
    is_complete = exit_kind in {"complete", "final-close"}
    assert records[0].output_tokens == (20 if is_complete else 1)
    assert records[0].known_cost_usd == pytest.approx(
        0.000135 if is_complete else 0.000097
    )
    assert records[0].has_unknown_cost is (not is_complete)
    totals = aggregate_usage(
        records, as_of=records[0].occurred_at, timezone=UTC
    ).selected
    assert totals.known_cost_usd == records[0].known_cost_usd
    # Known cost plus an unknown remainder is the aggregate's '+' marker contract.
    assert totals.has_known_cost
    assert totals.has_unknown_cost is (not is_complete)
    assert records[0].outcome == (
        UsageOutcome.COMPLETED if exit_kind == "complete" else UsageOutcome.INTERRUPTED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_failed_responses_preserve_usage_and_known_cost(streaming: bool) -> None:
    from chartreux.core.llm.backend.openai_responses import (
        OpenAIResponsesAdapter,
        OpenAIResponsesStreamError,
    )

    adapter = OpenAIResponsesAdapter()
    provider = ProviderConfig(name="provider", api_base="https://provider.invalid")
    response = {
        "object": "response",
        "status": "failed",
        "output": [],
        "error": {"code": "server_error", "message": "failed"},
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 10},
        },
    }
    with pytest.raises(OpenAIResponsesStreamError) as parsed:
        adapter.parse_response(
            {"type": "response.failed", "response": response}
            if streaming
            else response,
            provider,
        )
    records: list[UsageRecord] = []
    with pytest.raises(RuntimeError):
        if streaming:
            _ = [
                part
                async for part in LLMGateway().chat_streaming(
                    inputs(),
                    resources(StreamingAttemptBackend([], parsed.value), records),
                    transcript=lambda _: None,
                )
            ]
        else:
            await LLMGateway().complete(
                inputs(), resources(AttemptBackend(error=parsed.value), records)
            )
    assert len(records) == 1
    assert records[0].outcome == UsageOutcome.FAILED
    assert records[0].usage_state == UsageState.PARTIAL
    assert (records[0].input_tokens, records[0].output_tokens) == (100, 20)
    assert records[0].cached_input_tokens == 10
    assert records[0].known_cost_usd == pytest.approx(0.000135)
    assert records[0].has_unknown_cost
    totals = aggregate_usage(
        records, as_of=records[0].occurred_at, timezone=UTC
    ).selected
    assert totals.known_cost_usd == records[0].known_cost_usd
    assert totals.has_known_cost and totals.has_unknown_cost


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_task_creation_failure_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, streaming: bool
) -> None:
    def fail_create_task(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("event loop teardown")

    monkeypatch.setattr(asyncio, "create_task", fail_create_task)
    error = httpx.ConnectError("original failure")
    records: list[UsageRecord] = []
    with pytest.raises(RuntimeError) as caught:
        if streaming:
            _ = [
                part
                async for part in LLMGateway().chat_streaming(
                    inputs(),
                    resources(StreamingAttemptBackend([], error), records),
                    transcript=lambda _: None,
                )
            ]
        else:
            await LLMGateway().complete(
                inputs(), resources(AttemptBackend(error=error), records)
            )
    assert caught.value.__cause__ is error
    assert not records
    assert "coverage is degraded" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_error", [False, True])
async def test_stream_sink_failure_never_changes_inference(backend_error: bool) -> None:
    records: list[UsageRecord] = []
    error = httpx.ConnectError("offline") if backend_error else None
    backend = StreamingAttemptBackend([chunk()], error)

    async def sink(_record: UsageRecord) -> None:
        raise OSError("accounting unavailable")

    stream = LLMGateway().chat_streaming(
        inputs(),
        replace(resources(backend, records), accounting_sink=sink),
        transcript=lambda _: None,
    )
    if error is not None:
        with pytest.raises(RuntimeError) as caught:
            _ = [part async for part in stream]
        assert caught.value.__cause__ is error
        assert classify_failure(caught.value) == classify_failure(error)
    else:
        assert len([part async for part in stream]) == 1
    assert backend.calls == 1
