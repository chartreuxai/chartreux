"""Attempt-ordinal axis on the usage ledger (schema v2, no bump).

Pinned B1b contract: every finalized attempt record carries its 1-based ordinal
within the logical call, a replayed attempt is distinguishable as ordinal 2, the
attempt records of one call sum to the call's settled accounting, and
pre-existing v2 records without the field parse with None.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import pytest

from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
)
from chartreux.core.config import ModelConfig
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm.utility_completion import run_utility_completion
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role, StopInfo
from chartreux.core.session_types import AgentStats
from chartreux.core.usage import (
    UsageAttribution,
    UsageOutcome,
    UsagePrices,
    UsagePurpose,
    UsageRecord,
    UsageState,
)
from tests.conftest import build_test_vibe_config
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


def chunk() -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.assistant, content="answer"),
        usage=LLMUsage(prompt_tokens=100, completion_tokens=20, cached_tokens=10),
        stop=StopInfo(reason="stop"),
    )


def empty_chunk() -> LLMChunk:
    return chunk().model_copy(
        update={"message": LLMMessage(role=Role.assistant, content=" \n")}
    )


class ReplayBackend(FakeBackend):
    """One queued terminal response per transport attempt."""

    async def complete(self, **kwargs: Any) -> LLMChunk:
        notify_request_started()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        notify_request_started()
        async for part in super().complete_streaming(**kwargs):
            yield part


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


async def replay_records(*, streaming: bool) -> tuple[list[UsageRecord], AgentStats]:
    """One empty response followed by a good one: exactly two finalizations."""
    records: list[UsageRecord] = []
    call_resources = resources(ReplayBackend([[empty_chunk()], [chunk()]]), records)
    if streaming:
        _ = [
            part
            async for part in LLMGateway().chat_streaming(
                inputs(), call_resources, transcript=lambda _: None
            )
        ]
    else:
        await LLMGateway().complete(inputs(), call_resources)
    return records, call_resources.stats


@pytest.mark.asyncio
async def test_attempt_ordinal_written_per_attempt() -> None:
    for streaming in (False, True):
        records, _stats = await replay_records(streaming=streaming)
        assert [record.attempt_ordinal for record in records] == [1, 2], streaming
        assert records[0].record_id != records[1].record_id, streaming


@pytest.mark.asyncio
async def test_single_attempt_ordinal_is_one() -> None:
    for streaming in (False, True):
        records: list[UsageRecord] = []
        call_resources = resources(ReplayBackend([[chunk()]]), records)
        if streaming:
            _ = [
                part
                async for part in LLMGateway().chat_streaming(
                    inputs(), call_resources, transcript=lambda _: None
                )
            ]
        else:
            await LLMGateway().complete(inputs(), call_resources)
        assert len(records) == 1, streaming
        assert records[0].attempt_ordinal == 1, streaming


@pytest.mark.asyncio
async def test_attempt_records_reconcile_to_call_totals() -> None:
    records, stats = await replay_records(streaming=False)
    assert [record.attempt_ordinal for record in records] == [1, 2]
    assert (
        sum(record.input_tokens or 0 for record in records)
        == stats.session_prompt_tokens
    )
    assert (
        sum(record.output_tokens or 0 for record in records)
        == stats.session_completion_tokens
    )
    assert (
        sum(record.cached_input_tokens or 0 for record in records)
        == stats.session_cached_tokens
    )
    assert stats.known_cost_total == pytest.approx(
        sum(record.known_cost_usd for record in records)
    )


def test_records_without_ordinal_parse_none() -> None:
    record = UsageRecord(
        record_id="record",
        occurred_at=datetime(2026, 3, 1, tzinfo=UTC),
        root_session_id="root",
        session_id="child",
        parent_session_id="root",
        agent_role="subagent",
        agent_profile="coder",
        model="model",
        provider="provider",
        wire_name="wire",
        project_key="project",
        outcome=UsageOutcome.COMPLETED,
        usage_state=UsageState.COMPLETE,
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=10,
        prices_usd_per_million=UsagePrices(input=1, output=2, cached_input=0.5),
        known_cost_usd=0.00016,
        has_unknown_cost=False,
    )
    written = record.model_dump(mode="json")
    del written["attempt_ordinal"]
    parsed = UsageRecord.model_validate(written)
    assert parsed.schema_version == 2
    assert parsed.attempt_ordinal is None


@pytest.mark.asyncio
async def test_utility_attempt_ordinal_written(monkeypatch: pytest.MonkeyPatch) -> None:
    from chartreux.core.llm import utility_completion

    records: list[UsageRecord] = []

    async def sink(record: UsageRecord) -> None:
        records.append(record)

    class UtilityBackend(FakeBackend):
        async def complete(self, **kwargs: Any) -> LLMChunk:
            notify_request_started()
            return chunk()

    def fake_create_backend(**_kwargs: Any) -> UtilityBackend:
        return UtilityBackend()

    monkeypatch.setattr(utility_completion, "create_backend", fake_create_backend)
    content = await run_utility_completion(
        config=build_test_vibe_config(),
        system_prompt="system",
        user_content="utility ask",
        max_tokens=16,
        request_timeout_seconds=5.0,
        retry_budget_seconds=5.0,
        accounting_sink=sink,
        usage_attribution=ATTRIBUTION,
        purpose=UsagePurpose.TITLE,
    )

    assert content == "answer"
    assert len(records) == 1
    assert records[0].attempt_ordinal == 1
    assert records[0].purpose == UsagePurpose.TITLE
