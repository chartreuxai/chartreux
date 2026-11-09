"""P3 chunk 3b: request-shape telemetry captured at the request-build sites.

The settled contract under test: the coordinator's real build site measures the
seven telemetry fields once per build and carries them on the frozen
CompletionInputs; every gateway attempt of that logical call — retries
included — records them on its own ledger row through the same settlement that
records its usage; a KEEPALIVE-purpose row is typed like any other purpose; an
attempt with unknown usage stays a row; and the gateway's existing tolerance
for a missing sink or attribution is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from chartreux.core.agent_loop._completion_coordinator import CompletionCoordinator
from chartreux.core.agent_loop.llm_gateway import (
    CallFinalizer,
    CallResources,
    CompletionInputs,
    LLMGateway,
    request_concurrency_snapshot,
)
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm_models import (
    AvailableFunction,
    AvailableTool,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StopInfo,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.usage import (
    UsageAttribution,
    UsageOutcome,
    UsagePurpose,
    UsageRecord,
    UsageState,
)
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend

SYSTEM_PROMPT = "You are a focused helper."
TELEMETRY_FIELDS = (
    "system_prompt_bytes",
    "tool_def_bytes",
    "prefix_stability_hash",
    "gap_since_previous_call_ms",
    "concurrency_snapshot",
    "toolset_prompt_fingerprint",
    "declared_max_tokens",
)
USAGE = LLMUsage(prompt_tokens=100, completion_tokens=20, cached_tokens=10)
ATTRIBUTION = UsageAttribution(
    root_session_id="root",
    session_id="session",
    agent_role="root",
    model="stale-model",
    provider="stale-provider",
    wire_name="stale-wire",
    project_key="project",
    content_class="implementation",
    run_id="run-1",
)


def _model() -> ModelConfig:
    return ModelConfig(
        name="wire",
        provider="test",
        alias="alias-x",
        supports_images=True,
        input_price=1.0,
        output_price=2.0,
        cached_input_price=0.5,
    )


def _tool(name: str = "read", description: str = "Read a file.") -> AvailableTool:
    return AvailableTool(
        function=AvailableFunction(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    )


def _messages() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content=SYSTEM_PROMPT),
        LLMMessage(role=Role.user, content="go"),
    ]


def _coordinator(max_tokens: int | None = 8192) -> CompletionCoordinator:
    """A coordinator whose input build runs the real production capture."""
    config = build_test_vibe_config(
        providers=[ProviderConfig(name="test", api_base="https://test.invalid")]
    )
    coordinator = CompletionCoordinator.__new__(CompletionCoordinator)
    coordinator._config_getter = lambda: config
    coordinator._session_id_getter = lambda: "session-1"
    coordinator._backend_metadata = lambda call_type: {}
    coordinator._max_tokens_getter = lambda: max_tokens
    coordinator._last_request_envelope = None
    return coordinator


def _build(
    coordinator: CompletionCoordinator,
    *,
    tools: list[AvailableTool] | None = None,
    purpose: UsagePurpose = UsagePurpose.CONVERSATION,
) -> CompletionInputs:
    return coordinator.build_completion_inputs(
        model=_model(),
        messages=_messages(),
        tools=tools,
        tool_choice=None,
        purpose=purpose,
    )


def _chunk(*, usage: LLMUsage | None = USAGE) -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.assistant, content="answer"),
        usage=usage,
        stop=StopInfo(reason="stop"),
    )


class _AttemptBackend(FakeBackend):
    """A backend that reports its transport start, as the real one does."""

    def __init__(self, result: LLMChunk | None = None) -> None:
        super().__init__()
        self._result = result if result is not None else _chunk()

    async def complete(self, **_kwargs: Any) -> LLMChunk:
        notify_request_started()
        return self._result


class _ReplayBackend(FakeBackend):
    """One empty attempt, then a real answer (the bounded in-call replay)."""

    def __init__(self, first: LLMChunk, second: LLMChunk) -> None:
        super().__init__([[first], [second]])

    async def complete(self, **kwargs: Any) -> LLMChunk:
        notify_request_started()
        return await super().complete(**kwargs)


class _FailingBackend(FakeBackend):
    """A transport failure: the attempt reports no usage at all."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self._error = error

    async def complete(self, **_kwargs: Any) -> LLMChunk:
        notify_request_started()
        raise self._error


def _resources(
    backend: FakeBackend,
    records: list[UsageRecord],
    *,
    attribution: UsageAttribution | None = ATTRIBUTION,
) -> CallResources:
    async def sink(record: UsageRecord) -> None:
        records.append(record)

    return CallResources(
        backend=backend,
        stats=AgentStats(),
        process_message=lambda message: message,
        accounting_sink=sink,
        usage_attribution=attribution,
    )


def _expected_tool_json(tools: list[AvailableTool]) -> str:
    return json.dumps(
        [tool.model_dump(mode="json") for tool in tools],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_the_real_build_site_measures_every_telemetry_field() -> None:
    tools = [_tool()]
    inputs = _build(_coordinator(max_tokens=4096), tools=tools)

    shape = inputs.request_shape
    assert shape is not None
    assert inputs.max_tokens == 4096
    assert shape.system_prompt_bytes == len(SYSTEM_PROMPT.encode())
    assert shape.tool_def_bytes == len(_expected_tool_json(tools).encode())
    assert shape.tool_def_bytes > 0
    assert len(shape.prefix_stability_hash) == 16
    assert int(shape.prefix_stability_hash, 16) >= 0
    assert len(shape.toolset_prompt_fingerprint) == 16
    assert int(shape.toolset_prompt_fingerprint, 16) >= 0
    assert shape.gap_since_previous_call_ms is None
    assert shape.concurrency_snapshot == request_concurrency_snapshot()
    assert shape.declared_max_tokens == 4096


def test_an_undeclared_output_limit_stays_unset() -> None:
    shape = _build(_coordinator(max_tokens=None), tools=[_tool()]).request_shape

    assert shape is not None
    assert shape.declared_max_tokens is None
    assert "declared_max_tokens" not in shape.attribution_fields()


def test_each_build_reports_the_gap_since_its_previous_build() -> None:
    coordinator = _coordinator()
    first = _build(coordinator, tools=[_tool()]).request_shape
    time.sleep(0.02)
    second = _build(coordinator, tools=[_tool()]).request_shape

    assert first is not None and second is not None
    assert first.gap_since_previous_call_ms is None
    assert second.gap_since_previous_call_ms is not None
    assert second.gap_since_previous_call_ms >= 15


def test_the_prefix_hash_tracks_bytes_while_the_fingerprint_tracks_shape() -> None:
    coordinator = _coordinator()
    baseline = _build(coordinator, tools=[_tool()]).request_shape
    repeated = _build(coordinator, tools=[_tool()]).request_shape
    reworded = _build(
        coordinator, tools=[_tool(description="Read a file, verbatim.")]
    ).request_shape
    reshaped = _build(
        coordinator,
        tools=[
            AvailableTool(
                function=AvailableFunction(
                    name="read",
                    description="Read a file.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "offset": {"type": "integer"},
                        },
                    },
                )
            )
        ],
    ).request_shape

    assert baseline is not None and repeated is not None
    assert reworded is not None and reshaped is not None
    assert baseline.prefix_stability_hash == repeated.prefix_stability_hash
    assert baseline.toolset_prompt_fingerprint == repeated.toolset_prompt_fingerprint
    assert reworded.prefix_stability_hash != baseline.prefix_stability_hash
    assert reworded.toolset_prompt_fingerprint == baseline.toolset_prompt_fingerprint
    assert reshaped.prefix_stability_hash != baseline.prefix_stability_hash
    assert reshaped.toolset_prompt_fingerprint != baseline.toolset_prompt_fingerprint


@pytest.mark.asyncio
async def test_every_telemetry_field_lands_on_the_attempt_row() -> None:
    records: list[UsageRecord] = []
    inputs = _build(_coordinator(max_tokens=1234), tools=[_tool()])
    shape = inputs.request_shape
    assert shape is not None

    await LLMGateway().complete(inputs, _resources(_AttemptBackend(), records))

    assert len(records) == 1
    record = records[0]
    assert record.purpose == UsagePurpose.CONVERSATION
    assert record.usage_state == UsageState.COMPLETE
    for field, expected in shape.attribution_fields().items():
        assert getattr(record, field) == expected
    assert record.declared_max_tokens == 1234
    assert record.attempt_ordinal == 1


@pytest.mark.asyncio
async def test_each_retry_attempt_is_recorded_distinctly_with_its_telemetry() -> None:
    records: list[UsageRecord] = []
    empty = _chunk().model_copy(
        update={"message": LLMMessage(role=Role.assistant, content=" \n")}
    )
    inputs = _build(_coordinator(max_tokens=512), tools=[_tool()])
    shape = inputs.request_shape
    assert shape is not None

    await LLMGateway().complete(
        inputs, _resources(_ReplayBackend(empty, _chunk()), records)
    )

    assert len(records) == 2
    assert [record.attempt_ordinal for record in records] == [1, 2]
    assert records[0].record_id != records[1].record_id
    for record in records:
        assert record.system_prompt_bytes == shape.system_prompt_bytes
        assert record.tool_def_bytes == shape.tool_def_bytes
        assert record.prefix_stability_hash == shape.prefix_stability_hash
        assert record.concurrency_snapshot == shape.concurrency_snapshot
        assert record.toolset_prompt_fingerprint == shape.toolset_prompt_fingerprint
        assert record.declared_max_tokens == 512


@pytest.mark.asyncio
async def test_keepalive_purpose_routes_through_the_same_commit_boundary() -> None:
    records: list[UsageRecord] = []
    inputs = _build(
        _coordinator(max_tokens=256), tools=[_tool()], purpose=UsagePurpose.KEEPALIVE
    )
    shape = inputs.request_shape
    assert shape is not None

    await LLMGateway().complete(inputs, _resources(_AttemptBackend(), records))

    assert len(records) == 1
    record = records[0]
    assert record.purpose is UsagePurpose.KEEPALIVE
    assert json.loads(record.model_dump_json())["purpose"] == "keepalive"
    # A utility purpose carries no work-class label and no run identity.
    assert record.content_class is None
    assert record.run_id is None
    assert record.system_prompt_bytes == shape.system_prompt_bytes
    assert record.declared_max_tokens == 256


@pytest.mark.asyncio
async def test_an_unknown_usage_attempt_is_retained_as_a_row() -> None:
    records: list[UsageRecord] = []
    inputs = _build(_coordinator(max_tokens=2048), tools=[_tool()])
    shape = inputs.request_shape
    assert shape is not None

    with pytest.raises(RuntimeError):
        await LLMGateway().complete(
            inputs, _resources(_FailingBackend(httpx.ConnectError("offline")), records)
        )

    assert len(records) == 1
    record = records[0]
    assert record.outcome == UsageOutcome.FAILED
    assert record.usage_state == UsageState.MISSING
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.cached_input_tokens is None
    assert record.has_unknown_cost is True
    # The sensitivity appendix prices an unknown call by its declared limit.
    assert record.declared_max_tokens == 2048
    assert record.system_prompt_bytes == shape.system_prompt_bytes
    assert record.tool_def_bytes == shape.tool_def_bytes


@pytest.mark.asyncio
async def test_a_partially_reported_attempt_keeps_its_partial_row() -> None:
    records: list[UsageRecord] = []
    partial = LLMUsage.from_reported(prompt_tokens=100, cached_tokens=10)
    inputs = _build(_coordinator(max_tokens=2048), tools=[_tool()])

    await LLMGateway().complete(
        inputs, _resources(_AttemptBackend(_chunk(usage=partial)), records)
    )

    assert len(records) == 1
    record = records[0]
    assert record.usage_state == UsageState.PARTIAL
    assert record.input_tokens == 100
    assert record.output_tokens is None
    assert record.has_unknown_cost is True
    assert record.declared_max_tokens == 2048


@pytest.mark.asyncio
async def test_the_missing_sink_and_attribution_tolerance_is_preserved() -> None:
    records: list[UsageRecord] = []
    inputs = _build(_coordinator(), tools=[_tool()])
    no_sink = CallResources(
        backend=_AttemptBackend(),
        stats=AgentStats(),
        process_message=lambda message: message,
    )
    no_attribution = _resources(_AttemptBackend(), records, attribution=None)

    for resources in (no_sink, no_attribution):
        finalizer = CallFinalizer(inputs, resources)
        finalizer.start()
        finalizer.usage = USAGE
        finalizer.outcome = UsageOutcome.COMPLETED
        await finalizer.finalize()
        await finalizer.finalize()

    assert records == []
    assert request_concurrency_snapshot() >= 0


@pytest.mark.asyncio
async def test_the_concurrency_snapshot_counts_requests_in_flight() -> None:
    baseline = request_concurrency_snapshot()
    finalizer = CallFinalizer(
        _build(_coordinator(), tools=[_tool()]),
        CallResources(
            backend=_AttemptBackend(),
            stats=AgentStats(),
            process_message=lambda message: message,
        ),
    )
    finalizer.start()
    # A backend's per-wire retries may re-fire the observer: the count is
    # per in-flight request, never per notification.
    finalizer.start()
    try:
        assert request_concurrency_snapshot() == baseline + 1
        shape = _build(_coordinator(), tools=[_tool()]).request_shape
        assert shape is not None
        assert shape.concurrency_snapshot == baseline + 1
    finally:
        await finalizer.finalize()
    assert request_concurrency_snapshot() == baseline


@pytest.mark.asyncio
async def test_an_unmeasured_build_leaves_the_row_telemetry_free() -> None:
    records: list[UsageRecord] = []
    inputs = CompletionInputs(
        model=_model(),
        provider_name="test",
        emits_finish_reason=True,
        messages=(LLMMessage(role=Role.user, content="go"),),
        tools=None,
        tool_choice=None,
        extra_headers={},
        metadata={},
        max_tokens=None,
    )
    assert inputs.request_shape is None

    await LLMGateway().complete(inputs, _resources(_AttemptBackend(), records))

    assert len(records) == 1
    payload = json.loads(records[0].model_dump_json())
    assert not set(TELEMETRY_FIELDS) & set(payload)
    assert all(getattr(records[0], field) is None for field in TELEMETRY_FIELDS)


@pytest.mark.asyncio
async def test_the_in_flight_count_survives_a_cancelled_settlement() -> None:
    baseline = request_concurrency_snapshot()
    entered, release = asyncio.Event(), asyncio.Event()

    async def sink(_record: UsageRecord) -> None:
        entered.set()
        await release.wait()

    finalizer = CallFinalizer(
        _build(_coordinator(), tools=[_tool()]),
        CallResources(
            backend=_AttemptBackend(),
            stats=AgentStats(),
            process_message=lambda message: message,
            accounting_sink=sink,
            usage_attribution=ATTRIBUTION,
        ),
    )
    finalizer.start()
    finalizer.usage = USAGE
    finalizer.outcome = UsageOutcome.COMPLETED
    task = asyncio.create_task(finalizer.finalize())
    await entered.wait()
    assert request_concurrency_snapshot() == baseline + 1
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert request_concurrency_snapshot() == baseline
