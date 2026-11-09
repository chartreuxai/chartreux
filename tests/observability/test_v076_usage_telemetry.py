"""P3 chunk 3c: the closeout integration slice for usage telemetry.

The settled contract under test: a keepalive ping driven through the production
build site and the production gateway boundary settles every attempt into a
real on-disk usage ledger carrying the full request-shape telemetry, and a cold
ledger reader rebuilds that per-attempt history end to end — attempt ordinals,
outcomes, distinct record identities and measurements included.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from chartreux.core.agent_loop._completion_coordinator import CompletionCoordinator
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    LLMGateway,
    RequestShapeTelemetry,
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
    AsyncUsageWriter,
    SnapshotState,
    UsageAttribution,
    UsageOutcome,
    UsagePurpose,
    UsageReader,
    UsageRecord,
    UsageState,
)
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend

SYSTEM_PROMPT = "You are a focused helper."
PING_MAX_TOKENS = 512
USAGE = LLMUsage(prompt_tokens=100, completion_tokens=20, cached_tokens=10)
TELEMETRY_FIELDS = (
    "system_prompt_bytes",
    "tool_def_bytes",
    "prefix_stability_hash",
    "gap_since_previous_call_ms",
    "concurrency_snapshot",
    "toolset_prompt_fingerprint",
    "declared_max_tokens",
)
ATTRIBUTION = UsageAttribution(
    root_session_id="root",
    session_id="session",
    agent_role="root",
    model="stale-model",
    provider="stale-provider",
    wire_name="stale-wire",
    project_key="project",
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


def _tool() -> AvailableTool:
    return AvailableTool(
        function=AvailableFunction(
            name="read",
            description="Read a file.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    )


def _messages() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content=SYSTEM_PROMPT),
        LLMMessage(role=Role.user, content="go"),
    ]


def _coordinator(max_tokens: Any) -> CompletionCoordinator:
    """A coordinator whose input build runs the real production capture."""
    config = build_test_vibe_config(
        providers=[ProviderConfig(name="test", api_base="https://test.invalid")]
    )
    coordinator = CompletionCoordinator.__new__(CompletionCoordinator)
    coordinator._config_getter = lambda: config
    coordinator._session_id_getter = lambda: "session-1"
    coordinator._backend_metadata = lambda call_type: {}
    coordinator._max_tokens_getter = max_tokens
    coordinator._last_request_envelope = None
    return coordinator


@dataclass(frozen=True)
class ScriptedAttempt:
    """One scripted attempt of a ping: the chunk the backend returns or raises."""

    result: LLMChunk | BaseException


def answer(content: str = "pong", *, usage: LLMUsage | None = USAGE) -> ScriptedAttempt:
    return ScriptedAttempt(
        LLMChunk(
            message=LLMMessage(role=Role.assistant, content=content),
            usage=usage,
            stop=StopInfo(reason="stop"),
        )
    )


def empty_answer() -> ScriptedAttempt:
    return answer(" \n")


def offline() -> ScriptedAttempt:
    return ScriptedAttempt(httpx.ConnectError("offline"))


class _ScriptedBackend(FakeBackend):
    """A backend that plays one attempt script, reporting each transport start."""

    def __init__(self, attempts: Sequence[ScriptedAttempt]) -> None:
        super().__init__()
        self._attempts = list(attempts)

    async def complete(self, **_kwargs: Any) -> LLMChunk:
        notify_request_started()
        result = self._attempts.pop(0).result
        if isinstance(result, BaseException):
            raise result
        return result


class ScriptedKeepaliveLoop:
    """A scripted ping loop over the production build, gateway, and ledger.

    A ping is one logical keepalive call: the real build site measures the
    request shape, the gateway settles one row per attempt through the real
    async writer, and each row is appended to a real on-disk ledger.
    """

    def __init__(self, usage_dir: Path) -> None:
        self.usage_dir = usage_dir
        self.shapes: list[RequestShapeTelemetry] = []
        self._max_tokens: int | None = PING_MAX_TOKENS
        self._writer = AsyncUsageWriter(usage_dir)
        self._coordinator = _coordinator(lambda: self._max_tokens)

    async def ping(
        self,
        attempts: Sequence[ScriptedAttempt],
        *,
        max_tokens: int | None = PING_MAX_TOKENS,
    ) -> RequestShapeTelemetry:
        self._max_tokens = max_tokens
        inputs = self._coordinator.build_completion_inputs(
            model=_model(),
            messages=_messages(),
            tools=[_tool()],
            tool_choice=None,
            purpose=UsagePurpose.KEEPALIVE,
        )
        shape = inputs.request_shape
        assert shape is not None
        self.shapes.append(shape)
        await LLMGateway().complete(
            inputs,
            CallResources(
                backend=_ScriptedBackend(attempts),
                stats=AgentStats(),
                process_message=lambda message: message,
                accounting_sink=self._writer,
                usage_attribution=ATTRIBUTION,
            ),
        )
        return shape

    def rows(self) -> list[dict[str, Any]]:
        """The ledger rows exactly as appended, in append order."""
        path = self.usage_dir / ATTRIBUTION.root_session_id / "usage.jsonl"
        return [json.loads(line) for line in path.read_bytes().splitlines()]

    def history(self) -> tuple[UsageRecord, ...]:
        """Rebuild the per-attempt history with a cold reader over the files."""
        snapshot = UsageReader(self.usage_dir).reconcile()
        assert snapshot.state is SnapshotState.READY
        assert snapshot.warnings == ()
        return snapshot.records

    async def aclose(self) -> None:
        await self._writer.aclose()


@pytest_asyncio.fixture
async def scripted_loop(tmp_path: Path) -> AsyncIterator[ScriptedKeepaliveLoop]:
    loop = ScriptedKeepaliveLoop(tmp_path / "usage")
    try:
        yield loop
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_a_keepalive_ping_lands_in_the_ledger_with_all_telemetry_fields(
    scripted_loop: ScriptedKeepaliveLoop,
) -> None:
    first = await scripted_loop.ping([answer()])
    second = await scripted_loop.ping([answer()])

    rows = scripted_loop.rows()
    assert len(rows) == 2
    assert [row["purpose"] for row in rows] == ["keepalive", "keepalive"]
    assert [row["attempt_ordinal"] for row in rows] == [1, 1]
    # The first build of a loop has no idle gap to report, so that field is
    # omitted; the second build measures it and the row carries all seven.
    assert first.gap_since_previous_call_ms is None
    assert "gap_since_previous_call_ms" not in rows[0]
    assert second.gap_since_previous_call_ms is not None
    latest = rows[1]
    assert set(TELEMETRY_FIELDS) <= set(latest)
    for field in TELEMETRY_FIELDS:
        assert latest[field] == getattr(second, field)
    assert latest["schema_version"] == 2
    assert latest["declared_max_tokens"] == PING_MAX_TOKENS
    assert latest["system_prompt_bytes"] == len(SYSTEM_PROMPT.encode())
    assert latest["tool_def_bytes"] > 0

    records = scripted_loop.history()
    assert len(records) == 2
    assert all(record.purpose is UsagePurpose.KEEPALIVE for record in records)
    restored = records[1]
    for field in TELEMETRY_FIELDS:
        assert getattr(restored, field) == getattr(second, field)


@pytest.mark.asyncio
async def test_the_reader_reconstructs_the_per_attempt_history_end_to_end(
    scripted_loop: ScriptedKeepaliveLoop,
) -> None:
    await scripted_loop.ping([answer()])
    await scripted_loop.ping([empty_answer(), answer()])
    with pytest.raises(RuntimeError):
        await scripted_loop.ping([offline()])

    single, replayed, failed = scripted_loop.shapes
    records = scripted_loop.history()

    assert [record.attempt_ordinal for record in records] == [1, 1, 2, 1]
    assert [record.outcome for record in records] == [
        UsageOutcome.COMPLETED,
        UsageOutcome.FAILED,
        UsageOutcome.COMPLETED,
        UsageOutcome.FAILED,
    ]
    assert len({record.record_id for record in records}) == len(records)
    assert all(record.purpose is UsagePurpose.KEEPALIVE for record in records)
    for record, shape in zip(
        records, [single, replayed, replayed, failed], strict=True
    ):
        for field in TELEMETRY_FIELDS:
            assert getattr(record, field) == getattr(shape, field)
    # The replayed empty attempt keeps the usage it reported; the offline
    # attempt reports nothing and stays an un-fabricated row.
    assert records[1].usage_state is UsageState.COMPLETE
    assert records[1].input_tokens == 100
    assert records[3].usage_state is UsageState.MISSING
    assert records[3].input_tokens is None
    assert records[3].output_tokens is None
    assert records[3].cached_input_tokens is None


@pytest.mark.asyncio
async def test_an_undeclared_ping_limit_stays_unset_on_the_ledger_row(
    scripted_loop: ScriptedKeepaliveLoop,
) -> None:
    await scripted_loop.ping([answer()])
    shape = await scripted_loop.ping([answer()], max_tokens=None)

    assert shape.declared_max_tokens is None
    latest = scripted_loop.rows()[-1]
    assert "declared_max_tokens" not in latest
    assert scripted_loop.history()[-1].declared_max_tokens is None
