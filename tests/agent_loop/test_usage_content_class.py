"""Content-class attribution on the usage ledger (schema v2).

Pinned B1a contract: v1 records parse unchanged, v2 records carry the
content_class work-class label through a write/read round trip, unknown
schema versions stay gated, and routed loops attribute their launch purpose
while direct launches carry None.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from chartreux.core.agent_loop._loop_types import RootAccountingOwner
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role, StopInfo
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.usage import (
    AsyncUsageWriter,
    CoverageWarningCode,
    UsageOutcome,
    UsagePrices,
    UsagePurpose,
    UsageReader,
    UsageRecord,
    UsageState,
)
from tests.conftest import build_test_agent_loop
from tests.stubs.fake_backend import FakeBackend


def record(record_id: str = "one", root: str = "root") -> UsageRecord:
    return UsageRecord(
        record_id=record_id,
        occurred_at=datetime(2026, 2, 20, tzinfo=UTC),
        root_session_id=root,
        session_id="child",
        parent_session_id=root,
        agent_role="subagent",
        agent_profile="coder",
        model="model",
        provider="provider",
        wire_name="deployment",
        project_key="project",
        outcome=UsageOutcome.COMPLETED,
        usage_state=UsageState.COMPLETE,
        input_tokens=10,
        output_tokens=2,
        cached_input_tokens=0,
        prices_usd_per_million=UsagePrices(input=1, output=2),
        known_cost_usd=0.000014,
        has_unknown_cost=False,
    )


def put(usage_dir: Path, root: str, content: bytes) -> Path:
    path = usage_dir / root / "usage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def line(item: dict[str, Any]) -> bytes:
    import json

    return (json.dumps(item) + "\n").encode()


def test_schema_v1_records_still_parse(tmp_path: Path) -> None:
    written = record().model_dump(mode="json")
    assert written.pop("content_class") is None
    written["schema_version"] = 1
    put(tmp_path, "root", line(written))
    records = UsageReader(tmp_path).reconcile().records
    assert records == (record().model_copy(update={"schema_version": 1}),)
    assert records[0].content_class is None


def test_schema_v2_roundtrip_carries_content_class(tmp_path: Path) -> None:
    written = record().model_dump(mode="json")
    written["content_class"] = "implementation"
    put(tmp_path, "root", line(written))
    records = UsageReader(tmp_path).reconcile().records
    assert len(records) == 1
    assert records[0].schema_version == 2
    assert records[0].content_class == "implementation"


def test_unknown_schema_version_gated(tmp_path: Path) -> None:
    written = record().model_dump(mode="json")
    written["schema_version"] = 3
    put(tmp_path, "root", line(written) + line(record("two").model_dump(mode="json")))
    snapshot = UsageReader(tmp_path).reconcile()
    assert [entry.record_id for entry in snapshot.records] == ["two"]
    assert {warning.code for warning in snapshot.warnings} == {
        CoverageWarningCode.UNSUPPORTED_SCHEMA
    }


class AttemptBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.result = LLMChunk(
            message=LLMMessage(role=Role.assistant, content="answer"),
            usage=LLMUsage(prompt_tokens=100, completion_tokens=20, cached_tokens=10),
            stop=StopInfo(reason="stop"),
        )

    async def complete(self, **kwargs: Any) -> LLMChunk:
        notify_request_started()
        return self.result


async def _loop_records(
    tmp_path: Path, **loop_kwargs: Any
) -> tuple[list[UsageRecord], AsyncUsageWriter]:
    records: list[UsageRecord] = []
    writer = AsyncUsageWriter(tmp_path / "usage")
    unsubscribe = writer.subscribe(lambda entry, _result: records.append(entry))
    try:
        owner = RootAccountingOwner(
            root_session_id="root-ledger", project_key="project", writer=writer
        )
        agent = build_test_agent_loop(
            backend=AttemptBackend(), accounting_owner=owner, **loop_kwargs
        )
        assert (await agent._chat()).message.content == "answer"
        await agent.aclose()
        await writer.drain()
    finally:
        unsubscribe()
        await writer.aclose()
    return records, writer


@pytest.mark.asyncio
async def test_loop_attribution_carries_launch_purpose(tmp_path: Path) -> None:
    records, _writer = await _loop_records(tmp_path, launch_purpose="implementation")
    assert records
    assert all(
        entry.content_class == "implementation"
        and entry.schema_version == 2
        and entry.agent_role == "root"
        for entry in records
    )


@pytest.mark.asyncio
async def test_direct_launch_content_class_none(tmp_path: Path) -> None:
    records, _writer = await _loop_records(tmp_path)
    assert records
    assert all(
        entry.content_class is None and entry.schema_version == 2 for entry in records
    )


class CompactionBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.result = LLMChunk(
            message=LLMMessage(role=Role.assistant, content="<summary>done</summary>"),
            usage=LLMUsage(prompt_tokens=100, completion_tokens=20, cached_tokens=10),
            stop=StopInfo(reason="stop"),
        )

    async def complete(self, **kwargs: Any) -> LLMChunk:
        notify_request_started()
        return self.result


def _compaction_config() -> ChartreuxConfigSchema:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-first": {"api_base": "https://first.invalid"}},
            "models": {
                "base": {"deployments": [{"provider": "test-first", "name": "first"}]},
                "compact": {
                    "deployments": [{"provider": "test-first", "name": "compact-first"}]
                },
            },
            "roles": {},
        }),
        "test-revision",
    )
    return ChartreuxConfigSchema.model_validate(
        {"active_model": "base", "compaction_model": "compact"},
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)


@pytest.mark.asyncio
async def test_compaction_content_class_none(tmp_path: Path) -> None:
    records: list[UsageRecord] = []
    writer = AsyncUsageWriter(tmp_path / "usage")
    unsubscribe = writer.subscribe(lambda entry, _result: records.append(entry))
    try:
        owner = RootAccountingOwner(
            root_session_id="root-ledger", project_key="project", writer=writer
        )
        agent = build_test_agent_loop(
            backend=CompactionBackend(),
            accounting_owner=owner,
            launch_purpose="implementation",
            config=_compaction_config(),
        )
        agent.run_identity = "run-1"
        assert (await agent._chat()).message.content == "<summary>done</summary>"
        assert await agent.compaction_manager.compact() == "done"
        await agent.aclose()
        await writer.drain()
    finally:
        unsubscribe()
        await writer.aclose()

    conversation = [r for r in records if r.purpose is UsagePurpose.CONVERSATION]
    compaction = [r for r in records if r.purpose is UsagePurpose.COMPACTION]
    assert conversation and compaction
    # Routed work keeps the launch-purpose label and run id; the utility
    # compaction call carries neither.
    assert all(
        r.content_class == "implementation" and r.run_id == "run-1"
        for r in conversation
    )
    assert all(r.content_class is None and r.run_id is None for r in compaction)
