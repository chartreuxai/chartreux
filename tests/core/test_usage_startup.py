from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chartreux.core import _usage_startup as startup
from chartreux.core.usage import (
    AsyncUsageWriter,
    UsageOutcome,
    UsagePrices,
    UsageReader,
    UsageRecord,
    UsageState,
    UsageWriteDisposition,
)
from tests.conftest import build_test_vibe_config


def startup_record(context: startup.StartupAccountingContext) -> UsageRecord:
    model = build_test_vibe_config().get_active_model()
    return UsageRecord(
        **context.attribution(model).model_dump(),
        record_id="startup-call",
        occurred_at=datetime.now(UTC),
        outcome=UsageOutcome.COMPLETED,
        usage_state=UsageState.MISSING,
        prices_usd_per_million=UsagePrices(),
        known_cost_usd=0,
        has_unknown_cost=True,
    )


def test_claim_is_thread_safe_and_terminal(tmp_path: Path) -> None:
    context = startup._allocate("original-project", tmp_path / "usage")

    def claim() -> bool:
        try:
            context.claim()
        except RuntimeError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(lambda _: claim(), range(16))) == 1
    context.abandon()
    assert context.state == "abandoned"
    with pytest.raises(RuntimeError, match="single-use"):
        context.claim()
    with pytest.raises(FrozenInstanceError):
        setattr(context.identity, "project_key", "changed")  # noqa: B010


@pytest.mark.asyncio
async def test_collision_rerolls_and_reservation_is_not_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage_dir = tmp_path / "usage"
    (usage_dir / "collision").mkdir(parents=True)
    ids = iter(["collision", "reserved"])
    monkeypatch.setattr(startup, "generate_session_id", lambda: next(ids))
    context = await startup.create_startup_accounting_context(
        tmp_path, usage_dir=usage_dir
    )
    assert context.identity.root_session_id == "reserved"
    assert list((usage_dir / "reserved").iterdir()) == []
    assert UsageReader(usage_dir).reconcile().records == ()
    assert not (tmp_path / "sessions").exists()


def test_collision_allocation_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "collision").mkdir()
    monkeypatch.setattr(startup, "generate_session_id", lambda: "collision")
    with pytest.raises(RuntimeError, match="unique"):
        startup._allocate("project", tmp_path)


def test_cross_loop_facades_reuse_blocking_writer(tmp_path: Path) -> None:
    async def early() -> startup.StartupAccountingContext:
        context = await startup.create_startup_accounting_context(
            tmp_path, usage_dir=tmp_path / "usage"
        )
        facade = context.early_writer()
        assert facade._writer is context.writer
        await facade.append(startup_record(context))
        await facade.aclose()
        return context

    context = asyncio.run(early())

    async def main() -> None:
        facade = AsyncUsageWriter(writer=context.writer)
        assert facade._writer is context.writer
        result = await facade.append(startup_record(context))
        assert result.disposition == UsageWriteDisposition.ALREADY_PRESENT
        await facade.aclose()

    asyncio.run(main())
    assert len(context.early_settlements) == 1
    records = UsageReader(tmp_path / "usage").reconcile().records
    assert len(records) == 1
    record = records[0]
    assert (
        record.root_session_id == record.session_id == context.identity.root_session_id
    )
    assert record.agent_role == "startup"
    assert record.parent_session_id is record.agent_profile is None
    assert record.purpose == "worktree-naming"
    assert record.project_key == context.identity.project_key


@pytest.mark.asyncio
async def test_reservation_failure_retains_identity_and_degrades_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(*args: object) -> None:
        raise PermissionError("private-path")

    monkeypatch.setattr(startup, "reserve_usage_root", fail)
    context = await startup.create_startup_accounting_context(
        tmp_path, usage_dir=tmp_path / "usage"
    )
    assert context.identity.root_session_id
    assert context.warnings[0].root_session_id == context.identity.root_session_id
    assert "coverage is degraded" in caplog.text
    assert "private-path" not in caplog.text
    assert not (tmp_path / "usage").exists()


def test_existing_writer_and_directory_are_mutually_exclusive(tmp_path: Path) -> None:
    context = startup._allocate("project", tmp_path)
    with pytest.raises(ValueError, match="existing writer"):
        AsyncUsageWriter(tmp_path, writer=context.writer)


def test_exclusive_reservation_has_only_one_winner(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: startup.reserve_usage_root(tmp_path / "usage", "same-id"),
                range(16),
            )
        )
    assert sum(results) == 1
    assert list((tmp_path / "usage" / "same-id").iterdir()) == []


def test_parent_store_error_is_not_misclassified_as_collision(tmp_path: Path) -> None:
    store = tmp_path / "usage"
    store.write_text("not a directory")
    context = startup._allocate("project", store)
    assert context.state == "available"
    assert context.warnings
