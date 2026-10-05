from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, get_ident
from zoneinfo import ZoneInfo

import pytest

from chartreux.core import usage
from chartreux.core._usage_calendar import calendar_windows
from chartreux.core.usage import (
    CoverageWarning,
    CoverageWarningCode,
    SnapshotState,
    UsageOutcome,
    UsagePrices,
    UsageReader,
    UsageReadSnapshot,
    UsageRecord,
    UsageService,
    UsageState,
    UsageWindow,
    aggregate_usage,
)

AS_OF = datetime(2026, 3, 8, 18, tzinfo=UTC)
PRICES = UsagePrices(input=1, output=2, cached_input=0.5)


def record(
    name: str = "priced",
    *,
    cost: float = 1.25,
    unknown: bool = False,
    prices: UsagePrices = PRICES,
    counts: tuple[int | None, int | None, int | None] = (100, 20, 30),
    project: str | None = "project",
    model: str = "model",
    provider: str = "provider",
    wire_name: str = "deployment",
    occurred_at: datetime = AS_OF,
    state: UsageState | None = None,
) -> UsageRecord:
    presence = (
        UsageState.MISSING
        if all(count is None for count in counts)
        else UsageState.PARTIAL
        if any(count is None for count in counts)
        else UsageState.COMPLETE
    )
    return UsageRecord(
        record_id=name,
        occurred_at=occurred_at,
        root_session_id="root",
        session_id="child",
        agent_role="agent",
        model=model,
        provider=provider,
        wire_name=wire_name,
        project_key=project,
        outcome=UsageOutcome.COMPLETED,
        usage_state=state or presence,
        input_tokens=counts[0],
        output_tokens=counts[1],
        cached_input_tokens=counts[2],
        prices_usd_per_million=prices,
        known_cost_usd=cost,
        has_unknown_cost=unknown,
    )


def test_mixed_priced_unknown_free_and_missing() -> None:
    records = (
        record(model="priced"),
        record("unknown", model="unknown", cost=0, unknown=True, prices=UsagePrices()),
        record(
            "free",
            model="free",
            cost=0,
            prices=UsagePrices(input=0, output=0, cached_input=0),
        ),
        record(
            "missing", model="missing", cost=0, unknown=True, counts=(None, None, None)
        ),
    )
    result = aggregate_usage(records, as_of=AS_OF, timezone=UTC)
    for summary in (
        result.summaries.day,
        result.summaries.week,
        result.summaries.month,
    ):
        totals = summary.totals
        assert totals.request_count == 4
        assert totals.input_tokens == 300
        assert totals.output_tokens == 60
        assert totals.cached_input_tokens == 90
        assert totals.known_cost_usd == 1.25
        assert totals.has_known_cost and totals.has_unknown_cost
        assert totals.has_unknown_tokens
        assert totals.input.tokens == 210
        assert totals.input.has_unknown_tokens
    rows = {row.model: row for row in result.models}
    assert rows["priced"].has_known_cost and not rows["priced"].has_unknown_cost
    assert rows["free"].has_known_cost and not rows["free"].has_unknown_cost
    assert rows["free"].known_cost_usd == 0
    assert not rows["unknown"].has_known_cost and rows["unknown"].has_unknown_cost
    assert not rows["missing"].has_known_cost and rows["missing"].has_unknown_cost


def test_stored_costs_never_repriced(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Aggregation must not invoke pricing or the model catalog")

    monkeypatch.setattr(usage, "price_usage", forbidden)
    monkeypatch.setattr(usage, "capture_prices", forbidden)
    result = aggregate_usage(
        (record(cost=12), record("later", cost=3, prices=UsagePrices(input=99999))),
        as_of=AS_OF,
        timezone=UTC,
    )
    assert result.selected.known_cost_usd == 15
    assert result.models[0].known_cost_usd == 15
    # Component cost cannot be reconstructed from a stored total: expose only
    # its observed token lower bound rather than silently repricing history.
    assert result.selected.input.tokens == 140
    assert result.selected.input.known_cost_usd == 0
    assert result.selected.input.has_unknown_cost


@pytest.mark.parametrize(
    "project, expected", [(None, 3), ("project", 1), ("other", 1), ("missing", 0)]
)
def test_project_filter(project: str | None, expected: int) -> None:
    records = (record(), record("other", project="other"), record("null", project=None))
    # Null project keys must also survive actual record parsing.
    assert (
        UsageRecord.model_validate_json(records[2].model_dump_json()).project_key
        is None
    )
    result = aggregate_usage(records, as_of=AS_OF, timezone=UTC, project_key=project)
    assert result.selected.request_count == expected
    assert result.project_filter == project


@pytest.mark.parametrize("window", ["day", "week", "month"])
@pytest.mark.parametrize("timezone", ["America/New_York", "Asia/Kathmandu"])
def test_boundaries(window: UsageWindow, timezone: str) -> None:
    calendar = calendar_windows(AS_OF, timezone)
    boundary = getattr(calendar, window)
    records = (
        record("before", occurred_at=boundary.start_utc - timedelta(microseconds=1)),
        record("start", occurred_at=boundary.start_utc),
        record("inside", occurred_at=boundary.end_utc - timedelta(microseconds=1)),
        record("end", occurred_at=boundary.end_utc),
    )
    result = aggregate_usage(records, as_of=AS_OF, timezone=timezone, window=window)
    assert result.calendar == calendar
    assert getattr(result.summaries, window).boundaries == boundary
    assert result.selected.request_count == 2
    assert result.selected.known_cost_usd == 2.5


def test_deployment_triple_grouping_and_sorted_rows() -> None:
    records = (
        record("one", provider="z", wire_name="b"),
        record("two", provider="z", wire_name="b"),
        record("three", provider="z", wire_name="a"),
        record("four", provider="a", wire_name="b"),
        record("five", model="other"),
    )
    result = aggregate_usage(records, as_of=AS_OF, timezone=UTC)
    keys = [(row.model, row.provider, row.wire_name) for row in result.models]
    assert keys == sorted(keys)
    assert [row.request_count for row in result.models] == [1, 1, 2, 1]
    assert (
        sum(row.input.tokens for row in result.models) == result.selected.input.tokens
    )


@pytest.mark.parametrize(
    "counts, uncached, unknown",
    [
        ((100, 20, 30), 70, False),
        ((100, None, 30), 70, True),
        ((100, 20, None), 0, True),
        ((None, 20, 30), 0, True),
        ((0, 0, 0), 0, False),
    ],
)
def test_component_lower_bounds(
    counts: tuple[int | None, int | None, int | None], uncached: int, unknown: bool
) -> None:
    result = aggregate_usage((record(counts=counts),), as_of=AS_OF, timezone=UTC)
    totals = result.selected
    assert totals.input.tokens == uncached
    assert totals.cached_input.tokens == (counts[2] or 0)
    assert totals.output.tokens == (counts[1] or 0)
    assert totals.has_unknown_tokens == unknown
    if not unknown:
        assert totals.input.tokens + totals.cached_input.tokens == totals.input_tokens
        assert totals.cached_input_tokens <= totals.input_tokens


def test_partial_stream_with_present_counts_remains_incomplete() -> None:
    result = aggregate_usage(
        (record(state=UsageState.PARTIAL, unknown=True),), as_of=AS_OF, timezone=UTC
    )
    assert result.selected.has_unknown_tokens
    assert result.models[0].has_unknown_tokens
    assert result.selected.input.has_unknown_tokens
    assert result.selected.input_tokens == 100


def test_empty_is_ready_zero_and_complete() -> None:
    result = aggregate_usage((), as_of=AS_OF, timezone=UTC)
    assert result.models == ()
    for summary in (
        result.summaries.day,
        result.summaries.week,
        result.summaries.month,
    ):
        totals = summary.totals
        assert totals.state == SnapshotState.READY
        assert totals.request_count == 0
        assert totals.known_cost_usd == 0
        assert not totals.has_unknown_cost and not totals.has_unknown_tokens
        assert not totals.input.has_unknown_tokens
        assert not totals.input.has_unknown_cost


@pytest.mark.parametrize("state", list(SnapshotState))
def test_service_preserves_snapshot_state_warnings_and_shared_instant(
    state: SnapshotState,
) -> None:
    warning = CoverageWarning(code=CoverageWarningCode.UNREADABLE)
    snapshot = UsageReadSnapshot((record(),), (warning,), state)
    calls = []

    def clock() -> datetime:
        calls.append("clock")
        return AS_OF

    def timezone() -> ZoneInfo:
        calls.append("timezone")
        return ZoneInfo("America/New_York")

    result = UsageService(snapshot, clock=clock, timezone_resolver=timezone).read(
        "month", "project"
    )
    assert calls == ["clock", "timezone"]
    assert result.as_of == AS_OF
    assert result.warnings == (warning,)
    assert result.selected is result.summaries.month.totals
    for summary in (
        result.summaries.day,
        result.summaries.week,
        result.summaries.month,
    ):
        assert summary.totals.state == state
        assert summary.totals.warnings == (warning,)


def test_service_reads_cache_without_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = UsageReader(tmp_path / "missing")
    reader.reconcile()

    def forbidden() -> UsageReadSnapshot:
        pytest.fail("read must not reconcile synchronously")

    monkeypatch.setattr(reader, "reconcile", forbidden)
    monkeypatch.setattr(reader, "refresh_invalidated", forbidden)
    service = UsageService(reader, clock=lambda: AS_OF, timezone_resolver=lambda: UTC)
    assert service.read().selected.state == SnapshotState.READY


@pytest.mark.asyncio
async def test_aggregation_runs_off_loop() -> None:
    records = tuple(record(str(index)) for index in range(1000))
    result = await asyncio.to_thread(
        aggregate_usage, records, as_of=AS_OF, timezone=UTC
    )
    assert result.selected.request_count == 1000
    assert result.selected.known_cost_usd == 1250


class ScheduledCall:
    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class Scheduler:
    def __init__(self) -> None:
        self.calls: list[ScheduledCall] = []

    def __call__(self, delay: float, callback: Callable[[], None]) -> ScheduledCall:
        assert delay == 0.25
        call = ScheduledCall(callback)
        self.calls.append(call)
        return call

    def fire(self) -> None:
        calls, self.calls = self.calls, []
        for call in calls:
            if not call.cancelled:
                call.callback()


@pytest.mark.asyncio
async def test_background_readiness_and_off_loop_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = UsageReader(tmp_path)
    entered = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    main_thread = get_ident()
    original = reader.reconcile
    aggregate = usage.aggregate_usage

    def blocked() -> UsageReadSnapshot:
        assert get_ident() != main_thread
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original()

    def checked(*args: object, **kwargs: object) -> usage.UsageServiceSnapshot:
        assert get_ident() != main_thread
        return aggregate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(reader, "reconcile", blocked)
    service = UsageService(reader, clock=lambda: AS_OF, timezone_resolver=lambda: UTC)
    updates: list[usage.UsageServiceSnapshot] = []
    service.subscribe(updates.append)
    task = service.start()
    assert not task.done()
    assert service.state == SnapshotState.LOADING
    assert service.read().selected.state == SnapshotState.LOADING
    monkeypatch.setattr(usage, "aggregate_usage", checked)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not task.done()  # The event loop ran while the scan was blocked.
    finally:
        release.set()
    await service.wait_ready()
    assert service.state == SnapshotState.READY
    assert service.read().selected.request_count == 0
    assert len(updates) == 1 and updates[0].revision == 1
    await service.aread("month", "project")
    await service.aclose()


@pytest.mark.asyncio
async def test_settlement_burst_coalesces_without_home_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = UsageReader(tmp_path)
    scheduler = Scheduler()
    writer = usage.AsyncUsageWriter(tmp_path)
    service = UsageService(
        reader, clock=lambda: AS_OF, timezone_resolver=lambda: UTC, scheduler=scheduler
    )
    service.attach_writer(writer)
    service.attach_writer(writer)
    updates: list[usage.UsageServiceSnapshot] = []
    service.subscribe(updates.append)
    await service.wait_ready()

    def forbidden() -> UsageReadSnapshot:
        pytest.fail("Local settlements must not enumerate the home")

    monkeypatch.setattr(reader, "reconcile", forbidden)
    await asyncio.gather(*(writer.append(record(str(i))) for i in range(20)))
    assert service.revision == 1
    assert len(scheduler.calls) == 1
    published = asyncio.Event()
    service.subscribe(lambda _: published.set())
    scheduler.fire()
    await asyncio.wait_for(published.wait(), 5)
    assert service.read().selected.request_count == 20
    assert [item.revision for item in updates] == [1, 2]
    assert (await service.aread("week", "project")).revision == 2
    await service.aclose()
    await writer.aclose()


@pytest.mark.asyncio
async def test_reconcile_external_append_and_unchanged_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = UsageReader(tmp_path)
    service = UsageService(reader, clock=lambda: AS_OF, timezone_resolver=lambda: UTC)
    await service.wait_ready()
    await asyncio.to_thread(usage.UsageWriter(tmp_path).append, record())
    result = await service.reconcile()
    assert result.selected.request_count == 1 and result.revision == 2

    def forbidden(*args: object) -> bytes:
        pytest.fail("Unchanged files must not be read again")

    from chartreux.core import _usage_io

    monkeypatch.setattr(_usage_io, "read_usage_file", forbidden)
    assert (await service.reconcile()).revision == 2
    await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "before, after, window",
    [
        (datetime(2026, 3, 8, 23, tzinfo=UTC), datetime(2026, 3, 9, tzinfo=UTC), "day"),
        (
            datetime(2026, 3, 8, 23, tzinfo=UTC),
            datetime(2026, 3, 9, tzinfo=UTC),
            "week",
        ),
        (
            datetime(2026, 3, 31, 23, tzinfo=UTC),
            datetime(2026, 4, 1, tzinfo=UTC),
            "month",
        ),
    ],
)
async def test_rollover_republishes_without_file_changes(
    before: datetime, after: datetime, window: UsageWindow
) -> None:
    now = before
    service = UsageService(
        UsageReadSnapshot((record(occurred_at=before),), state=SnapshotState.READY),
        clock=lambda: now,
        timezone_resolver=lambda: UTC,
    )
    updates: list[usage.UsageServiceSnapshot] = []
    service.subscribe(updates.append)
    await service.wait_ready()
    assert (await service.aread(window)).selected.request_count == 1
    now = after
    result = await service.aread(window)
    assert result.selected.request_count == 0
    assert result.revision == 2 and len(updates) == 2
    await service.aclose()


@pytest.mark.asyncio
async def test_timezone_change_invalidates_and_cached_read_schedules_rebuild() -> None:
    zone = ZoneInfo("UTC")
    now = datetime(2026, 3, 9, 1, tzinfo=UTC)
    service = UsageService(
        UsageReadSnapshot((record(occurred_at=AS_OF),), state=SnapshotState.READY),
        clock=lambda: now,
        timezone_resolver=lambda: zone,
    )
    await service.wait_ready()
    assert service.read().selected.request_count == 0
    updated = asyncio.Event()
    service.subscribe(lambda _: updated.set())
    zone = ZoneInfo("America/New_York")
    service.read()
    await asyncio.wait_for(updated.wait(), 5)
    assert service.read().selected.request_count == 1
    assert service.revision == 2
    await service.aclose()


@pytest.mark.asyncio
async def test_stale_scan_cannot_publish_over_newer_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = UsageReader(tmp_path)
    writer = usage.AsyncUsageWriter(tmp_path)
    await writer.append(record("old"))
    entered = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    original = reader._read_entry
    first = True

    def blocked(
        path: Path, cached: usage._UsageFileEntry | None
    ) -> usage._UsageFileEntry | None:
        nonlocal first
        result = original(path, cached)
        if first:
            first = False
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return result

    monkeypatch.setattr(reader, "_read_entry", blocked)
    service = UsageService(
        reader,
        clock=lambda: AS_OF,
        timezone_resolver=lambda: UTC,
        scheduler=Scheduler(),
    )
    service.attach_writer(writer)
    updates: list[usage.UsageServiceSnapshot] = []
    service.subscribe(updates.append)
    service.start()
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await writer.append(record("new"))
    finally:
        release.set()
    await service.wait_ready()
    assert reader.snapshot.state == SnapshotState.READY
    assert service.read().selected.request_count == 2
    assert [item.selected.request_count for item in updates] == [2]
    await service.aclose()
    await writer.aclose()


@pytest.mark.asyncio
async def test_shutdown_stops_publication_timer_and_drains_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = usage.AsyncUsageWriter(tmp_path)
    scheduler = Scheduler()
    service = UsageService(
        UsageReader(tmp_path),
        clock=lambda: AS_OF,
        timezone_resolver=lambda: UTC,
        scheduler=scheduler,
    )
    service.attach_writer(writer)
    await service.wait_ready()
    await writer.append(record("first"))
    timer = scheduler.calls[0]
    entered = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    original = writer._writer.append

    def blocked(item: UsageRecord) -> usage.UsageWriteResult:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original(item)

    monkeypatch.setattr(writer._writer, "append", blocked)
    append = asyncio.create_task(writer.append(record("pending")))
    await asyncio.wait_for(entered.wait(), 5)
    closing = asyncio.create_task(service.aclose())
    try:
        # Synchronize with unsubscribe, rather than sleeping for the close task.
        while writer._callbacks:
            await asyncio.sleep(0)
        assert timer.cancelled and not closing.done()
        scheduler.fire()
        assert service.revision == 1
    finally:
        release.set()
    await closing
    await append
    fresh = await asyncio.to_thread(UsageReader(tmp_path).reconcile)
    assert len(fresh.records) == 2
    assert service.revision == 1
    await writer.aclose()


@pytest.mark.asyncio
async def test_inflight_aggregation_is_superseded_by_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = usage.AsyncUsageWriter(tmp_path)
    scheduler = Scheduler()
    service = UsageService(
        UsageReader(tmp_path),
        clock=lambda: AS_OF,
        timezone_resolver=lambda: UTC,
        scheduler=scheduler,
    )
    service.attach_writer(writer)
    await service.wait_ready()
    entered = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    original = service._project
    first = True

    def blocked(
        snapshot: UsageReadSnapshot,
        calendar: usage.CalendarWindows,
        window: UsageWindow = "day",
        project_key: str | None = None,
    ) -> usage.UsageServiceSnapshot:
        nonlocal first
        result = original(snapshot, calendar, window, project_key)
        if first:
            first = False
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return result

    monkeypatch.setattr(service, "_project", blocked)
    updates: list[usage.UsageServiceSnapshot] = []
    published = asyncio.Event()
    service.subscribe(updates.append)
    unsubscribe = service.subscribe(lambda _: published.set())
    await writer.append(record("first"))
    scheduler.fire()
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await writer.append(record("second"))
    finally:
        release.set()
    await asyncio.wait_for(published.wait(), 5)
    assert [item.selected.request_count for item in updates] == [2]
    assert service.revision == 2
    unsubscribe()
    unsubscribe()
    assert (await service.reconcile()).revision == 2
    await service.aclose()
    await writer.aclose()
