from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import statistics
import sys
from threading import Event, get_ident
from time import perf_counter

import pytest

from chartreux.app_server.config import StatusLineConfigView
from chartreux.app_server.models import UsageWindowSummary
from chartreux.cli.textual_ui.widgets.session_status_line import (
    SessionStatusState,
    format_status_line,
)
from chartreux.core import _usage_io, usage
from tests.perf._metrics import machine_context, percentiles, record

pytestmark = [
    pytest.mark.perf,
    pytest.mark.timeout(120),
    pytest.mark.skipif(
        sys.platform != "linux", reason="performance scenarios require Linux"
    ),
]

_AS_OF = datetime(2026, 6, 15, 12, tzinfo=UTC)
# Match the concurrency harness's 5ms heartbeat; timings are reports, not limits.
_HEARTBEAT_SECONDS = 0.005


def _record(index: int, root: str = "session-000") -> usage.UsageRecord:
    return usage.UsageRecord(
        record_id=f"{root}-{index:06d}",
        root_session_id=root,
        session_id=root,
        agent_role="root",
        agent_profile="performance",
        model="performance-model",
        provider="fake",
        wire_name="performance-deployment",
        project_key="/workspace/" + "retained-project/" * 20,
        occurred_at=_AS_OF,
        outcome=usage.UsageOutcome.COMPLETED,
        usage_state=usage.UsageState.COMPLETE,
        input_tokens=1000,
        cached_input_tokens=100,
        output_tokens=200,
        prices_usd_per_million=usage.UsagePrices(input=1, cached_input=0.5, output=2),
        known_cost_usd=0.00135,
        has_unknown_cost=False,
    )


@pytest.fixture
def retained_home(tmp_path: Path) -> tuple[Path, int]:
    """50 retained roots, 200 content-free ~1KB records each, outside timing."""
    home = tmp_path / "usage"
    size = 0
    for session in range(50):
        root = f"session-{session:03d}"
        directory = home / root
        directory.mkdir(parents=True)
        content = "".join(_record(i, root).model_dump_json() + "\n" for i in range(200))
        data = content.encode()
        (directory / "usage.jsonl").write_bytes(data)
        size += len(data)
    return home, size


def _service(source: usage.UsageReader | usage.UsageReadSnapshot) -> usage.UsageService:
    return usage.UsageService(
        source, clock=lambda: _AS_OF, timezone_resolver=lambda: UTC
    )


async def _heartbeat(active: Event, stop: asyncio.Event, lags: list[float]) -> None:
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        expected = loop.time() + _HEARTBEAT_SECONDS
        await asyncio.sleep(_HEARTBEAT_SECONDS)
        # Count only ticks inside the actual blocking operation, not submission
        # or completion yields (which would also pass with synchronous work).
        if active.is_set():
            lags.append(max(0, loop.time() - expected) * 1000)


@pytest.mark.asyncio
async def test_cold_scan_responsiveness(
    retained_home: tuple[Path, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, size = retained_home
    reader = usage.UsageReader(home)
    service = _service(reader)
    original = reader.reconcile
    active = Event()
    entered = asyncio.Event()
    release = Event()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop_thread = get_ident()
    lags: list[float] = []
    scan_ms: list[float] = []

    def measured_scan() -> usage.UsageReadSnapshot:
        assert get_ident() != loop_thread
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), "scan release watchdog expired"
        active.set()
        started = perf_counter()
        try:
            return original()
        finally:
            scan_ms.append((perf_counter() - started) * 1000)
            active.clear()

    monkeypatch.setattr(reader, "reconcile", measured_scan)
    heartbeat = asyncio.create_task(_heartbeat(active, stop, lags))
    started = perf_counter()
    initial = service.start()
    try:
        # Immediate first read returns loading while a real worker is pending.
        assert service.read().selected.state == usage.SnapshotState.LOADING
        first_read = asyncio.create_task(service.aread())
        await asyncio.wait_for(entered.wait(), 10)
        assert not initial.done() and not first_read.done()
        release.set()
        snapshot = await first_read
        elapsed_ms = (perf_counter() - started) * 1000
        assert snapshot.selected.request_count == 10_000
        assert not snapshot.warnings
        assert lags, "no event-loop progress during the actual retained-home scan"
        assert service.read() is snapshot
        record(
            "usage_cold_scan",
            {
                "files": 50,
                "records": 10_000,
                "ledger_bytes": size,
                "scan_ms": percentiles(scan_ms),
                "first_aread_ms": elapsed_ms,
                "scan_progress_ticks": len(lags),
                "loop_lag_ms": percentiles(lags),
                "machine": machine_context(),
            },
        )
    finally:
        release.set()
        stop.set()
        await heartbeat
        await service.aclose()


@pytest.mark.asyncio
async def test_unchanged_refresh_is_stat_cached(
    retained_home: tuple[Path, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, _ = retained_home
    service = _service(usage.UsageReader(home))
    reads: list[Path] = []
    original_read = _usage_io.read_usage_file
    projects = 0
    original_project = service._project

    def counted_read(path: Path) -> bytes:
        reads.append(path)
        return original_read(path)

    def counted_project(*args, **kwargs):
        nonlocal projects
        projects += 1
        return original_project(*args, **kwargs)

    monkeypatch.setattr(_usage_io, "read_usage_file", counted_read)
    monkeypatch.setattr(service, "_project", counted_project)
    cold: list[float] = []
    warm: list[float] = []
    try:
        # Compare identical reconcile operations, clearing only the stat cache
        # for the cold samples. No fixed machine-dependent duration threshold.
        for _ in range(3):
            service._source = usage.UsageReader(home)
            started = perf_counter()
            await service.reconcile()
            cold.append((perf_counter() - started) * 1000)
        assert len(reads) == 150
        reads.clear()
        projects = 0
        revision = service.revision
        for _ in range(10):
            started = perf_counter()
            result = await service.reconcile()
            warm.append((perf_counter() - started) * 1000)
            assert result.selected.request_count == 10_000
        assert reads == []
        assert projects == 0
        assert service.revision == revision
        assert statistics.median(warm) < statistics.median(cold)
        record(
            "usage_unchanged_refresh",
            {
                "cold_ms": percentiles(cold),
                "warm_ms": percentiles(warm),
                "warm_file_reads": len(reads),
                "warm_aggregations": projects,
                "cold_to_warm_ratio": statistics.median(cold) / statistics.median(warm),
                "machine": machine_context(),
            },
        )
    finally:
        await service.aclose()


@dataclass
class _ScheduledCall:
    callback: Callable[[], None]
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


@pytest.mark.asyncio
async def test_local_append_burst_coalesces_and_reads_only_affected_file(
    retained_home: tuple[Path, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, _ = retained_home
    calls: list[_ScheduledCall] = []

    def schedule(delay: float, callback: Callable[[], None]) -> _ScheduledCall:
        # Hold the production debounce window open deterministically: disk speed
        # must not decide whether this synthetic burst fits in 250ms.
        assert delay == 0.25
        call = _ScheduledCall(callback)
        calls.append(call)
        return call

    reader = usage.UsageReader(home)
    service = usage.UsageService(
        reader, clock=lambda: _AS_OF, timezone_resolver=lambda: UTC, scheduler=schedule
    )
    writer = usage.AsyncUsageWriter(home)
    service.attach_writer(writer)
    updates: list[usage.UsageServiceSnapshot] = []
    service.subscribe(updates.append)
    await service.wait_ready()
    reads: list[Path] = []
    original_read = _usage_io.read_usage_file
    original_iterdir = Path.iterdir
    home_scans = 0

    def counted_read(path: Path) -> bytes:
        reads.append(path)
        return original_read(path)

    def counted_iterdir(path: Path):
        nonlocal home_scans
        if path == home:
            home_scans += 1
        return original_iterdir(path)

    monkeypatch.setattr(_usage_io, "read_usage_file", counted_read)
    monkeypatch.setattr(Path, "iterdir", counted_iterdir)
    try:
        started = perf_counter()
        results = await asyncio.gather(
            *(writer.append(_record(i)) for i in range(200, 250))
        )
        append_ms = (perf_counter() - started) * 1000
        assert all(
            r.disposition == usage.UsageWriteDisposition.APPENDED for r in results
        )
        assert len(calls) == 1
        assert len(updates) == 1
        published = asyncio.Event()
        service.subscribe(lambda _: published.set())
        started = perf_counter()
        calls[0].callback()
        await asyncio.wait_for(published.wait(), 10)
        refresh_ms = (perf_counter() - started) * 1000
        assert [snapshot.revision for snapshot in updates] == [1, 2]
        assert service.read().selected.request_count == 10_050
        assert reads == [home / "session-000" / "usage.jsonl"]
        assert home_scans == 0
        record(
            "usage_local_append",
            {
                "appends": 50,
                "append_burst_ms": append_ms,
                "refresh_ms": refresh_ms,
                "new_revisions": len(updates) - 1,
                "scheduled_flushes": len(calls),
                "affected_file_reads": len(reads),
                "home_scans": home_scans,
                "machine": machine_context(),
            },
        )
    finally:
        await service.aclose()
        await writer.aclose()


@pytest.mark.asyncio
async def test_slow_fsync_keeps_event_loop_servicing_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = usage.AsyncUsageWriter(tmp_path / "usage")
    entered = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    loop_thread = get_ident()
    original = os.fsync
    sync_calls = 0

    def slow_sync(fd: int) -> None:
        nonlocal sync_calls
        assert get_ident() != loop_thread
        sync_calls += 1
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), "fsync release watchdog expired"
        original(fd)

    monkeypatch.setattr(os, "fsync", slow_sync)
    task = asyncio.create_task(writer.append(_record(0)))
    started = perf_counter()
    ticks = 0
    try:
        await asyncio.wait_for(entered.wait(), 10)
        for _ in range(10):
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            assert not task.done(), "writer settled before its sync was released"
            ticks += 1
        release.set()
        result = await task
        assert result.disposition == usage.UsageWriteDisposition.APPENDED
        record(
            "usage_slow_fsync",
            {
                "progress_ticks_while_sync_blocked": ticks,
                "fsync_calls": sync_calls,
                "awaited_append_ms": (perf_counter() - started) * 1000,
                "machine": machine_context(),
            },
        )
    finally:
        release.set()
        await writer.aclose()


@pytest.mark.asyncio
async def test_large_filtered_aggregation_runs_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(_record(i) for i in range(50_000))
    source = usage.UsageReadSnapshot(records=records, state=usage.SnapshotState.READY)
    service = _service(source)
    await service.wait_ready()
    original = service._project
    active = Event()
    stop = asyncio.Event()
    loop_thread = get_ident()
    lags: list[float] = []
    durations: list[float] = []

    def measured_project(*args, **kwargs):
        assert get_ident() != loop_thread
        active.set()
        started = perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            durations.append((perf_counter() - started) * 1000)
            active.clear()

    monkeypatch.setattr(service, "_project", measured_project)
    heartbeat = asyncio.create_task(_heartbeat(active, stop, lags))
    try:
        result = await service.aread("month", records[0].project_key)
        assert result.selected.request_count == 50_000
        assert len(result.models) == 1
        assert len(durations) == 1
        assert lags, "no event-loop progress inside the actual large aggregation"
        record(
            "usage_aggregation",
            {
                "records": len(records),
                "aggregation_ms": percentiles(durations),
                "aggregation_progress_ticks": len(lags),
                "loop_lag_ms": percentiles(lags),
                "machine": machine_context(),
            },
        )
    finally:
        stop.set()
        await heartbeat
        await service.aclose()


def test_statusline_supplied_snapshot_formatting_has_no_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = UsageWindowSummary(
        known_cost_usd=12.34,
        requests=10_000,
        has_known_cost=True,
        start_local=_AS_OF.replace(hour=0),
        end_local=_AS_OF.replace(day=16, hour=0),
        start_utc=_AS_OF.replace(hour=0),
        end_utc=_AS_OF.replace(day=16, hour=0),
        timezone="UTC",
    )
    state = SessionStatusState(
        cwd="/workspace/project",
        home_directory=Path("/home/perf"),
        context_tokens=1000,
        auto_compact_threshold=10_000,
        usage_day=summary,
        usage_week=summary,
        usage_month=summary,
    )
    baseline = StatusLineConfigView(segments=["directory", "context"])
    spend = StatusLineConfigView(
        segments=["directory", "context", "spend-today", "spend-week", "spend-month"]
    )
    before = summary.model_dump()
    reads = 0

    def forbidden(*args, **kwargs):
        nonlocal reads
        reads += 1
        pytest.fail("statusline formatting performed filesystem IO")

    timings: dict[str, list[float]] = {"baseline": [], "spend": []}
    # Patch only during rendering: report/machine-context IO is not render IO.
    with monkeypatch.context() as guard:
        for target, name in (
            (_usage_io, "read_usage_file"),
            (Path, "open"),
            (Path, "read_bytes"),
            (Path, "read_text"),
            (Path, "stat"),
            (Path, "iterdir"),
            (os, "open"),
            (os, "stat"),
        ):
            guard.setattr(target, name, forbidden)
        guard.setattr("builtins.open", forbidden)
        for _ in range(10):
            for name, config in (("baseline", baseline), ("spend", spend)):
                started = perf_counter()
                rendered = ""
                for _ in range(1000):
                    rendered = format_status_line(state, config, 200)
                timings[name].append((perf_counter() - started) * 1000 / 1000)
                if name == "spend":
                    assert "Today $12.34" in rendered
                    assert "Week $12.34" in rendered
                    assert "Month $12.34" in rendered
    assert reads == 0
    assert summary.model_dump() == before
    record(
        "usage_statusline",
        {
            "renders_per_config": 10_000,
            "baseline_ms_per_render": percentiles(timings["baseline"]),
            "spend_ms_per_render": percentiles(timings["spend"]),
            "spend_to_baseline_ratio": statistics.median(timings["spend"])
            / statistics.median(timings["baseline"]),
            "file_reads": reads,
            "machine": machine_context(),
        },
    )
