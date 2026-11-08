"""P5-Reanchor contract node: the save-loaded C1-C3 producer cell.

C1 (worker scan wall), C2 (lock-held await wall, executor dispatch included)
and C3 (loop heartbeat excess) are all gated by the single contract node
``test_runtime_save_scan_contract``; the cell itself lives in
``tests/perf/_runtime_save_scan.py``.

The node drives the REAL save path on the shortened fixture the plan allows
in-suite (§5), then validates the run records the campaign consumes: the
worker scan runs off the event loop, the lock-held await brackets it and is
distinct from the write phase, the heartbeat rows are logged while the
workload runs, and every record carries its chunks coverage (saves per run,
heartbeat rows per run).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import pytest

from tests.perf._runtime_metrics import REPORT_ENV
from tests.perf._runtime_report import SCHEMA_ID, nearest_rank
from tests.perf._runtime_save_scan import (
    CONTRACT_SAVE_CELL,
    SAVE_HEARTBEAT_INTERVAL_MS,
    SAVE_MIN_HEARTBEAT_ROWS,
    SAVE_WORKER_TARGET,
    SaveCellRun,
    build_save_records,
    report_destination,
    run_save_cell,
    write_run_report,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="the save-cell heartbeat and thread identity assume Linux semantics",
    ),
]

_RUN_START_UTC = "2026-11-08T00:00:00Z"
_RUN_ID = "save-cell-contract-1"
_MACHINE_FIELDS = ("cpu_model", "core_count", "python_version", "platform")
_WORKER_PHASE = "worker_scan_ms"
_LOCK_PHASE = "lock_held_await_ms"
_SAMPLE_PRECISION = 4


def _assert_egress_blocked() -> None:
    """The pytest session's transport isolation installs the socket block.

    The cell starts no subprocess and reaches no provider, so the in-suite
    session fixture is the fail-closed egress path it runs under; this asserts
    that block is actually installed rather than assuming it.
    """
    installed = vars(socket.socket)
    assert "connect" in installed, "session provider transport isolation missing"
    assert "connect_ex" in installed, "session provider transport isolation missing"


def _assert_c1(run: SaveCellRun) -> None:
    """C1: the worker phase runs off the event loop, once per successful save."""
    assert run.measured_saves == CONTRACT_SAVE_CELL.measured_saves
    assert len(run.worker_samples) == CONTRACT_SAVE_CELL.measured_saves
    idents = run.worker_thread_idents()
    assert len(idents) == 1, f"the scan rotated worker threads: {idents}"
    assert idents[0] != run.loop_thread_ident
    for sample in run.worker_samples:
        assert sample.target == SAVE_WORKER_TARGET
        assert sample.outcome == "returned"
        assert sample.duration_ms >= 0.0
        assert sample.thread_ident == idents[0]
        assert sample.finished_ns >= sample.started_ns


def _assert_c2(run: SaveCellRun) -> None:
    """C2: the lock-held await brackets the scan and excludes the write phase."""
    assert len(run.lock_await_samples) == run.measured_saves
    assert len(run.persist_await_samples) == run.measured_saves
    for scan, worker, persist in zip(
        run.lock_await_samples,
        run.worker_samples,
        run.persist_await_samples,
        strict=True,
    ):
        assert scan.target == SAVE_WORKER_TARGET
        assert scan.outcome == "returned"
        assert scan.thread_ident == run.loop_thread_ident
        assert worker.thread_ident != scan.thread_ident
        # The await span opens before the worker starts (that span carries the
        # executor dispatch) and closes after the worker returned.
        assert scan.started_ns <= worker.started_ns
        assert scan.finished_ns >= worker.finished_ns
        assert scan.duration_ms >= worker.duration_ms
        # The write is a separate await: C2 is the scan await, never the full
        # lock occupancy.
        assert persist.target == "persist"
        assert persist.outcome == "returned"
        assert scan.finished_ns <= persist.started_ns
        assert persist.finished_ns >= persist.started_ns


def _assert_c3(run: SaveCellRun) -> None:
    """C3: heartbeat rows are logged across the measured save window."""
    assert run.measured_heartbeat_rows >= 1
    assert run.warmup_heartbeat_rows >= 0
    assert (
        run.warmup_heartbeat_rows + run.measured_heartbeat_rows
        == run.total_heartbeat_rows
    )
    for row in run.heartbeat_rows:
        assert row.scheduled_ms == SAVE_HEARTBEAT_INTERVAL_MS
        assert row.observed_ms >= 0.0
        assert row.excess_ms == max(0.0, row.observed_ms - row.scheduled_ms)
        assert row.finished_ns >= row.started_ns
    starts = [row.started_ns for row in run.heartbeat_rows]
    assert starts == sorted(starts)


def _assert_coverage(record: dict[str, Any], *, unit: str) -> dict[str, Any]:
    """Return the record's chunks coverage after checking it is populated."""
    coverage = record["chunks_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["unit"] == unit
    assert coverage["complete"] is True
    assert coverage["missing"] == []
    return coverage


def _assert_save_timing_records(
    run: SaveCellRun, records: dict[str, dict[str, Any]]
) -> None:
    """C1/C2: the save records carry both phases, the counts and the identities."""
    assert set(records) == {"C1", "C2", "C3"}
    saves = run.measured_saves
    for campaign_id, phase in (("C1", _WORKER_PHASE), ("C2", _LOCK_PHASE)):
        record = records[campaign_id]
        assert record["campaign_id"] == campaign_id
        assert record["cell"] == "save_loaded_200k"
        assert record["producer_wp"] == "P5-Reanchor"
        coverage = _assert_coverage(record, unit="saves")
        assert coverage["measured_ops"] == saves
        assert coverage["warmup_ops"] == CONTRACT_SAVE_CELL.warmup_saves
        assert coverage["unit_counts"] == {"saves": saves}
        timings = record["save_timings"]
        assert isinstance(timings, dict)
        assert timings["threads_distinct"] is True
        assert timings["worker_thread_ident"] == run.single_worker_thread_ident()
        assert timings["loop_thread_ident"] == run.loop_thread_ident
        assert len(timings[phase]) == saves
        assert record["samples"] == timings[phase], f"{phase} is not the gated series"
        assert record["eligibility"] == {"eligible": True, "reasons": []}
        summary = record["summary"]
        assert isinstance(summary, dict)
        assert summary["count"] == saves
        assert summary["p95"] == nearest_rank(record["samples"], fraction=0.95)
        assert summary["p95"] >= 0.0
        machine = record["machine"]
        assert isinstance(machine, dict)
        for field in _MACHINE_FIELDS:
            assert machine[field] is not None, f"machine context lacks {field}"
        assert record["fixture_hash"] == run.fixture_hash
    assert records["C1"]["metric"] == "save_worker_scan_p95_ms"
    assert records["C2"]["metric"] == "save_lock_held_await_p95_ms"


def _assert_heartbeat_record(
    run: SaveCellRun, records: dict[str, dict[str, Any]]
) -> None:
    """C3: the heartbeat record carries its rows and applies the row floor."""
    record = records["C3"]
    assert record["campaign_id"] == "C3"
    assert record["metric"] == "loop_heartbeat_excess_p95_ms"
    coverage = _assert_coverage(record, unit="heartbeat rows")
    assert coverage["measured_ops"] == run.measured_heartbeat_rows
    assert coverage["warmup_ops"] == run.warmup_heartbeat_rows
    samples = record["samples"]
    assert len(samples) == run.measured_heartbeat_rows
    assert samples == [
        round(value, _SAMPLE_PRECISION) for value in run.heartbeat_excess_ms()
    ]
    # The shortened in-suite fixture cannot reach the 1000-row floor, and the
    # record must say so rather than pass silently.
    reasons = record["eligibility"]["reasons"]
    assert ("ops_floor_measured" in reasons) == (
        run.measured_heartbeat_rows < SAVE_MIN_HEARTBEAT_ROWS
    )


def _assert_report_round_trip(
    records: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    """The run records write and read back as one validated report document."""
    destination = tmp_path / "reports" / "save-cell.json"
    written = write_run_report(list(records.values()), destination)
    assert written == destination
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["schema"] == SCHEMA_ID
    assert [entry["campaign_id"] for entry in document["runs"]] == ["C1", "C2", "C3"]
    if REPORT_ENV in os.environ:
        assert report_destination() is not None
    else:
        assert report_destination() is None


@pytest.mark.timeout(2)
def test_runtime_save_scan_contract(tmp_path: Path) -> None:
    """C1-C3 come off the landed P5 save path with their coverage populated."""
    _assert_egress_blocked()
    run = run_save_cell(CONTRACT_SAVE_CELL, save_root=tmp_path / "sessions")
    _assert_c1(run)
    _assert_c2(run)
    _assert_c3(run)
    records = build_save_records(run, run_id=_RUN_ID, run_start_utc=_RUN_START_UTC)
    _assert_save_timing_records(run, records)
    _assert_heartbeat_record(run, records)
    _assert_report_round_trip(records, tmp_path)
