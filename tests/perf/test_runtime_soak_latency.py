"""X3a contract node: the loaded latency soak mode and its F1/F2 quantities.

The node pins the committed soak shape (plan §5: 400 growing root turns over the
200k-equivalent fixture history, 8 agents, one synchronized child burst every
10th turn, turns 301-400 measured with turn 300 as the warm boundary, a 1 ms
heartbeat cadence, no tracemalloc) together with the F1/F2 row bindings of the
WP6 schema, then EXECUTES a shortened soak through the real harness and checks
what the mode actually produces: a growing transcript, bursts interleaved with
the late window, late-window heartbeat rows attributed by observation time, the
``late-window heartbeat rows`` chunks-coverage field populated, clean teardown,
and no allocation tracing anywhere in the run.

In-suite nodes run shortened fixtures (plan §5): a shortened run must never be
mistaken for campaign data, so the node also proves the fail-closed edge — the
shortened run refuses to emit gated records, and the pinned floors reject its
counts instead of admitting them. Full-scale soaks are opt-in external cells run
through :func:`tests.perf._runtime_soak.main`.

Negative controls keep the node from passing vacuously: a drifted shape changes
its digest, and a short heartbeat row count fails the F2 coverage floor.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import pytest

from tests.perf._runtime_fixture import (
    FIXTURE_HASH,
    FIXTURE_MESSAGE_CHARS,
    FIXTURE_MESSAGE_COUNT,
    FIXTURE_RESPONSE_CHARS,
    fixture_spec,
)
from tests.perf._runtime_report import (
    CELL_SOAK_LATENCY,
    COUNT_LATE_HEARTBEAT_ROWS,
    COVERAGE_LATE_HEARTBEAT_ROWS,
    SOAK_LATE_WINDOW,
    evaluate_run_floor,
    row_spec,
)
from tests.perf._runtime_scenario import (
    BURST_INTERVAL_TURNS,
    LOADED_CHILD_AGENTS,
    LOADED_TOTAL_AGENTS,
    RESPONSE_CHUNK_COUNT,
    ROOT_TURNS,
    ROOT_USER_CHARS,
    WORKLOAD_SEED,
    EgressGuard,
)
from tests.perf._runtime_soak import (
    LATENCY_SOAK_CONTRACT,
    LATENCY_SOAK_FULL,
    SOAK_BUDGET_S,
    SOAK_CHECKPOINT_TURNS,
    SOAK_HEARTBEAT_INTERVAL_S,
    SOAK_KILL_DEADLINE_S,
    SOAK_SEGMENT_TURNS,
    SOAK_SHAPE_HASHES,
    SOAK_SHAPES,
    SoakError,
    SoakLatencyRun,
    run_latency_soak,
    soak_shape,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the soak runtime cells are Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_CONTRACT_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S


def _assert_shape_definitions() -> None:
    """The committed soak shape and the F1/F2 row bindings are pinned."""
    assert set(SOAK_SHAPES) == {"full", "contract"}
    assert soak_shape("full") is LATENCY_SOAK_FULL
    assert soak_shape("contract") is LATENCY_SOAK_CONTRACT
    with pytest.raises(SoakError, match="unknown_shape"):
        soak_shape("uncommitted")

    full = LATENCY_SOAK_FULL
    fixture = fixture_spec()
    assert full.root_turns == ROOT_TURNS == 400
    assert full.history_messages == FIXTURE_MESSAGE_COUNT
    assert full.total_agents == LOADED_TOTAL_AGENTS == 8
    assert full.child_agents == full.burst_agents == LOADED_CHILD_AGENTS == 7
    assert full.burst_interval_turns == BURST_INTERVAL_TURNS == 10
    assert full.heartbeat_interval_s == SOAK_HEARTBEAT_INTERVAL_S == 0.001
    assert full.budget_s == SOAK_BUDGET_S == 360.0
    assert full.kill_deadline_s == SOAK_KILL_DEADLINE_S == 720.0
    assert full.checkpoint_turns == SOAK_CHECKPOINT_TURNS == (0, 100, 200, 300, 400)
    assert full.segment_turns == SOAK_SEGMENT_TURNS == 100

    # Plan §4 F1/F2: exactly turns 301-400, behind the turn-300 warm boundary.
    window = full.late_window
    assert window == SOAK_LATE_WINDOW
    assert window.warm_boundary_turn == 300
    assert window.first_measured_turn == 301
    assert window.last_measured_turn == 400
    assert window.measured_turns == 100
    assert len(range(window.first_measured_turn, window.last_measured_turn + 1)) == 100
    assert window.last_measured_turn == full.root_turns

    # The seed, the sizes and the response shape travel with the fixture hash.
    record = full.record()
    assert record["fixture_hash"] == fixture["fixture_hash"] == FIXTURE_HASH
    assert record["workload_seed"] == WORKLOAD_SEED
    assert record["root_user_chars"] == ROOT_USER_CHARS
    assert record["response_chars"] == FIXTURE_RESPONSE_CHARS
    assert record["response_chunks"] == RESPONSE_CHUNK_COUNT
    assert record["history_messages"] == fixture["message_count"]
    assert record["tracemalloc"] == "disabled"

    # The committed digests bind the shape to the fixture lineage.
    assert SOAK_SHAPE_HASHES["full"] == full.digest()
    assert SOAK_SHAPE_HASHES["contract"] == LATENCY_SOAK_CONTRACT.digest()
    drifted = replace(full, root_turns=399)
    assert drifted.digest() != SOAK_SHAPE_HASHES["full"]
    assert LATENCY_SOAK_FULL.gated is True
    assert LATENCY_SOAK_CONTRACT.gated is False

    # Row bindings: the soak cell, the producer, and F2's late-row coverage.
    f1 = row_spec("F1")
    f2 = row_spec("F2")
    for spec in (f1, f2):
        assert spec.cell == CELL_SOAK_LATENCY
        assert spec.producer_wp == "X3a"
        assert spec.late_window == SOAK_LATE_WINDOW
        assert spec.warmup_floor == window.warm_boundary_turn == 300
    assert f1.unit == f2.unit == "ms"
    assert f1.measured_floor == 100
    assert f1.exact_measured == 100
    assert f2.coverage_unit == COVERAGE_LATE_HEARTBEAT_ROWS
    assert COUNT_LATE_HEARTBEAT_ROWS in f2.required_unit_counts
    assert f2.measured_floor == f2.unit_count_floors[COUNT_LATE_HEARTBEAT_ROWS] == 1000

    # The floors admit exactly the committed full-scale numbers and nothing thin.
    assert evaluate_run_floor("F1", measured_ops=100, warmup_ops=300).eligible
    assert evaluate_run_floor(
        "F2",
        measured_ops=1000,
        warmup_ops=300,
        unit_counts={COUNT_LATE_HEARTBEAT_ROWS: 1000},
    ).eligible
    short_turns = evaluate_run_floor("F1", measured_ops=99, warmup_ops=300)
    assert not short_turns.eligible
    assert "ops_floor_measured" in short_turns.reasons
    short_rows = evaluate_run_floor(
        "F2",
        measured_ops=999,
        warmup_ops=300,
        unit_counts={COUNT_LATE_HEARTBEAT_ROWS: 999},
    )
    assert not short_rows.eligible
    assert f"coverage_floor_{COUNT_LATE_HEARTBEAT_ROWS}" in short_rows.reasons
    missing_rows = evaluate_run_floor("F2", measured_ops=1000, warmup_ops=300)
    assert f"coverage_missing_{COUNT_LATE_HEARTBEAT_ROWS}" in missing_rows.reasons


def _assert_growth_and_late_window(run: SoakLatencyRun) -> None:
    """The run grew its history and measured its late window, once each."""
    assert run.status == "ok"
    assert len(run.turns) == LATENCY_SOAK_CONTRACT.root_turns
    assert [turn.turn for turn in run.turns] == list(
        range(1, LATENCY_SOAK_CONTRACT.root_turns + 1)
    )
    assert run.all_late_turns_complete()

    window = LATENCY_SOAK_CONTRACT.late_window
    measured = run.measured_turns()
    assert [turn.turn for turn in measured] == [6, 7, 8]
    assert len(measured) == window.measured_turns
    assert run.warmup_turns() == window.warm_boundary_turn
    opened = run.window_open_s
    closed = run.window_close_s
    assert opened is not None and closed is not None
    assert closed > opened
    # No late turn lies outside the measured span it opened.
    assert opened <= measured[0].started_s
    assert closed >= measured[-1].finished_s

    growth = run.growth_record()
    assert growth["seeded_messages"] == LATENCY_SOAK_CONTRACT.history_messages + 1
    assert growth["grew"] is True
    assert growth["final_messages"] > growth["seeded_messages"]
    stamps = [checkpoint["history_messages"] for checkpoint in run.checkpoints]
    assert [checkpoint["turn"] for checkpoint in run.checkpoints] == [0, 4, 8]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)

    overheads = run.late_overheads_ms()
    assert len(overheads) == window.measured_turns
    assert run.invalid_late_samples() == 0
    assert all(value >= 0.0 for value in overheads)
    for turn, overhead in zip(measured, overheads, strict=True):
        assert overhead == pytest.approx(turn.wall_ms - turn.injected_delay_ms)
        assert turn.injected_delay_ms == 0.0


def _assert_late_heartbeat_rows(run: SoakLatencyRun) -> None:
    """The late-window heartbeat rows are scoped, counted and carried as coverage."""
    rows = run.late_window_rows()
    assert rows, "the late window logged no heartbeat rows"
    opened = run.window_open_s
    closed = run.window_close_s
    assert opened is not None and closed is not None
    assert all(opened <= row.observed_s <= closed for row in rows)
    assert [row.index for row in rows] == sorted(row.index for row in rows)
    assert [row.excess_ms for row in rows] == run.late_heartbeat_excess_ms()
    assert all(row.excess_ms >= 0.0 for row in rows)
    # Warm-window rows exist and are NOT charged to the late window.
    assert len(run.heartbeat.rows) > len(rows)

    coverage = run.chunks_coverage()
    assert coverage["unit"] == COVERAGE_LATE_HEARTBEAT_ROWS
    assert coverage["late_window_heartbeat_rows"] == len(rows)
    assert coverage["heartbeat_rows"] == len(run.heartbeat.rows)
    assert (
        coverage["measured_turns"] == LATENCY_SOAK_CONTRACT.late_window.measured_turns
    )
    assert (
        coverage["warmup_turns"] == LATENCY_SOAK_CONTRACT.late_window.warm_boundary_turn
    )
    assert coverage["complete"] is True
    assert coverage["missing"] == []
    # A synchronized child burst ran inside the late window: its rows are in scope.
    assert coverage["late_window_child_bursts"] >= 1
    assert coverage["child_bursts"] == 2
    assert coverage["child_turns"] == 2

    summary = run.heartbeat_summary()
    assert summary["late_window_heartbeat_rows"] == len(rows)
    assert summary["interval_ms"] == 1.0
    assert summary["p50"] is not None and summary["p95"] is not None
    assert summary["p50"] <= summary["p95"]


def _assert_segment_report(run: SoakLatencyRun) -> None:
    """The §5 segment report buckets overhead and loop lag, early to late."""
    segments = run.segments()
    width = LATENCY_SOAK_CONTRACT.segment_turns
    assert width == 4
    expected = -(-LATENCY_SOAK_CONTRACT.root_turns // width)
    assert len(segments) == expected == 2
    assert segments[0]["first_turn"] == 1
    assert segments[0]["last_turn"] == 4
    assert segments[-1]["first_turn"] == 5
    assert segments[-1]["last_turn"] == LATENCY_SOAK_CONTRACT.root_turns
    assert sum(segment["turns"] for segment in segments) == len(run.turns)
    for segment in segments:
        assert segment["turns"] > 0
        assert segment["overhead_p50_ms"] is not None
        assert segment["overhead_p50_ms"] <= segment["overhead_p95_ms"]
        assert segment["heartbeat_rows"] >= 0
    assert sum(segment["heartbeat_rows"] for segment in segments) > 0

    change = run.segment_change()
    assert change["segments"] == len(segments)
    early = change["early_overhead_p50_ms"]
    late = change["late_overhead_p50_ms"]
    assert early is not None and late is not None
    assert change["delta_ms"] == pytest.approx(late - early, abs=1e-3)

    report = run.report()
    assert report["segments"] == segments
    assert report["segment_change"] == change


def _assert_no_allocation_tracing(run: SoakLatencyRun) -> None:
    """A latency run carries no allocation tracing, and every sample agrees."""
    run.assert_no_tracemalloc()
    assert run.tracemalloc_active is False
    assert run.tracemalloc_samples
    assert not any(run.tracemalloc_samples)
    report = run.report()
    assert report["tracemalloc"] == {
        "active": False,
        "samples": len(run.tracemalloc_samples),
    }


def _assert_cell_edge_is_closed(run: SoakLatencyRun) -> None:
    """A shortened run refuses gated records and never passes the row floors."""
    with pytest.raises(SoakError, match="shape_not_committed"):
        run.gated_records()
    coverage = run.chunks_coverage()
    turns = evaluate_run_floor(
        "F1",
        measured_ops=coverage["measured_turns"],
        warmup_ops=coverage["warmup_turns"],
    )
    assert not turns.eligible
    assert "ops_floor_warmup" in turns.reasons
    rows = coverage["late_window_heartbeat_rows"]
    heartbeat = evaluate_run_floor(
        "F2",
        measured_ops=rows,
        warmup_ops=coverage["warmup_turns"],
        unit_counts={COUNT_LATE_HEARTBEAT_ROWS: rows},
    )
    assert not heartbeat.eligible
    assert "ops_floor_warmup" in heartbeat.reasons

    report = run.report()
    assert report["gated"] is False
    assert report["mode"] == "contract"
    assert report["shape_hash"] == SOAK_SHAPE_HASHES["contract"]
    assert report["fixture_hash"] == FIXTURE_HASH
    execution = report["execution"]
    assert execution["status"] == "ok"
    assert execution["turns_settled"] == LATENCY_SOAK_CONTRACT.root_turns
    assert execution["stopped_early"] is False
    assert execution["budget_s"] == LATENCY_SOAK_CONTRACT.budget_s
    assert execution["kill_deadline_s"] == LATENCY_SOAK_CONTRACT.kill_deadline_s
    assert report["growth"]["grew"] is True
    assert report["chunks_coverage"]["complete"] is True


def _assert_child_cohort_cleanup(run: SoakLatencyRun) -> None:
    """The cohort's records survived and nothing outlived the run."""
    scenario = run.scenario
    assert run.bursts and all(burst.completed == 1 for burst in run.bursts)
    assert scenario.retained_child_sessions == len(run.bursts)
    scenario.assert_child_session_records()
    scenario.assert_child_history_fidelity(
        minimum_chars=(LATENCY_SOAK_CONTRACT.history_messages * FIXTURE_MESSAGE_CHARS)
    )
    scenario.assert_cleanup()
    assert scenario.registry._children == {}


def _assert_contract_run(run: SoakLatencyRun) -> None:
    _assert_growth_and_late_window(run)
    _assert_late_heartbeat_rows(run)
    _assert_segment_report(run)
    _assert_no_allocation_tracing(run)
    _assert_cell_edge_is_closed(run)
    _assert_child_cohort_cleanup(run)


async def _run_and_assert(session_root: Path) -> None:
    run = await run_latency_soak(LATENCY_SOAK_CONTRACT, session_root=session_root)
    _assert_contract_run(run)


@pytest.mark.timeout(_CONTRACT_TIMEOUT_S)
def test_runtime_soak_latency_contract(tmp_path: Path) -> None:
    """The loaded latency soak holds its committed shape, F1/F2 and its guards."""
    started = perf_counter()
    _assert_shape_definitions()
    guard = EgressGuard(tmp_path / "egress").install()
    try:
        guard.assert_closed()
        asyncio.run(_run_and_assert(tmp_path / "sessions"))
        guard.assert_closed()
    finally:
        guard.restore()
    # The process is the caller's again: no socket hooks, no forced home.
    assert not guard._installed
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"soak latency contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
