"""X3b contract node: the loaded memory soak mode, G1, and its reporting extras.

Pinned here (plan §5, Appendix B row G1): the memory cell runs the loaded
schedule with allocation tracing ACTIVE and an RSS sample after every settled
turn; the gated quantity is one WITHIN-RUN p95 over the 101 rolling
least-squares RSS slopes of the closed [200,300] .. [300,400] window geometry,
computed from exactly the 201 samples of turns 200..400; and signed slopes are
recorded as measured while the gate sees them clamped to zero, so a plateaued
baseline disarms itself through the nonpositive-threshold dormancy rule instead
of gating. The five fresh processes — never the 101 overlapping windows — are
the independent replicates. The retained-set checkpoint diffs are
reporting-only: five structural stamps yield four diffs, and four diffs cannot
support a within-run p95.

The node then EXECUTES a shortened memory soak through the real harness and
checks what the mode actually produces: an RSS sample per settled turn, tracing
observed active at every turn, a closed window geometry whose slopes exist and
clamp correctly, the chunks-coverage fields (RSS samples + window slopes), two
retained-set diffs from three stamps, and clean teardown.

Mode boundaries are asserted, not assumed: a shortened memory run refuses its
gated record, the inherited F1/F2 record path refuses a memory shape outright,
the memory report carries no latency section, and the run leaves no tracing
behind — allocation instrumentation never co-runs with the latency rows.
Full-scale soaks are opt-in external cells (``--mode memory``).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc

import pytest

from tests.perf._runtime_fixture import (
    FIXTURE_HASH,
    FIXTURE_MESSAGE_COUNT,
    FIXTURE_RESPONSE_CHARS,
    FIXTURE_TURN_WIDTH,
    fixture_spec,
)
from tests.perf._runtime_report import (
    CELL_SOAK_LATENCY,
    CELL_SOAK_MEMORY,
    COUNT_RSS_SAMPLES,
    COUNT_WINDOW_SLOPES,
    COVERAGE_ROLLING_WINDOWS,
    MEMORY_WINDOW_SPAN,
    SLOPE_CLAMP_RULE,
    SLOPE_UNIT,
    evaluate_run_floor,
    row_spec,
)
from tests.perf._runtime_scenario import (
    BURST_INTERVAL_TURNS,
    LOADED_CHILD_AGENTS,
    LOADED_TOTAL_AGENTS,
    ROOT_TURNS,
    WORKLOAD_SEED,
    EgressGuard,
)
from tests.perf._runtime_soak import (
    LATENCY_SOAK_FULL,
    MEMORY_FIRST_SAMPLE_TURN,
    MEMORY_LAST_SAMPLE_TURN,
    MEMORY_OPEN_CAMPAIGN_NOTE,
    MEMORY_RETAINED_SET_GATE,
    MEMORY_RETAINED_SET_NULL_REASON,
    MEMORY_SLOPE_CLAMP,
    MEMORY_SLOPE_UNIT,
    MEMORY_SOAK_CONTRACT,
    MEMORY_SOAK_FULL,
    MEMORY_SOAK_SHAPE_HASHES,
    MEMORY_SOAK_SHAPES,
    MEMORY_WINDOW_COUNT,
    MEMORY_WINDOW_WIDTH_TURNS,
    MEMORY_WINDOWS_FULL,
    SOAK_BUDGET_S,
    SOAK_CHECKPOINT_TURNS,
    SOAK_KILL_DEADLINE_S,
    SOAK_MODE_CONTRACT,
    SOAK_SHAPES,
    SOAK_TRACEMALLOC_ACTIVE,
    SOAK_TRACEMALLOC_DISABLED,
    MemoryWindows,
    RssSample,
    SoakError,
    SoakMemoryRun,
    WindowSlope,
    least_squares_slope,
    memory_shape_gated,
    memory_soak_shape,
    rolling_window_slopes,
    run_memory_soak,
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

#: A 512-byte-per-turn ramp, used to check the slope arithmetic exactly.
_RAMP_BYTES_PER_TURN = 512


def _ramp(total: int, *, per_turn: int = _RAMP_BYTES_PER_TURN) -> dict[int, int]:
    """Return ``total`` per-turn RSS samples growing by ``per_turn`` bytes."""
    first = MEMORY_WINDOWS_FULL.first_sample_turn
    return {
        turn: 1_000_000 + per_turn * (turn - first)
        for turn in range(first, first + total)
    }


def _assert_memory_definitions() -> None:
    """The memory shapes, their geometry and the G1 row binding are pinned."""
    assert set(MEMORY_SOAK_SHAPES) == {"memory-full", "memory-contract"}
    assert memory_soak_shape("memory-full") is MEMORY_SOAK_FULL
    assert memory_soak_shape("memory-contract") is MEMORY_SOAK_CONTRACT
    with pytest.raises(SoakError, match="unknown_shape"):
        memory_soak_shape("full")

    # The memory registry is its own run set: the latency registry is untouched.
    assert set(SOAK_SHAPES) == {"full", "contract"}
    assert soak_shape("full") is LATENCY_SOAK_FULL
    assert LATENCY_SOAK_FULL.memory_windows is None
    assert LATENCY_SOAK_FULL.cell == CELL_SOAK_LATENCY

    full = MEMORY_SOAK_FULL
    fixture = fixture_spec()
    assert full.root_turns == ROOT_TURNS == 400
    assert full.history_messages == FIXTURE_MESSAGE_COUNT
    assert full.total_agents == LOADED_TOTAL_AGENTS == 8
    assert full.child_agents == full.burst_agents == LOADED_CHILD_AGENTS == 7
    assert full.burst_interval_turns == BURST_INTERVAL_TURNS == 10
    assert full.budget_s == SOAK_BUDGET_S == 360.0
    assert full.kill_deadline_s == SOAK_KILL_DEADLINE_S == 720.0
    assert full.checkpoint_turns == SOAK_CHECKPOINT_TURNS == (0, 100, 200, 300, 400)
    assert full.cell == CELL_SOAK_MEMORY != CELL_SOAK_LATENCY
    assert full.tracemalloc_mode == SOAK_TRACEMALLOC_ACTIVE
    assert full.memory_windows == MEMORY_WINDOWS_FULL

    # G1's geometry: 101 rolling 100-turn windows over the 201 sampled turns.
    geometry = MEMORY_WINDOWS_FULL
    assert geometry.first_sample_turn == MEMORY_FIRST_SAMPLE_TURN == 200
    assert geometry.last_sample_turn == MEMORY_LAST_SAMPLE_TURN == 400
    assert geometry.window_width_turns == MEMORY_WINDOW_WIDTH_TURNS == 100
    assert geometry.window_count == MEMORY_WINDOW_COUNT == 101
    assert geometry.sample_count() == 201
    assert geometry.warm_boundary_turn == 199
    windows = geometry.windows()
    assert len(windows) == 101
    assert windows[0] == (200, 300)
    assert windows[-1] == (300, 400)
    assert all(last - first == 100 for first, last in windows)
    assert all(len(range(first, last + 1)) == 101 for first, last in windows)
    assert list(geometry.sample_turns()) == list(range(200, 401))

    # The geometry IS the pinned row bound; a drift in either changes nothing
    # silently, because the record below is compared field by field.
    span = MEMORY_WINDOW_SPAN
    assert geometry.first_sample_turn == span.first_measured_turn
    assert geometry.last_sample_turn == span.last_measured_turn
    assert geometry.warm_boundary_turn == span.warm_boundary_turn
    assert geometry.sample_count() == span.sample_count
    assert geometry.window_width_turns == span.window_width_turns
    assert geometry.window_count == span.window_count

    # The shape record carries the memory identity and binds the digest.
    record = full.record()
    assert record["cell"] == CELL_SOAK_MEMORY
    assert record["tracemalloc"] == SOAK_TRACEMALLOC_ACTIVE
    assert record["memory_windows"] == geometry.record()
    assert record["slope_unit"] == MEMORY_SLOPE_UNIT
    assert record["slope_clamp"] == MEMORY_SLOPE_CLAMP
    assert "heartbeat_interval_ms" not in record
    assert record["fixture_hash"] == fixture["fixture_hash"] == FIXTURE_HASH
    assert record["workload_seed"] == WORKLOAD_SEED
    assert record["response_chars"] == FIXTURE_RESPONSE_CHARS
    latency_record = LATENCY_SOAK_FULL.record()
    assert latency_record["cell"] == CELL_SOAK_LATENCY
    assert latency_record["tracemalloc"] == SOAK_TRACEMALLOC_DISABLED
    assert latency_record["heartbeat_interval_ms"] == 1.0
    assert "memory_windows" not in latency_record

    assert MEMORY_SOAK_SHAPE_HASHES["memory-full"] == full.digest()
    assert MEMORY_SOAK_SHAPE_HASHES["memory-contract"] == (
        MEMORY_SOAK_CONTRACT.digest()
    )
    assert (
        replace(full, root_turns=399).digest()
        != MEMORY_SOAK_SHAPE_HASHES["memory-full"]
    )
    assert (
        replace(full, tracemalloc_mode=SOAK_TRACEMALLOC_DISABLED).digest()
        != (MEMORY_SOAK_SHAPE_HASHES["memory-full"])
    )

    # Gating: the memory predicate binds the cell, the tracing mode, the closed
    # geometry and the committed schedule; the latency predicate refuses it.
    assert memory_shape_gated(full) is True
    assert memory_shape_gated(MEMORY_SOAK_CONTRACT) is False
    assert memory_shape_gated(LATENCY_SOAK_FULL) is False
    assert not memory_shape_gated(replace(full, root_turns=399))
    assert not memory_shape_gated(replace(full, tracemalloc_mode="disabled"))
    assert not memory_shape_gated(replace(full, cell=CELL_SOAK_LATENCY))
    assert not memory_shape_gated(
        replace(full, memory_windows=MemoryWindows(200, 400, 100, 100))
    )
    assert full.gated is False
    assert MEMORY_SOAK_CONTRACT.gated is False
    assert LATENCY_SOAK_FULL.gated is True

    # The shortened shape runs the same machinery at the fixture's own scale.
    contract = MEMORY_SOAK_CONTRACT
    assert contract.history_messages == FIXTURE_TURN_WIDTH * 2
    assert contract.root_turns == 4
    assert contract.mode == SOAK_MODE_CONTRACT
    contract_geometry = contract.memory_windows
    assert contract_geometry is not None
    assert contract_geometry.sample_count() == 4
    assert contract_geometry.window_count == 3
    assert contract_geometry.warm_boundary_turn == 0

    # The G1 row: same cell, same producer, its own coverage unit and counts.
    spec = row_spec("G1")
    assert spec.cell == CELL_SOAK_MEMORY
    assert spec.producer_wp == "X3b"
    assert spec.unit == SLOPE_UNIT == "bytes/turn"
    assert spec.coverage_unit == COVERAGE_ROLLING_WINDOWS
    assert spec.measured_floor == spec.exact_measured == 101
    assert spec.unit_count_exact == {COUNT_RSS_SAMPLES: 201, COUNT_WINDOW_SLOPES: 101}
    assert spec.late_window == span
    assert spec.slope_unit == SLOPE_UNIT
    assert spec.gated_series_key == COUNT_WINDOW_SLOPES
    assert spec.warmup_floor == 10
    # The latency rows stay bound to their own cell and producer.
    for campaign_id in ("F1", "F2"):
        latency_row = row_spec(campaign_id)
        assert latency_row.cell == CELL_SOAK_LATENCY
        assert latency_row.producer_wp == "X3a"

    # The floors admit exactly the committed full-scale counts and nothing thin.
    exact = evaluate_run_floor(
        "G1",
        measured_ops=101,
        warmup_ops=span.warm_boundary_turn,
        unit_counts={COUNT_RSS_SAMPLES: 201, COUNT_WINDOW_SLOPES: 101},
    )
    assert exact.eligible
    thin = evaluate_run_floor(
        "G1",
        measured_ops=100,
        warmup_ops=span.warm_boundary_turn,
        unit_counts={COUNT_RSS_SAMPLES: 200, COUNT_WINDOW_SLOPES: 100},
    )
    assert not thin.eligible
    assert "ops_floor_measured" in thin.reasons
    assert "ops_floor_exact_measured" in thin.reasons
    assert f"coverage_exact_{COUNT_RSS_SAMPLES}" in thin.reasons
    assert f"coverage_exact_{COUNT_WINDOW_SLOPES}" in thin.reasons
    missing = evaluate_run_floor("G1", measured_ops=101, warmup_ops=199)
    assert f"coverage_missing_{COUNT_RSS_SAMPLES}" in missing.reasons
    assert f"coverage_missing_{COUNT_WINDOW_SLOPES}" in missing.reasons


def _assert_window_geometry_guards() -> None:
    """A geometry its windows cannot cover is rejected, never padded."""
    MEMORY_WINDOWS_FULL.validate()
    with pytest.raises(SoakError, match="window_width_invalid"):
        MemoryWindows(200, 201, 0, 2).validate()
    with pytest.raises(SoakError, match="window_count_invalid"):
        MemoryWindows(200, 200, 100, 0).validate()
    with pytest.raises(SoakError, match="window_span_invalid"):
        MemoryWindows(300, 200, 100, 1).validate()
    with pytest.raises(SoakError, match="window_geometry_invalid"):
        MemoryWindows(200, 400, 100, 100).validate()
    with pytest.raises(SoakError, match="window_geometry_invalid"):
        MemoryWindows(200, 201, 1, 2).validate()
    with pytest.raises(SoakError, match="window_geometry_invalid"):
        MemoryWindows(200, 398, 100, 101).validate()


def _assert_slope_math() -> None:
    """The window slope is least squares, signed, and clamped only for the gate."""
    turns = list(range(200, 301))
    ramp = [1_000_000.0 + _RAMP_BYTES_PER_TURN * (turn - 200) for turn in turns]
    assert least_squares_slope(turns, ramp) == pytest.approx(
        _RAMP_BYTES_PER_TURN, rel=1e-9
    )
    flat = [4_000_000.0 for _turn in turns]
    assert least_squares_slope(turns, flat) == 0.0
    shrinking = [4_000_000.0 - _RAMP_BYTES_PER_TURN * (turn - 200) for turn in turns]
    assert least_squares_slope(turns, shrinking) == pytest.approx(
        -_RAMP_BYTES_PER_TURN, rel=1e-9
    )
    with pytest.raises(SoakError, match="slope_shape_invalid"):
        least_squares_slope(turns, ramp[:-1])
    with pytest.raises(SoakError, match="slope_samples_insufficient"):
        least_squares_slope([200], [1.0])
    with pytest.raises(SoakError, match="slope_geometry_degenerate"):
        least_squares_slope([200, 200], [1.0, 2.0])

    # The signed value stays on the record; only the gated form is clamped.
    falling = WindowSlope(first_turn=200, last_turn=299, signed_bytes_per_turn=-8.5)
    record = falling.record()
    assert record["signed_bytes_per_turn"] == -8.5
    assert record["clamped_bytes_per_turn"] == 0.0
    assert falling.clamped() == 0.0
    rising = WindowSlope(first_turn=200, last_turn=299, signed_bytes_per_turn=32.25)
    assert rising.record()["clamped_bytes_per_turn"] == 32.25

    # Over the committed geometry: 101 slopes on a ramp, all of them the ramp.
    slopes = rolling_window_slopes(MEMORY_WINDOWS_FULL, _ramp(201))
    assert len(slopes) == 101
    assert slopes[0].first_turn == 200
    assert slopes[-1].last_turn == 400
    assert all(
        slope.signed_bytes_per_turn == pytest.approx(_RAMP_BYTES_PER_TURN, rel=1e-9)
        for slope in slopes
    )
    plateau = rolling_window_slopes(
        MEMORY_WINDOWS_FULL, dict.fromkeys(range(200, 401), 3_000_000)
    )
    assert len(plateau) == 101
    assert [slope.clamped() for slope in plateau] == [0.0] * 101
    # A plateaued baseline is exactly the degenerate-threshold premise: M = D = 0
    # and therefore T = 0, which dormants the row instead of gating it.
    ordered = sorted(slope.clamped() for slope in plateau)
    median = ordered[len(ordered) // 2]
    mad = sorted(abs(value - median) for value in ordered)[len(ordered) // 2]
    assert median == 0.0 and mad == 0.0

    # A window missing one of its samples yields NO slope for that window: the
    # first window is the only one carrying turn 200, so the other 100 survive.
    gapped = _ramp(201)
    del gapped[200]
    surviving = rolling_window_slopes(MEMORY_WINDOWS_FULL, gapped)
    assert len(surviving) == 100
    assert (surviving[0].first_turn, surviving[0].last_turn) == (201, 301)
    assert (surviving[-1].first_turn, surviving[-1].last_turn) == (300, 400)


def _assert_retained_set_is_reporting_only() -> None:
    """Five stamps are four diffs: a diff count cannot support a p95."""
    assert len(SOAK_CHECKPOINT_TURNS) == 5
    assert len(SOAK_CHECKPOINT_TURNS) - 1 == 4
    assert MEMORY_RETAINED_SET_GATE == "reporting_only"
    assert "diff count" in MEMORY_RETAINED_SET_NULL_REASON
    assert "p95" in MEMORY_RETAINED_SET_NULL_REASON


def _assert_contract_run(run: SoakMemoryRun) -> None:
    """The shortened memory run produced RSS coverage and refused to gate."""
    shape = MEMORY_SOAK_CONTRACT
    geometry = shape.memory_windows
    assert geometry is not None
    assert run.status == "ok"
    assert len(run.turns) == shape.root_turns == 4
    assert [turn.turn for turn in run.turns] == list(range(1, 5))

    # One RSS sample per settled turn, all of them readable.
    assert [sample.turn for sample in run.rss_samples] == list(range(1, 5))
    assert all(sample.rss_bytes is not None for sample in run.rss_samples)
    assert all(value > 0 for value in run.rss_bytes_by_turn().values())
    assert run.unavailable_rss_samples() == []
    assert set(run.rss_bytes_by_turn()) == set(range(1, 5))
    assert run.sampled_span_complete() is True

    # Tracing was active at every turn: the memory mode never runs blind.
    run.assert_tracemalloc_active()
    assert run.tracemalloc_observed() is True
    assert run.tracemalloc_samples and all(run.tracemalloc_samples)
    assert run.memory_gated is False

    # The closed geometry produced its three slopes, consistently clamped.
    slopes = run.window_slopes()
    assert len(slopes) == geometry.window_count == 3
    assert [(slope.first_turn, slope.last_turn) for slope in slopes] == [
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    signed = run.window_slopes_signed()
    clamped = run.window_slopes_clamped()
    assert len(signed) == len(clamped) == 3
    for measured, gated in zip(signed, clamped, strict=True):
        assert gated == max(0.0, measured)
    p95 = run.window_slope_p95()
    assert p95["count"] == 3
    assert p95["clamped"] is not None and p95["signed"] is not None
    assert "null_reason" not in p95

    # Chunks coverage carries both memory unit counts (plan §3).
    coverage = run.chunks_coverage()
    assert coverage["unit"] == COVERAGE_ROLLING_WINDOWS
    assert coverage["root_turns"] == 4
    assert coverage["measured_turns"] == 4
    assert coverage["warmup_turns"] == 0
    assert coverage["rss_samples"] == 4
    assert coverage["window_slopes"] == 3
    assert coverage["retained_set_checkpoint_diffs"] == 2
    assert coverage["child_bursts"] == 0
    assert coverage["child_turns"] == 0
    assert coverage["complete"] is True
    assert coverage["missing"] == []

    # Three structural stamps produced two diffs, reportable but never gated.
    assert [stamp["turn"] for stamp in run.retained_stamps] == [0, 2, 4]
    assert len(run.retained_diffs) == len(run.retained_stamps) - 1 == 2
    for diff in run.retained_diffs:
        assert isinstance(diff.net_bytes, int)
        assert isinstance(diff.top_growth, list)
    retained = run.retained_set_record()
    assert retained["gate"] == MEMORY_RETAINED_SET_GATE
    assert retained["p95_supported"] is False
    assert len(retained["diffs"]) == 2
    assert retained["stamps"] == list(run.retained_stamps)

    # The report is a memory report: RSS and slopes, never a latency section.
    report = run.report()
    assert report["cell"] == CELL_SOAK_MEMORY
    assert report["mode"] == SOAK_MODE_CONTRACT
    assert report["gated"] is False
    assert report["shape_hash"] == MEMORY_SOAK_SHAPE_HASHES["memory-contract"]
    assert report["fixture_hash"] == FIXTURE_HASH
    assert report["tracemalloc"]["active"] is True
    assert report["tracemalloc"]["retained_set_stamps"] == 3
    assert report["latency_gates"]["gated"] is False
    assert report["latency_gates"]["reason"]
    for latency_key in (
        "heartbeat",
        "late_turn_overheads_p95_ms",
        "invalid_late_samples",
        "segments",
        "segment_change",
    ):
        assert latency_key not in report
    memory = report["memory"]
    assert memory["windows"] == geometry.record()
    assert memory["sampled_turns"] == 4
    assert memory["unavailable_samples"] == []
    assert memory["slopes"]["unit"] == SLOPE_UNIT
    assert memory["slopes"]["clamp"] == SLOPE_CLAMP_RULE
    assert len(memory["slopes"]["signed"]) == 3
    assert len(memory["slopes"]["windows"]) == 3
    assert memory["p95"]["count"] == 3
    assert report["retained_set"]["p95_supported"] is False
    assert report["chunks_coverage"] == coverage
    assert report["growth"]["grew"] is True
    assert [checkpoint["turn"] for checkpoint in report["checkpoints"]] == [0, 2, 4]
    assert [turn["turn"] for turn in report["turns"]] == list(range(1, 5))
    assert "wall_ms" not in report["turns"][0]
    assert report["execution"]["status"] == "ok"
    assert report["execution"]["turns_settled"] == 4
    assert report["execution"]["stopped_early"] is False

    # The mode boundaries: no G1 record from a shortened run, and the inherited
    # F1/F2 path refuses a memory shape outright (no co-measured latency rows).
    with pytest.raises(SoakError, match="shape_not_committed"):
        run.g1_record()
    with pytest.raises(SoakError, match="shape_not_committed"):
        run.gated_records()
    thin = evaluate_run_floor(
        "G1",
        measured_ops=len(slopes),
        warmup_ops=coverage["warmup_turns"],
        unit_counts={
            COUNT_RSS_SAMPLES: coverage["rss_samples"],
            COUNT_WINDOW_SLOPES: coverage["window_slopes"],
        },
    )
    assert not thin.eligible

    # The shortened shape carries no cohort, and nothing outlived the teardown.
    scenario = run.scenario
    assert scenario.retained_child_sessions == len(run.bursts) == 0
    scenario.assert_child_session_records()
    scenario.assert_cleanup()
    assert scenario.registry._children == {}


async def _run_and_assert(session_root: Path) -> None:
    run = await run_memory_soak(MEMORY_SOAK_CONTRACT, session_root=session_root)
    _assert_contract_run(run)
    _assert_g1_record_assembly(run)


def _assert_g1_record_assembly(run: SoakMemoryRun) -> None:
    """The full-scale record path assembles a schema-valid G1 record.

    The RSS series here is a SYNTHETIC ramp over the real run's machinery: what
    is under test is the record path's exact counts, its late-window bounds and
    its signed-plus-clamped slope series (the full-scale measuring run itself is
    the opt-in external cell), and no measured G1 value is claimed.
    """
    run.shape = MEMORY_SOAK_FULL
    first = MEMORY_WINDOWS_FULL.first_sample_turn
    run.rss_samples = [
        RssSample(
            turn=turn,
            rss_bytes=1_000_000 + _RAMP_BYTES_PER_TURN * (turn - first),
            reason=None,
        )
        for turn in MEMORY_WINDOWS_FULL.sample_turns()
    ]
    record = run.g1_record()
    assert record["campaign_id"] == "G1"
    assert record["cell"] == CELL_SOAK_MEMORY
    assert record["metric_version"] == "chartreux.runtime_perf.v1"
    assert record["unit"] == SLOPE_UNIT
    assert record["mode"] == "measure"
    coverage = record["chunks_coverage"]
    assert coverage["unit"] == COVERAGE_ROLLING_WINDOWS
    assert coverage["measured_ops"] == 101
    assert coverage["warmup_ops"] == 199
    assert coverage["unit_counts"] == {COUNT_RSS_SAMPLES: 201, COUNT_WINDOW_SLOPES: 101}
    assert coverage["complete"] is True
    assert coverage["missing"] == []
    assert record["eligibility"] == {"eligible": True, "reasons": []}
    assert record["late_window"] == MEMORY_WINDOW_SPAN.as_record()
    assert len(record["samples"]) == 101
    signed = record["slopes"]["signed"]
    assert len(signed) == 101
    for measured, gated in zip(signed, record["samples"], strict=True):
        assert gated == max(0.0, measured)
        assert measured == pytest.approx(_RAMP_BYTES_PER_TURN, rel=1e-9)
    assert record["summary"]["count"] == 101
    assert record["summary"]["p95"] == max(record["samples"])
    assert record["summary"]["p95_signed"] is not None
    assert record["campaign"]["closure_status"] is None
    assert record["campaign"]["closure_note"] == MEMORY_OPEN_CAMPAIGN_NOTE
    # A run that kept no per-turn samples cannot fabricate a full-scale record.
    run.rss_samples = run.rss_samples[:200]
    with pytest.raises(SoakError, match="coverage_incomplete"):
        run.g1_record()


@pytest.mark.timeout(_CONTRACT_TIMEOUT_S)
def test_runtime_soak_memory_contract(tmp_path: Path) -> None:
    """The loaded memory soak holds its G1 shape, its coverage and its guards."""
    started = perf_counter()
    _assert_memory_definitions()
    _assert_window_geometry_guards()
    _assert_slope_math()
    _assert_retained_set_is_reporting_only()
    guard = EgressGuard(tmp_path / "egress").install()
    try:
        guard.assert_closed()
        asyncio.run(_run_and_assert(tmp_path / "sessions"))
        guard.assert_closed()
    finally:
        guard.restore()
    # The process is the caller's again: no socket hooks, no forced home, and no
    # allocation tracing left behind for a later latency node.
    assert not guard._installed
    assert not tracemalloc.is_tracing()
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"soak memory contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
