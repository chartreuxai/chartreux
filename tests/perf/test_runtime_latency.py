"""X1 contract node: end-to-end turn latency on both adapter paths.

Case-to-assertion map for this node (plan §3 X1 row, §4 rows A1-A4/B1-B2,
Appendix B rows A1-A4/B1-B2, §3 chunks coverage):

* A1  -- ``_assert_turn_records`` (raw wall, exact 30 ms injection, residual
         gated, raw_p95 and residual_p95 recorded separately).
* A2  -- ``_assert_ttfb_record`` (observed TTFB includes the 10 ms first byte).
* A3  -- ``_assert_gap_record`` (observed gaps include the 5 ms chunk gap, one
         per gap, >= 300 gaps per full-scale run by the row's floor).
* A4  -- ``_assert_heartbeat_records`` (1 ms scheduled heartbeat rows).
* B1/B2 -- ``_drive_loaded_cell`` (the 8-agent loaded cell produces both rows).
* Pilot-path matrix reports are observation-only -- ``_assert_pilot_matrix``.
* Injected-vs-application split -- asserted per turn and re-asserted on the
  synthetic negative-residual control, which must stay invalid and unclamped.
* Chunks coverage (turns / chunk gaps / heartbeat rows per run) --
  ``_assert_chunks_coverage`` on every run of the node.
* P2-memo-aware fidelity -- ``_assert_memo_context``.

Full-scale cells are external campaign runs (plan §5); this node runs the
shortened contract scale, so every run it produces is honestly ineligible as
baseline data (``cell_eligibility.baseline_eligible`` is False with the
``contract_scale_run`` reason), and the node asserts exactly that instead of a
pass that could be mistaken for eligible measurement.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import os
from pathlib import Path
import socket
import sys
from time import perf_counter
from typing import Any

import pytest

from tests.perf._runtime_latency import (
    CANONICAL_POINT,
    CANONICAL_POINT_KEY,
    GATED_ROWS_BY_CELL,
    HEARTBEAT_INTERVAL_MS,
    HEARTBEAT_WARMUP_ROWS,
    INVALID_SAMPLE_CAP,
    LOADED_CELL_POINT,
    LOADED_NO_INJECTION_REASON,
    MATRIX_POINTS,
    OBSERVATION_ONLY_REASON,
    PILOT_OBSERVATION_ONLY_REASON,
    HeadlessLatencyCell,
    InjectedDelayPoint,
    LatencyLog,
    TurnLatencySample,
    gated_point,
    matrix_point,
    observation_only_points,
    run_pilot_matrix,
)
from tests.perf._runtime_metrics import current_perf_paths
from tests.perf._runtime_pilot import PILOT_ADAPTER
from tests.perf._runtime_report import nearest_rank, validate_run_record
from tests.perf._runtime_scenario import (
    CELL_FRESH_HEADLESS,
    CELL_LOADED_HEADLESS,
    CONFIG_HOME_ENV,
    INJECTED_CHUNK_COUNT,
    INJECTED_CHUNK_GAP_MS,
    INJECTED_FIRST_BYTE_MS,
    LOADED_CHILD_AGENTS,
    LOADED_TOTAL_AGENTS,
    PROVIDER_CREDENTIAL_NAMES,
    SHORTENED_SCALE,
    WARMUP_TURNS,
    WORKLOAD_HASHES,
    EgressGuard,
    response_chunks,
    workload_spec,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the runtime latency cells are Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget inside a 2s
# hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_LATENCY_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_P95_FRACTION = 0.95
_EgressProbe = "192.0.2.1"
#: The fresh cell's contract geometry: ONE root agent and no child cohort, the
#: committed dimension of ``x1_fresh_headless_fb10_c4`` (SHORTENED_SCALE carries
#: a cohort for the loaded cell's own contract node). The turn count keeps the
#: event loop yielding long enough for the committed 10-row heartbeat warmup to
#: leave a measured window behind it; a turn costs single-digit milliseconds.
_FRESH_CONTRACT_SCALE = replace(
    SHORTENED_SCALE,
    name="latency_contract_fresh",
    root_turns=5,
    child_agents=0,
    burst_agents=0,
)
#: The loaded cell's contract geometry: one cohort only (its full scale is 7),
#: spread over enough turns for the same heartbeat window.
_LOADED_CONTRACT_SCALE = replace(
    SHORTENED_SCALE,
    name="latency_contract_loaded",
    root_turns=8,
    burst_interval_turns=7,
)
#: The observation-only points inject no delay at all, so their turns yield the
#: event loop far less than the canonical cell's 30 ms per turn; the probe run
#: therefore carries more turns to keep a measured heartbeat window behind the
#: committed 10-row warmup.
_PROBE_CONTRACT_SCALE = replace(
    _FRESH_CONTRACT_SCALE, name="latency_contract_probe", root_turns=12
)
_CONTRACT_TURNS = _FRESH_CONTRACT_SCALE.root_turns
_CONTRACT_MEASURED_TURNS = _CONTRACT_TURNS - 1
#: Gaps per turn of the committed c4 response shape (one per chunk boundary).
_GAPS_PER_TURN = INJECTED_CHUNK_COUNT - 1


# --------------------------------------------------------------------------
# Matrix and cell definitions
# --------------------------------------------------------------------------


def _assert_matrix_definitions() -> None:
    """The canonical cell is the only gated point; every other one says why."""
    assert CANONICAL_POINT.key == CANONICAL_POINT_KEY
    assert CANONICAL_POINT.first_byte_ms == INJECTED_FIRST_BYTE_MS == 10.0
    assert CANONICAL_POINT.chunk_gap_ms == INJECTED_CHUNK_GAP_MS == 5.0
    assert CANONICAL_POINT.chunk_count == INJECTED_CHUNK_COUNT == 4
    assert CANONICAL_POINT.total_delay_ms == 30.0
    assert CANONICAL_POINT.canonical is True
    assert CANONICAL_POINT.reason is None

    canonical = [point for point in MATRIX_POINTS if point.canonical]
    assert canonical == [CANONICAL_POINT]
    others = observation_only_points()
    assert others == tuple(
        point for point in MATRIX_POINTS if point is not CANONICAL_POINT
    )
    assert others and CANONICAL_POINT not in others
    for point in others:
        assert point.canonical is False
        assert point.reason == OBSERVATION_ONLY_REASON
        assert point.chunk_count == INJECTED_CHUNK_COUNT
        assert point.key != CANONICAL_POINT_KEY

    # The points are the matrix: distinct keys, distinct injection schedules.
    assert len({point.key for point in MATRIX_POINTS}) == len(MATRIX_POINTS)
    assert len({point.total_delay_ms for point in MATRIX_POINTS}) == len(MATRIX_POINTS)
    for point in MATRIX_POINTS:
        assert matrix_point(point.key) == point
        assert point.record()["total_delay_ms"] == point.total_delay_ms
    with pytest.raises(ValueError):
        matrix_point("fb99_c9x99")

    # The loaded cell carries no injection of its own: the load is the fixture.
    assert LOADED_CELL_POINT.first_byte_ms == 0.0
    assert LOADED_CELL_POINT.chunk_gap_ms == 0.0
    assert LOADED_CELL_POINT.total_delay_ms == 0.0
    assert LOADED_CELL_POINT.canonical is False
    assert LOADED_CELL_POINT.reason == LOADED_NO_INJECTION_REASON
    assert LOADED_CELL_POINT not in MATRIX_POINTS


def _assert_gated_configurations() -> None:
    """Only a cell's own gated configuration may produce gated row records."""
    assert gated_point(CELL_FRESH_HEADLESS) is CANONICAL_POINT
    assert gated_point(CELL_LOADED_HEADLESS) is LOADED_CELL_POINT
    with pytest.raises(ValueError):
        gated_point("not_a_latency_cell")

    assert GATED_ROWS_BY_CELL[CELL_FRESH_HEADLESS] == ("A1", "A2", "A3", "A4")
    assert GATED_ROWS_BY_CELL[CELL_LOADED_HEADLESS] == ("B1", "B2")

    fresh = workload_spec(CELL_FRESH_HEADLESS)
    loaded = workload_spec(CELL_LOADED_HEADLESS)
    # The freshly committed cell definitions the rows are anchored to.
    assert fresh.injected_turn_delay_ms == CANONICAL_POINT.total_delay_ms == 30.0
    assert loaded.injected_turn_delay_ms == LOADED_CELL_POINT.total_delay_ms == 0.0
    assert fresh.warmup_turns == loaded.warmup_turns == WARMUP_TURNS == 10
    # B1's binding dimension: 8 agents in total, root + 7 synchronized children.
    assert loaded.total_agents == LOADED_TOTAL_AGENTS == 8
    assert loaded.child_agents == LOADED_CHILD_AGENTS == 7
    assert loaded.digest() == WORKLOAD_HASHES[CELL_LOADED_HEADLESS]

    # The in-suite runs are shortened contract geometries, never full scale, and
    # the fresh ones keep the fresh cell's single-agent dimension.
    assert SHORTENED_SCALE.mode == "contract"
    assert _FRESH_CONTRACT_SCALE.mode == "contract"
    assert _FRESH_CONTRACT_SCALE.root_turns == _CONTRACT_TURNS
    assert _FRESH_CONTRACT_SCALE.child_agents == 0
    assert _FRESH_CONTRACT_SCALE.burst_agents == 0
    assert _LOADED_CONTRACT_SCALE.mode == "contract"
    assert _LOADED_CONTRACT_SCALE.child_agents == SHORTENED_SCALE.child_agents
    assert _LOADED_CONTRACT_SCALE.root_turns < loaded.root_turns

    assert HEARTBEAT_INTERVAL_MS == 1.0
    assert HEARTBEAT_WARMUP_ROWS == 10
    assert INVALID_SAMPLE_CAP == 0.10


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def _connect(host: str) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect((host, 80))
    finally:
        probe.close()


def _install_guard(tmp_path: Path) -> EgressGuard:
    """Fail-closed egress on the latency paths too (plan §1)."""
    guard = EgressGuard(tmp_path / "egress")
    guard.install()
    guard.assert_closed()
    assert Path(os.environ[CONFIG_HOME_ENV]) == guard.config_home
    assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    with pytest.raises(OSError, match="blocked"):
        _connect(_EgressProbe)
    return guard


def _assert_measurement_paths() -> None:
    """The run's measurement paths stay outside the repository (plan §1)."""
    paths = current_perf_paths()
    if paths.run_dir is not None:
        assert paths.repo_root not in paths.run_dir.parents
        assert paths.run_dir != paths.repo_root


# --------------------------------------------------------------------------
# Per-run assertions
# --------------------------------------------------------------------------


def _assert_turn_split(log: LatencyLog) -> None:
    """A1's raw/injected split, per turn and in aggregate."""
    assert log.point is CANONICAL_POINT
    assert len(log.turns) == _CONTRACT_TURNS
    measured = log.measured_turns()
    assert len(measured) == _CONTRACT_MEASURED_TURNS
    assert [turn.turn for turn in measured] == list(range(1, _CONTRACT_TURNS))

    for turn in log.turns:
        # The turn's exact injected delay is the canonical 30 ms, recorded
        # beside the raw wall and subtracted to the residual.
        assert turn.injected_ms == CANONICAL_POINT.total_delay_ms == 30.0
        assert turn.raw_ms > turn.injected_ms
        assert turn.residual_ms == turn.raw_ms - turn.injected_ms
        assert turn.valid is True
        assert turn.foreign_requests == 0
        assert turn.chunks == INJECTED_CHUNK_COUNT
        assert len(turn.chunk_gaps_ms) == _GAPS_PER_TURN
        # Observed values INCLUDE the injected delays by construction.
        assert turn.ttfb_ms is not None
        assert turn.ttfb_ms >= INJECTED_FIRST_BYTE_MS
        assert all(gap >= INJECTED_CHUNK_GAP_MS for gap in turn.chunk_gaps_ms)

    assert log.invalid_residuals() == []
    assert len(log.valid_residuals()) == _CONTRACT_MEASURED_TURNS
    assert log.foreign_requests() == 0
    summary = log.summary()
    assert summary["measured_turns"] == _CONTRACT_MEASURED_TURNS
    assert summary["invalid_samples"] == 0
    assert summary["raw_p95_ms"] > summary["residual_p95_ms"] > 0.0
    assert summary["ttfb_p95_ms"] >= INJECTED_FIRST_BYTE_MS
    assert summary["chunk_gap_p95_ms"] >= INJECTED_CHUNK_GAP_MS


def _assert_chunks_coverage(log: LatencyLog) -> dict[str, Any]:
    """Every run carries its turns / chunk gaps / heartbeat rows (§3)."""
    coverage = log.chunks_coverage()
    measured = log.measured_turns()
    gaps = log.measured_gaps()
    assert coverage["complete"] is True, coverage["missing"]
    assert coverage["missing"] == []
    assert coverage["unit"] == "turns"
    assert coverage["turns"] == len(measured) == len(log.turns) - log.warmup_turns
    assert coverage["warmup_turns"] == len(log.turns) - len(measured)
    assert coverage["chunk_gaps"] == len(gaps) == len(measured) * _GAPS_PER_TURN
    assert coverage["heartbeat_rows"] == len(log.measured_heartbeat()) > 0
    # The measured heartbeat window is at least as wide as the discarded one.
    assert coverage["heartbeat_rows"] >= coverage["heartbeat_warmup_rows"]
    assert coverage["heartbeat_interval_ms"] == HEARTBEAT_INTERVAL_MS
    assert coverage["requests"] >= coverage["turns"]
    return coverage


def _assert_memo_context(log: LatencyLog) -> None:
    """P2-memo-aware fidelity: the backend received the memoized request."""
    context = log.request_context
    assert isinstance(context, dict)
    assert context["memo_present"] is True
    assert context["memo_matches_request"] is True
    assert context["memo_messages"] == context["messages"]
    assert context["requests"] >= len(log.turns)

    probe = log.memo_probe
    assert isinstance(probe, dict)
    # Re-preparing the same live snapshot is an identity hit; the live version
    # re-read invalidates the memo once the transcript mutates.
    assert probe["identity_hit"] is True
    assert probe["live_reread_miss"] is True
    assert probe["first"] == probe["hit"]
    assert probe["mutated"] != probe["first"]


def _assert_common_record(record: dict[str, Any], log: LatencyLog, row: str) -> None:
    validate_run_record(record)
    assert record["campaign_id"] == row
    assert record["cell"] == log.cell
    assert record["mode"] == "measure"
    assert record["metric_version"] == "chartreux.runtime_perf.v1"
    assert record["fixture_hash"] == log.fixture_hash
    assert record["workload_hash"] == log.workload_hash
    # An open campaign records its closure reason, never a fabricated status.
    campaign = record["campaign"]
    assert campaign["campaign_id"] == row
    assert campaign["cell"] == log.cell
    assert campaign["closure_status"] is None
    assert campaign["closure_note"]
    coverage = record["chunks_coverage"]
    assert coverage["complete"] is True
    assert coverage["missing"] == []
    counts = coverage["unit_counts"]
    assert counts["turns"] == len(log.measured_turns())
    assert counts["chunk_gaps"] == len(log.measured_gaps())
    assert counts["heartbeat_rows"] == len(log.measured_heartbeat()) > 0
    # A shortened contract run carries its row floors as recorded reasons: it is
    # never presented as eligible baseline data.
    assert record["eligibility"]["eligible"] is False
    assert record["eligibility"]["reasons"]
    cell = record["cell_eligibility"]
    assert cell["floor_eligible"] is False
    assert cell["baseline_eligible"] is False
    assert cell["scale_mode"] == "contract"
    assert "contract_scale_run" in cell["reasons"]
    assert cell["invalid_cap_exceeded"] is False
    assert cell["invalid_samples"] == 0
    assert record["quiet"]["quiet"] in (True, False)


def _assert_turn_record(record: dict[str, Any], log: LatencyLog, row: str) -> None:
    """A1/B1: the gate compares the residual only; raw and injected stay apart."""
    _assert_common_record(record, log, row)
    assert record["gated_series"] == "residual_ms"
    measured = log.measured_turns()
    residuals = [round(turn.residual_ms, 4) for turn in measured]
    assert record["samples"] == residuals
    expected_p95 = nearest_rank(residuals, fraction=_P95_FRACTION)
    assert record["summary"]["p95"] == expected_p95
    assert record["residual_p95_ms"] == expected_p95
    assert record["residual_p95_ms"] > 0.0
    assert record["injected"]["subtracted"] is True
    assert record["injected"]["total_delay_ms"] == log.point.total_delay_ms
    assert record["injected"]["samples_ms"] == [log.point.total_delay_ms] * len(
        measured
    )
    if log.point.total_delay_ms > 0.0:
        # The split is real: the raw wall carries the injected delay.
        assert record["raw_p95_ms"] > record["residual_p95_ms"]
    else:
        # No injected delay in this cell: the residual IS the raw wall.
        assert record["raw_p95_ms"] == record["residual_p95_ms"]
    assert record["residual"]["invalid_samples_ms"] == []
    assert record["residual"]["clamped"] is False
    assert record["residual"]["valid_samples_ms"] == residuals
    assert record["chunks_coverage"]["unit"] == "turns"
    assert "ops_floor_measured" in record["eligibility"]["reasons"]


def _assert_ttfb_record(record: dict[str, Any], log: LatencyLog) -> None:
    """A2: TTFB is gated as observed, the injected first byte included."""
    _assert_common_record(record, log, "A2")
    samples = record["samples"]
    assert len(samples) == len(log.measured_turns())
    assert all(sample >= INJECTED_FIRST_BYTE_MS for sample in samples)
    assert record["chunks_coverage"]["measured_ops"] == len(samples)
    assert record["injected_included"] is True
    assert record["injected_first_byte_ms"] == INJECTED_FIRST_BYTE_MS
    assert record["delay_dominated"] is True


def _assert_gap_record(record: dict[str, Any], log: LatencyLog) -> None:
    """A3: one sample per gap, gated as observed, injected gap included."""
    _assert_common_record(record, log, "A3")
    samples = record["samples"]
    assert len(samples) == len(log.measured_gaps())
    assert len(samples) == len(log.measured_turns()) * _GAPS_PER_TURN
    assert record["chunks_coverage"]["unit_counts"]["chunk_gaps"] == len(samples)
    assert all(sample >= INJECTED_CHUNK_GAP_MS for sample in samples)
    assert record["gaps_per_turn"] == _GAPS_PER_TURN
    assert record["injected_included"] is True
    assert record["injected_chunk_gap_ms"] == INJECTED_CHUNK_GAP_MS
    # The row's full-scale floor is 300 gaps: 100 turns x 3 gaps per turn.
    assert record["chunks_coverage"]["measured_ops"] * _GAPS_PER_TURN >= len(samples)


def _assert_heartbeat_records(
    record: dict[str, Any], log: LatencyLog, row: str
) -> None:
    """A4/B2: the scheduled 1 ms heartbeat's per-row excess."""
    _assert_common_record(record, log, row)
    heartbeat = record["heartbeat"]
    assert heartbeat["interval_ms"] == HEARTBEAT_INTERVAL_MS
    assert heartbeat["warmup_rows"] == HEARTBEAT_WARMUP_ROWS
    assert heartbeat["rows"] == heartbeat["warmup_rows"] + heartbeat["measured_rows"]
    samples = record["samples"]
    assert len(samples) == heartbeat["measured_rows"] > 0
    assert record["chunks_coverage"]["unit"] == "heartbeat rows"
    assert record["chunks_coverage"]["measured_ops"] == len(samples)
    assert record["chunks_coverage"]["unit_counts"]["heartbeat_rows"] == len(samples)
    # A heartbeat row is a scheduling lateness, so the series is non-negative.
    assert all(sample >= 0.0 for sample in samples)
    assert record["summary"]["p95"] >= record["summary"]["p50"] >= 0.0


def _assert_negative_residual_is_invalid_not_clamped() -> None:
    """A negative corrected sample stays invalid; it is never clamped to zero."""
    sample = TurnLatencySample(
        turn=1,
        raw_ms=1.5,
        injected_ms=30.0,
        history_messages=3,
        requests=1,
        chunks=INJECTED_CHUNK_COUNT,
        ttfb_ms=10.5,
        chunk_gaps_ms=(5.0, 5.0, 5.0),
    )
    assert sample.residual_ms == -28.5
    assert sample.valid is False
    assert sample.record()["residual_ms"] == -28.5
    log = LatencyLog(
        cell=CELL_FRESH_HEADLESS,
        point=CANONICAL_POINT,
        warmup_turns=0,
        scale=SHORTENED_SCALE.dimensions(),
        turns=[sample],
    )
    assert log.invalid_residuals() == [-28.5]
    assert log.valid_residuals() == []
    assert log.summary()["residual_p95_ms"] is None
    assert log.summary()["invalid_samples"] == 1


async def _drive_observation_only(
    point: InjectedDelayPoint,
) -> tuple[LatencyLog, dict[str, Any]]:
    """A non-canonical matrix point reports, and may not produce row records."""
    cell = HeadlessLatencyCell(
        point=point, scale=_PROBE_CONTRACT_SCALE, run_id=f"x1-contract-{point.key}"
    )
    log = await cell.run()
    assert log.child_bursts == 0
    report = cell.observation_report()
    assert report["gated"] is False
    assert report["reason"] == OBSERVATION_ONLY_REASON
    assert report["injected_delay_point"]["key"] == point.key
    assert report["injected_delay_point"]["canonical"] is False
    _assert_chunks_coverage(log)
    # This zero-injection control point's residual IS its raw wall.
    for turn in log.turns:
        assert turn.injected_ms == point.total_delay_ms
        assert turn.residual_ms == turn.raw_ms
        assert turn.valid is True
    assert log.summary()["ttfb_p95_ms"] < CANONICAL_POINT.first_byte_ms
    # Observation-only means exactly that: no gated row records, ever.
    with pytest.raises(AssertionError, match="observation-only"):
        cell.build_records()
    return log, report


async def _drive_loaded_cell() -> tuple[LatencyLog, dict[str, dict[str, Any]]]:
    cell = HeadlessLatencyCell(
        cell=CELL_LOADED_HEADLESS,
        scale=_LOADED_CONTRACT_SCALE,
        run_id="x1-contract-loaded",
    )
    log = await cell.run()
    assert log.point is LOADED_CELL_POINT
    assert log.scale["mode"] == "contract"
    assert log.scale["child_agents"] == _LOADED_CONTRACT_SCALE.child_agents == 1
    # The committed burst schedule: one cohort every N turns, never at turn 0.
    expected_bursts = (
        _LOADED_CONTRACT_SCALE.root_turns - 1
    ) // _LOADED_CONTRACT_SCALE.burst_interval_turns
    assert log.child_bursts == expected_bursts == 1
    assert log.child_turns == expected_bursts * _LOADED_CONTRACT_SCALE.burst_agents
    for turn in log.turns:
        # No injected delay in the loaded cell: the load is the fixture.
        assert turn.injected_ms == 0.0
        assert turn.residual_ms == turn.raw_ms
        assert turn.valid is True
    records = cell.build_records()
    assert set(records) == {"B1", "B2"}
    _assert_chunks_coverage(log)
    _assert_turn_record(records["B1"], log, "B1")
    _assert_heartbeat_records(records["B2"], log, "B2")
    assert records["B1"]["injected"]["total_delay_ms"] == 0.0
    assert records["B1"]["residual"]["valid_samples_ms"] == [
        round(turn.residual_ms, 4) for turn in log.measured_turns()
    ]
    return log, records


def _assert_pilot_matrix(
    reports: list[dict[str, Any]], points: tuple[InjectedDelayPoint, ...]
) -> None:
    """Pilot reports are observation-only and carry their recorded reason."""
    assert len(reports) == len(points) == 2
    for report, point in zip(reports, points, strict=True):
        # Nothing on this adapter is gated: the gated rows are headless only.
        assert report["adapter"] == PILOT_ADAPTER
        assert report["gated"] is False
        assert report["reason"] == PILOT_OBSERVATION_ONLY_REASON
        assert report["cell"] == CELL_FRESH_HEADLESS
        assert report["workload_hash"] == WORKLOAD_HASHES[CELL_FRESH_HEADLESS]
        assert report["injected_delay_point"]["key"] == point.key
        assert report["injected_delay_point"]["total_delay_ms"] == point.total_delay_ms
        assert report["summary"]["raw_p95_ms"] is not None
        assert report["summary"]["residual_p95_ms"] is not None
        assert report["summary"]["invalid_samples"] == 0
        coverage = report["chunks_coverage"]
        assert coverage["complete"] is True
        assert coverage["missing"] == []
        assert coverage["chunks"] == coverage["measured_turns"] * INJECTED_CHUNK_COUNT
        assert coverage["flushes"] == coverage["chunks"]
        turns = report["turns"]
        assert [turn["turn"] for turn in turns] == [0, 1]
        for turn in turns:
            # The Pilot path carries the same injected schedule as the point.
            assert turn["injected_delay_ms"] == point.total_delay_ms
            assert turn["residual_ms"] == turn["raw_ms"] - point.total_delay_ms
            assert turn["valid"] is True
            assert turn["chunks"] == len(response_chunks(turn["turn"]))
            assert turn["flushes"] == turn["chunks"]
    # The canonical point carries the SAME 30 ms injected schedule on both
    # adapters, so the two residuals are comparable observations.
    assert reports[0]["injected_delay_point"]["total_delay_ms"] == (
        CANONICAL_POINT.total_delay_ms
    )


async def _exercise() -> None:
    _assert_negative_residual_is_invalid_not_clamped()

    canonical = HeadlessLatencyCell(
        scale=_FRESH_CONTRACT_SCALE, run_id="x1-contract-canonical"
    )
    log = await canonical.run()
    # The fresh cell is a single-agent cell: no cohort can interleave a request
    # into a measured turn's window, so A2's one-request-per-turn claim holds by
    # construction as well as by assertion.
    assert log.child_bursts == 0
    assert log.scale["child_agents"] == 0
    _assert_turn_split(log)
    _assert_chunks_coverage(log)
    _assert_memo_context(log)
    records = canonical.build_records()
    assert set(records) == {"A1", "A2", "A3", "A4"}
    _assert_turn_record(records["A1"], log, "A1")
    _assert_ttfb_record(records["A2"], log)
    _assert_gap_record(records["A3"], log)
    _assert_heartbeat_records(records["A4"], log, "A4")
    assert canonical.observation_report()["gated"] is True
    assert canonical.observation_report()["reason"] is None

    await _drive_observation_only(observation_only_points()[0])
    await _drive_loaded_cell()

    pilot_points = (CANONICAL_POINT, observation_only_points()[0])
    reports = await run_pilot_matrix(*pilot_points)
    _assert_pilot_matrix(reports, pilot_points)


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------


@pytest.mark.timeout(_LATENCY_TIMEOUT_S)
def test_runtime_latency_contract(tmp_path: Path) -> None:
    """Both adapter paths, the injected matrix, A1-A4 and B1-B2 hold."""
    started = perf_counter()
    _assert_matrix_definitions()
    _assert_gated_configurations()
    _assert_measurement_paths()
    guard = _install_guard(tmp_path)
    try:
        asyncio.run(_exercise())
        guard.assert_closed()
    finally:
        guard.restore()
    # After restore the process is the caller's again: no hooks, no forced home.
    assert not guard._installed
    assert socket.socket.connect is guard._original_connect
    assert os.environ.get(CONFIG_HOME_ENV) != str(guard.config_home)
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"runtime latency contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
