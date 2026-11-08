"""X8d contract node: the deterministic permission round-trip cell (E2).

The node proves the cell the E2 campaign measures through, not a stand-in: a
real dangerous-command approval is classified by the shell policy, asked
through the interaction broker, opened as a user-input callback on the live
projector, answered across the real transport pair, and applied to the waiting
tool decision. The only substitution is the deterministic fake responder
(plan §4 E2: "no human/network latency"), whose definition is committed as the
fake-responder fixture hash the row requires.

In-suite contract nodes run shortened fixtures (plan §5), so this node's
round-trip count is the contract scale and its run record is honestly
ineligible: a shortened run can never be mistaken for campaign data. The node
also pins the chunks-coverage fields E2 must populate (permission round trips
per run) and the WP6 row identity the cell reports against.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket
import sys
from time import perf_counter

import pytest

from tests.perf._runtime_permission import (
    FAKE_RESPONDER_FIXTURE_HASH,
    PERMISSION_ANSWER_LABEL,
    PERMISSION_CAMPAIGN_ID,
    PERMISSION_CELL,
    PERMISSION_COMMAND,
    PERMISSION_CONTRACT_MEASURED,
    PERMISSION_CONTRACT_WARMUPS,
    PERMISSION_COUNT_KEY,
    PERMISSION_COVERAGE_UNIT,
    PERMISSION_MEASURED_ROUND_TRIPS,
    PERMISSION_METRIC,
    PERMISSION_ORIGIN,
    PERMISSION_UNIT,
    PERMISSION_WARMUP_ROUND_TRIPS,
    PERMISSION_WINDOW,
    PermissionCell,
    PermissionCellResult,
    build_permission_run_record,
    fake_responder_fixture,
    fake_responder_fixture_digest,
    permission_cell_guard,
    prepare_permission_project,
)
from tests.perf._runtime_report import (
    CELL_PERMISSION,
    COUNT_PERMISSION_ROUND_TRIPS,
    COVERAGE_PERMISSION_ROUND_TRIPS,
    PERMISSION_ROUND_TRIP,
    ROW_SPECS,
    evaluate_run_floor,
)
from tests.perf._runtime_scenario import (
    CONFIG_HOME_ENV,
    PROVIDER_CREDENTIAL_NAMES,
    EgressGuard,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the permission cell is Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_PERMISSION_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_EgressProbe = "192.0.2.1"
_E2_PRODUCER_WP = "X8d"


def _assert_row_identity() -> None:
    """The cell reports exactly the WP6 E2 row the campaign gates on."""
    spec = ROW_SPECS[PERMISSION_CAMPAIGN_ID]
    assert PERMISSION_CAMPAIGN_ID == "E2"
    assert PERMISSION_CELL == CELL_PERMISSION == spec.cell
    assert PERMISSION_METRIC == PERMISSION_ROUND_TRIP == spec.metric
    assert PERMISSION_UNIT == spec.unit
    assert spec.producer_wp == _E2_PRODUCER_WP
    assert PERMISSION_COVERAGE_UNIT == COVERAGE_PERMISSION_ROUND_TRIPS
    assert PERMISSION_COUNT_KEY == COUNT_PERMISSION_ROUND_TRIPS
    # Appendix B E2 floors: >= 100 measured round trips after >= 10 warmups.
    assert spec.measured_floor == PERMISSION_MEASURED_ROUND_TRIPS == 100
    assert spec.warmup_floor == PERMISSION_WARMUP_ROUND_TRIPS == 10
    assert spec.required_unit_counts == (PERMISSION_COUNT_KEY,)
    assert PERMISSION_CONTRACT_WARMUPS < PERMISSION_WARMUP_ROUND_TRIPS
    assert PERMISSION_CONTRACT_MEASURED < PERMISSION_MEASURED_ROUND_TRIPS


def _assert_responder_fixture() -> None:
    """The fake responder is deterministic and pinned by its committed hash."""
    fixture = fake_responder_fixture()
    assert fake_responder_fixture_digest() == FAKE_RESPONDER_FIXTURE_HASH
    assert fixture["campaign_id"] == PERMISSION_CAMPAIGN_ID
    assert fixture["cell"] == PERMISSION_CELL
    assert fixture["metric"] == PERMISSION_METRIC
    assert fixture["command"] == PERMISSION_COMMAND
    assert fixture["answer_label"] == PERMISSION_ANSWER_LABEL
    assert fixture["origin"] == str(PERMISSION_ORIGIN) == "runtime_approval"
    assert fixture["warmup_round_trips"] == PERMISSION_WARMUP_ROUND_TRIPS
    assert fixture["measured_round_trips"] == PERMISSION_MEASURED_ROUND_TRIPS
    assert fixture["measurement_window"] == PERMISSION_WINDOW
    assert fixture["response_shape"]["streaming"] is False

    # Negative controls: the hash is the drift rule, so it must move when the
    # responder's command or its matching answer label changes.
    for key, value in (("command", "git push"), ("answer_label", "Deny")):
        drifted = dict(fixture)
        drifted[key] = value
        assert fake_responder_fixture_digest(drifted) != FAKE_RESPONDER_FIXTURE_HASH


def _connect(host: str) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect((host, 80))
    finally:
        probe.close()


def _assert_egress_guard(tmp_path: Path) -> EgressGuard:
    """The guard blocks egress, strips credentials and redirects the homes."""
    guard = permission_cell_guard(tmp_path / "egress")
    guard.assert_closed()
    assert guard.config_home.is_dir()
    assert Path(os.environ[CONFIG_HOME_ENV]) == guard.config_home
    assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    # Negative control: the block is address-selective, not a blanket kill.
    with pytest.raises(OSError, match="blocked"):
        _connect(_EgressProbe)
    return guard


def _assert_round_trips(result: PermissionCellResult) -> None:
    """Every round trip is a matched approval whose disposition was applied."""
    total = PERMISSION_CONTRACT_WARMUPS + PERMISSION_CONTRACT_MEASURED
    assert result.warmup_ops == PERMISSION_CONTRACT_WARMUPS
    assert result.measured_ops == PERMISSION_CONTRACT_MEASURED
    assert len(result.round_trips) == total
    assert result.applied_round_trips() == total
    assert result.executed_commands == total
    assert result.declined_responses == 0
    assert [trip.index for trip in result.round_trips] == list(range(total))

    for trip in result.round_trips:
        assert trip.applied
        assert trip.applied_status == "completed"
        # The request is a runtime approval for the committed command, and the
        # deterministic answer is one of the options the request offered.
        assert PERMISSION_COMMAND in trip.question
        assert PERMISSION_ANSWER_LABEL in trip.offered_labels
        assert trip.answer_label == PERMISSION_ANSWER_LABEL
        assert trip.elapsed_ms > 0.0
    assert len({trip.callback_id for trip in result.round_trips}) == total

    samples = result.samples_ms
    assert len(samples) == PERMISSION_CONTRACT_MEASURED
    assert samples == tuple(trip.elapsed_ms for trip in result.round_trips[1:])
    assert all(sample > 0.0 for sample in samples)
    assert result.setup_wall_ms > 0.0
    assert result.turn_wall_ms > 0.0

    summary = result.summary()
    assert summary["count"] == PERMISSION_CONTRACT_MEASURED
    assert summary["p50"] == min(samples)
    assert summary["p95"] == max(samples)
    assert summary["max"] == max(samples)


def _assert_chunks_coverage(result: PermissionCellResult) -> None:
    """E2 populates its per-run chunks coverage: permission round trips."""
    coverage = result.chunks_coverage()
    assert set(coverage) == {
        "unit",
        "measured_ops",
        "warmup_ops",
        "unit_counts",
        "complete",
        "missing",
    }
    assert coverage["unit"] == PERMISSION_COVERAGE_UNIT == "permission round trips"
    assert coverage["measured_ops"] == PERMISSION_CONTRACT_MEASURED
    assert coverage["warmup_ops"] == PERMISSION_CONTRACT_WARMUPS
    assert coverage["unit_counts"] == {
        PERMISSION_COUNT_KEY: PERMISSION_CONTRACT_MEASURED
    }
    assert coverage["complete"] is True
    assert coverage["missing"] == []

    record = result.run_record()
    assert record["campaign_id"] == PERMISSION_CAMPAIGN_ID
    assert record["cell"] == PERMISSION_CELL
    assert record["fixture_hash"] == FAKE_RESPONDER_FIXTURE_HASH
    assert record["measurement_window"] == PERMISSION_WINDOW
    assert record["chunks_coverage"] == coverage
    assert len(record["round_trips"]) == (
        PERMISSION_CONTRACT_WARMUPS + PERMISSION_CONTRACT_MEASURED
    )


def _assert_run_record(result: PermissionCellResult) -> None:
    """The shortened run's WP6 record is valid and honestly ineligible."""
    record = build_permission_run_record(
        result, run_id="permission-contract", run_start_utc="2026-11-08T00:00:00Z"
    )
    assert record["campaign_id"] == "E2"
    assert record["metric"] == PERMISSION_ROUND_TRIP
    assert record["unit"] == PERMISSION_UNIT
    assert record["cell"] == PERMISSION_CELL
    assert record["producer_wp"] == _E2_PRODUCER_WP
    assert record["fixture_hash"] == FAKE_RESPONDER_FIXTURE_HASH
    # The record carries the gated series at the schema's sample precision.
    assert record["samples"] == [round(sample, 4) for sample in result.samples_ms]
    assert record["chunks_coverage"]["unit_counts"] == {
        PERMISSION_COUNT_KEY: PERMISSION_CONTRACT_MEASURED
    }
    assert record["campaign"]["closure_status"] is None
    assert record["campaign"]["closure_note"]
    # A shortened in-suite run never claims eligibility for campaign data.
    assert record["eligibility"]["eligible"] is False
    assert set(record["eligibility"]["reasons"]) == {
        "ops_floor_measured",
        "ops_floor_warmup",
        f"coverage_floor_{PERMISSION_COUNT_KEY}",
    }

    # The committed full scale does meet the row's floors.
    full = evaluate_run_floor(
        PERMISSION_CAMPAIGN_ID,
        measured_ops=PERMISSION_MEASURED_ROUND_TRIPS,
        warmup_ops=PERMISSION_WARMUP_ROUND_TRIPS,
        unit_counts={PERMISSION_COUNT_KEY: PERMISSION_MEASURED_ROUND_TRIPS},
    )
    assert full.eligible is True
    assert full.reasons == ()


async def _exercise_cell(tmp_path: Path) -> None:
    project = prepare_permission_project(tmp_path / "project")
    cell = PermissionCell(
        project,
        warmups=PERMISSION_CONTRACT_WARMUPS,
        measured=PERMISSION_CONTRACT_MEASURED,
    )
    result = await cell.run()
    _assert_round_trips(result)
    _assert_chunks_coverage(result)
    _assert_run_record(result)


@pytest.mark.timeout(_PERMISSION_TIMEOUT_S)
def test_runtime_permission_contract(tmp_path: Path) -> None:
    """The cell's request/response round trips, coverage fields and record hold."""
    started = perf_counter()
    _assert_row_identity()
    _assert_responder_fixture()
    guard = _assert_egress_guard(tmp_path)
    try:
        asyncio.run(_exercise_cell(tmp_path))
        guard.assert_closed()
    finally:
        guard.restore()
    assert perf_counter() - started < _CONTRACT_WINDOW_S
