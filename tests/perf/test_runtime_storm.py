"""X8b contract node: the synchronized completion-storm cell (row E1).

The node proves the machinery the E1 baseline campaign measures through, not a
stand-in: eight children are launched into a real app-server registry behind a
synchronized barrier, released together, and awaited until all eight
completions have been delivered and the cohort's finalization tasks settled.
The cell's floor is 100 measured storms after 10 warmup storms, one fresh
cleaned cohort per storm; in-suite the storm count is shortened to a single
measured storm (plan §5) and the cohort stays at the cell's full eight agents.

Protecting the node from passing vacuously:

* the cohort's committed geometry and its digest are pinned, and a drifted
  geometry must change the digest;
* the run's record plumbing is exercised at the full 110-storm shape, where the
  floors are met, and at a 109-storm shape, where they are not;
* a repeated cohort (a reused agent id) is refused, so "fresh cohort per storm"
  is a checked invariant rather than a claim;
* the residual check must observe an all-zero in-memory completion history
  before the storm, after the storm and after the cleaning pass;
* the real storm's run record must be ineligible — one storm is far below the
  floor — with the ops and coverage reasons recorded, never a silent pass.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket
import sys
from time import perf_counter

import pytest

from tests.perf._runtime_report import COUNT_STORMS, nearest_rank, row_spec
from tests.perf._runtime_scenario import (
    CONFIG_HOME_ENV,
    PROVIDER_CREDENTIAL_NAMES,
    EgressGuard,
)
from tests.perf._runtime_storm import (
    RESIDUAL_FIELDS,
    STORM_AGENTS,
    STORM_CAMPAIGN_ID,
    STORM_CELL,
    STORM_CLEANING_DRAIN,
    STORM_COHORT_HISTORY_MESSAGES,
    STORM_COHORTS_PER_STORM,
    STORM_MEASURED_FLOOR,
    STORM_MEASURED_STORMS,
    STORM_SYNC_STARTED_BEFORE_RELEASE,
    STORM_WARMUP_STORMS,
    StormError,
    StormRecord,
    StormRun,
    assert_clean,
    build_storm_harness,
    cohort_residual,
    cohort_spec,
    empty_residual,
    rebuild_warmups,
    run_storms,
    storm_samples,
    storm_scale,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the storm runtime cells are Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window. The node drives a real 8-agent
# registry cohort, so its timeout marker keeps the matrix-class floor.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_STORM_TIMEOUT_S = 24

#: The committed cohort definition's digest; drift in any dimension moves it.
_COHORT_DIGEST = "fb68e1f0c416ec5c7d92907ec981db5af5c2492c307444d8fba4d5f772eaa39a"

_EGRESS_PROBE = "192.0.2.1"
_P95_FRACTION = 0.95
_RECORD_PRECISION_DIGITS = 4


def _synthetic_records(
    *, warmups: int, measured: int, base_ms: float = 20.0
) -> list[StormRecord]:
    """Build storm rows for record-plumbing coverage (no timing claim).

    These rows exercise the run record's floors, coverage fields and summary;
    the samples are placeholders and never describe a measurement.
    """
    records: list[StormRecord] = []
    for index in range(warmups + measured):
        records.append(
            StormRecord(
                index=index,
                cohort_size=STORM_AGENTS,
                completed=STORM_AGENTS,
                drain_ms=base_ms + index * 0.25,
                construct_ms=1.0,
                warmup=index < warmups,
                agent_ids=tuple(
                    f"agent-{index * STORM_AGENTS + position}"
                    for position in range(STORM_AGENTS)
                ),
                residual_before=empty_residual(),
            )
        )
    return records


def _synthetic_run(*, warmups: int, measured: int) -> StormRun:
    run = StormRun(warmup_storms=warmups)
    for record in _synthetic_records(warmups=warmups, measured=measured):
        run.add(record)
    return run


def _assert_cell_definition() -> None:
    """The cohort geometry, the row identity and the digest pin."""
    spec = cohort_spec()
    assert spec.agents == STORM_AGENTS == 8
    assert spec.history_messages == STORM_COHORT_HISTORY_MESSAGES
    assert spec.synchronization == STORM_SYNC_STARTED_BEFORE_RELEASE
    assert spec.cleaning == STORM_CLEANING_DRAIN
    assert spec.fixture_hash
    assert STORM_COHORTS_PER_STORM == 1
    assert spec.digest() == _COHORT_DIGEST

    # Every storm is one 8-child cohort; only the storm count shortens in-suite.
    scale = storm_scale()
    assert scale.child_agents == STORM_AGENTS
    assert scale.burst_agents == STORM_AGENTS
    assert scale.history_messages == STORM_COHORT_HISTORY_MESSAGES
    assert scale.mode == "contract"

    # The cell identity comes from the gated row, never from an invented one.
    row = row_spec(STORM_CAMPAIGN_ID)
    assert row.cell == STORM_CELL == "storm_8child"
    assert row.metric == "completion_storm_drain_p95_ms"
    assert row.unit == "ms"
    assert row.producer_wp == "X8b"
    assert row.measured_floor == STORM_MEASURED_FLOOR == STORM_MEASURED_STORMS
    assert row.warmup_floor == STORM_WARMUP_STORMS
    assert dict(row.unit_count_floors) == {COUNT_STORMS: STORM_MEASURED_FLOOR}

    # Negative control: a drifted geometry must move the digest.
    drifted = cohort_spec()
    object.__setattr__(drifted, "agents", drifted.agents + 1)
    assert drifted.digest() != _COHORT_DIGEST


def _assert_record_plumbing() -> None:
    """The 110-storm shape meets the floors; the 109-storm shape records why not."""
    full = _synthetic_run(warmups=STORM_WARMUP_STORMS, measured=STORM_MEASURED_STORMS)
    coverage = full.chunks_coverage()
    assert coverage["unit"] == "storms"
    assert coverage["storms_per_run"] == 110
    assert coverage["measured_storms"] == 100
    assert coverage["warmup_storms"] == 10
    assert coverage["cohorts_per_run"] == 110
    assert coverage["agents_per_cohort"] == STORM_AGENTS
    assert len(coverage["drain_ms_per_storm"]) == 110
    assert coverage["complete"] is True
    assert coverage["missing"] == []
    assert coverage["cohort_digest"] == _COHORT_DIGEST

    record = full.run_record(
        run_id="storm-contract-full",
        run_start_utc="2026-11-08T00:00:00Z",
        closure_note="contract coverage: campaign left open",
    )
    assert record["campaign_id"] == STORM_CAMPAIGN_ID
    assert record["cell"] == STORM_CELL
    assert record["metric"] == "completion_storm_drain_p95_ms"
    assert record["unit"] == "ms"
    assert record["producer_wp"] == "X8b"
    assert record["chunks_coverage"]["unit"] == "storms"
    assert record["chunks_coverage"]["measured_ops"] == 100
    assert record["chunks_coverage"]["warmup_ops"] == 10
    assert record["chunks_coverage"]["unit_counts"][COUNT_STORMS] == 100
    assert record["chunks_coverage"]["complete"] is True
    assert record["eligibility"] == {"eligible": True, "reasons": []}
    assert len(record["samples"]) == STORM_MEASURED_STORMS
    assert record["summary"]["count"] == STORM_MEASURED_STORMS
    assert record["summary"]["p95"] == pytest.approx(
        nearest_rank(storm_samples(full.measured), fraction=_P95_FRACTION),
        abs=10**-_RECORD_PRECISION_DIGITS,
    )
    section = record["storm"]
    assert section["cell"] == STORM_CELL
    assert section["measured_storms"] == 100
    assert section["warmup_storms"] == 10
    assert section["cohorts"] == 110
    assert section["fresh_cohort_per_storm"] is True
    assert section["cohort_digest"] == _COHORT_DIGEST
    assert len(section["storms"]) == 110
    assert [row["warmup"] for row in section["storms"]] == (
        [True] * STORM_WARMUP_STORMS + [False] * STORM_MEASURED_STORMS
    )
    assert all(row["residual_before"] == empty_residual() for row in section["storms"])
    assert all(row["cohort_size"] == STORM_AGENTS for row in section["storms"])

    # One storm short of the floor: the record is ineligible and says why.
    short = _synthetic_run(
        warmups=STORM_WARMUP_STORMS, measured=STORM_MEASURED_STORMS - 1
    )
    short_record = short.run_record(
        run_id="storm-contract-short",
        run_start_utc="2026-11-08T00:00:00Z",
        closure_note="contract coverage: campaign left open",
    )
    assert short_record["eligibility"]["eligible"] is False
    reasons = short_record["eligibility"]["reasons"]
    assert "ops_floor_measured" in reasons
    assert f"coverage_floor_{COUNT_STORMS}" in reasons
    assert short_record["storm"]["measured_storms"] == 99

    # The warmup boundary is a run-record property; re-marking keeps the order.
    remarked = rebuild_warmups(full, warmup_storms=5)
    assert len(remarked.measured) == 105
    assert [record.index for record in remarked.records] == list(range(110))

    # A repeated cohort is refused: a fresh cohort per storm is an invariant.
    repeated = StormRecord(
        index=110,
        cohort_size=STORM_AGENTS,
        completed=STORM_AGENTS,
        drain_ms=20.0,
        construct_ms=1.0,
        warmup=False,
        agent_ids=full.records[0].agent_ids,
        residual_before=empty_residual(),
    )
    with pytest.raises(StormError, match="reused cohort agents"):
        full.add(repeated)

    # So is a storm that arrives out of order or leaves residual history.
    with pytest.raises(StormError, match="out of order"):
        StormRun(warmup_storms=0).add(
            StormRecord(
                index=3,
                cohort_size=STORM_AGENTS,
                completed=STORM_AGENTS,
                drain_ms=20.0,
                construct_ms=1.0,
                warmup=False,
                agent_ids=tuple(f"agent-late-{position}" for position in range(8)),
                residual_before=empty_residual(),
            )
        )
    leftover = empty_residual()
    leftover["run_results"] = 1
    with pytest.raises(StormError, match="residual completion history"):
        assert_clean(leftover, where="negative control")
    assert set(leftover) == set(RESIDUAL_FIELDS)


def _connect(host: str) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect((host, 80))
    finally:
        probe.close()


async def _exercise_storm() -> None:
    """Run one real storm and assert its cohort, drain and record."""
    scenario = await build_storm_harness()
    try:
        assert scenario.projector is None
        run = await run_storms(scenario, warmup_storms=0, measured_storms=1)

        assert len(run.records) == 1
        storm = run.records[0]
        assert storm.index == 0
        assert storm.warmup is False
        assert storm.cohort_size == STORM_AGENTS
        assert storm.completed == STORM_AGENTS
        assert len(storm.agent_ids) == STORM_AGENTS
        assert len(set(storm.agent_ids)) == STORM_AGENTS
        assert storm.drain_ms > 0.0
        assert storm.construct_ms > 0.0
        assert storm.residual_before == empty_residual()
        # The barrier served exactly one request per child of the cohort.
        assert scenario._burst_backend.completed == STORM_AGENTS
        assert len(scenario._burst_backend.context_chars) == STORM_AGENTS

        # The cleaning pass left no completion history behind.
        assert cohort_residual(scenario) == empty_residual()
        assert scenario.registry._children == {}
        assert scenario.projector is None

        coverage = run.chunks_coverage()
        assert coverage["storms_per_run"] == 1
        assert coverage["measured_storms"] == 1
        assert coverage["warmup_storms"] == 0
        assert coverage["cohorts_per_run"] == 1
        assert coverage["complete"] is True

        record = run.run_record(
            run_id="storm-contract-single",
            run_start_utc="2026-11-08T00:00:00Z",
            closure_note="contract coverage: campaign left open",
        )
        assert record["campaign_id"] == STORM_CAMPAIGN_ID
        assert record["cell"] == STORM_CELL
        assert record["chunks_coverage"]["unit"] == "storms"
        assert record["chunks_coverage"]["measured_ops"] == 1
        assert record["chunks_coverage"]["unit_counts"][COUNT_STORMS] == 1
        assert record["samples"][0] == pytest.approx(
            storm.drain_ms, abs=10**-_RECORD_PRECISION_DIGITS
        )
        assert record["summary"]["count"] == 1
        assert record["summary"]["p95"] == pytest.approx(
            record["samples"][0], abs=10**-_RECORD_PRECISION_DIGITS
        )
        # One storm is far below the floor: the record is ineligible, with the
        # ops and coverage reasons preserved rather than a silent pass.
        assert record["eligibility"]["eligible"] is False
        reasons = record["eligibility"]["reasons"]
        assert "ops_floor_measured" in reasons
        assert f"coverage_floor_{COUNT_STORMS}" in reasons
        row = record["storm"]["storms"][0]
        assert row["storm"] == 0
        assert row["completed"] == STORM_AGENTS
        assert row["warmup"] is False
        assert row["residual_before"] == empty_residual()
        assert record["storm"]["cohort_digest"] == _COHORT_DIGEST
    finally:
        await scenario.close()
    scenario.assert_cleanup()
    assert scenario.registry._children == {}


@pytest.mark.timeout(_STORM_TIMEOUT_S)
def test_runtime_storm_contract(tmp_path: Path) -> None:
    """The storm cell holds its cohort, synchronization, drain and record."""
    started = perf_counter()
    # Plan §1: the timeout marker covers at least twice the node's budget.
    assert _STORM_TIMEOUT_S >= 2 * _CONTRACT_BUDGET_S
    _assert_cell_definition()
    _assert_record_plumbing()

    guard = EgressGuard(tmp_path / "egress").install()
    try:
        guard.assert_closed()
        assert Path(os.environ[CONFIG_HOME_ENV]) == guard.config_home
        assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
        with pytest.raises(OSError, match="blocked"):
            _connect(_EGRESS_PROBE)
        asyncio.run(_exercise_storm())
        guard.assert_closed()
    finally:
        guard.restore()
    assert not guard._installed
    assert socket.socket.connect is guard._original_connect
    assert os.environ.get(CONFIG_HOME_ENV) != str(guard.config_home)

    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"storm contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
