"""WP4 contract node: the Pilot (Textual) adapter path and cross-adapter parity.

The node proves the second adapter path the stage-2 cells measure through, and
that it drives the SAME committed workload as the headless path: the delivered
transcript of both adapters produces one identical workload fingerprint. The
Pilot side is a real Textual application under the production stylesheet — the
committed prompt mounted as the production user message, the committed response
streamed frame by frame through the production assistant widget — so the D1
flush records come from a live render path, not a stub.

Also pinned here: the EXISTING r12 idle cells (a 60 s requested window on both
adapters, gated on the achieved window within a 75 s budget and a kill deadline
of at least twice it). They are verified, never duplicated.

Negative controls keep the node from passing vacuously: the headless transcript
must match the committed response before parity is even considered, a drifted
turn breaks the fingerprint, a drifted shared topology label breaks the
structural comparison, and the egress guard must block a non-loopback connect.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import socket
import sys
from time import perf_counter
from typing import Any

import pytest

from chartreux.cli.textual_ui.widgets.messages import StreamingMessageBase
from tests.perf._runtime_fixture import (
    FIXTURE_HASH,
    FIXTURE_RESPONSE_CHARS,
    FIXTURE_SEED,
    FIXTURE_TURNS,
    response_content,
)
from tests.perf._runtime_pilot import (
    CONTRACT_TURNS,
    CROSS_ADAPTER_SHARED_LABELS,
    HEADLESS_ADAPTER,
    IDLE_CELL_ADAPTERS,
    IDLE_KILL_DEADLINE_S,
    IDLE_PHASES,
    IDLE_REQUESTED_WINDOW_S,
    IDLE_WINDOW_BUDGET_S,
    IDLE_WINDOW_TOLERANCE_S,
    PILOT_ADAPTER,
    PILOT_PARITY_SCALE,
    PILOT_SCENARIO_SCHEMA,
    PilotScenario,
    WorkloadTurn,
    assert_cross_adapter_topology,
    assert_workload_parity,
    evaluate_idle_window,
    headless_workload_turns,
    idle_cell,
    make_workload_turn,
    workload_fingerprint,
)
from tests.perf._runtime_scenario import (
    CELL_FRESH_HEADLESS,
    CONFIG_HOME_ENV,
    PROVIDER_CREDENTIAL_NAMES,
    SCENARIO_METRIC_VERSION,
    WARMUP_TURNS,
    WORKLOAD_HASHES,
    EgressGuard,
    build_headless_scenario,
    response_chunks,
    root_user_content,
    workload_spec,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the Pilot runtime cells are Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_PILOT_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

#: Enough warmup flushes to prove the D1 producer splits warmup from measured
#: samples on this path, while leaving most flushes measured.
_CONTRACT_WARMUP_FLUSHES = 2
_FULL_SCALE_TURNS = 400
_INJECTED_CHUNK_COUNT = 4
_EgressProbe = "192.0.2.1"


def _assert_workload_definitions() -> None:
    """The Pilot path drives the committed workload definition, unchanged."""
    scenario = PilotScenario()
    spec = scenario.spec
    # The workload definition is adapter-independent: the same cell, fixture
    # hash and committed digest the headless cells measure.
    assert spec.cell == CELL_FRESH_HEADLESS
    assert spec.fixture_hash == FIXTURE_HASH
    assert spec.fixture_spec["seed"] == FIXTURE_SEED
    assert spec.digest() == WORKLOAD_HASHES[CELL_FRESH_HEADLESS]
    assert spec.digest() == workload_spec(CELL_FRESH_HEADLESS).digest()
    assert spec.root_turns == FIXTURE_TURNS == _FULL_SCALE_TURNS
    assert spec.warmup_turns == WARMUP_TURNS == 10
    assert spec.response_chars == FIXTURE_RESPONSE_CHARS
    assert spec.response_chunks == _INJECTED_CHUNK_COUNT

    # The in-suite run is the shortened contract geometry: the plan §5 shape
    # minus the child cohort, and explicitly never eligible baseline data.
    assert scenario.scale is PILOT_PARITY_SCALE
    assert scenario.scale.mode == "contract"
    assert scenario.scale.name == "pilot_parity"
    assert scenario.scale.root_turns == CONTRACT_TURNS
    assert scenario.scale.child_agents == 0
    assert scenario.scale.burst_agents == 0

    # The committed response is split into the canonical c4 schedule and the
    # chunks reassemble to exactly the committed content.
    for turn in (0, 1):
        chunks = response_chunks(turn)
        assert len(chunks) == spec.response_chunks
        assert "".join(chunks) == response_content(turn)
        assert all(chunks)
        assert len(root_user_content(turn)) == spec.root_user_chars

    # Negative control: the digest moves when the committed seed moves.
    drifted = workload_spec(CELL_FRESH_HEADLESS)
    object.__setattr__(drifted, "seed", drifted.seed + 1)
    assert drifted.digest() != WORKLOAD_HASHES[CELL_FRESH_HEADLESS]

    # A child cohort is never a Pilot streaming transcript.
    with pytest.raises(ValueError):
        PilotScenario(scale=replace(PILOT_PARITY_SCALE, child_agents=1))


def _assert_idle_cells() -> None:
    """The r12 idle cells are the EXISTING definition, on both adapters."""
    assert IDLE_REQUESTED_WINDOW_S == 60.0
    assert IDLE_WINDOW_TOLERANCE_S == 15.0
    assert IDLE_WINDOW_BUDGET_S == 75.0
    assert IDLE_KILL_DEADLINE_S == 150.0
    assert IDLE_KILL_DEADLINE_S >= 2 * IDLE_WINDOW_BUDGET_S
    assert IDLE_PHASES == ("startup", "steady", "drain_teardown", "idle")
    assert IDLE_CELL_ADAPTERS == (HEADLESS_ADAPTER, PILOT_ADAPTER)

    headless_cell = idle_cell(HEADLESS_ADAPTER)
    pilot_cell = idle_cell(PILOT_ADAPTER)
    # One definition, two adapters: the Pilot idle cell is the headless cell's
    # numbers, so no second idle cell is introduced here.
    assert replace(headless_cell, adapter=PILOT_ADAPTER) == pilot_cell
    assert pilot_cell.adapter == PILOT_ADAPTER
    assert pilot_cell.gated_quantity == "measured_window"
    assert pilot_cell.phase == "idle"
    assert pilot_cell.kill_deadline_s >= 2 * pilot_cell.budget_s

    ok = evaluate_idle_window(IDLE_REQUESTED_WINDOW_S)
    assert (ok.status, ok.ok) == ("ok", True)
    assert ok.record()["gated_quantity"] == "measured_window"
    # The budget, not the requested window, is the gate; equality passes.
    assert evaluate_idle_window(IDLE_WINDOW_BUDGET_S).ok is True
    over = evaluate_idle_window(IDLE_WINDOW_BUDGET_S + 0.001)
    assert over.ok is False
    assert over.status == "over_budget"
    assert over.reasons == ("measured_window_over_budget",)
    # A window that never completed is incomplete, never ok.
    incomplete = evaluate_idle_window(12.0, completed=False)
    assert (incomplete.status, incomplete.ok) == ("incomplete", False)
    assert incomplete.reasons == ("window_not_completed",)

    with pytest.raises(ValueError):
        idle_cell("app_server")
    with pytest.raises(ValueError):
        evaluate_idle_window(-1.0)


def _connect(host: str) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect((host, 80))
    finally:
        probe.close()


def _install_guard(tmp_path: Path) -> EgressGuard:
    """Fail-closed egress on the Pilot path too (plan §1)."""
    guard = EgressGuard(tmp_path / "egress")
    guard.install()
    guard.assert_closed()
    assert Path(os.environ[CONFIG_HOME_ENV]) == guard.config_home
    assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    with pytest.raises(OSError, match="blocked"):
        _connect(_EgressProbe)
    return guard


async def _drive_headless() -> tuple[list[WorkloadTurn], dict[str, Any]]:
    """Run the shortened workload through the real headless adapter."""
    scenario = await build_headless_scenario(
        CELL_FRESH_HEADLESS, scale=PILOT_PARITY_SCALE
    )
    try:
        await scenario.run_workload()
        turns = headless_workload_turns(
            list(scenario.root_loop.messages), turns=CONTRACT_TURNS
        )
        topology = scenario.topology(projector_live=True)
        record = scenario.run_record()
    finally:
        await scenario.close()
    scenario.assert_cleanup()
    assert scenario.registry._children == {}

    # The headless transcript must be the committed response itself; parity is
    # only meaningful when the reference side is faithful.
    for turn in turns:
        committed = make_workload_turn(
            turn.turn,
            user_text=root_user_content(turn.turn),
            assistant_text="".join(response_chunks(turn.turn)),
        )
        assert turn == committed, f"headless turn {turn.turn} drifted from the workload"
    assert [turn.turn for turn in turns] == [0, 1]
    assert record["workload_hash"] == WORKLOAD_HASHES[CELL_FRESH_HEADLESS]
    return turns, topology


async def _drive_pilot() -> tuple[list[WorkloadTurn], dict[str, Any], str]:
    """Run the same shortened workload through the live Pilot adapter."""
    observer_before = StreamingMessageBase._flush_write_buffer
    pilot = PilotScenario(warmup_flushes=_CONTRACT_WARMUP_FLUSHES)
    await pilot.run_workload()

    # Instrumentation is installed and removed around the run, never left on
    # the production widget for the next node.
    assert StreamingMessageBase._flush_write_buffer is observer_before

    assert [turn.turn for turn in pilot.turns] == [0, 1]
    for turn in pilot.turns:
        # One frame per committed chunk: the streaming shape is production's.
        assert turn.chunks == len(response_chunks(turn.turn))
        assert turn.flushes == turn.chunks
        assert turn.wall_ms > 0.0

    record = pilot.run_record()
    assert record["schema"] == PILOT_SCENARIO_SCHEMA
    assert record["metric_version"] == SCENARIO_METRIC_VERSION
    assert record["metric_version"] == "chartreux.runtime_perf.v1"
    assert record["adapter"] == PILOT_ADAPTER
    assert record["fixture_hash"] == FIXTURE_HASH
    assert record["workload_hash"] == WORKLOAD_HASHES[CELL_FRESH_HEADLESS]
    return pilot.workload_turns(), record, pilot.workload_fingerprint()


def _assert_d1_producer(record: dict[str, Any]) -> None:
    """The D1 record on this path is the WP2 producer's, with real samples."""
    d1 = record["d1"]
    assert isinstance(d1, dict)
    assert d1["campaign_id"] == "D1"
    assert d1["metric"] == "render_flush_p95_ms"
    assert d1["cell"] == "pilot_streaming_fresh"
    assert d1["metric_version"] == SCENARIO_METRIC_VERSION

    coverage = d1["chunks_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["unit"] == "render flushes"
    assert coverage["completed_turns"] == CONTRACT_TURNS
    assert coverage["warmup_flushes"] == _CONTRACT_WARMUP_FLUSHES
    assert coverage["empty_flushes"] == 0
    assert coverage["skipped_flushes"] == 0
    measured = coverage["measured_flushes"]
    assert isinstance(measured, int)
    assert measured == CONTRACT_TURNS * _INJECTED_CHUNK_COUNT - _CONTRACT_WARMUP_FLUSHES

    samples = d1["samples_ms"]
    assert isinstance(samples, list)
    assert len(samples) == measured
    assert all(sample > 0 for sample in samples)
    # p50 is reported alongside the gated p95 (D1 row special notes).
    assert d1["render_flush_p50_ms"] is not None
    assert d1["render_flush_p95_ms"] is not None
    assert d1["render_flush_p95_ms"] >= d1["render_flush_p50_ms"] > 0

    # A shortened contract run fails the pinned D1 floor honestly.
    assert d1["eligibility"] == {
        "eligible": False,
        "reasons": ["ops_floor_completed_turns", "ops_floor_measured_flushes"],
    }

    pilot_coverage = record["chunks_coverage"]
    assert isinstance(pilot_coverage, dict)
    assert pilot_coverage["complete"] is True
    assert pilot_coverage["missing"] == []
    assert pilot_coverage["turns"] == CONTRACT_TURNS
    assert pilot_coverage["chunks"] == CONTRACT_TURNS * _INJECTED_CHUNK_COUNT
    assert pilot_coverage["measured_flushes"] == measured


def _assert_fingerprint_is_load_bearing(turns: list[WorkloadTurn]) -> None:
    """Negative control: a drifted delivered turn changes the fingerprint."""
    workload_hash = WORKLOAD_HASHES[CELL_FRESH_HEADLESS]
    baseline = workload_fingerprint(CELL_FRESH_HEADLESS, workload_hash, turns)
    drifted = list(turns)
    last = turns[-1]
    drifted[-1] = replace(last, assistant_digest=last.assistant_digest[:-1] + "0")
    assert workload_fingerprint(CELL_FRESH_HEADLESS, workload_hash, drifted) != baseline
    with pytest.raises(ValueError):
        workload_fingerprint(CELL_FRESH_HEADLESS, workload_hash, [])


async def _assert_cross_adapter_parity() -> None:
    headless_turns, headless_topology = await _drive_headless()
    pilot_turns, record, pilot_fingerprint = await _drive_pilot()

    # CROSS-ADAPTER FINGERPRINT PARITY: the same workload, the same fingerprint
    # — and it is computed from what each adapter actually delivered.
    fingerprint = assert_workload_parity(
        cell=CELL_FRESH_HEADLESS,
        workload_hash=WORKLOAD_HASHES[CELL_FRESH_HEADLESS],
        headless_turns=headless_turns,
        pilot_turns=pilot_turns,
    )
    assert fingerprint == pilot_fingerprint
    assert record["workload_fingerprint"] == fingerprint
    _assert_fingerprint_is_load_bearing(headless_turns)

    # The adapters differ structurally only where the adapter itself differs.
    assert_cross_adapter_topology(headless_topology, record["topology"])
    topology = record["topology"]
    assert isinstance(topology, dict)
    for label in CROSS_ADAPTER_SHARED_LABELS:
        assert topology[label] == headless_topology[label]
    assert topology["adapter"] == PILOT_ADAPTER
    # No app-server projector on this path: the transcript is the live surface,
    # and the flush samples above are what it rendered.
    assert topology["projector_live"] is False

    _assert_d1_producer(record)

    # Negative controls: the parity checks fail on real drift.
    with pytest.raises(AssertionError):
        assert_workload_parity(
            cell=CELL_FRESH_HEADLESS,
            workload_hash=WORKLOAD_HASHES[CELL_FRESH_HEADLESS],
            headless_turns=headless_turns,
            pilot_turns=pilot_turns[:-1],
        )
    with pytest.raises(AssertionError):
        assert_cross_adapter_topology(
            headless_topology, {**topology, "total_agents": 8}
        )
    with pytest.raises(AssertionError):
        assert_cross_adapter_topology(
            headless_topology, {**topology, "adapter": HEADLESS_ADAPTER}
        )


@pytest.mark.asyncio
@pytest.mark.timeout(_PILOT_TIMEOUT_S)
async def test_pilot_scenario_contract(tmp_path: Path) -> None:
    """The Pilot path holds its workload, parity, D1 and idle-cell contracts."""
    started = perf_counter()
    _assert_workload_definitions()
    _assert_idle_cells()
    guard = _install_guard(tmp_path)
    try:
        await _assert_cross_adapter_parity()
        guard.assert_closed()
    finally:
        guard.restore()
    # After restore the process is the caller's again: no hooks, no forced home.
    assert not guard._installed
    assert socket.socket.connect is guard._original_connect
    assert os.environ.get(CONFIG_HOME_ENV) != str(guard.config_home)
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"pilot scenario contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
