"""X2 contract node: transport observation and the B3/B4 notification producer.

The node proves the machinery the B3/B4 rows are collected with, on the real
WP3b scenario harness rather than a stand-in:

* every completed root turn's window closed on the projector's completed-state
  event, and its in-window notification count and UTF-8 byte total are the
  sink's send-time serialization record — B3/B4 add no second serialization;
* the transport boundary reports one request per model call with the committed
  chunk schedule, a positive TTFB and one inter-chunk gap per chunk boundary;
* the ``request_start_observer`` seam is used when a backend announces its
  transport start (probe) and the harness's own backend-advance entry is used
  when it does not (the deterministic scenario backend has no HTTP transport),
  with the in-scope production callback still delegated to;
* the retry/stall observation rides ``GenericBackend.on_retry`` over an offline
  ``httpx.MockTransport`` and stays observation-only;
* the produced B3/B4 run records validate against the WP6 schema, populate the
  chunks-coverage fields X2 owes (root turns counted, per-turn counts/bytes),
  and are honestly ineligible at the shortened in-suite scale — the only thing
  standing between them and eligibility is the row's op floor;
* the claim is within-backend with failover out of claim.

Negative controls keep the node from passing vacuously: an unannounced backend
must be labelled with the harness boundary, a window that never closed must
raise, an empty sample set must report the missing coverage field, and mutating
one turn's counted notifications must change the produced series.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import os
from pathlib import Path
import socket
import sys
from time import perf_counter, perf_counter_ns

import pytest

from chartreux.core.llm.backend.generic import request_start_observer
from tests.perf._runtime_report import COUNT_ROOT_TURNS, ROW_SPECS
from tests.perf._runtime_scenario import (
    CELL_LOADED_HEADLESS,
    CONFIG_HOME_ENV,
    PROVIDER_CREDENTIAL_NAMES,
    SHORTENED_SCALE,
    EgressGuard,
    NotificationWindow,
    RootTurnResult,
    response_chunks,
    workload_spec,
)
from tests.perf._runtime_transport import (
    B3_B4_DETERMINISM,
    B3_CAMPAIGN,
    B4_CAMPAIGN,
    CLAIM_SCOPE,
    FAILOVER_STANDING,
    HARNESS_ENTRY_BOUNDARY,
    MEASUREMENT_SERIALIZATION,
    NOTIFICATION_WINDOW_RULE,
    OBSERVATION_ONLY,
    OPEN_CAMPAIGN_NOTE,
    RETRY_SEAM,
    RETRY_STALL_REASON,
    ROOT_TURN_WINDOW_KIND,
    TRANSPORT_SCHEMA,
    TRANSPORT_START_BOUNDARY,
    WINDOW_CLOSE_EVENT,
    WINDOW_OPEN_EVENT,
    SeamProbeBackend,
    TransportCellRun,
    notification_coverage,
    notification_records,
    observe_transport,
    retry_stall_observation,
    root_turn_notifications,
    run_transport_cell,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the transport runtime cells are Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_TRANSPORT_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_EgressProbe = "192.0.2.1"

#: The canonical response shape: the c4 schedule, so four chunks per request.
_CHUNKS_PER_TURN = len(response_chunks(0))
_SEAM_PROBE_CHUNKS = 3
_SEAM_PROBE_GAP_MS = 5.0
_REQUIRED_MACHINE_FIELDS = ("cpu_model", "core_count", "python_version", "platform")


def _assert_window_and_row_pins() -> None:
    """The B3/B4 identity, window rule and coverage unit are the pinned ones."""
    assert NOTIFICATION_WINDOW_RULE == "turn_submission->projector_completed_state"
    assert WINDOW_OPEN_EVENT == "root_turn_user_message_submission"
    assert WINDOW_CLOSE_EVENT == "projector_completed_state"
    assert ROOT_TURN_WINDOW_KIND == "root_turn"
    assert TRANSPORT_SCHEMA == "chartreux.runtime_transport.v1"
    assert B3_CAMPAIGN == "B3"
    assert B4_CAMPAIGN == "B4"
    assert CLAIM_SCOPE == "within_backend"
    assert FAILOVER_STANDING == "out_of_claim"
    # The cell's committed workload carries the same window closure rule.
    assert workload_spec(CELL_LOADED_HEADLESS).notification_window == (
        NOTIFICATION_WINDOW_RULE
    )

    b3 = ROW_SPECS[B3_CAMPAIGN]
    b4 = ROW_SPECS[B4_CAMPAIGN]
    assert (b3.producer_wp, b4.producer_wp) == ("X2", "X2")
    assert b3.cell == b4.cell == CELL_LOADED_HEADLESS
    assert b3.coverage_unit == b4.coverage_unit == "root turns"
    assert COUNT_ROOT_TURNS in b3.required_unit_counts
    assert COUNT_ROOT_TURNS in b4.required_unit_counts
    assert b3.measured_floor == b4.measured_floor == 100
    assert (b3.unit, b4.unit) == ("events", "bytes")


def _connect(host: str) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect((host, 80))
    finally:
        probe.close()


def _assert_egress_guard(tmp_path: Path) -> EgressGuard:
    """The guard blocks egress, strips credentials and redirects every home."""
    guard = EgressGuard(tmp_path / "egress")
    guard.install()
    guard.assert_closed()
    assert guard.config_home.is_dir()
    assert Path(os.environ[CONFIG_HOME_ENV]) == guard.config_home
    assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    # Negative control: the block is address-selective, not a blanket kill.
    with pytest.raises(OSError, match="blocked"):
        _connect(_EgressProbe)
    return guard


async def _seam_probes() -> None:
    """The seam is used and delegated to; an unannounced backend is labelled."""
    announced = SeamProbeBackend(
        chunks=_SEAM_PROBE_CHUNKS, first_byte_ms=_SEAM_PROBE_GAP_MS
    )
    delegated: list[int] = []
    token = request_start_observer.set(lambda: delegated.append(1))
    try:
        with observe_transport(announced) as observer:
            chunks = [chunk async for chunk in announced.complete_streaming()]
    finally:
        request_start_observer.reset(token)

    assert len(chunks) == _SEAM_PROBE_CHUNKS
    assert announced.seam_firings == 1
    # Our recorder sits in front of the production callback and delegates to it.
    assert delegated == [1]
    request = observer.requests[0]
    assert request.boundary == TRANSPORT_START_BOUNDARY
    assert request.transport_starts == 1
    assert request.outcome == "closed"
    ttfb = request.ttfb_ms()
    assert ttfb is not None and ttfb >= _SEAM_PROBE_GAP_MS
    assert len(request.gap_ms()) == _SEAM_PROBE_CHUNKS - 1
    assert observer.record()["seam_requests"] == 1
    # The wrapper lived on the instance only and is removed on exit.
    assert "complete_streaming" not in vars(announced)

    # Negative control: no seam firing means the harness entry is the boundary
    # and the in-scope callback is never invoked.
    silent = SeamProbeBackend(
        chunks=_SEAM_PROBE_CHUNKS, first_byte_ms=0.0, fires_seam=False
    )
    untouched: list[int] = []
    token = request_start_observer.set(lambda: untouched.append(1))
    try:
        with observe_transport(silent) as silent_observer:
            silent_chunks = [chunk async for chunk in silent.complete_streaming()]
    finally:
        request_start_observer.reset(token)
    assert len(silent_chunks) == _SEAM_PROBE_CHUNKS
    assert untouched == []
    assert silent_observer.requests[0].boundary == HARNESS_ENTRY_BOUNDARY
    assert silent_observer.record()["seam_requests"] == 0


def _assert_transport_observation(cell: TransportCellRun) -> None:
    """The transport observer reports the committed request and chunk shape."""
    record = cell.observer.record()
    assert record["claim_scope"] == CLAIM_SCOPE
    assert record["gated"] is False
    assert record["standing"] == OBSERVATION_ONLY
    assert record["attribute"] == "complete_streaming"
    assert record["backend"] == "RootTurnBackend"
    assert record["requests"] == len(cell.observer.requests)
    # Every root turn streams through this backend; a child-completion injected
    # turn can add a request of its own, so the count is never below the turns.
    assert record["requests"] >= len(cell.scenario.turns) >= 2
    assert record["chunks"] == record["requests"] * _CHUNKS_PER_TURN
    assert len(record["ttfb_ms"]) == record["requests"]
    assert all(ttfb >= 0.0 for ttfb in record["ttfb_ms"])
    assert max(record["ttfb_ms"]) > 0.0
    assert len(record["gaps_ms"]) == record["requests"] * (_CHUNKS_PER_TURN - 1)

    for observation in record["observations"]:
        assert observation["outcome"] == "closed"
        assert observation["chunks"] == _CHUNKS_PER_TURN
        assert len(observation["gaps_ms"]) == _CHUNKS_PER_TURN - 1
        assert all(gap >= 0.0 for gap in observation["gaps_ms"])
        assert observation["duration_ms"] is not None

    # The deterministic backend announces no transport start, so the harness
    # entry is the boundary and nothing here is a retried transport attempt.
    assert record["seam_requests"] == 0
    assert record["harness_entry_requests"] == record["requests"]
    assert record["transport_starts"] == 0
    assert record["retried_requests"] == 0
    assert record["stall_ms"] == []
    # The observation was installed on the cell's own backend instance only and
    # is gone once the cell closed, so it cannot leak into the next run.
    assert "complete_streaming" not in vars(cell.scenario._backend)


def _assert_notification_accounting(cell: TransportCellRun) -> None:
    """B3/B4 read the send-time serialization record; no second serialization."""
    scenario = cell.scenario
    samples = cell.samples
    assert len(samples) == len(scenario.turns) == scenario.scale.root_turns

    coverage = cell.notification_coverage()
    assert coverage["unit"] == "root turns"
    assert coverage["root_turns"] == len(samples)
    assert coverage["notification_windows"] == len(samples)
    assert coverage["measured_root_turns"] == 0
    assert coverage["warmup_root_turns"] == len(samples)
    assert coverage["complete"] is True
    assert coverage["missing"] == []

    # The scenario's own coverage block agrees with X2's on every shared field.
    scenario_coverage = scenario.chunks_coverage()
    shared = (
        "root_turns",
        "warmup_root_turns",
        "measured_root_turns",
        "notification_windows",
        "notifications",
        "notification_bytes",
        "notifications_per_root_turn",
        "notification_bytes_per_root_turn",
    )
    for field_name in shared:
        assert coverage[field_name] == scenario_coverage[field_name], field_name

    windows = {turn.turn_id: turn.window for turn in scenario.turns}
    for sample in samples:
        window = windows[sample.turn_id]
        assert window.kind == ROOT_TURN_WINDOW_KIND
        assert not window.open
        assert window.duration_ms is not None
        assert sample.notifications > 0
        assert sample.notification_bytes > 0
        # The measured totals ARE the send-time record, not a second dump.
        assert sample.notifications == len(window.methods)
        assert sample.notification_bytes == window.wire_bytes

    # One serialization per notification, and the client end received it.
    assert len(scenario.delivered_notifications) == len(scenario.sink.serialized)
    serialized_bytes = sum(record.wire_bytes for record in scenario.sink.serialized)
    in_window_bytes = sum(window.wire_bytes for window in scenario.sink.windows)
    # Children notify too, and their notifications land outside the windows.
    assert in_window_bytes == coverage["notification_bytes"]
    assert in_window_bytes < serialized_bytes

    # Negative control: a window that never closed has no B3/B4 sample at all.
    open_window = NotificationWindow(
        kind=ROOT_TURN_WINDOW_KIND, turn_id="never-closed", opened_ns=perf_counter_ns()
    )
    unfinished = RootTurnResult(
        turn=0,
        turn_id="never-closed",
        status="completed",
        wall_ms=1.0,
        injected_delay_ms=0.0,
        window=open_window,
        history_messages=0,
    )
    with pytest.raises(AssertionError, match="never closed"):
        root_turn_notifications([unfinished], warmup_turns=0)

    # Negative control: the coverage field X2 owes is reported missing when
    # there is no counted root turn to report it for.
    empty = notification_coverage([])
    assert empty["root_turns"] == 0
    assert empty["missing"] == [COUNT_ROOT_TURNS]
    assert empty["complete"] is False


async def _assert_records(cell: TransportCellRun) -> None:
    """The produced B3/B4 records are schema-valid and honestly floor-limited."""
    records = cell.notification_records(run_id="x2-contract")
    assert set(records) == {B3_CAMPAIGN, B4_CAMPAIGN}
    for campaign_id, record in records.items():
        spec = ROW_SPECS[campaign_id]
        assert record["campaign_id"] == campaign_id
        assert record["cell"] == CELL_LOADED_HEADLESS
        assert record["metric"] == spec.metric
        assert record["unit"] == spec.unit
        assert record["producer_wp"] == "X2"
        assert record["mode"] == "measure"
        assert record["fixture_hash"] == cell.spec.fixture_hash
        coverage = record["chunks_coverage"]
        assert coverage["unit"] == "root turns"
        assert coverage["complete"] is True
        assert coverage["missing"] == []
        assert coverage["unit_counts"] == {COUNT_ROOT_TURNS: len(cell.samples)}
        assert coverage["measured_ops"] == 0
        assert coverage["warmup_ops"] == len(cell.samples)
        # The shortened in-suite scale discards every turn as warmup.
        assert record["samples"] == []
        assert record["eligibility"]["eligible"] is False
        assert record["eligibility"]["reasons"] == [
            "ops_floor_measured",
            "ops_floor_warmup",
            "coverage_floor_root_turns",
            "no_gated_samples",
        ]
        campaign = record["campaign"]
        assert campaign["closure_status"] is None
        assert campaign["closure_note"] == OPEN_CAMPAIGN_NOTE
        for field_name in _REQUIRED_MACHINE_FIELDS:
            assert record["machine"][field_name] is not None, field_name

    # Series wiring: with no warmup discarded, every counted turn is measured
    # and the row's series is exactly the per-turn counts (B3) / bytes (B4).
    counted = root_turn_notifications(cell.scenario.turns, warmup_turns=0)
    series_records = notification_records(
        counted,
        run_id="x2-contract-series",
        run_start_utc=cell.run_start_utc,
        fixture_hash=cell.spec.fixture_hash,
    )
    assert series_records[B3_CAMPAIGN]["samples"] == [
        float(sample.notifications) for sample in counted
    ]
    assert series_records[B4_CAMPAIGN]["samples"] == [
        float(sample.notification_bytes) for sample in counted
    ]
    for record in series_records.values():
        assert record["chunks_coverage"]["measured_ops"] == len(counted)
        assert record["chunks_coverage"]["warmup_ops"] == 0
        assert record["eligibility"]["reasons"] == [
            "ops_floor_measured",
            "ops_floor_warmup",
            "coverage_floor_root_turns",
        ]

    # Negative control: a turn's counted notifications actually drive the series.
    mutated = [replace(counted[0], notifications=counted[0].notifications + 1)]
    mutated.extend(counted[1:])
    mutated_records = notification_records(
        mutated, run_id="x2-contract-mutated", run_start_utc=cell.run_start_utc
    )
    assert (
        mutated_records[B3_CAMPAIGN]["samples"]
        != (series_records[B3_CAMPAIGN]["samples"])
    )


def _assert_within_backend_claim(cell: TransportCellRun) -> None:
    """The claim is one backend instance; failover is explicitly out of claim."""
    record = cell.record()
    assert record["schema"] == TRANSPORT_SCHEMA
    assert record["cell"] == CELL_LOADED_HEADLESS
    assert record["scale"]["mode"] == "contract"
    assert record["window"]["rule"] == NOTIFICATION_WINDOW_RULE
    assert record["window"]["workload_window"] == NOTIFICATION_WINDOW_RULE
    assert record["window"]["opens_at"] == WINDOW_OPEN_EVENT
    assert record["window"]["closes_at"] == WINDOW_CLOSE_EVENT
    assert record["claim"]["scope"] == CLAIM_SCOPE
    assert record["claim"]["failover"] == FAILOVER_STANDING
    assert record["claim"]["backend"] == type(cell.scenario._backend).__name__
    assert record["transport"]["claim_scope"] == CLAIM_SCOPE

    properties = record["properties"]
    assert properties["b3_b4_determinism"] == B3_B4_DETERMINISM
    assert "expected D over the five run-p95s is 0" in properties["b3_b4_determinism"]
    assert "max(6D, 0.20 x M)" in properties["b3_b4_determinism"]
    assert properties["measurement_serialization"] == MEASUREMENT_SERIALIZATION
    assert properties["failover"] == FAILOVER_STANDING
    assert properties["retry_stall"] == RETRY_STALL_REASON
    assert [sample["turn"] for sample in record["root_turns"]] == [
        turn.turn for turn in cell.scenario.turns
    ]


async def _assert_retry_stall() -> None:
    """The retry/stall seam is observed and stays observation-only."""
    observer = await retry_stall_observation()
    record = observer.record()
    assert record["seam"] == RETRY_SEAM
    assert record["gated"] is False
    assert record["standing"] == OBSERVATION_ONLY
    assert record["reason"] == RETRY_STALL_REASON
    assert record["retries"] == 2
    assert [event["category"] for event in record["events"]] == [
        "RATE_LIMITED",
        "SERVER_ERROR",
    ]
    assert [event["detail"] for event in record["events"]] == ["HTTP 429", "HTTP 503"]
    assert len(record["stall_ms"]) == record["retries"] - 1
    assert all(stall >= 0.0 for stall in record["stall_ms"])


async def _exercise(tmp_path: Path) -> None:
    """Run the shortened cell and assert every contract the node pins."""
    cell = await run_transport_cell(
        scale=SHORTENED_SCALE, session_root=tmp_path / "sessions"
    )
    cell.scenario.assert_cleanup()
    assert cell.scenario.retained_child_sessions == cell.scenario.scale.burst_agents
    await _seam_probes()
    _assert_transport_observation(cell)
    _assert_notification_accounting(cell)
    await _assert_records(cell)
    _assert_within_backend_claim(cell)
    await _assert_retry_stall()


@pytest.mark.timeout(_TRANSPORT_TIMEOUT_S)
def test_runtime_transport_contract(tmp_path: Path) -> None:
    """Transport observation and the B3/B4 producer hold their contracts."""
    started = perf_counter()
    _assert_window_and_row_pins()
    guard = _assert_egress_guard(tmp_path)
    try:
        asyncio.run(_exercise(tmp_path))
        guard.assert_closed()
    finally:
        guard.restore()
    # After restore the process is the caller's again: no hooks, no forced home.
    assert not guard._installed
    assert socket.socket.connect is guard._original_connect
    assert os.environ.get(CONFIG_HOME_ENV) != str(guard.config_home)
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"transport contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
