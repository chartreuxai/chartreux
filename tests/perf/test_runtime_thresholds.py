"""X7a contract node: deterministic eligibility and threshold construction.

The node pins the activation construction of the runtime gate. ACTIVATION is
built from the COMMITTED BASELINE SET ALONE: for every registry row the
committed run-p95s of its cell's promoted baseline are extracted, reduced to
M = median and D = MAD, and turned into T = M + max(6D, 0.20M). A row ACTIVATES
only when its campaign is decision-ready (``eligible_complete``, exactly five
selected runs) AND its T is finite and positive; otherwise it is DORMANT with
the FIRST applicable reason of the pinned order
machine_mismatch -> insufficient_baseline_data -> missing_determinants ->
degenerate_threshold. Run ineligibility stays a run-record reason and a
threshold comparison stays a comparison outcome: neither is ever an activation
disposition.

All fixtures are synthetic committed baselines built through X5a's validated
baseline layer — no cell is executed and nothing is measured, so the node stays
inside the plan's in-suite contract-node window. The two pinned artifacts are
read byte-exact and checked against their plan §7 sha256 first. Malformed or
ambiguous input must halt: every such case is asserted to raise, never to
produce a discretionary DORMANT.

X7b's node sits beside it and pins the reporting layer: one fresh comparison
run's marked states render into the WP6 report as an additive section under the
pinned row-first precedence (DORMANT > COMPARISON-INELIGIBLE > threshold
labels), a machine context that differs in ANY of the four required fields
emits a PERF-BLOB machine-mismatch record through the WP1 path guard and
records DORMANT machine_mismatch, and a label its own facts do not support is
refused rather than rendered.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any

import pytest

from tests.perf._runtime_baselines import (
    COMPARISON_LABEL_PRECEDENCE,
    DIRECTION_LOWER_IS_BETTER,
    INSUFFICIENT_DATA_REASON,
    MAX_ATTEMPTS,
    REASON_DEGENERATE_THRESHOLD,
    REASON_INVALID_SAMPLE_CAP,
    REASON_MACHINE_MISMATCH,
    REASON_MISSING_DETERMINANTS,
    REASON_ORDER,
    REASON_RUN_INELIGIBILITY,
    REASON_THRESHOLD_FAIL,
    REASON_THRESHOLD_PASS,
    SELECTED_RUNS,
    THRESHOLD_RULE,
    build_baseline,
    build_baseline_run,
    build_comparison,
    cell_baseline_spec,
    gated_cells,
    quiet_stamp,
    quiet_window,
    shared_run,
    spec_from_baseline,
    threshold_value,
    validate_baseline,
)
from tests.perf._runtime_metrics import REPORT_ENV, RUN_DIR_ENV, repository_root
from tests.perf._runtime_report import (
    CELL_FRESH_HEADLESS,
    CELL_PERMISSION,
    CELL_SOAK_MEMORY,
    CELL_STORM,
    CLOSURE_ELIGIBLE_COMPLETE,
    CLOSURE_INSUFFICIENT_DATA,
    COMPARISON_STATE_DORMANT,
    COMPARISON_STATE_FAIL,
    COMPARISON_STATE_INELIGIBLE,
    COMPARISON_STATE_PASS,
    COMPARISON_STATE_PRECEDENCE,
    COMPARISON_STATE_RULE,
    COMPARISON_STATES_SECTION,
    GATED_CAMPAIGN_IDS,
    PERF_BLOB_KIND,
    PERF_BLOB_MACHINE_MISMATCH,
    PERF_BLOB_SECTION,
    REPORT_SCHEMA,
    ROW_SPECS,
    SLOPE_CLAMP_RULE,
    ReportSchemaError,
    build_report,
    report_schema,
    reporting_extension,
    validate_perf_blob_section,
    validate_report,
    validate_reporting_sections,
)
from tests.perf._runtime_thresholds import (
    ACTIVATION_INDETERMINATE,
    ACTIVE_TEST_REQUIRED_PHRASES,
    DETERMINANTS_SHA256,
    DORMANCY_REASONS,
    DORMANCY_TIERS,
    NON_DORMANCY_REASONS,
    PERF_BLOB_FILENAME,
    PINNED_REASON_TIERS,
    REGISTRY_SHA256,
    RUN_RECORD_TIER,
    STATE_ACTIVE,
    STATE_DORMANT,
    THRESHOLD_TIER,
    THRESHOLDS_SCHEMA_ID,
    UNMEASURED_REASON,
    ThresholdError,
    active_test_rule,
    build_comparison_states,
    construct_threshold_table,
    emit_perf_blob,
    load_activation_inputs,
    machine_mismatch_blob,
    parse_reason_order,
    perf_blob_section,
    reporting_sections,
    validate_state_section,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="the runtime threshold contract is Linux-only machinery",
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_THRESHOLD_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S
#: The reporting node plans the same 1s budget inside the same 2s hard window.
_REPORTING_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_PERF_DIR = Path(__file__).resolve().parent
_REGISTRY_FILENAME = "runtime_metric_registry.json"
_DETERMINANTS_FILENAME = "runtime_metric_determinants.json"

_CORES = 12
_UNSET: Any = object()

#: The committed machine context a baseline cell is collected on.
_MACHINE: dict[str, Any] = {
    "cpu_model": "AMD Ryzen 9 5900X 12-Core Processor",
    "core_count": _CORES,
    "python_version": "3.12.13",
    "platform": "Linux-7.3.0-9-generic-x86_64-with-glibc2.44",
}

#: The committed dimensions of a cell run, as a capture records them.
_DIMENSIONS: dict[str, Any] = {
    "root_turns": 100,
    "history_messages": 4096,
    "child_agents": 0,
    "injected_first_byte_ms": 10.0,
    "injected_chunk_count": 4,
    "injected_chunk_gap_ms": 5.0,
    "mode": "measure",
}

#: Each fixture row's floor counts, at exactly its floor values.
_RECIPE: dict[str, tuple[int, int, dict[str, int]]] = {
    "A1": (100, 10, {}),
    "A2": (100, 10, {}),
    "A3": (100, 10, {"chunk_gaps": 300}),
    "A4": (1000, 10, {}),
    "E2": (100, 10, {"permission_round_trips": 100}),
    "G1": (101, 10, {"rss_samples": 201, "window_slopes": 101}),
}

#: Per-run bases of the decision-ready fixtures: the run-p95s become
#: 2, 3, 4, 5, 6, so M = 4, D = 1 and T = 10 exactly (dyadic values).
_BASES: dict[str, float] = {f"run-{index:02d}": float(index) for index in range(1, 6)}


# ---------------------------------------------------------------------------
# Fixture construction (synthetic committed baselines; nothing is measured)
# ---------------------------------------------------------------------------


def _series(length: int, base: float = 1.0) -> list[float]:
    """Return a dyadic series whose nearest-rank p95 is exactly ``base + 1.0``."""
    return [round(base + (index % 5) * 0.25, 4) for index in range(length)]


def _slopes(kind: str) -> list[float]:
    """Return G1's signed window slopes for one shape."""
    if kind == "growing":
        return [float(index) - 50.0 for index in range(101)]
    if kind == "shrinking":
        return [-float(index) - 1.0 for index in range(101)]
    raise AssertionError(kind)


def _quiet() -> dict[str, Any]:
    return quiet_window(quiet_stamp(0.4, _CORES), quiet_stamp(0.4, _CORES))


def _noisy() -> dict[str, Any]:
    return quiet_window(quiet_stamp(0.4, _CORES), quiet_stamp(9.0, _CORES))


def _spec_for(cell: str) -> Any:
    return cell_baseline_spec(
        cell,
        fixture_hash=f"fixture-hash-{cell}",
        dimensions=_DIMENSIONS,
        machine=_MACHINE,
    )


def _row_record(
    campaign_id: str,
    *,
    run_id: str,
    run_start_utc: str,
    spec: Any,
    base: float = 1.0,
    flat: float | None = None,
    slopes: list[float] | None = None,
    quiet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one row record of one run of the spec's cell."""
    measured, warmup, counts = _RECIPE[campaign_id]
    length = counts.get("chunk_gaps", measured)
    kwargs: dict[str, Any] = {
        "measured_ops": measured,
        "warmup_ops": warmup,
        "unit_counts": dict(counts),
    }
    if campaign_id == "G1":
        kwargs["slopes_signed"] = _slopes("growing") if slopes is None else list(slopes)
    elif flat is not None:
        kwargs["samples"] = [flat] * length
    else:
        kwargs["samples"] = _series(length, base)
    return build_baseline_run(
        campaign_id,
        run_id=run_id,
        run_start_utc=run_start_utc,
        fixture_hash=spec.fixture_hash,
        machine=dict(spec.machine),
        quiet=_quiet() if quiet is None else dict(quiet),
        dimensions=dict(spec.dimensions),
        **kwargs,
    )


def _cell_run(
    spec: Any,
    *,
    run_id: str,
    run_start_utc: str,
    base: float = 1.0,
    flat: float | None = None,
    slopes: list[float] | None = None,
    quiet: Mapping[str, Any] | None = None,
) -> Any:
    records = {
        campaign_id: _row_record(
            campaign_id,
            run_id=run_id,
            run_start_utc=run_start_utc,
            spec=spec,
            base=base,
            flat=flat,
            slopes=slopes,
            quiet=quiet,
        )
        for campaign_id in spec.campaign_ids
    }
    return shared_run(records, run_key=run_id)


def _baseline(
    spec: Any,
    *,
    count: int = SELECTED_RUNS,
    bases: Mapping[str, float] | None = None,
    flat: float | None = None,
    slopes: list[float] | None = None,
    noisy: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Build one cell's committed baseline from synthetic runs."""
    runs = []
    for index in range(1, count + 1):
        run_id = f"run-{index:02d}"
        runs.append(
            _cell_run(
                spec,
                run_id=run_id,
                run_start_utc=f"2026-11-08T01:{index:02d}:00Z",
                base=1.0 if bases is None else bases[run_id],
                flat=flat,
                slopes=slopes,
                quiet=_noisy() if run_id in noisy else None,
            )
        )
    return build_baseline(spec, runs)


def _input_set(**overrides: Any) -> dict[str, Any]:
    """Return a complete baseline input mapping: every gated cell accounted for."""
    mapping: dict[str, Any] = {cell: None for cell in gated_cells()}
    mapping.update(overrides)
    return mapping


def _fixtures() -> dict[str, dict[str, Any]]:
    """Build every synthetic input set the node needs, once."""
    fresh = _spec_for(CELL_FRESH_HEADLESS)
    permission = _spec_for(CELL_PERMISSION)
    memory = _spec_for(CELL_SOAK_MEMORY)
    thin_noise = frozenset(f"run-{index:02d}" for index in range(1, 9))
    return {
        "fresh": _input_set(**{CELL_FRESH_HEADLESS: _baseline(fresh, bases=_BASES)}),
        "permission": _input_set(**{CELL_PERMISSION: _baseline(permission)}),
        "permission_flat": _input_set(**{
            CELL_PERMISSION: _baseline(permission, flat=1.0)
        }),
        "permission_zero": _input_set(**{
            CELL_PERMISSION: _baseline(permission, flat=0.0)
        }),
        "permission_thin": _input_set(**{
            CELL_PERMISSION: _baseline(permission, count=MAX_ATTEMPTS, noisy=thin_noise)
        }),
        "memory_growing": _input_set(**{
            CELL_SOAK_MEMORY: _baseline(memory, slopes=_slopes("growing"))
        }),
        "memory_shrinking": _input_set(**{
            CELL_SOAK_MEMORY: _baseline(memory, slopes=_slopes("shrinking"))
        }),
    }


def _clamped_baseline() -> dict[str, Any]:
    """Return a G1 baseline whose committed run-p95s are negative.

    The producer clamps a run's slope series, so a committed negative p95 is
    only reachable through a hand-edited campaign block; X5a does not recompute
    p95s from records, so the block stays a valid input and exercises X7a's own
    positive-growth clamp.
    """
    document = copy.deepcopy(_baseline(_spec_for(CELL_SOAK_MEMORY)))
    document["campaigns"]["G1"]["run_p95s"] = [-3.0] * SELECTED_RUNS
    return _input_set(**{CELL_SOAK_MEMORY: document})


# ---------------------------------------------------------------------------
# The pinned inputs
# ---------------------------------------------------------------------------

_LOADED: dict[str, Any] = {}


def _inputs() -> Any:
    if "inputs" not in _LOADED:
        _LOADED["inputs"] = load_activation_inputs()
    return _LOADED["inputs"]


def _pinned() -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = _inputs()
    return inputs.registry, inputs.determinants


def _table(
    baselines: Mapping[str, Any],
    *,
    machine: Any = _UNSET,
    registry: Mapping[str, Any] | None = None,
    determinants: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the table from the pinned artifacts and the given inputs."""
    pinned_registry, pinned_determinants = _pinned()
    return construct_threshold_table(
        registry=pinned_registry if registry is None else registry,
        determinants=pinned_determinants if determinants is None else determinants,
        baselines=baselines,
        machine=_MACHINE if machine is _UNSET else machine,
    )


def _row(table: Mapping[str, Any], campaign_id: str) -> dict[str, Any]:
    for row in table["rows"]:
        if row["campaign_id"] == campaign_id:
            return row
    raise AssertionError(f"{campaign_id} is missing from the table")


def _assert_error(call: Callable[[], object], reason: str, label: str = "") -> None:
    with pytest.raises(ThresholdError) as excinfo:
        call()
    assert excinfo.value.reason == reason, f"{label}: {excinfo.value.detail}"


def _canonical(document: Any) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Pinned constants, artifacts and table invariants
# ---------------------------------------------------------------------------


def _assert_pinned_constants() -> None:
    assert THRESHOLDS_SCHEMA_ID == "chartreux.runtime_thresholds.v1"
    assert (STATE_ACTIVE, STATE_DORMANT) == ("ACTIVE", "DORMANT")
    assert DORMANCY_REASONS == (
        REASON_MACHINE_MISMATCH,
        INSUFFICIENT_DATA_REASON,
        REASON_MISSING_DETERMINANTS,
        REASON_DEGENERATE_THRESHOLD,
    )
    assert [tier.tier for tier in PINNED_REASON_TIERS] == [1, 2, 3, 4, 5, 6]
    assert DORMANCY_TIERS == PINNED_REASON_TIERS[:4]
    assert RUN_RECORD_TIER.recorded_reasons == (REASON_RUN_INELIGIBILITY,)
    assert RUN_RECORD_TIER.tier == 5
    assert THRESHOLD_TIER.recorded_reasons == (
        REASON_THRESHOLD_FAIL,
        REASON_THRESHOLD_PASS,
    )
    assert THRESHOLD_TIER.tier == 6
    # Tier 5 is a run-record reason and tier 6 labels comparisons: neither is a
    # dormancy reason, and the two vocabularies never overlap.
    assert REASON_RUN_INELIGIBILITY in NON_DORMANCY_REASONS
    assert REASON_INVALID_SAMPLE_CAP in NON_DORMANCY_REASONS
    assert set(NON_DORMANCY_REASONS) == {
        REASON_RUN_INELIGIBILITY,
        REASON_INVALID_SAMPLE_CAP,
        REASON_THRESHOLD_FAIL,
        REASON_THRESHOLD_PASS,
    }
    assert not set(NON_DORMANCY_REASONS) & set(DORMANCY_REASONS)
    # The parsed order flattens onto the vocabulary the comparison layer records.
    assert (
        tuple(
            reason for tier in PINNED_REASON_TIERS for reason in tier.recorded_reasons
        )
        == REASON_ORDER
    )
    assert THRESHOLD_RULE == (
        "T = M + max(6*D, 0.20*M); measured <= T passes, equality included"
    )
    assert threshold_value(4.0, 1.0) == 10.0
    # Six MADs dominate the twenty-percent band at D = 1, M = 4; the band
    # protects a degenerate D = 0 baseline instead.
    assert threshold_value(4.0, 0.0) == 4.0 + 0.20 * 4.0
    assert UNMEASURED_REASON == "activation_never_evaluates_a_measured_value"
    assert ACTIVATION_INDETERMINATE == "activation_machine_indeterminate"
    assert REGISTRY_SHA256 == (
        "7b336621b946b5760fb13ad673c70d4b4c14bf90d6325543ca9e530a35086de5"
    )
    assert DETERMINANTS_SHA256 == (
        "a1847c336f05c83cd3f50cfe3cc756ce2a4b453f79eb9a4a0ddc02a60a7cf21f"
    )
    assert SELECTED_RUNS == 5


def _assert_pinned_artifacts() -> None:
    """The two pinned artifacts are byte-exact, and the loader proves it."""
    registry_digest = hashlib.sha256(
        (_PERF_DIR / _REGISTRY_FILENAME).read_bytes()
    ).hexdigest()
    determinants_digest = hashlib.sha256(
        (_PERF_DIR / _DETERMINANTS_FILENAME).read_bytes()
    ).hexdigest()
    assert registry_digest == REGISTRY_SHA256, (
        "the pinned registry drifted from its plan §7 pin; STOP + ESCALATE — "
        "never re-emit or re-format it to force a match"
    )
    assert determinants_digest == DETERMINANTS_SHA256, (
        "the pinned determinants drifted from its plan §7 pin; STOP + ESCALATE — "
        "never re-emit or re-format it to force a match"
    )

    inputs = _inputs()
    assert inputs.pins == {
        "registry_sha256": REGISTRY_SHA256,
        "determinants_sha256": DETERMINANTS_SHA256,
    }
    # Every gated cell is looked up: an absent baseline is recorded as absent.
    assert set(inputs.baselines) == set(gated_cells())
    for cell, document in inputs.baselines.items():
        if document is not None:
            validate_baseline(document, spec_from_baseline(document))
            assert document["cell"] == cell

    common = inputs.determinants["common_determinants"]
    # The order this module applies is the order the artifact states, in full.
    assert parse_reason_order(common["reason_order"]) == PINNED_REASON_TIERS
    assert "T = M + max(6*D, 0.20*M)" in common["threshold"]
    rule = active_test_rule(inputs.determinants)
    for phrase in ACTIVE_TEST_REQUIRED_PHRASES:
        assert phrase in rule


def _assert_table_invariants(table: Mapping[str, Any]) -> None:
    rows = table["rows"]
    assert table["schema"] == THRESHOLDS_SCHEMA_ID
    assert table["threshold_rule"] == THRESHOLD_RULE
    assert [row["campaign_id"] for row in rows] == list(GATED_CAMPAIGN_IDS)
    assert table["row_count"] == len(rows) == 17
    assert table["active_count"] + table["dormant_count"] == len(rows)
    assert [tier["tier"] for tier in table["reason_order"]] == [1, 2, 3, 4, 5, 6]
    assert table["reason_order"][0]["recorded_reasons"] == [REASON_MACHINE_MISMATCH]
    assert table["dormancy_reasons"] == list(DORMANCY_REASONS)
    assert table["run_record_tier"]["tier"] == 5
    assert table["threshold_tier"]["tier"] == 6
    assert table["active_test"]["evaluates_fresh_measured_value"] is False
    assert table["slope_clamp"] == SLOPE_CLAMP_RULE
    assert table["activation_machine"] == {
        name: _MACHINE[name]
        for name in ("cpu_model", "core_count", "python_version", "platform")
    }
    for row in rows:
        assert row["state"] in (STATE_ACTIVE, STATE_DORMANT)
        # No activation disposition ever records a measured value.
        assert row["measured_value"] is None
        assert row["null_reasons"]["measured_value"] == UNMEASURED_REASON
        if row["state"] == STATE_ACTIVE:
            assert row["reason"] is None
            assert row["decision_ready"] is True
            assert row["threshold"] is not None
            assert math.isfinite(row["threshold"]) and row["threshold"] > 0
            assert "threshold" not in row["null_reasons"]
            assert len(row["selected_runs"]) == SELECTED_RUNS
        else:
            # Only the dormancy tiers can set a dormant state.
            assert row["reason"] in DORMANCY_REASONS
            assert row["reason"] not in NON_DORMANCY_REASONS
            assert row["reason"] not in (
                REASON_RUN_INELIGIBILITY,
                REASON_INVALID_SAMPLE_CAP,
                REASON_THRESHOLD_FAIL,
                REASON_THRESHOLD_PASS,
            )
            if row["threshold"] is None:
                assert row["null_reasons"].get("threshold") == row["reason"]
            else:
                assert row["threshold"] > 0


# ---------------------------------------------------------------------------
# Construction: ACTIVE, the twenty-percent band, absent and thin campaigns
# ---------------------------------------------------------------------------


def _assert_active_rows(fixtures: Mapping[str, dict[str, Any]]) -> None:
    table = _table(fixtures["fresh"])
    _assert_table_invariants(table)
    assert table["active_count"] == 4
    assert table["dormant_count"] == 13
    for campaign_id in ("A1", "A2", "A3", "A4"):
        row = _row(table, campaign_id)
        assert row["state"] == STATE_ACTIVE, row
        assert row["reason"] is None
        assert row["closure_status"] == CLOSURE_ELIGIBLE_COMPLETE
        assert row["decision_ready"] is True
        # The row-specific committed series, in run-start order.
        assert row["collected_run_p95s"] == [2.0, 3.0, 4.0, 5.0, 6.0]
        assert [run["run_id"] for run in row["selected_runs"]] == [
            "run-01",
            "run-02",
            "run-03",
            "run-04",
            "run-05",
        ]
        assert [run["run_start_utc"] for run in row["selected_runs"]] == [
            f"2026-11-08T01:0{index}:00Z" for index in range(1, 6)
        ]
        assert row["median"] == 4.0
        assert row["mad"] == 1.0
        assert row["threshold"] == 10.0  # M + 6D, not M + 0.20M
        assert row["machine_mismatch_fields"] == []
        assert row["slope_clamp"] is None
        assert row["collected_run_p95s_signed"] == [2.0, 3.0, 4.0, 5.0, 6.0]
    # The six MADs are the dominating term for this baseline.
    assert threshold_value(4.0, 1.0) == 4.0 + 6.0


def _assert_twenty_percent_band(fixtures: Mapping[str, dict[str, Any]]) -> None:
    """A degenerate D = 0 baseline is protected by the 20% band, not by 0."""
    table = _table(fixtures["permission_flat"])
    row = _row(table, "E2")
    assert row["state"] == STATE_ACTIVE
    assert row["collected_run_p95s"] == [1.0] * SELECTED_RUNS
    assert row["median"] == 1.0
    assert row["mad"] == 0.0
    assert row["threshold"] == pytest.approx(1.2)
    assert row["threshold"] == pytest.approx(1.0 + 0.20 * 1.0)
    assert row["threshold"] != 1.0


def _assert_absent_baseline(fixtures: Mapping[str, dict[str, Any]]) -> None:
    """A cell with no promoted baseline is not decision-ready, never ACTIVE."""
    table = _table(fixtures["fresh"])
    for campaign_id in ("B1", "B2", "B3", "B4", "C1", "D1", "E1", "E2", "F1", "F2"):
        row = _row(table, campaign_id)
        assert row["state"] == STATE_DORMANT
        assert row["reason"] == INSUFFICIENT_DATA_REASON
        assert row["closure_status"] is None
        assert row["decision_ready"] is False
        assert row["collected_run_p95s"] == []
        assert row["selected_runs"] == []
        assert row["threshold"] is None
        assert row["median"] is None and row["mad"] is None
        for name in ("collected_run_p95s", "median", "mad", "threshold"):
            assert row["null_reasons"][name] == INSUFFICIENT_DATA_REASON


def _assert_thin_campaign(fixtures: Mapping[str, dict[str, Any]]) -> None:
    """An insufficient_data closure is DORMANT and never populates a T."""
    baselines = fixtures["permission_thin"]
    document = baselines[CELL_PERMISSION]
    assert document["closure"]["status"] == CLOSURE_INSUFFICIENT_DATA
    assert document["attempt_count"] == MAX_ATTEMPTS
    ineligible = [
        attempt for attempt in document["attempts"] if attempt["status"] != "eligible"
    ]
    assert len(ineligible) == 8
    # Tier 5 is a RUN-RECORD reason: the attempts keep their own reasons...
    for attempt in ineligible:
        assert attempt["reasons"]
        assert attempt["failing_row"] is None
    table = _table(baselines)
    row = _row(table, "E2")
    assert row["state"] == STATE_DORMANT
    assert row["reason"] == INSUFFICIENT_DATA_REASON
    assert row["closure_status"] == CLOSURE_INSUFFICIENT_DATA
    assert row["decision_ready"] is False
    # ... and the activation row excludes them rather than re-labelling itself.
    assert len(row["collected_run_p95s"]) == 2
    assert row["reason"] != REASON_RUN_INELIGIBILITY
    assert row["threshold"] is None
    assert row["null_reasons"]["threshold"] == INSUFFICIENT_DATA_REASON
    # Thin data still supports what the two committed runs support.
    assert row["median"] == 2.0 and row["mad"] == 0.0


# ---------------------------------------------------------------------------
# Tier 1: the activation machine context
# ---------------------------------------------------------------------------


def _assert_machine_mismatch(fixtures: Mapping[str, dict[str, Any]]) -> None:
    healthy = fixtures["permission"]
    assert _row(_table(healthy), "E2")["state"] == STATE_ACTIVE

    one_field = _table(healthy, machine={**_MACHINE, "python_version": "3.13.0"})
    row = _row(one_field, "E2")
    assert row["state"] == STATE_DORMANT
    assert row["reason"] == REASON_MACHINE_MISMATCH
    assert row["machine_mismatch_fields"] == ["python_version"]
    # Dormancy records why the row does not gate; the arithmetic still stands.
    assert row["median"] == 2.0
    assert row["threshold"] == pytest.approx(2.4)
    assert "threshold" not in row["null_reasons"]
    assert one_field["active_count"] == 0

    two_fields = _table(
        healthy, machine={**_MACHINE, "core_count": 8, "platform": "Linux-other"}
    )
    assert _row(two_fields, "E2")["machine_mismatch_fields"] == [
        "core_count",
        "platform",
    ]
    assert _row(two_fields, "E2")["reason"] == REASON_MACHINE_MISMATCH


def _assert_indeterminate_machine(fixtures: Mapping[str, dict[str, Any]]) -> None:
    """An unevaluable activation context halts; it is never a DORMANT."""
    healthy = fixtures["permission"]
    for machine in (
        None,
        {},
        {"core_count": _CORES},
        {**_MACHINE, "cpu_model": None},
        {**_MACHINE, "cpu_model": "  "},
        {**_MACHINE, "core_count": 0},
        {**_MACHINE, "python_version": None},
        {**_MACHINE, "platform": ""},
    ):
        _assert_error(
            lambda machine=machine: _table(healthy, machine=machine),
            ACTIVATION_INDETERMINATE,
        )


# ---------------------------------------------------------------------------
# Tiers 2-4: closure, determinants, degenerate thresholds
# ---------------------------------------------------------------------------


def _determinants_copy() -> dict[str, Any]:
    return copy.deepcopy(_pinned()[1])


def _assert_missing_determinants(fixtures: Mapping[str, dict[str, Any]]) -> None:
    healthy = fixtures["permission"]

    absent = _determinants_copy()
    del absent["rows"]["E2"]
    table = _table(healthy, determinants=absent)
    row = _row(table, "E2")
    assert row["state"] == STATE_DORMANT
    assert row["reason"] == REASON_MISSING_DETERMINANTS
    assert row["threshold"] is None
    assert row["null_reasons"]["threshold"] == REASON_MISSING_DETERMINANTS
    # The row was decision-ready: the commitment is not what is missing.
    assert row["decision_ready"] is True

    foreign = _determinants_copy()
    foreign["rows"]["E2"] = {**foreign["rows"]["E2"], "metric": "another_metric"}
    foreign_table = _table(healthy, determinants=foreign)
    assert _row(foreign_table, "E2")["reason"] == REASON_MISSING_DETERMINANTS

    # A missing row for one cell never disturbs the other rows' dispositions.
    fresh = _table(fixtures["fresh"], determinants=absent)
    assert _row(fresh, "A1")["state"] == STATE_ACTIVE


def _assert_degenerate_threshold(fixtures: Mapping[str, dict[str, Any]]) -> None:
    """A nonpositive T is degenerate — no epsilon is ever invented."""
    table = _table(fixtures["permission_zero"])
    row = _row(table, "E2")
    assert row["state"] == STATE_DORMANT
    assert row["reason"] == REASON_DEGENERATE_THRESHOLD
    assert row["median"] == 0.0
    assert row["mad"] == 0.0
    assert row["threshold"] is None
    assert row["null_reasons"]["threshold"] == REASON_DEGENERATE_THRESHOLD
    assert threshold_value(0.0, 0.0) == 0.0


def _assert_g1_positive_growth(fixtures: Mapping[str, dict[str, Any]]) -> None:
    """G1's gated quantity is positive growth: clamp at zero, never pad."""
    growing = _table(fixtures["memory_growing"])
    row = _row(growing, "G1")
    assert row["state"] == STATE_ACTIVE
    assert row["collected_run_p95s"] == [45.0] * SELECTED_RUNS
    assert row["collected_run_p95s_signed"] == [45.0] * SELECTED_RUNS
    assert row["slope_clamp"] == SLOPE_CLAMP_RULE
    assert row["slope_clamped"] is False  # nothing needed clamping
    assert row["median"] == 45.0 and row["mad"] == 0.0
    assert row["threshold"] == pytest.approx(54.0)

    # A shrinking series is clamped by the PRODUCER: the committed run-p95 is
    # already 0, which is the row's gated quantity, so the baseline is a plateau.
    shrinking = _table(fixtures["memory_shrinking"])
    shrunk = _row(shrinking, "G1")
    assert shrunk["state"] == STATE_DORMANT
    assert shrunk["reason"] == REASON_DEGENERATE_THRESHOLD
    assert shrunk["collected_run_p95s"] == [0.0] * SELECTED_RUNS
    assert shrunk["median"] == 0.0 and shrunk["mad"] == 0.0
    assert shrunk["threshold"] is None
    assert shrunk["null_reasons"]["threshold"] == REASON_DEGENERATE_THRESHOLD

    # A committed negative p95 is clamped at extraction and kept signed: the
    # resulting nonpositive T is degenerate, and no epsilon is ever invented.
    clamped = _row(_table(_clamped_baseline()), "G1")
    assert clamped["collected_run_p95s"] == [0.0] * SELECTED_RUNS
    assert clamped["collected_run_p95s_signed"] == [-3.0] * SELECTED_RUNS
    assert clamped["slope_clamp"] == SLOPE_CLAMP_RULE
    assert clamped["slope_clamped"] is True
    assert clamped["state"] == STATE_DORMANT
    assert clamped["reason"] == REASON_DEGENERATE_THRESHOLD
    assert clamped["threshold"] is None
    assert threshold_value(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Reason-order precedence at activation
# ---------------------------------------------------------------------------


def _assert_reason_precedence(fixtures: Mapping[str, dict[str, Any]]) -> None:
    """The recorded reason is the FIRST applicable one in the pinned order."""
    mismatch = {**_MACHINE, "cpu_model": "Another CPU"}
    absent_determinants = _determinants_copy()
    del absent_determinants["rows"]["E2"]

    # Tier 1 beats tier 2: a mismatched machine on a thin campaign.
    thin_mismatch = _table(fixtures["permission_thin"], machine=mismatch)
    row = _row(thin_mismatch, "E2")
    assert row["reason"] == REASON_MACHINE_MISMATCH
    assert row["threshold"] is None
    assert row["null_reasons"]["threshold"] == INSUFFICIENT_DATA_REASON

    # Tier 1 beats tier 3.
    healthy_mismatch = _table(
        fixtures["permission"], machine=mismatch, determinants=absent_determinants
    )
    assert _row(healthy_mismatch, "E2")["reason"] == REASON_MACHINE_MISMATCH

    # Tier 2 beats tier 3.
    thin_absent = _table(fixtures["permission_thin"], determinants=absent_determinants)
    assert _row(thin_absent, "E2")["reason"] == INSUFFICIENT_DATA_REASON

    # Tier 3 beats tier 4: a degenerate baseline without its determinants.
    zero_absent = _table(fixtures["permission_zero"], determinants=absent_determinants)
    assert _row(zero_absent, "E2")["reason"] == REASON_MISSING_DETERMINANTS


# ---------------------------------------------------------------------------
# Activation never evaluates a measured value
# ---------------------------------------------------------------------------


def _assert_no_measured_value(fixtures: Mapping[str, dict[str, Any]]) -> None:
    """The constructor has no channel for a fresh measurement."""
    parameters = inspect.signature(construct_threshold_table).parameters
    assert set(parameters) == {
        "registry",
        "determinants",
        "baselines",
        "machine",
        "pins",
    }
    for name in parameters:
        assert not re.search(
            r"measured|measurement|sample|observation|latency|fresh", name
        ), name
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )

    table = _table(fixtures["fresh"])
    for row in table["rows"]:
        assert row["measured_value"] is None
        assert row["null_reasons"]["measured_value"] == UNMEASURED_REASON
    assert table["active_test"]["evaluates_fresh_measured_value"] is False


def _assert_determinism_and_purity(fixtures: Mapping[str, dict[str, Any]]) -> None:
    """Construction is a pure function of the pinned inputs."""
    registry, determinants = _pinned()
    baselines = fixtures["fresh"]
    before = (_canonical(registry), _canonical(determinants), _canonical(baselines))
    first = _table(baselines)
    second = _table(baselines)
    after = (_canonical(registry), _canonical(determinants), _canonical(baselines))
    assert before == after
    assert first == second
    assert first["table_digest"] == second["table_digest"]
    # The synthetic input set carries no pinned file digests of its own.
    assert first["inputs"]["digest_kind"] == "canonical_document"
    assert first["inputs"]["baselines"][CELL_FRESH_HEADLESS] == {
        "attempt_digest": baselines[CELL_FRESH_HEADLESS]["attempt_digest"],
        "closure_status": CLOSURE_ELIGIBLE_COMPLETE,
    }
    assert first["inputs"]["baselines"][CELL_STORM] is None


# ---------------------------------------------------------------------------
# Malformed or ambiguous input halts
# ---------------------------------------------------------------------------


def _assert_halts(fixtures: Mapping[str, dict[str, Any]]) -> None:
    registry, determinants = _pinned()
    healthy = fixtures["permission"]

    broken_schema = copy.deepcopy(registry)
    broken_schema["schema"] = "chartreux.runtime_metric_registry.v0"
    short = copy.deepcopy(registry)
    short["entries"] = short["entries"][:-1]
    drifted = copy.deepcopy(registry)
    drifted["entries"][0] = {**drifted["entries"][0], "metric": "another_metric"}
    duplicated = copy.deepcopy(registry)
    duplicated["entries"] = [*duplicated["entries"], duplicated["entries"][0]]

    order_text = determinants["common_determinants"]["reason_order"]
    pinned_snapshot = _canonical(determinants)
    swapped = copy.deepcopy(determinants)
    swapped["common_determinants"]["reason_order"] = order_text.replace(
        "(1) machine_mismatch; (2) campaign_not_decision_ready",
        "(1) campaign_not_decision_ready; (2) machine_mismatch",
    )
    truncated = copy.deepcopy(determinants)
    truncated["common_determinants"]["reason_order"] = order_text.split("(4)")[0]
    reworded = copy.deepcopy(determinants)
    reworded["common_determinants"]["active_test"] = reworded["common_determinants"][
        "active_test"
    ].replace("never evaluates a fresh measured value", "never evaluates a value")
    no_common = copy.deepcopy(determinants)
    del no_common["common_determinants"]
    no_rows = copy.deepcopy(determinants)
    del no_rows["rows"]
    other_schema = copy.deepcopy(determinants)
    other_schema["schema"] = "chartreux.runtime_metric_determinants.v0"

    open_campaign = _input_set(**{
        CELL_PERMISSION: _baseline(_spec_for(CELL_PERMISSION), count=3)
    })
    tampered = _input_set(**{CELL_PERMISSION: copy.deepcopy(healthy[CELL_PERMISSION])})
    tampered[CELL_PERMISSION]["attempts"][0]["status"] = "ineligible"
    foreign_cell = _input_set(**{CELL_STORM: healthy[CELL_PERMISSION]})
    incomplete = {
        cell: value for cell, value in healthy.items() if cell != CELL_SOAK_MEMORY
    }
    unknown = {**healthy, "not_a_gated_cell": None}
    not_a_document = {**healthy, CELL_PERMISSION: "not a baseline"}

    cases: tuple[tuple[str, Callable[[], object], str], ...] = (
        (
            "registry schema",
            lambda: _table(healthy, registry=broken_schema),
            "registry_invalid",
        ),
        (
            "registry row set",
            lambda: _table(healthy, registry=short),
            "registry_row_set_mismatch",
        ),
        (
            "registry row drift",
            lambda: _table(healthy, registry=drifted),
            "registry_row_drift",
        ),
        (
            "registry duplicate",
            lambda: _table(healthy, registry=duplicated),
            "registry_invalid",
        ),
        (
            "determinants schema",
            lambda: _table(healthy, determinants=other_schema),
            "determinants_schema_mismatch",
        ),
        (
            "determinants common block",
            lambda: _table(healthy, determinants=no_common),
            "determinants_common_missing",
        ),
        (
            "reason order swapped",
            lambda: _table(healthy, determinants=swapped),
            "determinants_reason_order_drift",
        ),
        (
            "reason order truncated",
            lambda: _table(healthy, determinants=truncated),
            "determinants_reason_order_drift",
        ),
        (
            "active test reworded",
            lambda: _table(healthy, determinants=reworded),
            "determinants_active_test_drift",
        ),
        (
            "determinants rows",
            lambda: _table(healthy, determinants=no_rows),
            "determinants_invalid",
        ),
        ("open campaign", lambda: _table(open_campaign), "baseline_invalid"),
        ("tampered attempts", lambda: _table(tampered), "baseline_invalid"),
        (
            "baseline under a foreign cell",
            lambda: _table(foreign_cell),
            "baseline_cell_mismatch",
        ),
        ("input set incomplete", lambda: _table(incomplete), "baselines_incomplete"),
        (
            "unknown cell in the input set",
            lambda: _table(unknown),
            "baselines_unknown_cell",
        ),
        (
            "baseline not a document",
            lambda: _table(not_a_document),
            "baseline_not_mapping",
        ),
    )
    for label, call, reason in cases:
        _assert_error(call, reason, label)
    # A refused input never mutates the artifact the refusal was read from.
    assert _canonical(determinants) == pinned_snapshot


# ---------------------------------------------------------------------------
# X7b: comparison states and the PERF-BLOB machine-mismatch record
# ---------------------------------------------------------------------------


def _assert_schema_error(
    call: Callable[[], object], reason: str, label: str = ""
) -> None:
    with pytest.raises(ReportSchemaError) as excinfo:
        call()
    assert excinfo.value.reason == reason, f"{label}: {excinfo.value.detail}"


#: The comparison machine of every mismatch fixture: one required field differs.
_MISMATCH_MACHINE: dict[str, Any] = {**_MACHINE, "python_version": "3.13.0"}


def _comparison_run(
    spec: Any,
    *,
    base: float = 1.0,
    quiet: Mapping[str, Any] | None = None,
    machine: Mapping[str, Any] | None = None,
    run_id: str = "cmp-01",
    run_start_utc: str = "2026-11-08T02:00:00Z",
) -> Any:
    """Build one fresh comparison run of the spec's cell (every row co-measured)."""
    records = {
        campaign_id: _row_record(
            campaign_id,
            run_id=run_id,
            run_start_utc=run_start_utc,
            spec=spec,
            base=base,
            quiet=quiet,
        )
        for campaign_id in spec.campaign_ids
    }
    if machine is not None:
        for record in records.values():
            record["machine"] = dict(machine)
    return shared_run(records, run_key=run_id)


def _comparison_documents(
    fixtures: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build one X5b comparison document per marked state the node pins."""
    fresh_spec = _spec_for(CELL_FRESH_HEADLESS)
    permission_spec = _spec_for(CELL_PERMISSION)
    fresh = fixtures["fresh"][CELL_FRESH_HEADLESS]
    thin = fixtures["permission_thin"][CELL_PERMISSION]
    degenerate = fixtures["permission_zero"][CELL_PERMISSION]
    return {
        "pass": build_comparison(
            fresh, _comparison_run(fresh_spec, base=1.0, run_id="cmp-pass")
        ),
        # The committed T is 10.0: 10.0 == T passes by equality.
        "equality": build_comparison(
            fresh, _comparison_run(fresh_spec, base=9.0, run_id="cmp-equality")
        ),
        "fail": build_comparison(
            fresh, _comparison_run(fresh_spec, base=9.25, run_id="cmp-fail")
        ),
        "ineligible": build_comparison(
            fresh, _comparison_run(fresh_spec, quiet=_noisy(), run_id="cmp-ineligible")
        ),
        "mismatch": build_comparison(
            fresh,
            _comparison_run(
                fresh_spec, machine=_MISMATCH_MACHINE, run_id="cmp-mismatch"
            ),
        ),
        # A dormant row outranks the comparison run's own ineligibility.
        "mismatch_noisy": build_comparison(
            fresh,
            _comparison_run(
                fresh_spec,
                quiet=_noisy(),
                machine=_MISMATCH_MACHINE,
                run_id="cmp-mismatch-noisy",
            ),
        ),
        "thin": build_comparison(thin, _comparison_run(permission_spec)),
        "degenerate": build_comparison(degenerate, _comparison_run(permission_spec)),
    }


def _assert_reporting_vocabulary() -> None:
    """The report's labels are X5b's comparison labels, in the same total order."""
    assert tuple(COMPARISON_STATE_PRECEDENCE) == (
        COMPARISON_STATE_DORMANT,
        COMPARISON_STATE_INELIGIBLE,
        COMPARISON_STATE_PASS,
        COMPARISON_STATE_FAIL,
    )
    assert tuple(COMPARISON_STATE_PRECEDENCE) == COMPARISON_LABEL_PRECEDENCE
    assert (COMPARISON_STATE_DORMANT, COMPARISON_STATE_PASS) == (
        "DORMANT",
        "ACTIVE-pass",
    )
    assert (COMPARISON_STATE_INELIGIBLE, COMPARISON_STATE_FAIL) == (
        "COMPARISON-INELIGIBLE",
        "ACTIVE-fail",
    )
    assert COMPARISON_STATE_DORMANT == STATE_DORMANT
    assert COMPARISON_STATE_RULE == (
        "row first: DORMANT > COMPARISON-INELIGIBLE > threshold labels"
    )
    assert (COMPARISON_STATES_SECTION, PERF_BLOB_SECTION) == (
        "comparison_states",
        "perf_blob",
    )
    assert PERF_BLOB_KIND == "perf_blob"
    assert PERF_BLOB_MACHINE_MISMATCH == REASON_MACHINE_MISMATCH
    assert PERF_BLOB_FILENAME == "perf_blob.json"
    # The WP6 core schema carries neither section until X7b merges them.
    assert COMPARISON_STATES_SECTION not in REPORT_SCHEMA
    assert PERF_BLOB_SECTION not in REPORT_SCHEMA


def _assert_comparison_states(documents: Mapping[str, dict[str, Any]]) -> None:
    """Every marked state renders from the comparison record's own facts."""
    passing = build_comparison_states([documents["pass"]])
    validate_state_section(passing)
    assert passing["rule"] == COMPARISON_STATE_RULE
    assert passing["precedence"] == list(COMPARISON_STATE_PRECEDENCE)
    assert passing["counts"] == {
        COMPARISON_STATE_DORMANT: 0,
        COMPARISON_STATE_INELIGIBLE: 0,
        COMPARISON_STATE_PASS: 4,
        COMPARISON_STATE_FAIL: 0,
    }
    assert [row["campaign_id"] for row in passing["rows"]] == ["A1", "A2", "A3", "A4"]
    for row in passing["rows"]:
        spec = ROW_SPECS[row["campaign_id"]]
        assert (row["cell"], row["metric"], row["unit"]) == (
            spec.cell,
            spec.metric,
            spec.unit,
        )
        assert row["state"] == COMPARISON_STATE_PASS
        assert row["precedence"] == 2
        assert row["reason"] == REASON_THRESHOLD_PASS
        assert row["direction"] == DIRECTION_LOWER_IS_BETTER
        assert row["measured_value"] == 2.0
        assert row["threshold"] == 10.0
        assert row["null_reasons"] == {}
        assert row["machine_mismatch_fields"] == []
        assert row["comparison_run"]["eligible"] is True

    # Equality passes: measured == T is a pass, never a failure.
    equality = build_comparison_states([documents["equality"]])
    for row in equality["rows"]:
        assert row["state"] == COMPARISON_STATE_PASS
        assert row["measured_value"] == row["threshold"] == 10.0
    assert equality["counts"][COMPARISON_STATE_PASS] == 4

    # Above T fails, and only above T.
    failing = build_comparison_states([documents["fail"]])
    for row in failing["rows"]:
        assert row["state"] == COMPARISON_STATE_FAIL
        assert row["precedence"] == 3
        assert row["reason"] == REASON_THRESHOLD_FAIL
        assert row["measured_value"] == 10.25
        assert row["threshold"] == 10.0
        assert row["measured_value"] > row["threshold"]
    assert failing["counts"][COMPARISON_STATE_FAIL] == 4

    # An ineligible comparison run on an eligible row: no threshold outcome.
    ineligible = build_comparison_states([documents["ineligible"]])
    for row in ineligible["rows"]:
        assert row["state"] == COMPARISON_STATE_INELIGIBLE
        assert row["precedence"] == 1
        assert row["reason"] == REASON_RUN_INELIGIBILITY
        assert row["measured_value"] is None
        assert row["null_reasons"]["measured_value"] == REASON_RUN_INELIGIBILITY
        # The row is not dormant, so its threshold is populated and unused.
        assert row["threshold"] == 10.0
        assert row["comparison_run"]["eligible"] is False
        assert row["comparison_run"]["reasons"]
    assert ineligible["counts"][COMPARISON_STATE_INELIGIBLE] == 4

    # A thin campaign is DORMANT with its dormancy reason and no populated T.
    thin = build_comparison_states([documents["thin"]])
    row = thin["rows"][0]
    assert [entry["campaign_id"] for entry in thin["rows"]] == ["E2"]
    assert row["state"] == COMPARISON_STATE_DORMANT
    assert row["precedence"] == 0
    assert row["reason"] == INSUFFICIENT_DATA_REASON
    assert row["measured_value"] is None
    assert row["threshold"] is None
    assert row["null_reasons"]["threshold"] == INSUFFICIENT_DATA_REASON
    assert row["comparison_run"]["eligible"] is None
    assert row["comparison_run"]["null_reason"] == "row_dormant"

    # A nonpositive committed T is degenerate, never padded.
    degenerate = build_comparison_states([documents["degenerate"]])
    degenerate_row = degenerate["rows"][0]
    assert degenerate_row["state"] == COMPARISON_STATE_DORMANT
    assert degenerate_row["reason"] == REASON_DEGENERATE_THRESHOLD
    assert degenerate_row["threshold"] is None
    assert degenerate_row["null_reasons"]["threshold"] == REASON_DEGENERATE_THRESHOLD

    # Several cells render into one section, in registry order within each cell.
    combined = build_comparison_states([documents["pass"], documents["thin"]])
    assert [row["campaign_id"] for row in combined["rows"]] == [
        "A1",
        "A2",
        "A3",
        "A4",
        "E2",
    ]
    assert combined["counts"] == {
        COMPARISON_STATE_DORMANT: 1,
        COMPARISON_STATE_INELIGIBLE: 0,
        COMPARISON_STATE_PASS: 4,
        COMPARISON_STATE_FAIL: 0,
    }
    assert _canonical(combined) == _canonical(
        build_comparison_states([documents["pass"], documents["thin"]])
    )


def _assert_state_precedence(documents: Mapping[str, dict[str, Any]]) -> None:
    """Row first: a dormant row is labelled before any run-level evaluation."""
    section = build_comparison_states([documents["mismatch_noisy"]])
    for row in section["rows"]:
        assert row["state"] == COMPARISON_STATE_DORMANT
        assert row["reason"] == REASON_MACHINE_MISMATCH
        assert row["machine_mismatch_fields"] == ["python_version"]
        # The run's own ineligibility never becomes a dormant row's label.
        assert row["comparison_run"]["eligible"] is None
        assert row["comparison_run"]["null_reason"] == "row_dormant"
        assert row["measured_value"] is None
    assert section["counts"][COMPARISON_STATE_DORMANT] == 4

    # The same kind of run on an eligible row is COMPARISON-INELIGIBLE instead.
    ineligible = build_comparison_states([documents["ineligible"]])
    assert {row["state"] for row in ineligible["rows"]} == {COMPARISON_STATE_INELIGIBLE}
    assert {row["comparison_run"]["eligible"] for row in ineligible["rows"]} == {False}

    # No rendered label ever leaves the four marked labels: there is no FAILED.
    for document in documents.values():
        rendered = build_comparison_states([document])
        for row in rendered["rows"]:
            assert row["state"] in COMPARISON_STATE_PRECEDENCE
            assert row["precedence"] == COMPARISON_STATE_PRECEDENCE.index(row["state"])


def _assert_machine_mismatch_blob(documents: Mapping[str, dict[str, Any]]) -> None:
    """A four-field mismatch emits the PERF-BLOB record and never ACTIVATES."""
    cell = CELL_FRESH_HEADLESS
    blob = machine_mismatch_blob(
        cell=cell, baseline_machine=_MACHINE, comparison_machine=_MISMATCH_MACHINE
    )
    assert blob is not None
    assert blob["kind"] == PERF_BLOB_KIND
    assert blob["record"] == PERF_BLOB_MACHINE_MISMATCH
    assert blob["cell"] == cell
    assert blob["campaign_ids"] == ["A1", "A2", "A3", "A4"]
    assert blob["mismatch_fields"] == ["python_version"]
    assert blob["baseline_machine"] == _MACHINE
    assert blob["comparison_machine"] == _MISMATCH_MACHINE
    assert blob["state"] == COMPARISON_STATE_DORMANT == STATE_DORMANT
    assert blob["reason"] == REASON_MACHINE_MISMATCH
    assert blob["never_active"] is True
    assert blob["never_failed"] is True
    validate_perf_blob_section(perf_blob_section([blob]))

    # The rendered rows and the PERF-BLOB record agree on what differs.
    section = build_comparison_states([documents["mismatch"]])
    assert section["counts"][COMPARISON_STATE_DORMANT] == 4
    for row in section["rows"]:
        assert row["state"] == COMPARISON_STATE_DORMANT
        assert row["reason"] == REASON_MACHINE_MISMATCH
        assert row["machine_mismatch_fields"] == blob["mismatch_fields"]
        assert row["measured_value"] is None
        assert row["threshold"] == 10.0  # dormant, never activated by it

    # ANY of the four required fields is a mismatch, not core_count alone.
    for field_name, value in (
        ("cpu_model", "Another CPU"),
        ("core_count", 8),
        ("python_version", "3.13.0"),
        ("platform", "Linux-other"),
    ):
        record = machine_mismatch_blob(
            cell=cell,
            baseline_machine=_MACHINE,
            comparison_machine={**_MACHINE, field_name: value},
        )
        assert record is not None, field_name
        assert record["mismatch_fields"] == [field_name]
    two = machine_mismatch_blob(
        cell=cell,
        baseline_machine=_MACHINE,
        comparison_machine={**_MACHINE, "core_count": 8, "platform": "Linux-other"},
    )
    assert two is not None
    assert two["mismatch_fields"] == ["core_count", "platform"]

    # A matched machine emits no record at all, and an empty section is empty.
    assert (
        machine_mismatch_blob(
            cell=cell, baseline_machine=_MACHINE, comparison_machine=_MACHINE
        )
        is None
    )
    assert perf_blob_section([]) == {"kind": PERF_BLOB_KIND, "records": []}

    # A field the comparison machine cannot report is a run-record defect, not
    # evidence the machine differs; an unevaluable baseline machine halts.
    unusable = machine_mismatch_blob(
        cell=cell,
        baseline_machine=_MACHINE,
        comparison_machine={**_MACHINE, "cpu_model": None, "platform": "Linux-other"},
    )
    assert unusable is not None
    assert unusable["mismatch_fields"] == ["platform"]
    assert "cpu_model" not in unusable["comparison_machine"]
    _assert_error(
        lambda: machine_mismatch_blob(
            cell=cell,
            baseline_machine={"core_count": _CORES},
            comparison_machine=_MACHINE,
        ),
        "baseline_machine_indeterminate",
    )
    _assert_error(
        lambda: machine_mismatch_blob(
            cell="not_a_gated_cell",
            baseline_machine=_MACHINE,
            comparison_machine={**_MACHINE, "platform": "Linux-other"},
        ),
        "blob_cell_invalid",
    )


def _assert_reporting_schema(documents: Mapping[str, dict[str, Any]]) -> None:
    """The sections merge into the WP6 report additively, core sections intact."""
    blob = machine_mismatch_blob(
        cell=CELL_FRESH_HEADLESS,
        baseline_machine=_MACHINE,
        comparison_machine=_MISMATCH_MACHINE,
    )
    assert blob is not None
    sections = reporting_sections(
        comparisons=[documents["pass"], documents["thin"]], blobs=[blob]
    )
    assert set(sections) == {COMPARISON_STATES_SECTION, PERF_BLOB_SECTION}
    validate_state_section(sections[COMPARISON_STATES_SECTION])
    validate_perf_blob_section(sections[PERF_BLOB_SECTION])

    record = _row_record(
        "A1",
        run_id="report-01",
        run_start_utc="2026-11-08T03:00:00Z",
        spec=_spec_for(CELL_FRESH_HEADLESS),
    )
    report = {**build_report([record]), **sections}
    validate_report(report)
    validate_reporting_sections(report)
    assert report["schema"] == "chartreux.runtime_perf.v1"
    assert report["metric_version"] == "chartreux.runtime_perf.v1"
    assert [run["campaign_id"] for run in report["runs"]] == ["A1"]
    assert report[COMPARISON_STATES_SECTION]["counts"][COMPARISON_STATE_PASS] == 4
    assert report[PERF_BLOB_SECTION]["records"] == [blob]

    # The descriptors merge through the schema's own extension mechanism.
    extended = report_schema(extra_sections=reporting_extension())
    assert extended["rows"] == REPORT_SCHEMA["rows"]
    assert extended["run_fields"] == REPORT_SCHEMA["run_fields"]
    assert extended["modes"] == REPORT_SCHEMA["modes"]
    assert extended[COMPARISON_STATES_SECTION]["precedence"] == list(
        COMPARISON_STATE_PRECEDENCE
    )
    assert extended[COMPARISON_STATES_SECTION]["rule"] == COMPARISON_STATE_RULE
    assert extended[PERF_BLOB_SECTION]["record"] == PERF_BLOB_MACHINE_MISMATCH
    assert COMPARISON_STATES_SECTION not in REPORT_SCHEMA
    assert PERF_BLOB_SECTION not in REPORT_SCHEMA
    # The additive sections are disjoint from every core section, so the merge
    # can never be a core replacement.
    assert not set(reporting_extension()) & set(REPORT_SCHEMA)


def _assert_reporting_emit(
    documents: Mapping[str, dict[str, Any]], tmp_path: Path
) -> None:
    """The PERF-BLOB record is written through the WP1 path guard, RUN_DIR-only."""
    blob = machine_mismatch_blob(
        cell=CELL_FRESH_HEADLESS,
        baseline_machine=_MACHINE,
        comparison_machine=_MISMATCH_MACHINE,
    )
    assert blob is not None
    section = perf_blob_section([blob])
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "perf-report.json"

    written = emit_perf_blob(
        section,
        environ={RUN_DIR_ENV: str(run_dir), REPORT_ENV: str(report_path)},
        scratchpad_root=tmp_path,
    )
    assert written is not None
    assert written == report_path
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload == json.loads(json.dumps(section))

    # RUN_DIR alone is the fallback destination, still guard-checked.
    fallback = emit_perf_blob(
        section, environ={RUN_DIR_ENV: str(run_dir)}, scratchpad_root=tmp_path
    )
    assert fallback is not None
    assert fallback == run_dir / PERF_BLOB_FILENAME
    assert json.loads(fallback.read_text(encoding="utf-8")) == payload

    # A section with no record is written nowhere at all.
    assert (
        emit_perf_blob(
            perf_blob_section([]),
            environ={RUN_DIR_ENV: str(run_dir)},
            scratchpad_root=tmp_path,
        )
        is None
    )

    # Nothing escaped into the repository.
    repo = repository_root()
    for path in (written, fallback):
        assert repo not in path.resolve().parents

    outside = tmp_path.parent / "outside-run"
    # No destination at all is its own refusal: the record goes to a guarded
    # path or nowhere.
    _assert_error(
        lambda: emit_perf_blob(section, environ={}, scratchpad_root=tmp_path),
        "report_path_unset",
    )
    for label, environ, scratchpad in (
        ("report without run dir", {REPORT_ENV: str(report_path)}, tmp_path),
        ("relative run dir", {RUN_DIR_ENV: "run"}, tmp_path),
        (
            "run dir under the repository",
            {RUN_DIR_ENV: str(repo / "tests" / "perf")},
            tmp_path,
        ),
        (
            "report under the repository",
            {
                RUN_DIR_ENV: str(run_dir),
                REPORT_ENV: str(repo / "tests" / "perf" / "report.json"),
            },
            tmp_path,
        ),
        (
            "report outside the run dir",
            {RUN_DIR_ENV: str(run_dir), REPORT_ENV: str(tmp_path / "elsewhere.json")},
            tmp_path,
        ),
        ("run dir outside the scratchpad", {RUN_DIR_ENV: str(outside)}, tmp_path),
        ("run dir missing", {RUN_DIR_ENV: str(tmp_path / "absent")}, tmp_path),
    ):
        _assert_error(
            lambda environ=environ, scratchpad=scratchpad: emit_perf_blob(
                section, environ=environ, scratchpad_root=scratchpad
            ),
            "report_path_rejected",
            label,
        )
    _assert_error(
        lambda: emit_perf_blob(
            section,
            environ={RUN_DIR_ENV: str(run_dir)},
            scratchpad_root=tmp_path,
            filename="../perf_blob.json",
        ),
        "filename_invalid",
    )
    # A refusal never writes anything beside the run directory.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run"]
    assert not outside.exists()


def _assert_reporting_refusals(documents: Mapping[str, dict[str, Any]]) -> None:
    """A label its own facts do not support is refused, never rendered."""
    tampered = copy.deepcopy(documents["mismatch"])
    tampered["records"]["A1"]["measured_value"] = 1.0
    _assert_error(
        lambda: build_comparison_states([tampered]), "comparison_dormant_measured_value"
    )

    tampered = copy.deepcopy(documents["mismatch"])
    record = tampered["records"]["A1"]
    record["outcome"] = COMPARISON_STATE_PASS
    record["reason"] = REASON_THRESHOLD_PASS
    record["measured_value"] = 2.0
    record["comparison_run"]["eligible"] = True
    _assert_error(
        lambda: build_comparison_states([tampered]),
        "comparison_active_mismatched_hardware",
    )

    tampered = copy.deepcopy(documents["pass"])
    tampered["records"]["A1"]["measured_value"] = 99.0
    _assert_error(
        lambda: build_comparison_states([tampered]),
        "comparison_threshold_relation_mismatch",
    )

    tampered = copy.deepcopy(documents["ineligible"])
    tampered["records"]["A1"]["measured_value"] = 2.0
    _assert_error(
        lambda: build_comparison_states([tampered]),
        "comparison_ineligible_measured_value",
    )

    tampered = copy.deepcopy(documents["thin"])
    tampered["records"]["E2"]["reason"] = REASON_THRESHOLD_PASS
    _assert_error(
        lambda: build_comparison_states([tampered]),
        "comparison_dormancy_reason_invalid",
    )

    tampered = copy.deepcopy(documents["pass"])
    tampered["records"]["A1"]["outcome"] = "FAILED"
    _assert_error(
        lambda: build_comparison_states([tampered]), "comparison_label_invalid"
    )

    tampered = copy.deepcopy(documents["pass"])
    del tampered["records"]["A4"]
    _assert_error(lambda: build_comparison_states([tampered]), "comparison_row_missing")

    tampered = copy.deepcopy(documents["pass"])
    tampered["records"]["E2"] = copy.deepcopy(tampered["records"]["A1"])
    _assert_error(
        lambda: build_comparison_states([tampered]), "comparison_row_unexpected"
    )

    tampered = copy.deepcopy(documents["pass"])
    tampered["schema"] = "chartreux.runtime_comparison.v0"
    _assert_error(
        lambda: build_comparison_states([tampered]), "comparison_schema_mismatch"
    )

    _assert_error(
        lambda: build_comparison_states([documents["pass"], documents["pass"]]),
        "comparison_row_duplicated",
    )
    _assert_error(lambda: build_comparison_states([]), "comparison_documents_missing")

    # A rendered section is re-derived row by row, so an edited label cannot
    # survive even when its structure still looks right.
    section = build_comparison_states([documents["pass"]])
    tampered_section = copy.deepcopy(section)
    tampered_section["rows"][0]["state"] = COMPARISON_STATE_FAIL
    _assert_schema_error(
        lambda: validate_state_section(tampered_section),
        "comparison_state_precedence_mismatch",
    )

    tampered_section = copy.deepcopy(section)
    tampered_section["rows"][0]["precedence"] = 0
    _assert_schema_error(
        lambda: validate_state_section(tampered_section),
        "comparison_state_precedence_mismatch",
    )

    tampered_section = copy.deepcopy(section)
    tampered_section["rows"][0]["measured_value"] = None
    tampered_section["rows"][0]["null_reasons"]["measured_value"] = "test"
    _assert_schema_error(
        lambda: validate_state_section(tampered_section),
        "comparison_state_threshold_outcome",
    )

    tampered_section = copy.deepcopy(section)
    tampered_section["rows"][0]["machine_mismatch_fields"] = ["python_version"]
    _assert_schema_error(
        lambda: validate_state_section(tampered_section),
        "comparison_state_mismatched_hardware",
    )

    tampered_section = copy.deepcopy(section)
    tampered_section["counts"][COMPARISON_STATE_PASS] = 3
    _assert_schema_error(
        lambda: validate_state_section(tampered_section),
        "comparison_states_counts_mismatch",
    )

    tampered_section = copy.deepcopy(section)
    tampered_section["rule"] = "threshold labels first"
    _assert_schema_error(
        lambda: validate_state_section(tampered_section), "comparison_states_rule_drift"
    )

    tampered_section = copy.deepcopy(section)
    tampered_section["rows"][0]["campaign_id"] = "Z9"
    _assert_schema_error(
        lambda: validate_state_section(tampered_section), "unknown_campaign_id"
    )

    tampered_section = copy.deepcopy(section)
    tampered_section["rows"][0]["reason"] = REASON_THRESHOLD_FAIL
    _assert_error(
        lambda: validate_state_section(tampered_section),
        "comparison_threshold_reason_invalid",
    )

    tampered_section = copy.deepcopy(section)
    tampered_section["rows"][0]["direction"] = "higher_is_better"
    _assert_error(
        lambda: validate_state_section(tampered_section),
        "comparison_threshold_relation_mismatch",
    )

    # The PERF-BLOB section is the machine-mismatch record and nothing else.
    blob = machine_mismatch_blob(
        cell=CELL_FRESH_HEADLESS,
        baseline_machine=_MACHINE,
        comparison_machine=_MISMATCH_MACHINE,
    )
    assert blob is not None
    tampered_blob = copy.deepcopy(blob)
    tampered_blob["state"] = COMPARISON_STATE_PASS
    _assert_schema_error(
        lambda: perf_blob_section([tampered_blob]), "perf_blob_state_not_dormant"
    )

    tampered_blob = copy.deepcopy(blob)
    tampered_blob["never_failed"] = False
    _assert_schema_error(
        lambda: perf_blob_section([tampered_blob]), "perf_blob_disposition_invalid"
    )

    tampered_blob = copy.deepcopy(blob)
    tampered_blob["mismatch_fields"] = []
    _assert_schema_error(
        lambda: perf_blob_section([tampered_blob]), "perf_blob_mismatch_fields_invalid"
    )

    tampered_blob = copy.deepcopy(blob)
    tampered_blob["comparison_machine"]["python_version"] = _MACHINE["python_version"]
    _assert_schema_error(
        lambda: perf_blob_section([tampered_blob]), "perf_blob_mismatch_unproven"
    )

    tampered_blob = copy.deepcopy(blob)
    del tampered_blob["comparison_machine"]["python_version"]
    _assert_schema_error(
        lambda: perf_blob_section([tampered_blob]), "perf_blob_machine_field_missing"
    )

    tampered_blob = copy.deepcopy(blob)
    tampered_blob["record"] = "threshold_fail"
    _assert_schema_error(
        lambda: perf_blob_section([tampered_blob]), "perf_blob_record_kind_invalid"
    )

    tampered_blob = copy.deepcopy(blob)
    del tampered_blob["cell"]
    _assert_schema_error(
        lambda: perf_blob_section([tampered_blob]), "perf_blob_field_missing"
    )
    _assert_schema_error(
        lambda: validate_reporting_sections({PERF_BLOB_SECTION: {"records": []}}),
        "perf_blob_kind_invalid",
    )


# ---------------------------------------------------------------------------
# The contract node
# ---------------------------------------------------------------------------


@pytest.mark.timeout(_THRESHOLD_TIMEOUT_S)
def test_runtime_threshold_eligibility_contract() -> None:
    """X7a builds activation from the committed set alone, or halts."""
    started = perf_counter()
    _assert_pinned_constants()
    _assert_pinned_artifacts()
    fixtures = _fixtures()
    _assert_active_rows(fixtures)
    _assert_twenty_percent_band(fixtures)
    _assert_absent_baseline(fixtures)
    _assert_thin_campaign(fixtures)
    _assert_machine_mismatch(fixtures)
    _assert_indeterminate_machine(fixtures)
    _assert_missing_determinants(fixtures)
    _assert_degenerate_threshold(fixtures)
    _assert_g1_positive_growth(fixtures)
    _assert_reason_precedence(fixtures)
    _assert_no_measured_value(fixtures)
    _assert_determinism_and_purity(fixtures)
    _assert_halts(fixtures)
    assert perf_counter() - started <= _CONTRACT_WINDOW_S


@pytest.mark.timeout(_REPORTING_TIMEOUT_S)
def test_runtime_threshold_reporting_contract(tmp_path: Path) -> None:
    """X7b renders the four comparison states and the PERF-BLOB mismatch record."""
    started = perf_counter()
    _assert_reporting_vocabulary()
    documents = _comparison_documents(_fixtures())
    _assert_comparison_states(documents)
    _assert_state_precedence(documents)
    _assert_machine_mismatch_blob(documents)
    _assert_reporting_schema(documents)
    _assert_reporting_emit(documents, tmp_path)
    _assert_reporting_refusals(documents)
    assert perf_counter() - started <= _CONTRACT_WINDOW_S
