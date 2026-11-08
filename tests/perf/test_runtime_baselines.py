"""X5a contract node: per-metric baseline validation before promotion.

The node pins the validation contract of the baseline layer. A run of a cell's
SHARED run set is eligible only when the CONJUNCTION of every co-measured row's
requirements holds — its floors, its chunks coverage (a record with missing
coverage is ineligible: dimension completeness), its sample validity and the
invalid-sample cap — and the run's own identity: fixture hash, dimensions,
observer mode and the pinned ``chartreux.runtime_perf.v1`` metric version, the
four required machine-context fields, and BOTH bracketing quiet stamps.

Selection is the latest five eligible runs ordered by run START time with the run
id as the tie-breaker, and this node proves the selection never consults a
measured value: runs are built with deliberately favorable and unfavorable
series on both sides of the cut, and the cut follows time only.

Closure is bounded (``eligible_complete`` at five eligible runs or
``insufficient_data`` at ten attempted runs), every attempt is preserved with its
status, and the promotion gate recomputes the closure and the selection from the
attempt list so a committed baseline cannot drop an attempt or substitute a
favorable five.

Nothing here measures anything: every run record is synthetic and no cell is
executed, so the node stays inside the plan's in-suite contract-node window.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import pytest

from tests.perf._runtime_baselines import (
    BASELINE_SCHEMA_ID,
    BASELINES_DIRNAME,
    COMPARISON_FILENAME,
    COMPARISON_KIND,
    COMPARISON_LABEL_PRECEDENCE,
    COMPARISON_RUN_DIR_ENV,
    COMPARISON_SCHEMA_ID,
    DEFAULT_DIRECTION,
    DIRECTION_HIGHER_IS_BETTER,
    INVALID_SAMPLE_CAP,
    MACHINE_FIELDS,
    MACHINE_MISMATCH_RULE,
    MAX_ATTEMPTS,
    OUTCOME_DORMANT,
    OUTCOME_FAIL,
    OUTCOME_INELIGIBLE,
    OUTCOME_LABELS,
    OUTCOME_PASS,
    QUIET_LOADAVG_FACTOR,
    QUIET_RULE,
    REASON_CELL_MISMATCH,
    REASON_CHUNKS_INCOMPLETE,
    REASON_CONTRACT_SCALE,
    REASON_DEGENERATE_THRESHOLD,
    REASON_DIMENSIONS_MISMATCH,
    REASON_DIMENSIONS_MISSING,
    REASON_FIXTURE_MISMATCH,
    REASON_FIXTURE_MISSING,
    REASON_INVALID_SAMPLE_CAP,
    REASON_MACHINE_FIELD,
    REASON_MACHINE_MISMATCH,
    REASON_METRIC_VERSION,
    REASON_MISSING_DETERMINANTS,
    REASON_OBSERVER_FLOW,
    REASON_ORDER,
    REASON_QUIET_INVALID,
    REASON_QUIET_MISSING,
    REASON_QUIET_NOT_QUIET,
    REASON_ROW_RECORD_MISMATCH,
    REASON_ROW_RECORD_MISSING,
    REASON_RUN_INELIGIBILITY,
    REASON_RUN_MODE,
    REASON_THRESHOLD_FAIL,
    REASON_THRESHOLD_PASS,
    SELECTED_RUNS,
    SELECTION_RULE,
    STATUS_ELIGIBLE,
    STATUS_INELIGIBLE,
    THRESHOLD_MAD_FACTOR,
    THRESHOLD_RELATIVE_FACTOR,
    THRESHOLD_RULE,
    AttemptStatus,
    BaselineError,
    CellBaselineSpec,
    ComparisonError,
    RunIdentity,
    RunVerdict,
    build_baseline,
    build_baseline_run,
    build_comparison,
    cell_baseline_spec,
    cell_campaign_ids,
    committed_baseline_path,
    committed_baselines_dir,
    comparison_machine_mismatch,
    comparison_run_dir,
    comparison_write_path,
    evaluate_closure,
    evaluate_comparison_row,
    evaluate_shared_run,
    load_committed_baseline,
    machine_fields_missing,
    machine_mismatch_fields,
    mad,
    median,
    quiet_stamp,
    quiet_window,
    select_latest_eligible,
    shared_run,
    spec_from_baseline,
    threshold_value,
    validate_baseline,
    write_comparison,
)
from tests.perf._runtime_report import (
    CELL_FRESH_HEADLESS,
    CELL_LOADED_HEADLESS,
    CELL_PERMISSION,
    CELL_PILOT_STREAMING,
    CELL_SAVE_LOADED,
    CELL_SOAK_LATENCY,
    CELL_SOAK_MEMORY,
    CELL_STORM,
    GATED_CAMPAIGN_IDS,
    METRIC_VERSION,
    ROW_SPECS,
    SaveTimings,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="the runtime perf baseline contract is Linux-only machinery",
    ),
]

_PERF_DIR = Path(__file__).resolve().parent

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_BASELINE_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_CORES = 12
_START = "2026-11-08T00:00:00Z"

#: The committed machine context a baseline cell is collected on.
_MACHINE: dict[str, Any] = {
    "cpu_model": "AMD Ryzen 9 5900X 12-Core Processor",
    "core_count": _CORES,
    "python_version": "3.12.13",
    "platform": "Linux-7.3.0-9-generic-x86_64-with-glibc2.44",
}

#: The committed dimensions of a cell run (agent count, payload, chunk and
#: injected-delay schedule), as a capture records them.
_DIMENSIONS: dict[str, Any] = {
    "root_turns": 100,
    "history_messages": 4096,
    "child_agents": 0,
    "injected_first_byte_ms": 10.0,
    "injected_chunk_count": 4,
    "injected_chunk_gap_ms": 5.0,
    "mode": "measure",
}

#: Each gated row's Appendix B floor, at exactly its floor value, and the series
#: length its record carries (the row's gated series, not always the op count).
_ROW_RECIPE: dict[str, dict[str, Any]] = {
    "A1": {"measured_ops": 100, "warmup_ops": 10, "unit_counts": {}, "series": 100},
    "A2": {"measured_ops": 100, "warmup_ops": 10, "unit_counts": {}, "series": 100},
    "A3": {
        "measured_ops": 100,
        "warmup_ops": 10,
        "unit_counts": {"chunk_gaps": 300},
        "series": 300,
    },
    "A4": {"measured_ops": 1000, "warmup_ops": 10, "unit_counts": {}, "series": 1000},
    "B1": {"measured_ops": 100, "warmup_ops": 10, "unit_counts": {}, "series": 100},
    "B2": {"measured_ops": 1000, "warmup_ops": 10, "unit_counts": {}, "series": 1000},
    "B3": {
        "measured_ops": 100,
        "warmup_ops": 10,
        "unit_counts": {"root_turns": 100},
        "series": 100,
    },
    "B4": {
        "measured_ops": 100,
        "warmup_ops": 10,
        "unit_counts": {"root_turns": 100},
        "series": 100,
    },
    "C1": {
        "measured_ops": 100,
        "warmup_ops": 10,
        "unit_counts": {"saves": 100},
        "series": 100,
        "save": True,
    },
    "C2": {
        "measured_ops": 100,
        "warmup_ops": 10,
        "unit_counts": {"saves": 100},
        "series": 100,
        "save": True,
    },
    "C3": {"measured_ops": 1000, "warmup_ops": 10, "unit_counts": {}, "series": 1000},
    "D1": {
        "measured_ops": 100,
        "warmup_ops": 10,
        "unit_counts": {"render_flushes": 100, "turns": 100},
        "series": 100,
    },
    "E1": {
        "measured_ops": 100,
        "warmup_ops": 10,
        "unit_counts": {"storms": 100},
        "series": 100,
    },
    "E2": {
        "measured_ops": 100,
        "warmup_ops": 10,
        "unit_counts": {"permission_round_trips": 100},
        "series": 100,
    },
    "F1": {"measured_ops": 100, "warmup_ops": 300, "unit_counts": {}, "series": 100},
    "F2": {
        "measured_ops": 1000,
        "warmup_ops": 300,
        "unit_counts": {"late_window_heartbeat_rows": 1000},
        "series": 1000,
    },
    "G1": {
        "measured_ops": 101,
        "warmup_ops": 10,
        "unit_counts": {"rss_samples": 201, "window_slopes": 101},
        "series": 101,
        "slopes": True,
    },
}


# ---------------------------------------------------------------------------
# Fixtures (synthetic run records only: nothing is measured)
# ---------------------------------------------------------------------------


def _load_artifact(name: str) -> dict[str, Any]:
    return json.loads((_PERF_DIR / name).read_text(encoding="utf-8"))


def _series(length: int, base: float = 1.0) -> list[float]:
    """Return a deterministic series whose nearest-rank p95 is base + 0.4."""
    return [round(base + (index % 5) * 0.1, 4) for index in range(length)]


def _quiet_window(loadavg_1m: float = 0.4) -> dict[str, Any]:
    """Return a quiet bracketing window at ``loadavg_1m`` on both stamps."""
    return quiet_window(
        quiet_stamp(loadavg_1m, _CORES), quiet_stamp(loadavg_1m, _CORES)
    )


def _spec_for(
    cell: str = CELL_FRESH_HEADLESS,
    *,
    fixture_hash: str | None = None,
    workload_hash: str | None = None,
) -> CellBaselineSpec:
    """Return one cell's committed baseline identity."""
    return cell_baseline_spec(
        cell,
        fixture_hash=(f"fixture-hash-{cell}" if fixture_hash is None else fixture_hash),
        dimensions=_DIMENSIONS,
        machine=_MACHINE,
        workload_hash=workload_hash,
    )


def _row_kwargs(campaign_id: str, *, base: float = 1.0) -> dict[str, Any]:
    """Return one row record's floor counts and its gated series."""
    recipe = _ROW_RECIPE[campaign_id]
    length = int(recipe["series"])
    kwargs: dict[str, Any] = {
        "measured_ops": int(recipe["measured_ops"]),
        "warmup_ops": int(recipe["warmup_ops"]),
        "unit_counts": dict(recipe["unit_counts"]),
    }
    if recipe.get("save"):
        kwargs["save_timings"] = SaveTimings(
            worker_scan_ms=_series(length, base),
            lock_held_await_ms=_series(length, base + 1.0),
            worker_thread_ident=4242,
            loop_thread_ident=4240,
        )
    elif recipe.get("slopes"):
        kwargs["slopes_signed"] = [float(index) - 50.0 for index in range(length)]
    else:
        kwargs["samples"] = _series(length, base)
    return kwargs


def _row_record(
    campaign_id: str,
    *,
    run_id: str,
    run_start_utc: str,
    spec: CellBaselineSpec,
    base: float = 1.0,
    quiet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one row record of one run of the spec's cell."""
    return build_baseline_run(
        campaign_id,
        run_id=run_id,
        run_start_utc=run_start_utc,
        fixture_hash=spec.fixture_hash,
        machine=dict(spec.machine),
        quiet=_quiet_window() if quiet is None else dict(quiet),
        dimensions=dict(spec.dimensions),
        workload_hash=spec.workload_hash,
        **_row_kwargs(campaign_id, base=base),
    )


def _records_for(
    spec: CellBaselineSpec,
    *,
    run_id: str = "run-1",
    run_start_utc: str = _START,
    base: float = 1.0,
    quiet: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build every co-measured row record of one run of the spec's cell."""
    return {
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


def _verdict(
    spec: CellBaselineSpec,
    records: Mapping[str, dict[str, Any]],
    *,
    run_id: str = "run-1",
) -> RunVerdict:
    """Evaluate one shared run assembled from the supplied row records."""
    return evaluate_shared_run(spec, shared_run(records, run_key=run_id))


def _assert_error(call: Callable[[], object], reason: str) -> None:
    with pytest.raises(BaselineError) as excinfo:
        call()
    assert excinfo.value.reason == reason, excinfo.value.detail


def _shared_cell_runs(
    spec: CellBaselineSpec,
    starts: Mapping[str, str],
    *,
    bases: Mapping[str, float] | None = None,
    noisy: frozenset[str] = frozenset(),
) -> list[Any]:
    """Build one shared run per (run id, start) pair, optionally noisy."""
    runs = []
    for run_id, start in starts.items():
        base = 1.0 if bases is None else bases[run_id]
        quiet = (
            quiet_window(quiet_stamp(0.4, _CORES), quiet_stamp(8.0, _CORES))
            if run_id in noisy
            else _quiet_window()
        )
        records = _records_for(
            spec, run_id=run_id, run_start_utc=start, base=base, quiet=quiet
        )
        runs.append(shared_run(records, run_key=run_id))
    return runs


# ---------------------------------------------------------------------------
# Pinned contract constants and cell row sets
# ---------------------------------------------------------------------------


def _assert_pinned_constants(registry: dict[str, Any]) -> None:
    assert BASELINE_SCHEMA_ID == "chartreux.runtime_baseline.v1"
    assert BASELINE_SCHEMA_ID != METRIC_VERSION
    assert METRIC_VERSION == "chartreux.runtime_perf.v1"
    assert SELECTED_RUNS == 5
    assert MAX_ATTEMPTS == 10
    assert INVALID_SAMPLE_CAP == 0.10
    assert QUIET_LOADAVG_FACTOR == 0.5
    assert MACHINE_FIELDS == ("cpu_model", "core_count", "python_version", "platform")
    assert QUIET_RULE == "loadavg_1m <= 0.5 * core_count at both bracketing stamps"
    assert "run start" in SELECTION_RULE and "run id" in SELECTION_RULE
    for word in ("p95", "median", "measured value", "latency"):
        assert word not in SELECTION_RULE
    assert "any required machine-context field" in MACHINE_MISMATCH_RULE
    assert tuple(_ROW_RECIPE) == GATED_CAMPAIGN_IDS

    # The cell row sets come from the committed registry, not from a local guess.
    by_cell: dict[str, list[str]] = {}
    for entry in registry["entries"]:
        by_cell.setdefault(entry["cell"], []).append(entry["campaign_id"])
    assert len(by_cell) == 8
    for cell, campaign_ids in by_cell.items():
        assert list(cell_campaign_ids(cell)) == campaign_ids
        for campaign_id in campaign_ids:
            assert ROW_SPECS[campaign_id].cell == cell
    assert cell_campaign_ids(CELL_FRESH_HEADLESS) == ("A1", "A2", "A3", "A4")
    assert cell_campaign_ids(CELL_LOADED_HEADLESS) == ("B1", "B2", "B3", "B4")
    assert cell_campaign_ids(CELL_SAVE_LOADED) == ("C1", "C2", "C3")
    assert cell_campaign_ids(CELL_SOAK_LATENCY) == ("F1", "F2")
    assert cell_campaign_ids(CELL_PILOT_STREAMING) == ("D1",)
    assert cell_campaign_ids(CELL_STORM) == ("E1",)
    assert cell_campaign_ids(CELL_PERMISSION) == ("E2",)
    assert cell_campaign_ids(CELL_SOAK_MEMORY) == ("G1",)
    assert cell_campaign_ids("not_a_cell") == ()


# ---------------------------------------------------------------------------
# Machine context and quiet stamps
# ---------------------------------------------------------------------------


def _assert_machine_context() -> None:
    assert machine_fields_missing(_MACHINE) == ()
    assert machine_fields_missing(None) == MACHINE_FIELDS
    assert machine_fields_missing({}) == MACHINE_FIELDS
    for field in MACHINE_FIELDS:
        assert machine_fields_missing({**_MACHINE, field: None}) == (field,)
    assert machine_fields_missing({**_MACHINE, "cpu_model": "  "}) == ("cpu_model",)
    assert machine_fields_missing({**_MACHINE, "core_count": 0}) == ("core_count",)
    assert machine_fields_missing({**_MACHINE, "core_count": "12"}) == ("core_count",)

    assert machine_mismatch_fields(_MACHINE, _MACHINE) == ()
    assert machine_mismatch_fields(
        _MACHINE, {**_MACHINE, "platform": "Linux-other"}
    ) == ("platform",)
    assert machine_mismatch_fields(_MACHINE, {**_MACHINE, "core_count": 8}) == (
        "core_count",
    )
    # Inequality in ANY field is a mismatch, never core_count alone.
    both = machine_mismatch_fields(
        _MACHINE, {**_MACHINE, "core_count": 8, "python_version": "3.13.0"}
    )
    assert both == ("core_count", "python_version")
    assert machine_mismatch_fields({}, _MACHINE) == MACHINE_FIELDS


def _assert_quiet_stamps() -> None:
    stamp = quiet_stamp(3.0, _CORES)
    assert stamp == {
        "loadavg_1m": 3.0,
        "core_count": _CORES,
        "bound": 6.0,
        "quiet": True,
    }
    # Equality at the core-count-relative bound is quiet.
    assert quiet_stamp(6.0, _CORES)["quiet"] is True
    assert quiet_stamp(6.001, _CORES)["quiet"] is False
    assert quiet_stamp(None, _CORES) == {
        "loadavg_1m": None,
        "core_count": _CORES,
        "bound": 6.0,
        "quiet": False,
    }
    window = _quiet_window()
    assert window["quiet"] is True
    assert window["rule"] == QUIET_RULE
    assert window["before"]["bound"] == QUIET_LOADAVG_FACTOR * _CORES


def _assert_run_level_refusals() -> None:
    spec = _spec_for()
    happy = _verdict(spec, _records_for(spec))
    assert happy.eligible is True
    assert happy.reasons == ()
    assert happy.failing_row is None
    assert happy.machine_mismatch_fields == ()
    assert happy.machine_fields_missing == ()

    # Both bracketing stamps are required: a noisy after-stamp kills the run.
    noisy_after = _verdict(
        spec,
        _records_for(
            spec, quiet=quiet_window(quiet_stamp(0.4, _CORES), quiet_stamp(8.0, _CORES))
        ),
    )
    assert noisy_after.eligible is False
    assert REASON_QUIET_NOT_QUIET in noisy_after.reasons
    assert REASON_QUIET_NOT_QUIET in noisy_after.run_reasons
    assert noisy_after.failing_row is None

    # A stamp that claims quiet while the sample is above the bound is refused:
    # the recorded verdict is never trusted.
    lying_after = dict(quiet_stamp(8.0, _CORES))
    lying_after["quiet"] = True
    lying = _verdict(
        spec,
        _records_for(
            spec,
            quiet={
                "before": quiet_stamp(0.4, _CORES),
                "after": lying_after,
                "quiet": True,
                "rule": QUIET_RULE,
            },
        ),
    )
    assert REASON_QUIET_INVALID in lying.reasons

    # A stamp whose bound is not core-count-relative is malformed.
    wrong_bound = dict(quiet_stamp(0.4, _CORES))
    wrong_bound["bound"] = 99.0
    bad_bound = _verdict(
        spec,
        _records_for(
            spec,
            quiet={
                "before": wrong_bound,
                "after": quiet_stamp(0.4, _CORES),
                "quiet": True,
                "rule": QUIET_RULE,
            },
        ),
    )
    assert REASON_QUIET_INVALID in bad_bound.reasons

    # A stamp sampled against another core count cannot certify this machine.
    other_cores = _verdict(
        spec,
        _records_for(
            spec,
            quiet={
                "before": quiet_stamp(0.4, _CORES + 4),
                "after": quiet_stamp(0.4, _CORES),
                "quiet": True,
                "rule": QUIET_RULE,
            },
        ),
    )
    assert REASON_QUIET_INVALID in other_cores.reasons

    # An unavailable loadavg certifies nothing, and is not a pass.
    unavailable = _verdict(
        spec,
        _records_for(
            spec,
            quiet=quiet_window(quiet_stamp(0.4, _CORES), quiet_stamp(None, _CORES)),
        ),
    )
    assert REASON_QUIET_NOT_QUIET in unavailable.reasons

    # One bracketing stamp missing is a missing stamp, not a silent pass.
    partial = _records_for(spec)
    for record in partial.values():
        record["quiet"] = {
            "before": quiet_stamp(0.4, _CORES),
            "quiet": True,
            "rule": QUIET_RULE,
        }
    assert REASON_QUIET_MISSING in _verdict(spec, partial).reasons

    no_window = _records_for(spec)
    for record in no_window.values():
        del record["quiet"]
    assert REASON_QUIET_MISSING in _verdict(spec, no_window).reasons

    # A required machine field is mandatory on every run record.
    missing_machine = _records_for(spec)
    for record in missing_machine.values():
        record["machine"] = {**_MACHINE, "cpu_model": None}
    missing_verdict = _verdict(spec, missing_machine)
    assert missing_verdict.eligible is False
    assert REASON_MACHINE_FIELD in missing_verdict.reasons
    assert missing_verdict.machine_fields_missing == ("cpu_model",)

    # A machine that DIFFERS from the committed baseline is recorded, not
    # silently accepted: the run stays eligible and the mismatch is reported,
    # because machine_mismatch is a cell dormancy reason, never ineligibility.
    different_platform = _records_for(spec)
    for record in different_platform.values():
        record["machine"] = {**_MACHINE, "platform": "Linux-9.9.9-other"}
    mismatch = _verdict(spec, different_platform)
    assert mismatch.eligible is True
    assert mismatch.machine_mismatch_fields == ("platform",)


# ---------------------------------------------------------------------------
# Shared-run assembly
# ---------------------------------------------------------------------------


def _assert_shared_run_assembly() -> None:
    spec = _spec_for()
    records = _records_for(spec)
    derived = shared_run(records)
    assert derived.run_key == "run-1"
    assert derived.run_start_utc == _START
    assert derived.identity.run_id == "run-1"

    # Producers may suffix the row run id (X2 does): a key is then required.
    suffixed = {
        campaign_id: {**record, "run_id": f"run-1-{campaign_id.lower()}"}
        for campaign_id, record in records.items()
    }
    _assert_error(lambda: shared_run(suffixed), "run_key_required")
    keyed = shared_run(suffixed, run_key="run-1")
    assert keyed.run_key == "run-1"
    assert evaluate_shared_run(spec, keyed).eligible is True

    mixed = {**records, "A4": {**records["A4"], "cell": CELL_LOADED_HEADLESS}}
    _assert_error(lambda: shared_run(mixed), "mixed_cell_records")

    drifted_start = {
        **records,
        "A2": {**records["A2"], "run_start_utc": "2026-11-08T00:05:00Z"},
    }
    _assert_error(lambda: shared_run(drifted_start), "inconsistent_run_start")

    naive = {
        campaign_id: {**record, "run_start_utc": "2026-11-08T00:00:00"}
        for campaign_id, record in records.items()
    }
    _assert_error(lambda: shared_run(naive), "run_start_naive")

    unorderable = {
        campaign_id: {**record, "run_start_utc": "not-a-time"}
        for campaign_id, record in records.items()
    }
    _assert_error(lambda: shared_run(unorderable), "run_start_invalid")

    _assert_error(lambda: shared_run({}), "empty_run")
    _assert_error(lambda: shared_run(records, run_key="  "), "run_key_required")


# ---------------------------------------------------------------------------
# Identity of fixture, dimensions, observer mode and metric version
# ---------------------------------------------------------------------------


def _assert_identity_refusals() -> None:
    spec = _spec_for()

    def verdict_after(mutate: Callable[[str, dict[str, Any]], None]) -> RunVerdict:
        records = copy.deepcopy(_records_for(spec))
        for campaign_id, record in records.items():
            mutate(campaign_id, record)
        return _verdict(spec, records)

    def drop(key: str) -> Callable[[str, dict[str, Any]], None]:
        def mutate(_campaign_id: str, record: dict[str, Any]) -> None:
            record.pop(key, None)

        return mutate

    assert (
        REASON_FIXTURE_MISMATCH
        in verdict_after(
            lambda _c, record: record.update({"fixture_hash": "fixture-hash-other"})
        ).reasons
    )
    assert REASON_FIXTURE_MISSING in verdict_after(drop("fixture_hash")).reasons
    assert REASON_DIMENSIONS_MISSING in verdict_after(drop("dimensions")).reasons

    def shorten_dimensions(_campaign_id: str, record: dict[str, Any]) -> None:
        record["dimensions"] = {**_DIMENSIONS, "root_turns": 99}

    dimensions_verdict = verdict_after(shorten_dimensions)
    assert dimensions_verdict.eligible is False
    assert REASON_DIMENSIONS_MISMATCH in dimensions_verdict.reasons

    # The workload digest carries the seed and the response shape: drift in it
    # is dimension drift and makes the run ineligible for the committed cell.
    workload_spec = _spec_for(workload_hash="workload-hash-committed")
    committed = _records_for(workload_spec)
    assert evaluate_shared_run(workload_spec, shared_run(committed)).eligible is True
    drifted = {
        campaign_id: {**record, "workload_hash": "workload-hash-other"}
        for campaign_id, record in committed.items()
    }
    assert REASON_DIMENSIONS_MISMATCH in _verdict(workload_spec, drifted).reasons
    undeclared = {
        campaign_id: {k: v for k, v in record.items() if k != "workload_hash"}
        for campaign_id, record in committed.items()
    }
    assert REASON_DIMENSIONS_MISSING in _verdict(workload_spec, undeclared).reasons

    # The metric version is pinned to the report schema, deliberately distinct
    # from the determinants document's own schema field.
    determinants_schema = "chartreux.runtime_metric_determinants.v1"
    assert determinants_schema != METRIC_VERSION
    version_verdict = verdict_after(
        lambda _c, record: record.update({"metric_version": determinants_schema})
    )
    assert version_verdict.eligible is False
    assert REASON_METRIC_VERSION in version_verdict.reasons

    assert (
        REASON_RUN_MODE
        in verdict_after(lambda _c, record: record.update({"mode": "profile"})).reasons
    )

    def contract_scale(_campaign_id: str, record: dict[str, Any]) -> None:
        record["dimensions"] = {**_DIMENSIONS, "mode": "contract"}

    assert REASON_CONTRACT_SCALE in verdict_after(contract_scale).reasons

    assert (
        REASON_OBSERVER_FLOW
        in verdict_after(
            lambda _c, record: record.update({
                "observer": {"mode": "observer", "control_flow_changed": True}
            })
        ).reasons
    )
    # A control-flow-neutral observer declaration is accepted.
    neutral = verdict_after(
        lambda _c, record: record.update({
            "observer": {"mode": "observer", "control_flow_changed": False}
        })
    )
    assert neutral.eligible is True

    assert (
        REASON_CELL_MISMATCH
        in verdict_after(
            lambda _c, record: record.update({"cell": CELL_LOADED_HEADLESS})
        ).reasons
    )


# ---------------------------------------------------------------------------
# The shared-set conjunction
# ---------------------------------------------------------------------------


def _assert_conjunction() -> None:
    spec = _spec_for()

    # A missing co-measured row makes the run ineligible for ALL the cell's
    # campaign IDs, and the failing row is recorded — never silently dropped.
    missing_row = _records_for(spec)
    del missing_row["A3"]
    verdict = _verdict(spec, missing_row)
    assert verdict.eligible is False
    assert verdict.failing_row == "A3"
    assert verdict.row_reasons["A3"] == (REASON_ROW_RECORD_MISSING,)

    # A short heartbeat run fails A4's floor alone and takes the whole run down.
    short_heartbeat = _records_for(spec)
    short_heartbeat["A4"] = build_baseline_run(
        "A4",
        run_id="run-1",
        run_start_utc=_START,
        fixture_hash=spec.fixture_hash,
        machine=dict(spec.machine),
        quiet=_quiet_window(),
        dimensions=dict(spec.dimensions),
        measured_ops=999,
        warmup_ops=10,
        samples=_series(999),
    )
    verdict = _verdict(spec, short_heartbeat)
    assert verdict.eligible is False
    assert verdict.failing_row == "A4"
    assert "ops_floor_measured" in verdict.row_reasons["A4"]
    assert "A1" not in verdict.row_reasons

    # Missing chunks coverage is ineligibility: dimension completeness is part
    # of the conjunction, and the coverage reason names the missing count.
    no_gaps = _records_for(spec)
    no_gaps["A3"] = build_baseline_run(
        "A3",
        run_id="run-1",
        run_start_utc=_START,
        fixture_hash=spec.fixture_hash,
        machine=dict(spec.machine),
        quiet=_quiet_window(),
        dimensions=dict(spec.dimensions),
        measured_ops=100,
        warmup_ops=10,
        samples=_series(300),
    )
    verdict = _verdict(spec, no_gaps)
    assert verdict.eligible is False
    assert verdict.failing_row == "A3"
    assert REASON_CHUNKS_INCOMPLETE in verdict.row_reasons["A3"]
    assert "coverage_missing_chunk_gaps" in verdict.row_reasons["A3"]

    # A record filed under the wrong row is refused, not accepted as that row.
    misfiled = _records_for(spec)
    misfiled["A3"] = misfiled["A2"]
    verdict = _verdict(spec, misfiled)
    assert verdict.eligible is False
    assert REASON_ROW_RECORD_MISMATCH in verdict.row_reasons["A3"]

    # The conjunction is cell-wide: the same run is ineligible for every row.
    document = build_baseline(spec, [shared_run(missing_row, run_key="run-1")])
    assert document["attempts"][0]["status"] == STATUS_INELIGIBLE
    for campaign_id in spec.campaign_ids:
        assert document["campaigns"][campaign_id]["run_ids"] == []


def _assert_invalid_sample_cap() -> None:
    spec = _spec_for()

    def with_invalid(count: int) -> RunVerdict:
        records = _records_for(spec)
        records["A1"]["cell_eligibility"] = {"invalid_samples": count}
        return _verdict(spec, records)

    # 12 invalid of 112 is above the 10 % cap; 11 of 111 stays within it.
    over = with_invalid(12)
    assert over.eligible is False
    assert over.failing_row == "A1"
    assert REASON_INVALID_SAMPLE_CAP in over.row_reasons["A1"]

    within = with_invalid(11)
    assert within.eligible is True
    assert within.reasons == ()

    # A malformed count cannot be evaluated, so the run is not eligible.
    malformed = _records_for(spec)
    malformed["A2"]["cell_eligibility"] = {"invalid_samples": "12"}
    assert _verdict(spec, malformed).eligible is False


# ---------------------------------------------------------------------------
# Selection: by run start then run id, NEVER by a measured value
# ---------------------------------------------------------------------------


def _assert_selection_never_reads_a_value() -> None:
    spec = _spec_for()

    # Seven eligible runs. The two EARLIEST carry a far better series (0.01 ms)
    # than the five later ones (900 ms). Selection must still take the later
    # five: a favorable subset is never selected.
    starts = {
        "run-1": "2026-11-08T00:01:00Z",
        "run-2": "2026-11-08T00:02:00Z",
        "run-3": "2026-11-08T00:03:00Z",
        "run-4": "2026-11-08T00:04:00Z",
        "run-5": "2026-11-08T00:05:00Z",
        "run-6": "2026-11-08T00:06:00Z",
        "run-7": "2026-11-08T00:07:00Z",
    }
    bases = {f"run-{index}": (0.01 if index <= 2 else 900.0) for index in range(1, 8)}
    runs = _shared_cell_runs(spec, starts, bases=bases)
    document = build_baseline(spec, runs)
    closure = document["closure"]
    expected = ("run-3", "run-4", "run-5", "run-6", "run-7")
    assert closure["status"] == "eligible_complete"
    assert tuple(closure["selected_run_ids"]) == expected
    assert closure["selected_run_count"] == SELECTED_RUNS
    assert closure["eligible_run_count"] == 7
    assert closure["reason"] is None
    # Every attempt survives, in capture order, including the two unselected.
    assert document["attempt_count"] == 7
    assert tuple(attempt["run_id"] for attempt in document["attempts"]) == tuple(starts)
    assert all(attempt["status"] == STATUS_ELIGIBLE for attempt in document["attempts"])
    # The per-campaign run-p95s are the SELECTED runs', so they are the large
    # series: selection followed time, not the measurement.
    for campaign_id in spec.campaign_ids:
        block = document["campaigns"][campaign_id]
        assert tuple(block["run_ids"]) == expected
        assert len(block["run_p95s"]) == SELECTED_RUNS
        assert all(value > 100.0 for value in block["run_p95s"])
        assert block["null_reason"] is None

    # The mirror image: when the LATEST five are the favorable ones they ARE
    # selected. Same 7 runs, bases inverted — the cut never moves.
    inverted = {
        f"run-{index}": (900.0 if index <= 2 else 0.01) for index in range(1, 8)
    }
    flipped = build_baseline(spec, _shared_cell_runs(spec, starts, bases=inverted))
    assert tuple(flipped["closure"]["selected_run_ids"]) == expected
    for campaign_id in spec.campaign_ids:
        assert all(
            value < 1.0 for value in flipped["campaigns"][campaign_id]["run_p95s"]
        )

    # Capture order is irrelevant: the selection is ordered by start time.
    shuffled = {
        "run-7": starts["run-7"],
        "run-1": starts["run-1"],
        "run-5": starts["run-5"],
        "run-2": starts["run-2"],
        "run-6": starts["run-6"],
        "run-3": starts["run-3"],
        "run-4": starts["run-4"],
    }
    unordered = build_baseline(spec, _shared_cell_runs(spec, shuffled, bases=bases))
    assert tuple(unordered["closure"]["selected_run_ids"]) == expected
    assert tuple(unordered["closure"]["eligible_run_ids"]) == tuple(sorted(starts))
    assert tuple(attempt["run_id"] for attempt in unordered["attempts"]) == tuple(
        shuffled
    )

    # The tie-breaker is the run id: two runs starting at the same instant are
    # ordered by id, so the higher id is the later one and outlives the cut.
    tied = {
        "run-a": "2026-11-08T00:01:00Z",
        "run-b": "2026-11-08T00:01:00Z",
        "run-c": "2026-11-08T00:02:00Z",
        "run-d": "2026-11-08T00:03:00Z",
        "run-e": "2026-11-08T00:04:00Z",
        "run-f": "2026-11-08T00:05:00Z",
    }
    tied_doc = build_baseline(spec, _shared_cell_runs(spec, tied))
    assert tuple(tied_doc["closure"]["selected_run_ids"]) == (
        "run-b",
        "run-c",
        "run-d",
        "run-e",
        "run-f",
    )
    assert "run-a" not in tied_doc["closure"]["selected_run_ids"]

    # Selection cannot even see a value: it takes identities only, and fewer
    # than five identities select nothing at all.
    assert select_latest_eligible([]) == ()
    four = [RunIdentity(run_id, starts[run_id]) for run_id in list(starts)[:4]]
    assert select_latest_eligible(four) == ()
    five = [*four, RunIdentity("run-5", starts["run-5"])]
    assert select_latest_eligible(five) == ("run-1", "run-2", "run-3", "run-4", "run-5")


# ---------------------------------------------------------------------------
# Closure and attempt preservation
# ---------------------------------------------------------------------------


def _assert_closure_states() -> None:
    spec = _spec_for(CELL_STORM)
    assert cell_campaign_ids(CELL_STORM) == ("E1",)

    starts = {
        f"run-{index}": f"2026-11-08T00:{index:02d}:00Z" for index in range(1, 11)
    }
    starts_eleven = {**starts, "run-11": "2026-11-08T00:11:00Z"}
    ordered = list(starts)

    # The closure object itself: open below the cap, insufficient_data at the
    # cap without five eligible runs, and a hard refusal beyond the cap.
    def attempt_statuses(count: int, *, eligible: int = 0) -> list[AttemptStatus]:
        return [
            AttemptStatus(
                run_id=f"run-{index}",
                run_start_utc=f"2026-11-08T00:{index:02d}:00Z",
                eligible=index - 1 < eligible,
            )
            for index in range(1, count + 1)
        ]

    empty_closure = evaluate_closure([])
    assert empty_closure.status is None
    assert empty_closure.reason == "campaign_open"
    assert evaluate_closure(attempt_statuses(9)).status is None
    assert evaluate_closure(attempt_statuses(10)).status == "insufficient_data"
    assert evaluate_closure(attempt_statuses(10, eligible=4)).eligible_run_count == 4
    assert (
        evaluate_closure(attempt_statuses(10, eligible=5)).status == "eligible_complete"
    )
    _assert_error(
        lambda: evaluate_closure(attempt_statuses(11)), "attempt_cap_exceeded"
    )

    # Ten attempted runs, none eligible: the honest insufficient_data closure.
    all_noisy = build_baseline(
        spec, _shared_cell_runs(spec, starts, noisy=frozenset(starts))
    )
    assert all_noisy["closure"]["status"] == "insufficient_data"
    assert all_noisy["closure"]["reason"] == "insufficient_baseline_data"
    assert all_noisy["closure"]["selected_run_ids"] == []
    assert all_noisy["closure"]["eligible_run_ids"] == []
    assert all_noisy["attempt_count"] == MAX_ATTEMPTS
    for attempt in all_noisy["attempts"]:
        assert attempt["status"] == STATUS_INELIGIBLE
        assert REASON_QUIET_NOT_QUIET in attempt["reasons"]
    block = all_noisy["campaigns"]["E1"]
    assert block["run_ids"] == []
    assert block["run_p95s"] == []
    assert block["null_reason"] == "insufficient_baseline_data"
    validate_baseline(all_noisy, spec)

    # Four eligible of ten attempts: still insufficient_data, and the fewer than
    # five collected run-p95s are carried without a threshold or a selection.
    thin_doc = build_baseline(
        spec, _shared_cell_runs(spec, starts, noisy=frozenset(ordered[4:]))
    )
    closure = thin_doc["closure"]
    assert closure["status"] == "insufficient_data"
    assert closure["eligible_run_count"] == 4
    assert closure["selected_run_ids"] == []
    assert closure["eligible_run_ids"] == ["run-1", "run-2", "run-3", "run-4"]
    block = thin_doc["campaigns"]["E1"]
    assert block["run_ids"] == closure["eligible_run_ids"]
    assert len(block["run_p95s"]) == 4 < SELECTED_RUNS
    assert block["null_reason"] == "insufficient_baseline_data"
    validate_baseline(thin_doc, spec)

    # Five eligible within the cap closes eligible_complete, not insufficient.
    five_ok = frozenset(list(starts)[5:])
    five_doc = build_baseline(spec, _shared_cell_runs(spec, starts, noisy=five_ok))
    assert five_doc["closure"]["status"] == "eligible_complete"
    assert five_doc["closure"]["eligible_run_count"] == 5
    assert tuple(five_doc["closure"]["selected_run_ids"]) == (
        "run-1",
        "run-2",
        "run-3",
        "run-4",
        "run-5",
    )

    # Below the cap without five eligible the campaign is OPEN: NULL status with
    # its reason, refused by the promotion gate.
    open_starts = dict(list(starts.items())[:4])
    open_doc = build_baseline(spec, _shared_cell_runs(spec, open_starts))
    assert open_doc["closure"]["status"] is None
    assert open_doc["closure"]["reason"] == "campaign_open"
    assert open_doc["closure"]["selected_run_ids"] == []
    assert open_doc["campaigns"]["E1"]["null_reason"] == "campaign_open"
    validate_baseline(open_doc, spec, closed_required=False)
    _assert_error(lambda: validate_baseline(open_doc, spec), "campaign_not_closed")

    # The attempt cap is a hard bound, never an unbounded retry.
    _assert_error(
        lambda: build_baseline(spec, _shared_cell_runs(spec, starts_eleven)),
        "attempt_cap_exceeded",
    )
    _assert_error(lambda: build_baseline(spec, []), "empty_run_set")
    duplicated = _shared_cell_runs(spec, dict(list(starts.items())[:2]))
    _assert_error(
        lambda: build_baseline(spec, [duplicated[0], duplicated[0]]),
        "duplicate_attempt",
    )


# ---------------------------------------------------------------------------
# The promotion gate
# ---------------------------------------------------------------------------


def _assert_attempt_preservation_and_gate() -> None:
    spec = _spec_for()
    starts = {f"run-{index}": f"2026-11-08T00:{index:02d}:00Z" for index in range(1, 7)}
    noisy = frozenset({"run-2", "run-4"})
    document = build_baseline(spec, _shared_cell_runs(spec, starts, noisy=noisy))
    assert document["schema"] == BASELINE_SCHEMA_ID
    assert document["cell"] == CELL_FRESH_HEADLESS
    assert document["campaign_ids"] == list(spec.campaign_ids)
    assert document["attempt_count"] == 6
    statuses = {
        attempt["run_id"]: attempt["status"] for attempt in document["attempts"]
    }
    assert statuses == {
        "run-1": STATUS_ELIGIBLE,
        "run-2": STATUS_INELIGIBLE,
        "run-3": STATUS_ELIGIBLE,
        "run-4": STATUS_INELIGIBLE,
        "run-5": STATUS_ELIGIBLE,
        "run-6": STATUS_ELIGIBLE,
    }
    assert document["closure"]["eligible_run_ids"] == [
        "run-1",
        "run-3",
        "run-5",
        "run-6",
    ]
    # Four eligible of six attempts: the campaign cannot decide and says so.
    assert document["closure"]["status"] is None
    validate_baseline(document, spec, closed_required=False)

    # A committed-and-elected document: five eligible of six attempts.
    five_starts = {
        f"run-{index}": f"2026-11-08T00:{index:02d}:00Z" for index in range(1, 7)
    }
    elected = build_baseline(
        spec, _shared_cell_runs(spec, five_starts, noisy=frozenset({"run-3"}))
    )
    assert elected["closure"]["status"] == "eligible_complete"
    assert tuple(elected["closure"]["selected_run_ids"]) == (
        "run-1",
        "run-2",
        "run-4",
        "run-5",
        "run-6",
    )
    validate_baseline(elected, spec)

    # Identity is re-checked at the gate: a different committed cell refuses.
    _assert_error(
        lambda: validate_baseline(elected, _spec_for(fixture_hash="other")),
        "identity_mismatch",
    )

    tampered = copy.deepcopy(elected)
    tampered["schema"] = "chartreux.runtime_baseline.v0"
    _assert_error(lambda: validate_baseline(tampered, spec), "schema_mismatch")

    short_cell = copy.deepcopy(elected)
    short_cell["campaign_ids"] = list(spec.campaign_ids)[:2]
    _assert_error(lambda: validate_baseline(short_cell, spec), "cell_incomplete")

    incomplete_machine = copy.deepcopy(elected)
    incomplete_machine["machine"] = {**_MACHINE, "platform": None}
    _assert_error(
        lambda: validate_baseline(incomplete_machine, spec), "machine_field_missing"
    )

    # An attempt cannot be relabelled eligible: its recorded reasons contradict it.
    relabelled = copy.deepcopy(elected)
    for attempt in relabelled["attempts"]:
        if attempt["status"] == STATUS_INELIGIBLE:
            attempt["status"] = STATUS_ELIGIBLE
    _assert_error(
        lambda: validate_baseline(relabelled, spec), "attempt_status_conflict"
    )

    # An ineligible attempt that records no reason is refused too.
    unexplained = copy.deepcopy(elected)
    for attempt in unexplained["attempts"]:
        if attempt["status"] == STATUS_INELIGIBLE:
            attempt["reasons"] = []
    _assert_error(
        lambda: validate_baseline(unexplained, spec), "attempt_reason_missing"
    )

    # Nor can an attempt's capture data be edited after the fact.
    edited = copy.deepcopy(elected)
    edited["attempts"][0]["run_start_utc"] = "2026-11-08T23:59:59Z"
    _assert_error(lambda: validate_baseline(edited, spec), "attempt_digest_mismatch")

    # Nor can the selection be substituted, even by another five eligible runs.
    substituted = copy.deepcopy(elected)
    substituted["closure"]["selected_run_ids"] = [
        "run-1",
        "run-2",
        "run-3",
        "run-4",
        "run-5",
    ]
    _assert_error(lambda: validate_baseline(substituted, spec), "closure_mismatch")

    substituted_block = copy.deepcopy(elected)
    substituted_block["campaigns"]["A1"]["run_ids"] = [
        "run-1",
        "run-2",
        "run-3",
        "run-4",
        "run-5",
    ]
    _assert_error(
        lambda: validate_baseline(substituted_block, spec), "selection_mismatch"
    )

    # Dropping an attempt changes the closure: the gate recomputes and refuses.
    dropped = copy.deepcopy(elected)
    dropped["attempts"] = [
        attempt for attempt in dropped["attempts"] if attempt["run_id"] != "run-3"
    ]
    dropped["attempt_count"] = len(dropped["attempts"])
    dropped["attempt_digest"] = "0" * 64
    _assert_error(lambda: validate_baseline(dropped, spec), "attempt_digest_mismatch")
    counts_only = copy.deepcopy(elected)
    counts_only["attempt_count"] = 99
    _assert_error(
        lambda: validate_baseline(counts_only, spec), "attempt_count_mismatch"
    )


# ---------------------------------------------------------------------------
# Every gated row validates in its own cell
# ---------------------------------------------------------------------------


def _assert_every_row_eligible() -> None:
    cells = tuple(
        dict.fromkeys(ROW_SPECS[campaign_id].cell for campaign_id in GATED_CAMPAIGN_IDS)
    )
    assert len(cells) == 8
    for cell in cells:
        spec = _spec_for(cell)
        assert all(campaign_id in _ROW_RECIPE for campaign_id in spec.campaign_ids)
        document = build_baseline(
            spec, [shared_run(_records_for(spec), run_key="run-1")]
        )
        assert document["closure"]["status"] is None
        assert document["attempts"][0]["status"] == STATUS_ELIGIBLE
        assert document["attempts"][0]["failing_row"] is None
        validate_baseline(document, spec, closed_required=False)


@pytest.mark.timeout(_BASELINE_TIMEOUT_S)
def test_runtime_baseline_contract() -> None:
    """X5a validates per-metric baselines and never selects on a measured value."""
    started = perf_counter()
    registry = _load_artifact("runtime_metric_registry.json")
    _assert_pinned_constants(registry)
    _assert_machine_context()
    _assert_quiet_stamps()
    _assert_run_level_refusals()
    _assert_shared_run_assembly()
    _assert_identity_refusals()
    _assert_conjunction()
    _assert_invalid_sample_cap()
    _assert_selection_never_reads_a_value()
    _assert_closure_states()
    _assert_attempt_preservation_and_gate()
    _assert_every_row_eligible()
    assert perf_counter() - started <= _CONTRACT_WINDOW_S


# ---------------------------------------------------------------------------
# X5b: the comparison tool — marked outcomes over a committed baseline
# ---------------------------------------------------------------------------

_COMPARISON_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S
_COMPARISON_RUN_START = "2026-11-08T02:00:00Z"


def _hourly_starts(count: int, *, hour: int = 1) -> dict[str, str]:
    """Return ``count`` run starts one minute apart, all in one hour."""
    return {
        f"run-{index}": f"2026-11-08T{hour:02d}:{index:02d}:00Z"
        for index in range(1, count + 1)
    }


def _committed_baseline(
    spec: CellBaselineSpec,
    *,
    count: int = SELECTED_RUNS,
    bases: Mapping[str, float] | None = None,
    noisy: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Build one cell's baseline document from synthetic runs."""
    return build_baseline(
        spec, _shared_cell_runs(spec, _hourly_starts(count), bases=bases, noisy=noisy)
    )


def _comparison_run(
    spec: CellBaselineSpec,
    *,
    base: float = 1.0,
    quiet: Mapping[str, Any] | None = None,
    run_id: str = "cmp-1",
) -> Any:
    """Build one fresh comparison run of the spec's cell."""
    return shared_run(
        _records_for(
            spec,
            run_id=run_id,
            run_start_utc=_COMPARISON_RUN_START,
            base=base,
            quiet=quiet,
        ),
        run_key=run_id,
    )


def _constant_run(
    spec: CellBaselineSpec, campaign_id: str, *, value: float, run_id: str, start: str
) -> Any:
    """Build one single-row run whose gated series is the constant ``value``."""
    recipe = _ROW_RECIPE[campaign_id]
    record = build_baseline_run(
        campaign_id,
        run_id=run_id,
        run_start_utc=start,
        fixture_hash=spec.fixture_hash,
        machine=dict(spec.machine),
        quiet=_quiet_window(),
        dimensions=dict(spec.dimensions),
        measured_ops=int(recipe["measured_ops"]),
        warmup_ops=int(recipe["warmup_ops"]),
        unit_counts=dict(recipe["unit_counts"]),
        samples=[value] * int(recipe["series"]),
    )
    return shared_run({campaign_id: record}, run_key=run_id)


def _assert_comparison_error(call: Callable[[], object], reason: str) -> None:
    with pytest.raises(ComparisonError) as excinfo:
        call()
    assert excinfo.value.reason == reason, excinfo.value.detail


def _assert_comparison_constants(registry: dict[str, Any]) -> None:
    assert COMPARISON_SCHEMA_ID == "chartreux.runtime_comparison.v1"
    assert COMPARISON_SCHEMA_ID not in (BASELINE_SCHEMA_ID, METRIC_VERSION)
    assert COMPARISON_KIND == "comparison_outcome"
    assert COMPARISON_FILENAME == "comparison.json"
    assert COMPARISON_RUN_DIR_ENV == "CHARTREUX_PERF_RUN_DIR"
    assert BASELINES_DIRNAME == "baselines"

    # The committed-baseline read path is pinned to the promoted directory.
    assert committed_baselines_dir() == _PERF_DIR / BASELINES_DIRNAME
    for entry in registry["entries"]:
        path = committed_baseline_path(entry["cell"])
        assert path.parent == _PERF_DIR / BASELINES_DIRNAME
        assert path.name == f"{entry['cell']}.json"
    _assert_comparison_error(
        lambda: committed_baseline_path("not_a_cell"), "unknown_cell"
    )

    # The four marked labels, and the total precedence a record obeys.
    assert OUTCOME_LABELS == (
        OUTCOME_DORMANT,
        OUTCOME_INELIGIBLE,
        OUTCOME_PASS,
        OUTCOME_FAIL,
    )
    assert OUTCOME_DORMANT == "DORMANT"
    assert OUTCOME_INELIGIBLE == "COMPARISON-INELIGIBLE"
    assert {OUTCOME_PASS, OUTCOME_FAIL} == {"ACTIVE-pass", "ACTIVE-fail"}
    precedence = COMPARISON_LABEL_PRECEDENCE
    assert precedence.index(OUTCOME_DORMANT) < precedence.index(OUTCOME_INELIGIBLE)
    assert precedence.index(OUTCOME_INELIGIBLE) < precedence.index(OUTCOME_PASS)

    # The reason order: dormancy tiers (1)-(4), the run-record tier (5), then the
    # threshold labels (6).
    assert REASON_ORDER == (
        "machine_mismatch",
        "insufficient_baseline_data",
        "missing_determinants",
        "degenerate_threshold",
        "run_ineligibility",
        "threshold_fail",
        "threshold_pass",
    )
    assert THRESHOLD_MAD_FACTOR == 6.0
    assert THRESHOLD_RELATIVE_FACTOR == 0.20
    for marker in ("6*D", "0.20*M", "equality"):
        assert marker in THRESHOLD_RULE

    # No gated row declares a higher-is-better margin.
    assert DEFAULT_DIRECTION == "lower_is_better"


def _assert_comparison_statistics() -> None:
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert median([7.0]) == 7.0
    assert mad([1.0, 2.0, 3.0]) == 1.0
    assert mad([2.0, 2.0, 2.0]) == 0.0
    assert mad([1.0, 1.0, 3.0]) == 0.0

    # max(6*D, 0.20*M): the MAD band dominates a spread baseline, the relative
    # band dominates a steady one.
    assert threshold_value(100.0, 10.0) == 160.0
    assert threshold_value(100.0, 1.0) == 120.0
    assert threshold_value(0.0, 0.0) == 0.0
    _assert_comparison_error(lambda: median([]), "median_empty")


def _assert_every_row_defaults_lower_is_better() -> None:
    cells = tuple(
        dict.fromkeys(ROW_SPECS[campaign_id].cell for campaign_id in GATED_CAMPAIGN_IDS)
    )
    assert len(cells) == 8
    for cell in cells:
        spec = _spec_for(cell)
        baseline = build_baseline(
            spec, [shared_run(_records_for(spec), run_key="run-1")]
        )
        document = build_comparison(baseline, _comparison_run(spec))
        assert tuple(document["records"]) == cell_campaign_ids(cell)
        for record in document["records"].values():
            assert record["direction"] == DEFAULT_DIRECTION == "lower_is_better"


def _assert_comparison_outcomes() -> None:
    spec = _spec_for()
    baseline = _committed_baseline(spec)
    assert baseline["closure"]["status"] == "eligible_complete"
    selected = baseline["closure"]["selected_run_ids"]
    assert len(selected) == SELECTED_RUNS

    # The rebuilt identity IS the committed identity: a comparison run is judged
    # by exactly the rules that selected the baseline.
    assert spec_from_baseline(baseline) == spec

    document = build_comparison(baseline, _comparison_run(spec))
    assert document["schema"] == COMPARISON_SCHEMA_ID
    assert document["kind"] == COMPARISON_KIND
    assert document["cell"] == CELL_FRESH_HEADLESS
    assert document["closure_status"] == "eligible_complete"
    assert document["committed_baseline_schema"] == BASELINE_SCHEMA_ID
    assert document["committed_attempt_digest"] == baseline["attempt_digest"]
    assert document["comparison_run"]["run_id"] == "cmp-1"
    assert tuple(document["label_precedence"]) == COMPARISON_LABEL_PRECEDENCE
    assert tuple(document["reason_order"]) == REASON_ORDER
    assert tuple(document["records"]) == cell_campaign_ids(CELL_FRESH_HEADLESS)

    for campaign_id, record in document["records"].items():
        row = ROW_SPECS[campaign_id]
        assert record["kind"] == COMPARISON_KIND
        assert record["campaign_id"] == campaign_id
        assert record["cell"] == CELL_FRESH_HEADLESS
        assert record["metric"] == row.metric
        assert record["unit"] == row.unit
        assert record["direction"] == DEFAULT_DIRECTION
        assert record["closure_status"] == "eligible_complete"
        assert [run["run_id"] for run in record["selected_runs"]] == list(selected)
        assert all(run["run_start_utc"] for run in record["selected_runs"])
        assert record["collected_run_p95s"] == [pytest.approx(1.4)] * SELECTED_RUNS
        assert record["median"] == pytest.approx(1.4)
        assert record["mad"] == pytest.approx(0.0)
        assert record["threshold"] == pytest.approx(1.68)
        assert record["measured_value"] == pytest.approx(1.4)
        assert record["measured_value"] <= record["threshold"]
        assert record["outcome"] == OUTCOME_PASS
        assert record["reason"] == REASON_THRESHOLD_PASS
        assert record["null_reasons"] == {}
        assert record["machine_mismatch_fields"] == []
        assert record["comparison_run"]["eligible"] is True
        assert record["comparison_run"]["failing_row"] is None
        # A marked outcome is never a bare pass/fail flag, nor an activation state.
        for forbidden in ("passed", "failed", "activation", "activation_state"):
            assert forbidden not in record

    # Above the threshold the row fails; the label is marked, not a flag.
    slower = build_comparison(baseline, _comparison_run(spec, base=2.0))
    for record in slower["records"].values():
        assert record["threshold"] == pytest.approx(1.68)
        assert record["measured_value"] == pytest.approx(2.4)
        assert record["measured_value"] > record["threshold"]
        assert record["outcome"] == OUTCOME_FAIL
        assert record["reason"] == REASON_THRESHOLD_FAIL
        assert "passed" not in record
        assert "failed" not in record

    # A row that declared a higher-is-better margin would invert the operator;
    # no gated row declares one, so only the default is exercised end to end.
    higher = evaluate_comparison_row(
        baseline, "A1", _comparison_run(spec), direction=DIRECTION_HIGHER_IS_BETTER
    )
    assert higher["direction"] == "higher_is_better"
    assert higher["outcome"] == OUTCOME_FAIL
    assert higher["reason"] == REASON_THRESHOLD_FAIL


def _assert_threshold_boundary_and_degeneracy() -> None:
    spec = _spec_for(CELL_PERMISSION)
    assert cell_campaign_ids(CELL_PERMISSION) == ("E2",)
    starts = list(_hourly_starts(SELECTED_RUNS).values())

    # A constant-series baseline: M = D = 2.5, so T = 2.5 + 0.20*2.5 = 3.0.
    steady = [
        _constant_run(spec, "E2", value=2.5, run_id=f"run-{index}", start=start)
        for index, start in enumerate(starts, start=1)
    ]
    baseline = build_baseline(spec, steady)
    assert baseline["closure"]["status"] == "eligible_complete"

    for value, outcome, reason in (
        (2.5, OUTCOME_PASS, REASON_THRESHOLD_PASS),
        (3.0, OUTCOME_PASS, REASON_THRESHOLD_PASS),
        (3.001, OUTCOME_FAIL, REASON_THRESHOLD_FAIL),
    ):
        record = build_comparison(
            baseline,
            _constant_run(
                spec, "E2", value=value, run_id="cmp", start=_COMPARISON_RUN_START
            ),
        )["records"]["E2"]
        assert record["median"] == 2.5
        assert record["mad"] == 0.0
        assert record["threshold"] == 3.0
        assert record["measured_value"] == value
        assert record["outcome"] == outcome
        assert record["reason"] == reason

    # An all-zero baseline has T = 0: degenerate, never padded with an epsilon.
    zeroed = [
        _constant_run(spec, "E2", value=0.0, run_id=f"run-{index}", start=start)
        for index, start in enumerate(starts, start=1)
    ]
    zero_baseline = build_baseline(spec, zeroed)
    record = build_comparison(
        zero_baseline,
        _constant_run(spec, "E2", value=0.0, run_id="cmp", start=_COMPARISON_RUN_START),
    )["records"]["E2"]
    assert record["median"] == 0.0
    assert record["mad"] == 0.0
    assert record["threshold"] is None
    assert record["null_reasons"]["threshold"] == REASON_DEGENERATE_THRESHOLD
    assert record["outcome"] == OUTCOME_DORMANT
    assert record["reason"] == REASON_DEGENERATE_THRESHOLD
    assert record["measured_value"] is None
    assert record["null_reasons"]["measured_value"] == REASON_DEGENERATE_THRESHOLD


def _assert_missing_determinants() -> None:
    spec = _spec_for()
    baseline = _committed_baseline(spec)

    # A committed campaign whose row block is absent supplies no determinants for
    # that row: DORMANT missing_determinants, never a fabricated threshold.
    deleted = copy.deepcopy(baseline)
    del deleted["campaigns"]["A2"]
    record = build_comparison(deleted, _comparison_run(spec))["records"]["A2"]
    assert record["outcome"] == OUTCOME_DORMANT
    assert record["reason"] == REASON_MISSING_DETERMINANTS
    assert record["threshold"] is None
    assert record["null_reasons"]["threshold"] == REASON_MISSING_DETERMINANTS
    assert record["collected_run_p95s"] == []
    assert record["measured_value"] is None

    # A block describing another row is not this row's determinants either.
    misfiled = copy.deepcopy(baseline)
    misfiled["campaigns"]["A2"] = dict(misfiled["campaigns"]["A1"])
    record = build_comparison(misfiled, _comparison_run(spec))["records"]["A2"]
    assert record["outcome"] == OUTCOME_DORMANT
    assert record["reason"] == REASON_MISSING_DETERMINANTS
    assert record["threshold"] is None


def _assert_comparison_machine_refusal() -> None:
    spec = _spec_for()
    baseline = _committed_baseline(spec)

    for field, value in (
        ("cpu_model", "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz"),
        ("core_count", 8),
        ("python_version", "3.13.0"),
        ("platform", "Linux-9.9.9-other-x86_64-with-glibc2.44"),
    ):
        records = _records_for(
            spec, run_id="cmp-1", run_start_utc=_COMPARISON_RUN_START
        )
        for record in records.values():
            record["machine"] = {**_MACHINE, field: value}
        document = build_comparison(baseline, shared_run(records, run_key="cmp-1"))
        for record in document["records"].values():
            assert record["outcome"] == OUTCOME_DORMANT
            assert record["reason"] == REASON_MACHINE_MISMATCH
            assert tuple(record["machine_mismatch_fields"]) == (field,)
            # No comparison happened: no measured value, no threshold label.
            assert record["measured_value"] is None
            assert record["null_reasons"]["measured_value"] == "machine_mismatch"
            # The committed baseline's own statistics stay computable and recorded.
            assert record["threshold"] == pytest.approx(1.68)
            # A dormant row performs no run-level evaluation at all.
            assert record["comparison_run"]["eligible"] is None
            assert record["comparison_run"]["null_reason"] == "row_dormant"

    # Inequality is judged per field, not for core_count alone.
    records = _records_for(spec, run_id="cmp-2", run_start_utc=_COMPARISON_RUN_START)
    for record in records.values():
        record["machine"] = {**_MACHINE, "core_count": 8, "platform": "Linux-other"}
    document = build_comparison(baseline, shared_run(records, run_key="cmp-2"))
    for record in document["records"].values():
        assert tuple(record["machine_mismatch_fields"]) == ("core_count", "platform")

    # A field the run cannot report is a run-record defect, not a machine
    # difference: it is judged by the run rules.
    records = _records_for(spec, run_id="cmp-3", run_start_utc=_COMPARISON_RUN_START)
    for record in records.values():
        record["machine"] = {**_MACHINE, "cpu_model": None}
    run = shared_run(records, run_key="cmp-3")
    assert comparison_machine_mismatch(_MACHINE, run) == ()
    record = build_comparison(baseline, run)["records"]["A1"]
    assert record["outcome"] == OUTCOME_INELIGIBLE
    assert record["reason"] == REASON_RUN_INELIGIBILITY
    assert "machine_field_missing" in record["run_ineligibility_reasons"]


def _assert_comparison_insufficient_and_high_load() -> None:
    # A campaign that closed insufficient_data is not decision-ready: DORMANT
    # with insufficient_baseline_data and NO populated threshold.
    storm = _spec_for(CELL_STORM)
    starts = _hourly_starts(MAX_ATTEMPTS)
    all_noisy = build_baseline(
        storm, _shared_cell_runs(storm, starts, noisy=frozenset(starts))
    )
    assert all_noisy["closure"]["status"] == "insufficient_data"
    record = build_comparison(all_noisy, _comparison_run(storm))["records"]["E1"]
    assert record["closure_status"] == "insufficient_data"
    assert record["outcome"] == OUTCOME_DORMANT
    assert record["reason"] == "insufficient_baseline_data"
    assert record["selected_runs"] == []
    assert record["collected_run_p95s"] == []
    assert record["median"] is None
    assert record["mad"] is None
    assert record["threshold"] is None
    assert record["null_reasons"]["threshold"] == "insufficient_baseline_data"
    assert record["null_reasons"]["median"] == "insufficient_baseline_data"
    assert record["measured_value"] is None

    # Thin data (four eligible of ten attempts) carries what those four runs
    # support — M and D — and still no threshold.
    thin = build_baseline(
        storm, _shared_cell_runs(storm, starts, noisy=frozenset(list(starts)[4:]))
    )
    assert thin["closure"]["status"] == "insufficient_data"
    record = build_comparison(thin, _comparison_run(storm))["records"]["E1"]
    assert len(record["collected_run_p95s"]) == 4 < SELECTED_RUNS
    assert record["median"] == pytest.approx(1.4)
    assert record["mad"] == pytest.approx(0.0)
    assert record["threshold"] is None
    assert record["null_reasons"]["threshold"] == "insufficient_baseline_data"
    assert record["outcome"] == OUTCOME_DORMANT
    assert record["measured_value"] is None

    # High load on the comparison run: a marked outcome, never a pass or a fail.
    spec = _spec_for()
    baseline = _committed_baseline(spec)
    noisy_run = _comparison_run(
        spec, quiet=quiet_window(quiet_stamp(0.4, _CORES), quiet_stamp(8.0, _CORES))
    )
    for record in build_comparison(baseline, noisy_run)["records"].values():
        assert record["outcome"] == OUTCOME_INELIGIBLE
        assert record["reason"] == REASON_RUN_INELIGIBILITY
        assert "quiet_stamp_not_quiet" in record["run_ineligibility_reasons"]
        assert record["measured_value"] is None
        assert record["null_reasons"]["measured_value"] == "run_ineligibility"
        assert record["threshold"] == pytest.approx(1.68)
        assert record["comparison_run"]["eligible"] is False
        assert "passed" not in record
        assert "failed" not in record

    # The comparison run's own eligibility is the selection conjunction: one
    # row's floor miss takes every co-measured row down with it.
    short = _records_for(spec, run_id="cmp-4", run_start_utc=_COMPARISON_RUN_START)
    short["A4"] = build_baseline_run(
        "A4",
        run_id="cmp-4",
        run_start_utc=_COMPARISON_RUN_START,
        fixture_hash=spec.fixture_hash,
        machine=dict(spec.machine),
        quiet=_quiet_window(),
        dimensions=dict(spec.dimensions),
        measured_ops=999,
        warmup_ops=10,
        samples=_series(999),
    )
    document = build_comparison(baseline, shared_run(short, run_key="cmp-4"))
    for record in document["records"].values():
        assert record["outcome"] == OUTCOME_INELIGIBLE
        assert record["comparison_run"]["failing_row"] == "A4"
        assert "ops_floor_measured" in record["run_ineligibility_reasons"]

    # The invalid-sample cap is a recorded subtype reason of ineligibility.
    invalid = _records_for(spec, run_id="cmp-5", run_start_utc=_COMPARISON_RUN_START)
    invalid["A1"]["cell_eligibility"] = {"invalid_samples": 12}
    document = build_comparison(baseline, shared_run(invalid, run_key="cmp-5"))
    for record in document["records"].values():
        assert record["outcome"] == OUTCOME_INELIGIBLE
        assert record["reason"] == REASON_RUN_INELIGIBILITY
        assert "invalid_sample_cap" in record["run_ineligibility_reasons"]

    # One run of another cell is not a comparison run for this baseline.
    other = _records_for(
        _spec_for(CELL_LOADED_HEADLESS),
        run_id="cmp-6",
        run_start_utc=_COMPARISON_RUN_START,
    )
    _assert_comparison_error(
        lambda: build_comparison(baseline, shared_run(other, run_key="cmp-6")),
        "comparison_run_cell_mismatch",
    )


def _assert_comparison_precedence() -> None:
    # DORMANT > COMPARISON-INELIGIBLE: a mismatched machine AND an ineligible run
    # records the row's dormancy, never the run's ineligibility.
    spec = _spec_for()
    baseline = _committed_baseline(spec)
    records = _records_for(spec, run_id="cmp-1", run_start_utc=_COMPARISON_RUN_START)
    for record in records.values():
        record["machine"] = {**_MACHINE, "platform": "Linux-other"}
        record["quiet"] = quiet_window(
            quiet_stamp(0.4, _CORES), quiet_stamp(8.0, _CORES)
        )
    document = build_comparison(baseline, shared_run(records, run_key="cmp-1"))
    assert COMPARISON_LABEL_PRECEDENCE.index(OUTCOME_DORMANT) == 0
    for record in document["records"].values():
        assert record["outcome"] == OUTCOME_DORMANT
        assert record["reason"] == REASON_MACHINE_MISMATCH
        assert record["comparison_run"]["eligible"] is None
        assert record["comparison_run"]["null_reason"] == "row_dormant"

    # DORMANT > COMPARISON-INELIGIBLE for a non-decision-ready baseline too.
    storm = _spec_for(CELL_STORM)
    starts = _hourly_starts(MAX_ATTEMPTS)
    all_noisy = build_baseline(
        storm, _shared_cell_runs(storm, starts, noisy=frozenset(starts))
    )
    record = build_comparison(all_noisy, _comparison_run(storm, base=2.0))["records"][
        "E1"
    ]
    assert record["outcome"] == OUTCOME_DORMANT
    assert record["reason"] == "insufficient_baseline_data"

    # An unknown row or an unusable direction is a refusal, not a guess.
    _assert_comparison_error(
        lambda: evaluate_comparison_row(baseline, "Z9", _comparison_run(spec)),
        "unknown_campaign_id",
    )
    _assert_comparison_error(
        lambda: evaluate_comparison_row(
            baseline, "A1", _comparison_run(spec), direction="sideways"
        ),
        "direction_invalid",
    )
    _assert_comparison_error(
        lambda: spec_from_baseline({"cell": CELL_STORM}), "identity_missing"
    )


def _assert_comparison_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec_for()
    baseline = _committed_baseline(spec)
    document = build_comparison(baseline, _comparison_run(spec))

    # RUN_DIR is mandatory: there is no default and no fallback into the tree.
    monkeypatch.delenv(COMPARISON_RUN_DIR_ENV, raising=False)
    _assert_comparison_error(comparison_run_dir, "run_dir_unset")
    _assert_comparison_error(lambda: write_comparison(document), "run_dir_unset")

    run_dir = tmp_path / "run-dir"
    run_dir.mkdir()
    monkeypatch.setenv(COMPARISON_RUN_DIR_ENV, str(run_dir))
    assert comparison_run_dir() == run_dir.resolve()

    # A relative or repository-resident RUN_DIR is refused outright.
    monkeypatch.setenv(COMPARISON_RUN_DIR_ENV, "relative/run-dir")
    _assert_comparison_error(comparison_run_dir, "run_dir_relative")
    monkeypatch.setenv(COMPARISON_RUN_DIR_ENV, str(_PERF_DIR))
    _assert_comparison_error(comparison_run_dir, "run_dir_in_repo")
    monkeypatch.setenv(COMPARISON_RUN_DIR_ENV, str(_PERF_DIR / BASELINES_DIRNAME))
    _assert_comparison_error(comparison_run_dir, "run_dir_in_repo")

    monkeypatch.setenv(COMPARISON_RUN_DIR_ENV, str(run_dir))
    outside = tmp_path / "outside"
    outside.mkdir()
    _assert_comparison_error(
        lambda: comparison_write_path(run_dir=outside), "run_dir_outside_run_dir"
    )
    _assert_comparison_error(
        lambda: comparison_write_path(run_dir="relative"), "run_dir_relative"
    )
    _assert_comparison_error(
        lambda: comparison_write_path(filename="nested/comparison.json"),
        "filename_invalid",
    )
    _assert_comparison_error(
        lambda: comparison_write_path(filename="comparison.txt"), "filename_invalid"
    )

    # A symlinked destination that escapes RUN_DIR is refused, and the link's
    # target is never created.
    escaping = run_dir / "escaping.json"
    escaping.symlink_to(_PERF_DIR / "escaped-comparison.json")
    _assert_comparison_error(
        lambda: comparison_write_path(filename="escaping.json"), "write_escape"
    )
    assert not (_PERF_DIR / "escaped-comparison.json").exists()

    # The one legal destination: RUN_DIR, or a subdirectory of it.
    written = write_comparison(document)
    assert written == run_dir.resolve() / COMPARISON_FILENAME
    text = written.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == json.loads(json.dumps(document, sort_keys=True))

    nested = write_comparison(
        document, filename="x5b-comparison.json", run_dir=run_dir / "nested"
    )
    assert nested.parent == (run_dir / "nested").resolve()
    assert json.loads(nested.read_text(encoding="utf-8"))["cell"] == (
        CELL_FRESH_HEADLESS
    )

    # Nothing landed in the tree, and the committed baselines are untouched.
    assert not (committed_baselines_dir() / COMPARISON_FILENAME).exists()
    assert not (committed_baselines_dir() / "x5b-comparison.json").exists()

    # A document carrying a non-finite value is refused, never written.
    permission = _spec_for(CELL_PERMISSION)
    infinite = [
        _constant_run(
            permission, "E2", value=float("inf"), run_id=f"run-{index}", start=start
        )
        for index, start in enumerate(_hourly_starts(SELECTED_RUNS).values(), start=1)
    ]
    infinite_baseline = build_baseline(permission, infinite)
    infinite_document = build_comparison(
        infinite_baseline,
        _constant_run(
            permission, "E2", value=0.0, run_id="cmp", start=_COMPARISON_RUN_START
        ),
    )
    record = infinite_document["records"]["E2"]
    assert record["median"] is None
    assert record["null_reasons"]["median"] == REASON_DEGENERATE_THRESHOLD
    assert record["threshold"] is None
    assert record["outcome"] == OUTCOME_DORMANT
    assert record["reason"] == REASON_DEGENERATE_THRESHOLD
    _assert_comparison_error(
        lambda: write_comparison(infinite_document, filename="infinite.json"),
        "document_not_json_safe",
    )
    assert not (run_dir / "infinite.json").exists()


def _assert_committed_baseline_reading(root: Path) -> None:
    root.mkdir(parents=True)
    spec = _spec_for()
    committed = _committed_baseline(spec)
    baselines = root / BASELINES_DIRNAME

    # Reading never creates the promoted directory.
    assert load_committed_baseline(CELL_FRESH_HEADLESS, baselines_dir=baselines) is None
    assert not baselines.exists()

    baselines.mkdir()
    path = committed_baseline_path(CELL_FRESH_HEADLESS, baselines_dir=baselines)
    assert path == baselines / f"{CELL_FRESH_HEADLESS}.json"
    path.write_text(json.dumps(committed), encoding="utf-8")
    loaded = load_committed_baseline(CELL_FRESH_HEADLESS, baselines_dir=baselines)
    assert loaded is not None
    assert loaded["attempt_digest"] == committed["attempt_digest"]
    validate_baseline(loaded, spec)
    from_file = build_comparison(loaded, _comparison_run(spec))
    assert from_file["committed_attempt_digest"] == committed["attempt_digest"]
    assert {record["outcome"] for record in from_file["records"].values()} == {
        OUTCOME_PASS
    }

    # A file that is not a committed baseline is refused, never skipped.
    path.write_text("[]", encoding="utf-8")
    _assert_comparison_error(
        lambda: load_committed_baseline(CELL_FRESH_HEADLESS, baselines_dir=baselines),
        "baseline_unreadable",
    )
    path.write_text(json.dumps({"schema": "chartreux.runtime_baseline.v0"}))
    _assert_comparison_error(
        lambda: load_committed_baseline(CELL_FRESH_HEADLESS, baselines_dir=baselines),
        "baseline_schema_invalid",
    )
    path.write_text("{not json", encoding="utf-8")
    _assert_comparison_error(
        lambda: load_committed_baseline(CELL_FRESH_HEADLESS, baselines_dir=baselines),
        "baseline_unreadable",
    )


def _assert_comparison_is_not_an_activation_oracle(root: Path) -> None:
    root.mkdir(parents=True)
    spec = _spec_for()
    baseline = _committed_baseline(spec)
    before = copy.deepcopy(baseline)

    document = build_comparison(baseline, _comparison_run(spec))
    # The committed baseline is an input, never an output.
    assert baseline == before
    # A comparison artifact names itself a comparison and carries no activation
    # state and no freshly constructed threshold table.
    assert document["kind"] == COMPARISON_KIND
    for forbidden in ("activation", "activation_state", "thresholds", "baseline_set"):
        assert forbidden not in document
        for record in document["records"].values():
            assert forbidden not in record
    for record in document["records"].values():
        assert record["outcome"] in OUTCOME_LABELS
    # Building a comparison writes nothing at all.
    assert tuple(root.iterdir()) == ()


@pytest.mark.timeout(_COMPARISON_TIMEOUT_S)
def test_runtime_comparison_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X5b compares a fresh run against a committed baseline, marking outcomes."""
    started = perf_counter()
    registry = _load_artifact("runtime_metric_registry.json")
    _assert_comparison_constants(registry)
    _assert_comparison_statistics()
    _assert_every_row_defaults_lower_is_better()
    _assert_comparison_outcomes()
    _assert_threshold_boundary_and_degeneracy()
    _assert_missing_determinants()
    _assert_comparison_machine_refusal()
    _assert_comparison_insufficient_and_high_load()
    _assert_comparison_precedence()
    _assert_comparison_writer(tmp_path, monkeypatch)
    _assert_committed_baseline_reading(tmp_path / "reader")
    _assert_comparison_is_not_an_activation_oracle(tmp_path / "oracle")
    assert perf_counter() - started <= _CONTRACT_WINDOW_S
