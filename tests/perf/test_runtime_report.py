"""WP6 contract node: the runtime-perf run-record and report schema.

The node pins the schema id to the metric version of every run record (and to
the distinct determinants-document schema field), binds the schema's 17 gated
rows to the COMMITTED registry — a missing or invented campaign id fails here
rather than silently editing the schema — checks every row's chunks-coverage
floors against Appendix B, exercises the record's row extensions (late windows,
signed-plus-clamped slopes, save timings, campaign closure), and drives the
validator's refusals so a record cannot claim eligibility its own counts do not
support.

Nothing here measures anything: it reads the pinned artifacts and the schema,
so the node stays inside the plan's in-suite contract-node window.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import pytest

from tests.perf._runtime_report import (
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
    METRIC_VERSION,
    MODE_PROFILE,
    PERF_BLOB_KIND,
    PERF_BLOB_MACHINE_MISMATCH,
    PERF_BLOB_SECTION,
    REPORT_SCHEMA,
    ROW_SPECS,
    SAVE_LOCK_PHASE,
    SCHEMA_EXAMPLE,
    SCHEMA_ID,
    ReportSchemaError,
    SaveTimings,
    build_report,
    build_run_record,
    evaluate_run_floor,
    nearest_rank,
    report_schema,
    reporting_extension,
    validate_comparison_states_section,
    validate_perf_blob_section,
    validate_report,
    validate_reporting_sections,
    validate_run_record,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="the runtime perf schema is Linux-only machinery",
    ),
]

_PERF_DIR = Path(__file__).resolve().parent

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_SCHEMA_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_RUN_START = "2026-11-08T00:00:00Z"

#: The rows' floors as Appendix B pins them: measured ops, warmup ops, and the
#: row-specific unit counts at exactly their floor value.
_AT_FLOOR: tuple[tuple[str, int, int, dict[str, int]], ...] = (
    ("A1", 100, 10, {}),
    ("A2", 100, 10, {}),
    ("A3", 100, 10, {"chunk_gaps": 300}),
    ("A4", 1000, 10, {}),
    ("B1", 100, 10, {}),
    ("B2", 1000, 10, {}),
    ("B3", 100, 10, {"root_turns": 100}),
    ("B4", 100, 10, {"root_turns": 100}),
    ("C1", 100, 10, {"saves": 100}),
    ("C2", 100, 10, {"saves": 100}),
    ("C3", 1000, 10, {}),
    ("D1", 100, 10, {"render_flushes": 100, "turns": 100}),
    ("E1", 100, 10, {"storms": 100}),
    ("E2", 100, 10, {"permission_round_trips": 100}),
    ("F1", 100, 300, {}),
    ("F2", 1000, 300, {"late_window_heartbeat_rows": 1000}),
    ("G1", 101, 10, {"rss_samples": 201, "window_slopes": 101}),
)

#: Plan §3's row-specific unit counts, per campaign ID.
_EXPECTED_UNIT_COUNTS: dict[str, tuple[str, ...]] = {
    "A3": ("chunk_gaps",),
    "B3": ("root_turns",),
    "B4": ("root_turns",),
    "C1": ("saves",),
    "C2": ("saves",),
    "D1": ("render_flushes", "turns"),
    "E1": ("storms",),
    "E2": ("permission_round_trips",),
    "F2": ("late_window_heartbeat_rows",),
    "G1": ("rss_samples", "window_slopes"),
}


def _load_artifact(name: str) -> dict[str, Any]:
    return json.loads((_PERF_DIR / name).read_text(encoding="utf-8"))


def _assert_reason(call: Callable[[], object], reason: str) -> None:
    with pytest.raises(ReportSchemaError) as excinfo:
        call()
    assert excinfo.value.reason == reason, excinfo.value.detail


def _a1_record(
    *,
    measured_ops: int = 100,
    warmup_ops: int = 10,
    samples: list[float] | None = None,
    mode: str = "measure",
    closure_status: str | None = CLOSURE_ELIGIBLE_COMPLETE,
    closure_note: str | None = None,
) -> dict[str, Any]:
    series = (
        [float(index) for index in range(measured_ops)] if samples is None else samples
    )
    return build_run_record(
        "A1",
        run_id="a1-run-1",
        run_start_utc=_RUN_START,
        measured_ops=measured_ops,
        warmup_ops=warmup_ops,
        samples=series,
        mode=mode,
        closure_status=closure_status,
        closure_note=closure_note,
    )


def _f2_record() -> dict[str, Any]:
    return build_run_record(
        "F2",
        run_id="f2-run-1",
        run_start_utc=_RUN_START,
        measured_ops=1000,
        warmup_ops=300,
        samples=[float(index % 4) + 1.0 for index in range(1000)],
        unit_counts={"late_window_heartbeat_rows": 1000},
        closure_status=CLOSURE_ELIGIBLE_COMPLETE,
    )


def _c1_record(*, worker_ident: int = 99) -> dict[str, Any]:
    return build_run_record(
        "C1",
        run_id="c1-run-1",
        run_start_utc=_RUN_START,
        measured_ops=100,
        warmup_ops=10,
        unit_counts={"saves": 100},
        save_timings=SaveTimings(
            worker_scan_ms=[float(index) + 1.0 for index in range(100)],
            lock_held_await_ms=[float(index) + 3.0 for index in range(100)],
            worker_thread_ident=worker_ident,
            loop_thread_ident=1,
        ),
        closure_status=CLOSURE_ELIGIBLE_COMPLETE,
    )


def _assert_pinned_identity(determinants: dict[str, Any]) -> None:
    assert SCHEMA_ID == "chartreux.runtime_perf.v1"
    assert METRIC_VERSION == SCHEMA_ID
    assert REPORT_SCHEMA["schema"] == SCHEMA_ID
    assert REPORT_SCHEMA["metric_version"] == METRIC_VERSION
    assert REPORT_SCHEMA["modes"] == ("measure", "profile")
    assert REPORT_SCHEMA["closure_statuses"] == (
        "eligible_complete",
        "insufficient_data",
    )
    # The run-record metric version is deliberately distinct from the
    # determinants document's own schema field.
    assert determinants["schema"] == "chartreux.runtime_metric_determinants.v1"
    assert determinants["schema"] != METRIC_VERSION
    eligibility = determinants["common_rules"]["run_eligibility"]["metric_version"]
    assert METRIC_VERSION in eligibility


def _assert_registry_binding(registry: dict[str, Any]) -> None:
    entries = registry["entries"]
    assert registry["schema"] == "chartreux.runtime_metric_registry.v1"
    assert tuple(entry["campaign_id"] for entry in entries) == GATED_CAMPAIGN_IDS
    assert len(GATED_CAMPAIGN_IDS) == 17
    assert tuple(ROW_SPECS) == GATED_CAMPAIGN_IDS
    assert tuple(REPORT_SCHEMA["rows"]) == GATED_CAMPAIGN_IDS
    for entry in entries:
        spec = ROW_SPECS[entry["campaign_id"]]
        assert (spec.metric, spec.unit, spec.cell, spec.producer_wp) == (
            entry["metric"],
            entry["unit"],
            entry["cell"],
            entry["producer_wp"],
        )
        row = REPORT_SCHEMA["rows"][entry["campaign_id"]]
        assert (row["metric"], row["unit"], row["cell"]) == (
            entry["metric"],
            entry["unit"],
            entry["cell"],
        )


def _assert_coverage_binding() -> None:
    for campaign_id in GATED_CAMPAIGN_IDS:
        spec = ROW_SPECS[campaign_id]
        expected = _EXPECTED_UNIT_COUNTS.get(campaign_id, ())
        assert tuple(sorted(spec.required_unit_counts)) == tuple(sorted(expected))
    assert REPORT_SCHEMA["chunks_coverage_fields"] == (
        "unit",
        "measured_ops",
        "warmup_ops",
        "unit_counts",
        "complete",
        "missing",
    )
    # The gated series comes from the row's own source, never from a guess.
    assert ROW_SPECS["A3"].gated_series_key == "chunk_gaps"
    assert ROW_SPECS["G1"].gated_series_key == "window_slopes"
    assert ROW_SPECS["C1"].save_phase == "worker_scan_ms"
    assert ROW_SPECS["C2"].save_phase == SAVE_LOCK_PHASE
    assert ROW_SPECS["A1"].gated_series_key is None
    assert ROW_SPECS["A1"].save_phase is None


def _assert_floors() -> None:
    assert tuple(row[0] for row in _AT_FLOOR) == GATED_CAMPAIGN_IDS
    for campaign_id, measured, warmup, counts in _AT_FLOOR:
        verdict = evaluate_run_floor(
            campaign_id, measured_ops=measured, warmup_ops=warmup, unit_counts=counts
        )
        assert verdict.eligible is True, (campaign_id, verdict.reasons)
        assert verdict.reasons == ()
        short_ops = evaluate_run_floor(
            campaign_id,
            measured_ops=measured - 1,
            warmup_ops=warmup,
            unit_counts=counts,
        )
        assert short_ops.eligible is False, campaign_id
        assert short_ops.reasons
        short_warmup = evaluate_run_floor(
            campaign_id,
            measured_ops=measured,
            warmup_ops=warmup - 1,
            unit_counts=counts,
        )
        assert short_warmup.eligible is False, campaign_id

    assert evaluate_run_floor(
        "A3", measured_ops=100, warmup_ops=10, unit_counts={"chunk_gaps": 299}
    ).reasons == ("coverage_floor_chunk_gaps",)
    assert evaluate_run_floor("A4", measured_ops=999, warmup_ops=10).reasons == (
        "ops_floor_measured",
    )
    assert evaluate_run_floor("A3", measured_ops=100, warmup_ops=10).reasons == (
        "coverage_missing_chunk_gaps",
    )
    assert evaluate_run_floor("F1", measured_ops=99, warmup_ops=300).reasons == (
        "ops_floor_measured",
        "ops_floor_exact_measured",
    )
    assert evaluate_run_floor(
        "G1",
        measured_ops=101,
        warmup_ops=200,
        unit_counts={"rss_samples": 201, "window_slopes": 100},
    ).reasons == ("coverage_exact_window_slopes",)
    assert evaluate_run_floor(
        "G1",
        measured_ops=101,
        warmup_ops=200,
        unit_counts={"rss_samples": 200, "window_slopes": 101},
    ).reasons == ("coverage_exact_rss_samples",)


def _assert_late_windows() -> None:
    soak = {
        "warm_boundary_turn": 300,
        "first_measured_turn": 301,
        "last_measured_turn": 400,
        "measured_turns": 100,
    }
    memory = {
        "warm_boundary_turn": 199,
        "first_measured_turn": 200,
        "last_measured_turn": 400,
        "sample_count": 201,
        "window_width_turns": 100,
        "window_count": 101,
    }
    f1 = ROW_SPECS["F1"].late_window
    f2 = ROW_SPECS["F2"].late_window
    g1 = ROW_SPECS["G1"].late_window
    assert f1 is not None and f2 is not None and g1 is not None
    assert f1.as_record() == soak
    assert f2.as_record() == soak
    assert g1.as_record() == memory
    late_rows = [
        campaign_id
        for campaign_id in GATED_CAMPAIGN_IDS
        if ROW_SPECS[campaign_id].late_window is not None
    ]
    assert late_rows == ["F1", "F2", "G1"]


def _assert_example() -> None:
    validate_report(SCHEMA_EXAMPLE)
    assert SCHEMA_EXAMPLE["schema"] == SCHEMA_ID
    assert SCHEMA_EXAMPLE["metric_version"] == METRIC_VERSION
    runs = SCHEMA_EXAMPLE["runs"]
    assert [run["campaign_id"] for run in runs] == ["A3", "C2", "G1"]
    for run in runs:
        assert run["metric_version"] == METRIC_VERSION
        assert run["summary"]["count"] == len(run["samples"])
        assert run["summary"]["p95"] == nearest_rank(run["samples"], fraction=0.95)
        assert run["eligibility"] == {"eligible": True, "reasons": []}

    a3, c2, g1 = runs
    # A3: the gated series is per gap while the measured ops are turns.
    assert a3["chunks_coverage"] == {
        "unit": "turns",
        "measured_ops": 100,
        "warmup_ops": 10,
        "unit_counts": {"chunk_gaps": 300},
        "complete": True,
        "missing": [],
    }
    assert len(a3["samples"]) == 300
    assert a3["campaign"]["closure_status"] is None
    assert isinstance(a3["campaign"]["closure_note"], str)
    assert "late_window" not in a3
    assert "slopes" not in a3
    assert "save_timings" not in a3

    # C2: both save phases are recorded, and the worker is never the loop.
    timings = c2["save_timings"]
    assert len(timings["worker_scan_ms"]) == 120
    assert len(timings[SAVE_LOCK_PHASE]) == 120
    assert timings["worker_thread_ident"] != timings["loop_thread_ident"]
    assert timings["threads_distinct"] is True
    assert c2["samples"] == timings[SAVE_LOCK_PHASE]
    assert c2["campaign"]["closure_status"] == CLOSURE_INSUFFICIENT_DATA

    # G1: signed slopes as measured, clamped to positive for the gate.
    slopes = g1["slopes"]
    signed = slopes["signed"]
    clamped = slopes["clamped"]
    assert len(signed) == len(clamped) == 101
    assert min(signed) < 0
    assert clamped == [max(0.0, value) for value in signed]
    assert g1["samples"] == clamped
    assert g1["unit"] == "bytes/turn"
    assert g1["late_window"]["window_count"] == 101
    assert g1["summary"]["p95_signed"] == nearest_rank(signed, fraction=0.95)
    assert g1["campaign"]["closure_status"] == CLOSURE_ELIGIBLE_COMPLETE


def _assert_builder_refusals() -> None:
    _assert_reason(
        lambda: build_run_record(
            "Z9", run_id="z9", run_start_utc=_RUN_START, measured_ops=1, warmup_ops=0
        ),
        "unknown_campaign_id",
    )
    _assert_reason(lambda: _a1_record(mode="benchmark"), "mode_invalid")
    _assert_reason(lambda: _a1_record(closure_status=None), "null_without_reason")
    _assert_reason(
        lambda: _a1_record(closure_status="pending"), "closure_status_invalid"
    )
    _assert_reason(
        lambda: _a1_record(samples=[float(index) for index in range(99)]),
        "sample_count_mismatch",
    )
    _assert_reason(
        lambda: build_run_record(
            "A1",
            run_id="a1-slopes",
            run_start_utc=_RUN_START,
            measured_ops=100,
            warmup_ops=10,
            samples=[float(index) for index in range(100)],
            slopes_signed=[1.0],
            closure_status=CLOSURE_ELIGIBLE_COMPLETE,
        ),
        "slopes_not_applicable",
    )
    _assert_reason(
        lambda: build_run_record(
            "A1",
            run_id="a1-saves",
            run_start_utc=_RUN_START,
            measured_ops=100,
            warmup_ops=10,
            samples=[float(index) for index in range(100)],
            save_timings=SaveTimings([1.0], [1.0], 2, 1),
            closure_status=CLOSURE_ELIGIBLE_COMPLETE,
        ),
        "save_timings_not_applicable",
    )
    _assert_reason(lambda: _c1_record(worker_ident=1), "save_thread_identity")
    _assert_reason(
        lambda: build_run_record(
            "G1",
            run_id="g1-run-1",
            run_start_utc=_RUN_START,
            measured_ops=101,
            warmup_ops=200,
            unit_counts={"rss_samples": 201, "window_slopes": 101},
            slopes_signed=[float(index) for index in range(101)],
            samples=[0.0] * 101,
            closure_status=CLOSURE_ELIGIBLE_COMPLETE,
        ),
        "samples_slope_mismatch",
    )
    _assert_reason(
        lambda: build_run_record(
            "C2",
            run_id="c2-run-1",
            run_start_utc=_RUN_START,
            measured_ops=120,
            warmup_ops=10,
            unit_counts={"saves": 120},
            save_timings=SaveTimings(
                worker_scan_ms=[1.0] * 120,
                lock_held_await_ms=[2.0] * 120,
                worker_thread_ident=2,
                loop_thread_ident=1,
            ),
            samples=[3.0] * 120,
            closure_status=CLOSURE_ELIGIBLE_COMPLETE,
        ),
        "samples_phase_mismatch",
    )

    # The valid shapes the refusals above are near misses of.
    profile_record = _a1_record(mode=MODE_PROFILE)
    assert profile_record["mode"] == "profile"
    validate_run_record(profile_record)
    save_record = _c1_record()
    assert save_record["samples"] == save_record["save_timings"]["worker_scan_ms"]
    assert save_record["samples"] != save_record["save_timings"][SAVE_LOCK_PHASE]
    f2 = _f2_record()
    assert f2["late_window"]["first_measured_turn"] == 301
    assert f2["chunks_coverage"]["unit_counts"]["late_window_heartbeat_rows"] == 1000


def _assert_validator_refusals() -> None:
    base = _a1_record()

    tampered = copy.deepcopy(base)
    tampered["metric_version"] = "chartreux.runtime_metric_determinants.v1"
    _assert_reason(lambda: validate_run_record(tampered), "metric_version_mismatch")

    tampered = copy.deepcopy(base)
    tampered["schema"] = "chartreux.runtime_perf.v2"
    _assert_reason(lambda: validate_run_record(tampered), "schema_mismatch")

    tampered = copy.deepcopy(base)
    tampered["campaign_id"] = "Z9"
    _assert_reason(lambda: validate_run_record(tampered), "unknown_campaign_id")

    tampered = copy.deepcopy(base)
    del tampered["chunks_coverage"]
    _assert_reason(lambda: validate_run_record(tampered), "missing_run_field")

    tampered = copy.deepcopy(base)
    tampered["cell"] = "x2_loaded_headless_8x200k"
    _assert_reason(lambda: validate_run_record(tampered), "cell_mismatch")

    tampered = copy.deepcopy(base)
    tampered["mode"] = "benchmark"
    _assert_reason(lambda: validate_run_record(tampered), "mode_invalid")

    tampered = copy.deepcopy(base)
    tampered["samples"] = tampered["samples"][:-1]
    _assert_reason(lambda: validate_run_record(tampered), "sample_count_mismatch")

    tampered = copy.deepcopy(base)
    tampered["summary"]["p95"] = 1.0
    _assert_reason(lambda: validate_run_record(tampered), "summary_mismatch")

    tampered = copy.deepcopy(base)
    tampered["summary"]["p95"] = None
    _assert_reason(lambda: validate_run_record(tampered), "summary_mismatch")

    tampered = copy.deepcopy(base)
    tampered["eligibility"] = {"eligible": False, "reasons": []}
    _assert_reason(lambda: validate_run_record(tampered), "eligibility_reason_missing")

    tampered = copy.deepcopy(base)
    tampered["eligibility"] = {"eligible": True, "reasons": ["ops_floor_measured"]}
    _assert_reason(
        lambda: validate_run_record(tampered), "eligibility_reasons_conflict"
    )

    # A sub-floor run cannot claim eligibility: the counts decide.
    sub_floor = _a1_record(
        measured_ops=50, samples=[float(index) for index in range(50)]
    )
    assert sub_floor["eligibility"] == {
        "eligible": False,
        "reasons": ["ops_floor_measured"],
    }
    sub_floor["eligibility"] = {"eligible": True, "reasons": []}
    _assert_reason(lambda: validate_run_record(sub_floor), "unsupported_eligibility")

    tampered = copy.deepcopy(base)
    tampered["campaign"]["closure_status"] = "pending"
    _assert_reason(lambda: validate_run_record(tampered), "closure_status_invalid")

    tampered = copy.deepcopy(base)
    tampered["campaign"]["closure_status"] = None
    tampered["campaign"]["closure_note"] = ""
    _assert_reason(lambda: validate_run_record(tampered), "null_without_reason")

    tampered = copy.deepcopy(base)
    tampered["campaign"]["campaign_id"] = "B1"
    _assert_reason(lambda: validate_run_record(tampered), "campaign_identity_mismatch")

    tampered = copy.deepcopy(base)
    tampered["late_window"] = {"warm_boundary_turn": 0}
    _assert_reason(lambda: validate_run_record(tampered), "late_window_not_applicable")

    a3 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][0])
    a3["chunks_coverage"]["unit"] = "saves"
    _assert_reason(lambda: validate_run_record(a3), "coverage_unit_mismatch")

    a3 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][0])
    del a3["chunks_coverage"]["unit_counts"]["chunk_gaps"]
    _assert_reason(lambda: validate_run_record(a3), "coverage_missing_fields")

    a3 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][0])
    a3["chunks_coverage"]["complete"] = False
    _assert_reason(
        lambda: validate_run_record(a3), "coverage_completeness_inconsistent"
    )

    a3 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][0])
    a3["chunks_coverage"]["unit_counts"]["chunk_gaps"] = -1
    _assert_reason(lambda: validate_run_record(a3), "coverage_count_invalid")

    g1 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][2])
    g1["slopes"]["clamped"][0] = 99.0
    _assert_reason(lambda: validate_run_record(g1), "slope_clamp_mismatch")

    g1 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][2])
    g1["slopes"]["signed"] = None
    _assert_reason(lambda: validate_run_record(g1), "slope_null_mismatch")

    g1 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][2])
    g1["slopes"]["signed"] = None
    g1["slopes"]["clamped"] = None
    _assert_reason(lambda: validate_run_record(g1), "null_without_reason")

    g1 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][2])
    g1["late_window"]["window_count"] = 100
    _assert_reason(lambda: validate_run_record(g1), "late_window_mismatch")

    g1 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][2])
    del g1["late_window"]
    _assert_reason(lambda: validate_run_record(g1), "late_window_missing")

    c2 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][1])
    c2["save_timings"]["worker_thread_ident"] = c2["save_timings"]["loop_thread_ident"]
    _assert_reason(lambda: validate_run_record(c2), "save_thread_identity")

    c2 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][1])
    del c2["save_timings"]
    _assert_reason(lambda: validate_run_record(c2), "missing_save_timings")

    c2 = copy.deepcopy(SCHEMA_EXAMPLE["runs"][1])
    c2["save_timings"][SAVE_LOCK_PHASE] = [1.0] * 10
    _assert_reason(lambda: validate_run_record(c2), "save_phase_count_mismatch")


def _assert_report_level() -> None:
    _assert_reason(lambda: validate_report({}), "schema_mismatch")
    _assert_reason(
        lambda: validate_report({
            "schema": SCHEMA_ID,
            "metric_version": "other",
            "runs": [],
        }),
        "metric_version_mismatch",
    )
    _assert_reason(
        lambda: validate_report({
            "schema": SCHEMA_ID,
            "metric_version": METRIC_VERSION,
            "runs": "nope",
        }),
        "report_runs_invalid",
    )
    duplicate = build_report((_a1_record(),))
    duplicate["runs"].append(copy.deepcopy(duplicate["runs"][0]))
    _assert_reason(lambda: validate_report(duplicate), "duplicate_run")


def _blob_record() -> dict[str, Any]:
    """Return one structurally valid PERF-BLOB machine-mismatch record."""
    return {
        "kind": PERF_BLOB_KIND,
        "record": PERF_BLOB_MACHINE_MISMATCH,
        "cell": "x1_fresh_headless_fb10_c4",
        "campaign_ids": ["A1", "A2", "A3", "A4"],
        "mismatch_fields": ["python_version"],
        "baseline_machine": {
            "cpu_model": "AMD Ryzen 9 5900X 12-Core Processor",
            "core_count": 12,
            "python_version": "3.12.13",
            "platform": "Linux-7.3.0-9-generic-x86_64-with-glibc2.44",
        },
        "comparison_machine": {
            "cpu_model": "AMD Ryzen 9 5900X 12-Core Processor",
            "core_count": 12,
            "python_version": "3.13.0",
            "platform": "Linux-7.3.0-9-generic-x86_64-with-glibc2.44",
        },
        "state": COMPARISON_STATE_DORMANT,
        "reason": PERF_BLOB_MACHINE_MISMATCH,
        "never_active": True,
        "never_failed": True,
    }


def _assert_reporting_extension() -> None:
    """X7b's sections are additive: descriptors merge, core sections never move."""
    extension = reporting_extension()
    assert set(extension) == {COMPARISON_STATES_SECTION, PERF_BLOB_SECTION}
    assert COMPARISON_STATES_SECTION not in REPORT_SCHEMA
    assert PERF_BLOB_SECTION not in REPORT_SCHEMA
    # The labels are the marked comparison vocabulary, in precedence order.
    assert COMPARISON_STATE_PRECEDENCE == (
        COMPARISON_STATE_DORMANT,
        COMPARISON_STATE_INELIGIBLE,
        COMPARISON_STATE_PASS,
        COMPARISON_STATE_FAIL,
    )
    assert (COMPARISON_STATE_DORMANT, COMPARISON_STATE_INELIGIBLE) == (
        "DORMANT",
        "COMPARISON-INELIGIBLE",
    )
    assert (COMPARISON_STATE_PASS, COMPARISON_STATE_FAIL) == (
        "ACTIVE-pass",
        "ACTIVE-fail",
    )
    extended = report_schema(extra_sections=extension)
    assert extended["rows"] == REPORT_SCHEMA["rows"]
    assert extended["run_fields"] == REPORT_SCHEMA["run_fields"]
    assert extended["slope_rule"] == REPORT_SCHEMA["slope_rule"]
    assert extended[COMPARISON_STATES_SECTION] == extension[COMPARISON_STATES_SECTION]
    assert extended[PERF_BLOB_SECTION] == extension[PERF_BLOB_SECTION]
    # The additive sections are disjoint from every core section, so the merge
    # can never be a core replacement.
    assert not set(extension) & set(REPORT_SCHEMA)

    # A PERF-BLOB record is a machine mismatch, never ACTIVE, never FAILED.
    record = _blob_record()
    validate_perf_blob_section({"kind": PERF_BLOB_KIND, "records": [record]})
    refusals: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
        (
            "state",
            lambda blob: blob.update({"state": COMPARISON_STATE_FAIL}),
            "perf_blob_state_not_dormant",
        ),
        (
            "never active",
            lambda blob: blob.update({"never_active": False}),
            "perf_blob_disposition_invalid",
        ),
        (
            "never failed",
            lambda blob: blob.update({"never_failed": False}),
            "perf_blob_disposition_invalid",
        ),
        (
            "no mismatch",
            lambda blob: blob.update({"mismatch_fields": []}),
            "perf_blob_mismatch_fields_invalid",
        ),
        (
            "record kind",
            lambda blob: blob.update({"record": "threshold_fail"}),
            "perf_blob_record_kind_invalid",
        ),
        (
            "unproven mismatch",
            lambda blob: blob["comparison_machine"].update({
                "python_version": "3.12.13"
            }),
            "perf_blob_mismatch_unproven",
        ),
    )
    for label, mutate, reason in refusals:
        tampered = copy.deepcopy(record)
        mutate(tampered)
        _assert_reason(
            lambda tampered=tampered: validate_perf_blob_section({
                "kind": PERF_BLOB_KIND,
                "records": [tampered],
            }),
            reason,
        )
        assert label
    # The core row rules are untouched by the extension.
    assert tuple(REPORT_SCHEMA["rows"]) == GATED_CAMPAIGN_IDS
    assert REPORT_SCHEMA["chunks_coverage_fields"][0] == "unit"


def _assert_extension_surface() -> None:
    # A later report edit lands as an added section, never a core rewrite.
    assert "comparisons" not in REPORT_SCHEMA
    extended = report_schema(
        extra_sections={
            "comparisons": {"schema": "chartreux.runtime_comparison.v1", "rows": []}
        }
    )
    assert extended["comparisons"] == {
        "schema": "chartreux.runtime_comparison.v1",
        "rows": [],
    }
    assert extended["rows"] == REPORT_SCHEMA["rows"]
    assert extended["metric_version"] == METRIC_VERSION
    assert "comparisons" not in REPORT_SCHEMA
    _assert_reason(
        lambda: report_schema(extra_sections={"rows": {}}), "section_conflict"
    )
    # X7b's own sections merge the same way and validate as rendered.
    with_added_sections = {
        **SCHEMA_EXAMPLE,
        "comparisons": [],
        COMPARISON_STATES_SECTION: {
            "rule": COMPARISON_STATE_RULE,
            "precedence": list(COMPARISON_STATE_PRECEDENCE),
            "counts": {label: 0 for label in COMPARISON_STATE_PRECEDENCE},
            "rows": [],
        },
        PERF_BLOB_SECTION: {"kind": PERF_BLOB_KIND, "records": [_blob_record()]},
    }
    validate_report(with_added_sections)
    validate_reporting_sections(with_added_sections)
    # An absent section is fine; a malformed one is refused.
    validate_reporting_sections({})
    _assert_reason(
        lambda: validate_comparison_states_section({"rule": "labels first"}),
        "comparison_states_rule_drift",
    )
    _assert_reason(
        lambda: validate_reporting_sections({PERF_BLOB_SECTION: {"records": []}}),
        "perf_blob_kind_invalid",
    )


@pytest.mark.timeout(_SCHEMA_TIMEOUT_S)
def test_report_schema_contract() -> None:
    """The schema pins 17 gated rows and holds every one of its refusals."""
    started = perf_counter()
    registry = _load_artifact("runtime_metric_registry.json")
    determinants = _load_artifact("runtime_metric_determinants.json")
    _assert_pinned_identity(determinants)
    _assert_registry_binding(registry)
    _assert_coverage_binding()
    _assert_floors()
    _assert_late_windows()
    _assert_example()
    _assert_builder_refusals()
    _assert_validator_refusals()
    _assert_report_level()
    _assert_extension_surface()
    _assert_reporting_extension()
    assert perf_counter() - started <= _CONTRACT_WINDOW_S
