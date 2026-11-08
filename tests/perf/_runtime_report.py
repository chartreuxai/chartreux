"""Run-record and report schema for the v0.7.5 stage-2 runtime gate (WP6).

The schema id :data:`SCHEMA_ID` (``chartreux.runtime_perf.v1``) is also the
pinned ``metric_version`` of every run record (Appendix B
``run_eligibility.metric_version``). It is deliberately distinct from the
determinants document's own schema field
(``chartreux.runtime_metric_determinants.v1``), which identifies that document
and is never a run-record value.

Per gated campaign ID the schema carries the row's identity (metric, unit,
cell, producing work package), its chunks-coverage requirements — the measured
op count, the discarded warmup count, and the row-specific unit counts of plan
§3 — its floors, and the row's record extension: the late-window bounds of the
F1/F2/G1 soak rows, G1's signed-plus-clamped slope record, and the C1/C2 save
timings. Records are built and validated here so every gated-row producer
(X1, X2, X3a, X3b, X8b, X8d, P5-Reanchor, WP2) writes one envelope and X5a's
floor validation consumes the same fields.

The extension is additive: :data:`REPORT_SCHEMA` is a document built from the
row table, :func:`report_schema` merges extra sections without touching the
core ones, and :func:`validate_report` tolerates unknown top-level sections.
X7b's comparison states and PERF-BLOB records therefore land as an added
section rather than an edit to the run-record rules.

Nullable fields follow Appendix B's rule: a field that cannot be computed is
recorded as NULL together with its reason, never fabricated or back-filled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

SCHEMA_ID = "chartreux.runtime_perf.v1"
#: The schema id is the pinned metric version of every run record (Appendix B).
METRIC_VERSION = SCHEMA_ID
#: The determinants document's own schema field; never a run-record value.
DETERMINANTS_SCHEMA_ID = "chartreux.runtime_metric_determinants.v1"

MODE_MEASURE = "measure"
MODE_PROFILE = "profile"
MODES: tuple[str, ...] = (MODE_MEASURE, MODE_PROFILE)

CLOSURE_ELIGIBLE_COMPLETE = "eligible_complete"
CLOSURE_INSUFFICIENT_DATA = "insufficient_data"
CLOSURE_STATUSES: tuple[str, ...] = (
    CLOSURE_ELIGIBLE_COMPLETE,
    CLOSURE_INSUFFICIENT_DATA,
)

#: The generic warmup floor: >= 10 operations discarded and never counted.
DEFAULT_WARMUP_OPS = 10

SAVE_UNIT = "ms"
SAVE_WORKER_PHASE = "worker_scan_ms"
SAVE_LOCK_PHASE = "lock_held_await_ms"
SAVE_PHASES: tuple[str, ...] = (SAVE_WORKER_PHASE, SAVE_LOCK_PHASE)

#: G1's gated quantity is positive RSS growth; the signed slope stays recorded.
SLOPE_CLAMP_RULE = "clamp_negative_to_zero"
SLOPE_UNIT = "bytes/turn"

#: The gated campaign IDs of the closed registry, in registry order.
GATED_CAMPAIGN_IDS: tuple[str, ...] = (
    "A1",
    "A2",
    "A3",
    "A4",
    "B1",
    "B2",
    "B3",
    "B4",
    "C1",
    "C2",
    "C3",
    "D1",
    "E1",
    "E2",
    "F1",
    "F2",
    "G1",
)

CELL_FRESH_HEADLESS = "x1_fresh_headless_fb10_c4"
CELL_LOADED_HEADLESS = "x2_loaded_headless_8x200k"
CELL_SAVE_LOADED = "save_loaded_200k"
CELL_PILOT_STREAMING = "pilot_streaming_fresh"
CELL_STORM = "storm_8child"
CELL_PERMISSION = "permission_fake_response"
CELL_SOAK_LATENCY = "soak_loaded_latency_turns301_400"
CELL_SOAK_MEMORY = "soak_loaded_memory_windows200_400"

TURN_OVERHEAD = "turn_application_overhead_p95_ms"
TRANSPORT_TTFB = "transport_ttfb_p95_ms"
CHUNK_GAP = "transport_chunk_gap_p95_ms"
HEARTBEAT_EXCESS = "loop_heartbeat_excess_p95_ms"
NOTIFICATION_COUNT = "notification_count_p95_per_root_turn"
NOTIFICATION_BYTES = "notification_bytes_p95_per_root_turn"
SAVE_WORKER_SCAN = "save_worker_scan_p95_ms"
SAVE_LOCK_HELD_AWAIT = "save_lock_held_await_p95_ms"
RENDER_FLUSH = "render_flush_p95_ms"
STORM_DRAIN = "completion_storm_drain_p95_ms"
PERMISSION_ROUND_TRIP = "permission_round_trip_p95_ms"
RSS_SLOPE = "rss_slope_p95_bytes_per_turn"

UNIT_MS = "ms"
UNIT_EVENTS = "events"
UNIT_BYTES = "bytes"

COVERAGE_TURNS = "turns"
COVERAGE_HEARTBEAT_ROWS = "heartbeat rows"
COVERAGE_ROOT_TURNS = "root turns"
COVERAGE_SAVES = "saves"
COVERAGE_RENDER_FLUSHES = "render flushes"
COVERAGE_STORMS = "storms"
COVERAGE_PERMISSION_ROUND_TRIPS = "permission round trips"
COVERAGE_LATE_HEARTBEAT_ROWS = "late-window heartbeat rows"
COVERAGE_ROLLING_WINDOWS = "rolling windows"

COUNT_TURNS = "turns"
COUNT_CHUNK_GAPS = "chunk_gaps"
COUNT_ROOT_TURNS = "root_turns"
COUNT_SAVES = "saves"
COUNT_RENDER_FLUSHES = "render_flushes"
COUNT_STORMS = "storms"
COUNT_PERMISSION_ROUND_TRIPS = "permission_round_trips"
COUNT_LATE_HEARTBEAT_ROWS = "late_window_heartbeat_rows"
COUNT_RSS_SAMPLES = "rss_samples"
COUNT_WINDOW_SLOPES = "window_slopes"

#: Fields every run record carries; the row extensions below are conditional.
REQUIRED_RUN_FIELDS: tuple[str, ...] = (
    "schema",
    "metric_version",
    "run_id",
    "run_start_utc",
    "mode",
    "campaign_id",
    "cell",
    "metric",
    "unit",
    "producer_wp",
    "chunks_coverage",
    "samples",
    "summary",
    "eligibility",
    "campaign",
)

CHUNKS_COVERAGE_FIELDS: tuple[str, ...] = (
    "unit",
    "measured_ops",
    "warmup_ops",
    "unit_counts",
    "complete",
    "missing",
)

_P50_FRACTION = 0.50
_P95_FRACTION = 0.95
_SAMPLE_PRECISION = 4


class ReportSchemaError(ValueError):
    """A run record or report document violates the runtime-perf schema."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"runtime report rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class LateWindow:
    """The pinned measured-window bounds of one late-window row."""

    warm_boundary_turn: int
    first_measured_turn: int
    last_measured_turn: int
    measured_turns: int | None = None
    sample_count: int | None = None
    window_width_turns: int | None = None
    window_count: int | None = None

    def as_record(self) -> dict[str, int]:
        """Return the record form: pinned bounds, inapplicable fields omitted."""
        record = {
            "warm_boundary_turn": self.warm_boundary_turn,
            "first_measured_turn": self.first_measured_turn,
            "last_measured_turn": self.last_measured_turn,
        }
        optional = (
            ("measured_turns", self.measured_turns),
            ("sample_count", self.sample_count),
            ("window_width_turns", self.window_width_turns),
            ("window_count", self.window_count),
        )
        for key, value in optional:
            if value is not None:
                record[key] = value
        return record


#: Plan §4 F1/F2: turns 301-400 are measured, turn 300 is the warm boundary.
SOAK_LATE_WINDOW = LateWindow(
    warm_boundary_turn=300,
    first_measured_turn=301,
    last_measured_turn=400,
    measured_turns=100,
)

#: Plan §4 G1: windows [200,300]..[300,400] over turns 200..400's RSS samples.
MEMORY_WINDOW_SPAN = LateWindow(
    warm_boundary_turn=199,
    first_measured_turn=200,
    last_measured_turn=400,
    sample_count=201,
    window_width_turns=100,
    window_count=101,
)


@dataclass(frozen=True, slots=True)
class RowSpec:
    """One gated campaign row: identity, floors, and its record extension."""

    campaign_id: str
    metric: str
    unit: str
    cell: str
    producer_wp: str
    coverage_unit: str
    measured_floor: int
    warmup_floor: int = DEFAULT_WARMUP_OPS
    exact_measured: int | None = None
    unit_count_floors: Mapping[str, int] = field(default_factory=dict)
    unit_count_exact: Mapping[str, int] = field(default_factory=dict)
    #: The unit-count key the gated sample series is counted by, when it is
    #: not the measured op count itself (A3's gaps, G1's window slopes).
    gated_series_key: str | None = None
    late_window: LateWindow | None = None
    slope_unit: str | None = None
    #: The C1/C2 save phase whose series is the row's gated samples.
    save_phase: str | None = None

    @property
    def required_unit_counts(self) -> tuple[str, ...]:
        """Return the row-specific count fields a run record must carry."""
        return tuple(dict.fromkeys((*self.unit_count_floors, *self.unit_count_exact)))

    @property
    def series_source(self) -> str:
        """Return where the row's gated sample series comes from."""
        if self.save_phase is not None:
            return self.save_phase
        return self.gated_series_key or "measured_ops"


_ROW_SPECS: tuple[RowSpec, ...] = (
    RowSpec(
        campaign_id="A1",
        metric=TURN_OVERHEAD,
        unit=UNIT_MS,
        cell=CELL_FRESH_HEADLESS,
        producer_wp="X1",
        coverage_unit=COVERAGE_TURNS,
        measured_floor=100,
    ),
    RowSpec(
        campaign_id="A2",
        metric=TRANSPORT_TTFB,
        unit=UNIT_MS,
        cell=CELL_FRESH_HEADLESS,
        producer_wp="X1",
        coverage_unit=COVERAGE_TURNS,
        measured_floor=100,
    ),
    RowSpec(
        campaign_id="A3",
        metric=CHUNK_GAP,
        unit=UNIT_MS,
        cell=CELL_FRESH_HEADLESS,
        producer_wp="X1",
        coverage_unit=COVERAGE_TURNS,
        measured_floor=100,
        unit_count_floors={COUNT_CHUNK_GAPS: 300},
        gated_series_key=COUNT_CHUNK_GAPS,
    ),
    RowSpec(
        campaign_id="A4",
        metric=HEARTBEAT_EXCESS,
        unit=UNIT_MS,
        cell=CELL_FRESH_HEADLESS,
        producer_wp="X1",
        coverage_unit=COVERAGE_HEARTBEAT_ROWS,
        measured_floor=1000,
    ),
    RowSpec(
        campaign_id="B1",
        metric=TURN_OVERHEAD,
        unit=UNIT_MS,
        cell=CELL_LOADED_HEADLESS,
        producer_wp="X1",
        coverage_unit=COVERAGE_TURNS,
        measured_floor=100,
    ),
    RowSpec(
        campaign_id="B2",
        metric=HEARTBEAT_EXCESS,
        unit=UNIT_MS,
        cell=CELL_LOADED_HEADLESS,
        producer_wp="X1",
        coverage_unit=COVERAGE_HEARTBEAT_ROWS,
        measured_floor=1000,
    ),
    RowSpec(
        campaign_id="B3",
        metric=NOTIFICATION_COUNT,
        unit=UNIT_EVENTS,
        cell=CELL_LOADED_HEADLESS,
        producer_wp="X2",
        coverage_unit=COVERAGE_ROOT_TURNS,
        measured_floor=100,
        unit_count_floors={COUNT_ROOT_TURNS: 100},
    ),
    RowSpec(
        campaign_id="B4",
        metric=NOTIFICATION_BYTES,
        unit=UNIT_BYTES,
        cell=CELL_LOADED_HEADLESS,
        producer_wp="X2",
        coverage_unit=COVERAGE_ROOT_TURNS,
        measured_floor=100,
        unit_count_floors={COUNT_ROOT_TURNS: 100},
    ),
    RowSpec(
        campaign_id="C1",
        metric=SAVE_WORKER_SCAN,
        unit=UNIT_MS,
        cell=CELL_SAVE_LOADED,
        producer_wp="P5-Reanchor",
        coverage_unit=COVERAGE_SAVES,
        measured_floor=100,
        unit_count_floors={COUNT_SAVES: 100},
        save_phase=SAVE_WORKER_PHASE,
    ),
    RowSpec(
        campaign_id="C2",
        metric=SAVE_LOCK_HELD_AWAIT,
        unit=UNIT_MS,
        cell=CELL_SAVE_LOADED,
        producer_wp="P5-Reanchor",
        coverage_unit=COVERAGE_SAVES,
        measured_floor=100,
        unit_count_floors={COUNT_SAVES: 100},
        save_phase=SAVE_LOCK_PHASE,
    ),
    RowSpec(
        campaign_id="C3",
        metric=HEARTBEAT_EXCESS,
        unit=UNIT_MS,
        cell=CELL_SAVE_LOADED,
        producer_wp="P5-Reanchor",
        coverage_unit=COVERAGE_HEARTBEAT_ROWS,
        measured_floor=1000,
    ),
    RowSpec(
        campaign_id="D1",
        metric=RENDER_FLUSH,
        unit=UNIT_MS,
        cell=CELL_PILOT_STREAMING,
        producer_wp="WP2",
        coverage_unit=COVERAGE_RENDER_FLUSHES,
        measured_floor=100,
        unit_count_floors={COUNT_RENDER_FLUSHES: 100, COUNT_TURNS: 100},
    ),
    RowSpec(
        campaign_id="E1",
        metric=STORM_DRAIN,
        unit=UNIT_MS,
        cell=CELL_STORM,
        producer_wp="X8b",
        coverage_unit=COVERAGE_STORMS,
        measured_floor=100,
        unit_count_floors={COUNT_STORMS: 100},
    ),
    RowSpec(
        campaign_id="E2",
        metric=PERMISSION_ROUND_TRIP,
        unit=UNIT_MS,
        cell=CELL_PERMISSION,
        producer_wp="X8d",
        coverage_unit=COVERAGE_PERMISSION_ROUND_TRIPS,
        measured_floor=100,
        unit_count_floors={COUNT_PERMISSION_ROUND_TRIPS: 100},
    ),
    RowSpec(
        campaign_id="F1",
        metric=TURN_OVERHEAD,
        unit=UNIT_MS,
        cell=CELL_SOAK_LATENCY,
        producer_wp="X3a",
        coverage_unit=COVERAGE_TURNS,
        measured_floor=100,
        warmup_floor=300,
        exact_measured=100,
        late_window=SOAK_LATE_WINDOW,
    ),
    RowSpec(
        campaign_id="F2",
        metric=HEARTBEAT_EXCESS,
        unit=UNIT_MS,
        cell=CELL_SOAK_LATENCY,
        producer_wp="X3a",
        coverage_unit=COVERAGE_LATE_HEARTBEAT_ROWS,
        measured_floor=1000,
        warmup_floor=300,
        unit_count_floors={COUNT_LATE_HEARTBEAT_ROWS: 1000},
        late_window=SOAK_LATE_WINDOW,
    ),
    RowSpec(
        campaign_id="G1",
        metric=RSS_SLOPE,
        unit=SLOPE_UNIT,
        cell=CELL_SOAK_MEMORY,
        producer_wp="X3b",
        coverage_unit=COVERAGE_ROLLING_WINDOWS,
        measured_floor=101,
        exact_measured=101,
        unit_count_exact={COUNT_RSS_SAMPLES: 201, COUNT_WINDOW_SLOPES: 101},
        gated_series_key=COUNT_WINDOW_SLOPES,
        late_window=MEMORY_WINDOW_SPAN,
        slope_unit=SLOPE_UNIT,
    ),
)

ROW_SPECS: dict[str, RowSpec] = {spec.campaign_id: spec for spec in _ROW_SPECS}


def row_spec(campaign_id: str) -> RowSpec:
    """Return the gated row's spec, raising for an unknown campaign id."""
    spec = ROW_SPECS.get(campaign_id)
    if spec is None:
        raise ReportSchemaError(
            "unknown_campaign_id", f"{campaign_id!r} is not a gated campaign id"
        )
    return spec


def nearest_rank(samples: Sequence[float], *, fraction: float) -> float:
    """Return the nearest-rank percentile: the k-th smallest, k = ceil(f*n)."""
    if not samples:
        raise ValueError("nearest_rank requires at least one sample")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(samples)
    return ordered[max(1, math.ceil(fraction * len(ordered))) - 1]


def _sample_series(values: Sequence[float]) -> list[float]:
    return [round(float(value), _SAMPLE_PRECISION) for value in values]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(_is_number(item) for item in value)


@dataclass(frozen=True, slots=True)
class FloorVerdict:
    """One run's chunks-coverage floor outcome, one reason per failure."""

    eligible: bool
    reasons: tuple[str, ...]


def evaluate_run_floor(
    campaign_id: str,
    *,
    measured_ops: int,
    warmup_ops: int,
    unit_counts: Mapping[str, int] | None = None,
) -> FloorVerdict:
    """Apply the row's chunks-coverage floors to one run's counts."""
    spec = row_spec(campaign_id)
    counts: Mapping[str, int] = {} if unit_counts is None else unit_counts
    reasons: list[str] = []
    if measured_ops < spec.measured_floor:
        reasons.append("ops_floor_measured")
    if spec.exact_measured is not None and measured_ops != spec.exact_measured:
        reasons.append("ops_floor_exact_measured")
    if warmup_ops < spec.warmup_floor:
        reasons.append("ops_floor_warmup")
    for key in spec.required_unit_counts:
        if key not in counts:
            reasons.append(f"coverage_missing_{key}")
            continue
        floor = spec.unit_count_floors.get(key)
        if floor is not None and counts[key] < floor:
            reasons.append(f"coverage_floor_{key}")
        exact = spec.unit_count_exact.get(key)
        if exact is not None and counts[key] != exact:
            reasons.append(f"coverage_exact_{key}")
    return FloorVerdict(eligible=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True, slots=True)
class SaveTimings:
    """Both C1/C2 save-timing phases of one save-cell run.

    ``worker_thread_ident`` is the ``to_thread`` worker that runs the scan and
    ``loop_thread_ident`` is the awaiting event-loop thread; the plan requires
    their distinctness (and the thread identity) to be asserted, so the record
    carries both.
    """

    worker_scan_ms: Sequence[float]
    lock_held_await_ms: Sequence[float]
    worker_thread_ident: int
    loop_thread_ident: int


def _slope_record(slope_unit: str, signed: Sequence[float] | None) -> dict[str, Any]:
    """Build the signed-plus-clamped slope record of a G1 run."""
    if signed is None:
        return {
            "unit": slope_unit,
            "signed": None,
            "clamped": None,
            "clamp": SLOPE_CLAMP_RULE,
            "null_reason": "no_slopes_measured",
        }
    signed_values = _sample_series(signed)
    return {
        "unit": slope_unit,
        "signed": signed_values,
        "clamped": [max(0.0, value) for value in signed_values],
        "clamp": SLOPE_CLAMP_RULE,
    }


def _save_record(timings: SaveTimings | None) -> dict[str, Any]:
    """Build the C1/C2 save-timing record, both phases included."""
    if timings is None:
        return {
            "unit": SAVE_UNIT,
            SAVE_WORKER_PHASE: None,
            SAVE_LOCK_PHASE: None,
            "null_reason": "no_save_timings",
        }
    distinct = timings.worker_thread_ident != timings.loop_thread_ident
    return {
        "unit": SAVE_UNIT,
        SAVE_WORKER_PHASE: _sample_series(timings.worker_scan_ms),
        SAVE_LOCK_PHASE: _sample_series(timings.lock_held_await_ms),
        "worker_thread_ident": timings.worker_thread_ident,
        "loop_thread_ident": timings.loop_thread_ident,
        "threads_distinct": distinct,
    }


def _coverage_record(
    spec: RowSpec, *, measured_ops: int, warmup_ops: int, counts: Mapping[str, int]
) -> dict[str, Any]:
    """Build the chunks-coverage block, absent row counts recorded as missing."""
    missing = [key for key in spec.required_unit_counts if key not in counts]
    return {
        "unit": spec.coverage_unit,
        "measured_ops": measured_ops,
        "warmup_ops": warmup_ops,
        "unit_counts": {key: counts[key] for key in sorted(counts)},
        "complete": not missing,
        "missing": missing,
    }


def _summary_record(
    gated: Sequence[float] | None, *, signed_slopes: Sequence[float] | None = None
) -> dict[str, Any]:
    """Summarize the gated series; an empty series carries NULL and its reason."""
    if not gated:
        summary: dict[str, Any] = {
            "count": 0,
            "p50": None,
            "p95": None,
            "max": None,
            "null_reason": "no_gated_samples",
        }
        if signed_slopes is not None:
            summary["p95_signed"] = None
        return summary
    summary = {
        "count": len(gated),
        "p50": nearest_rank(gated, fraction=_P50_FRACTION),
        "p95": nearest_rank(gated, fraction=_P95_FRACTION),
        "max": max(gated),
    }
    if signed_slopes is not None:
        summary["p95_signed"] = nearest_rank(signed_slopes, fraction=_P95_FRACTION)
    return summary


def build_run_record(
    campaign_id: str,
    *,
    run_id: str,
    run_start_utc: str,
    measured_ops: int,
    warmup_ops: int,
    samples: Sequence[float] | None = None,
    unit_counts: Mapping[str, int] | None = None,
    mode: str = MODE_MEASURE,
    slopes_signed: Sequence[float] | None = None,
    save_timings: SaveTimings | None = None,
    closure_status: str | None = None,
    closure_note: str | None = None,
    fixture_hash: str | None = None,
    machine: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one run record for a gated row, validating it before returning.

    The gated sample series is derived from the row's own source: C1/C2 records
    take it from the matching save phase, G1 records from the clamped slopes,
    and every other row from ``samples``. ``closure_status`` may be NULL for a
    campaign that has not closed yet, and then ``closure_note`` records why.
    """
    spec = row_spec(campaign_id)
    if mode not in MODES:
        raise ReportSchemaError("mode_invalid", f"{mode!r} is not a run mode")
    if closure_status is not None and closure_status not in CLOSURE_STATUSES:
        raise ReportSchemaError(
            "closure_status_invalid", f"{closure_status!r} is not a closure status"
        )
    if closure_status is None and not _is_non_empty_text(closure_note):
        raise ReportSchemaError(
            "null_without_reason", "an open campaign records its closure reason"
        )
    if spec.slope_unit is None and slopes_signed is not None:
        raise ReportSchemaError(
            "slopes_not_applicable", f"{campaign_id} carries no slope record"
        )
    if spec.save_phase is None and save_timings is not None:
        raise ReportSchemaError(
            "save_timings_not_applicable", f"{campaign_id} carries no save timings"
        )

    counts = dict(unit_counts or {})
    slopes = (
        _slope_record(spec.slope_unit, slopes_signed)
        if spec.slope_unit is not None
        else None
    )
    saves = _save_record(save_timings) if spec.save_phase is not None else None
    signed_values = slopes["signed"] if slopes is not None else None
    clamped = slopes["clamped"] if slopes is not None else None
    phase_series = (
        saves[spec.save_phase] if saves is not None and spec.save_phase else None
    )

    if spec.save_phase is not None:
        gated = phase_series
        if gated is not None and samples is not None:
            if _sample_series(samples) != gated:
                raise ReportSchemaError(
                    "samples_phase_mismatch",
                    f"C1/C2 samples must be the {spec.save_phase} series",
                )
    elif spec.slope_unit is not None:
        gated = clamped
        if gated is not None and samples is not None:
            if _sample_series(samples) != gated:
                raise ReportSchemaError(
                    "samples_slope_mismatch", "G1 samples must be the clamped slopes"
                )
    else:
        gated = None if samples is None else _sample_series(samples)

    coverage = _coverage_record(
        spec, measured_ops=measured_ops, warmup_ops=warmup_ops, counts=counts
    )
    reasons = list(
        evaluate_run_floor(
            campaign_id,
            measured_ops=measured_ops,
            warmup_ops=warmup_ops,
            unit_counts=counts,
        ).reasons
    )
    if not gated:
        reasons.append("no_gated_samples")

    record: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "metric_version": METRIC_VERSION,
        "run_id": run_id,
        "run_start_utc": run_start_utc,
        "mode": mode,
        "campaign_id": campaign_id,
        "cell": spec.cell,
        "metric": spec.metric,
        "unit": spec.unit,
        "producer_wp": spec.producer_wp,
        "chunks_coverage": coverage,
        "samples": [] if gated is None else gated,
        "summary": _summary_record(gated, signed_slopes=signed_values),
        "eligibility": {"eligible": not reasons, "reasons": reasons},
        "campaign": {
            "campaign_id": campaign_id,
            "cell": spec.cell,
            "closure_status": closure_status,
            "closure_note": closure_note,
        },
    }
    if spec.late_window is not None:
        record["late_window"] = spec.late_window.as_record()
    if slopes is not None:
        record["slopes"] = slopes
    if saves is not None:
        record["save_timings"] = saves
    if fixture_hash is not None:
        record["fixture_hash"] = fixture_hash
    if machine is not None:
        record["machine"] = dict(machine)
    validate_run_record(record)
    return record


def _check_identity(record: Mapping[str, Any], spec: RowSpec) -> None:
    if not _is_non_empty_text(record["run_id"]):
        raise ReportSchemaError("run_id_invalid", f"{record['run_id']!r}")
    if not _is_non_empty_text(record["run_start_utc"]):
        raise ReportSchemaError("run_start_invalid", f"{record['run_start_utc']!r}")
    identity = (
        ("cell", spec.cell, "cell_mismatch"),
        ("metric", spec.metric, "metric_mismatch"),
        ("unit", spec.unit, "unit_mismatch"),
        ("producer_wp", spec.producer_wp, "producer_mismatch"),
    )
    for field_name, expected, reason in identity:
        if record[field_name] != expected:
            raise ReportSchemaError(
                reason, f"{field_name}={record[field_name]!r}, expected {expected!r}"
            )


def _non_negative_count(container: Mapping[str, Any], name: str) -> int:
    """Return a non-negative integer count field, or reject the record."""
    value = container.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportSchemaError("coverage_count_invalid", f"{name}={value!r}")
    return value


def _coverage_fields(
    record: Mapping[str, Any], spec: RowSpec
) -> tuple[int, dict[str, int]]:
    """Validate the chunks-coverage block and return its counts."""
    coverage = record["chunks_coverage"]
    if not isinstance(coverage, Mapping):
        raise ReportSchemaError("chunks_coverage_invalid", "not a mapping")
    if coverage.get("unit") != spec.coverage_unit:
        raise ReportSchemaError(
            "coverage_unit_mismatch",
            f"{coverage.get('unit')!r}, expected {spec.coverage_unit!r}",
        )
    measured_ops = _non_negative_count(coverage, "measured_ops")
    _non_negative_count(coverage, "warmup_ops")
    raw_counts = coverage.get("unit_counts")
    if not isinstance(raw_counts, Mapping):
        raise ReportSchemaError(
            "coverage_count_invalid", "unit_counts is not a mapping"
        )
    counts: dict[str, int] = {}
    for key, value in raw_counts.items():
        if not isinstance(key, str) or isinstance(value, bool):
            raise ReportSchemaError(
                "coverage_count_invalid", f"unit_counts[{key!r}]={value!r}"
            )
        if not isinstance(value, int) or value < 0:
            raise ReportSchemaError(
                "coverage_count_invalid", f"unit_counts[{key!r}]={value!r}"
            )
        counts[key] = value
    missing = coverage.get("missing")
    if not isinstance(missing, list) or not all(
        isinstance(key, str) for key in missing
    ):
        raise ReportSchemaError(
            "coverage_completeness_inconsistent", f"missing={missing!r}"
        )
    expected_missing = [key for key in spec.required_unit_counts if key not in counts]
    if sorted(missing) != sorted(expected_missing):
        raise ReportSchemaError(
            "coverage_missing_fields",
            f"missing={sorted(missing)!r}, expected {sorted(expected_missing)!r}",
        )
    complete = coverage.get("complete")
    if not isinstance(complete, bool) or complete != (not expected_missing):
        raise ReportSchemaError(
            "coverage_completeness_inconsistent", f"complete={complete!r}"
        )
    return measured_ops, counts


def _check_late_window(record: Mapping[str, Any], spec: RowSpec) -> None:
    expected = None if spec.late_window is None else spec.late_window.as_record()
    actual = record.get("late_window")
    if expected is None:
        if actual is not None:
            raise ReportSchemaError(
                "late_window_not_applicable", f"{spec.campaign_id} has no late window"
            )
        return
    if not isinstance(actual, Mapping):
        raise ReportSchemaError("late_window_missing", "late-window row without bounds")
    if dict(actual) != expected:
        raise ReportSchemaError(
            "late_window_mismatch", f"{dict(actual)!r}, expected {expected!r}"
        )


def _slope_series(record: Mapping[str, Any]) -> list[float] | None:
    slopes = record["slopes"]
    signed = slopes.get("signed") if isinstance(slopes, Mapping) else None
    return None if signed is None else list(signed)


def _check_slopes(record: Mapping[str, Any], spec: RowSpec) -> None:
    slopes = record.get("slopes")
    if spec.slope_unit is None:
        if slopes is not None:
            raise ReportSchemaError(
                "slopes_not_applicable", f"{spec.campaign_id} carries no slope record"
            )
        return
    if not isinstance(slopes, Mapping):
        raise ReportSchemaError("slopes_missing", "slope row without a slope record")
    if slopes.get("unit") != spec.slope_unit:
        raise ReportSchemaError("slope_unit_mismatch", f"{slopes.get('unit')!r}")
    if slopes.get("clamp") != SLOPE_CLAMP_RULE:
        raise ReportSchemaError("slope_clamp_rule_mismatch", f"{slopes.get('clamp')!r}")
    signed = slopes.get("signed")
    clamped = slopes.get("clamped")
    if signed is None or clamped is None:
        if signed is not None or clamped is not None:
            raise ReportSchemaError("slope_null_mismatch", "one slope series is NULL")
        if not _is_non_empty_text(slopes.get("null_reason")):
            raise ReportSchemaError(
                "null_without_reason", "NULL slopes without a reason"
            )
        return
    if not _is_number_list(signed) or not _is_number_list(clamped):
        raise ReportSchemaError("slope_series_invalid", "slopes must be numbers")
    if len(signed) != len(clamped):
        raise ReportSchemaError(
            "slope_length_mismatch", f"{len(signed)} vs {len(clamped)}"
        )
    for measured, gated in zip(signed, clamped, strict=True):
        if gated != max(0.0, measured):
            raise ReportSchemaError(
                "slope_clamp_mismatch", f"clamped {gated!r} for signed {measured!r}"
            )


def _check_save_timings(
    record: Mapping[str, Any], spec: RowSpec, measured_ops: int
) -> None:
    timings = record.get("save_timings")
    if spec.save_phase is None:
        if timings is not None:
            raise ReportSchemaError(
                "save_timings_not_applicable", f"{spec.campaign_id} has no save phases"
            )
        return
    if not isinstance(timings, Mapping):
        raise ReportSchemaError("missing_save_timings", "save row without timings")
    if timings.get("unit") != SAVE_UNIT:
        raise ReportSchemaError("save_unit_mismatch", f"{timings.get('unit')!r}")
    phases = [timings.get(phase) for phase in SAVE_PHASES]
    if all(phase is None for phase in phases):
        if not _is_non_empty_text(timings.get("null_reason")):
            raise ReportSchemaError(
                "null_without_reason", "NULL save timings without a reason"
            )
        return
    for phase in SAVE_PHASES:
        series = timings.get(phase)
        if not isinstance(series, list) or not all(_is_number(item) for item in series):
            raise ReportSchemaError("save_phase_missing", f"{phase} is not a series")
        if len(series) != measured_ops:
            raise ReportSchemaError(
                "save_phase_count_mismatch",
                f"{phase} carries {len(series)} samples for {measured_ops} saves",
            )
    worker = timings.get("worker_thread_ident")
    loop = timings.get("loop_thread_ident")
    if not _is_int(worker) or not _is_int(loop):
        raise ReportSchemaError(
            "save_thread_identity", "thread idents must be integers"
        )
    if worker == loop:
        raise ReportSchemaError(
            "save_thread_identity",
            "the scan runs in a worker thread, never on the loop thread",
        )
    if timings.get("threads_distinct") is not True:
        raise ReportSchemaError(
            "save_thread_flag_mismatch", "threads_distinct is not True"
        )


def _expected_series_length(
    spec: RowSpec, measured_ops: int, counts: Mapping[str, int]
) -> int | None:
    if spec.save_phase is not None:
        return measured_ops
    if spec.gated_series_key is None:
        return measured_ops
    return counts.get(spec.gated_series_key)


def _check_samples(
    record: Mapping[str, Any],
    spec: RowSpec,
    measured_ops: int,
    counts: Mapping[str, int],
) -> list[float]:
    samples = record["samples"]
    if not _is_number_list(samples):
        raise ReportSchemaError("samples_invalid", "samples must be numbers")
    gated = list(samples)
    if not gated:
        return gated
    expected = _expected_series_length(spec, measured_ops, counts)
    if expected is not None and len(gated) != expected:
        raise ReportSchemaError(
            "sample_count_mismatch",
            f"{len(gated)} samples for {expected} gated {spec.series_source}",
        )
    if spec.slope_unit is not None:
        clamped = record["slopes"].get("clamped")
        if clamped is not None and gated != list(clamped):
            raise ReportSchemaError(
                "samples_slope_mismatch", "G1 samples are not the clamped slopes"
            )
    if spec.save_phase is not None:
        phase = record["save_timings"].get(spec.save_phase)
        if phase is not None and gated != list(phase):
            raise ReportSchemaError(
                "samples_phase_mismatch",
                f"samples are not the {spec.save_phase} series",
            )
    return gated


def _check_summary(
    record: Mapping[str, Any], gated: Sequence[float], spec: RowSpec
) -> None:
    summary = record["summary"]
    if not isinstance(summary, Mapping):
        raise ReportSchemaError("summary_invalid", "summary is not a mapping")
    count = summary.get("count")
    if not _is_int(count) or count != len(gated):
        raise ReportSchemaError(
            "summary_count_mismatch", f"count={count!r} for {len(gated)} samples"
        )
    if not gated:
        for key in ("p50", "p95", "max"):
            if summary.get(key) is not None:
                raise ReportSchemaError("summary_mismatch", f"{key} is not NULL")
        if spec.slope_unit is not None and summary.get("p95_signed") is not None:
            raise ReportSchemaError("summary_mismatch", "p95_signed is not NULL")
        if not _is_non_empty_text(summary.get("null_reason")):
            raise ReportSchemaError(
                "null_without_reason", "NULL summary without a reason"
            )
        return
    expected = {
        "p50": nearest_rank(gated, fraction=_P50_FRACTION),
        "p95": nearest_rank(gated, fraction=_P95_FRACTION),
        "max": max(gated),
    }
    for key, value in expected.items():
        if not _is_number(summary.get(key)) or summary[key] != value:
            raise ReportSchemaError("summary_mismatch", f"{key}={summary.get(key)!r}")
    signed = _slope_series(record) if spec.slope_unit is not None else None
    if signed:
        wanted = nearest_rank(signed, fraction=_P95_FRACTION)
        if not _is_number(summary.get("p95_signed")) or summary["p95_signed"] != wanted:
            raise ReportSchemaError(
                "summary_mismatch", f"p95_signed={summary.get('p95_signed')!r}"
            )


def _check_eligibility(
    record: Mapping[str, Any],
    spec: RowSpec,
    measured_ops: int,
    counts: Mapping[str, int],
    gated: Sequence[float],
) -> None:
    eligibility = record["eligibility"]
    if not isinstance(eligibility, Mapping):
        raise ReportSchemaError("eligibility_invalid", "not a mapping")
    eligible = eligibility.get("eligible")
    reasons = eligibility.get("reasons")
    if not isinstance(eligible, bool) or not isinstance(reasons, list):
        raise ReportSchemaError("eligibility_invalid", f"{eligibility!r}")
    if not all(isinstance(reason, str) for reason in reasons):
        raise ReportSchemaError("eligibility_invalid", "reasons must be strings")
    if not eligible:
        if not reasons:
            raise ReportSchemaError(
                "eligibility_reason_missing", "an ineligible run records why"
            )
        return
    if reasons:
        raise ReportSchemaError(
            "eligibility_reasons_conflict", "an eligible run carries no reasons"
        )
    if not gated:
        raise ReportSchemaError("unsupported_eligibility", "no gated samples")
    coverage = record["chunks_coverage"]
    assert isinstance(coverage, Mapping)
    if coverage.get("complete") is not True:
        raise ReportSchemaError("unsupported_eligibility", "incomplete chunks coverage")
    verdict = evaluate_run_floor(
        spec.campaign_id,
        measured_ops=measured_ops,
        warmup_ops=int(coverage["warmup_ops"]),
        unit_counts=counts,
    )
    if not verdict.eligible:
        raise ReportSchemaError(
            "unsupported_eligibility", f"counts below the row floor: {verdict.reasons}"
        )


def _check_campaign(record: Mapping[str, Any], spec: RowSpec) -> None:
    campaign = record["campaign"]
    if not isinstance(campaign, Mapping):
        raise ReportSchemaError("campaign_invalid", "campaign is not a mapping")
    if campaign.get("campaign_id") != record["campaign_id"]:
        raise ReportSchemaError(
            "campaign_identity_mismatch", f"{campaign.get('campaign_id')!r}"
        )
    if campaign.get("cell") != spec.cell:
        raise ReportSchemaError(
            "campaign_identity_mismatch", f"{campaign.get('cell')!r}"
        )
    status = campaign.get("closure_status")
    note = campaign.get("closure_note")
    if status is None:
        if not _is_non_empty_text(note):
            raise ReportSchemaError(
                "null_without_reason", "an open campaign records its closure reason"
            )
        return
    if status not in CLOSURE_STATUSES:
        raise ReportSchemaError("closure_status_invalid", f"{status!r}")
    if note is not None and not isinstance(note, str):
        raise ReportSchemaError("closure_note_invalid", f"{note!r}")


def validate_run_record(record: Mapping[str, Any]) -> None:
    """Validate one run record against the schema, raising on the first breach."""
    if not isinstance(record, Mapping):
        raise ReportSchemaError("record_not_mapping", "a run record is a mapping")
    for field_name in REQUIRED_RUN_FIELDS:
        if field_name not in record:
            raise ReportSchemaError("missing_run_field", field_name)
    if record["schema"] != SCHEMA_ID:
        raise ReportSchemaError("schema_mismatch", f"{record['schema']!r}")
    if record["metric_version"] != METRIC_VERSION:
        raise ReportSchemaError(
            "metric_version_mismatch", f"{record['metric_version']!r}"
        )
    campaign_id = record["campaign_id"]
    if not isinstance(campaign_id, str):
        raise ReportSchemaError("unknown_campaign_id", f"{campaign_id!r}")
    spec = row_spec(campaign_id)
    if record["mode"] not in MODES:
        raise ReportSchemaError("mode_invalid", f"{record['mode']!r}")
    _check_identity(record, spec)
    measured_ops, counts = _coverage_fields(record, spec)
    _check_late_window(record, spec)
    _check_slopes(record, spec)
    _check_save_timings(record, spec, measured_ops)
    gated = _check_samples(record, spec, measured_ops, counts)
    _check_summary(record, gated, spec)
    _check_eligibility(record, spec, measured_ops, counts, gated)
    _check_campaign(record, spec)


def validate_report(document: Mapping[str, Any]) -> None:
    """Validate a report document's core sections, ignoring added sections."""
    if not isinstance(document, Mapping):
        raise ReportSchemaError("report_not_mapping", "a report is a mapping")
    if document.get("schema") != SCHEMA_ID:
        raise ReportSchemaError("schema_mismatch", f"{document.get('schema')!r}")
    if document.get("metric_version") != METRIC_VERSION:
        raise ReportSchemaError(
            "metric_version_mismatch", f"{document.get('metric_version')!r}"
        )
    runs = document.get("runs")
    if not isinstance(runs, list):
        raise ReportSchemaError("report_runs_invalid", f"{runs!r}")
    seen: set[tuple[str, str]] = set()
    for record in runs:
        validate_run_record(record)
        key = (record["campaign_id"], record["run_id"])
        if key in seen:
            raise ReportSchemaError("duplicate_run", f"{key!r} appears twice")
        seen.add(key)


def build_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assemble and validate a report document from run records."""
    document: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "metric_version": METRIC_VERSION,
        "runs": [dict(record) for record in records],
    }
    validate_report(document)
    return document


def _row_schema(spec: RowSpec) -> dict[str, Any]:
    return {
        "campaign_id": spec.campaign_id,
        "metric": spec.metric,
        "unit": spec.unit,
        "cell": spec.cell,
        "producer_wp": spec.producer_wp,
        "coverage_unit": spec.coverage_unit,
        "measured_floor": spec.measured_floor,
        "warmup_floor": spec.warmup_floor,
        "exact_measured": spec.exact_measured,
        "unit_counts": {
            "required": list(spec.required_unit_counts),
            "floors": dict(spec.unit_count_floors),
            "exact": dict(spec.unit_count_exact),
        },
        "gated_series": spec.series_source,
        "late_window": (
            None if spec.late_window is None else spec.late_window.as_record()
        ),
        "slope_unit": spec.slope_unit,
        "save_phase": spec.save_phase,
    }


def report_schema(*, extra_sections: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the schema document, merging additive sections into a copy.

    Core sections are never overwritten: a conflicting extra section is
    rejected, so a later package (X7b's comparison states, the PERF-BLOB
    record) adds its own section instead of editing the row rules.
    """
    document: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "metric_version": METRIC_VERSION,
        "report_fields": ("schema", "metric_version", "runs"),
        "run_fields": REQUIRED_RUN_FIELDS,
        "modes": MODES,
        "closure_statuses": CLOSURE_STATUSES,
        "chunks_coverage_fields": CHUNKS_COVERAGE_FIELDS,
        "slope_rule": {
            "unit": SLOPE_UNIT,
            "clamp": SLOPE_CLAMP_RULE,
            "reports": ("signed", "clamped"),
        },
        "save_phases": SAVE_PHASES,
        "rows": {spec.campaign_id: _row_schema(spec) for spec in _ROW_SPECS},
    }
    if extra_sections:
        conflicts = sorted(set(extra_sections) & set(document))
        if conflicts:
            raise ReportSchemaError(
                "section_conflict", f"core sections cannot be replaced: {conflicts}"
            )
        document.update(extra_sections)
    return document


REPORT_SCHEMA: dict[str, Any] = report_schema()


def _example_g1_slopes() -> tuple[float, ...]:
    """Return a slope series that is genuinely negative in its early windows."""
    return tuple(float(index) - 50.0 for index in range(101))


_EXAMPLE_CHUNK_GAPS = tuple(3.0 + (index % 7) * 0.5 for index in range(300))
_EXAMPLE_SAVE_PHASES = tuple(12.0 + (index % 5) * 0.25 for index in range(120))

SCHEMA_EXAMPLE: dict[str, Any] = build_report((
    build_run_record(
        "A3",
        run_id="a3-example-1",
        run_start_utc="2026-11-08T00:00:00Z",
        measured_ops=100,
        warmup_ops=10,
        samples=_EXAMPLE_CHUNK_GAPS,
        unit_counts={"chunk_gaps": 300},
        closure_status=None,
        closure_note="campaign open: 2 of 5 eligible runs collected",
    ),
    build_run_record(
        "C2",
        run_id="c2-example-1",
        run_start_utc="2026-11-08T00:10:00Z",
        measured_ops=120,
        warmup_ops=10,
        unit_counts={"saves": 120},
        save_timings=SaveTimings(
            worker_scan_ms=_EXAMPLE_SAVE_PHASES,
            lock_held_await_ms=_EXAMPLE_SAVE_PHASES,
            worker_thread_ident=4242,
            loop_thread_ident=4240,
        ),
        closure_status=CLOSURE_INSUFFICIENT_DATA,
        closure_note="10 attempted runs without 5 eligible runs",
    ),
    build_run_record(
        "G1",
        run_id="g1-example-1",
        run_start_utc="2026-11-08T01:00:00Z",
        measured_ops=101,
        warmup_ops=200,
        unit_counts={"rss_samples": 201, "window_slopes": 101},
        slopes_signed=_example_g1_slopes(),
        closure_status=CLOSURE_ELIGIBLE_COMPLETE,
    ),
))


# ---------------------------------------------------------------------------
# X7b: the additive comparison-states and PERF-BLOB reporting sections
# ---------------------------------------------------------------------------

#: The two sections X7b renders into the report. Neither is a core section, so
#: both merge through :func:`report_schema`'s ``extra_sections`` instead of
#: editing the run-record rules.
COMPARISON_STATES_SECTION = "comparison_states"
PERF_BLOB_SECTION = "perf_blob"

#: The marked comparison labels of Appendix B ``marked_comparison_outcomes``,
#: listed in their pinned total order: DORMANT > COMPARISON-INELIGIBLE >
#: threshold labels.
COMPARISON_STATE_DORMANT = "DORMANT"
COMPARISON_STATE_INELIGIBLE = "COMPARISON-INELIGIBLE"
COMPARISON_STATE_PASS = "ACTIVE-pass"
COMPARISON_STATE_FAIL = "ACTIVE-fail"
COMPARISON_STATE_PRECEDENCE: tuple[str, ...] = (
    COMPARISON_STATE_DORMANT,
    COMPARISON_STATE_INELIGIBLE,
    COMPARISON_STATE_PASS,
    COMPARISON_STATE_FAIL,
)
COMPARISON_STATE_RULE = "row first: DORMANT > COMPARISON-INELIGIBLE > threshold labels"

#: One rendered comparison row: the label with its precedence index, the reason
#: order's first applicable reason, the run-level facts, and the nullable
#: comparison fields with their reasons.
COMPARISON_STATE_ROW_FIELDS: tuple[str, ...] = (
    "campaign_id",
    "cell",
    "metric",
    "unit",
    "state",
    "precedence",
    "reason",
    "machine_mismatch_fields",
    "comparison_run",
    "direction",
    "measured_value",
    "threshold",
    "null_reasons",
)

#: The PERF-BLOB record kind and the one record type X7b emits.
PERF_BLOB_KIND = "perf_blob"
PERF_BLOB_MACHINE_MISMATCH = "machine_mismatch"
PERF_BLOB_RECORD_FIELDS: tuple[str, ...] = (
    "kind",
    "record",
    "cell",
    "campaign_ids",
    "mismatch_fields",
    "baseline_machine",
    "comparison_machine",
    "state",
    "reason",
    "never_active",
    "never_failed",
)


def reporting_extension() -> dict[str, Any]:
    """Return the X7b sections' descriptors for :func:`report_schema`.

    The descriptors name the labels, their precedence, and the fields of each
    rendered record; they are additive sections, so :func:`report_schema`
    merges them while refusing any attempt to replace a core section.
    """
    return {
        COMPARISON_STATES_SECTION: {
            "rule": COMPARISON_STATE_RULE,
            "precedence": list(COMPARISON_STATE_PRECEDENCE),
            "row_fields": list(COMPARISON_STATE_ROW_FIELDS),
        },
        PERF_BLOB_SECTION: {
            "kind": PERF_BLOB_KIND,
            "record": PERF_BLOB_MACHINE_MISMATCH,
            "state": COMPARISON_STATE_DORMANT,
            "reason": PERF_BLOB_MACHINE_MISMATCH,
            "fields": list(PERF_BLOB_RECORD_FIELDS),
        },
    }


def _comparison_state_row(row: Any) -> Mapping[str, Any]:
    """Return one rendered row, refusing a row that breaks the label rules."""
    if not isinstance(row, Mapping):
        raise ReportSchemaError("comparison_state_row_invalid", repr(row))
    for field_name in COMPARISON_STATE_ROW_FIELDS:
        if field_name not in row:
            raise ReportSchemaError("comparison_state_row_field_missing", field_name)
    spec = (
        ROW_SPECS.get(row["campaign_id"])
        if isinstance(row["campaign_id"], str)
        else None
    )
    if spec is None:
        raise ReportSchemaError("unknown_campaign_id", repr(row["campaign_id"]))
    if row["cell"] != spec.cell:
        raise ReportSchemaError("comparison_state_cell_mismatch", repr(row["cell"]))
    state = row["state"]
    if state not in COMPARISON_STATE_PRECEDENCE:
        raise ReportSchemaError("comparison_state_label_invalid", repr(state))
    if row["precedence"] != COMPARISON_STATE_PRECEDENCE.index(state):
        raise ReportSchemaError(
            "comparison_state_precedence_mismatch", f"{state}: {row['precedence']!r}"
        )
    if not _is_non_empty_text(row["reason"]):
        raise ReportSchemaError("comparison_state_reason_missing", repr(row["reason"]))
    if not _is_non_empty_text(row["direction"]):
        raise ReportSchemaError(
            "comparison_state_direction_missing", repr(row["direction"])
        )
    mismatch = row["machine_mismatch_fields"]
    if not isinstance(mismatch, list) or not all(
        isinstance(field, str) for field in mismatch
    ):
        raise ReportSchemaError(
            "comparison_state_mismatch_fields_invalid", repr(mismatch)
        )
    run_block = row["comparison_run"]
    if not isinstance(run_block, Mapping):
        raise ReportSchemaError("comparison_state_run_invalid", repr(run_block))
    eligible = run_block.get("eligible")
    if eligible not in (True, False, None):
        raise ReportSchemaError(
            "comparison_state_run_invalid", f"eligible={eligible!r}"
        )
    if eligible is None and not _is_non_empty_text(run_block.get("null_reason")):
        raise ReportSchemaError("null_without_reason", "a NULL run verdict records why")
    nulls = row["null_reasons"]
    if not isinstance(nulls, Mapping):
        raise ReportSchemaError("comparison_state_null_reasons_invalid", repr(nulls))
    measured = row["measured_value"]
    threshold = row["threshold"]
    for field_name, value in (("measured_value", measured), ("threshold", threshold)):
        if value is not None and not _is_number(value):
            raise ReportSchemaError(
                "comparison_state_field_invalid", f"{field_name}={value!r}"
            )
        if value is None and not _is_non_empty_text(nulls.get(field_name)):
            raise ReportSchemaError(
                "null_without_reason", f"{field_name} is NULL without its reason"
            )
    if mismatch and state != COMPARISON_STATE_DORMANT:
        raise ReportSchemaError(
            "comparison_state_mismatched_hardware",
            "a machine mismatch is DORMANT, never a threshold label",
        )
    if state == COMPARISON_STATE_DORMANT:
        if measured is not None:
            raise ReportSchemaError(
                "comparison_state_dormant_measured", "a dormant row compares nothing"
            )
        if eligible is not None:
            raise ReportSchemaError(
                "comparison_state_dormant_run",
                "row-first precedence forbids a run-level verdict",
            )
        return row
    if threshold is None or threshold <= 0:
        raise ReportSchemaError(
            "comparison_state_dormant_threshold",
            "only a non-dormant row carries a threshold",
        )
    if state == COMPARISON_STATE_INELIGIBLE:
        if measured is not None:
            raise ReportSchemaError(
                "comparison_state_ineligible_measured",
                "an ineligible run carries no threshold outcome",
            )
        if eligible is not False:
            raise ReportSchemaError(
                "comparison_state_ineligible_run", f"eligible={eligible!r}"
            )
        return row
    if measured is None or eligible is not True:
        raise ReportSchemaError(
            "comparison_state_threshold_outcome",
            f"{state}: measured={measured!r} eligible={eligible!r}",
        )
    return row


def validate_comparison_states_section(section: Mapping[str, Any]) -> None:
    """Validate a rendered comparison-states section's structure and labels."""
    if not isinstance(section, Mapping):
        raise ReportSchemaError("comparison_states_invalid", repr(section))
    if section.get("rule") != COMPARISON_STATE_RULE:
        raise ReportSchemaError(
            "comparison_states_rule_drift", repr(section.get("rule"))
        )
    precedence = section.get("precedence")
    if (
        not isinstance(precedence, list)
        or tuple(precedence) != COMPARISON_STATE_PRECEDENCE
    ):
        raise ReportSchemaError("comparison_states_precedence_drift", repr(precedence))
    rows = section.get("rows")
    if not isinstance(rows, list):
        raise ReportSchemaError("comparison_states_rows_invalid", repr(rows))
    counts = {label: 0 for label in COMPARISON_STATE_PRECEDENCE}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        checked = _comparison_state_row(row)
        counts[checked["state"]] += 1
        key = (str(checked["cell"]), str(checked["campaign_id"]))
        if key in seen:
            raise ReportSchemaError("comparison_states_duplicate_row", repr(key))
        seen.add(key)
    declared = section.get("counts")
    if not isinstance(declared, Mapping) or dict(declared) != counts:
        raise ReportSchemaError(
            "comparison_states_counts_mismatch", f"{declared!r} != {counts!r}"
        )


def _perf_blob_record(record: Any) -> Mapping[str, Any]:
    """Return one PERF-BLOB record, refusing one that is not a machine mismatch."""
    if not isinstance(record, Mapping):
        raise ReportSchemaError("perf_blob_record_invalid", repr(record))
    for field_name in PERF_BLOB_RECORD_FIELDS:
        if field_name not in record:
            raise ReportSchemaError("perf_blob_field_missing", field_name)
    if record["kind"] != PERF_BLOB_KIND:
        raise ReportSchemaError("perf_blob_kind_invalid", repr(record["kind"]))
    if record["record"] != PERF_BLOB_MACHINE_MISMATCH:
        raise ReportSchemaError("perf_blob_record_kind_invalid", repr(record["record"]))
    if record["state"] != COMPARISON_STATE_DORMANT:
        raise ReportSchemaError(
            "perf_blob_state_not_dormant",
            f"{record['state']!r}: a machine mismatch is DORMANT, never"
            " ACTIVE and never FAILED",
        )
    if record["reason"] != PERF_BLOB_MACHINE_MISMATCH:
        raise ReportSchemaError("perf_blob_reason_invalid", repr(record["reason"]))
    if record["never_active"] is not True or record["never_failed"] is not True:
        raise ReportSchemaError(
            "perf_blob_disposition_invalid",
            "a machine mismatch is never marked ACTIVE and never marked FAILED",
        )
    if not _is_non_empty_text(record["cell"]):
        raise ReportSchemaError("perf_blob_cell_invalid", repr(record["cell"]))
    ids = record["campaign_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or not all(_is_non_empty_text(campaign_id) for campaign_id in ids)
    ):
        raise ReportSchemaError("perf_blob_campaign_ids_invalid", repr(ids))
    mismatch = record["mismatch_fields"]
    if (
        not isinstance(mismatch, list)
        or not mismatch
        or not all(_is_non_empty_text(field) for field in mismatch)
    ):
        raise ReportSchemaError("perf_blob_mismatch_fields_invalid", repr(mismatch))
    baseline = record["baseline_machine"]
    other = record["comparison_machine"]
    if not isinstance(baseline, Mapping) or not isinstance(other, Mapping):
        raise ReportSchemaError(
            "perf_blob_machine_invalid", "both machine contexts are mappings"
        )
    for field_name in mismatch:
        if field_name not in baseline or field_name not in other:
            raise ReportSchemaError(
                "perf_blob_machine_field_missing",
                f"{field_name!r} is not recorded on both sides",
            )
        if baseline[field_name] == other[field_name]:
            raise ReportSchemaError(
                "perf_blob_mismatch_unproven",
                f"{field_name!r} is equal in both machine contexts",
            )
    return record


def validate_perf_blob_section(section: Mapping[str, Any]) -> None:
    """Validate a PERF-BLOB section: its records are machine mismatches only."""
    if not isinstance(section, Mapping):
        raise ReportSchemaError("perf_blob_invalid", repr(section))
    if section.get("kind") != PERF_BLOB_KIND:
        raise ReportSchemaError("perf_blob_kind_invalid", repr(section.get("kind")))
    records = section.get("records")
    if not isinstance(records, list):
        raise ReportSchemaError("perf_blob_records_invalid", repr(records))
    for record in records:
        _perf_blob_record(record)


def validate_reporting_sections(document: Mapping[str, Any]) -> None:
    """Validate whichever additive reporting sections a report carries."""
    if not isinstance(document, Mapping):
        raise ReportSchemaError("report_not_mapping", "a report is a mapping")
    if COMPARISON_STATES_SECTION in document:
        validate_comparison_states_section(document[COMPARISON_STATES_SECTION])
    if PERF_BLOB_SECTION in document:
        validate_perf_blob_section(document[PERF_BLOB_SECTION])
