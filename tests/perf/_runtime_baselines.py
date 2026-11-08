"""Per-metric baseline validation for the v0.7.5 stage-2 runtime gate (X5a).

A baseline campaign collects five fresh-process runs of one CELL. Every campaign
ID measured in that cell is co-measured in the same runs, so the validation unit
is the cell's SHARED RUN SET rather than one row's series: a run is eligible only
when the CONJUNCTION of every co-measured row's requirements holds for it — each
row's floors, its chunks-coverage completeness, its sample validity and the
invalid-sample cap. A run failing any one row is excluded for ALL the cell's
campaign IDs, and its attempt record is preserved with the failing row's id.

Selection is the latest five eligible runs ordered by run start time with the
run id as the tie-breaker. :func:`select_latest_eligible` accepts run identities
only, so a measured value is structurally unable to take part in selection.

A campaign closes in exactly one of two ways: ``eligible_complete`` (five
eligible runs collected) or ``insufficient_data`` (ten attempted runs without
five eligible). An open campaign records NULL with its reason and is refused by
the promotion gate. The assembled baseline carries every attempt, the selected
runs' per-row p95s, the closure, and a digest of the attempt list, so a promoted
artifact that drops an attempt or re-orders the selection fails here.

X5a validates and assembles: no threshold and no dormancy disposition is
computed for promotion. X5b — the comparison tool in the second half of this
module — reads the committed baseline and records MARKED comparison outcomes for
one fresh comparison run (``ACTIVE-pass`` / ``ACTIVE-fail`` /
``COMPARISON-INELIGIBLE`` / ``DORMANT``), writing them under
``CHARTREUX_PERF_RUN_DIR`` only. A comparison is never an activation state:
activation is constructed from the committed baseline set alone (X7a), so a
machine context that differs in any required field refuses the comparison as
DORMANT ``machine_mismatch`` — never ACTIVE, never FAILED — and thin or high-load
inputs are marked outcomes rather than passes or failures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from tests.perf._runtime_report import (
    CLOSURE_ELIGIBLE_COMPLETE,
    CLOSURE_INSUFFICIENT_DATA,
    METRIC_VERSION,
    MODE_MEASURE,
    ROW_SPECS,
    ReportSchemaError,
    SaveTimings,
    build_run_record,
    evaluate_run_floor,
    validate_run_record,
)

__all__ = [
    "BASELINES_DIRNAME",
    "BASELINE_SCHEMA_ID",
    "COMPARISON_FILENAME",
    "COMPARISON_KIND",
    "COMPARISON_LABEL_PRECEDENCE",
    "COMPARISON_RUN_DIR_ENV",
    "COMPARISON_SCHEMA_ID",
    "DEFAULT_DIRECTION",
    "DIRECTIONS",
    "DIRECTION_HIGHER_IS_BETTER",
    "DIRECTION_LOWER_IS_BETTER",
    "INVALID_SAMPLE_CAP",
    "MACHINE_FIELDS",
    "MACHINE_MISMATCH_RULE",
    "MAX_ATTEMPTS",
    "OUTCOME_DORMANT",
    "OUTCOME_FAIL",
    "OUTCOME_INELIGIBLE",
    "OUTCOME_LABELS",
    "OUTCOME_PASS",
    "QUIET_LOADAVG_FACTOR",
    "QUIET_RULE",
    "REASON_DEGENERATE_THRESHOLD",
    "REASON_MACHINE_MISMATCH",
    "REASON_MISSING_DETERMINANTS",
    "REASON_ORDER",
    "REASON_RUN_INELIGIBILITY",
    "REASON_THRESHOLD_FAIL",
    "REASON_THRESHOLD_PASS",
    "SELECTED_RUNS",
    "SELECTION_RULE",
    "STATUS_ELIGIBLE",
    "STATUS_INELIGIBLE",
    "THRESHOLD_MAD_FACTOR",
    "THRESHOLD_RELATIVE_FACTOR",
    "THRESHOLD_RULE",
    "AttemptStatus",
    "BaselineError",
    "CellBaselineSpec",
    "Closure",
    "ComparisonError",
    "RunIdentity",
    "RunVerdict",
    "SharedRun",
    "build_baseline",
    "build_baseline_run",
    "build_comparison",
    "cell_baseline_spec",
    "cell_campaign_ids",
    "committed_baseline_path",
    "committed_baselines_dir",
    "comparison_machine_mismatch",
    "comparison_run_dir",
    "comparison_write_path",
    "evaluate_closure",
    "evaluate_comparison_row",
    "evaluate_shared_run",
    "load_committed_baseline",
    "machine_fields_missing",
    "machine_mismatch_fields",
    "mad",
    "median",
    "quiet_stamp",
    "quiet_window",
    "select_latest_eligible",
    "shared_run",
    "spec_from_baseline",
    "threshold_value",
    "validate_baseline",
    "write_comparison",
]

#: Committed-baseline schema id (the promotion artifact, distinct from the run
#: records' own ``chartreux.runtime_perf.v1`` metric version it validates).
BASELINE_SCHEMA_ID = "chartreux.runtime_baseline.v1"

#: Exactly five eligible runs make a campaign decision-ready.
SELECTED_RUNS = 5
#: The bounded campaign closure attempt cap for one cell's shared run set.
MAX_ATTEMPTS = 10
#: Quiet means 1-minute loadavg <= 0.5 * core_count at BOTH bracketing stamps.
QUIET_LOADAVG_FACTOR = 0.5
#: More than 10 % invalid samples in a run makes the run ineligible.
INVALID_SAMPLE_CAP = 0.10

#: The four required machine-context fields (``tests/perf/_metrics.py:95-108``).
MACHINE_FIELDS: tuple[str, ...] = (
    "cpu_model",
    "core_count",
    "python_version",
    "platform",
)
QUIET_RULE = "loadavg_1m <= 0.5 * core_count at both bracketing stamps"
SELECTION_RULE = "latest 5 eligible runs by run start time, run id tie-break"
MACHINE_MISMATCH_RULE = "inequality in any required machine-context field"

STATUS_ELIGIBLE = "eligible"
STATUS_INELIGIBLE = "ineligible"

#: A campaign that has neither closed nor hit its attempt cap.
OPEN_CAMPAIGN_REASON = "campaign_open"
#: The recorded reason of an insufficient_data closure (reason-order tier 2).
INSUFFICIENT_DATA_REASON = "insufficient_baseline_data"

# Run-level reason vocabulary (shared-set eligibility).
REASON_CELL_MISMATCH = "cell_mismatch"
REASON_METRIC_VERSION = "metric_version_mismatch"
REASON_FIXTURE_MISSING = "fixture_missing"
REASON_FIXTURE_MISMATCH = "fixture_mismatch"
REASON_DIMENSIONS_MISSING = "dimensions_missing"
REASON_DIMENSIONS_MISMATCH = "dimensions_mismatch"
REASON_RUN_MODE = "run_mode_not_measure"
REASON_CONTRACT_SCALE = "contract_scale_run"
REASON_OBSERVER_FLOW = "observer_mode_mismatch"
REASON_MACHINE_FIELD = "machine_field_missing"
REASON_QUIET_MISSING = "quiet_stamp_missing"
REASON_QUIET_INVALID = "quiet_stamp_invalid"
REASON_QUIET_NOT_QUIET = "quiet_stamp_not_quiet"
REASON_RUN_INCONSISTENT = "run_identity_inconsistent"

# Row-level reason vocabulary (one co-measured row of one run).
REASON_ROW_RECORD_MISSING = "row_record_missing"
REASON_ROW_RECORD_MISMATCH = "row_record_mismatch"
REASON_RECORD_INVALID = "record_schema_invalid"
REASON_CHUNKS_INCOMPLETE = "chunks_coverage_incomplete"
REASON_NO_SAMPLES = "no_gated_samples"
REASON_INVALID_SAMPLE_CAP = "invalid_sample_cap"
REASON_INVALID_SAMPLES_MALFORMED = "invalid_samples_malformed"


class BaselineError(ValueError):
    """A baseline capture or a promoted baseline violates the X5a contract."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"baseline rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _canonical(value: Any) -> str:
    """Return a stable text form, so two record blocks compare exactly."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _jsonable(value: Any) -> Any:
    """Return a JSON-safe deep copy of one record block."""
    return json.loads(_canonical(value))


def _ordered_unique(reasons: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(reasons))


def _parse_utc(text: Any) -> datetime:
    """Parse a run-start timestamp; a naive stamp is refused as unorderable."""
    if not _is_text(text):
        raise BaselineError("run_start_invalid", f"{text!r} is not a run start")
    candidate = text.strip()
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise BaselineError("run_start_invalid", f"{text!r}: {error}") from error
    if parsed.tzinfo is None:
        raise BaselineError(
            "run_start_naive",
            f"{text!r} carries no time zone, so run order would be ambiguous",
        )
    return parsed.astimezone(UTC)


# ---------------------------------------------------------------------------
# Cell identity and machine context
# ---------------------------------------------------------------------------


def cell_campaign_ids(cell: str) -> tuple[str, ...]:
    """Return the cell's co-measured campaign ids in registry order."""
    return tuple(
        campaign_id for campaign_id, spec in ROW_SPECS.items() if spec.cell == cell
    )


def gated_cells() -> tuple[str, ...]:
    """Return every gated cell, in registry order."""
    return tuple(dict.fromkeys(spec.cell for spec in ROW_SPECS.values()))


def machine_fields_missing(machine: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return the required machine fields that are absent or unusable."""
    if not isinstance(machine, Mapping):
        return MACHINE_FIELDS
    missing: list[str] = []
    for name in MACHINE_FIELDS:
        value = machine.get(name)
        if name == "core_count":
            if not _is_count(value) or value == 0:
                missing.append(name)
        elif not _is_text(value):
            missing.append(name)
    return tuple(missing)


def machine_mismatch_fields(
    baseline_machine: Mapping[str, Any] | None, run_machine: Mapping[str, Any] | None
) -> tuple[str, ...]:
    """Return the required machine fields whose values differ (tier-1 rule).

    Inequality in ANY field is a mismatch, exactly as Appendix B pins it: not
    core_count alone. An absent field on either side counts as a difference; the
    separate presence rule is :func:`machine_fields_missing`.
    """
    base: Mapping[str, Any] = (
        baseline_machine if isinstance(baseline_machine, Mapping) else {}
    )
    other: Mapping[str, Any] = run_machine if isinstance(run_machine, Mapping) else {}
    return tuple(
        field for field in MACHINE_FIELDS if base.get(field) != other.get(field)
    )


# ---------------------------------------------------------------------------
# Quiet stamps
# ---------------------------------------------------------------------------


def quiet_stamp(loadavg_1m: float | None, core_count: int) -> dict[str, Any]:
    """Return one 1-minute loadavg sample and its core-count-relative verdict."""
    bound = QUIET_LOADAVG_FACTOR * core_count
    if loadavg_1m is None:
        return {
            "loadavg_1m": None,
            "core_count": core_count,
            "bound": bound,
            "quiet": False,
        }
    return {
        "loadavg_1m": round(float(loadavg_1m), 3),
        "core_count": core_count,
        "bound": bound,
        "quiet": loadavg_1m <= bound,
    }


def quiet_window(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bracketing quiet record: BOTH stamps must certify quiet."""
    return {
        "before": dict(before),
        "after": dict(after),
        "quiet": before.get("quiet") is True and after.get("quiet") is True,
        "rule": QUIET_RULE,
    }


def _certified(stamp: Mapping[str, Any]) -> bool:
    loadavg = stamp.get("loadavg_1m")
    bound = stamp.get("bound")
    if isinstance(loadavg, bool) or not isinstance(loadavg, (int, float)):
        return False
    if isinstance(bound, bool) or not isinstance(bound, (int, float)):
        return False
    return float(loadavg) <= float(bound)


def _stamp_reasons(stamp: Mapping[str, Any], core_count: Any) -> tuple[str, ...]:
    """Return one quiet stamp's reasons; a recorded flag is never trusted."""
    reasons: list[str] = []
    stamp_core = stamp.get("core_count")
    bound = stamp.get("bound")
    flag = stamp.get("quiet")
    loadavg = stamp.get("loadavg_1m")
    if isinstance(stamp_core, bool) or not isinstance(stamp_core, int):
        return (REASON_QUIET_INVALID,)
    if (
        stamp_core <= 0
        or isinstance(bound, bool)
        or not isinstance(bound, (int, float))
    ):
        return (REASON_QUIET_INVALID,)
    if not isinstance(flag, bool):
        return (REASON_QUIET_INVALID,)
    if not math.isclose(float(bound), QUIET_LOADAVG_FACTOR * stamp_core):
        reasons.append(REASON_QUIET_INVALID)
    if _is_count(core_count) and core_count > 0 and stamp_core != core_count:
        reasons.append(REASON_QUIET_INVALID)
    if loadavg is None:
        if flag:
            reasons.append(REASON_QUIET_INVALID)
        reasons.append(REASON_QUIET_NOT_QUIET)
        return _ordered_unique(reasons)
    if isinstance(loadavg, bool) or not isinstance(loadavg, (int, float)):
        reasons.append(REASON_QUIET_INVALID)
        return _ordered_unique(reasons)
    certified = float(loadavg) <= float(bound)
    if flag != certified:
        reasons.append(REASON_QUIET_INVALID)
    if not certified:
        reasons.append(REASON_QUIET_NOT_QUIET)
    return _ordered_unique(reasons)


def _quiet_reasons(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate both bracketing quiet stamps of one run record."""
    window = record.get("quiet")
    if not isinstance(window, Mapping):
        return (REASON_QUIET_MISSING,)
    stamps: list[Mapping[str, Any]] = []
    for side in ("before", "after"):
        stamp = window.get(side)
        if not isinstance(stamp, Mapping):
            return (REASON_QUIET_MISSING,)
        stamps.append(stamp)
    machine = _record_machine(record)
    core_count = machine.get("core_count")
    reasons: list[str] = []
    for stamp in stamps:
        reasons.extend(_stamp_reasons(stamp, core_count))
    window_flag = window.get("quiet")
    both_quiet = all(_certified(stamp) for stamp in stamps)
    if not isinstance(window_flag, bool) or window_flag != both_quiet:
        reasons.append(REASON_QUIET_INVALID)
    return _ordered_unique(reasons)


# ---------------------------------------------------------------------------
# The committed cell identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CellBaselineSpec:
    """The committed identity every run of one cell's baseline must match."""

    cell: str
    fixture_hash: str
    dimensions: Mapping[str, Any]
    machine: Mapping[str, Any]
    workload_hash: str | None = None
    metric_version: str = METRIC_VERSION

    @property
    def campaign_ids(self) -> tuple[str, ...]:
        """Return the cell's co-measured campaign ids in registry order."""
        return cell_campaign_ids(self.cell)

    def as_record(self) -> dict[str, Any]:
        """Return the identity block a committed baseline carries."""
        return {
            "cell": self.cell,
            "fixture_hash": self.fixture_hash,
            "workload_hash": self.workload_hash,
            "dimensions": _jsonable(self.dimensions),
            "metric_version": self.metric_version,
        }


def cell_baseline_spec(
    cell: str,
    *,
    fixture_hash: str,
    dimensions: Mapping[str, Any],
    machine: Mapping[str, Any],
    workload_hash: str | None = None,
) -> CellBaselineSpec:
    """Build and validate one cell's committed baseline identity."""
    if cell not in gated_cells():
        raise BaselineError("unknown_cell", f"{cell!r} is not a gated cell")
    if not _is_text(fixture_hash):
        raise BaselineError(
            "fixture_missing", f"{fixture_hash!r} is not a fixture hash"
        )
    if not isinstance(dimensions, Mapping) or not dimensions:
        raise BaselineError("dimensions_missing", f"{cell}: no committed dimensions")
    missing = machine_fields_missing(machine)
    if missing:
        raise BaselineError(
            "machine_field_missing", f"{cell}: baseline machine lacks {list(missing)}"
        )
    if workload_hash is not None and not _is_text(workload_hash):
        raise BaselineError("dimensions_missing", f"{workload_hash!r}")
    return CellBaselineSpec(
        cell=cell,
        fixture_hash=fixture_hash,
        dimensions=dict(dimensions),
        machine=dict(machine),
        workload_hash=workload_hash,
    )


# ---------------------------------------------------------------------------
# One shared run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """The selection-relevant identity of one run: start time and run id."""

    run_id: str
    run_start_utc: str

    @property
    def start(self) -> datetime:
        """Return the run's start instant in UTC."""
        return _parse_utc(self.run_start_utc)


@dataclass(frozen=True, slots=True)
class SharedRun:
    """One run of a shared cell run set: its key, start, and row records."""

    run_key: str
    run_start_utc: str
    records: Mapping[str, Mapping[str, Any]]

    @property
    def identity(self) -> RunIdentity:
        """Return this run's selection identity."""
        return RunIdentity(self.run_key, self.run_start_utc)


def shared_run(
    records: Mapping[str, Mapping[str, Any]], *, run_key: str | None = None
) -> SharedRun:
    """Assemble one shared run from its per-row records.

    The records of one run share their cell and their run start time. Their run
    ids may differ per row (a producer may suffix the row id), so an explicit run
    key is required exactly then; otherwise the shared id is used.
    """
    if not records:
        raise BaselineError("empty_run", "a shared run needs at least one record")
    cells = {_canonical(record.get("cell")) for record in records.values()}
    if len(cells) != 1:
        raise BaselineError("mixed_cell_records", f"one run cannot span cells: {cells}")
    starts = {record.get("run_start_utc") for record in records.values()}
    if len(starts) != 1:
        raise BaselineError(
            "inconsistent_run_start",
            f"row records of one run disagree: {sorted(map(str, starts))}",
        )
    run_start = next(iter(starts))
    _parse_utc(run_start)
    if run_key is None:
        row_ids = {record.get("run_id") for record in records.values()}
        if len(row_ids) != 1:
            raise BaselineError(
                "run_key_required",
                f"row run ids differ, so the run needs a key: "
                f"{sorted(map(str, row_ids))}",
            )
        run_key = next(iter(row_ids))
    if not _is_text(run_key):
        raise BaselineError("run_key_required", f"{run_key!r} is not a run key")
    return SharedRun(
        run_key=str(run_key), run_start_utc=str(run_start), records=dict(records)
    )


def _record_machine(record: Mapping[str, Any]) -> Mapping[str, Any]:
    machine = record.get("machine")
    return machine if isinstance(machine, Mapping) else {}


def _dimension_block(record: Mapping[str, Any]) -> Any:
    """Return the run's declared dimensions (``dimensions`` or a scale block)."""
    block = record.get("dimensions")
    return block if block is not None else record.get("scale")


def _identity_reasons(
    record: Mapping[str, Any], spec: CellBaselineSpec
) -> tuple[str, ...]:
    """Check fixture, dimensions, cell and metric-version identity of a record."""
    reasons: list[str] = []
    if record.get("cell") != spec.cell:
        reasons.append(REASON_CELL_MISMATCH)
    if record.get("metric_version") != METRIC_VERSION:
        reasons.append(REASON_METRIC_VERSION)
    fixture = record.get("fixture_hash")
    if not _is_text(fixture):
        reasons.append(REASON_FIXTURE_MISSING)
    elif fixture != spec.fixture_hash:
        reasons.append(REASON_FIXTURE_MISMATCH)
    block = _dimension_block(record)
    if block is None:
        reasons.append(REASON_DIMENSIONS_MISSING)
    elif _canonical(block) != _canonical(spec.dimensions):
        reasons.append(REASON_DIMENSIONS_MISMATCH)
    if spec.workload_hash is not None:
        declared = record.get("workload_hash")
        if not _is_text(declared):
            reasons.append(REASON_DIMENSIONS_MISSING)
        elif declared != spec.workload_hash:
            reasons.append(REASON_DIMENSIONS_MISMATCH)
    return _ordered_unique(reasons)


def _instrumentation_reasons(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Check the run measured in observer mode without changing control flow."""
    reasons: list[str] = []
    if record.get("mode") != MODE_MEASURE:
        reasons.append(REASON_RUN_MODE)
    block = _dimension_block(record)
    if isinstance(block, Mapping) and block.get("mode") not in (None, MODE_MEASURE):
        reasons.append(REASON_CONTRACT_SCALE)
    observer = record.get("observer")
    if observer is not None and (
        not isinstance(observer, Mapping)
        or observer.get("control_flow_changed") is not False
    ):
        reasons.append(REASON_OBSERVER_FLOW)
    return _ordered_unique(reasons)


def _consistency_reasons(records: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Refuse a run whose row records disagree on the run's own identity."""
    reasons: list[str] = []
    for key in ("fixture_hash", "metric_version", "mode"):
        if len({_canonical(record.get(key)) for record in records.values()}) > 1:
            reasons.append(REASON_RUN_INCONSISTENT)
            break
    if len({_canonical(_record_machine(record)) for record in records.values()}) > 1:
        reasons.append(REASON_RUN_INCONSISTENT)
    if len({_canonical(_dimension_block(record)) for record in records.values()}) > 1:
        reasons.append(REASON_RUN_INCONSISTENT)
    return _ordered_unique(reasons)


def _invalid_sample_reasons(
    record: Mapping[str, Any], measured_ops: int
) -> tuple[str, ...]:
    """Apply the invalid-sample cap to one row record (subtype of ineligibility)."""
    cell = record.get("cell_eligibility")
    if not isinstance(cell, Mapping):
        return ()
    invalid = cell.get("invalid_samples")
    if invalid is None:
        return ()
    if not _is_count(invalid):
        return (REASON_INVALID_SAMPLES_MALFORMED,)
    total = measured_ops + invalid
    if total <= 0:
        return ()
    if invalid / total > INVALID_SAMPLE_CAP:
        return (REASON_INVALID_SAMPLE_CAP,)
    return ()


def _coverage_counts(coverage: Mapping[str, Any]) -> dict[str, int]:
    raw = coverage.get("unit_counts")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): int(value) for key, value in raw.items() if _is_count(value)}


def _row_reasons(campaign_id: str, record: Mapping[str, Any]) -> tuple[str, ...]:
    """Apply one co-measured row's requirements to its record of one run."""
    reasons: list[str] = []
    if record.get("campaign_id") != campaign_id:
        reasons.append(REASON_ROW_RECORD_MISMATCH)
    coverage = record.get("chunks_coverage")
    measured_ops = 0
    if isinstance(coverage, Mapping):
        if coverage.get("complete") is not True or coverage.get("missing"):
            reasons.append(REASON_CHUNKS_INCOMPLETE)
        raw_measured = coverage.get("measured_ops")
        raw_warmup = coverage.get("warmup_ops")
        if (
            isinstance(raw_measured, int)
            and not isinstance(raw_measured, bool)
            and raw_measured >= 0
            and isinstance(raw_warmup, int)
            and not isinstance(raw_warmup, bool)
            and raw_warmup >= 0
        ):
            measured_ops = raw_measured
            verdict = evaluate_run_floor(
                campaign_id,
                measured_ops=raw_measured,
                warmup_ops=raw_warmup,
                unit_counts=_coverage_counts(coverage),
            )
            reasons.extend(verdict.reasons)
        else:
            reasons.append(REASON_CHUNKS_INCOMPLETE)
    else:
        reasons.append(REASON_CHUNKS_INCOMPLETE)
    try:
        validate_run_record(record)
    except ReportSchemaError as error:
        reasons.append(REASON_RECORD_INVALID)
        reasons.append(error.reason)
    eligibility = record.get("eligibility")
    if isinstance(eligibility, Mapping) and eligibility.get("eligible") is not True:
        declared = eligibility.get("reasons")
        if isinstance(declared, list):
            reasons.extend(item for item in declared if isinstance(item, str))
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples:
        reasons.append(REASON_NO_SAMPLES)
    reasons.extend(_invalid_sample_reasons(record, measured_ops))
    return _ordered_unique(reasons)


@dataclass(frozen=True, slots=True)
class RunVerdict:
    """One run's shared-set eligibility: the conjunction of every row's rules."""

    eligible: bool
    reasons: tuple[str, ...]
    run_reasons: tuple[str, ...]
    row_reasons: Mapping[str, tuple[str, ...]]
    failing_row: str | None
    machine_mismatch_fields: tuple[str, ...]
    machine_fields_missing: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        """Return the attempt record's verdict fields."""
        return {
            "status": STATUS_ELIGIBLE if self.eligible else STATUS_INELIGIBLE,
            "reasons": list(self.reasons),
            "run_reasons": list(self.run_reasons),
            "row_reasons": {
                row: list(values) for row, values in self.row_reasons.items()
            },
            "failing_row": self.failing_row,
            "machine_mismatch_fields": list(self.machine_mismatch_fields),
            "machine_fields_missing": list(self.machine_fields_missing),
        }


def _machine_mismatch_union(
    spec: CellBaselineSpec, records: Mapping[str, Mapping[str, Any]]
) -> tuple[str, ...]:
    found: set[str] = set()
    for record in records.values():
        found.update(machine_mismatch_fields(spec.machine, _record_machine(record)))
    return tuple(field for field in MACHINE_FIELDS if field in found)


def _machine_missing_union(records: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    found: set[str] = set()
    for record in records.values():
        machine = record.get("machine")
        found.update(
            machine_fields_missing(machine if isinstance(machine, Mapping) else None)
        )
    return tuple(field for field in MACHINE_FIELDS if field in found)


def evaluate_shared_run(spec: CellBaselineSpec, run: SharedRun) -> RunVerdict:
    """Apply the shared-set eligibility conjunction to one run of the cell."""
    run_reasons: list[str] = []
    for record in run.records.values():
        run_reasons.extend(_identity_reasons(record, spec))
        run_reasons.extend(_instrumentation_reasons(record))
        run_reasons.extend(_quiet_reasons(record))
    missing_fields = _machine_missing_union(run.records)
    if missing_fields:
        run_reasons.append(REASON_MACHINE_FIELD)
    run_reasons.extend(_consistency_reasons(run.records))

    row_reasons: dict[str, tuple[str, ...]] = {}
    failing_row: str | None = None
    for campaign_id in spec.campaign_ids:
        record = run.records.get(campaign_id)
        reasons = (
            (REASON_ROW_RECORD_MISSING,)
            if record is None
            else _row_reasons(campaign_id, record)
        )
        if reasons:
            row_reasons[campaign_id] = reasons
            if failing_row is None:
                failing_row = campaign_id

    ordered_run = _ordered_unique(run_reasons)
    flat = list(ordered_run)
    for campaign_id in spec.campaign_ids:
        flat.extend(row_reasons.get(campaign_id, ()))
    reasons = _ordered_unique(flat)
    return RunVerdict(
        eligible=not reasons,
        reasons=reasons,
        run_reasons=ordered_run,
        row_reasons=row_reasons,
        failing_row=failing_row,
        machine_mismatch_fields=_machine_mismatch_union(spec, run.records),
        machine_fields_missing=missing_fields,
    )


# ---------------------------------------------------------------------------
# Selection and bounded campaign closure
# ---------------------------------------------------------------------------


def select_latest_eligible(identities: Sequence[RunIdentity]) -> tuple[str, ...]:
    """Return the latest five run ids by run start then run id.

    Only identities reach this function — no measured value is available to it,
    so selection cannot be influenced by one.
    """
    if len(identities) < SELECTED_RUNS:
        return ()
    ordered = sorted(identities, key=lambda identity: (identity.start, identity.run_id))
    return tuple(identity.run_id for identity in ordered[-SELECTED_RUNS:])


@dataclass(frozen=True, slots=True)
class AttemptStatus:
    """The selection-relevant projection of one attempt: identity and status."""

    run_id: str
    run_start_utc: str
    eligible: bool

    @property
    def identity(self) -> RunIdentity:
        """Return this attempt's selection identity."""
        return RunIdentity(self.run_id, self.run_start_utc)


def _ordered_statuses(statuses: Sequence[AttemptStatus]) -> list[AttemptStatus]:
    return sorted(statuses, key=lambda status: (status.identity.start, status.run_id))


@dataclass(frozen=True, slots=True)
class Closure:
    """One shared cell run set's bounded campaign closure."""

    status: str | None
    selected_run_ids: tuple[str, ...]
    eligible_run_ids: tuple[str, ...]
    eligible_run_count: int
    reason: str | None
    note: str

    def as_record(self) -> dict[str, Any]:
        """Return the closure block a committed baseline carries."""
        return {
            "status": self.status,
            "selected_run_ids": list(self.selected_run_ids),
            "selected_run_count": len(self.selected_run_ids),
            "eligible_run_ids": list(self.eligible_run_ids),
            "eligible_run_count": self.eligible_run_count,
            "reason": self.reason,
            "note": self.note,
        }


def evaluate_closure(statuses: Sequence[AttemptStatus]) -> Closure:
    """Close (or leave open) one cell's campaign within the attempt cap."""
    if len(statuses) > MAX_ATTEMPTS:
        raise BaselineError(
            "attempt_cap_exceeded",
            f"{len(statuses)} attempted runs for one cell, cap is {MAX_ATTEMPTS}",
        )
    ordered = _ordered_statuses(statuses)
    eligible = [status for status in ordered if status.eligible]
    eligible_ids = tuple(status.run_id for status in eligible)
    count = len(eligible)
    if count >= SELECTED_RUNS:
        return Closure(
            status=CLOSURE_ELIGIBLE_COMPLETE,
            selected_run_ids=select_latest_eligible([
                status.identity for status in eligible
            ]),
            eligible_run_ids=eligible_ids,
            eligible_run_count=count,
            reason=None,
            note=(
                f"{count} eligible runs collected; the latest {SELECTED_RUNS}"
                " are selected"
            ),
        )
    if len(statuses) >= MAX_ATTEMPTS:
        return Closure(
            status=CLOSURE_INSUFFICIENT_DATA,
            selected_run_ids=(),
            eligible_run_ids=eligible_ids,
            eligible_run_count=count,
            reason=INSUFFICIENT_DATA_REASON,
            note=(f"{len(statuses)} attempted runs without {SELECTED_RUNS} eligible"),
        )
    return Closure(
        status=None,
        selected_run_ids=(),
        eligible_run_ids=eligible_ids,
        eligible_run_count=count,
        reason=OPEN_CAMPAIGN_REASON,
        note=f"campaign open: {len(statuses)} attempted, {count} eligible",
    )


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def build_baseline_run(
    campaign_id: str,
    *,
    run_id: str,
    run_start_utc: str,
    fixture_hash: str,
    machine: Mapping[str, Any],
    quiet: Mapping[str, Any],
    dimensions: Mapping[str, Any],
    measured_ops: int,
    warmup_ops: int,
    samples: Sequence[float] | None = None,
    unit_counts: Mapping[str, int] | None = None,
    slopes_signed: Sequence[float] | None = None,
    save_timings: SaveTimings | None = None,
    workload_hash: str | None = None,
    observer: Mapping[str, Any] | None = None,
    closure_status: str | None = None,
    closure_note: str | None = None,
    mode: str = MODE_MEASURE,
) -> dict[str, Any]:
    """Build one baseline run record, identity blocks attached and validated.

    The identity blocks (fixture hash, machine context, quiet window, dimensions)
    are required here rather than optional: a record that cannot prove its run is
    the committed cell's run is never a baseline record.
    """
    record = build_run_record(
        campaign_id,
        run_id=run_id,
        run_start_utc=run_start_utc,
        measured_ops=measured_ops,
        warmup_ops=warmup_ops,
        samples=samples,
        unit_counts=unit_counts,
        mode=mode,
        slopes_signed=slopes_signed,
        save_timings=save_timings,
        closure_status=closure_status,
        closure_note=(
            "campaign open: this run is one attempt of the cell's campaign"
            if closure_note is None and closure_status is None
            else closure_note
        ),
        fixture_hash=fixture_hash,
        machine=machine,
    )
    record["quiet"] = dict(quiet)
    record["dimensions"] = _jsonable(dimensions)
    if workload_hash is not None:
        record["workload_hash"] = workload_hash
    if observer is not None:
        record["observer"] = dict(observer)
    validate_run_record(record)
    return record


def _run_p95(record: Mapping[str, Any] | None) -> float | None:
    if record is None:
        return None
    summary = record.get("summary")
    if not isinstance(summary, Mapping):
        return None
    value = summary.get("p95")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _attempt_record(run: SharedRun, verdict: RunVerdict) -> dict[str, Any]:
    return {
        "run_id": run.run_key,
        "run_start_utc": run.run_start_utc,
        **verdict.as_record(),
    }


def _attempt_digest(attempts: Sequence[Mapping[str, Any]]) -> str:
    payload = _canonical([dict(attempt) for attempt in attempts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_declared_closure(runs: Sequence[SharedRun], closure: Closure) -> None:
    """Refuse a capture whose records already claim a different closure."""
    for run in runs:
        for record in run.records.values():
            campaign = record.get("campaign")
            if not isinstance(campaign, Mapping):
                continue
            declared = campaign.get("closure_status")
            if declared is not None and declared != closure.status:
                raise BaselineError(
                    "closure_status_mismatch",
                    f"{run.run_key}/{record.get('campaign_id')}: {declared!r}"
                    f" != {closure.status!r}",
                )


def _campaign_blocks(
    spec: CellBaselineSpec, runs: Sequence[SharedRun], closure: Closure
) -> dict[str, Any]:
    """Return each campaign id's selected runs and their collected run-p95s."""
    by_key = {run.run_key: run for run in runs}
    if closure.status == CLOSURE_ELIGIBLE_COMPLETE:
        collected = closure.selected_run_ids
        null_reason: str | None = None
    elif closure.status == CLOSURE_INSUFFICIENT_DATA:
        collected = closure.eligible_run_ids
        null_reason = INSUFFICIENT_DATA_REASON
    else:
        collected = closure.eligible_run_ids
        null_reason = OPEN_CAMPAIGN_REASON
    blocks: dict[str, Any] = {}
    for campaign_id in spec.campaign_ids:
        row = ROW_SPECS[campaign_id]
        p95s = [
            _run_p95(by_key[run_id].records.get(campaign_id)) for run_id in collected
        ]
        blocks[campaign_id] = {
            "campaign_id": campaign_id,
            "metric": row.metric,
            "unit": row.unit,
            "cell": row.cell,
            "run_ids": list(collected),
            "run_p95s": p95s,
            "null_reason": null_reason,
        }
    return blocks


def build_baseline(spec: CellBaselineSpec, runs: Sequence[SharedRun]) -> dict[str, Any]:
    """Validate a cell's shared run set and assemble its baseline document.

    Every supplied run becomes an attempt, in the order supplied; no run is
    dropped for being ineligible and no run is preferred for measuring well.
    """
    if not runs:
        raise BaselineError("empty_run_set", f"{spec.cell}: no runs supplied")
    keys = [run.run_key for run in runs]
    if len(set(keys)) != len(keys):
        raise BaselineError("duplicate_attempt", f"{spec.cell}: {sorted(keys)}")
    attempts: list[dict[str, Any]] = []
    statuses: list[AttemptStatus] = []
    for run in runs:
        verdict = evaluate_shared_run(spec, run)
        attempts.append(_attempt_record(run, verdict))
        statuses.append(
            AttemptStatus(
                run_id=run.run_key,
                run_start_utc=run.run_start_utc,
                eligible=verdict.eligible,
            )
        )
    closure = evaluate_closure(statuses)
    _check_declared_closure(runs, closure)
    document: dict[str, Any] = {
        "schema": BASELINE_SCHEMA_ID,
        "cell": spec.cell,
        "cell_identity": spec.as_record(),
        "machine": _jsonable(spec.machine),
        "campaign_ids": list(spec.campaign_ids),
        "attempts": attempts,
        "attempt_count": len(attempts),
        "eligible_run_ids": list(closure.eligible_run_ids),
        "closure": closure.as_record(),
        "campaigns": _campaign_blocks(spec, runs, closure),
        "attempt_digest": _attempt_digest(attempts),
    }
    validate_baseline(document, spec, closed_required=False)
    return document


# ---------------------------------------------------------------------------
# The promotion gate
# ---------------------------------------------------------------------------


def _document_attempts(
    document: Mapping[str, Any], spec: CellBaselineSpec
) -> list[AttemptStatus]:
    """Read the attempt list back into statuses, refusing an unusable one."""
    attempts = document.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise BaselineError("attempts_missing", "a baseline preserves every attempt")
    statuses: list[AttemptStatus] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise BaselineError("attempt_invalid", f"{attempt!r} is not an attempt")
        run_id = attempt.get("run_id")
        run_start = attempt.get("run_start_utc")
        status = attempt.get("status")
        if not _is_text(run_id):
            raise BaselineError("attempt_invalid", f"run id {run_id!r}")
        _parse_utc(run_start)
        if status not in (STATUS_ELIGIBLE, STATUS_INELIGIBLE):
            raise BaselineError("attempt_invalid", f"{run_id}: status {status!r}")
        reasons = attempt.get("reasons")
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) for reason in reasons
        ):
            raise BaselineError("attempt_invalid", f"{run_id}: reasons {reasons!r}")
        failing_row = attempt.get("failing_row")
        if status == STATUS_ELIGIBLE:
            if reasons or failing_row is not None:
                raise BaselineError(
                    "attempt_status_conflict",
                    f"{run_id}: an eligible run carries no reason",
                )
        elif not reasons:
            raise BaselineError(
                "attempt_reason_missing", f"{run_id}: an ineligible run records why"
            )
        if failing_row is not None and failing_row not in spec.campaign_ids:
            raise BaselineError("attempt_invalid", f"{run_id}: row {failing_row!r}")
        statuses.append(
            AttemptStatus(
                run_id=str(run_id),
                run_start_utc=str(run_start),
                eligible=status == STATUS_ELIGIBLE,
            )
        )
    ids = [status.run_id for status in statuses]
    if len(set(ids)) != len(ids):
        raise BaselineError("duplicate_attempt", f"{sorted(ids)}")
    return statuses


def _check_campaign_blocks(
    document: Mapping[str, Any], spec: CellBaselineSpec, closure: Closure
) -> None:
    """Check every campaign block against the recomputed closure."""
    blocks = document.get("campaigns")
    if not isinstance(blocks, Mapping):
        raise BaselineError("campaigns_missing", f"{spec.cell}: no campaign blocks")
    if list(blocks) != list(spec.campaign_ids):
        raise BaselineError(
            "cell_incomplete",
            f"{spec.cell}: blocks {list(blocks)} != {list(spec.campaign_ids)}",
        )
    expected_collected = (
        closure.selected_run_ids
        if closure.status == CLOSURE_ELIGIBLE_COMPLETE
        else closure.eligible_run_ids
    )
    expected_reason = (
        None
        if closure.status == CLOSURE_ELIGIBLE_COMPLETE
        else INSUFFICIENT_DATA_REASON
        if closure.status == CLOSURE_INSUFFICIENT_DATA
        else OPEN_CAMPAIGN_REASON
    )
    for campaign_id in spec.campaign_ids:
        block = blocks[campaign_id]
        row = ROW_SPECS[campaign_id]
        if not isinstance(block, Mapping):
            raise BaselineError("campaign_block_invalid", campaign_id)
        identity = (
            block.get("campaign_id"),
            block.get("metric"),
            block.get("unit"),
            block.get("cell"),
        )
        if identity != (campaign_id, row.metric, row.unit, row.cell):
            raise BaselineError(
                "campaign_block_invalid", f"{campaign_id}: identity {identity!r}"
            )
        run_ids = block.get("run_ids")
        p95s = block.get("run_p95s")
        if not isinstance(run_ids, list) or not all(
            _is_text(run_id) for run_id in run_ids
        ):
            raise BaselineError("campaign_block_invalid", f"{campaign_id}: run_ids")
        if tuple(run_ids) != expected_collected:
            raise BaselineError(
                "selection_mismatch",
                f"{campaign_id}: {run_ids} != {list(expected_collected)}",
            )
        if not isinstance(p95s, list) or len(p95s) != len(run_ids):
            raise BaselineError("campaign_block_invalid", f"{campaign_id}: run_p95s")
        for value in p95s:
            if not _is_number(value):
                raise BaselineError("run_p95_missing", f"{campaign_id}: {value!r}")
        if block.get("null_reason") != expected_reason:
            raise BaselineError(
                "campaign_block_invalid",
                f"{campaign_id}: null_reason {block.get('null_reason')!r}",
            )
        if closure.status == CLOSURE_ELIGIBLE_COMPLETE:
            if len(run_ids) != SELECTED_RUNS:
                raise BaselineError(
                    "selection_incomplete",
                    f"{campaign_id}: {len(run_ids)} selected runs",
                )
        elif len(run_ids) >= SELECTED_RUNS:
            raise BaselineError(
                "selection_incomplete",
                f"{campaign_id}: thin data cannot select {len(run_ids)} runs",
            )


def validate_baseline(
    document: Mapping[str, Any], spec: CellBaselineSpec, *, closed_required: bool = True
) -> None:
    """Re-check a promoted baseline: identity, closure, selection, preservation.

    The closure and the selection are recomputed from the attempt list, so a
    promotable artifact cannot drop an attempt or substitute a favorable five;
    the attempt digest refuses any edit of the list after capture. With
    ``closed_required`` (the promotion default) an open campaign is refused.
    """
    if not isinstance(document, Mapping):
        raise BaselineError("baseline_not_mapping", "a baseline is a mapping")
    if document.get("schema") != BASELINE_SCHEMA_ID:
        raise BaselineError("schema_mismatch", f"{document.get('schema')!r}")
    if document.get("cell") != spec.cell:
        raise BaselineError(
            "cell_mismatch", f"{document.get('cell')!r}, expected {spec.cell!r}"
        )
    recorded_ids = document.get("campaign_ids")
    if not isinstance(recorded_ids, list) or recorded_ids != list(spec.campaign_ids):
        raise BaselineError(
            "cell_incomplete",
            f"campaign ids {recorded_ids!r} != {list(spec.campaign_ids)}",
        )
    identity = document.get("cell_identity")
    if not isinstance(identity, Mapping) or dict(identity) != spec.as_record():
        raise BaselineError("identity_mismatch", f"{identity!r}")
    machine = document.get("machine")
    missing = machine_fields_missing(machine if isinstance(machine, Mapping) else None)
    if missing:
        raise BaselineError(
            "machine_field_missing", f"baseline machine lacks {list(missing)}"
        )
    statuses = _document_attempts(document, spec)
    attempts = document["attempts"]
    if document.get("attempt_count") != len(attempts):
        raise BaselineError(
            "attempt_count_mismatch",
            f"{document.get('attempt_count')!r} for {len(attempts)} attempts",
        )
    if document.get("attempt_digest") != _attempt_digest(attempts):
        raise BaselineError(
            "attempt_digest_mismatch", "the attempt list was edited after capture"
        )
    closure = evaluate_closure(statuses)
    recorded_eligible = document.get("eligible_run_ids")
    if recorded_eligible != list(closure.eligible_run_ids):
        raise BaselineError(
            "eligible_run_ids_mismatch",
            f"{recorded_eligible!r} != {list(closure.eligible_run_ids)!r}",
        )
    recorded_closure = document.get("closure")
    if (
        not isinstance(recorded_closure, Mapping)
        or dict(recorded_closure) != closure.as_record()
    ):
        raise BaselineError(
            "closure_mismatch", f"{recorded_closure!r} != {closure.as_record()!r}"
        )
    if closed_required and closure.status is None:
        raise BaselineError("campaign_not_closed", closure.note)
    _check_campaign_blocks(document, spec, closure)


# ---------------------------------------------------------------------------
# The comparison tool: marked outcomes over a committed baseline (X5b)
# ---------------------------------------------------------------------------

#: Comparison record schema: a RUN_DIR-only outcome artifact, never an
#: activation input (activation is constructed from the committed baseline set).
COMPARISON_SCHEMA_ID = "chartreux.runtime_comparison.v1"
COMPARISON_KIND = "comparison_outcome"

OUTCOME_PASS = "ACTIVE-pass"
OUTCOME_FAIL = "ACTIVE-fail"
OUTCOME_INELIGIBLE = "COMPARISON-INELIGIBLE"
OUTCOME_DORMANT = "DORMANT"
#: The marked labels a comparison record may carry: never a bare pass or fail.
OUTCOME_LABELS: tuple[str, ...] = (
    OUTCOME_DORMANT,
    OUTCOME_INELIGIBLE,
    OUTCOME_PASS,
    OUTCOME_FAIL,
)
#: Total order: a dormant row is labelled before any run-level evaluation, and
#: only an eligible row with an eligible comparison run reaches a threshold label.
COMPARISON_LABEL_PRECEDENCE: tuple[str, ...] = (
    OUTCOME_DORMANT,
    OUTCOME_INELIGIBLE,
    OUTCOME_PASS,
    OUTCOME_FAIL,
)

#: Dormancy reasons a comparison row records (reason-order tiers 1-4).
REASON_MACHINE_MISMATCH = "machine_mismatch"
REASON_MISSING_DETERMINANTS = "missing_determinants"
REASON_DEGENERATE_THRESHOLD = "degenerate_threshold"
#: Tier 5: a run-record reason; at comparison it labels COMPARISON-INELIGIBLE.
REASON_RUN_INELIGIBILITY = "run_ineligibility"
#: Tier 6: the reasons of the two threshold labels.
REASON_THRESHOLD_PASS = "threshold_pass"
REASON_THRESHOLD_FAIL = "threshold_fail"

#: The order a record's single ``reason`` field is taken from (Appendix B).
REASON_ORDER: tuple[str, ...] = (
    REASON_MACHINE_MISMATCH,
    INSUFFICIENT_DATA_REASON,
    REASON_MISSING_DETERMINANTS,
    REASON_DEGENERATE_THRESHOLD,
    REASON_RUN_INELIGIBILITY,
    REASON_THRESHOLD_FAIL,
    REASON_THRESHOLD_PASS,
)

DIRECTION_LOWER_IS_BETTER = "lower_is_better"
DIRECTION_HIGHER_IS_BETTER = "higher_is_better"
DIRECTIONS: tuple[str, ...] = (DIRECTION_LOWER_IS_BETTER, DIRECTION_HIGHER_IS_BETTER)
#: No gated row declares a higher-is-better margin, so lower-is-better is default.
DEFAULT_DIRECTION = DIRECTION_LOWER_IS_BETTER

#: T = M + max(6*D, 0.20*M); a measured value <= T passes, equality included.
THRESHOLD_MAD_FACTOR = 6.0
THRESHOLD_RELATIVE_FACTOR = 0.20
THRESHOLD_RULE = "T = M + max(6*D, 0.20*M); measured <= T passes, equality included"

#: Comparisons are written under this run directory and nowhere else.
COMPARISON_RUN_DIR_ENV = "CHARTREUX_PERF_RUN_DIR"
COMPARISON_FILENAME = "comparison.json"
#: The committed baseline directory (parent/release-owner promotion only).
BASELINES_DIRNAME = "baselines"

_PERF_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PERF_DIR.parents[1]


class ComparisonError(ValueError):
    """The comparison tool refuses an input or a destination it cannot trust."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"comparison rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


def _descendant(path: Path, root: Path) -> bool:
    """Return True when ``path`` is ``root`` or below it."""
    return path == root or root in path.parents


# ---------------------------------------------------------------------------
# Committed baselines: read-only inputs
# ---------------------------------------------------------------------------


def committed_baselines_dir() -> Path:
    """Return the promoted-baseline directory; reading it is the only access."""
    return _PERF_DIR / BASELINES_DIRNAME


def committed_baseline_path(
    cell: str, *, baselines_dir: Path | str | None = None
) -> Path:
    """Return one cell's committed-baseline path, refusing an unknown cell."""
    if cell not in gated_cells():
        raise ComparisonError("unknown_cell", f"{cell!r} is not a gated cell")
    root = (
        Path(baselines_dir) if baselines_dir is not None else committed_baselines_dir()
    )
    return root / f"{cell}.json"


def load_committed_baseline(
    cell: str, *, baselines_dir: Path | str | None = None
) -> dict[str, Any] | None:
    """Read one cell's committed baseline, or None when none is promoted yet.

    The reader never creates, repairs or writes anything: an absent directory
    means "not promoted", and a file that exists but is not a baseline document
    is a hard refusal rather than a silently skipped input.
    """
    path = committed_baseline_path(cell, baselines_dir=baselines_dir)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComparisonError("baseline_unreadable", f"{path}: {error}") from error
    if not isinstance(document, dict):
        raise ComparisonError("baseline_unreadable", f"{path}: not an object")
    if document.get("schema") != BASELINE_SCHEMA_ID:
        raise ComparisonError(
            "baseline_schema_invalid", f"{path}: {document.get('schema')!r}"
        )
    return document


# ---------------------------------------------------------------------------
# Statistics and the threshold
# ---------------------------------------------------------------------------


def median(values: Sequence[float]) -> float:
    """Return the median: the middle value, or the mean of the middle pair."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ComparisonError("median_empty", "a median needs at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def mad(values: Sequence[float]) -> float:
    """Return the median absolute deviation of ``values`` from their median."""
    centre = median(values)
    return median([abs(float(value) - centre) for value in values])


def threshold_value(median_value: float, mad_value: float) -> float:
    """Return T = M + max(6*D, 0.20*M) for one decision-ready baseline."""
    return median_value + max(
        THRESHOLD_MAD_FACTOR * mad_value, THRESHOLD_RELATIVE_FACTOR * median_value
    )


# ---------------------------------------------------------------------------
# Reading the committed cell state
# ---------------------------------------------------------------------------


def _baseline_cell(document: Mapping[str, Any]) -> str:
    cell = document.get("cell")
    if not isinstance(cell, str) or cell not in gated_cells():
        raise ComparisonError(
            "baseline_cell_invalid", f"{cell!r} is not a committed gated cell"
        )
    return cell


def _closure_status(document: Mapping[str, Any]) -> str | None:
    closure = document.get("closure")
    if not isinstance(closure, Mapping):
        return None
    status = closure.get("status")
    return status if isinstance(status, str) else None


def _selected_runs(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the selected run ids with their start timestamps, in order."""
    starts: dict[str, Any] = {}
    attempts = document.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, Mapping) and _is_text(attempt.get("run_id")):
                starts[str(attempt["run_id"])] = attempt.get("run_start_utc")
    closure = document.get("closure")
    ids = closure.get("selected_run_ids") if isinstance(closure, Mapping) else None
    if not isinstance(ids, list):
        return []
    return [
        {"run_id": str(run_id), "run_start_utc": starts.get(str(run_id))}
        for run_id in ids
        if _is_text(run_id)
    ]


def _campaign_block(document: Mapping[str, Any], campaign_id: str) -> Any:
    campaigns = document.get("campaigns")
    if not isinstance(campaigns, Mapping):
        return None
    block = campaigns.get(campaign_id)
    return block if isinstance(block, Mapping) else None


def _collected_p95s(block: Any) -> list[float] | None:
    """Return the block's collected run-p95s, or None when they are unusable."""
    if not isinstance(block, Mapping):
        return None
    values = block.get("run_p95s")
    if not isinstance(values, list) or not all(_is_number(item) for item in values):
        return None
    return [float(item) for item in values]


def spec_from_baseline(document: Mapping[str, Any]) -> CellBaselineSpec:
    """Rebuild the committed cell identity from a committed baseline.

    A comparison run is then judged by exactly the rules that selected the
    baseline: the same identity, the same floors, the same quiet stamps.
    """
    identity = document.get("cell_identity")
    if not isinstance(identity, Mapping):
        raise ComparisonError(
            "identity_missing", "the committed baseline carries no cell identity"
        )
    cell = _baseline_cell(document)
    if identity.get("cell") != cell:
        raise ComparisonError(
            "identity_mismatch", f"{identity.get('cell')!r} != {cell!r}"
        )
    fixture_hash = identity.get("fixture_hash")
    dimensions = identity.get("dimensions")
    machine = document.get("machine")
    if not isinstance(fixture_hash, str) or not fixture_hash.strip():
        raise ComparisonError("identity_missing", f"{cell}: no committed fixture hash")
    if not isinstance(dimensions, Mapping):
        raise ComparisonError("identity_missing", f"{cell}: no committed dimensions")
    if not isinstance(machine, Mapping):
        raise ComparisonError("identity_missing", f"{cell}: no committed machine")
    declared = identity.get("workload_hash")
    if declared is not None and not isinstance(declared, str):
        raise ComparisonError("identity_missing", f"{cell}: unusable workload hash")
    return cell_baseline_spec(
        cell,
        fixture_hash=fixture_hash,
        dimensions=dict(dimensions),
        machine=machine,
        workload_hash=declared if isinstance(declared, str) else None,
    )


def comparison_machine_mismatch(
    baseline_machine: Mapping[str, Any] | None, run: SharedRun
) -> tuple[str, ...]:
    """Return the required machine fields a comparison run actually differs in.

    Inequality in ANY required field is a mismatch. A field the run cannot
    report is not evidence that the machine differs: that is a run-record defect
    (``machine_field_missing``, a tier-5 run reason) and is judged as one.
    """
    found: set[str] = set()
    for record in run.records.values():
        machine = _record_machine(record)
        usable = tuple(
            name
            for name in MACHINE_FIELDS
            if name not in machine_fields_missing(machine)
        )
        present = {name: machine[name] for name in usable}
        found.update(
            field
            for field in machine_mismatch_fields(baseline_machine, present)
            if field in present
        )
    return tuple(field for field in MACHINE_FIELDS if field in found)


# ---------------------------------------------------------------------------
# One row's marked outcome
# ---------------------------------------------------------------------------


def _marked_dormant(record: dict[str, Any], reason: str) -> dict[str, Any]:
    """Label a row DORMANT with its dormancy reason and the null reasons."""
    record["outcome"] = OUTCOME_DORMANT
    record["reason"] = reason
    nulls: dict[str, str] = record["null_reasons"]
    nulls["measured_value"] = reason
    for name in ("median", "mad", "threshold"):
        if record[name] is None:
            nulls.setdefault(name, reason)
    run_block: dict[str, Any] = record["comparison_run"]
    if run_block["eligible"] is None:
        run_block["null_reason"] = "row_dormant"
    return record


def evaluate_comparison_row(
    document: Mapping[str, Any],
    campaign_id: str,
    run: SharedRun,
    *,
    direction: str | None = None,
) -> dict[str, Any]:
    """Evaluate one fresh comparison run against one row of a committed baseline.

    The label is decided ROW FIRST (DORMANT > COMPARISON-INELIGIBLE > threshold
    labels): the row's dormancy, read from the committed baseline and from the
    run's machine context, is recorded before any run-level evaluation, and a
    dormant row performs none. The threshold comparison runs only for an eligible
    row and an eligible comparison run, and equality passes.
    """
    row = ROW_SPECS.get(campaign_id)
    if row is None:
        raise ComparisonError(
            "unknown_campaign_id", f"{campaign_id!r} is not a gated row"
        )
    declared_direction = DEFAULT_DIRECTION if direction is None else direction
    if declared_direction not in DIRECTIONS:
        raise ComparisonError(
            "direction_invalid", f"{direction!r} is not a comparison direction"
        )
    cell = _baseline_cell(document)
    for row_record in run.records.values():
        if row_record.get("cell") != cell:
            raise ComparisonError(
                "comparison_run_cell_mismatch",
                f"{row_record.get('cell')!r} != {cell!r}",
            )

    block = _campaign_block(document, campaign_id)
    collected = _collected_p95s(block)
    selected = _selected_runs(document)
    closure_status = _closure_status(document)
    decision_ready = (
        closure_status == CLOSURE_ELIGIBLE_COMPLETE and len(selected) == SELECTED_RUNS
    )
    row_inputs_present = (
        block is not None
        and collected is not None
        and len(collected) == SELECTED_RUNS
        and block.get("metric") == row.metric
        and block.get("unit") == row.unit
    )

    record: dict[str, Any] = {
        "kind": COMPARISON_KIND,
        "campaign_id": campaign_id,
        "cell": cell,
        "metric": row.metric,
        "unit": row.unit,
        "direction": declared_direction,
        "closure_status": closure_status,
        "selected_runs": selected,
        "collected_run_p95s": [] if collected is None else list(collected),
        "median": None,
        "mad": None,
        "threshold": None,
        "measured_value": None,
        "outcome": None,
        "reason": None,
        "null_reasons": {},
        "machine_mismatch_fields": [],
        "comparison_run": {
            "run_id": run.run_key,
            "run_start_utc": run.run_start_utc,
            "eligible": None,
            "reasons": [],
            "failing_row": None,
            "null_reason": None,
        },
    }
    nulls: dict[str, str] = record["null_reasons"]
    if collected is None:
        nulls["collected_run_p95s"] = REASON_MISSING_DETERMINANTS
    if record["collected_run_p95s"]:
        centre = median(record["collected_run_p95s"])
        deviation = mad(record["collected_run_p95s"])
        if math.isfinite(centre):
            record["median"] = centre
        else:
            nulls["median"] = REASON_DEGENERATE_THRESHOLD
        if math.isfinite(deviation):
            record["mad"] = deviation
        else:
            nulls["mad"] = REASON_DEGENERATE_THRESHOLD

    # Threshold population is authorized only for a decision-ready campaign
    # whose committed row block describes this row: thin data never gets a T.
    if not decision_ready:
        nulls["threshold"] = INSUFFICIENT_DATA_REASON
    elif not row_inputs_present:
        nulls["threshold"] = REASON_MISSING_DETERMINANTS
    elif record["median"] is None or record["mad"] is None:
        nulls["threshold"] = REASON_DEGENERATE_THRESHOLD
    else:
        candidate = threshold_value(record["median"], record["mad"])
        if math.isfinite(candidate) and candidate > 0:
            record["threshold"] = candidate
        else:
            nulls["threshold"] = REASON_DEGENERATE_THRESHOLD

    # Tier 1: the machine context. Inequality in ANY required field refuses the
    # comparison as DORMANT machine_mismatch — never ACTIVE, never FAILED.
    mismatch = comparison_machine_mismatch(document.get("machine"), run)
    if mismatch:
        record["machine_mismatch_fields"] = list(mismatch)
        return _marked_dormant(record, REASON_MACHINE_MISMATCH)

    # Tier 2: the committed campaign is not decision-ready.
    if not decision_ready:
        return _marked_dormant(record, INSUFFICIENT_DATA_REASON)

    # Tier 3: the row's committed determinants are absent or describe another row.
    if not row_inputs_present:
        return _marked_dormant(record, REASON_MISSING_DETERMINANTS)

    # Tier 4: a nonpositive or non-finite threshold is degenerate, never padded.
    threshold = record["threshold"]
    if threshold is None:
        return _marked_dormant(record, REASON_DEGENERATE_THRESHOLD)

    # Tier 5: the comparison run's own eligibility, by the selection rules.
    verdict = evaluate_shared_run(spec_from_baseline(document), run)
    run_block = record["comparison_run"]
    run_block["eligible"] = verdict.eligible
    run_block["reasons"] = list(verdict.reasons)
    run_block["failing_row"] = verdict.failing_row
    if not verdict.eligible:
        record["outcome"] = OUTCOME_INELIGIBLE
        record["reason"] = REASON_RUN_INELIGIBILITY
        record["run_ineligibility_reasons"] = list(verdict.reasons)
        nulls["measured_value"] = REASON_RUN_INELIGIBILITY
        return record

    # Tier 6: the threshold comparison, equality included.
    measured = _run_p95(run.records.get(campaign_id))
    if measured is None:
        raise ComparisonError("measured_value_missing", f"{run.run_key}/{campaign_id}")
    record["measured_value"] = measured
    if declared_direction == DIRECTION_LOWER_IS_BETTER:
        passes = measured <= threshold
    else:
        passes = measured >= threshold
    record["outcome"] = OUTCOME_PASS if passes else OUTCOME_FAIL
    record["reason"] = REASON_THRESHOLD_PASS if passes else REASON_THRESHOLD_FAIL
    return record


def build_comparison(
    document: Mapping[str, Any], run: SharedRun, *, direction: str | None = None
) -> dict[str, Any]:
    """Evaluate one fresh comparison run against a committed cell baseline.

    The document carries the committed inputs it derives from, the label
    precedence and reason order it obeys, and one marked outcome per co-measured
    campaign id of the cell. It is a record of a comparison, never an activation
    state, and nothing here writes anything: :func:`write_comparison` is the only
    writer and it writes under RUN_DIR.
    """
    cell = _baseline_cell(document)
    return {
        "schema": COMPARISON_SCHEMA_ID,
        "kind": COMPARISON_KIND,
        "cell": cell,
        "committed_baseline_schema": document.get("schema"),
        "committed_attempt_digest": document.get("attempt_digest"),
        "closure_status": _closure_status(document),
        "comparison_run": {"run_id": run.run_key, "run_start_utc": run.run_start_utc},
        "label_precedence": list(COMPARISON_LABEL_PRECEDENCE),
        "reason_order": list(REASON_ORDER),
        "records": {
            campaign_id: evaluate_comparison_row(
                document, campaign_id, run, direction=direction
            )
            for campaign_id in cell_campaign_ids(cell)
        },
    }


# ---------------------------------------------------------------------------
# The RUN_DIR-only writer
# ---------------------------------------------------------------------------


def comparison_run_dir() -> Path:
    """Return the canonical RUN_DIR comparisons are written under.

    RUN_DIR comes from ``CHARTREUX_PERF_RUN_DIR`` only: there is no default and
    no fallback, and a RUN_DIR that is not disjoint from the repository is
    refused, so a comparison can never land in the tree.
    """
    raw = os.environ.get(COMPARISON_RUN_DIR_ENV)
    if not raw or not raw.strip():
        raise ComparisonError(
            "run_dir_unset",
            f"{COMPARISON_RUN_DIR_ENV} is not set; comparisons are RUN_DIR-only",
        )
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ComparisonError("run_dir_relative", f"{raw!r} is not absolute")
    root = candidate.resolve()
    if _descendant(root, _REPO_ROOT) or _descendant(_REPO_ROOT, root):
        raise ComparisonError(
            "run_dir_in_repo", f"{root} is not disjoint from {_REPO_ROOT}"
        )
    return root


def comparison_write_path(
    *, filename: str = COMPARISON_FILENAME, run_dir: Path | str | None = None
) -> Path:
    """Return the checked write target for one comparison document.

    The target must be a plain ``.json`` name under RUN_DIR (or a subdirectory of
    it), outside the repository and never in the committed baselines directory.
    """
    root = comparison_run_dir()
    if run_dir is None:
        directory = root
    else:
        candidate = Path(run_dir)
        if not candidate.is_absolute():
            raise ComparisonError("run_dir_relative", f"{run_dir!r} is not absolute")
        directory = candidate.resolve()
        if not _descendant(directory, root):
            raise ComparisonError(
                "run_dir_outside_run_dir", f"{directory} is not under {root}"
            )
    name = Path(filename)
    if name.name != filename or name.suffix != ".json":
        raise ComparisonError(
            "filename_invalid", f"{filename!r} is not a plain .json name"
        )
    target = (directory / filename).resolve()
    if not _descendant(target.parent, root):
        raise ComparisonError("write_escape", f"{target} escapes {root}")
    if _descendant(target, _REPO_ROOT):
        raise ComparisonError("write_in_repo", f"{target} is inside {_REPO_ROOT}")
    if _descendant(target, committed_baselines_dir()):
        raise ComparisonError(
            "write_in_baselines", f"{target} would write the committed baselines"
        )
    return target


def write_comparison(
    document: Mapping[str, Any],
    *,
    filename: str = COMPARISON_FILENAME,
    run_dir: Path | str | None = None,
) -> Path:
    """Write one comparison document under RUN_DIR and return its path."""
    target = comparison_write_path(filename=filename, run_dir=run_dir)
    try:
        text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    except ValueError as error:
        raise ComparisonError(
            "document_not_json_safe", f"a non-finite field cannot be written: {error}"
        ) from error
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{text}\n", encoding="utf-8")
    return target
