"""X7a: pure eligibility and threshold construction for the runtime gate.

ACTIVATION is constructed from the COMMITTED BASELINE SET ALONE. This module
takes the pinned inputs — the committed registry, the pinned determinants
artifact, the promoted per-cell baselines, and the activation machine context —
and returns one deterministic ACTIVE/DORMANT disposition per registry row. It
measures nothing, writes nothing, and no argument of
:func:`construct_threshold_table` can carry a measurement, so the activation
event is structurally unable to consult a fresh value.

Per row the construction is: extract that row's committed run-p95s from its
cell's promoted baseline (the row-specific gated series the producer measured —
A1's residual overhead, A3's chunk gaps, B3/B4's per-root-turn counts, C1/C2's
save phases, G1's slopes), take M = median and D = MAD of the five, and build
T = M + max(6D, 0.20M). A row ACTIVATES only when its campaign is
decision-ready (``eligible_complete`` with exactly five selected runs) AND its T
is finite and positive; otherwise it is DORMANT with the FIRST applicable reason
of the reason order the artifact pins. Only the dormancy tiers (1)-(4) can set
the state here: run ineligibility (tier 5) stays a run-record reason and a
threshold comparison (tier 6) is never evaluated at activation.

Positive-growth rows (G1's RSS slope) clamp a negative committed p95 to zero
before the arithmetic and keep the signed value beside it; a nonpositive T is
DORMANT ``degenerate_threshold`` — never a padded epsilon.

What the module refuses to decide matters as much as what it decides. A
malformed, drifting or ambiguous input halts with :class:`ThresholdError`: an
unparseable or reordered reason order, an activation machine context with a
required field missing (an INDETERMINATE activation input, never a discretionary
DORMANT or ACTIVE), a registry row that disagrees with the in-repo row table, or
a promoted baseline that fails X5a's validation. Nothing is improvised. The
module makes no network call and spawns no process, so it carries no egress
path to close.

X7b adds the reporting layer. One fresh comparison run's marked states are
rendered into the WP6 report as an additive section, with the pinned row-first
precedence held: a dormant row records DORMANT with its dormancy reason before
any run-level evaluation, an eligible row with an ineligible comparison run
records COMPARISON-INELIGIBLE without a threshold outcome, and only an eligible
row with an eligible run records ACTIVE-pass / ACTIVE-fail. A machine context
that differs from the committed baseline machine in ANY of the four required
fields emits a PERF-BLOB machine-mismatch record through the WP1 path guard (so
it can only land under RUN_DIR) and records DORMANT ``machine_mismatch``: the
cell is never ACTIVE on mismatched hardware and never FAILED either.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from tests.perf._runtime_baselines import (
    COMPARISON_SCHEMA_ID,
    DEFAULT_DIRECTION,
    DIRECTION_LOWER_IS_BETTER,
    DIRECTIONS,
    INSUFFICIENT_DATA_REASON,
    MACHINE_FIELDS,
    OUTCOME_DORMANT,
    OUTCOME_INELIGIBLE,
    OUTCOME_PASS,
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
    BaselineError,
    ComparisonError,
    cell_campaign_ids,
    gated_cells,
    load_committed_baseline,
    machine_fields_missing,
    machine_mismatch_fields,
    mad,
    median,
    spec_from_baseline,
    threshold_value,
    validate_baseline,
)
from tests.perf._runtime_metrics import PerfPathError, resolve_perf_paths
from tests.perf._runtime_report import (
    CLOSURE_ELIGIBLE_COMPLETE,
    COMPARISON_STATE_DORMANT,
    COMPARISON_STATE_PRECEDENCE,
    COMPARISON_STATE_RULE,
    COMPARISON_STATES_SECTION,
    DETERMINANTS_SCHEMA_ID,
    GATED_CAMPAIGN_IDS,
    PERF_BLOB_KIND,
    PERF_BLOB_MACHINE_MISMATCH,
    PERF_BLOB_SECTION,
    ROW_SPECS,
    RSS_SLOPE,
    SLOPE_CLAMP_RULE,
    validate_comparison_states_section,
    validate_perf_blob_section,
)
from tests.perf._runtime_usage import (
    DETERMINANTS_FILENAME,
    PERF_DIR_NAME,
    REGISTRY_FILENAME,
    ReleaseRow,
    ReleaseUsageError,
    determinant_rows,
    release_rows,
    repository_root,
)

__all__ = [
    "ACTIVATION_INDETERMINATE",
    "ACTIVE_TEST_REQUIRED_PHRASES",
    "DETERMINANTS_SHA256",
    "DORMANCY_REASONS",
    "DORMANCY_TIERS",
    "NON_DORMANCY_REASONS",
    "PERF_BLOB_FILENAME",
    "PINNED_REASON_TIERS",
    "REGISTRY_SHA256",
    "RUN_RECORD_TIER",
    "STATE_ACTIVE",
    "STATE_DORMANT",
    "THRESHOLDS_SCHEMA_ID",
    "THRESHOLD_TIER",
    "UNMEASURED_REASON",
    "ActivationInputs",
    "ComparisonState",
    "RowDisposition",
    "RowPolicy",
    "RowSeries",
    "ThresholdError",
    "active_test_rule",
    "build_activation_table",
    "build_comparison_states",
    "comparison_state",
    "construct_row",
    "construct_threshold_table",
    "determinants_rows",
    "emit_perf_blob",
    "extract_row_series",
    "load_activation_inputs",
    "machine_mismatch_blob",
    "parse_reason_order",
    "perf_blob_section",
    "registry_identity",
    "reporting_sections",
    "row_policy",
    "validate_state_section",
]

#: The constructed threshold table's own schema id (an activation artifact).
THRESHOLDS_SCHEMA_ID = "chartreux.runtime_thresholds.v1"

STATE_ACTIVE = "ACTIVE"
STATE_DORMANT = "DORMANT"

#: An unevaluable activation-machine context is an INDETERMINATE activation
#: input: it raises this escalation signal and never becomes a state.
ACTIVATION_INDETERMINATE = "activation_machine_indeterminate"

#: No activation disposition records a measured value, ever.
UNMEASURED_REASON = "activation_never_evaluates_a_measured_value"

#: Plan §7 pins of the P-Artifacts commit. A mismatch is STOP + ESCALATE:
#: re-emitting, re-formatting or editing the JSON to force a match is never
#: authorized without a user-accepted plan amendment.
REGISTRY_SHA256 = "7b336621b946b5760fb13ad673c70d4b4c14bf90d6325543ca9e530a35086de5"
DETERMINANTS_SHA256 = "a1847c336f05c83cd3f50cfe3cc756ce2a4b453f79eb9a4a0ddc02a60a7cf21f"
PINNED_DIGESTS: dict[str, str] = {
    "registry_sha256": REGISTRY_SHA256,
    "determinants_sha256": DETERMINANTS_SHA256,
}


class ThresholdError(ValueError):
    """An activation input cannot be trusted, or an indeterminate context is met."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"threshold construction rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ReasonTier:
    """One tier of the pinned reason order, as the artifact states it."""

    tier: int
    rule: str
    recorded_reasons: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        """Return the tier in table form."""
        return {
            "tier": self.tier,
            "rule": self.rule,
            "recorded_reasons": list(self.recorded_reasons),
        }


#: The reason order this module implements, in the order the artifact pins.
#: Tiers 1-4 are the dormancy rules and set the activation state; tier 5 is a
#: RUN-RECORD reason and tier 6 labels comparisons, so neither is ever an
#: activation disposition.
PINNED_REASON_TIERS: tuple[ReasonTier, ...] = (
    ReasonTier(1, "machine_mismatch", (REASON_MACHINE_MISMATCH,)),
    ReasonTier(2, "campaign_not_decision_ready", (INSUFFICIENT_DATA_REASON,)),
    ReasonTier(3, "missing_determinants", (REASON_MISSING_DETERMINANTS,)),
    ReasonTier(4, "degenerate_threshold", (REASON_DEGENERATE_THRESHOLD,)),
    ReasonTier(5, "run_ineligibility", (REASON_RUN_INELIGIBILITY,)),
    ReasonTier(
        6,
        "threshold_fail / threshold_pass",
        (REASON_THRESHOLD_FAIL, REASON_THRESHOLD_PASS),
    ),
)
DORMANCY_TIERS: tuple[ReasonTier, ...] = PINNED_REASON_TIERS[:4]
RUN_RECORD_TIER: ReasonTier = PINNED_REASON_TIERS[4]
THRESHOLD_TIER: ReasonTier = PINNED_REASON_TIERS[5]

#: The reasons a DORMANT activation row may record, most significant first.
DORMANCY_REASONS: tuple[str, ...] = tuple(
    reason for tier in DORMANCY_TIERS for reason in tier.recorded_reasons
)
#: Vocabulary that must NEVER appear as an activation disposition: tier 5 is a
#: run-record reason and tier 6 belongs to an eligible fresh comparison run.
NON_DORMANCY_REASONS: tuple[str, ...] = (
    REASON_RUN_INELIGIBILITY,
    REASON_INVALID_SAMPLE_CAP,
    REASON_THRESHOLD_FAIL,
    REASON_THRESHOLD_PASS,
)

#: The commitments the pinned active-test statement must still carry, word for
#: word. The artifact is the binding authority, so a reworded statement is a
#: drift this constructor refuses to interpret.
ACTIVE_TEST_REQUIRED_PHRASES: tuple[str, ...] = (
    "COMMITTED BASELINE SET ALONE",
    "exactly 5 eligible selected runs",
    "finite and positive",
    "never evaluates a fresh measured value",
    "INDETERMINATE",
    "halts and escalates",
)

_ORDER_MARKER = "in this order:"
_TIER_MARKER = re.compile(r"\((\d)\)")
_RECORDED_REASON = re.compile(r"recorded reason ([a-z_]+)")
_RECORDED_SUBTYPE = re.compile(r"recorded subtype ([a-z_]+)")
#: Where a tier's leading rule token ends; the em dash is escaped so the source
#: stays ASCII while matching the artifact's prose.
_RULE_DELIMITERS: tuple[str, ...] = (";", ".", " \u2014 ")


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _leading_rule(segment: str) -> str:
    """Return the leading rule token of one tier's text."""
    text = segment.strip()
    cut = len(text)
    for delimiter in _RULE_DELIMITERS:
        found = text.find(delimiter)
        if found != -1:
            cut = min(cut, found)
    return text[:cut].strip()


def _flatten(tiers: Sequence[ReasonTier]) -> tuple[str, ...]:
    return tuple(reason for tier in tiers for reason in tier.recorded_reasons)


def _tier_reasons(tier: int, segment: str) -> tuple[str, ...]:
    """Return one tier's recorded reasons, refusing a reworded tier."""
    if tier == 5:
        subtype = _RECORDED_SUBTYPE.search(segment)
        if subtype is None or subtype.group(1) != REASON_INVALID_SAMPLE_CAP:
            raise ThresholdError(
                "determinants_reason_order_drift",
                "tier 5 must record the invalid_sample_cap subtype",
            )
        return (REASON_RUN_INELIGIBILITY,)
    if tier == 6:
        return tuple(
            label.strip() for label in _leading_rule(segment).split("/") if label
        )
    if tier == 2:
        recorded = _RECORDED_REASON.search(segment)
        if recorded is None:
            raise ThresholdError(
                "determinants_reason_order_drift",
                "tier 2 must record insufficient_baseline_data",
            )
        return (recorded.group(1),)
    return (_leading_rule(segment),)


def parse_reason_order(text: Any) -> tuple[ReasonTier, ...]:
    """Parse the pinned reason order out of the determinants artifact.

    The artifact is the binding authority for the order, so the order is read
    from it rather than assumed, and a parsed order that is not exactly the
    pinned one (or that no longer flattens onto the reason vocabulary the
    comparison layer records) is a drift this constructor halts on.
    """
    if not isinstance(text, str) or _ORDER_MARKER not in text:
        raise ThresholdError("determinants_reason_order_missing", f"{text!r}")
    tail = text.split(_ORDER_MARKER, 1)[1]
    # The enumeration is the strictly increasing marker run 1, 2, 3, ...; the
    # taxonomy prose that follows it also numbers things in parentheses, so the
    # run stops at the first marker that is not the next expected tier.
    markers: list[re.Match[str]] = []
    for expected, marker in enumerate(_TIER_MARKER.finditer(tail), start=1):
        if int(marker.group(1)) != expected:
            break
        markers.append(marker)
    if len(markers) != len(PINNED_REASON_TIERS):
        raise ThresholdError(
            "determinants_reason_order_drift",
            f"{len(markers)} tiers, expected {len(PINNED_REASON_TIERS)}",
        )
    tiers: list[ReasonTier] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(tail)
        segment = tail[marker.end() : end]
        tier = int(marker.group(1))
        tiers.append(
            ReasonTier(tier, _leading_rule(segment), _tier_reasons(tier, segment))
        )
    parsed = tuple(tiers)
    if parsed != PINNED_REASON_TIERS:
        raise ThresholdError(
            "determinants_reason_order_drift",
            f"[{', '.join(str(tier.as_record()) for tier in parsed)}]",
        )
    if _flatten(parsed) != REASON_ORDER:
        raise ThresholdError(
            "determinants_reason_order_drift",
            f"{list(_flatten(parsed))} != {list(REASON_ORDER)}",
        )
    return parsed


def _common_determinants(determinants: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the artifact's common determinants, refusing an unusable one."""
    if not isinstance(determinants, Mapping):
        raise ThresholdError("determinants_not_mapping", repr(determinants))
    if determinants.get("schema") != DETERMINANTS_SCHEMA_ID:
        raise ThresholdError(
            "determinants_schema_mismatch", repr(determinants.get("schema"))
        )
    common = determinants.get("common_determinants")
    if not isinstance(common, Mapping):
        raise ThresholdError("determinants_common_missing", repr(common))
    return common


def active_test_rule(determinants: Mapping[str, Any]) -> str:
    """Return the pinned activation statement, refusing a reworded one."""
    text = _common_determinants(determinants).get("active_test")
    if not _is_text(text):
        raise ThresholdError("determinants_active_test_missing", repr(text))
    assert isinstance(text, str)
    missing = [phrase for phrase in ACTIVE_TEST_REQUIRED_PHRASES if phrase not in text]
    if missing:
        raise ThresholdError(
            "determinants_active_test_drift",
            f"the pinned activation statement no longer carries {missing}",
        )
    return text


def determinants_rows(determinants: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the artifact's per-row methodology section."""
    try:
        return determinant_rows(determinants)
    except ReleaseUsageError as error:
        raise ThresholdError(
            "determinants_invalid", f"{error.reason}: {error.detail}"
        ) from error


def registry_identity(registry: Mapping[str, Any]) -> tuple[ReleaseRow, ...]:
    """Return the registry's rows, bound to the in-repo row table."""
    try:
        rows = release_rows(registry)
    except ReleaseUsageError as error:
        raise ThresholdError(
            "registry_invalid", f"{error.reason}: {error.detail}"
        ) from error
    ids = tuple(row.campaign_id for row in rows)
    if ids != GATED_CAMPAIGN_IDS:
        raise ThresholdError(
            "registry_row_set_mismatch", f"{list(ids)} != {list(GATED_CAMPAIGN_IDS)}"
        )
    for row in rows:
        spec = ROW_SPECS[row.campaign_id]
        identity = (spec.metric, spec.unit, spec.cell, spec.producer_wp)
        if identity != (row.metric, row.unit, row.cell, row.producer_wp):
            raise ThresholdError(
                "registry_row_drift",
                f"{row.campaign_id}: {identity!r} != "
                f"{(row.metric, row.unit, row.cell, row.producer_wp)!r}",
            )
    return rows


@dataclass(frozen=True, slots=True)
class RowPolicy:
    """One registry row's extraction policy: its gated quantity and clamp rule."""

    campaign_id: str
    metric: str
    unit: str
    cell: str
    #: True for a positive-growth row, whose negative committed values are
    #: clamped to zero for gate comparison.
    clamp_non_negative: bool


def row_policy(row: ReleaseRow) -> RowPolicy:
    """Return the row's own extraction policy."""
    return RowPolicy(
        campaign_id=row.campaign_id,
        metric=row.metric,
        unit=row.unit,
        cell=row.cell,
        clamp_non_negative=row.metric == RSS_SLOPE,
    )


def _determinants_match(row: ReleaseRow, entry: Any) -> bool:
    """True when the artifact's row describes exactly this registry row."""
    if not isinstance(entry, Mapping):
        return False
    identity = (
        entry.get("campaign_id"),
        entry.get("metric"),
        entry.get("cell"),
        entry.get("producer_wp"),
    )
    if identity != (row.campaign_id, row.metric, row.cell, row.producer_wp):
        return False
    return entry.get("unit", row.unit) == row.unit


@dataclass(frozen=True, slots=True)
class RowSeries:
    """One row's committed run-p95s, as the promoted baseline carries them."""

    selected_runs: tuple[tuple[str, str], ...]
    p95s: tuple[float, ...]
    signed_p95s: tuple[float, ...]
    slope_clamped: bool


def _attempt_starts(document: Mapping[str, Any]) -> dict[str, str]:
    """Return every attempt's run start, refusing an unusable attempt list."""
    attempts = document.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ThresholdError("baseline_attempts_missing", repr(attempts))
    starts: dict[str, str] = {}
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ThresholdError("baseline_attempt_invalid", repr(attempt))
        run_id = attempt.get("run_id")
        run_start = attempt.get("run_start_utc")
        if not _is_text(run_id) or not _is_text(run_start):
            raise ThresholdError(
                "baseline_attempt_invalid", f"{run_id!r} at {run_start!r}"
            )
        assert isinstance(run_id, str)
        assert isinstance(run_start, str)
        starts[run_id] = run_start
    return starts


def _campaign_block(
    document: Mapping[str, Any], policy: RowPolicy
) -> Mapping[str, Any]:
    """Return one row's committed campaign block, refusing another row's."""
    campaigns = document.get("campaigns")
    if not isinstance(campaigns, Mapping):
        raise ThresholdError("baseline_campaigns_missing", policy.cell)
    block = campaigns.get(policy.campaign_id)
    if not isinstance(block, Mapping):
        raise ThresholdError(
            "baseline_campaign_missing", f"{policy.cell}/{policy.campaign_id}"
        )
    identity = (
        block.get("campaign_id"),
        block.get("metric"),
        block.get("unit"),
        block.get("cell"),
    )
    expected = (policy.campaign_id, policy.metric, policy.unit, policy.cell)
    if identity != expected:
        raise ThresholdError(
            "baseline_row_identity_drift",
            f"{policy.cell}/{policy.campaign_id}: {identity!r} != {expected!r}",
        )
    return block


def extract_row_series(document: Mapping[str, Any], policy: RowPolicy) -> RowSeries:
    """Extract one row's committed run-p95s from its cell's baseline.

    The series is the row-specific gated quantity the producer measured and X5a
    selected; a positive-growth row is clamped at zero here, with the signed
    values kept beside the clamp.
    """
    block = _campaign_block(document, policy)
    run_ids = block.get("run_ids")
    if (
        not isinstance(run_ids, list)
        or not run_ids
        or not all(_is_text(run_id) for run_id in run_ids)
    ):
        raise ThresholdError("baseline_run_ids_invalid", repr(run_ids))
    raw = block.get("run_p95s")
    if not isinstance(raw, list) or len(raw) != len(run_ids):
        raise ThresholdError(
            "baseline_p95s_invalid",
            f"{policy.cell}/{policy.campaign_id}: {raw!r} for {len(run_ids)} runs",
        )
    if len(run_ids) > SELECTED_RUNS:
        raise ThresholdError(
            "baseline_selection_too_wide",
            f"{policy.cell}/{policy.campaign_id}: {len(run_ids)} runs",
        )
    starts = _attempt_starts(document)
    values: list[float] = []
    selected: list[tuple[str, str]] = []
    for run_id, value in zip(run_ids, raw, strict=True):
        assert isinstance(run_id, str)
        start = starts.get(run_id)
        if start is None:
            raise ThresholdError("baseline_attempt_missing", run_id)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ThresholdError(
                "baseline_p95s_invalid", f"{policy.campaign_id}/{run_id}: {value!r}"
            )
        values.append(float(value))
        selected.append((run_id, start))
    if policy.clamp_non_negative:
        gated = tuple(max(0.0, value) for value in values)
        clamped = any(gate != value for gate, value in zip(gated, values, strict=True))
    else:
        gated = tuple(values)
        clamped = False
    return RowSeries(
        selected_runs=tuple(selected),
        p95s=gated,
        signed_p95s=tuple(values),
        slope_clamped=clamped,
    )


def _closure_status(document: Mapping[str, Any]) -> str | None:
    closure = document.get("closure")
    if not isinstance(closure, Mapping):
        return None
    status = closure.get("status")
    return status if isinstance(status, str) else None


def _decision_ready(document: Mapping[str, Any]) -> bool:
    """True when the committed campaign is decision-ready (eligible_complete, 5)."""
    if _closure_status(document) != CLOSURE_ELIGIBLE_COMPLETE:
        return False
    closure = document.get("closure")
    selected = closure.get("selected_run_ids") if isinstance(closure, Mapping) else None
    return (
        isinstance(selected, list)
        and len(selected) == SELECTED_RUNS
        and all(_is_text(run_id) for run_id in selected)
    )


def _statistics(
    p95s: Sequence[float], null_reasons: dict[str, str], reason: str
) -> tuple[float | None, float | None]:
    """Return M and D, recording a non-computable one as NULL with its reason."""
    if not p95s:
        null_reasons["median"] = reason
        null_reasons["mad"] = reason
        return None, None
    raw_centre = median(p95s)
    raw_deviation = mad(p95s)
    centre = raw_centre if math.isfinite(raw_centre) else None
    deviation = raw_deviation if math.isfinite(raw_deviation) else None
    if centre is None:
        null_reasons["median"] = REASON_DEGENERATE_THRESHOLD
    if deviation is None:
        null_reasons["mad"] = REASON_DEGENERATE_THRESHOLD
    return centre, deviation


def _authorized_threshold(
    centre: float | None,
    deviation: float | None,
    *,
    ready: bool,
    determinants_match: bool,
    clamp_non_negative: bool,
    null_reasons: dict[str, str],
) -> float | None:
    """Return T when its population is authorized, else NULL with the reason.

    Thin data never gets a threshold, and neither does a row whose committed
    determinants are absent: the row-specific extraction policy is part of what
    the artifact pins. A nonpositive result is degenerate, never padded.
    """
    if not ready:
        null_reasons["threshold"] = INSUFFICIENT_DATA_REASON
        return None
    if not determinants_match:
        null_reasons["threshold"] = REASON_MISSING_DETERMINANTS
        return None
    if centre is None or deviation is None:
        null_reasons["threshold"] = REASON_DEGENERATE_THRESHOLD
        return None
    candidate = threshold_value(centre, deviation)
    if clamp_non_negative:
        candidate = max(0.0, candidate)
    if math.isfinite(candidate) and candidate > 0:
        return candidate
    null_reasons["threshold"] = REASON_DEGENERATE_THRESHOLD
    return None


@dataclass(frozen=True, slots=True)
class RowDisposition:
    """One registry row's constructed activation disposition."""

    campaign_id: str
    metric: str
    unit: str
    cell: str
    closure_status: str | None
    decision_ready: bool
    selected_runs: tuple[tuple[str, str], ...]
    collected_p95s: tuple[float, ...]
    signed_p95s: tuple[float, ...]
    slope_clamped: bool
    clamp_rule: str | None
    median: float | None
    mad: float | None
    threshold: float | None
    state: str
    reason: str | None
    machine_mismatch_fields: tuple[str, ...]
    null_reasons: Mapping[str, str]

    @property
    def active(self) -> bool:
        """True when the row ACTIVATES at stage-2 exit."""
        return self.state == STATE_ACTIVE

    def as_record(self) -> dict[str, Any]:
        """Return the disposition in table form."""
        return {
            "campaign_id": self.campaign_id,
            "metric": self.metric,
            "unit": self.unit,
            "cell": self.cell,
            "closure_status": self.closure_status,
            "decision_ready": self.decision_ready,
            "selected_runs": [
                {"run_id": run_id, "run_start_utc": run_start}
                for run_id, run_start in self.selected_runs
            ],
            "collected_run_p95s": list(self.collected_p95s),
            "collected_run_p95s_signed": list(self.signed_p95s),
            "slope_clamp": self.clamp_rule,
            "slope_clamped": self.slope_clamped,
            "median": self.median,
            "mad": self.mad,
            "threshold": self.threshold,
            "state": self.state,
            "reason": self.reason,
            "machine_mismatch_fields": list(self.machine_mismatch_fields),
            "measured_value": None,
            "null_reasons": dict(self.null_reasons),
        }


def _clamp_rule(policy: RowPolicy) -> str | None:
    """Return the row's clamp rule: a positive-growth row clamps at zero."""
    return SLOPE_CLAMP_RULE if policy.clamp_non_negative else None


def _absent_baseline(row: ReleaseRow) -> RowDisposition:
    """A cell with no promoted baseline is not decision-ready (tier 2)."""
    policy = row_policy(row)
    return RowDisposition(
        campaign_id=row.campaign_id,
        metric=row.metric,
        unit=row.unit,
        cell=row.cell,
        closure_status=None,
        decision_ready=False,
        selected_runs=(),
        collected_p95s=(),
        signed_p95s=(),
        slope_clamped=False,
        clamp_rule=_clamp_rule(policy),
        median=None,
        mad=None,
        threshold=None,
        state=STATE_DORMANT,
        reason=INSUFFICIENT_DATA_REASON,
        machine_mismatch_fields=(),
        null_reasons={
            "collected_run_p95s": INSUFFICIENT_DATA_REASON,
            "median": INSUFFICIENT_DATA_REASON,
            "mad": INSUFFICIENT_DATA_REASON,
            "threshold": INSUFFICIENT_DATA_REASON,
            "measured_value": UNMEASURED_REASON,
        },
    )


def construct_row(
    *,
    row: ReleaseRow,
    entry: Any,
    document: Mapping[str, Any] | None,
    machine: Mapping[str, Any],
) -> RowDisposition:
    """Construct one registry row's disposition from its committed baseline.

    The tiers are applied in the pinned reason order and only tiers 1-4 can set
    the state. The threshold is populated whenever its population is authorized
    by the committed set, so a DORMANT row is never left un-analysed: dormancy
    records why a row does not gate, not that nothing was computed.
    """
    if document is None:
        return _absent_baseline(row)
    policy = row_policy(row)
    series = extract_row_series(document, policy)
    ready = _decision_ready(document)
    if ready and len(series.p95s) != SELECTED_RUNS:
        raise ThresholdError(
            "baseline_selection_incomplete",
            f"{policy.cell}/{policy.campaign_id}: {len(series.p95s)} run-p95s"
            " for a decision-ready campaign",
        )
    null_reasons: dict[str, str] = {"measured_value": UNMEASURED_REASON}
    if not series.p95s:
        null_reasons["collected_run_p95s"] = INSUFFICIENT_DATA_REASON
    centre, deviation = _statistics(
        series.p95s,
        null_reasons,
        INSUFFICIENT_DATA_REASON if not ready else REASON_DEGENERATE_THRESHOLD,
    )
    determinants_match = _determinants_match(row, entry)
    threshold = _authorized_threshold(
        centre,
        deviation,
        ready=ready,
        determinants_match=determinants_match,
        clamp_non_negative=policy.clamp_non_negative,
        null_reasons=null_reasons,
    )
    mismatch = machine_mismatch_fields(document.get("machine"), machine)

    state = STATE_ACTIVE
    reason: str | None = None
    if mismatch:
        state, reason = STATE_DORMANT, REASON_MACHINE_MISMATCH
    elif not ready:
        state, reason = STATE_DORMANT, INSUFFICIENT_DATA_REASON
    elif not determinants_match:
        state, reason = STATE_DORMANT, REASON_MISSING_DETERMINANTS
    elif threshold is None:
        state, reason = STATE_DORMANT, REASON_DEGENERATE_THRESHOLD
    return RowDisposition(
        campaign_id=row.campaign_id,
        metric=row.metric,
        unit=row.unit,
        cell=row.cell,
        closure_status=_closure_status(document),
        decision_ready=ready,
        selected_runs=series.selected_runs,
        collected_p95s=series.p95s,
        signed_p95s=series.signed_p95s,
        slope_clamped=series.slope_clamped,
        clamp_rule=_clamp_rule(policy),
        median=centre,
        mad=deviation,
        threshold=threshold,
        state=state,
        reason=reason,
        machine_mismatch_fields=mismatch,
        null_reasons=null_reasons,
    )


def _checked_baselines(
    baselines: Mapping[str, Any],
) -> dict[str, Mapping[str, Any] | None]:
    """Validate every supplied baseline, refusing an incomplete input set."""
    if not isinstance(baselines, Mapping):
        raise ThresholdError("baselines_not_mapping", repr(baselines))
    unknown = sorted({str(key) for key in baselines} - set(gated_cells()))
    if unknown:
        raise ThresholdError("baselines_unknown_cell", str(unknown))
    missing = [cell for cell in gated_cells() if cell not in baselines]
    if missing:
        raise ThresholdError(
            "baselines_incomplete",
            f"no entry supplied for {missing}; an omitted cell is not 'absent'",
        )
    checked: dict[str, Mapping[str, Any] | None] = {}
    for cell in gated_cells():
        document = baselines[cell]
        if document is None:
            checked[cell] = None
            continue
        if not isinstance(document, Mapping):
            raise ThresholdError("baseline_not_mapping", cell)
        if document.get("cell") != cell:
            raise ThresholdError(
                "baseline_cell_mismatch", f"{cell}: {document.get('cell')!r}"
            )
        try:
            validate_baseline(document, spec_from_baseline(document))
        except (BaselineError, ComparisonError) as error:
            raise ThresholdError("baseline_invalid", f"{cell}: {error}") from error
        checked[cell] = document
    return checked


def _canonical_digest(document: Any) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _baseline_input_identity(
    document: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the committed baseline identity the table cites, when present."""
    if document is None:
        return None
    return {
        "attempt_digest": document.get("attempt_digest"),
        "closure_status": _closure_status(document),
    }


def construct_threshold_table(
    *,
    registry: Mapping[str, Any],
    determinants: Mapping[str, Any],
    baselines: Mapping[str, Any],
    machine: Mapping[str, Any] | None,
    pins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Construct every registry row's ACTIVE/DORMANT disposition.

    The inputs are the pinned artifacts, the promoted baselines, and the
    activation machine context — nothing else exists in this signature, so no
    measured value can enter an activation decision. The reason order is read
    from the artifact, the machine context is checked before any row is
    dispositioned, and every present baseline is validated before it is read.
    """
    order = parse_reason_order(_common_determinants(determinants).get("reason_order"))
    active_test = active_test_rule(determinants)
    rows = registry_identity(registry)
    entries = determinants_rows(determinants)
    if not isinstance(machine, Mapping):
        raise ThresholdError(
            ACTIVATION_INDETERMINATE,
            "the activation machine context is absent; halt and escalate",
        )
    missing = machine_fields_missing(machine)
    if missing:
        raise ThresholdError(
            ACTIVATION_INDETERMINATE,
            f"the activation machine context lacks {list(missing)};"
            " halt and escalate, never a discretionary disposition",
        )
    checked = _checked_baselines(baselines)
    dispositions = [
        construct_row(
            row=row,
            entry=entries.get(row.campaign_id),
            document=checked[row.cell],
            machine=machine,
        )
        for row in rows
    ]
    records = [disposition.as_record() for disposition in dispositions]
    active_count = sum(1 for disposition in dispositions if disposition.active)
    pinned = PINNED_DIGESTS if pins is None else dict(pins)
    return {
        "schema": THRESHOLDS_SCHEMA_ID,
        "threshold_rule": THRESHOLD_RULE,
        "active_test": {
            "rule": active_test,
            "evaluates_fresh_measured_value": False,
            "indeterminate_context": ACTIVATION_INDETERMINATE,
        },
        "reason_order": [tier.as_record() for tier in order],
        "dormancy_tiers": [tier.as_record() for tier in DORMANCY_TIERS],
        "dormancy_reasons": list(DORMANCY_REASONS),
        "run_record_tier": RUN_RECORD_TIER.as_record(),
        "threshold_tier": THRESHOLD_TIER.as_record(),
        "slope_clamp": SLOPE_CLAMP_RULE,
        "activation_machine": {name: machine[name] for name in MACHINE_FIELDS},
        "row_count": len(records),
        "active_count": active_count,
        "dormant_count": len(records) - active_count,
        "rows": records,
        "table_digest": _canonical_digest(records),
        "inputs": {
            "digest_kind": "pinned_file_sha256" if pins else "canonical_document",
            "registry_sha256": pinned["registry_sha256"],
            "determinants_sha256": pinned["determinants_sha256"],
            "baselines": {
                cell: _baseline_input_identity(checked[cell]) for cell in gated_cells()
            },
        },
    }


@dataclass(frozen=True, slots=True)
class ActivationInputs:
    """The pinned inputs one activation construction reads from the checkout."""

    repo_root: Path
    registry: Mapping[str, Any]
    determinants: Mapping[str, Any]
    baselines: Mapping[str, Mapping[str, Any] | None]
    pins: Mapping[str, str]


def _read_pinned(
    root: Path, filename: str, digest: str, label: str
) -> tuple[dict[str, Any], str]:
    """Read one pinned artifact, refusing a drift or an unreadable file."""
    path = root / PERF_DIR_NAME / filename
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ThresholdError(
            "pinned_artifact_unreadable", f"{path}: {error}"
        ) from error
    found = hashlib.sha256(payload).hexdigest()
    if found != digest:
        raise ThresholdError(
            "pinned_artifact_drift",
            f"{label} {found} != {digest}: STOP + ESCALATE — never re-emit or"
            " re-format it to force a match",
        )
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ThresholdError("pinned_artifact_malformed", f"{path}: {error}") from error
    if not isinstance(document, dict):
        raise ThresholdError("pinned_artifact_malformed", f"{path}: not an object")
    return document, found


def load_activation_inputs(
    *, repo_root: Path | None = None, baselines_dir: Path | str | None = None
) -> ActivationInputs:
    """Load the pinned registry and determinants plus every committed baseline.

    Both pinned artifacts are verified against their plan §7 sha256 before they
    are read, and every gated cell is looked up explicitly so an absent cell is
    recorded as absent rather than silently omitted.
    """
    root = repository_root() if repo_root is None else Path(repo_root)
    registry, registry_digest = _read_pinned(
        root, REGISTRY_FILENAME, REGISTRY_SHA256, "registry"
    )
    determinants, determinants_digest = _read_pinned(
        root, DETERMINANTS_FILENAME, DETERMINANTS_SHA256, "determinants"
    )
    directory = (
        root / PERF_DIR_NAME / "baselines"
        if baselines_dir is None
        else Path(baselines_dir)
    )
    baselines: dict[str, Mapping[str, Any] | None] = {}
    for cell in gated_cells():
        try:
            baselines[cell] = load_committed_baseline(cell, baselines_dir=directory)
        except ComparisonError as error:
            raise ThresholdError(
                "baseline_unreadable", f"{cell}: {error.reason}: {error.detail}"
            ) from error
    return ActivationInputs(
        repo_root=root,
        registry=registry,
        determinants=determinants,
        baselines=baselines,
        pins={
            "registry_sha256": registry_digest,
            "determinants_sha256": determinants_digest,
        },
    )


def build_activation_table(
    *,
    machine: Mapping[str, Any] | None,
    repo_root: Path | None = None,
    baselines_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load the committed inputs from the checkout and construct the table."""
    inputs = load_activation_inputs(repo_root=repo_root, baselines_dir=baselines_dir)
    return construct_threshold_table(
        registry=inputs.registry,
        determinants=inputs.determinants,
        baselines=inputs.baselines,
        machine=machine,
        pins=inputs.pins,
    )


# ---------------------------------------------------------------------------
# X7b: comparison states and the PERF-BLOB machine-mismatch record
# ---------------------------------------------------------------------------

#: The PERF-BLOB section is written through the WP1 path guard: to the declared
#: report path when one exists, otherwise to this file under RUN_DIR.
PERF_BLOB_FILENAME = "perf_blob.json"


@dataclass(frozen=True, slots=True)
class ComparisonState:
    """One comparison record's marked state, rendered as a report row."""

    campaign_id: str
    cell: str
    metric: str
    unit: str
    state: str
    reason: str
    direction: str
    machine_mismatch_fields: tuple[str, ...]
    comparison_run: Mapping[str, Any]
    measured_value: float | None
    threshold: float | None
    null_reasons: Mapping[str, str]

    @property
    def precedence(self) -> int:
        """Return the label's index in the pinned total order."""
        return COMPARISON_STATE_PRECEDENCE.index(self.state)

    def as_record(self) -> dict[str, Any]:
        """Return the row the report's comparison-states section carries."""
        return {
            "campaign_id": self.campaign_id,
            "cell": self.cell,
            "metric": self.metric,
            "unit": self.unit,
            "state": self.state,
            "precedence": self.precedence,
            "reason": self.reason,
            "machine_mismatch_fields": list(self.machine_mismatch_fields),
            "comparison_run": dict(self.comparison_run),
            "direction": self.direction,
            "measured_value": self.measured_value,
            "threshold": self.threshold,
            "null_reasons": dict(self.null_reasons),
        }


def _nullable_number(
    container: Mapping[str, Any], name: str, campaign_id: str
) -> float | None:
    """Return one comparison field's number, NULL included, or refuse it."""
    value = container.get(name)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ThresholdError(
            "comparison_field_invalid", f"{campaign_id}: {name}={value!r}"
        )
    return float(value)


def _checked_state(
    *,
    campaign_id: str,
    cell: Any,
    metric: Any,
    unit: Any,
    outcome: Any,
    reason: Any,
    direction: Any,
    mismatch: tuple[str, ...],
    run_block: Any,
    measured: float | None,
    threshold: float | None,
    null_reasons: Mapping[str, str],
) -> ComparisonState:
    """Apply the marked-outcome rules to one row's facts, row first.

    The label is never trusted on its own: each branch re-derives the facts the
    label claims. A dormant row compared nothing and has no run-level verdict;
    an ineligible comparison run has no threshold outcome; a threshold label
    must agree with the relation ``measured <= T`` (equality passes) or its
    higher-is-better counterpart.
    """
    spec = ROW_SPECS.get(campaign_id)
    if spec is None:
        raise ThresholdError("comparison_campaign_unknown", repr(campaign_id))
    if (cell, metric, unit) != (spec.cell, spec.metric, spec.unit):
        raise ThresholdError(
            "comparison_row_identity_drift", f"{campaign_id}: {(cell, metric, unit)!r}"
        )
    if outcome not in COMPARISON_STATE_PRECEDENCE:
        raise ThresholdError("comparison_label_invalid", f"{campaign_id}: {outcome!r}")
    if not _is_text(reason):
        raise ThresholdError("comparison_reason_missing", f"{campaign_id}: {reason!r}")
    if direction not in DIRECTIONS:
        raise ThresholdError(
            "comparison_direction_invalid", f"{campaign_id}: {direction!r}"
        )
    unknown = [field for field in mismatch if field not in MACHINE_FIELDS]
    if unknown:
        raise ThresholdError(
            "comparison_mismatch_fields_invalid", f"{campaign_id}: {unknown}"
        )
    ordered = tuple(field for field in MACHINE_FIELDS if field in mismatch)
    for name, value in (("measured_value", measured), ("threshold", threshold)):
        if value is None and not _is_text(null_reasons.get(name)):
            raise ThresholdError(
                "comparison_null_without_reason", f"{campaign_id}: {name}"
            )
    if not isinstance(run_block, Mapping):
        raise ThresholdError("comparison_run_missing", campaign_id)
    eligible = run_block.get("eligible")
    if eligible is None and not _is_text(run_block.get("null_reason")):
        raise ThresholdError("comparison_run_null_without_reason", campaign_id)
    assert isinstance(reason, str)
    assert isinstance(direction, str)

    if outcome == OUTCOME_DORMANT:
        # Row first: the dormancy reason stands before any run-level evaluation,
        # so a comparison run's own ineligibility never becomes this label.
        if reason not in DORMANCY_REASONS:
            raise ThresholdError(
                "comparison_dormancy_reason_invalid",
                f"{campaign_id}: {reason!r} is not a dormancy reason",
            )
        if bool(ordered) != (reason == REASON_MACHINE_MISMATCH):
            raise ThresholdError(
                "comparison_machine_mismatch_inconsistent",
                f"{campaign_id}: fields {list(ordered)} with reason {reason!r}",
            )
        if measured is not None:
            raise ThresholdError(
                "comparison_dormant_measured_value",
                f"{campaign_id}: a dormant row compares nothing",
            )
        if eligible is not None:
            raise ThresholdError(
                "comparison_dormant_run_evaluated",
                f"{campaign_id}: row-first precedence forbids a run-level verdict",
            )
    elif outcome == OUTCOME_INELIGIBLE:
        if reason != REASON_RUN_INELIGIBILITY:
            raise ThresholdError(
                "comparison_ineligible_reason_invalid", f"{campaign_id}: {reason!r}"
            )
        if ordered:
            raise ThresholdError(
                "comparison_ineligible_mismatched_hardware",
                f"{campaign_id}: {list(ordered)} is a dormant row, not this label",
            )
        if threshold is None or threshold <= 0:
            raise ThresholdError(
                "comparison_ineligible_dormant_row",
                f"{campaign_id}: only a non-dormant row evaluates its run",
            )
        if measured is not None:
            raise ThresholdError(
                "comparison_ineligible_measured_value",
                f"{campaign_id}: an ineligible run carries no threshold outcome",
            )
        if eligible is not False:
            raise ThresholdError(
                "comparison_ineligible_run_eligible", f"{campaign_id}: {eligible!r}"
            )
    else:
        expected = (
            REASON_THRESHOLD_PASS if outcome == OUTCOME_PASS else REASON_THRESHOLD_FAIL
        )
        if reason != expected:
            raise ThresholdError(
                "comparison_threshold_reason_invalid",
                f"{campaign_id}: {reason!r} for {outcome!r}",
            )
        if ordered:
            raise ThresholdError(
                "comparison_active_mismatched_hardware",
                f"{campaign_id}: {list(ordered)} is never ACTIVE",
            )
        if eligible is not True:
            raise ThresholdError(
                "comparison_threshold_run_ineligible", f"{campaign_id}: {eligible!r}"
            )
        if threshold is None or threshold <= 0:
            raise ThresholdError(
                "comparison_threshold_degenerate", f"{campaign_id}: {threshold!r}"
            )
        if measured is None:
            raise ThresholdError("comparison_measured_value_missing", campaign_id)
        if direction == DIRECTION_LOWER_IS_BETTER:
            passes = measured <= threshold
        else:
            passes = measured >= threshold
        if passes != (outcome == OUTCOME_PASS):
            raise ThresholdError(
                "comparison_threshold_relation_mismatch",
                f"{campaign_id}: {measured} against T={threshold} records {outcome!r}",
            )
    return ComparisonState(
        campaign_id=campaign_id,
        cell=spec.cell,
        metric=spec.metric,
        unit=spec.unit,
        state=str(outcome),
        reason=reason,
        direction=direction,
        machine_mismatch_fields=ordered,
        comparison_run=dict(run_block),
        measured_value=measured,
        threshold=threshold,
        null_reasons=dict(null_reasons),
    )


def comparison_state(record: Mapping[str, Any]) -> ComparisonState:
    """Return one X5b comparison record's marked state, row-first precedence held.

    The record's own fields are re-checked rather than trusted: a machine
    mismatch is DORMANT ``machine_mismatch`` before any run-level evaluation
    (never ACTIVE, never FAILED), an ineligible comparison run on an eligible
    row is COMPARISON-INELIGIBLE with no threshold outcome, and a threshold
    label must agree with the relation itself.
    """
    if not isinstance(record, Mapping):
        raise ThresholdError("comparison_record_not_mapping", repr(record))
    campaign_id = record.get("campaign_id")
    if not isinstance(campaign_id, str):
        raise ThresholdError("comparison_campaign_unknown", repr(campaign_id))
    declared = record.get("machine_mismatch_fields")
    if declared is None:
        declared = []
    if not isinstance(declared, list) or not all(
        isinstance(field, str) for field in declared
    ):
        raise ThresholdError(
            "comparison_mismatch_fields_invalid", f"{campaign_id}: {declared!r}"
        )
    nulls = record.get("null_reasons")
    if not isinstance(nulls, Mapping):
        raise ThresholdError("comparison_null_reasons_missing", campaign_id)
    direction = record.get("direction")
    return _checked_state(
        campaign_id=campaign_id,
        cell=record.get("cell"),
        metric=record.get("metric"),
        unit=record.get("unit"),
        outcome=record.get("outcome"),
        reason=record.get("reason"),
        direction=DEFAULT_DIRECTION if direction is None else direction,
        mismatch=tuple(declared),
        run_block=record.get("comparison_run"),
        measured=_nullable_number(record, "measured_value", campaign_id),
        threshold=_nullable_number(record, "threshold", campaign_id),
        null_reasons={str(key): str(value) for key, value in nulls.items()},
    )


def build_comparison_states(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Render X5b comparison documents into the report's comparison-states section.

    Every co-measured row of the cell is present, in registry order, with the
    label its own facts support; a missing, duplicated or foreign row is a
    malformed input and halts rather than being skipped.
    """
    if isinstance(comparisons, (str, bytes)) or not comparisons:
        raise ThresholdError(
            "comparison_documents_missing", "no comparison document supplied"
        )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for document in comparisons:
        if (
            not isinstance(document, Mapping)
            or document.get("schema") != COMPARISON_SCHEMA_ID
        ):
            raise ThresholdError("comparison_schema_mismatch", repr(document)[:120])
        cell = document.get("cell")
        if not isinstance(cell, str) or cell not in gated_cells():
            raise ThresholdError("comparison_cell_invalid", repr(cell))
        records = document.get("records")
        if not isinstance(records, Mapping):
            raise ThresholdError("comparison_records_missing", cell)
        expected = cell_campaign_ids(cell)
        unexpected = sorted(str(key) for key in set(records) - set(expected))
        if unexpected:
            raise ThresholdError("comparison_row_unexpected", f"{cell}: {unexpected}")
        for campaign_id in expected:
            record = records.get(campaign_id)
            if record is None:
                raise ThresholdError("comparison_row_missing", f"{cell}/{campaign_id}")
            row = comparison_state(record)
            if row.cell != cell:
                raise ThresholdError(
                    "comparison_row_cell_mismatch", f"{campaign_id}: {row.cell!r}"
                )
            key = (cell, campaign_id)
            if key in seen:
                raise ThresholdError("comparison_row_duplicated", str(key))
            seen.add(key)
            rows.append(row.as_record())
    counts = {label: 0 for label in COMPARISON_STATE_PRECEDENCE}
    for row in rows:
        counts[row["state"]] += 1
    section = {
        "rule": COMPARISON_STATE_RULE,
        "precedence": list(COMPARISON_STATE_PRECEDENCE),
        "counts": counts,
        "rows": rows,
    }
    validate_state_section(section)
    return section


def validate_state_section(section: Mapping[str, Any]) -> None:
    """Re-derive every rendered row: the section is valid when the rules agree.

    The structural rules live with the schema
    (:func:`validate_comparison_states_section`); this layer re-applies the
    marked-outcome rules to each row's own fields and requires the rendered row
    to be exactly what they produce, so a hand-edited label cannot survive.
    """
    validate_comparison_states_section(section)
    for row in section["rows"]:
        derived = _checked_state(
            campaign_id=row["campaign_id"],
            cell=row["cell"],
            metric=row["metric"],
            unit=row["unit"],
            outcome=row["state"],
            reason=row["reason"],
            direction=row["direction"],
            mismatch=tuple(row["machine_mismatch_fields"]),
            run_block=row["comparison_run"],
            measured=_nullable_number(row, "measured_value", row["campaign_id"]),
            threshold=_nullable_number(row, "threshold", row["campaign_id"]),
            null_reasons={
                str(key): str(value) for key, value in row["null_reasons"].items()
            },
        ).as_record()
        if _canonical_digest(derived) != _canonical_digest(dict(row)):
            raise ThresholdError(
                "comparison_state_row_drift",
                f"{row['campaign_id']}: the rendered row is not what the"
                " marked-outcome rules produce",
            )


def machine_mismatch_blob(
    *,
    cell: str,
    baseline_machine: Mapping[str, Any] | None,
    comparison_machine: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return one cell's PERF-BLOB machine-mismatch record, or None.

    A mismatch is inequality in ANY of the four required machine-context fields,
    never core_count alone. A field the comparison machine cannot report is a
    run-record defect (``machine_field_missing``, a run-level reason), not
    evidence the machine differs, so it is excluded exactly as the comparison
    tool excludes it. A baseline machine that cannot be evaluated has nothing
    trustworthy to compare against: that halts instead of emitting a record.
    """
    if cell not in gated_cells():
        raise ThresholdError("blob_cell_invalid", repr(cell))
    baseline_missing = machine_fields_missing(baseline_machine)
    if baseline_missing:
        raise ThresholdError(
            "baseline_machine_indeterminate",
            f"{cell}: the committed baseline machine lacks {list(baseline_missing)}",
        )
    assert isinstance(baseline_machine, Mapping)
    comparison_missing = machine_fields_missing(comparison_machine)
    usable = {
        name: comparison_machine[name]
        for name in MACHINE_FIELDS
        if isinstance(comparison_machine, Mapping) and name not in comparison_missing
    }
    mismatch = tuple(
        name
        for name in MACHINE_FIELDS
        if name in usable and baseline_machine.get(name) != usable[name]
    )
    if not mismatch:
        return None
    return {
        "kind": PERF_BLOB_KIND,
        "record": PERF_BLOB_MACHINE_MISMATCH,
        "cell": cell,
        "campaign_ids": list(cell_campaign_ids(cell)),
        "mismatch_fields": list(mismatch),
        "baseline_machine": {name: baseline_machine[name] for name in MACHINE_FIELDS},
        "comparison_machine": {
            name: usable[name] for name in MACHINE_FIELDS if name in usable
        },
        "state": COMPARISON_STATE_DORMANT,
        "reason": REASON_MACHINE_MISMATCH,
        "never_active": True,
        "never_failed": True,
    }


def perf_blob_section(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the PERF-BLOB section: every machine-mismatch record of the run."""
    section = {"kind": PERF_BLOB_KIND, "records": [dict(record) for record in records]}
    validate_perf_blob_section(section)
    return section


def reporting_sections(
    *, comparisons: Sequence[Mapping[str, Any]], blobs: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Return the two additive report sections X7b renders.

    They merge into a report through the schema's own extension mechanism
    (``report_schema(extra_sections=...)``), never by editing a core section.
    """
    return {
        COMPARISON_STATES_SECTION: build_comparison_states(comparisons),
        PERF_BLOB_SECTION: perf_blob_section(blobs),
    }


def emit_perf_blob(
    section: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    scratchpad_root: Path | None = None,
    repo_root: Path | None = None,
    filename: str = PERF_BLOB_FILENAME,
) -> Path | None:
    """Write the PERF-BLOB section through the WP1 path guard, under RUN_DIR.

    The destination is the guard-checked report path when one is declared and
    otherwise a file under the guard-checked RUN_DIR; there is no third
    destination, so a PERF-BLOB record can never land inside the repository. A
    section with no record is emitted nowhere and returns NULL. The module
    spawns no process and opens no socket, so it carries no egress path.
    """
    validate_perf_blob_section(section)
    if not section["records"]:
        return None
    name = Path(filename)
    if name.name != filename or name.suffix != ".json":
        raise ThresholdError(
            "filename_invalid", f"{filename!r} is not a plain .json name"
        )
    try:
        paths = resolve_perf_paths(
            environ, scratchpad_root=scratchpad_root, repo_root=repo_root
        )
    except PerfPathError as error:
        raise ThresholdError(
            "report_path_rejected", f"{error.reason}: {error.detail}"
        ) from error
    if paths.report is not None:
        target = paths.report
    elif paths.run_dir is not None:
        target = paths.run_dir / filename
    else:
        raise ThresholdError(
            "report_path_unset",
            "neither a report path nor a RUN_DIR is declared: the record is"
            " written to a guarded path or nowhere",
        )
    try:
        text = json.dumps(section, indent=2, sort_keys=True, allow_nan=False)
    except ValueError as error:
        raise ThresholdError("report_not_json_safe", str(error)) from error
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{text}\n", encoding="utf-8")
    return target
