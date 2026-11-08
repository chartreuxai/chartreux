"""Test-side render-flush observer for the D1 cell (v0.7.5 stage 2, WP2).

D1 gates the wall duration of every non-empty streaming render flush on the
Pilot path (campaign ``D1``, cell ``pilot_streaming_fresh``). The observer is
reimplemented here rather than imported from production: it times the real
flush boundary of ``StreamingMessageBase._flush_write_buffer``, so the
instrumentation adds fields and never changes control flow.

The run record carries the D1 floor's chunks-coverage fields (completed turns,
discarded warmup flushes, measured non-empty flushes) and reports p50
separately from the gated nearest-rank p95.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from time import perf_counter_ns
from typing import Any

import pytest

from chartreux.cli.textual_ui.widgets.messages import StreamingMessageBase

D1_CAMPAIGN_ID = "D1"
D1_CELL = "pilot_streaming_fresh"
D1_METRIC = "render_flush_p95_ms"
D1_METRIC_UNIT = "ms"
# The run record's metric version (Appendix B run eligibility) — distinct from
# the determinants document's own schema field.
D1_METRIC_VERSION = "chartreux.runtime_perf.v1"
D1_COVERAGE_UNIT = "render flushes"
# D1 floor, pinned by the determinants row: >= 100 completed turns AND >= 100
# non-empty render flushes per run after >= 10 warmup flushes that are
# discarded and never counted.
D1_WARMUP_FLUSHES = 10
D1_MIN_COMPLETED_TURNS = 100
D1_MIN_MEASURED_FLUSHES = 100

_P50_FRACTION = 0.50
_P95_FRACTION = 0.95
_HISTOGRAM_BUCKETS_MS: tuple[float, ...] = (
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    1000.0,
)
_SAMPLE_PRECISION = 4


def nearest_rank(samples: Sequence[float], *, fraction: float) -> float:
    """Return the nearest-rank percentile: the k-th smallest with k = ceil(f*n).

    Nearest rank (not interpolation) is the determinants' p95 definition, so a
    handful of slow flushes moves the recorded value onto a real sample.
    """
    if not samples:
        raise ValueError("nearest_rank requires at least one sample")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(samples)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def histogram_ms(samples: Sequence[float]) -> dict[str, int]:
    """Bucket flush durations (ms) into the run record's histogram."""
    labels = [f"<={bucket:g}" for bucket in _HISTOGRAM_BUCKETS_MS]
    overflow_label = f">{_HISTOGRAM_BUCKETS_MS[-1]:g}"
    counts = dict.fromkeys([*labels, overflow_label], 0)
    for sample in samples:
        for bucket, label in zip(_HISTOGRAM_BUCKETS_MS, labels, strict=True):
            if sample <= bucket:
                counts[label] += 1
                break
        else:
            counts[overflow_label] += 1
    return counts


@dataclass(slots=True)
class RenderFlushLog:
    """One run's flush calls, split into discarded warmup and measured windows."""

    warmup_ms: list[float] = field(default_factory=list)
    measured_ms: list[float] = field(default_factory=list)
    empty_flushes: int = 0
    skipped_flushes: int = 0
    warmup_required: int = D1_WARMUP_FLUSHES

    @property
    def non_empty_flushes(self) -> int:
        """Count every flush that rendered content, warmup included."""
        return len(self.warmup_ms) + len(self.measured_ms)


@dataclass(frozen=True, slots=True)
class FloorVerdict:
    """D1 floor outcome, with one reason code per unmet requirement."""

    eligible: bool
    reasons: tuple[str, ...]


def evaluate_d1_floor(*, completed_turns: int, measured_flushes: int) -> FloorVerdict:
    """Apply the D1 floor: turns AND measured non-empty flushes, both inclusive."""
    reasons: list[str] = []
    if completed_turns < D1_MIN_COMPLETED_TURNS:
        reasons.append("ops_floor_completed_turns")
    if measured_flushes < D1_MIN_MEASURED_FLUSHES:
        reasons.append("ops_floor_measured_flushes")
    return FloorVerdict(eligible=not reasons, reasons=tuple(reasons))


class RenderTimingObserver:
    """Times every non-empty streaming render flush at the widget flush boundary."""

    def __init__(self, *, warmup_flushes: int = D1_WARMUP_FLUSHES) -> None:
        if warmup_flushes < 0:
            raise ValueError("warmup_flushes must be non-negative")
        self.warmup_flushes = warmup_flushes
        self.log = RenderFlushLog(warmup_required=warmup_flushes)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hook the real flush boundary; instrumentation only, no flow change."""
        original_flush = StreamingMessageBase._flush_write_buffer

        async def timed_flush(message: StreamingMessageBase) -> None:
            pending = message._to_write_buffer
            started_ns = perf_counter_ns()
            try:
                await original_flush(message)
            finally:
                self.record_flush(
                    pending=pending,
                    remaining=message._to_write_buffer,
                    elapsed_ms=(perf_counter_ns() - started_ns) / 1_000_000,
                )

        monkeypatch.setattr(StreamingMessageBase, "_flush_write_buffer", timed_flush)

    def record_flush(self, *, pending: str, remaining: str, elapsed_ms: float) -> None:
        """Classify one flush call: rendered content, skipped, or empty.

        A call that renders nothing is never a D1 sample: an empty write buffer
        is counted separately, and buffered content the widget declined to
        render (collapsed reasoning, scrolled away) as skipped.
        """
        if not pending:
            self.log.empty_flushes += 1
            return
        if remaining:
            self.log.skipped_flushes += 1
            return
        if len(self.log.warmup_ms) < self.warmup_flushes:
            self.log.warmup_ms.append(elapsed_ms)
        else:
            self.log.measured_ms.append(elapsed_ms)


def build_run_record(
    log: RenderFlushLog, *, completed_turns: int, fixture: str | None = None
) -> dict[str, Any]:
    """Build one D1 run record, chunks coverage and floor verdict included.

    A record with no measured flush carries NULL statistics with its floor
    reason rather than a fabricated value (the determinants' nullable-fields
    rule); p50 is always reported separately from the gated p95.
    """
    measured = log.measured_ms
    verdict = evaluate_d1_floor(
        completed_turns=completed_turns, measured_flushes=len(measured)
    )
    samples = [round(sample, _SAMPLE_PRECISION) for sample in measured]
    run_record: dict[str, Any] = {
        "campaign_id": D1_CAMPAIGN_ID,
        "metric": D1_METRIC,
        "unit": D1_METRIC_UNIT,
        "metric_version": D1_METRIC_VERSION,
        "cell": D1_CELL,
        "chunks_coverage": {
            "unit": D1_COVERAGE_UNIT,
            "completed_turns": completed_turns,
            "warmup_flushes": len(log.warmup_ms),
            "measured_flushes": len(measured),
            "empty_flushes": log.empty_flushes,
            "skipped_flushes": log.skipped_flushes,
        },
        "warmup_flushes_required": log.warmup_required,
        "samples_ms": samples,
        "histogram_ms": histogram_ms(samples),
        "eligibility": {"eligible": verdict.eligible, "reasons": list(verdict.reasons)},
    }
    if fixture is not None:
        run_record["fixture"] = fixture
    if samples:
        run_record["render_flush_p50_ms"] = nearest_rank(
            samples, fraction=_P50_FRACTION
        )
        run_record["render_flush_p95_ms"] = nearest_rank(
            samples, fraction=_P95_FRACTION
        )
        run_record["render_flush_max_ms"] = max(samples)
    else:
        run_record["render_flush_p50_ms"] = None
        run_record["render_flush_p95_ms"] = None
        run_record["render_flush_max_ms"] = None
    return run_record
