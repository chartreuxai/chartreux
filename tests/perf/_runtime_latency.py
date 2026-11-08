"""End-to-end turn latency on both adapter paths (v0.7.5 stage 2, X1).

The gated rows this module produces are A1-A4 (fresh headless canonical cell,
``x1_fresh_headless_fb10_c4``) and B1-B2 (long-history loaded headless cell,
``x2_loaded_headless_8x200k``). The canonical cell is the committed one -- 10 ms
first byte plus four 5 ms chunk gaps, 30 ms injected per turn -- and it is the
only matrix point that may produce gated row records: every other
:data:`MATRIX_POINTS` entry is observation-only and carries its recorded reason,
and the Pilot path's reports are observation-only by plan §4.

The injected-vs-application split is the A1/B1 definition: the raw turn wall
(user-message submission -> model-visible completion) is recorded beside the
turn's EXACT injected delay, and the gated series is the residual. A negative
residual is an invalid sample for that turn -- it is kept and recorded as
invalid, excluded from the gated series and never clamped to zero.

A2/A3 are co-measured in the same run. The transport boundary observed here is
the streaming backend's own delivery: the request start is the stream's first
resume, which is the same visit the gateway arms its request-start observer on,
so TTFB and inter-chunk gaps include the injected delay by construction and are
gated as observed (the cell is delay-dominated by construction -- recorded as a
property, not a defect). A turn's own stream is the LAST one started inside its
window; a background child-completion turn may start an earlier one, which is
recorded as a foreign request and never folded into the turn's samples.

A4/B2 come from a 1 ms scheduled loop heartbeat logged throughout the run -- armed
before the scenario runs and stopped during cleanup -- and each row is the observed
interval's excess over the scheduled interval, with the first
:data:`HEARTBEAT_WARMUP_ROWS` rows discarded.

The Pilot path cannot inject transport delay through the headless backend, so
:class:`PilotLatencyRun` owns the chunk arrival schedule itself -- which is
exactly what the injected-delay matrix models -- and renders each chunk through
the real production flush boundary of the WP4 application. Its reports say
``gated: False`` with the recorded reason.

P2-memo-aware fidelity: every cell run records the request context the backend
actually received (memo presence and message identity) and the memo identity
probe, so a latency run is never taken while the gateway re-prepares a request
it should have served from the exact-snapshot memo.

Nothing here changes production behaviour: observers are installed on instances
or as tasks and removed on exit, and the workload definitions are WP3b's.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import functools
import os
from time import perf_counter_ns
from typing import Any

from textual.containers import Vertical
from textual.pilot import Pilot

from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from tests.perf._metrics import machine_context
from tests.perf._runtime_pilot import (
    PILOT_ADAPTER,
    PILOT_FRAME_DELAY_S,
    PILOT_HISTORY_CONTAINER_ID,
    PILOT_PARITY_SCALE,
    PilotScenarioApp,
    _settle_flush,
)
from tests.perf._runtime_report import MODE_MEASURE, build_run_record, nearest_rank
from tests.perf._runtime_scenario import (
    CELL_FRESH_HEADLESS,
    CELL_LOADED_HEADLESS,
    INJECTED_CHUNK_COUNT,
    INJECTED_CHUNK_GAP_MS,
    INJECTED_FIRST_BYTE_MS,
    SCENARIO_METRIC_VERSION,
    SHORTENED_SCALE,
    HeadlessScenario,
    ScenarioScale,
    build_headless_scenario,
    response_chunks,
    root_user_content,
    workload_spec,
)

LATENCY_SCHEMA = "chartreux.runtime_latency.v1"
PILOT_MATRIX_SCHEMA = "chartreux.runtime_latency_pilot_matrix.v1"

#: X1's canonical gated matrix point: 10 ms first byte, four 5 ms chunk gaps.
CANONICAL_POINT_KEY = "fb10_c4x5"
#: Recorded reason of every non-canonical matrix point (plan §4).
OBSERVATION_ONLY_REASON = "non_canonical_matrix_point_observation_only"
#: Recorded reason of the loaded cell's own no-injection definition.
LOADED_NO_INJECTION_REASON = "loaded_cell_carries_no_injected_delay"
#: Recorded reason of every Pilot-path report (plan §4).
PILOT_OBSERVATION_ONLY_REASON = "pilot_path_latency_observation_only"

#: The X1 heartbeat cadence and its discarded warmup rows (determinants A4/B2).
HEARTBEAT_INTERVAL_MS = 1.0
HEARTBEAT_WARMUP_ROWS = 10
#: Contract-scale warmup turns; full-scale cells use the committed spec's 10.
CONTRACT_WARMUP_TURNS = 1
#: The determinants' invalid-sample cap: > 10 % invalid samples = ineligible.
INVALID_SAMPLE_CAP = 0.10

#: The gated rows each headless cell produces, in registry order.
GATED_ROWS_BY_CELL: dict[str, tuple[str, ...]] = {
    CELL_FRESH_HEADLESS: ("A1", "A2", "A3", "A4"),
    CELL_LOADED_HEADLESS: ("B1", "B2"),
}
TURN_ROWS: tuple[str, ...] = ("A1", "B1")
TTFB_ROW = "A2"
GAP_ROW = "A3"
HEARTBEAT_ROWS: tuple[str, ...] = ("A4", "B2")

_P95_FRACTION = 0.95
_P50_FRACTION = 0.5
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_SAMPLE_PRECISION = 4


def run_start_utc() -> str:
    """Return the run record's UTC start stamp."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _milliseconds(nanoseconds: int) -> float:
    return nanoseconds / _NANOSECONDS_PER_MILLISECOND


def _rounded(values: list[float]) -> list[float]:
    return [round(float(value), _SAMPLE_PRECISION) for value in values]


# --------------------------------------------------------------------------
# The injected-delay matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InjectedDelayPoint:
    """One injected-delay matrix point: the delays a turn's transport carries.

    ``canonical`` marks the single gated cell of the matrix (fb=10 ms plus four
    5 ms chunk gaps). Every other point is observation-only and records why.
    """

    key: str
    first_byte_ms: float
    chunk_gap_ms: float
    chunk_count: int
    canonical: bool
    reason: str | None

    @property
    def total_delay_ms(self) -> float:
        """The exact total injected delay of one turn of this point."""
        return self.first_byte_ms + self.chunk_count * self.chunk_gap_ms

    def record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "first_byte_ms": self.first_byte_ms,
            "chunk_gap_ms": self.chunk_gap_ms,
            "chunk_count": self.chunk_count,
            "total_delay_ms": self.total_delay_ms,
            "canonical": self.canonical,
            "reason": self.reason,
        }


CANONICAL_POINT = InjectedDelayPoint(
    key=CANONICAL_POINT_KEY,
    first_byte_ms=INJECTED_FIRST_BYTE_MS,
    chunk_gap_ms=INJECTED_CHUNK_GAP_MS,
    chunk_count=INJECTED_CHUNK_COUNT,
    canonical=True,
    reason=None,
)

#: The fresh cell's injected-delay matrix. Exactly one point is canonical.
MATRIX_POINTS: tuple[InjectedDelayPoint, ...] = (
    CANONICAL_POINT,
    InjectedDelayPoint(
        key="fb0_c4x0",
        first_byte_ms=0.0,
        chunk_gap_ms=0.0,
        chunk_count=INJECTED_CHUNK_COUNT,
        canonical=False,
        reason=OBSERVATION_ONLY_REASON,
    ),
    InjectedDelayPoint(
        key="fb25_c4x10",
        first_byte_ms=25.0,
        chunk_gap_ms=10.0,
        chunk_count=INJECTED_CHUNK_COUNT,
        canonical=False,
        reason=OBSERVATION_ONLY_REASON,
    ),
    InjectedDelayPoint(
        key="fb2_c4x20",
        first_byte_ms=2.0,
        chunk_gap_ms=20.0,
        chunk_count=INJECTED_CHUNK_COUNT,
        canonical=False,
        reason=OBSERVATION_ONLY_REASON,
    ),
)

#: The loaded cell carries no injected delay: the load itself is the fixture.
LOADED_CELL_POINT = InjectedDelayPoint(
    key="loaded_no_injection",
    first_byte_ms=0.0,
    chunk_gap_ms=0.0,
    chunk_count=INJECTED_CHUNK_COUNT,
    canonical=False,
    reason=LOADED_NO_INJECTION_REASON,
)

GATED_POINT_BY_CELL: dict[str, InjectedDelayPoint] = {
    CELL_FRESH_HEADLESS: CANONICAL_POINT,
    CELL_LOADED_HEADLESS: LOADED_CELL_POINT,
}


def matrix_point(key: str) -> InjectedDelayPoint:
    """Return the matrix point with ``key``, raising for an unknown key."""
    for point in MATRIX_POINTS:
        if point.key == key:
            return point
    raise ValueError(f"no injected-delay matrix point: {key}")


def gated_point(cell: str) -> InjectedDelayPoint:
    """Return the point that may produce ``cell``'s gated row records."""
    point = GATED_POINT_BY_CELL.get(cell)
    if point is None:
        raise ValueError(f"no gated latency point for cell: {cell}")
    return point


def observation_only_points() -> tuple[InjectedDelayPoint, ...]:
    """Return the matrix points that may never produce gated row records."""
    return tuple(point for point in MATRIX_POINTS if not point.canonical)


# --------------------------------------------------------------------------
# Transport observation (A2/A3)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StreamCallTiming:
    """One streaming backend call: request start, first chunk, chunk arrivals.

    ``started_ns`` is stamped at the stream's first resume -- the same visit the
    gateway arms its request-start observer on -- so TTFB spans request start to
    the first delivered chunk, injected delay included.
    """

    call_index: int
    started_ns: int
    first_chunk_ns: int | None = None
    finished_ns: int | None = None
    chunk_times_ns: list[int] = field(default_factory=list)
    injected_ms: float | None = None

    @property
    def ttfb_ms(self) -> float | None:
        if self.first_chunk_ns is None:
            return None
        return _milliseconds(self.first_chunk_ns - self.started_ns)

    @property
    def chunks(self) -> int:
        return len(self.chunk_times_ns)

    @property
    def gaps_ms(self) -> list[float]:
        return [
            _milliseconds(following - leading)
            for leading, following in zip(
                self.chunk_times_ns, self.chunk_times_ns[1:], strict=False
            )
        ]

    def record(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "ttfb_ms": None if self.ttfb_ms is None else round(self.ttfb_ms, 4),
            "chunks": self.chunks,
            "gaps_ms": _rounded(self.gaps_ms),
            "injected_ms": self.injected_ms,
        }


class StreamingChunkObserver:
    """Times the delivery boundary of one backend instance's streams.

    Installed on the instance (never the class) and removed on exit, so the
    gateway's own request accounting, ordering and retry behaviour are
    untouched: the wrapper delegates and re-raises whatever the stream does.
    """

    def __init__(self) -> None:
        self.calls: list[StreamCallTiming] = []
        self._restorers: list[Callable[[], None]] = []

    def install(
        self, backend: Any, *, injected_log: list[float] | None = None
    ) -> StreamingChunkObserver:
        """Wrap ``backend.complete_streaming`` and start recording calls."""
        original = backend.complete_streaming
        instance_attributes = getattr(backend, "__dict__", {})
        had_own_attribute = (
            isinstance(instance_attributes, dict)
            and "complete_streaming" in instance_attributes
        )
        observer = self

        @functools.wraps(original)
        async def observed(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            timing = StreamCallTiming(
                call_index=len(observer.calls), started_ns=perf_counter_ns()
            )
            observer.calls.append(timing)
            injected_index = 0 if injected_log is None else len(injected_log)
            try:
                async for chunk in original(*args, **kwargs):
                    now = perf_counter_ns()
                    if timing.first_chunk_ns is None:
                        timing.first_chunk_ns = now
                    timing.chunk_times_ns.append(now)
                    yield chunk
            finally:
                timing.finished_ns = perf_counter_ns()
                if injected_log is not None and injected_index < len(injected_log):
                    timing.injected_ms = injected_log[injected_index]

        backend.complete_streaming = observed
        self._restorers.append(
            functools.partial(
                _restore_attribute,
                backend,
                "complete_streaming",
                original,
                had_own_attribute,
            )
        )
        return self

    def restore(self) -> None:
        """Undo every installation, most recent first."""
        while self._restorers:
            self._restorers.pop()()


def _restore_attribute(
    owner: Any, attribute: str, original: Any, had_own_attribute: bool
) -> None:
    if had_own_attribute:
        setattr(owner, attribute, original)
        return
    try:
        delattr(owner, attribute)
    except AttributeError:  # pragma: no cover - the attribute was ours
        pass


# --------------------------------------------------------------------------
# Loop heartbeat (A4/B2)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class HeartbeatLog:
    """One run's heartbeat rows: the signed excess over the scheduled interval."""

    interval_ms: float
    warmup_rows: int
    rows_ms: list[float] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return len(self.rows_ms)

    @property
    def measured_rows(self) -> int:
        return max(0, len(self.rows_ms) - self.warmup_rows)

    def measured_ms(self) -> list[float]:
        return list(self.rows_ms[self.warmup_rows :])

    def record(self) -> dict[str, Any]:
        return {
            "interval_ms": self.interval_ms,
            "rows": self.total_rows,
            "warmup_rows": min(self.warmup_rows, self.total_rows),
            "measured_rows": self.measured_rows,
        }


class LoopHeartbeatObserver:
    """Logs a scheduled loop heartbeat and each row's scheduling excess.

    One row is the wake's lateness against its scheduled deadline, which is the
    observed interval's excess over the scheduled 1 ms cadence, so a congested
    event loop produces larger rows. The deadline is re-armed each iteration
    (never carried across overshoots) so a late frame is recorded once instead
    of being refunded as a burst of sub-interval rows. The task is started and
    stopped around the measured workload and never left running.
    """

    def __init__(
        self,
        *,
        interval_ms: float = HEARTBEAT_INTERVAL_MS,
        warmup_rows: int = HEARTBEAT_WARMUP_ROWS,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if warmup_rows < 0:
            raise ValueError("warmup_rows must be non-negative")
        self.log = HeartbeatLog(interval_ms=interval_ms, warmup_rows=warmup_rows)
        self._task: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None

    async def start(self) -> LoopHeartbeatObserver:
        """Start the heartbeat task on the running loop."""
        if self._task is not None:
            raise RuntimeError("the heartbeat observer is already running")
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._beat(), name="perf-latency-heartbeat")
        return self

    async def stop(self) -> None:
        """Stop the heartbeat task and join it."""
        stop, task = self._stop, self._task
        self._task = None
        self._stop = None
        if stop is not None:
            stop.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def _beat(self) -> None:
        loop = asyncio.get_running_loop()
        interval_seconds = self.log.interval_ms / 1000
        stop = self._stop
        if stop is None:  # pragma: no cover - start() always arms it
            return
        while not stop.is_set():
            expected_wake = loop.time() + interval_seconds
            await asyncio.sleep(interval_seconds)
            excess_ms = (loop.time() - expected_wake) * 1000
            self.log.rows_ms.append(excess_ms)


# --------------------------------------------------------------------------
# The latency log of one cell run
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TurnLatencySample:
    """One measured root turn: raw wall, injected delay, transport samples."""

    turn: int
    raw_ms: float
    injected_ms: float
    history_messages: int
    requests: int
    chunks: int
    ttfb_ms: float | None
    chunk_gaps_ms: tuple[float, ...]

    @property
    def residual_ms(self) -> float:
        return self.raw_ms - self.injected_ms

    @property
    def valid(self) -> bool:
        return self.residual_ms >= 0.0

    @property
    def foreign_requests(self) -> int:
        return max(0, self.requests - 1)

    def record(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "raw_ms": round(self.raw_ms, _SAMPLE_PRECISION),
            "injected_delay_ms": round(self.injected_ms, _SAMPLE_PRECISION),
            "residual_ms": round(self.residual_ms, _SAMPLE_PRECISION),
            "valid": self.valid,
            "requests": self.requests,
            "foreign_requests": self.foreign_requests,
            "chunks": self.chunks,
            "ttfb_ms": None if self.ttfb_ms is None else round(self.ttfb_ms, 4),
            "chunk_gaps_ms": _rounded(list(self.chunk_gaps_ms)),
            "history_messages": self.history_messages,
        }


@dataclass(slots=True)
class LatencyLog:
    """One latency run's samples: turns, transport chunks and heartbeat rows."""

    cell: str
    point: InjectedDelayPoint
    warmup_turns: int
    scale: dict[str, Any]
    turns: list[TurnLatencySample] = field(default_factory=list)
    heartbeat: HeartbeatLog | None = None
    child_bursts: int = 0
    child_turns: int = 0
    request_context: dict[str, Any] | None = None
    memo_probe: dict[str, Any] | None = None
    workload_hash: str | None = None
    fixture_hash: str | None = None
    machine: dict[str, Any] | None = None
    quiet: dict[str, Any] | None = None

    # -- measured windows --------------------------------------------------

    def measured_turns(self) -> list[TurnLatencySample]:
        return [turn for turn in self.turns if turn.turn >= self.warmup_turns]

    def measured_raw(self) -> list[float]:
        return [turn.raw_ms for turn in self.measured_turns()]

    def valid_residuals(self) -> list[float]:
        return [turn.residual_ms for turn in self.measured_turns() if turn.valid]

    def invalid_residuals(self) -> list[float]:
        return [turn.residual_ms for turn in self.measured_turns() if not turn.valid]

    def measured_ttfb(self) -> list[float]:
        samples: list[float] = []
        for turn in self.measured_turns():
            if turn.ttfb_ms is None:
                raise AssertionError(f"turn {turn.turn} recorded no request start")
            samples.append(turn.ttfb_ms)
        return samples

    def measured_gaps(self) -> list[float]:
        gaps: list[float] = []
        for turn in self.measured_turns():
            gaps.extend(turn.chunk_gaps_ms)
        return gaps

    def measured_heartbeat(self) -> list[float]:
        return [] if self.heartbeat is None else self.heartbeat.measured_ms()

    def foreign_requests(self) -> int:
        return sum(turn.foreign_requests for turn in self.measured_turns())

    # -- records -----------------------------------------------------------

    def chunks_coverage(self) -> dict[str, Any]:
        """Return the run's turns / chunk-gap / heartbeat-row coverage (§3)."""
        measured = self.measured_turns()
        gaps = len(self.measured_gaps())
        heartbeat_rows = 0 if self.heartbeat is None else self.heartbeat.measured_rows
        missing: list[str] = []
        if not measured:
            missing.append("turns")
        if not gaps:
            missing.append("chunk_gaps")
        if not heartbeat_rows:
            missing.append("heartbeat_rows")
        return {
            "unit": "turns",
            "turns": len(measured),
            "warmup_turns": len(self.turns) - len(measured),
            "chunk_gaps": gaps,
            "heartbeat_rows": heartbeat_rows,
            "heartbeat_interval_ms": HEARTBEAT_INTERVAL_MS,
            "heartbeat_warmup_rows": (
                0
                if self.heartbeat is None
                else min(self.heartbeat.warmup_rows, self.heartbeat.total_rows)
            ),
            "requests": sum(turn.requests for turn in measured),
            "foreign_requests": self.foreign_requests(),
            "child_bursts": self.child_bursts,
            "child_turns": self.child_turns,
            "complete": not missing,
            "missing": missing,
        }

    def summary(self) -> dict[str, Any]:
        """Summarize every measured series; an empty series stays NULL."""
        return {
            "raw_p95_ms": _p95_or_none(self.measured_raw()),
            "residual_p95_ms": _p95_or_none(self.valid_residuals()),
            "ttfb_p95_ms": _p95_or_none(self.measured_ttfb()),
            "chunk_gap_p95_ms": _p95_or_none(self.measured_gaps()),
            "heartbeat_excess_p95_ms": _p95_or_none(self.measured_heartbeat()),
            "heartbeat_excess_p50_ms": _p50_or_none(self.measured_heartbeat()),
            "measured_turns": len(self.measured_turns()),
            "invalid_samples": len(self.invalid_residuals()),
        }

    def observation_report(self) -> dict[str, Any]:
        """Return the run's observation report (gated or observation-only)."""
        return {
            "schema": LATENCY_SCHEMA,
            "metric_version": SCENARIO_METRIC_VERSION,
            "adapter": "headless",
            "cell": self.cell,
            "gated": self.point == gated_point(self.cell),
            "reason": (
                None
                if self.point == gated_point(self.cell)
                else (self.point.reason or OBSERVATION_ONLY_REASON)
            ),
            "injected_delay_point": self.point.record(),
            "scale": self.scale,
            "warmup_turns": self.warmup_turns,
            "chunks_coverage": self.chunks_coverage(),
            "summary": self.summary(),
            "turns": [turn.record() for turn in self.turns],
            "request_context": self.request_context,
            "memo_probe": self.memo_probe,
            "quiet": self.quiet,
        }


def _p95_or_none(samples: list[float]) -> float | None:
    if not samples:
        return None
    return round(nearest_rank(samples, fraction=_P95_FRACTION), _SAMPLE_PRECISION)


def _p50_or_none(samples: list[float]) -> float | None:
    if not samples:
        return None
    return round(nearest_rank(samples, fraction=_P50_FRACTION), _SAMPLE_PRECISION)


# --------------------------------------------------------------------------
# Environment context
# --------------------------------------------------------------------------


def _quiet_stamp() -> dict[str, Any]:
    """Return one loadavg sample and its core-count-relative bound result."""
    cores = os.cpu_count() or 1
    bound = 0.5 * cores
    try:
        loadavg = os.getloadavg()[0]
    except OSError:  # pragma: no cover - Linux always reports it
        return {"loadavg_1m": None, "core_count": cores, "bound": bound, "quiet": False}
    return {
        "loadavg_1m": round(loadavg, 3),
        "core_count": cores,
        "bound": bound,
        "quiet": loadavg <= bound,
    }


def _quiet_window(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "before": before,
        "after": after,
        "quiet": bool(before["quiet"] and after["quiet"]),
        "rule": "loadavg_1m <= 0.5 * core_count at both bracketing stamps",
    }


def _request_context(scenario: HeadlessScenario) -> dict[str, Any]:
    """Report the request context the backend actually received (P2 memo)."""
    context = scenario.backend_request_context()
    return {
        "requests": context.request_count,
        "messages": context.message_count,
        "context_chars": context.context_chars,
        "memo_present": context.memo_present,
        "memo_messages": context.memo_message_count,
        "memo_matches_request": context.memo_matches_request,
    }


# --------------------------------------------------------------------------
# The headless latency cell
# --------------------------------------------------------------------------


class HeadlessLatencyCell:
    """One headless latency run over one injected-delay point.

    The cell mutates only its own backend instance's injection parameters (they
    are the matrix), drives the committed burst/turn schedule itself so each
    turn's window has a measurable transport boundary, and reports the exact
    per-turn injected delay the backend recorded next to the raw wall.
    """

    def __init__(
        self,
        *,
        cell: str = CELL_FRESH_HEADLESS,
        point: InjectedDelayPoint | None = None,
        scale: ScenarioScale | None = None,
        warmup_turns: int = CONTRACT_WARMUP_TURNS,
        closure_note: str = "campaign open: run collected, selection pending",
        run_id: str | None = None,
    ) -> None:
        if warmup_turns < 0:
            raise ValueError("warmup_turns must be non-negative")
        self.cell = cell
        self.point = gated_point(cell) if point is None else point
        self.scale = SHORTENED_SCALE if scale is None else scale
        self.warmup_turns = warmup_turns
        self.closure_note = closure_note
        self.run_id = run_id or f"{cell}-{self.point.key}"
        self.run_start_utc = run_start_utc()
        self.log: LatencyLog | None = None

    @property
    def gated(self) -> bool:
        """Whether this run may produce the cell's gated row records."""
        return self.point == gated_point(self.cell)

    async def run(self) -> LatencyLog:
        """Drive the workload and return the run's latency log."""
        if self.log is not None:
            raise RuntimeError("a latency cell drives one workload")
        quiet_before = _quiet_stamp()
        log = LatencyLog(
            cell=self.cell,
            point=self.point,
            warmup_turns=self.warmup_turns,
            scale=self.scale.dimensions(),
        )
        observer = StreamingChunkObserver()
        # The A4/B2 heartbeat is logged throughout the run, not only inside the
        # turn loop (Appendix B A4: "throughout each run"), so it is armed before
        # the scenario is built and disarmed after the last observed activity.
        heartbeat = LoopHeartbeatObserver()
        await heartbeat.start()
        scenario: HeadlessScenario | None = None
        original_injection: tuple[float, float] | None = None
        try:
            scenario = await build_headless_scenario(self.cell, scale=self.scale)
            backend: Any = scenario._backend
            original_injection = (backend.first_byte_ms, backend.chunk_gap_ms)
            backend.first_byte_ms = self.point.first_byte_ms
            backend.chunk_gap_ms = self.point.chunk_gap_ms
            observer.install(backend, injected_log=backend.injected_ms)
            await self._drive(scenario, observer, log)
            log.request_context = _request_context(scenario)
            probe = scenario.prepared_messages_identity_probe()
            log.memo_probe = {
                "first": probe[0],
                "hit": probe[1],
                "mutated": probe[2],
                "identity_hit": probe[0] == probe[1],
                "live_reread_miss": probe[2] != probe[0],
            }
            log.workload_hash = scenario.spec.digest()
            log.fixture_hash = scenario.spec.fixture_hash
        finally:
            await heartbeat.stop()
            observer.restore()
            if scenario is not None:
                if original_injection is not None:
                    backend = scenario._backend
                    backend.first_byte_ms, backend.chunk_gap_ms = original_injection
                await scenario.close()
        log.heartbeat = heartbeat.log
        log.quiet = _quiet_window(quiet_before, _quiet_stamp())
        log.machine = machine_context()
        self.log = log
        return log

    async def _drive(
        self,
        scenario: HeadlessScenario,
        observer: StreamingChunkObserver,
        log: LatencyLog,
    ) -> None:
        """Run the committed schedule, observing each turn's own transport."""
        for index in range(self.scale.root_turns):
            if (
                self.scale.child_agents
                and index
                and index % self.scale.burst_interval_turns == 0
            ):
                burst = await scenario.run_child_burst(
                    index // self.scale.burst_interval_turns - 1
                )
                log.child_bursts += 1
                log.child_turns += burst.completed
            started = len(observer.calls)
            result = await scenario.run_root_turn(index)
            window_calls = observer.calls[started:]
            if not window_calls:
                raise AssertionError(f"turn {index} started no observed stream")
            # A background child-completion turn may start an earlier stream in
            # this window; the turn's OWN request is the last one started.
            own = window_calls[-1]
            if own.injected_ms is None:
                raise AssertionError(f"turn {index} recorded no injected delay")
            if own.injected_ms != self.point.total_delay_ms:
                raise AssertionError(
                    f"turn {index} carried {own.injected_ms} ms injected,"
                    f" the point commits {self.point.total_delay_ms} ms"
                )
            log.turns.append(
                TurnLatencySample(
                    turn=result.turn,
                    raw_ms=result.wall_ms,
                    injected_ms=own.injected_ms,
                    history_messages=result.history_messages,
                    requests=len(window_calls),
                    chunks=own.chunks,
                    ttfb_ms=own.ttfb_ms,
                    chunk_gaps_ms=tuple(own.gaps_ms),
                )
            )

    # -- records -----------------------------------------------------------

    def require_log(self) -> LatencyLog:
        if self.log is None:
            raise RuntimeError("the latency cell has not run yet")
        return self.log

    def observation_report(self) -> dict[str, Any]:
        return self.require_log().observation_report()

    def build_records(self) -> dict[str, dict[str, Any]]:
        """Build and validate this run's gated row records."""
        log = self.require_log()
        if not self.gated:
            raise AssertionError(
                f"{self.point.key} is observation-only: it may produce no gated"
                f" row records for {self.cell}"
            )
        return build_cell_records(
            log,
            run_id=self.run_id,
            run_start_utc=self.run_start_utc,
            closure_note=self.closure_note,
        )


def build_cell_records(
    log: LatencyLog, *, run_id: str, run_start_utc: str, closure_note: str
) -> dict[str, dict[str, Any]]:
    """Build the cell's gated row records from one latency log."""
    if log.point != gated_point(log.cell):
        raise AssertionError(
            f"{log.point.key} is not {log.cell}'s gated configuration:"
            " a non-canonical point produces no gated row records"
        )
    coverage = log.chunks_coverage()
    counts = {
        "turns": coverage["turns"],
        "chunk_gaps": coverage["chunk_gaps"],
        "heartbeat_rows": coverage["heartbeat_rows"],
    }
    common: dict[str, Any] = {
        "run_id": run_id,
        "run_start_utc": run_start_utc,
        "mode": MODE_MEASURE,
        "unit_counts": counts,
        "closure_status": None,
        "closure_note": closure_note,
        "fixture_hash": log.fixture_hash,
        "machine": log.machine,
    }
    rows = GATED_ROWS_BY_CELL[log.cell]
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        record = _row_record(log, row, common)
        records[row] = record
    return records


def _row_record(log: LatencyLog, row: str, common: dict[str, Any]) -> dict[str, Any]:
    if row in TURN_ROWS:
        record = _turn_row_record(log, row, common)
    elif row == TTFB_ROW:
        record = _ttfb_row_record(log, row, common)
    elif row == GAP_ROW:
        record = _gap_row_record(log, row, common)
    elif row in HEARTBEAT_ROWS:
        record = _heartbeat_row_record(log, row, common)
    else:  # pragma: no cover - the row table is closed
        raise AssertionError(f"no latency producer for campaign id {row}")
    record.update(_record_extras(log))
    record["cell_eligibility"] = _cell_eligibility(log, record)
    return record


def _turn_row_record(
    log: LatencyLog, row: str, common: dict[str, Any]
) -> dict[str, Any]:
    """A1/B1: the gated series is the residual, raw and injected kept apart."""
    measured = log.measured_turns()
    raw = [turn.raw_ms for turn in measured]
    injected = [turn.injected_ms for turn in measured]
    valid = log.valid_residuals()
    invalid = log.invalid_residuals()
    record = build_run_record(
        row,
        measured_ops=len(valid),
        warmup_ops=log.warmup_turns,
        samples=valid,
        **common,
    )
    record["gated_series"] = "residual_ms"
    record["raw_p95_ms"] = _p95_or_none(raw)
    record["residual_p95_ms"] = record["summary"]["p95"]
    record["injected"] = {
        "subtracted": True,
        "total_delay_ms": log.point.total_delay_ms,
        "samples_ms": _rounded(injected),
        "p95_ms": _p95_or_none(injected),
    }
    record["residual"] = {
        "valid_samples_ms": _rounded(valid),
        "invalid_samples_ms": _rounded(invalid),
        "invalid_rule": "negative_corrected_sample_invalid_not_clamped",
        "clamped": False,
    }
    return record


def _ttfb_row_record(
    log: LatencyLog, row: str, common: dict[str, Any]
) -> dict[str, Any]:
    """A2: observed TTFB per request, injected first byte included."""
    if log.foreign_requests():
        raise AssertionError(
            f"{row} requires one request per measured turn,"
            f" {log.foreign_requests()} foreign requests were observed"
        )
    samples = log.measured_ttfb()
    record = build_run_record(
        row,
        measured_ops=len(log.measured_turns()),
        warmup_ops=log.warmup_turns,
        samples=samples,
        **common,
    )
    record["injected_included"] = True
    record["injected_first_byte_ms"] = log.point.first_byte_ms
    record["delay_dominated"] = True
    return record


def _gap_row_record(
    log: LatencyLog, row: str, common: dict[str, Any]
) -> dict[str, Any]:
    """A3: observed inter-chunk gaps, injected chunk gap included."""
    samples = log.measured_gaps()
    record = build_run_record(
        row,
        measured_ops=len(log.measured_turns()),
        warmup_ops=log.warmup_turns,
        samples=samples,
        **common,
    )
    record["injected_included"] = True
    record["injected_chunk_gap_ms"] = log.point.chunk_gap_ms
    record["delay_dominated"] = True
    record["gaps_per_turn"] = max(0, log.point.chunk_count - 1)
    return record


def _heartbeat_row_record(
    log: LatencyLog, row: str, common: dict[str, Any]
) -> dict[str, Any]:
    """A4/B2: the scheduled heartbeat's per-row excess over its interval."""
    samples = log.measured_heartbeat()
    heartbeat = log.heartbeat
    warmup_rows = (
        0 if heartbeat is None else min(heartbeat.warmup_rows, heartbeat.total_rows)
    )
    record = build_run_record(
        row,
        measured_ops=len(samples),
        warmup_ops=warmup_rows,
        samples=samples,
        **common,
    )
    record["heartbeat"] = (
        {"interval_ms": HEARTBEAT_INTERVAL_MS, "schedule": "scheduled_interval_excess"}
        if heartbeat is None
        else heartbeat.record()
    )
    return record


def _record_extras(log: LatencyLog) -> dict[str, Any]:
    """Attach the run's cell-level context to one row record."""
    return {
        "adapter": "headless",
        "injected_delay_point": log.point.record(),
        "scale": log.scale,
        "workload_hash": log.workload_hash,
        "cell_chunks_coverage": log.chunks_coverage(),
        "request_context": log.request_context,
        "memo_probe": log.memo_probe,
        "quiet": log.quiet,
    }


def _cell_eligibility(log: LatencyLog, record: dict[str, Any]) -> dict[str, Any]:
    """Combine the row floors, the invalid-sample cap and the run's scale."""
    measured = len(log.measured_turns())
    invalid = len(log.invalid_residuals())
    ratio = 0.0 if not measured else invalid / measured
    reasons = list(record["eligibility"]["reasons"])
    cap_exceeded = ratio > INVALID_SAMPLE_CAP
    if cap_exceeded:
        reasons.append("invalid_sample_cap")
    mode = str(log.scale.get("mode", ""))
    if mode != "measure":
        reasons.append("contract_scale_run")
    return {
        "floor_eligible": record["eligibility"]["eligible"],
        "invalid_samples": invalid,
        "invalid_ratio": round(ratio, _SAMPLE_PRECISION),
        "invalid_sample_cap": INVALID_SAMPLE_CAP,
        "invalid_cap_exceeded": cap_exceeded,
        "scale_mode": mode,
        "baseline_eligible": not reasons,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------
# The Pilot path (observation-only)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PilotTurnSample:
    """One turn driven through the Pilot path with an injected arrival schedule."""

    turn: int
    raw_ms: float
    injected_ms: float
    chunks: int
    flushes: int

    @property
    def residual_ms(self) -> float:
        return self.raw_ms - self.injected_ms

    @property
    def valid(self) -> bool:
        return self.residual_ms >= 0.0

    def record(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "raw_ms": round(self.raw_ms, _SAMPLE_PRECISION),
            "injected_delay_ms": round(self.injected_ms, _SAMPLE_PRECISION),
            "residual_ms": round(self.residual_ms, _SAMPLE_PRECISION),
            "valid": self.valid,
            "chunks": self.chunks,
            "flushes": self.flushes,
        }


class PilotLatencyRun:
    """One injected-delay matrix point driven through the live Pilot path.

    The headless backend is not on this path, so the run owns the chunk arrival
    schedule (the first-byte hold and the per-chunk gap) and renders every chunk
    through the production flush boundary under the WP4 application. The report
    is observation-only: the gated rows are the headless cell's alone.
    """

    def __init__(
        self,
        *,
        cell: str = CELL_FRESH_HEADLESS,
        point: InjectedDelayPoint | None = None,
        scale: ScenarioScale = PILOT_PARITY_SCALE,
        warmup_turns: int = CONTRACT_WARMUP_TURNS,
        frame_delay_s: float = PILOT_FRAME_DELAY_S,
    ) -> None:
        if scale.child_agents or scale.burst_agents:
            raise ValueError("the Pilot latency path is a single-root transcript")
        if warmup_turns < 0:
            raise ValueError("warmup_turns must be non-negative")
        self.cell = cell
        self.point = CANONICAL_POINT if point is None else point
        self.scale = scale
        self.warmup_turns = warmup_turns
        self.frame_delay_s = frame_delay_s
        self.turns: list[PilotTurnSample] = []
        self.workload_hash: str | None = None

    @property
    def adapter(self) -> str:
        return PILOT_ADAPTER

    async def run(self) -> list[PilotTurnSample]:
        """Drive the committed workload through a live Pilot application."""
        if self.turns:
            raise RuntimeError("a Pilot latency run drives one workload")
        app = PilotScenarioApp()
        async with app.run_test() as pilot:
            history = app.query_one(f"#{PILOT_HISTORY_CONTAINER_ID}", Vertical)
            for turn in range(self.scale.root_turns):
                self.turns.append(await self._run_turn(pilot, history, turn))
        self.workload_hash = workload_spec(self.cell).digest()
        return list(self.turns)

    async def _run_turn(
        self, pilot: Pilot[None], history: Vertical, turn: int
    ) -> PilotTurnSample:
        prompt = root_user_content(turn)
        started_ns = perf_counter_ns()
        user = UserMessage(prompt)
        await history.mount(user)
        message = AssistantMessage("")
        await history.mount(message)
        # Steer this widget's own frame pacing; the flush boundary stays
        # production's (the WP4 contract run does the same).
        message._frame_delay = self.frame_delay_s
        await pilot.pause()
        if self.point.first_byte_ms:
            await asyncio.sleep(self.point.first_byte_ms / 1000)
        expected = ""
        chunks = response_chunks(turn)
        for chunk in chunks:
            expected += chunk
            await message.append_content(chunk)
            await _settle_flush(pilot, message, expected)
            if self.point.chunk_gap_ms:
                await asyncio.sleep(self.point.chunk_gap_ms / 1000)
        raw_ms = _milliseconds(perf_counter_ns() - started_ns)
        delivered = message.get_content()
        if delivered != expected:
            raise AssertionError(
                f"turn {turn} delivered {len(delivered)} chars,"
                f" the committed response is {len(expected)}"
            )
        if user.get_content() != prompt:
            raise AssertionError(f"turn {turn} rendered a different user prompt")
        return PilotTurnSample(
            turn=turn,
            raw_ms=raw_ms,
            injected_ms=self.point.total_delay_ms,
            chunks=len(chunks),
            flushes=len(chunks),
        )

    def measured_turns(self) -> list[PilotTurnSample]:
        return [turn for turn in self.turns if turn.turn >= self.warmup_turns]

    def report(self) -> dict[str, Any]:
        """Return the observation-only matrix report for this Pilot run."""
        measured = self.measured_turns()
        raw = [turn.raw_ms for turn in measured]
        residuals = [turn.residual_ms for turn in measured if turn.valid]
        return {
            "schema": PILOT_MATRIX_SCHEMA,
            "metric_version": SCENARIO_METRIC_VERSION,
            "adapter": self.adapter,
            "cell": self.cell,
            "gated": False,
            "reason": PILOT_OBSERVATION_ONLY_REASON,
            "injected_delay_point": self.point.record(),
            "scale": self.scale.dimensions(),
            "warmup_turns": self.warmup_turns,
            "workload_hash": self.workload_hash,
            "chunks_coverage": {
                "turns": len(self.turns),
                "measured_turns": len(measured),
                "chunks": sum(turn.chunks for turn in measured),
                "flushes": sum(turn.flushes for turn in measured),
                "complete": bool(measured)
                and all(turn.chunks > 0 for turn in measured),
                "missing": [] if measured else ["turns"],
            },
            "summary": {
                "raw_p95_ms": _p95_or_none(raw),
                "residual_p95_ms": _p95_or_none(residuals),
                "invalid_samples": len(measured) - len(residuals),
            },
            "turns": [turn.record() for turn in self.turns],
        }


async def run_pilot_matrix(
    *points: InjectedDelayPoint,
    cell: str = CELL_FRESH_HEADLESS,
    scale: ScenarioScale = PILOT_PARITY_SCALE,
    warmup_turns: int = CONTRACT_WARMUP_TURNS,
) -> list[dict[str, Any]]:
    """Run each matrix point through the Pilot path; return their reports."""
    reports: list[dict[str, Any]] = []
    for point in points:
        run = PilotLatencyRun(
            cell=cell, point=point, scale=scale, warmup_turns=warmup_turns
        )
        await run.run()
        reports.append(run.report())
    return reports


__all__ = [
    "CANONICAL_POINT",
    "CANONICAL_POINT_KEY",
    "CONTRACT_WARMUP_TURNS",
    "GATED_POINT_BY_CELL",
    "GATED_ROWS_BY_CELL",
    "HEARTBEAT_INTERVAL_MS",
    "HEARTBEAT_WARMUP_ROWS",
    "INVALID_SAMPLE_CAP",
    "LOADED_CELL_POINT",
    "MATRIX_POINTS",
    "OBSERVATION_ONLY_REASON",
    "PILOT_MATRIX_SCHEMA",
    "PILOT_OBSERVATION_ONLY_REASON",
    "HeadlessLatencyCell",
    "HeartbeatLog",
    "InjectedDelayPoint",
    "LatencyLog",
    "LoopHeartbeatObserver",
    "PilotLatencyRun",
    "PilotTurnSample",
    "StreamCallTiming",
    "StreamingChunkObserver",
    "TurnLatencySample",
    "build_cell_records",
    "gated_point",
    "matrix_point",
    "observation_only_points",
    "run_pilot_matrix",
    "run_start_utc",
]
