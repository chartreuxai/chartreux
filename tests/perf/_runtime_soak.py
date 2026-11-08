"""Loaded latency soak mode (v0.7.5 stage 2, X3a).

The soak cells exist for the pathology the campaign is built around: a long
session whose history keeps growing while a loaded cohort keeps working. Plan §5
pins the shape. The run starts from the integrity-valid 200k-equivalent fixture
history (WP3a) and then ADDS 400 synthetic root turns, so the transcript grows
throughout the run; the loaded variant holds 8 agents (root + 7 synchronized
children) and releases one gated child burst every 10th root turn. The seed, the
per-turn message sizes and the deterministic response shape are the committed
workload definitions of the WP3b harness, and :meth:`SoakShape.digest` binds
them to the fixture hash in :data:`SOAK_SHAPE_HASHES` — a drift in any of them
changes the digest and makes the cell ineligible.

Two gated quantities come out of a latency run (Appendix B rows F1/F2):

* F1 — per-turn application overhead (wall from the turn's user-message
  submission to model-visible completion, minus the turn's injected delay; the
  loaded cell injects none), over exactly turns 301-400: 100 late turns, with
  turns 1-300 as the warm window and turn 300 the warm boundary.
* F2 — loop heartbeat excess (observed heartbeat interval minus the 1 ms
  scheduled cadence) over the same late window, with >= 1,000 late-window rows
  in scope and the row count carried as chunks coverage.

NO tracemalloc: a latency run with allocation tracing active is invalid, and the
gated record path refuses to emit one. The memory soak (X3b) reuses this shape
and driver with RSS sampling plus tracing, on a separate run set; allocation
instrumentation never co-runs with the latency gates. :meth:`SoakLatencyRun.
record_turn` is the per-turn seam X3b extends, and retained-set stamps,
compaction and pre-seal occurrences belong to that mode's reporting — this mode
reports turn overhead and loop responsiveness by 100-turn segment (§5) and does
not conflate the two.

The memory mode (:class:`SoakMemoryRun`) drives the same committed schedule with
allocation tracing ACTIVE and an RSS sample after every settled turn, and
produces one gated quantity (Appendix B row G1): the within-run p95 of the 101
rolling least-squares RSS slopes, each over a 100-turn window spanning
[200,300] .. [300,400] on exactly the 201 samples of turns 200..400. Signed
slopes are recorded as measured and clamped to zero for the gate; the five fresh
processes — never the 101 overlapping windows — are the independent replicates,
so a plateaued baseline disarms itself through the nonpositive-threshold
dormancy rule instead of gating. The same runs report the retained-set
checkpoint diffs of their structural stamps (turns 0/100/200/300/400 yield four
diffs, because a checkpoint count is a DIFF count), reporting-only: four diffs
per run cannot support a within-run p95. The memory mode emits NO latency gate
and keeps a run set the latency rows never share — allocation instrumentation
and the latency rows are never co-measured.

In-suite contract nodes run SHORTENED shapes (plan §5): a shortened run reports
its late-window series and its coverage, and refuses to emit gated records,
because records carry the pinned late-window bounds and a shortened run must
never be mistaken for campaign data. Full-scale runs are opt-in external cells:
:func:`main` is their entry point (path guard, egress guard, one report).
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import tracemalloc
from typing import Any

from tests.perf._metrics import alloc_breakdown_mib, machine_context
from tests.perf._runtime_fixture import (
    FIXTURE_HASH,
    FIXTURE_MESSAGE_COUNT,
    FIXTURE_RESPONSE_CHARS,
    FIXTURE_TURN_WIDTH,
)
from tests.perf._runtime_metrics import current_perf_paths
from tests.perf._runtime_process_tree import read_rss_bytes
from tests.perf._runtime_report import (
    CELL_SOAK_LATENCY,
    CELL_SOAK_MEMORY,
    COUNT_LATE_HEARTBEAT_ROWS,
    COUNT_RSS_SAMPLES,
    COUNT_WINDOW_SLOPES,
    COVERAGE_LATE_HEARTBEAT_ROWS,
    COVERAGE_ROLLING_WINDOWS,
    MEMORY_WINDOW_SPAN,
    METRIC_VERSION,
    MODE_MEASURE,
    SLOPE_CLAMP_RULE,
    SLOPE_UNIT,
    SOAK_LATE_WINDOW,
    LateWindow,
    build_report,
    build_run_record,
    nearest_rank,
)
from tests.perf._runtime_scenario import (
    BURST_INTERVAL_TURNS,
    CELL_LOADED_HEADLESS,
    LOADED_CHILD_AGENTS,
    RESPONSE_CHUNK_COUNT,
    ROOT_TURNS,
    ROOT_USER_CHARS,
    WORKLOAD_HASHES,
    WORKLOAD_SEED,
    ChildBurstResult,
    EgressGuard,
    HeadlessScenario,
    RootTurnResult,
    ScenarioScale,
    build_headless_scenario,
    workload_spec,
)

SOAK_SCHEMA = "chartreux.runtime_soak.v1"

#: The soak's committed schedule: a 1 ms heartbeat cadence (the A4/B2 observer's
#: cadence, reused so F2 and B2 measure the same quantity) over the late window.
SOAK_HEARTBEAT_INTERVAL_S = 0.001

#: Plan §5 budgets: 360 s measured window, 720 s kill deadline incl. setup.
SOAK_BUDGET_S = 360.0
SOAK_KILL_DEADLINE_S = 720.0

#: Plan §5 checkpoints: turns 0/100/200/300/400 (turn 0 is the starting state).
SOAK_CHECKPOINT_TURNS: tuple[int, ...] = (0, 100, 200, 300, 400)

#: Plan §5 segment report: 100-turn buckets of turn overhead and loop lag.
SOAK_SEGMENT_TURNS = 100

#: The degenerate-cell mode label: a shortened run is never campaign data.
SOAK_MODE_CONTRACT = "contract"

#: The external runner's two modes. Their run sets are separate: the memory mode
#: never co-measures with the latency rows.
SOAK_MODE_LATENCY = "latency"
SOAK_MODE_MEMORY = "memory"
SOAK_RUNNER_MODES: tuple[str, ...] = (SOAK_MODE_LATENCY, SOAK_MODE_MEMORY)

_P50_FRACTION = 0.50
_P95_FRACTION = 0.95
_MILLISECONDS_PER_SECOND = 1_000.0
_LOADAVG_SAMPLE_COUNT = 3
_SAMPLE_PRECISION = 4

#: An open campaign: this run measures one attempt of a shared run set, and the
#: campaign's closure status is the parent's artifact, never the run's.
SOAK_OPEN_CAMPAIGN_NOTE = (
    "open soak campaign: one measured attempt of the F1/F2 shared run set"
)

_MISSING_LATE_TURNS = "measured_turns"
_MISSING_LATE_ROWS = "late_window_heartbeat_rows"
_MISSING_TURNS = "root_turns"

#: The memory mode's tracing labels: a G1 run traces allocations, and a latency
#: run never does (Appendix B rows F1/F2/G1).
SOAK_TRACEMALLOC_DISABLED = "disabled"
SOAK_TRACEMALLOC_ACTIVE = "active"

#: G1's committed window geometry (Appendix B row G1): 101 rolling 100-turn
#: windows spanning [200,300] .. [300,400] over the 201 RSS samples of turns
#: 200..400. The contract node re-checks these against
#: :data:`~tests.perf._runtime_report.MEMORY_WINDOW_SPAN`.
MEMORY_FIRST_SAMPLE_TURN = 200
MEMORY_LAST_SAMPLE_TURN = 400
MEMORY_WINDOW_WIDTH_TURNS = 100
MEMORY_WINDOW_COUNT = 101

#: The gated slope unit and clamp rule: the gated quantity is positive growth,
#: and the signed slope stays recorded beside the clamped one.
MEMORY_SLOPE_UNIT = SLOPE_UNIT
MEMORY_SLOPE_CLAMP = SLOPE_CLAMP_RULE

#: A retained-set checkpoint count is a DIFF count: five stamps yield four
#: diffs per run, and four diffs cannot support a within-run p95.
MEMORY_RETAINED_SET_GATE = "reporting_only"
MEMORY_RETAINED_SET_NULL_REASON = (
    "a retained-set checkpoint count is a diff count: four diffs per full-scale "
    "run cannot support a within-run p95"
)

#: The memory cell runs a run set the latency rows never share.
MEMORY_EXCLUDED_LATENCY_REASON = (
    "memory mode gates RSS growth only: latency and heartbeat quantities are not "
    "measured here, and allocation tracing never co-runs with the latency rows"
)

MEMORY_OPEN_CAMPAIGN_NOTE = (
    "open soak campaign: one measured attempt of the G1 memory run set, never "
    "co-measured with the F1/F2 latency runs"
)

_PROC_ROOT = Path("/proc")
_RETAINED_SET_TOP_SOURCES = 5


class SoakError(RuntimeError):
    """A soak run, shape or report violates the committed definition."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"soak rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class MemoryWindows:
    """The memory cell's rolling-window geometry (Appendix B row G1).

    The geometry is CLOSED: ``window_count`` consecutive windows, each spanning
    ``window_width_turns`` turns between its first and last turn (both ends
    inclusive, so a window carries one more sample than its width), start at
    ``first_sample_turn``. The last window must therefore end exactly on
    ``last_sample_turn``, and the span must carry exactly
    ``window_count + window_width_turns`` per-turn RSS samples — the plan's
    windows [200,300] .. [300,400] are 101 windows of 100-turn spans over 201
    samples. A geometry its windows cannot cover is rejected, never padded.
    """

    first_sample_turn: int
    last_sample_turn: int
    window_width_turns: int
    window_count: int

    @property
    def warm_boundary_turn(self) -> int:
        """Return the turn before the span: the warm boundary, never sampled."""
        return self.first_sample_turn - 1

    def sample_count(self) -> int:
        """Return how many per-turn RSS samples the geometry requires."""
        return self.last_sample_turn - self.first_sample_turn + 1

    def sample_turns(self) -> range:
        """Return the turns whose RSS samples feed the windows."""
        return range(self.first_sample_turn, self.last_sample_turn + 1)

    def windows(self) -> list[tuple[int, int]]:
        """Return the inclusive (first, last) turn pair of every window."""
        return [
            (start, start + self.window_width_turns)
            for start in range(
                self.first_sample_turn, self.first_sample_turn + self.window_count
            )
        ]

    def validate(self) -> None:
        """Reject a geometry whose closed span its windows cannot cover."""
        if self.window_width_turns < 1:
            raise SoakError(
                "window_width_invalid",
                f"a {self.window_width_turns}-turn window has no slope",
            )
        if self.window_count < 1:
            raise SoakError("window_count_invalid", f"{self.window_count} windows")
        if self.last_sample_turn < self.first_sample_turn:
            raise SoakError(
                "window_span_invalid",
                f"[{self.first_sample_turn},{self.last_sample_turn}] is empty",
            )
        expected = self.window_count + self.window_width_turns
        if self.sample_count() != expected:
            raise SoakError(
                "window_geometry_invalid",
                f"{self.sample_count()} samples for {self.window_count} windows "
                f"spanning {self.window_width_turns} turns (needs {expected})",
            )
        last = self.windows()[-1]
        if last[1] != self.last_sample_turn:
            raise SoakError(
                "window_geometry_invalid",
                f"the last window ends at {last[1]}, not {self.last_sample_turn}",
            )

    def record(self) -> dict[str, int]:
        """Return the geometry's record form, with its derived sample count."""
        return {
            "first_sample_turn": self.first_sample_turn,
            "last_sample_turn": self.last_sample_turn,
            "window_width_turns": self.window_width_turns,
            "window_count": self.window_count,
            "sample_count": self.sample_count(),
        }


#: The gated G1 geometry: windows [200,300] .. [300,400] over 201 RSS samples.
MEMORY_WINDOWS_FULL = MemoryWindows(
    first_sample_turn=MEMORY_FIRST_SAMPLE_TURN,
    last_sample_turn=MEMORY_LAST_SAMPLE_TURN,
    window_width_turns=MEMORY_WINDOW_WIDTH_TURNS,
    window_count=MEMORY_WINDOW_COUNT,
)


@dataclass(frozen=True, slots=True)
class SoakShape:
    """One committed soak shape: the schedule, the bounds and the budgets.

    ``late_window`` carries the measured bounds; the full latency shape's window
    IS :data:`~tests.perf._runtime_report.SOAK_LATE_WINDOW` (turns 301-400 behind
    the turn-300 warm boundary), and ``gated`` reports whether a run of this
    shape may emit gated records at all. ``cell`` and ``tracemalloc_mode`` carry
    the mode's identity into the shape record and digest; ``memory_windows`` is
    the memory mode's rolling-window geometry and is ``None`` for a latency mode.
    """

    name: str
    mode: str
    root_turns: int
    history_messages: int
    child_agents: int
    burst_agents: int
    burst_interval_turns: int
    heartbeat_interval_s: float
    budget_s: float
    kill_deadline_s: float
    checkpoint_turns: tuple[int, ...]
    late_window: LateWindow
    segment_turns: int = SOAK_SEGMENT_TURNS
    cell: str = CELL_SOAK_LATENCY
    tracemalloc_mode: str = SOAK_TRACEMALLOC_DISABLED
    memory_windows: MemoryWindows | None = None

    @property
    def gated(self) -> bool:
        """Return whether a run of this shape is eligible gated data."""
        return (
            self.cell == CELL_SOAK_LATENCY
            and self.mode == MODE_MEASURE
            and self.late_window == SOAK_LATE_WINDOW
            and self.root_turns == ROOT_TURNS
            and self.history_messages == FIXTURE_MESSAGE_COUNT
            and self.child_agents == LOADED_CHILD_AGENTS
            and self.burst_agents == LOADED_CHILD_AGENTS
            and self.burst_interval_turns == BURST_INTERVAL_TURNS
        )

    @property
    def total_agents(self) -> int:
        return self.child_agents + 1

    def scenario_scale(self) -> ScenarioScale:
        """Return the WP3b harness scale this shape drives."""
        return ScenarioScale(
            name=self.name,
            root_turns=self.root_turns,
            history_messages=self.history_messages,
            child_agents=self.child_agents,
            burst_agents=self.burst_agents,
            burst_interval_turns=self.burst_interval_turns,
            mode=self.mode,
        )

    def record(self) -> dict[str, Any]:
        """Return the run-record shape block, bound to the live definitions."""
        record: dict[str, Any] = {
            "schema": SOAK_SCHEMA,
            "cell": self.cell,
            "shape": self.name,
            "mode": self.mode,
            "fixture_hash": FIXTURE_HASH,
            "workload_hash": WORKLOAD_HASHES[CELL_LOADED_HEADLESS],
            "workload_seed": WORKLOAD_SEED,
            "root_user_chars": ROOT_USER_CHARS,
            "response_chars": FIXTURE_RESPONSE_CHARS,
            "response_chunks": RESPONSE_CHUNK_COUNT,
            "root_turns": self.root_turns,
            "history_messages": self.history_messages,
            "total_agents": self.total_agents,
            "child_agents": self.child_agents,
            "burst_agents": self.burst_agents,
            "burst_interval_turns": self.burst_interval_turns,
            "tracemalloc": self.tracemalloc_mode,
            "late_window": self.late_window.as_record(),
            "checkpoint_turns": list(self.checkpoint_turns),
            "segment_turns": self.segment_turns,
            "budget_s": self.budget_s,
            "kill_deadline_s": self.kill_deadline_s,
        }
        if self.memory_windows is None:
            record["heartbeat_interval_ms"] = round(
                self.heartbeat_interval_s * _MILLISECONDS_PER_SECOND, 6
            )
        else:
            record["memory_windows"] = self.memory_windows.record()
            record["slope_unit"] = MEMORY_SLOPE_UNIT
            record["slope_clamp"] = MEMORY_SLOPE_CLAMP
        return record

    def digest(self) -> str:
        """Return the sha256 digest binding this shape to the fixture hash."""
        encoded = json.dumps(
            self.record(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(f"{SOAK_SCHEMA}|{FIXTURE_HASH}|".encode())
        digest.update(encoded)
        return digest.hexdigest()


#: The gated full-scale latency soak: 400 growing turns, 8 agents, one
#: synchronized child burst every 10th turn, turns 301-400 measured.
LATENCY_SOAK_FULL = SoakShape(
    name="full",
    mode=MODE_MEASURE,
    root_turns=ROOT_TURNS,
    history_messages=FIXTURE_MESSAGE_COUNT,
    child_agents=LOADED_CHILD_AGENTS,
    burst_agents=LOADED_CHILD_AGENTS,
    burst_interval_turns=BURST_INTERVAL_TURNS,
    heartbeat_interval_s=SOAK_HEARTBEAT_INTERVAL_S,
    budget_s=SOAK_BUDGET_S,
    kill_deadline_s=SOAK_KILL_DEADLINE_S,
    checkpoint_turns=SOAK_CHECKPOINT_TURNS,
    late_window=SOAK_LATE_WINDOW,
    segment_turns=SOAK_SEGMENT_TURNS,
)

#: The in-suite shape: short enough for the contract node's hard window, with
#: the same cadence, the same burst rule and a late window it can actually reach.
LATENCY_SOAK_CONTRACT = SoakShape(
    name="contract",
    mode=SOAK_MODE_CONTRACT,
    root_turns=8,
    history_messages=FIXTURE_TURN_WIDTH * 2,
    child_agents=1,
    burst_agents=1,
    burst_interval_turns=3,
    heartbeat_interval_s=SOAK_HEARTBEAT_INTERVAL_S,
    budget_s=5.0,
    kill_deadline_s=10.0,
    checkpoint_turns=(0, 4, 8),
    late_window=LateWindow(
        warm_boundary_turn=5,
        first_measured_turn=6,
        last_measured_turn=8,
        measured_turns=3,
    ),
    segment_turns=4,
)

SOAK_SHAPES: dict[str, SoakShape] = {
    LATENCY_SOAK_FULL.name: LATENCY_SOAK_FULL,
    LATENCY_SOAK_CONTRACT.name: LATENCY_SOAK_CONTRACT,
}

#: Committed shape digests. A digest that no longer matches this table means the
#: soak definition (or the fixture lineage it binds) drifted: ineligible.
SOAK_SHAPE_HASHES: dict[str, str] = {
    LATENCY_SOAK_FULL.name: (
        "e032fc47f148c6f03fab983483d08d9c71d3bd10b66e88dd0e7993aa45809a68"
    ),
    LATENCY_SOAK_CONTRACT.name: (
        "ffc2f56f0178a23963f6f947eab84cc292b29dbdde5afff99e21488471805de2"
    ),
}


def soak_shape(name: str) -> SoakShape:
    """Return one committed soak shape by name."""
    shape = SOAK_SHAPES.get(name)
    if shape is None:
        raise SoakError("unknown_shape", f"{name!r} is not a committed soak shape")
    return shape


#: The gated full-scale memory soak (G1): the loaded schedule with allocation
#: tracing active, RSS sampled after every turn, and the closed [200,300] ..
#: [300,400] window geometry over turns 200..400.
MEMORY_SOAK_FULL = SoakShape(
    name="memory-full",
    mode=MODE_MEASURE,
    root_turns=ROOT_TURNS,
    history_messages=FIXTURE_MESSAGE_COUNT,
    child_agents=LOADED_CHILD_AGENTS,
    burst_agents=LOADED_CHILD_AGENTS,
    burst_interval_turns=BURST_INTERVAL_TURNS,
    heartbeat_interval_s=SOAK_HEARTBEAT_INTERVAL_S,
    budget_s=SOAK_BUDGET_S,
    kill_deadline_s=SOAK_KILL_DEADLINE_S,
    checkpoint_turns=SOAK_CHECKPOINT_TURNS,
    late_window=MEMORY_WINDOW_SPAN,
    segment_turns=SOAK_SEGMENT_TURNS,
    cell=CELL_SOAK_MEMORY,
    tracemalloc_mode=SOAK_TRACEMALLOC_ACTIVE,
    memory_windows=MEMORY_WINDOWS_FULL,
)

#: The in-suite memory shape: the same sampling and window machinery at a scale
#: the contract node can reach with allocation tracing active (the loaded
#: cohort belongs to the latency mode's shortened run and to the full-scale
#: memory shape, which keeps all 7 children).
MEMORY_SOAK_CONTRACT = SoakShape(
    name="memory-contract",
    mode=SOAK_MODE_CONTRACT,
    root_turns=4,
    history_messages=FIXTURE_TURN_WIDTH * 2,
    child_agents=0,
    burst_agents=0,
    burst_interval_turns=2,
    heartbeat_interval_s=SOAK_HEARTBEAT_INTERVAL_S,
    budget_s=5.0,
    kill_deadline_s=10.0,
    checkpoint_turns=(0, 2, 4),
    late_window=LateWindow(
        warm_boundary_turn=0,
        first_measured_turn=1,
        last_measured_turn=4,
        measured_turns=4,
    ),
    segment_turns=2,
    cell=CELL_SOAK_MEMORY,
    tracemalloc_mode=SOAK_TRACEMALLOC_ACTIVE,
    memory_windows=MemoryWindows(
        first_sample_turn=1, last_sample_turn=4, window_width_turns=1, window_count=3
    ),
)

#: The memory mode's own registry: its run set is separate from the latency
#: shapes, and the latency registry above stays exactly the two latency shapes.
MEMORY_SOAK_SHAPES: dict[str, SoakShape] = {
    MEMORY_SOAK_FULL.name: MEMORY_SOAK_FULL,
    MEMORY_SOAK_CONTRACT.name: MEMORY_SOAK_CONTRACT,
}

#: Committed memory-shape digests, on the same drift rule as the latency ones.
MEMORY_SOAK_SHAPE_HASHES: dict[str, str] = {
    MEMORY_SOAK_FULL.name: (
        "c34ded062a63881bedfa21ebe29816f37d70027e7b65981bf679e867ea8b1e02"
    ),
    MEMORY_SOAK_CONTRACT.name: (
        "0bdddd3d7f197411b4d8218279484eb8206e9751eca432052882213cfb50716d"
    ),
}


def memory_soak_shape(name: str) -> SoakShape:
    """Return one committed memory soak shape by name."""
    shape = MEMORY_SOAK_SHAPES.get(name)
    if shape is None:
        raise SoakError(
            "unknown_shape", f"{name!r} is not a committed memory soak shape"
        )
    return shape


# --------------------------------------------------------------------------
# Loop heartbeat
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeartbeatRow:
    """One logged heartbeat row: when it was scheduled and how late it woke."""

    index: int
    scheduled_s: float
    observed_s: float
    excess_ms: float


class SoakHeartbeat:
    """A 1 ms-cadence event-loop heartbeat task over one soak run.

    Rows are logged for the whole run and attributed to the late window after
    the fact by observation time, so a row logged while turns 301-400 are in
    flight — including the rows of the child bursts interleaved between them —
    is exactly a late-window row.
    """

    def __init__(self, interval_s: float) -> None:
        if interval_s <= 0.0:
            raise SoakError("interval_invalid", f"interval {interval_s!r} is not > 0")
        self._interval_s = interval_s
        self.rows: list[HeartbeatRow] = []
        self._task: asyncio.Task[None] | None = None

    @property
    def interval_s(self) -> float:
        return self._interval_s

    def start(self) -> asyncio.Task[None]:
        """Start the heartbeat task on the running loop."""
        if self._task is not None:
            raise SoakError("heartbeat_running", "the heartbeat is already running")
        self._task = asyncio.get_running_loop().create_task(self._beat())
        return self._task

    async def stop(self) -> None:
        """Cancel the heartbeat task and await its cancellation."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _beat(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            scheduled = loop.time()
            await asyncio.sleep(self._interval_s)
            observed = loop.time()
            self.rows.append(
                HeartbeatRow(
                    index=len(self.rows),
                    scheduled_s=scheduled,
                    observed_s=observed,
                    excess_ms=max(
                        0.0,
                        (observed - scheduled - self._interval_s)
                        * _MILLISECONDS_PER_SECOND,
                    ),
                )
            )

    def rows_between(
        self, opened_s: float | None, closed_s: float | None
    ) -> list[HeartbeatRow]:
        """Return the rows logged while the late window was in flight."""
        if opened_s is None or closed_s is None or closed_s < opened_s:
            return []
        return [row for row in self.rows if opened_s <= row.observed_s <= closed_s]


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SoakTurn:
    """One settled synthetic root turn of the soak."""

    turn: int
    wall_ms: float
    injected_delay_ms: float
    overhead_ms: float
    started_s: float
    finished_s: float
    history_messages: int
    status: str

    def record(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "wall_ms": round(self.wall_ms, _SAMPLE_PRECISION),
            "injected_delay_ms": self.injected_delay_ms,
            "application_overhead_ms": round(self.overhead_ms, _SAMPLE_PRECISION),
            "history_messages": self.history_messages,
            "status": self.status,
        }


def _loadavg() -> tuple[float, ...] | None:
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


def _percentile_or_none(values: Sequence[float], fraction: float) -> float | None:
    """Return one nearest-rank percentile, or NULL when there are no samples."""
    if not values:
        return None
    return round(nearest_rank(values, fraction=fraction), _SAMPLE_PRECISION)


class SoakLatencyRun:
    """The measured state of one loaded-latency soak run.

    ``record_turn`` is the per-turn seam: the memory soak subclasses this run
    and samples RSS there instead of gating latency.
    """

    def __init__(
        self,
        *,
        shape: SoakShape,
        scenario: HeadlessScenario,
        run_id: str,
        started_utc: str,
        loadavg_before: tuple[float, ...] | None,
        seeded_messages: int,
    ) -> None:
        self.shape = shape
        self.scenario = scenario
        self.run_id = run_id
        self.started_utc = started_utc
        self.loadavg_before = loadavg_before
        self.loadavg_after: tuple[float, ...] | None = None
        self.seeded_messages = seeded_messages
        self.turns: list[SoakTurn] = []
        self.bursts: list[ChildBurstResult] = []
        self.burst_spans: list[tuple[float, float]] = []
        self.heartbeat = SoakHeartbeat(shape.heartbeat_interval_s)
        self.window_open_s: float | None = None
        self.window_close_s: float | None = None
        self.tracemalloc_samples: list[bool] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.stopped_early = False
        self.wall_s = 0.0
        self.status = "incomplete"

    # -- measurement seams -------------------------------------------------

    def record_turn(self, turn: SoakTurn) -> None:
        """Record one settled turn (the memory soak samples RSS here)."""
        self.turns.append(turn)
        self.tracemalloc_samples.append(tracemalloc.is_tracing())
        first = self.shape.late_window.first_measured_turn
        last = self.shape.late_window.last_measured_turn
        if first <= turn.turn <= last:
            if self.window_open_s is None:
                self.window_open_s = turn.started_s
            self.window_close_s = turn.finished_s

    def record_checkpoint(self, turn: int) -> None:
        """Record one checkpoint's transcript size (the §5 turn stamps)."""
        self.checkpoints.append({
            "turn": turn,
            "history_messages": len(self.scenario.root_loop.messages),
        })

    def finish(self, wall_s: float) -> None:
        """Close the run with its wall and its over-budget/killed verdict."""
        self.wall_s = wall_s
        self.loadavg_after = _loadavg()
        self.status = self._status()

    def _status(self) -> str:
        if self.stopped_early:
            return "killed"
        if len(self.turns) < self.shape.root_turns:
            return "incomplete"
        if len(self.measured_turns()) != self.shape.late_window.measured_turns:
            return "incomplete"
        if self.wall_s > self.shape.budget_s:
            return "over_budget"
        return "ok"

    # -- late window -------------------------------------------------------

    def measured_turns(self) -> list[SoakTurn]:
        """Return the turns inside the shape's late window, in turn order."""
        first = self.shape.late_window.first_measured_turn
        last = self.shape.late_window.last_measured_turn
        return [turn for turn in self.turns if first <= turn.turn <= last]

    def warmup_turns(self) -> int:
        """Return how many turns were warm window, never measured."""
        return len(self.turns) - len(self.measured_turns())

    def late_overheads_ms(self) -> list[float]:
        """Return the late turns' application overhead (valid samples only)."""
        return [
            turn.overhead_ms
            for turn in self.measured_turns()
            if turn.overhead_ms >= 0.0
        ]

    def invalid_late_samples(self) -> int:
        """Return the late turns whose residual overhead went negative."""
        return sum(1 for turn in self.measured_turns() if turn.overhead_ms < 0.0)

    def late_window_rows(self) -> list[HeartbeatRow]:
        """Return the heartbeat rows logged while the late window was in flight."""
        return self.heartbeat.rows_between(self.window_open_s, self.window_close_s)

    def bursts_in_late_window(self) -> int:
        """Return how many child bursts fell inside the late window's span."""
        opened = self.window_open_s
        closed = self.window_close_s
        if opened is None or closed is None:
            return 0
        return sum(
            1
            for started, finished in self.burst_spans
            if started <= closed and finished >= opened
        )

    def late_heartbeat_excess_ms(self) -> list[float]:
        """Return the late-window heartbeat excess series (the F2 samples)."""
        return [row.excess_ms for row in self.late_window_rows()]

    def all_late_turns_complete(self) -> bool:
        """Return whether every late turn reached its completed state."""
        measured = self.measured_turns()
        return len(measured) == self.shape.late_window.measured_turns and all(
            turn.status == "completed" for turn in measured
        )

    # -- tracemalloc -------------------------------------------------------

    @property
    def tracemalloc_active(self) -> bool:
        """Return whether allocation tracing was observed at any sample."""
        return any(self.tracemalloc_samples) or tracemalloc.is_tracing()

    def assert_no_tracemalloc(self) -> None:
        """Fail when a latency run had allocation tracing active."""
        if self.tracemalloc_active:
            raise SoakError(
                "tracemalloc_active",
                "a latency run with allocation tracing active is invalid",
            )

    # -- records -----------------------------------------------------------

    def chunks_coverage(self) -> dict[str, Any]:
        """Return the run's chunks-coverage block (plan §3)."""
        row_count = len(self.late_window_rows())
        measured = len(self.measured_turns())
        missing: list[str] = []
        if not self.turns:
            missing.append(_MISSING_TURNS)
        if measured != self.shape.late_window.measured_turns:
            missing.append(_MISSING_LATE_TURNS)
        if not row_count:
            missing.append(_MISSING_LATE_ROWS)
        return {
            "unit": COVERAGE_LATE_HEARTBEAT_ROWS,
            "root_turns": len(self.turns),
            "warmup_turns": self.warmup_turns(),
            "measured_turns": measured,
            "late_window_heartbeat_rows": row_count,
            "late_window_child_bursts": self.bursts_in_late_window(),
            "heartbeat_rows": len(self.heartbeat.rows),
            "child_bursts": len(self.bursts),
            "child_turns": sum(burst.completed for burst in self.bursts),
            "complete": not missing,
            "missing": missing,
        }

    def growth_record(self) -> dict[str, Any]:
        """Report the history growth the run actually produced."""
        final = len(self.scenario.root_loop.messages)
        expected = self.seeded_messages + 2 * len(self.turns)
        return {
            "seeded_messages": self.seeded_messages,
            "final_messages": final,
            "expected_minimum": expected,
            "grew": final >= expected,
        }

    def segments(self) -> list[dict[str, Any]]:
        """Return the §5 segment report: per-bucket overhead and loop lag."""
        width = self.shape.segment_turns
        report: list[dict[str, Any]] = []
        for start in range(0, len(self.turns), width):
            bucket = self.turns[start : start + width]
            overheads = [turn.overhead_ms for turn in bucket if turn.overhead_ms >= 0.0]
            excess = [
                row.excess_ms
                for row in self.heartbeat.rows_between(
                    bucket[0].started_s, bucket[-1].finished_s
                )
            ]
            report.append({
                "segment": len(report),
                "first_turn": bucket[0].turn,
                "last_turn": bucket[-1].turn,
                "turns": len(bucket),
                "overhead_p50_ms": _percentile_or_none(overheads, _P50_FRACTION),
                "overhead_p95_ms": _percentile_or_none(overheads, _P95_FRACTION),
                "heartbeat_rows": len(excess),
                "heartbeat_excess_p50_ms": _percentile_or_none(excess, _P50_FRACTION),
                "heartbeat_excess_p95_ms": _percentile_or_none(excess, _P95_FRACTION),
            })
        return report

    def segment_change(self) -> dict[str, Any]:
        """Return the early-to-late change between the first and last segment."""
        segments = self.segments()
        if len(segments) < 2:
            return {"segments": len(segments), "null_reason": "single_segment"}
        early = segments[0]["overhead_p50_ms"]
        late = segments[-1]["overhead_p50_ms"]
        if early is None or late is None:
            return {"segments": len(segments), "null_reason": "no_overhead_samples"}
        return {
            "segments": len(segments),
            "early_overhead_p50_ms": early,
            "late_overhead_p50_ms": late,
            "delta_ms": round(late - early, _SAMPLE_PRECISION),
            "ratio": round(late / early, 4) if early > 0 else None,
        }

    def _require_gated(self) -> None:
        if not self.shape.gated:
            raise SoakError(
                "shape_not_committed",
                f"{self.shape.name!r} is not a gated full-scale soak shape",
            )
        self.assert_no_tracemalloc()
        if self.status != "ok":
            raise SoakError("run_not_complete", f"run status is {self.status!r}")
        if not self.all_late_turns_complete():
            raise SoakError(
                "late_turns_incomplete",
                f"{len(self.measured_turns())} late turns, "
                f"{self.shape.late_window.measured_turns} required",
            )

    def gated_records(self) -> dict[str, dict[str, Any]]:
        """Build the validated F1/F2 run records of a gated run.

        Only a full-scale, tracing-free, complete run reaches this path: a
        shortened run carries bounds it never measured and must not emit data.
        """
        self._require_gated()
        machine = machine_context()
        warmup_ops = self.shape.late_window.warm_boundary_turn
        overheads = self.late_overheads_ms()
        excess = self.late_heartbeat_excess_ms()
        return {
            "F1": build_run_record(
                "F1",
                run_id=self.run_id,
                run_start_utc=self.started_utc,
                measured_ops=len(overheads),
                warmup_ops=warmup_ops,
                samples=overheads,
                mode=MODE_MEASURE,
                closure_status=None,
                closure_note=SOAK_OPEN_CAMPAIGN_NOTE,
                fixture_hash=FIXTURE_HASH,
                machine=machine,
            ),
            "F2": build_run_record(
                "F2",
                run_id=self.run_id,
                run_start_utc=self.started_utc,
                measured_ops=len(excess),
                warmup_ops=warmup_ops,
                samples=excess,
                unit_counts={COUNT_LATE_HEARTBEAT_ROWS: len(excess)},
                mode=MODE_MEASURE,
                closure_status=None,
                closure_note=SOAK_OPEN_CAMPAIGN_NOTE,
                fixture_hash=FIXTURE_HASH,
                machine=machine,
            ),
        }

    def heartbeat_summary(self) -> dict[str, Any]:
        """Summarize the late-window heartbeat rows."""
        excess = self.late_heartbeat_excess_ms()
        summary: dict[str, Any] = {
            "late_window_heartbeat_rows": len(excess),
            "rows": len(self.heartbeat.rows),
            "interval_ms": round(
                self.shape.heartbeat_interval_s * _MILLISECONDS_PER_SECOND, 6
            ),
        }
        if not excess:
            summary["p50"] = None
            summary["p95"] = None
            summary["null_reason"] = "no_late_window_rows"
            return summary
        summary["p50"] = round(
            nearest_rank(excess, fraction=_P50_FRACTION), _SAMPLE_PRECISION
        )
        summary["p95"] = round(
            nearest_rank(excess, fraction=_P95_FRACTION), _SAMPLE_PRECISION
        )
        return summary

    def execution_record(self) -> dict[str, Any]:
        """Return the run's execution block: wall, budget and quiet stamps."""
        return {
            "status": self.status,
            "wall_s": round(self.wall_s, 3),
            "budget_s": self.shape.budget_s,
            "kill_deadline_s": self.shape.kill_deadline_s,
            "turns_requested": self.shape.root_turns,
            "turns_settled": len(self.turns),
            "stopped_early": self.stopped_early,
            "loadavg_before": self.loadavg_before,
            "loadavg_after": self.loadavg_after,
            "loadavg_sample_count": _LOADAVG_SAMPLE_COUNT,
        }

    def report(self) -> dict[str, Any]:
        """Return the soak run report (the external cell's artifact)."""
        coverage = self.chunks_coverage()
        overheads = self.late_overheads_ms()
        return {
            "schema": SOAK_SCHEMA,
            "metric_version": METRIC_VERSION,
            "cell": CELL_SOAK_LATENCY,
            "mode": self.shape.mode,
            "run_id": self.run_id,
            "run_start_utc": self.started_utc,
            "shape": self.shape.record(),
            "shape_hash": self.shape.digest(),
            "gated": self.shape.gated,
            "fixture_hash": FIXTURE_HASH,
            "machine": machine_context(),
            "execution": self.execution_record(),
            "tracemalloc": {
                "active": self.tracemalloc_active,
                "samples": len(self.tracemalloc_samples),
            },
            "late_window": self.shape.late_window.as_record(),
            "late_turn_overheads_p95_ms": (
                round(
                    nearest_rank(overheads, fraction=_P95_FRACTION), _SAMPLE_PRECISION
                )
                if overheads
                else None
            ),
            "invalid_late_samples": self.invalid_late_samples(),
            "heartbeat": self.heartbeat_summary(),
            "chunks_coverage": coverage,
            "checkpoints": list(self.checkpoints),
            "growth": self.growth_record(),
            "segments": self.segments(),
            "segment_change": self.segment_change(),
            "turns": [turn.record() for turn in self.turns],
            "child_bursts": [burst.record() for burst in self.bursts],
        }


# --------------------------------------------------------------------------
# Memory soak (X3b): RSS samples, rolling window slopes, retained-set diffs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RssSample:
    """One per-turn RSS sample: the value, or the reason it is unavailable."""

    turn: int
    rss_bytes: int | None
    reason: str | None

    def record(self) -> dict[str, Any]:
        return {"turn": self.turn, "rss_bytes": self.rss_bytes, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class WindowSlope:
    """One rolling window's least-squares RSS slope, signed and clamped."""

    first_turn: int
    last_turn: int
    signed_bytes_per_turn: float

    def clamped(self) -> float:
        """Return the gated form: positive growth only, negatives to zero."""
        return max(0.0, self.signed_bytes_per_turn)

    def record(self) -> dict[str, Any]:
        return {
            "first_turn": self.first_turn,
            "last_turn": self.last_turn,
            "signed_bytes_per_turn": self.signed_bytes_per_turn,
            "clamped_bytes_per_turn": self.clamped(),
        }


@dataclass(frozen=True, slots=True)
class RetainedSetDiff:
    """The traced retained-set change between two consecutive stamps."""

    from_turn: int
    to_turn: int
    net_bytes: int
    top_growth: list[dict[str, Any]]

    def record(self) -> dict[str, Any]:
        return {
            "from_turn": self.from_turn,
            "to_turn": self.to_turn,
            "net_bytes": self.net_bytes,
            "top_growth": list(self.top_growth),
        }


def least_squares_slope(turns: Sequence[int], values: Sequence[float]) -> float:
    """Return the least-squares slope of ``values`` over ``turns``, per turn.

    The window slope is bytes per turn, so a flat window is exactly 0 and a
    shrinking window is negative: the caller clamps for the gate and keeps the
    signed value in the record.
    """
    if len(turns) != len(values):
        raise SoakError(
            "slope_shape_invalid", f"{len(turns)} turns for {len(values)} values"
        )
    if len(turns) < 2:
        raise SoakError(
            "slope_samples_insufficient", f"a slope needs 2 samples, got {len(turns)}"
        )
    count = float(len(turns))
    mean_turn = sum(turns) / count
    mean_value = sum(values) / count
    numerator = sum(
        (turn - mean_turn) * (value - mean_value)
        for turn, value in zip(turns, values, strict=True)
    )
    denominator = sum((turn - mean_turn) ** 2 for turn in turns)
    if denominator <= 0.0:
        raise SoakError(
            "slope_geometry_degenerate", "the window's turns are not distinct"
        )
    return numerator / denominator


def rolling_window_slopes(
    geometry: MemoryWindows, rss_bytes_by_turn: Mapping[int, int]
) -> list[WindowSlope]:
    """Return the geometry's rolling-window slopes over the sampled turns.

    A window missing any of its per-turn samples yields NO slope: the value is
    never interpolated or back-filled, so the caller sees a short series and its
    chunks coverage stays incomplete.
    """
    slopes: list[WindowSlope] = []
    for first_turn, last_turn in geometry.windows():
        window = list(range(first_turn, last_turn + 1))
        if any(turn not in rss_bytes_by_turn for turn in window):
            continue
        slopes.append(
            WindowSlope(
                first_turn=first_turn,
                last_turn=last_turn,
                signed_bytes_per_turn=round(
                    least_squares_slope(
                        window, [float(rss_bytes_by_turn[turn]) for turn in window]
                    ),
                    _SAMPLE_PRECISION,
                ),
            )
        )
    return slopes


def memory_shape_gated(shape: SoakShape) -> bool:
    """Return whether a shape is eligible full-scale G1 data.

    The memory cell's run set is separate from the latency rows, so this
    predicate — not the latency ``gated`` — decides whether a memory run may
    emit its record, and it binds the closed window geometry alongside the
    committed schedule.
    """
    return (
        shape.cell == CELL_SOAK_MEMORY
        and shape.mode == MODE_MEASURE
        and shape.tracemalloc_mode == SOAK_TRACEMALLOC_ACTIVE
        and shape.memory_windows == MEMORY_WINDOWS_FULL
        and shape.root_turns == ROOT_TURNS
        and shape.history_messages == FIXTURE_MESSAGE_COUNT
        and shape.child_agents == LOADED_CHILD_AGENTS
        and shape.burst_agents == LOADED_CHILD_AGENTS
        and shape.burst_interval_turns == BURST_INTERVAL_TURNS
    )


class SoakMemoryRun(SoakLatencyRun):
    """The measured state of one loaded-memory soak run (X3b).

    The run reuses the latency mode's schedule and driver and changes what is
    measured: RSS is sampled after every settled turn, allocation tracing is
    ACTIVE for the whole run, and the structural checkpoints take retained-set
    stamps. No latency gate is emitted — the inherited F1/F2 record path refuses
    a memory shape, and this class's own record path yields G1 only.
    """

    def __init__(
        self,
        *,
        shape: SoakShape,
        scenario: HeadlessScenario,
        run_id: str,
        started_utc: str,
        loadavg_before: tuple[float, ...] | None,
        seeded_messages: int,
    ) -> None:
        super().__init__(
            shape=shape,
            scenario=scenario,
            run_id=run_id,
            started_utc=started_utc,
            loadavg_before=loadavg_before,
            seeded_messages=seeded_messages,
        )
        self.rss_samples: list[RssSample] = []
        self.retained_stamps: list[dict[str, Any]] = []
        self.retained_diffs: list[RetainedSetDiff] = []
        self._retained_before: tracemalloc.Snapshot | None = None
        self._retained_net_before = 0
        self._retained_turn_before: int | None = None

    # -- measurement seams -------------------------------------------------

    def record_turn(self, turn: SoakTurn) -> None:
        """Record one settled turn and its RSS sample."""
        super().record_turn(turn)
        self.rss_samples.append(self.sample_rss(turn.turn))

    def record_checkpoint(self, turn: int) -> None:
        """Record one structural checkpoint and its retained-set stamp."""
        super().record_checkpoint(turn)
        self._stamp_retained_set(turn)

    def sample_rss(self, turn: int) -> RssSample:
        """Read this process's RSS after ``turn`` settled, or why it failed."""
        rss_bytes, reason = read_rss_bytes(os.getpid(), _PROC_ROOT)
        return RssSample(turn=turn, rss_bytes=rss_bytes, reason=reason)

    def _stamp_retained_set(self, turn: int) -> None:
        """Stamp the traced retained set and diff it against the prior stamp."""
        if not tracemalloc.is_tracing():
            raise SoakError(
                "tracemalloc_inactive",
                "a memory run stamps its retained set with tracing active",
            )
        current, _peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        self.retained_stamps.append({"turn": turn, "traced_bytes": current})
        previous_turn = self._retained_turn_before
        if self._retained_before is not None and previous_turn is not None:
            self.retained_diffs.append(
                RetainedSetDiff(
                    from_turn=previous_turn,
                    to_turn=turn,
                    net_bytes=current - self._retained_net_before,
                    top_growth=alloc_breakdown_mib(
                        self._retained_before, snapshot, limit=_RETAINED_SET_TOP_SOURCES
                    ),
                )
            )
        self._retained_before = snapshot
        self._retained_net_before = current
        self._retained_turn_before = turn

    # -- gating ------------------------------------------------------------

    def require_windows(self) -> MemoryWindows:
        """Return the shape's closed window geometry, or refuse the run."""
        geometry = self.shape.memory_windows
        if geometry is None:
            raise SoakError(
                "shape_not_committed",
                f"{self.shape.name!r} carries no memory window geometry",
            )
        return geometry

    @property
    def memory_gated(self) -> bool:
        """Return whether a run of this shape is eligible G1 data."""
        return memory_shape_gated(self.shape)

    def tracemalloc_observed(self) -> bool:
        """Return whether tracing was observed active at every turn sample."""
        return bool(self.tracemalloc_samples) and all(self.tracemalloc_samples)

    def assert_tracemalloc_active(self) -> None:
        """Fail when a memory run did not trace allocations at every turn."""
        if not self.tracemalloc_observed():
            raise SoakError(
                "tracemalloc_inactive",
                "a memory run samples its RSS with allocation tracing active",
            )

    # -- RSS samples and window slopes -------------------------------------

    def rss_bytes_by_turn(self) -> dict[int, int]:
        """Return the readable RSS samples inside the window span, by turn."""
        geometry = self.shape.memory_windows
        if geometry is None:
            return {}
        wanted = set(geometry.sample_turns())
        series: dict[int, int] = {}
        for sample in self.rss_samples:
            value = sample.rss_bytes
            if value is not None and sample.turn in wanted:
                series[sample.turn] = value
        return series

    def unavailable_rss_samples(self) -> list[dict[str, Any]]:
        """Return the samples whose RSS could not be read, with their reasons."""
        return [
            sample.record() for sample in self.rss_samples if sample.rss_bytes is None
        ]

    def window_slopes(self) -> list[WindowSlope]:
        """Return the rolling-window slopes over the sampled span."""
        geometry = self.shape.memory_windows
        if geometry is None:
            return []
        return rolling_window_slopes(geometry, self.rss_bytes_by_turn())

    def window_slopes_signed(self) -> list[float]:
        """Return the signed slope series (the G1 record's slope series)."""
        return [slope.signed_bytes_per_turn for slope in self.window_slopes()]

    def window_slopes_clamped(self) -> list[float]:
        """Return the clamped slope series (the gated G1 samples)."""
        return [slope.clamped() for slope in self.window_slopes()]

    def window_slope_p95(self) -> dict[str, Any]:
        """Return the within-run p95 over the clamped window slopes."""
        clamped = self.window_slopes_clamped()
        if not clamped:
            return {
                "count": 0,
                "clamped": None,
                "signed": None,
                "null_reason": "no_window_slopes",
            }
        signed = self.window_slopes_signed()
        return {
            "count": len(clamped),
            "clamped": round(
                nearest_rank(clamped, fraction=_P95_FRACTION), _SAMPLE_PRECISION
            ),
            "signed": round(
                nearest_rank(signed, fraction=_P95_FRACTION), _SAMPLE_PRECISION
            ),
        }

    def sampled_span_complete(self) -> bool:
        """Return whether every turn of the window span carries an RSS sample."""
        geometry = self.shape.memory_windows
        if geometry is None:
            return True
        series = self.rss_bytes_by_turn()
        return all(turn in series for turn in geometry.sample_turns())

    # -- status and coverage -----------------------------------------------

    def _status(self) -> str:
        if self.stopped_early:
            return "killed"
        if len(self.turns) < self.shape.root_turns:
            return "incomplete"
        if not self.sampled_span_complete():
            return "incomplete"
        if self.wall_s > self.shape.budget_s:
            return "over_budget"
        return "ok"

    def chunks_coverage(self) -> dict[str, Any]:
        """Return the run's chunks-coverage block (plan §3: RSS and slopes)."""
        geometry = self.shape.memory_windows
        samples = len(self.rss_bytes_by_turn())
        slopes = len(self.window_slopes())
        expected_samples = 0 if geometry is None else geometry.sample_count()
        expected_slopes = 0 if geometry is None else geometry.window_count
        missing: list[str] = []
        if not self.turns:
            missing.append(_MISSING_TURNS)
        if samples != expected_samples:
            missing.append(COUNT_RSS_SAMPLES)
        if slopes != expected_slopes:
            missing.append(COUNT_WINDOW_SLOPES)
        return {
            "unit": COVERAGE_ROLLING_WINDOWS,
            "root_turns": len(self.turns),
            "warmup_turns": self.warmup_turns(),
            "measured_turns": len(self.measured_turns()),
            "rss_samples": samples,
            "window_slopes": slopes,
            "retained_set_checkpoint_diffs": len(self.retained_diffs),
            "child_bursts": len(self.bursts),
            "child_turns": sum(burst.completed for burst in self.bursts),
            "complete": not missing,
            "missing": missing,
        }

    # -- records -----------------------------------------------------------

    def _require_memory_gated(self) -> None:
        if not self.memory_gated:
            raise SoakError(
                "shape_not_committed",
                f"{self.shape.name!r} is not a gated full-scale memory soak shape",
            )
        self.assert_tracemalloc_active()
        if self.status != "ok":
            raise SoakError("run_not_complete", f"run status is {self.status!r}")
        geometry = self.require_windows()
        samples = len(self.rss_bytes_by_turn())
        slopes = len(self.window_slopes())
        if samples != geometry.sample_count() or slopes != geometry.window_count:
            raise SoakError(
                "coverage_incomplete",
                f"{samples} RSS samples and {slopes} slopes for "
                f"{geometry.sample_count()}/{geometry.window_count}",
            )

    def g1_record(self) -> dict[str, Any]:
        """Build the validated G1 run record of a gated memory run.

        Only a full-scale, tracing-active, complete run reaches this path: a
        shortened run carries geometry it never measured and must not emit data.
        """
        self._require_memory_gated()
        geometry = self.require_windows()
        signed = self.window_slopes_signed()
        clamped = [max(0.0, value) for value in signed]
        return build_run_record(
            "G1",
            run_id=self.run_id,
            run_start_utc=self.started_utc,
            measured_ops=len(clamped),
            warmup_ops=geometry.warm_boundary_turn,
            samples=clamped,
            unit_counts={
                COUNT_RSS_SAMPLES: len(self.rss_bytes_by_turn()),
                COUNT_WINDOW_SLOPES: len(clamped),
            },
            slopes_signed=signed,
            mode=MODE_MEASURE,
            closure_status=None,
            closure_note=MEMORY_OPEN_CAMPAIGN_NOTE,
            fixture_hash=FIXTURE_HASH,
            machine=machine_context(),
        )

    def retained_set_record(self) -> dict[str, Any]:
        """Return the retained-set checkpoint diffs, reporting-only."""
        return {
            "gate": MEMORY_RETAINED_SET_GATE,
            "p95_supported": False,
            "null_reason": MEMORY_RETAINED_SET_NULL_REASON,
            "stamps": list(self.retained_stamps),
            "diffs": [diff.record() for diff in self.retained_diffs],
        }

    def report(self) -> dict[str, Any]:
        """Return the memory soak run report (the external cell's artifact).

        The report carries the RSS series, the signed and clamped window slopes
        and the retained-set diffs, and it deliberately omits the latency and
        loop-lag sections: the memory mode measures RSS growth only, and no
        value here may be read as an F1/F2 observation.
        """
        geometry = self.shape.memory_windows
        slopes = self.window_slopes()
        return {
            "schema": SOAK_SCHEMA,
            "metric_version": METRIC_VERSION,
            "cell": CELL_SOAK_MEMORY,
            "mode": self.shape.mode,
            "run_id": self.run_id,
            "run_start_utc": self.started_utc,
            "shape": self.shape.record(),
            "shape_hash": self.shape.digest(),
            "gated": self.memory_gated,
            "fixture_hash": FIXTURE_HASH,
            "machine": machine_context(),
            "execution": self.execution_record(),
            "tracemalloc": {
                "active": self.tracemalloc_observed(),
                "samples": len(self.tracemalloc_samples),
                "retained_set_stamps": len(self.retained_stamps),
            },
            "latency_gates": {"gated": False, "reason": MEMORY_EXCLUDED_LATENCY_REASON},
            "memory": {
                "windows": None if geometry is None else geometry.record(),
                "rss_samples": [sample.record() for sample in self.rss_samples],
                "sampled_turns": len(self.rss_bytes_by_turn()),
                "unavailable_samples": self.unavailable_rss_samples(),
                "slopes": {
                    "unit": MEMORY_SLOPE_UNIT,
                    "clamp": MEMORY_SLOPE_CLAMP,
                    "signed": [slope.signed_bytes_per_turn for slope in slopes],
                    "clamped": [slope.clamped() for slope in slopes],
                    "windows": [slope.record() for slope in slopes],
                },
                "p95": self.window_slope_p95(),
            },
            "retained_set": self.retained_set_record(),
            "chunks_coverage": self.chunks_coverage(),
            "checkpoints": list(self.checkpoints),
            "growth": self.growth_record(),
            "turns": [
                {
                    "turn": turn.turn,
                    "history_messages": turn.history_messages,
                    "status": turn.status,
                }
                for turn in self.turns
            ],
            "child_bursts": [burst.record() for burst in self.bursts],
        }


async def run_memory_soak(
    shape: SoakShape = MEMORY_SOAK_FULL, *, session_root: Path | None = None
) -> SoakMemoryRun:
    """Run one loaded-memory soak at ``shape`` and return its measured state.

    Allocation tracing is ACTIVE for the whole run (started here when the caller
    had it off, and restored to off afterwards), no heartbeat is logged because
    no latency quantity is measured, and RSS is sampled after every settled
    turn. A shortened shape runs and reports; only the full-scale shape reaches
    the G1 record path.
    """
    geometry = shape.memory_windows
    if shape.cell != CELL_SOAK_MEMORY or geometry is None:
        raise SoakError(
            "cell_mismatch", "a memory soak runs the memory cell's window geometry"
        )
    if shape.tracemalloc_mode != SOAK_TRACEMALLOC_ACTIVE:
        raise SoakError(
            "tracemalloc_inactive", "a memory soak runs with allocation tracing"
        )
    geometry.validate()
    spec = workload_spec(CELL_LOADED_HEADLESS)
    if spec.injected_turn_delay_ms != 0.0:
        raise SoakError(
            "injected_delay_present",
            "the loaded cell charges no injected delay, and neither may the soak",
        )
    started_at = datetime.now(UTC)
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    try:
        loadavg_before = _loadavg()
        scenario = await build_headless_scenario(
            CELL_LOADED_HEADLESS,
            scale=shape.scenario_scale(),
            session_root=session_root,
        )
        run = SoakMemoryRun(
            shape=shape,
            scenario=scenario,
            run_id=_run_identity(shape, started_at),
            started_utc=started_at.isoformat(timespec="microseconds"),
            loadavg_before=loadavg_before,
            seeded_messages=len(scenario.root_loop.messages),
        )
        started_wall = perf_counter()
        try:
            await _drive_workload(
                scenario, shape, run, started_wall + shape.kill_deadline_s
            )
        finally:
            await run.heartbeat.stop()
            await scenario.close()
        run.finish(perf_counter() - started_wall)
        scenario.assert_cleanup()
        return run
    finally:
        if started_tracing:
            tracemalloc.stop()


# --------------------------------------------------------------------------
# Driving the soak
# --------------------------------------------------------------------------


def _burst_due(shape: SoakShape, index: int) -> bool:
    """Return whether the scheduled child burst precedes this root turn."""
    return bool(
        shape.child_agents and index and index % shape.burst_interval_turns == 0
    )


def _turn_record(
    turn_number: int, result: RootTurnResult, started_s: float, finished_s: float
) -> SoakTurn:
    return SoakTurn(
        turn=turn_number,
        wall_ms=result.wall_ms,
        injected_delay_ms=result.injected_delay_ms,
        overhead_ms=result.wall_ms - result.injected_delay_ms,
        started_s=started_s,
        finished_s=finished_s,
        history_messages=result.history_messages,
        status=result.status,
    )


async def _drive_workload(
    scenario: HeadlessScenario,
    shape: SoakShape,
    run: SoakLatencyRun,
    deadline_at: float,
) -> None:
    """Run the committed schedule: a burst before every Nth turn, then the turn."""
    loop = asyncio.get_running_loop()
    if 0 in shape.checkpoint_turns:
        run.record_checkpoint(0)
    for index in range(shape.root_turns):
        if perf_counter() > deadline_at:
            run.stopped_early = True
            return
        if _burst_due(shape, index):
            burst_started = loop.time()
            run.bursts.append(await scenario.run_child_burst(len(run.bursts)))
            run.burst_spans.append((burst_started, loop.time()))
        started_s = loop.time()
        result = await scenario.run_root_turn(index)
        finished_s = loop.time()
        run.record_turn(_turn_record(index + 1, result, started_s, finished_s))
        if index + 1 in shape.checkpoint_turns:
            run.record_checkpoint(index + 1)


def _run_identity(shape: SoakShape, started: datetime) -> str:
    return f"{shape.cell}-{shape.name}-{started.strftime('%Y%m%dT%H%M%S%f')}"


async def run_latency_soak(
    shape: SoakShape = LATENCY_SOAK_FULL, *, session_root: Path | None = None
) -> SoakLatencyRun:
    """Run one loaded-latency soak at ``shape`` and return its measured state.

    The run refuses to start with allocation tracing active, seeds the committed
    fixture history, grows it with the committed synthetic turns, and returns the
    late-window series and coverage. Gated records are the caller's next step and
    exist only for the full-scale shape.
    """
    if tracemalloc.is_tracing():
        raise SoakError(
            "tracemalloc_active", "a latency soak must run without allocation tracing"
        )
    spec = workload_spec(CELL_LOADED_HEADLESS)
    if spec.injected_turn_delay_ms != 0.0:
        raise SoakError(
            "injected_delay_present",
            "the loaded cell charges no injected delay, and neither may the soak",
        )
    started_at = datetime.now(UTC)
    run_id = _run_identity(shape, started_at)
    loadavg_before = _loadavg()
    scenario = await build_headless_scenario(
        CELL_LOADED_HEADLESS, scale=shape.scenario_scale(), session_root=session_root
    )
    run = SoakLatencyRun(
        shape=shape,
        scenario=scenario,
        run_id=run_id,
        started_utc=started_at.isoformat(timespec="microseconds"),
        loadavg_before=loadavg_before,
        seeded_messages=len(scenario.root_loop.messages),
    )
    started_wall = perf_counter()
    run.heartbeat.start()
    try:
        await _drive_workload(
            scenario, shape, run, started_wall + shape.kill_deadline_s
        )
    finally:
        await run.heartbeat.stop()
        await scenario.close()
    run.finish(perf_counter() - started_wall)
    scenario.assert_cleanup()
    return run


# --------------------------------------------------------------------------
# External cell runner (opt-in; never part of the in-suite nodes)
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runtime-soak",
        description=(
            "Run one full-scale loaded soak as an external campaign attempt and "
            "write its report: the latency mode (F1/F2) or the memory mode (G1). "
            "The two modes are separate run sets and are never co-measured."
        ),
    )
    parser.add_argument(
        "--report", help="report path (defaults to CHARTREUX_PERF_REPORT)"
    )
    parser.add_argument(
        "--mode",
        choices=SOAK_RUNNER_MODES,
        default=SOAK_MODE_LATENCY,
        help="soak mode to run (default: latency)",
    )
    return parser


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _memory_cell(run_dir: Path, report: Path, guard: EgressGuard) -> int:
    """Run the full-scale memory soak as one G1 attempt and write its report."""
    run = asyncio.run(
        run_memory_soak(MEMORY_SOAK_FULL, session_root=run_dir / "sessions")
    )
    guard.assert_closed()
    document = run.report()
    if document["gated"]:
        try:
            record = run.g1_record()
        except SoakError as error:
            document["records_refused"] = {
                "reason": error.reason,
                "detail": error.detail,
            }
        else:
            document["records"] = {"G1": record}
            document["runtime_perf"] = build_report([record])
    _write_json(report, document)
    print(
        f"soak memory {run.status}: {len(run.turns)} turns, "
        f"{len(run.window_slopes())} window slopes, {run.wall_s:.1f}s",
        flush=True,
    )
    return 0 if run.status == "ok" else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run one opt-in full-scale soak, fail-closed on egress.

    The paths come from the WP1 guard (a canonical run directory strictly inside
    the session scratchpad, a report strictly inside it), the egress guard closes
    the network and strips provider credentials, and the report carries the soak
    record plus — for a gated run — the validated runtime-perf records: F1/F2 in
    the latency mode (the default) and G1 in the memory mode.
    """
    args = _parser().parse_args(argv)
    paths = current_perf_paths()
    run_dir = paths.run_dir
    report = paths.report if args.report is None else Path(args.report)
    if run_dir is None:
        print("run directory not configured (CHARTREUX_PERF_RUN_DIR)", flush=True)
        return 2
    if report is None:
        print(
            "report path not configured (--report or CHARTREUX_PERF_REPORT)", flush=True
        )
        return 2
    guard = EgressGuard(run_dir / "egress").install()
    try:
        guard.assert_closed()
        if args.mode == SOAK_MODE_MEMORY:
            return _memory_cell(run_dir, report, guard)
        run = asyncio.run(
            run_latency_soak(LATENCY_SOAK_FULL, session_root=run_dir / "sessions")
        )
        guard.assert_closed()
        document = run.report()
        if document["gated"]:
            try:
                records = run.gated_records()
            except SoakError as error:
                document["records_refused"] = {
                    "reason": error.reason,
                    "detail": error.detail,
                }
            else:
                ordered = [records[key] for key in sorted(records)]
                document["records"] = {
                    record["campaign_id"]: record for record in ordered
                }
                document["runtime_perf"] = build_report(ordered)
        _write_json(report, document)
        print(
            f"soak latency {run.status}: {len(run.turns)} turns, "
            f"{len(run.late_window_rows())} late-window heartbeat rows, "
            f"{run.wall_s:.1f}s",
            flush=True,
        )
        return 0 if run.status == "ok" else 1
    finally:
        guard.restore()


if __name__ == "__main__":  # pragma: no cover - external runner entry point
    raise SystemExit(main())
