"""Save-loaded producer cell for C1-C3 (v0.7.5 stage 2, P5-Reanchor).

The cell measures the LANDED P5 save path, not the superseded on-loop scan:
``save_interaction`` runs ``_check_transcript_persistence`` in a lock-scoped
``asyncio.to_thread`` worker inside ``_save_lock`` BEFORE the staleness fence
(``chartreux/core/session/session_logger.py:1028-1032`` is the worker call,
``:1035-1037`` the fence, ``:1123-1149`` the timed wrapper).

Three co-measured rows come out of one save run set:

* C1 (``save_worker_scan_p95_ms``): the worker's own synchronous execution of
  the wrapper, timed by the test-side worker wrapper. The worker thread
  identity is asserted distinct from the event-loop thread.
* C2 (``save_lock_held_await_p95_ms``): the lock-held await wall, from entry
  into the ``to_thread`` call until that await returns, INCLUDING the executor
  dispatch. The phase is that one await — never the full lock occupancy — and
  the write-phase ``persist`` await is observed separately as the boundary.
* C3 (``loop_heartbeat_excess_p95_ms``): loop heartbeat excess while the save
  workload runs (responsiveness, not scan cost).

Cell dimensions: the workload runs on a single-worker default executor, so the
recorded worker-thread identity is stable — the loop's default pool can
transiently spawn a second thread — and the campaign cell keeps saving past
the 100-save floor until the heartbeat rows reach the C3 floor, bounded by its
save cap. The production save path itself is untouched.

Every run record carries the row's chunks coverage through the WP6 schema
(``saves`` per run for C1/C2, heartbeat rows per run for C3), so a run with
missing coverage is ineligible downstream. In-suite contract nodes use a
shortened fixture (plan §5); the external campaign cell uses the committed
200k-equivalent fixture and records its digest.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from time import perf_counter_ns
from typing import Any
from unittest.mock import MagicMock

from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage
from chartreux.core.session import session_logger as session_logger_module
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config
from tests.perf._metrics import machine_context
from tests.perf._runtime_fixture import (
    FIXTURE_HASH,
    FIXTURE_TURN_WIDTH,
    FIXTURE_TURNS,
    build_fixture,
    fixture_digest,
)
from tests.perf._runtime_metrics import (
    AwaitSample,
    WorkerSample,
    current_perf_paths,
    observe_save_scan,
)
from tests.perf._runtime_report import SaveTimings, build_report, build_run_record

SAVE_CELL = "save_loaded_200k"
SAVE_CAMPAIGN_IDS: tuple[str, ...] = ("C1", "C2", "C3")
SAVE_TIMING_CAMPAIGN_IDS: tuple[str, ...] = ("C1", "C2")
SAVE_HEARTBEAT_CAMPAIGN_ID = "C3"
#: The worker call inside the lock, and the write-phase await that follows it.
SAVE_WORKER_TARGET = "_check_transcript_persistence"
SAVE_PERSIST_TARGET = "persist"
#: A4's pinned cadence for the heartbeat rows (plan §4 row A4, shared by C3).
SAVE_HEARTBEAT_INTERVAL_MS = 1.0
#: Floors of the save rows (Appendix B rows C1/C2/C3).
SAVE_MIN_MEASURED_SAVES = 100
SAVE_MIN_HEARTBEAT_ROWS = 1000
SAVE_WARMUP_SAVES = 10
#: Campaign-cell sizing. Measured first-hand on the 200k fixture at the
#: development HEAD: ~5.6 ms per save, and the loop heartbeat ticks every
#: ~4.9 ms while a save workload runs (1.07 ms when idle), so a save buys only
#: ~1.2 heartbeat rows. The campaign cell therefore extends past the 100-save
#: floor until 1000 rows are logged rather than pinning a save count that a
#: faster machine would undershoot; the 3000-save cap keeps the run inside the
#: §6 ~20 s per-run price. A run reports its actual coverage; a short run is
#: invalid, never relaxed.
SAVE_FULL_CELL_MAX_SAVES = 3000
#: In-suite contract-node shape (plan §5: shortened fixtures in-suite).
SAVE_CONTRACT_TURNS = 8
SAVE_CONTRACT_MEASURED_SAVES = SAVE_MIN_MEASURED_SAVES
SAVE_OPEN_CAMPAIGN_NOTE = "campaign open: no closure recorded for this run"

_NANOSECONDS_PER_MILLISECOND = 1_000_000


class SaveCellError(RuntimeError):
    """The save cell could not produce a coherent C1-C3 run set."""


@dataclass(frozen=True, slots=True)
class SaveCellSpec:
    """One save-cell run's shape: transcript size, save counts, cadence."""

    turns: int = FIXTURE_TURNS
    warmup_saves: int = SAVE_WARMUP_SAVES
    measured_saves: int = SAVE_MIN_MEASURED_SAVES
    #: Keep saving past ``measured_saves`` until this many heartbeat rows exist;
    #: ``None`` fixes the workload at ``measured_saves`` instead.
    heartbeat_row_target: int | None = SAVE_MIN_HEARTBEAT_ROWS
    max_measured_saves: int = SAVE_FULL_CELL_MAX_SAVES
    heartbeat_interval_ms: float = SAVE_HEARTBEAT_INTERVAL_MS
    label: str = SAVE_CELL

    def __post_init__(self) -> None:
        if not 0 < self.turns <= FIXTURE_TURNS:
            raise ValueError(f"turns must be in (0, {FIXTURE_TURNS}]: {self.turns}")
        if self.warmup_saves < 0 or self.measured_saves <= 0:
            raise ValueError("the cell needs measured saves and non-negative warmups")
        if self.max_measured_saves < self.measured_saves:
            raise ValueError("the save cap must cover the measured-save floor")
        if self.heartbeat_row_target is not None and self.heartbeat_row_target <= 0:
            raise ValueError("the heartbeat row target must be positive")
        if self.heartbeat_interval_ms <= 0:
            raise ValueError("the heartbeat interval must be positive")


#: The campaign cell: the committed 200k fixture, sized to the row floors.
CAMPAIGN_SAVE_CELL = SaveCellSpec()
#: The in-suite contract node's cell: a turn-aligned shortened fixture (§5),
#: with a fixed workload (it does not chase the campaign's heartbeat floor).
CONTRACT_SAVE_CELL = SaveCellSpec(
    turns=SAVE_CONTRACT_TURNS,
    measured_saves=SAVE_CONTRACT_MEASURED_SAVES,
    heartbeat_row_target=None,
    label="save_loaded_shortened",
)


def save_cell_messages(turns: int = FIXTURE_TURNS) -> list[LLMMessage]:
    """Return the cell transcript: a turn-aligned prefix of the committed fixture."""
    if not 0 < turns <= FIXTURE_TURNS:
        raise ValueError(f"turns must be in (0, {FIXTURE_TURNS}]: {turns}")
    fixture = build_fixture()
    if turns == FIXTURE_TURNS:
        return fixture
    return fixture[: turns * FIXTURE_TURN_WIDTH]


@dataclass(frozen=True, slots=True)
class HeartbeatRow:
    """One logged heartbeat tick of the C3 loop observer."""

    index: int
    scheduled_ms: float
    observed_ms: float
    excess_ms: float
    started_ns: int
    finished_ns: int


class LoopHeartbeat:
    """A scheduled-cadence loop heartbeat for the C3 row.

    Each row records the observed interval between consecutive wake-ups and
    its excess over the scheduled cadence; a heartbeat that wakes early is not
    an excess, so the excess clamps at zero.
    """

    def __init__(self, interval_ms: float = SAVE_HEARTBEAT_INTERVAL_MS) -> None:
        if interval_ms <= 0:
            raise ValueError("the heartbeat interval must be positive")
        self.interval_ms = interval_ms
        self.rows: list[HeartbeatRow] = []
        self._stop: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the heartbeat task on the running loop."""
        if self._task is not None:
            raise RuntimeError("the heartbeat is already started")
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._beat(), name="perf-save-cell-heartbeat")

    async def stop(self) -> None:
        """Stop the heartbeat task and let the in-flight row finish."""
        if self._stop is None or self._task is None:
            raise RuntimeError("the heartbeat is not started")
        self._stop.set()
        await self._task
        self._task = None
        self._stop = None

    async def _beat(self) -> None:
        assert self._stop is not None
        interval_s = self.interval_ms / 1000.0
        while not self._stop.is_set():
            started = perf_counter_ns()
            await asyncio.sleep(interval_s)
            finished = perf_counter_ns()
            observed_ms = (finished - started) / _NANOSECONDS_PER_MILLISECOND
            self.rows.append(
                HeartbeatRow(
                    index=len(self.rows),
                    scheduled_ms=self.interval_ms,
                    observed_ms=observed_ms,
                    excess_ms=max(0.0, observed_ms - self.interval_ms),
                    started_ns=started,
                    finished_ns=finished,
                )
            )


@dataclass(frozen=True, slots=True)
class SaveCellRun:
    """One completed save-cell run: the measured C1/C2/C3 sample sets."""

    spec: SaveCellSpec
    fixture_hash: str
    loop_thread_ident: int
    worker_samples: tuple[WorkerSample, ...]
    lock_await_samples: tuple[AwaitSample, ...]
    persist_await_samples: tuple[AwaitSample, ...]
    heartbeat_rows: tuple[HeartbeatRow, ...]
    warmup_heartbeat_rows: int
    total_heartbeat_rows: int

    @property
    def measured_saves(self) -> int:
        """Return the successful measured saves (C1/C2's unit count)."""
        return len(self.worker_samples)

    @property
    def measured_heartbeat_rows(self) -> int:
        """Return the heartbeat rows logged inside the measured save window."""
        return len(self.heartbeat_rows)

    def worker_thread_idents(self) -> tuple[int, ...]:
        """Return the distinct worker thread identities, in first-seen order."""
        idents: list[int] = []
        for sample in self.worker_samples:
            if sample.thread_ident not in idents:
                idents.append(sample.thread_ident)
        return tuple(idents)

    def single_worker_thread_ident(self) -> int:
        """Return the one worker identity, raising if the pool rotated threads."""
        idents = self.worker_thread_idents()
        if len(idents) != 1:
            raise SaveCellError(
                f"the scan ran on {len(idents)} worker threads: {idents}"
            )
        return idents[0]

    def worker_ms(self) -> list[float]:
        """Return the C1 samples: the worker's own execution wall."""
        return [sample.duration_ms for sample in self.worker_samples]

    def lock_await_ms(self) -> list[float]:
        """Return the C2 samples: the lock-held await wall."""
        return [sample.duration_ms for sample in self.lock_await_samples]

    def heartbeat_excess_ms(self) -> list[float]:
        """Return the C3 samples: per-row heartbeat excess."""
        return [row.excess_ms for row in self.heartbeat_rows]

    def coverage_counts(self) -> dict[str, int]:
        """Return the C1/C2 chunks-coverage counts: saves per run."""
        return {"saves": self.measured_saves}


@dataclass(frozen=True, slots=True)
class _WorkloadOutcome:
    """The heartbeat facts of one driven save workload."""

    loop_thread_ident: int
    attempted_saves: int
    heartbeat_rows: tuple[HeartbeatRow, ...]
    warmup_heartbeat_rows: int
    total_heartbeat_rows: int


def _save_inputs(
    spec: SaveCellSpec, save_root: Path, session_prefix: str
) -> tuple[SessionLogger, ChartreuxConfigSchema, AgentStats, MagicMock]:
    """Build the logger and the save call's argument fixtures."""
    session_config = SessionLoggingConfig(
        save_dir=str(save_root), session_prefix=session_prefix, enabled=True
    )
    logger = SessionLogger(session_config, f"{session_prefix}-{spec.label}")
    tool_manager = MagicMock(spec=ToolManager)
    tool_manager.available_tools = {}
    tool_manager.available_tool_specs.return_value = []
    return logger, build_test_vibe_config(), AgentStats(), tool_manager


def _needs_more_saves(spec: SaveCellSpec, attempts: int, measured_rows: int) -> bool:
    """Return whether the workload still owes the cell a save.

    The save floor is binding; ``heartbeat_row_target`` extends the workload
    past it until the C3 row floor can be met, and ``max_measured_saves``
    bounds that extension.
    """
    if attempts >= spec.max_measured_saves:
        return False
    if attempts < spec.measured_saves:
        return True
    target = spec.heartbeat_row_target
    return target is not None and measured_rows < target


async def _drive_saves(
    spec: SaveCellSpec,
    logger: SessionLogger,
    messages: list[LLMMessage],
    config: ChartreuxConfigSchema,
    stats: AgentStats,
    tool_manager: ToolManager,
) -> _WorkloadOutcome:
    """Run the warmup saves, then the measured saves, under the heartbeat."""
    # The cell pins ONE executor worker. The loop's default pool transiently
    # spawns a second thread when a submit lands before the finishing worker
    # re-acquires its idle token, which would make the recorded worker identity
    # a race; a single-worker executor makes it a fact without touching the
    # production save path. ``asyncio.run`` shuts this executor down.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="perf-save-cell")
    )
    heartbeat = LoopHeartbeat(spec.heartbeat_interval_ms)
    heartbeat.start()
    try:
        for _ in range(spec.warmup_saves):
            await logger.save_interaction(
                messages, stats=stats, config=config, tool_manager=tool_manager
            )
        # The measured window opens here; rows completed from this point are the
        # C3 series, so a row straddling the boundary counts with the window it
        # completed in.
        warmup_rows = len(heartbeat.rows)
        attempts = 0
        while _needs_more_saves(spec, attempts, len(heartbeat.rows) - warmup_rows):
            await logger.save_interaction(
                messages, stats=stats, config=config, tool_manager=tool_manager
            )
            attempts += 1
    finally:
        await heartbeat.stop()
    rows = tuple(heartbeat.rows)
    return _WorkloadOutcome(
        loop_thread_ident=threading.get_ident(),
        attempted_saves=attempts,
        heartbeat_rows=rows[warmup_rows:],
        warmup_heartbeat_rows=warmup_rows,
        total_heartbeat_rows=len(rows),
    )


def _measured_samples(spec: SaveCellSpec, samples: Sequence[Any]) -> list[Any]:
    """Drop the warmup saves' samples, keeping the measured save samples."""
    return list(samples[spec.warmup_saves :])


def _successful(worker: WorkerSample, scan: AwaitSample, persist: AwaitSample) -> bool:
    """Return whether all three recorded phases of one save returned normally."""
    return (
        worker.outcome == "returned"
        and scan.outcome == "returned"
        and persist.outcome == "returned"
    )


def run_save_cell(
    spec: SaveCellSpec, *, save_root: Path, session_prefix: str = "save-cell"
) -> SaveCellRun:
    """Drive one save-cell run on the landed P5 save path and collect C1-C3."""
    messages = save_cell_messages(spec.turns)
    logger, config, stats, tool_manager = _save_inputs(spec, save_root, session_prefix)
    loop_thread_ident = threading.get_ident()
    with observe_save_scan(logger, session_logger_module) as scan:
        outcome = asyncio.run(
            _drive_saves(spec, logger, messages, config, stats, tool_manager)
        )
    if outcome.loop_thread_ident != loop_thread_ident:
        raise SaveCellError(
            "the cell's workload did not run on the calling thread's loop"
        )

    worker_samples = _measured_samples(spec, scan.worker.samples)
    scan_awaits = _measured_samples(
        spec, scan.lock_await.for_target(SAVE_WORKER_TARGET)
    )
    persist_awaits = _measured_samples(
        spec, scan.lock_await.for_target(SAVE_PERSIST_TARGET)
    )
    if not (
        len(worker_samples)
        == len(scan_awaits)
        == len(persist_awaits)
        == outcome.attempted_saves
    ):
        raise SaveCellError(
            "the save phases disagree on the measured save count: "
            f"{len(worker_samples)} scans, {len(scan_awaits)} lock-held awaits, "
            f"{len(persist_awaits)} writes for {outcome.attempted_saves} saves"
        )
    kept = [
        (worker, scan_sample, persist)
        for worker, scan_sample, persist in zip(
            worker_samples, scan_awaits, persist_awaits, strict=True
        )
        if _successful(worker, scan_sample, persist)
    ]
    if not kept:
        raise SaveCellError("no save completed successfully in the measured window")

    transcript = logger.messages_filepath
    if not transcript.exists() or transcript.stat().st_size == 0:
        raise SaveCellError("the save cell persisted no transcript")

    return SaveCellRun(
        spec=spec,
        fixture_hash=fixture_digest(messages),
        loop_thread_ident=outcome.loop_thread_ident,
        worker_samples=tuple(item[0] for item in kept),
        lock_await_samples=tuple(item[1] for item in kept),
        persist_await_samples=tuple(item[2] for item in kept),
        heartbeat_rows=outcome.heartbeat_rows,
        warmup_heartbeat_rows=outcome.warmup_heartbeat_rows,
        total_heartbeat_rows=outcome.total_heartbeat_rows,
    )


def build_save_records(
    run: SaveCellRun,
    *,
    run_id: str,
    run_start_utc: str,
    closure_status: str | None = None,
    closure_note: str | None = SAVE_OPEN_CAMPAIGN_NOTE,
    machine: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the C1, C2 and C3 run records of one save-cell run.

    The C1/C2 records share one ``SaveTimings`` pair of phases (each record
    gates its own phase), carry the run's ``saves`` coverage, and name the
    distinct worker and loop thread identities. The C3 record carries the
    measured heartbeat rows and the discarded warmup rows.
    """
    saves = run.measured_saves
    timings = SaveTimings(
        worker_scan_ms=run.worker_ms(),
        lock_held_await_ms=run.lock_await_ms(),
        worker_thread_ident=run.single_worker_thread_ident(),
        loop_thread_ident=run.loop_thread_ident,
    )
    context = dict(machine_context() if machine is None else machine)
    counts = run.coverage_counts()
    records: dict[str, dict[str, Any]] = {}
    for campaign_id in SAVE_TIMING_CAMPAIGN_IDS:
        records[campaign_id] = build_run_record(
            campaign_id,
            run_id=run_id,
            run_start_utc=run_start_utc,
            measured_ops=saves,
            warmup_ops=run.spec.warmup_saves,
            unit_counts=counts,
            save_timings=timings,
            closure_status=closure_status,
            closure_note=closure_note,
            fixture_hash=run.fixture_hash,
            machine=context,
        )
    records[SAVE_HEARTBEAT_CAMPAIGN_ID] = build_run_record(
        SAVE_HEARTBEAT_CAMPAIGN_ID,
        run_id=run_id,
        run_start_utc=run_start_utc,
        measured_ops=run.measured_heartbeat_rows,
        warmup_ops=run.warmup_heartbeat_rows,
        samples=run.heartbeat_excess_ms(),
        closure_status=closure_status,
        closure_note=closure_note,
        fixture_hash=run.fixture_hash,
        machine=context,
    )
    return records


def report_destination() -> Path | None:
    """Return this run's configured report path, validated by the path guard."""
    return current_perf_paths().report


def write_run_report(records: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    """Validate and write the run report, returning the written path."""
    document = build_report(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


__all__ = [
    "CAMPAIGN_SAVE_CELL",
    "CONTRACT_SAVE_CELL",
    "FIXTURE_HASH",
    "SAVE_CAMPAIGN_IDS",
    "SAVE_CELL",
    "SAVE_CONTRACT_MEASURED_SAVES",
    "SAVE_CONTRACT_TURNS",
    "SAVE_FULL_CELL_MAX_SAVES",
    "SAVE_HEARTBEAT_INTERVAL_MS",
    "SAVE_MIN_HEARTBEAT_ROWS",
    "SAVE_MIN_MEASURED_SAVES",
    "SAVE_TIMING_CAMPAIGN_IDS",
    "SAVE_WARMUP_SAVES",
    "HeartbeatRow",
    "LoopHeartbeat",
    "SaveCellError",
    "SaveCellRun",
    "SaveCellSpec",
    "build_save_records",
    "report_destination",
    "run_save_cell",
    "save_cell_messages",
    "write_run_report",
]
