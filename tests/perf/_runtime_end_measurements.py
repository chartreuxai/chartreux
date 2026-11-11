"""Offline end-measurement harness for the v0.7.7 runtime (WP9 [9A]).

The plan's WP9 chunk [9A] asks for a small set of seeded, egress-blocked,
single-machine observations of the multi-process runtime: PTY input and paint
gaps, the zero-delay turn wall, per-process and total RSS/PSS at 0 and 8
children, the coordinator's loop delay, CPU, shed frames and control latency,
and process start-to-ready. This module holds the recording surface those
observations share; ``test_runtime_end_measurements.py`` holds the nodes.

Nothing here gates a number. Every figure is emitted with its raw samples, its
statistics, and the two labels design.md:45-48 binds to a figure: the
interpreter/tree it was taken on and its evidence class (never compare an
installed-3.14.4 figure with a Python-3.12 clone figure; a replay figure is not
a live finding):

``offline``
    A same-tree measurement of the runtime itself, no provider involved.
``replay``
    A seeded, scripted-provider measurement: the transport and runtime paths
    are real, the model is a deterministic stub.
``live``
    A real provider turn. Not produced by this chunk (no live provider calls).
``ablation``
    A variant measurement against a control. Not produced by this chunk.

Output discipline: every artifact this harness writes lands OUTSIDE the
repository. ``results_root()`` refuses a path inside the checked-out tree, and
every temporary directory the measurements create is redirected there, because
the worker output boundary covers incidental in-repo writes too.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ENV = "CHARTREUX_WP9_RESULTS"
"""Optional override of the results directory; the default is external too."""

DEFAULT_RESULTS_ROOT = Path("/home/pav/.cache/chartreux-wp9/results")

MEASUREMENT_SEED = 20261111
"""The one seed every [9A] observation is taken with."""

SCHEMA = "chartreux.wp9.end_measurement.v1"

EVIDENCE_OFFLINE = "offline"
EVIDENCE_REPLAY = "replay"
EVIDENCE_LIVE = "live"
EVIDENCE_ABLATION = "ablation"
EVIDENCE_CLASSES = (EVIDENCE_OFFLINE, EVIDENCE_REPLAY, EVIDENCE_LIVE, EVIDENCE_ABLATION)

P99_MIN_SAMPLES = 100
"""Plan §9 acceptance: a p99 figure needs at least this many samples."""

SERIAL_NOTE = "serial-only: every [9A] node assumes -n0 (no xdist worker)"

_CLOCK_TICKS = int(os.sysconf("SC_CLK_TCK"))


class MeasurementError(RuntimeError):
    """A measurement could not be placed or taken; the reason is the message."""


def repository_root() -> Path:
    """The checked-out tree the observations are labeled with."""
    return REPO_ROOT


def results_root() -> Path:
    """The external directory every artifact of this chunk is written to."""
    raw = os.environ.get(RESULTS_ENV)
    root = Path(raw).expanduser() if raw else DEFAULT_RESULTS_ROOT
    root = Path(os.path.abspath(root))
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        raise MeasurementError(
            f"measurement results must stay outside the repository: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


_prepared_scratch: Path | None = None


def scratch_root() -> Path:
    """The external scratch directory, and the process's tempfile root.

    The first call redirects ``TMPDIR`` and ``tempfile.tempdir``: /tmp is
    inode-saturated on this machine and, more generally, nothing this chunk
    creates may land in a shared temp namespace it does not own.
    """
    global _prepared_scratch
    if _prepared_scratch is None:
        path = results_root() / "tmp"
        path.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(path)
        tempfile.tempdir = str(path)
        _prepared_scratch = path
    return _prepared_scratch


def prepare() -> None:
    """Redirect temp destinations before any measurement creates one."""
    scratch_root()


def machine_record() -> dict[str, Any]:
    """The machine an observation was taken on."""
    return {
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu_count": os.cpu_count(),
        "host": socket.gethostname(),
    }


def interpreter_record() -> dict[str, Any]:
    """The interpreter an observation was taken on (never mixed across runs)."""
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "executable": sys.executable,
        "build": sys.version,
    }


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return f"<git {arguments[0]} failed: {completed.stderr.strip()}>"
    return completed.stdout


def tree_record() -> dict[str, Any]:
    """The tree (git revision) an observation was taken on."""
    porcelain = [line for line in _git("status", "--porcelain").splitlines() if line]
    return {
        "revision": _git("rev-parse", "HEAD").strip(),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty_paths": porcelain,
    }


def producer_command() -> dict[str, Any]:
    """The command that produced a node's observations.

    ``argv`` is exactly what the running interpreter saw; ``command`` is the
    same invocation with the uv wrapper this chunk is run under, kept so the
    recorded command is the one a reader would re-issue.
    """
    argv = list(sys.argv)
    executable = Path(argv[0]).name or "pytest"
    return {
        "argv": argv,
        "command": "uv run --no-sync " + " ".join([executable, *argv[1:]]),
    }


def _nearest_rank(ordered: Sequence[float], fraction: float) -> float:
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarise(samples: Sequence[float]) -> dict[str, Any]:
    """Median/p99/min/max for one metric, with p99's eligibility stated."""
    if not samples:
        raise MeasurementError("an observation needs at least one sample")
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p99": (
            _nearest_rank(ordered, 0.99) if len(ordered) >= P99_MIN_SAMPLES else None
        ),
        "p99_eligible": len(ordered) >= P99_MIN_SAMPLES,
    }


@dataclass(frozen=True)
class Observation:
    """One measured quantity, with its raw samples and its labels."""

    metric: str
    topology: str
    unit: str
    samples: tuple[float, ...]
    evidence_class: str
    note: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise MeasurementError(f"unknown evidence class: {self.evidence_class}")

    def record(self) -> dict[str, Any]:
        """The JSON form: statistics plus the untouched raw samples."""
        return {
            "metric": self.metric,
            "topology": self.topology,
            "unit": self.unit,
            "evidence_class": self.evidence_class,
            "note": self.note,
            "statistics": summarise(self.samples),
            "samples": list(self.samples),
            "detail": dict(self.detail),
        }


def emit_family(
    name: str,
    observations: Sequence[Observation],
    *,
    seed: int = MEASUREMENT_SEED,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write one node's complete observation set into the results directory."""
    prepare()
    document = {
        "schema": SCHEMA,
        "family": name,
        "seed": seed,
        "serial_only": True,
        "note": SERIAL_NOTE,
        "command": producer_command()["command"],
        "argv": producer_command()["argv"],
        "producer_exit_status": 0,
        "tree": tree_record(),
        "interpreter": interpreter_record(),
        "machine": machine_record(),
        "started_utc": _utc_stamp(),
        "observations": [observation.record() for observation in observations],
        "extra": dict(extra or {}),
    }
    path = results_root() / f"{name}.json"
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _utc_stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def read_pss_bytes(pid: int) -> int | None:
    """Proportional set size of one process, or ``None`` when unavailable.

    PSS is the honest tree total: summing RSS counts a shared page once per
    process that maps it.
    """
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("Pss:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def read_command_line(pid: int, *, limit: int = 90) -> str:
    """A short, single-line form of a process's command line."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return "<gone>"
    text = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    return text[:limit]


def memory_rows(sample: Any) -> list[dict[str, Any]]:
    """Per-process RSS/PSS rows for one process-tree sample."""
    rows: list[dict[str, Any]] = []
    for process in sample.processes:
        rows.append({
            "pid": process.pid,
            "ppid": process.ppid,
            "rss_bytes": process.rss_bytes,
            "pss_bytes": read_pss_bytes(process.pid),
            "cpu_self_s": process.cpu_self_s,
            "command": read_command_line(process.pid),
        })
    return rows


def tree_memory(sample: Any) -> dict[str, Any]:
    """The tree's rows plus its RSS and PSS totals (PSS is the honest total)."""
    rows = memory_rows(sample)
    rss = [row["rss_bytes"] for row in rows if row["rss_bytes"] is not None]
    pss = [row["pss_bytes"] for row in rows if row["pss_bytes"] is not None]
    return {
        "outcome": sample.outcome,
        "reasons": list(sample.reasons),
        "process_count": len(rows),
        "rows": rows,
        "total_rss_bytes": sum(rss) if rss else None,
        "total_pss_bytes": sum(pss) if pss else None,
        "pss_missing": sum(1 for row in rows if row["pss_bytes"] is None),
    }


def own_cpu_seconds() -> float | None:
    """This process's user+system CPU time, from ``/proc/self/stat``."""
    from tests.perf._runtime_process_tree import PROC_ROOT, read_stat

    fields, _failure = read_stat(os.getpid(), PROC_ROOT)
    if fields is None:
        return None
    return (fields.utime_ticks + fields.stime_ticks) / _CLOCK_TICKS


class LoopLagObserver:
    """Event-loop heartbeat lateness at a fixed cadence.

    The heartbeat asks to wake every ``interval_s``; the lateness is how much
    later than that target it actually ran, which is what a busy coordinator
    loop shows as (the campaign's shared-loop signature, C1).
    """

    def __init__(self, *, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self.samples_ms: list[float] = []
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> LoopLagObserver:
        self._task = asyncio.create_task(self._beat())
        return self

    async def stop(self) -> list[float]:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return self.samples_ms

    async def _beat(self) -> None:
        loop = asyncio.get_running_loop()
        target = loop.time() + self.interval_s
        while True:
            await asyncio.sleep(max(0.0, target - loop.time()))
            now = loop.time()
            self.samples_ms.append(max(0.0, (now - target) * 1000.0))
            target += self.interval_s


async def measure_loop_window(
    *, seconds: float, interval_s: float = 0.01
) -> dict[str, Any]:
    """Loop delay, CPU time and CPU share over one quiescent window."""
    observer = LoopLagObserver(interval_s=interval_s)
    cpu_before = own_cpu_seconds()
    started = perf_counter()
    await observer.start()
    await asyncio.sleep(seconds)
    lags = await observer.stop()
    elapsed = perf_counter() - started
    cpu_after = own_cpu_seconds()
    cpu_s = (
        (cpu_after - cpu_before)
        if cpu_after is not None and cpu_before is not None
        else None
    )
    return {
        "loop_delay_ms": lags,
        "window_s": elapsed,
        "cpu_s": cpu_s,
        "cpu_share": (cpu_s / elapsed) if cpu_s is not None else None,
    }


def shed_frames_under_saturation(
    *, offered: int, payload_bytes: int, per_stream_budget: int
) -> dict[str, Any]:
    """Frames the relay hop sheds when a presentation stream exceeds its budget.

    A structural measurement of the shedding rule (design.md:465-470), not a
    live drop count: the hop is never started, so no frame ever leaves the
    per-stream FIFO and every frame past the budget is shed by construction.
    """
    from chartreux.app_server.relay import RelayHop

    class _NeverSend:
        async def send(self, message: dict[str, Any]) -> None:
            raise AssertionError("an unstarted hop must never send")

        def abort(self, reason: str) -> int:
            return 0

    hop = RelayHop(
        lambda _stream_id: _NeverSend(),
        presentation_per_stream=per_stream_budget,
        presentation_per_hop=per_stream_budget * 4,
        control_capacity=1024 * 1024,
    )
    hop.open_stream("measurement-stream")
    payload = "x" * payload_bytes
    outcomes = [
        hop.enqueue(
            "measurement-stream",
            {
                "jsonrpc": "2.0",
                "method": "turn/updated",
                "params": {"padding": payload},
            },
        )
        for _ in range(offered)
    ]
    dispositions = [str(outcome.disposition) for outcome in outcomes]
    return {
        "offered": offered,
        "shed": hop.shed_frames,
        "queued": dispositions.count("queued"),
        "stream_presentation_bytes": _stream_bytes(hop),
        "presentation_capacity": hop.presentation_capacity,
        "per_stream_budget": per_stream_budget,
    }


def _stream_bytes(hop: Any) -> int | None:
    stream = hop.stream("measurement-stream")
    return None if stream is None else stream.presentation_bytes


def child_environment(guard_dir: Path) -> dict[str, str]:
    """A spawned agent's environment: fail-closed egress, external homes."""
    from tests.differential import egress

    environment = dict(os.environ)
    environment.update(egress.child_environment(egress.MODE_ALL))
    environment["CHARTREUX_TRANSPORT"] = "multi_process"
    paths = [str(REPO_ROOT), str(guard_dir)]
    inherited = environment.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TMPDIR"] = str(scratch_root())
    return environment


def seed_content(seed: int, *, count: int) -> list[str]:
    """Deterministic per-turn content from the measurement seed."""
    import random

    generator = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    return [
        "".join(generator.choice(alphabet) for _ in range(24)).strip()
        for _ in range(count)
    ]


__all__ = [
    "DEFAULT_RESULTS_ROOT",
    "EVIDENCE_ABLATION",
    "EVIDENCE_CLASSES",
    "EVIDENCE_LIVE",
    "EVIDENCE_OFFLINE",
    "EVIDENCE_REPLAY",
    "MEASUREMENT_SEED",
    "P99_MIN_SAMPLES",
    "RESULTS_ENV",
    "SCHEMA",
    "LoopLagObserver",
    "MeasurementError",
    "Observation",
    "child_environment",
    "emit_family",
    "interpreter_record",
    "machine_record",
    "measure_loop_window",
    "memory_rows",
    "own_cpu_seconds",
    "prepare",
    "producer_command",
    "read_command_line",
    "read_pss_bytes",
    "repository_root",
    "results_root",
    "scratch_root",
    "seed_content",
    "shed_frames_under_saturation",
    "summarise",
    "tree_memory",
    "tree_record",
]
