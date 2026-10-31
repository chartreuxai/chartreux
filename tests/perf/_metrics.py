from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter_ns
import tracemalloc

_INVOCATION_CWD: Path = Path.cwd()
_MIB = 1024 * 1024


def record(scenario: str, metrics: dict[str, object]) -> None:
    """Print a metrics blob and optionally append it to the report file."""
    blob = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    line = f"PERF-BLOB:{scenario}:{blob}"
    print(line)

    if report_path := os.environ.get("CHARTREUX_PERF_REPORT"):
        report_file_path = Path(report_path)
        if not report_file_path.is_absolute():
            report_file_path = _INVOCATION_CWD / report_file_path
        with report_file_path.open("a", encoding="utf-8") as report_file:
            report_file.write(f"{line}\n")


def _percentile(sorted_samples: list[float], fraction: float) -> float:
    position = (len(sorted_samples) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_samples) - 1)
    weight = position - lower_index
    return (
        sorted_samples[lower_index] * (1 - weight)
        + sorted_samples[upper_index] * weight
    )


def percentiles(samples: list[float]) -> dict[str, float]:
    """Return summary statistics using linearly interpolated percentiles."""
    if not samples:
        raise ValueError("percentiles requires at least one sample")

    ordered = sorted(samples)
    return {
        "min": round(ordered[0], 3),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "p99": round(_percentile(ordered, 0.99), 3),
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def _cpu_model() -> str:
    processor = platform.processor()
    if processor:
        return processor

    try:
        cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return platform.machine()

    for line in cpu_info.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"model name", "hardware"}:
            return value.strip()
    return platform.machine()


def _current_commit() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            cwd=repository,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def machine_context() -> dict[str, object]:
    """Return basic host and checkout information for a performance report."""
    try:
        loadavg: tuple[float, float, float] | None = os.getloadavg()
    except (AttributeError, OSError):
        loadavg = None

    return {
        "cpu_model": _cpu_model(),
        "core_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "loadavg": loadavg,
        "commit": _current_commit(),
    }


def rss_mib() -> float:
    """Return this process's current resident set size in MiB."""
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return rss_peak_mib()
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 3)
    return rss_peak_mib()


def rss_peak_mib() -> float:
    """Return the process RSS high-water mark in MiB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (AttributeError, OSError):
        return 0.0
    # ru_maxrss is KiB on Linux and bytes on macOS.
    divisor = 1024 if sys.platform == "darwin" else 1
    return round(usage.ru_maxrss / divisor / 1024, 3)


@dataclass(slots=True)
class AllocTrace:
    """Traced allocation for one measured block, in MiB."""

    baseline_mib: float = 0.0
    peak_mib: float = 0.0
    net_mib: float = 0.0


@contextmanager
def alloc_peak() -> Iterator[AllocTrace]:
    """Trace allocations inside the block and report peak and net MiB.

    ``tracemalloc`` inflates allocation-heavy loops several times over, so
    latency must be measured with tracing stopped; this is for memory only.
    """
    trace = AllocTrace()
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    trace.baseline_mib = round(baseline / _MIB, 4)
    try:
        tracemalloc.reset_peak()
        yield trace
    finally:
        current, peak = tracemalloc.get_traced_memory()
        trace.peak_mib = round(peak / _MIB, 4)
        trace.net_mib = round((current - baseline) / _MIB, 4)
        if not already_tracing:
            tracemalloc.stop()


def alloc_snapshot() -> tracemalloc.Snapshot:
    """Take a traced allocation snapshot, starting tracing if necessary."""
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    return tracemalloc.take_snapshot()


def alloc_breakdown_mib(
    before: tracemalloc.Snapshot, after: tracemalloc.Snapshot, *, limit: int = 8
) -> list[dict[str, object]]:
    """Attribute traced growth between two snapshots to source locations.

    The per-line entries are what a caller buckets into residency categories
    (parsed records, raw text buffers, serialization, atomic-write cost).
    """
    entries: list[dict[str, object]] = []
    for statistic in after.compare_to(before, "lineno")[:limit]:
        frame = statistic.traceback[0] if statistic.traceback else None
        source = (
            f"{frame.filename}:{frame.lineno}" if frame is not None else "<unknown>"
        )
        entries.append({
            "source": source,
            "size_mib": round(statistic.size_diff / _MIB, 4),
            "count": statistic.count_diff,
        })
    return entries


def timed_ms(call: Callable[[], object]) -> float:
    """Return one call's wall-clock duration in milliseconds."""
    started = perf_counter_ns()
    call()
    return (perf_counter_ns() - started) / 1_000_000


def repeat_timed_ms(call: Callable[[], object], *, reps: int) -> list[float]:
    """Return one wall-clock millisecond sample per repetition."""
    return [timed_ms(call) for _ in range(reps)]


async def measure_loop_lag(
    work: Awaitable[object], *, interval_s: float = 0.001
) -> dict[str, object]:
    """Measure event-loop scheduling lag while awaiting ``work``.

    A heartbeat task sleeps in ``interval_s`` steps and records how much later
    than requested it resumed. A synchronous block inside ``work`` (an admission
    pass, for example) starves the heartbeat and shows up as a large sample.
    """
    loop = asyncio.get_running_loop()
    delays: list[float] = []
    finished = False

    async def heartbeat() -> None:
        while not finished:
            scheduled = loop.time()
            await asyncio.sleep(interval_s)
            delays.append((loop.time() - scheduled - interval_s) * 1000)

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    try:
        await work
    finally:
        finished = True
        await asyncio.gather(beat, return_exceptions=True)

    if not delays:
        return {"sample_count": 0}
    return {"sample_count": len(delays), **percentiles(delays)}
