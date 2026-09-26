from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess

_INVOCATION_CWD: Path = Path.cwd()


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
