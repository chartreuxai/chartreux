"""Per-trace digest ledger for the both-mode gate (WP6 [6D]).

The gate runs the differential suite twice — once with the process/stdio
transport selected and once with ``CHARTREUX_TRANSPORT=in_process`` — and
compares what each run recorded. A trace that wants to take part records its
normalized observation stream here; the gate then asserts that every distinct
seed-and-script pair produced the same digest in both modes.

The ledger directory comes from ``VIBE_TEST_DIFFERENTIAL_LEDGER``. The marker
is deliberately outside the ``CHARTREUX_`` namespace for the same reason as the
[6A] egress marker: the config layer treats an unknown ``CHARTREUX_*`` name as
a fatal configuration error, so a harness marker there would break every
spawned agent that loads configuration.

One file per trace key keeps parallel runs (and xdist workers) from racing:
a key is written by exactly one trace instance, and the gate merges the
directory afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path

from tests.differential.replay import canonical_json, digest, normalize_frames

LEDGER_ENV = "VIBE_TEST_DIFFERENTIAL_LEDGER"

FAMILY_PRESENTATION = "presentation"
FAMILY_ROOT = "root"
FAMILY_CHILD_CRASH = "child_crash"
FAMILY_REPLAY = "replay"
FAMILY_SHUTDOWN = "shutdown"


def enabled() -> bool:
    """Whether this run records a ledger (the gate's nested runs set the marker)."""
    return bool(os.environ.get(LEDGER_ENV, ""))


def trace_key(family: str, script: object) -> str:
    """A content-addressed key: the same script gets the same key in every mode."""
    material = canonical_json({"family": family, "script": script})
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def record(family: str, script: object, frames: Iterable[Mapping[str, object]]) -> None:
    """Record one trace's normalized digest; a no-op when the ledger is off."""
    directory = os.environ.get(LEDGER_ENV, "")
    if not directory:
        return
    normalized = normalize_frames(frames)
    key = trace_key(family, script)
    entry = {
        "family": family,
        "key": key,
        "digest": digest(normalized),
        "frames": len(normalized),
    }
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / f".{key}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target / f"{key}.json")


def read(directory: Path) -> dict[str, dict[str, object]]:
    """Read a ledger directory back, keyed by trace key."""
    entries: dict[str, dict[str, object]] = {}
    for path in sorted(Path(directory).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "key" not in value:
            raise ValueError(f"malformed ledger entry: {path}")
        entries[str(value["key"])] = value
    return entries


def families(entries: Mapping[str, Mapping[str, object]]) -> dict[str, int]:
    """Count recorded traces per family."""
    counts: dict[str, int] = {}
    for entry in entries.values():
        family = str(entry.get("family", ""))
        counts[family] = counts.get(family, 0) + 1
    return counts


__all__ = [
    "FAMILY_CHILD_CRASH",
    "FAMILY_PRESENTATION",
    "FAMILY_REPLAY",
    "FAMILY_ROOT",
    "FAMILY_SHUTDOWN",
    "LEDGER_ENV",
    "enabled",
    "families",
    "read",
    "record",
    "trace_key",
]
