"""Shared regression-recording mechanics for the added §6.3 differential cases.

The base differential suite records every trace's normalized digest in the
per-family ledger (``tests.differential.ledger``) and the both-mode gate
compares the two transport runs. Point 1 of v0.7.9 adds seven stateful
race/partial-outcome cases to that suite (``stdio-parity-design.md`` §6.3:
mutation-timeout, root-death, resume, stale-snapshot,
concurrent-recovery-admission, revocation-race, and MCP partial-convergence).

Each case lands its test node with the carrier owner that implements the
behavior. This module is the shared plumbing those nodes use, so a case records
through one path and the both-mode gate can discover it instead of hardcoding:

* :data:`ADDED_CASES` declares each case once: its ledger family, the chunk that
  owns the node, and the trace module that hosts it. A case is *landed* only
  when its node exists. The gate ignores a case that has not landed, so this
  scaffolding chunk passes before any case node lands and a landed case is
  gated automatically.
* :func:`record_case` normalizes the trace and records the per-mode digest in
  the shared ledger, exactly as the base families do.
* The same call maintains the checked-in regression baseline. By default a
  trace whose normalized digest drifts from the recorded baseline fails, and
  :data:`REFRESH_ENV` (``VIBE_TEST_DIFFERENTIAL_REFRESH=1``) rewrites the
  baseline instead. A landed case with no baseline entry fails with the refresh
  instruction rather than passing silently.

The refresh marker is deliberately outside the ``CHARTREUX_`` namespace for the
same reason as the ledger and egress markers: the configuration layer treats an
unknown ``CHARTREUX_*`` name as a fatal configuration error, so a harness marker
there would break every spawned agent that loads configuration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path

from tests.differential import ledger
from tests.differential.replay import digest, normalize_frames

REFRESH_ENV = "VIBE_TEST_DIFFERENTIAL_REFRESH"

#: The checked-in regression baseline. Read on every recorded run; rewritten
#: only under :data:`REFRESH_ENV`, which is an explicit developer action.
BASELINE_PATH = Path(__file__).with_name("regression_baseline.json")

MUTATION_TIMEOUT = "mutation-timeout"
ROOT_DEATH = "root-death"
RESUME = "resume"
STALE_SNAPSHOT = "stale-snapshot"
CONCURRENT_RECOVERY = "concurrent-recovery-admission"
REVOCATION_RACE = "revocation-race"
MCP_PARTIAL_CONVERGENCE = "mcp-partial-convergence"


@dataclass(frozen=True)
class CaseSpec:
    """One added §6.3 case: its owner, ledger family, and recording options."""

    case_id: str
    owner: str
    family: str
    trace_module: str
    records_to_ledger: bool = True
    baseline: bool = True
    landed: bool = False


#: Every added §6.3 case and the chunk that lands its node. ``landed`` flips in
#: the owner's chunk, together with the node and (for a baselined case) the
#: refreshed baseline entry.
ADDED_CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        MUTATION_TIMEOUT,
        "C2c",
        ledger.FAMILY_MUTATION_TIMEOUT,
        "tests/differential/test_p_parity.py",
        records_to_ledger=False,
        landed=True,
    ),
    CaseSpec(
        ROOT_DEATH,
        "C2b",
        ledger.FAMILY_ROOT_DEATH,
        "tests/differential/test_parity_root.py",
        landed=True,
    ),
    CaseSpec(
        RESUME,
        "C2b",
        ledger.FAMILY_RESUME,
        "tests/differential/test_parity_root.py",
        landed=True,
    ),
    CaseSpec(
        STALE_SNAPSHOT,
        "C2a",
        ledger.FAMILY_STALE_SNAPSHOT,
        "tests/differential/test_parity_root.py",
        landed=True,
    ),
    CaseSpec(
        CONCURRENT_RECOVERY,
        "C1g",
        ledger.FAMILY_CONCURRENT_RECOVERY,
        "tests/differential/test_parity_root.py",
        landed=True,
    ),
    CaseSpec(
        REVOCATION_RACE,
        "C1c",
        ledger.FAMILY_REVOCATION_RACE,
        "tests/differential/test_p_parity.py",
        records_to_ledger=False,
        landed=True,
    ),
    CaseSpec(
        MCP_PARTIAL_CONVERGENCE,
        "C1h",
        ledger.FAMILY_MCP_CONVERGENCE,
        "tests/differential/test_parity_root.py",
        landed=True,
    ),
)


def refresh_enabled() -> bool:
    """Whether this run rewrites the regression baseline instead of asserting it."""
    return bool(os.environ.get(REFRESH_ENV, ""))


def recording_enabled() -> bool:
    """Whether a trace records at all: the gate's ledger marker or a refresh run."""
    return ledger.enabled() or refresh_enabled()


def case(case_id: str) -> CaseSpec:
    """The declared spec for ``case_id``; an undeclared case is a hard error."""
    for spec in ADDED_CASES:
        if spec.case_id == case_id:
            return spec
    raise KeyError(f"undeclared differential case: {case_id!r}")


def landed_cases() -> tuple[CaseSpec, ...]:
    """The added cases whose nodes have landed (none while scaffolding)."""
    return tuple(spec for spec in ADDED_CASES if spec.landed)


def landed_modules() -> tuple[str, ...]:
    """The trace modules the both-mode gate must run for landed ledger cases."""
    return tuple(
        dict.fromkeys(
            spec.trace_module for spec in landed_cases() if spec.records_to_ledger
        )
    )


def expected_family_minimums() -> dict[str, int]:
    """One recorded trace per landed ledger case, grouped by family."""
    minimums: dict[str, int] = {}
    for spec in landed_cases():
        if spec.records_to_ledger:
            minimums[spec.family] = minimums.get(spec.family, 0) + 1
    return minimums


def merge_family_minimums(base: Mapping[str, int]) -> dict[str, int]:
    """The gate's family coverage: the base families plus landed case families."""
    merged = dict(base)
    for family, minimum in expected_family_minimums().items():
        merged[family] = merged.get(family, 0) + minimum
    return merged


def read_baseline(path: Path | None = None) -> dict[str, dict[str, object]]:
    """Read the regression baseline, keyed by case id; an absent file is empty."""
    target = BASELINE_PATH if path is None else Path(path)
    if not target.exists():
        return {}
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"malformed regression baseline: {target}")
    entries: dict[str, dict[str, object]] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"malformed regression baseline entry: {target}")
        entries[str(key)] = entry
    return entries


def _write_baseline(
    path: Path, case_id: str, key: str, observed: str, frames: int
) -> None:
    entries = read_baseline(path)
    entries[case_id] = {"key": key, "digest": observed, "frames": frames}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(entries, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def record_case(
    spec: CaseSpec,
    script: object,
    frames: Iterable[Mapping[str, object]],
    *,
    baseline_path: Path | None = None,
) -> str:
    """Record one added §6.3 trace and return its normalized digest.

    The cross-mode digest goes through the shared ledger when the case uses it
    (the ledger marker gates the write, exactly as for the base families). The
    checked-in baseline is asserted unless :data:`REFRESH_ENV` is set, in which
    case it is rewritten. A landed case that has no baseline entry fails with
    the refresh instruction instead of passing silently.
    """
    normalized = normalize_frames(frames)
    observed = digest(normalized)
    if spec.records_to_ledger:
        ledger.record(spec.family, script, normalized)
    if not spec.baseline:
        return observed

    path = BASELINE_PATH if baseline_path is None else Path(baseline_path)
    if refresh_enabled():
        _write_baseline(
            path,
            spec.case_id,
            ledger.trace_key(spec.family, script),
            observed,
            len(normalized),
        )
        return observed

    entry = read_baseline(path).get(spec.case_id)
    if entry is None:
        if spec.landed:
            raise AssertionError(
                f"no recorded regression baseline for {spec.case_id!r}; "
                f"refresh it once with {REFRESH_ENV}=1"
            )
        return observed
    recorded = str(entry.get("digest", ""))
    if recorded != observed:
        raise AssertionError(
            f"regression baseline drift for {spec.case_id!r}: "
            f"{recorded} != {observed}; if the change is intended, "
            f"refresh with {REFRESH_ENV}=1"
        )
    return observed


__all__ = [
    "ADDED_CASES",
    "BASELINE_PATH",
    "CONCURRENT_RECOVERY",
    "MCP_PARTIAL_CONVERGENCE",
    "MUTATION_TIMEOUT",
    "REFRESH_ENV",
    "RESUME",
    "REVOCATION_RACE",
    "ROOT_DEATH",
    "STALE_SNAPSHOT",
    "CaseSpec",
    "case",
    "expected_family_minimums",
    "landed_cases",
    "landed_modules",
    "merge_family_minimums",
    "read_baseline",
    "record_case",
    "recording_enabled",
    "refresh_enabled",
]
