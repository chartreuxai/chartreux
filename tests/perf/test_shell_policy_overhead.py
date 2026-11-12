"""Shell-policy overhead perf carrier — F1pkg1(b) (v0.7.1 wave-2 slot).

This file OWNS the two frozen pre-change denominators of the wave-2 shell-policy
slot and the per-case ceilings the post-change tree enforces:

1. Per-case corpus replay durations — every ``corpus.jsonl`` entry is replayed
   through its family observer (``test_shell_policy_corpus.observe_entry``) and
   timed individually, once per trial, for ``TRIALS`` trials.
2. Executor-pinning overhead — ``Bash.resolve_permission`` is timed per command
   over the shell-command battery (every corpus ``permission``/``executor``
   family argv plus the fixed shell-option/glob forms below), once per trial.

Ceiling form (per Amendment XX F7 the ceiling is per-case, never suite-total):
for each case, the median of its ``TRIALS`` samples must satisfy

    median <= max(frozen_median * CASE_CEILING_FACTOR,
                  frozen_median + CASE_CEILING_SLACK_S)

Corpus entries added AFTER the freeze (ids absent from the frozen map) are
bounded by their family's frozen p95 under the same factor+slack, so a correct
row addition never fails the suite while an unbounded per-row regression does.
A ceiling miss is investigated with per-trial attribution, never silently raised.

Freeze discipline (T1b/T2b precedent): the denominators were frozen BEFORE any
contract code change, on a git worktree PINNED at pre-change commit f8fd5823
("Release v0.7.0") with ONLY this carrier file copied in, run serial (-n0) from
the worktree root on a quiet machine (no concurrent test runs; loadavg recorded
with every freeze artifact). The frozen values live in the pure-data companion
file ``_shell_policy_frozen.json`` beside this carrier; the provenance comment
block records the worktree commit, the freeze timestamps, the trial batches, and
the machine context. While that data file is absent or a mapping is empty (the
PLACEHOLDER state), the ceilings are NOT enforced and every run records its raw
per-case samples as a freeze candidate.

Run post-change in the main tree, serial only, on a quiet machine:

    uv run pytest tests/perf/test_shell_policy_overhead.py -n0
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import json
import os
from pathlib import Path
import shlex
import statistics
import time
from typing import cast

import pytest

import chartreux
from chartreux.core.tools.base import BaseToolState
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from tests.perf._metrics import machine_context, record
from tests.tools import test_shell_policy_corpus as corpus

#: Number of measured trials per case (each trial times every case once).
TRIALS = 5

#: Per-case ceiling factor over the frozen median.
CASE_CEILING_FACTOR = 2.0

#: Per-case absolute slack (seconds) added to the scaled frozen median.
CASE_CEILING_SLACK_S = 0.002

#: The executor the resolve battery pins (the ambient SHELL of the freeze host).
PINNED_SHELL = "/usr/bin/zsh"

# ---------------------------------------------------------------------------
# FROZEN DENOMINATORS — landed by the F1pkg1(b) freeze on 2026-11-02.
#
# Provenance: frozen BEFORE any contract code change on a git worktree PINNED
# at pre-change commit f8fd5823 ("Release v0.7.0",
# f8fd58238aa820df43c9f0808b201fe385a5b52f) with ONLY this carrier file copied
# in, run serial (-n0) from the worktree root with PYTHONPATH=<worktree> so
# chartreux imports from the worktree (import probe in every freeze artifact:
# chartreux.__file__ = <worktree>/chartreux/__init__.py). Two freeze batches
# (A started 2026-11-02T14:10:24Z, B started 2026-11-02T14:10:56Z), each
# TRIALS=5 trials over 3703 corpus cases and 1432 resolve-battery cases;
# freeze artifacts: f1b-freeze-A.jsonl / f1b-freeze-B.jsonl (raw per-case
# samples), logs f1b-freeze-A.log / f1b-freeze-B.log. Frozen value per case =
# median of all 10 samples (both batches); family p95 = p95 of the per-case
# medians per corpus family plus the "resolve" battery family. Quiet machine,
# no concurrent test runs: AMD Ryzen 9 5900X 12-Core Processor, 24 cores,
# Linux-7.3.0-9-generic-x86_64 glibc 2.44, Python 3.12.13; loadavg 1-min
# 1.61-2.27 on 24 cores across both batches (recorded per freeze artifact).
# Post-change measured trials run in the main tree AFTER the contract landed;
# their timestamps postdate this freeze (see the acceptance record).
# The data file _shell_policy_frozen.json holds {case_id: median_seconds} for
# each scenario plus the family p95s.
# ---------------------------------------------------------------------------

# Fixed shell-command battery for the executor-pinning overhead denominator:
# the option-contract forms F1pkg1(b) re-decides, the glob forms, and wrapper
# shapes. The corpus-derived argv are added on top at runtime.
FIXED_RESOLVE_COMMANDS = (
    "echo hi",
    "bash -c 'echo hi'",
    "bash +O extglob -c 'echo hi'",
    "bash -O extglob -c 'echo hi'",
    "bash +o errexit -c 'echo hi'",
    "env CI=1 bash +o errexit -c 'echo hi'",
    "zsh -c 'echo hi'",
    "zsh -o extendedglob -c 'echo hi'",
    "zsh +O extendedglob -c 'echo hi'",
    "sh -c 'echo hi'",
    "bash --noprofile -ic 'echo hi'",
    "setopt extendedglob; echo hi",
    "shopt -s extglob; echo hi",
    "set -o extglob; echo hi",
    "cat *.py",
    "cat *.txt",
    "ls *.py",
    "wc *.md",
    "bash -c 'cat *.py'",
    "bash -c 'ls *'",
    "echo @(foo)",
    "echo ^foo",
    "echo foo#",
    "uv run python -c 'print(1)'",
    "bash -c 'uv run rm -r build'",
    "git status",
    "rm -r build",
)


#: Pure-data companion holding the frozen denominators (see the provenance
#: block above); a missing file or an empty mapping is the PLACEHOLDER.
_FROZEN_DATA_PATH = Path(__file__).parent / "_shell_policy_frozen.json"


def _load_frozen(mapping: str) -> dict[str, float] | None:
    """Load one frozen-denominator mapping; absent data decodes to None."""
    if not _FROZEN_DATA_PATH.is_file():
        return None
    payload = json.loads(_FROZEN_DATA_PATH.read_text(encoding="utf-8"))
    values = payload.get(mapping) or {}
    if not values:
        return None
    return {str(key): float(value) for key, value in values.items()}


_FROZEN_CORPUS_CASE_S = _load_frozen("corpus_case_s")
_FROZEN_RESOLVE_CASE_S = _load_frozen("resolve_case_s")
_FROZEN_FAMILY_P95_S = _load_frozen("family_p95_s")


def _ceiling(
    frozen: dict[str, float] | None,
    case: str,
    family_p95: dict[str, float] | None,
    family: str,
) -> float | None:
    """The per-case ceiling, or None while the denominators are PLACEHOLDER."""
    if frozen is None or family_p95 is None:
        return None
    base = frozen.get(case)
    if base is None:
        base = family_p95.get(family, 0.0)
    return max(base * CASE_CEILING_FACTOR, base + CASE_CEILING_SLACK_S)


def _corpus_fixtures(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize the corpus fixture tree the way the corpus suite does."""
    root = corpus.materialize_fixtures(
        corpus.FIXTURES_SOURCE, tmp_path_factory.mktemp("overhead-corpus")
    )
    for builder_name in (
        "build_recursive_rm_fixtures",
        "build_bare_metadata_fixtures",
        "build_curl_destination_fixtures",
        "build_glob_fixtures",
        "build_git_workflow_fixtures",
    ):
        builder = getattr(corpus, builder_name, None)
        if builder is not None:
            builder(root)
    return root


def _resolve_battery() -> list[str]:
    """The per-case resolve battery: corpus argv plus the fixed forms."""
    commands = {command for command in FIXED_RESOLVE_COMMANDS}
    for entry in corpus.load_corpus():
        family = str(entry.get("family"))
        if family in {
            "permission",
            "executor",
            "policy",
            "paths",
            "executor_aux",
            "divergence",
        }:
            if entry.get("raw") is not None:
                commands.add(str(entry["raw"]))
            elif isinstance(entry.get("argv"), list):
                commands.add(shlex.join([str(token) for token in entry["argv"]]))
    return sorted(commands)


def _time_cases(
    cases: Sequence[tuple[str, str, Callable[[], object]]], trials: int
) -> dict[str, list[float]]:
    """Time every case once per trial; return per-case sample lists."""
    samples: dict[str, list[float]] = {case_id: [] for case_id, _, _ in cases}
    for _trial in range(trials):
        for case_id, _family, call in cases:
            start = time.perf_counter()
            call()
            samples[case_id].append(time.perf_counter() - start)
    return samples


def _report_and_enforce(
    *,
    scenario: str,
    samples: dict[str, list[float]],
    frozen: dict[str, float] | None,
    family_p95: dict[str, float] | None,
    family_of: Callable[[str], str],
    machine: dict[str, object],
    loadavg: tuple[float, float, float] | None,
) -> None:
    medians = {case: statistics.median(values) for case, values in samples.items()}
    violations: list[dict[str, float | str | None]] = []
    for case, median in sorted(medians.items()):
        ceiling = _ceiling(frozen, case, family_p95, family_of(case))
        if ceiling is not None and median > ceiling:
            violations.append({
                "case": case,
                "median_s": median,
                "frozen_s": (frozen or {}).get(case),
                "ceiling_s": ceiling,
            })
    frozen_status = (
        "PLACEHOLDER — not enforced; raw samples recorded as freeze candidates"
        if frozen is None or family_p95 is None
        else "measured against the provenance-frozen per-case ceilings"
    )
    record(
        scenario,
        {
            "status": frozen_status,
            "trials": trials_count(),
            "case_count": len(medians),
            "per_case_median_s": medians,
            "per_case_samples_s": samples if frozen is None else None,
            "frozen_median_s": frozen,
            "family_p95_s": family_p95,
            "ceiling_factor": CASE_CEILING_FACTOR,
            "ceiling_slack_s": CASE_CEILING_SLACK_S,
            "violations": violations,
            "machine": machine,
            "loadavg_at_report": loadavg,
            "chartreux_file": chartreux.__file__,
        },
    )
    print(
        f"{scenario}: {len(medians)} cases x {trials_count()} trials; "
        f"ceiling enforcement: {'ON' if frozen is not None else 'OFF (placeholder)'}; "
        f"violations: {len(violations)}; chartreux from {chartreux.__file__}"
    )
    for violation in violations:
        median_s = cast(float, violation["median_s"])
        ceiling_s = cast(float, violation["ceiling_s"])
        frozen_s = cast(float, violation["frozen_s"] or 0)
        print(
            f"  CEILING MISS {violation['case']}: median "
            f"{median_s * 1_000:.3f} ms > ceiling "
            f"{ceiling_s * 1_000:.3f} ms (frozen "
            f"{frozen_s * 1_000:.3f} ms) — investigate with "
            "per-trial attribution, never raise the constant"
        )
    assert not violations, (
        f"{scenario}: {len(violations)} per-case ceilings exceeded; see the "
        "PERF-BLOB violations list"
    )


def trials_count() -> int:
    return TRIALS


@pytest.fixture
def pinned_shell(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the resolved executor so pre- and post-change runs are comparable.

    The process-level resolution — not the ambient ``$SHELL`` — is what the
    analysis reads since the shipped bash default was introduced, so the pin
    installs the frozen denominators' executor directly.
    """
    from chartreux.core.utils import shell as shell_module

    monkeypatch.setenv("SHELL", PINNED_SHELL)
    shell_module.apply_executor_selection(PINNED_SHELL)
    try:
        yield
    finally:
        shell_module.reset_executor_selection()


@pytest.fixture
def hermetic_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the hostname resolver seam for the resolve battery (S4b2a).

    Since v0.7.1 S4b2a, ``resolve_permission`` on an ssh hostname row runs the
    resolve-validate model, so the battery must not do real DNS (the frozen
    pre-change denominators were recorded without any hostname resolution).
    The corpus module's fixed answer table is reused, so corpus replay and
    the resolve battery measure the same hermetic resolution.
    """
    from chartreux.core.tools.builtins import _remote_exec

    monkeypatch.setattr(
        _remote_exec, "_hostname_resolver", corpus.fixed_hostname_resolver
    )


@pytest.mark.perf
@pytest.mark.timeout(900)
def test_shell_policy_overhead_smoke(
    tmp_path_factory: pytest.TempPathFactory,
    pinned_shell: None,
    hermetic_resolution: None,
) -> None:
    """SMOKE CHECK ONLY: one pass over both case sets; numbers are NOT
    denominators (the pinned-worktree freeze owns those). Proves the harness
    executes, the corpus module and chartreux import from the running tree,
    and the fixture tree materializes.
    """
    root = _corpus_fixtures(tmp_path_factory)
    entries = corpus.load_corpus()
    assert entries, "corpus.jsonl must load"
    battery = _resolve_battery()
    corpus_cases = [
        (
            str(entry["id"]),
            str(entry["family"]),
            (lambda e=entry: corpus.observe_entry(e, root)),
        )
        for entry in entries
    ]
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=root)
    resolve_cases = [
        (
            f"resolve:{command}",
            "resolve",
            (lambda c=command: tool.resolve_permission(BashArgs(command=c))),
        )
        for command in battery
    ]
    machine = machine_context()
    loadavg = os.getloadavg()
    corpus_samples = _time_cases(corpus_cases, 1)
    resolve_samples = _time_cases(resolve_cases, 1)
    print(
        f"shell-policy-overhead SMOKE (NOT a denominator): {len(corpus_cases)} "
        f"corpus cases + {len(resolve_cases)} resolve cases executed once; "
        f"chartreux from {chartreux.__file__}; machine {machine}"
    )
    record(
        "shell_policy_overhead_smoke",
        {
            "status": "SMOKE ONLY — not a denominator; the pinned-worktree freeze "
            "owns the denominators",
            "corpus_cases": len(corpus_cases),
            "resolve_cases": len(resolve_cases),
            "corpus_total_s": sum(sum(v) for v in corpus_samples.values()),
            "resolve_total_s": sum(sum(v) for v in resolve_samples.values()),
            "machine": machine,
            "loadavg": loadavg,
            "chartreux_file": chartreux.__file__,
        },
    )


@pytest.mark.perf
@pytest.mark.timeout(900)
def test_corpus_replay_cases_within_frozen_ceiling(
    tmp_path_factory: pytest.TempPathFactory, pinned_shell: None
) -> None:
    """Per-case corpus replay durations vs the frozen per-case denominators."""
    root = _corpus_fixtures(tmp_path_factory)
    entries = corpus.load_corpus()
    cases = [
        (
            str(entry["id"]),
            str(entry["family"]),
            (lambda e=entry: corpus.observe_entry(e, root)),
        )
        for entry in entries
    ]
    machine = machine_context()
    loadavg = os.getloadavg()
    samples = _time_cases(cases, TRIALS)
    _report_and_enforce(
        scenario="shell_policy_corpus_replay_cases",
        samples=samples,
        frozen=_FROZEN_CORPUS_CASE_S,
        family_p95=_FROZEN_FAMILY_P95_S,
        family_of=lambda case: case.split(":", 1)[0],
        machine=machine,
        loadavg=loadavg,
    )


@pytest.mark.perf
@pytest.mark.timeout(900)
def test_executor_pinning_overhead_cases_within_frozen_ceiling(
    tmp_path_factory: pytest.TempPathFactory,
    pinned_shell: None,
    hermetic_resolution: None,
) -> None:
    """Per-case resolve_permission durations (the executor-pinning overhead
    denominator) vs the frozen per-case denominators.
    """
    root = _corpus_fixtures(tmp_path_factory)
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=root)
    battery = _resolve_battery()
    cases = [
        (
            f"resolve:{command}",
            "resolve",
            (lambda c=command: tool.resolve_permission(BashArgs(command=c))),
        )
        for command in battery
    ]
    machine = machine_context()
    loadavg = os.getloadavg()
    samples = _time_cases(cases, TRIALS)
    _report_and_enforce(
        scenario="shell_policy_executor_pinning_overhead_cases",
        samples=samples,
        frozen=_FROZEN_RESOLVE_CASE_S,
        family_p95=_FROZEN_FAMILY_P95_S,
        family_of=lambda case: "resolve",
        machine=machine,
        loadavg=loadavg,
    )
