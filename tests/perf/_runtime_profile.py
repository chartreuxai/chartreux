"""Matrix runner for the v0.7.5 stage-2 runtime profiling matrix (WP7a).

The stage-2 workload cells are driven by their producer packages, one fresh
process per run. This module is the machinery around those runs: the expanded
cell matrix (both adapter paths, both run modes), the repriced budget each cell
carries, and the per-cell subprocess isolation that makes a cell a genuinely
fresh process.

Isolation is the point. Every cell runs as its own interpreter with a
purpose-built environment: an allowlisted environment so no provider credential
or repository config can be inherited, ``PYTHONDONTWRITEBYTECODE`` preserved,
and a cell-shaped ``CHARTREUX_PERF_RUN_DIR`` inside a strict descendant of the
cell's own declared scratchpad, with ``CHARTREUX_PERF_REPORT`` under it. The
child installs the harness egress guard itself and asserts it closed before and
after the workload — the pytest fixtures never reach a subprocess — and writes
its report atomically, so a killed cell can never leave adoptable output.

Timeout is a halt, never a retry (plan §1): a cell that outlives its kill
deadline is killed, its partial output is discarded rather than adopted, and
the matrix stops there. No attempt is repeated.

Reports are WP6 schema documents (``chartreux.runtime_perf.v1``) carrying the
run mode (``measure`` or ``profile``) and the chunks-coverage fields of plan
§3; the matrix report adds the cell outcomes and the matrix manifest as extra
sections, which the schema tolerates by design.

Budgeting: r12's 931 s / 1,100 s matrix numbers are planning estimates to
revalidate, not measurements to trust (plan §8). :func:`matrix_manifest` prices
the expanded matrix from plan §6's per-run prices plus an explicit setup and
profiler allowance, so WP7b's one-shot run is compared against a stated basis
instead of an inherited number.

Profiler readiness (WP7b). A cell publishes its own pid and a phase line before
it touches the workload, so a profiler attach names a process that provably
started and reached the workload — never a pid guessed from the outside.
:func:`wait_for_handshake` is the parent half of that handshake, bounded by both
the handshake window and the cell's own invocation deadline. A cell that never
publishes is a halt, not a retry, and the killed cell's handshake is discarded
with the rest of its unadoptable output.

py-spy absence is an honest ``unavailable`` branch: profiling is observational
(Appendix B ``run_eligibility.observer_mode``), so a missing or failing profiler
is recorded and never allowed to change a cell's outcome.

The one-shot run (WP7b) executes the full expanded matrix exactly once under a
deadline that is never shorter than the expanded estimate. A cell killed by its
own or by the matrix deadline halts the run, its partial output is discarded,
nothing is rerun, and the report carries the escalation together with the
measured wall beside the repriced estimate and r12's superseded numbers.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from time import perf_counter, sleep
from typing import Any

from tests.perf._runtime_metrics import (
    REPORT_ENV,
    RUN_DIR_ENV,
    SCRATCHPAD_ENV,
    PerfPathError,
    PerfPaths,
    current_perf_paths,
    repository_root,
    resolve_perf_paths,
)
from tests.perf._runtime_report import (
    CELL_FRESH_HEADLESS,
    CELL_LOADED_HEADLESS,
    METRIC_VERSION,
    MODE_MEASURE,
    MODE_PROFILE,
    MODES,
    ReportSchemaError,
    build_report,
    build_run_record,
    validate_report,
)

MATRIX_SCHEMA = "chartreux.runtime_profile_matrix.v1"
ONE_SHOT_SCHEMA = "chartreux.runtime_profile_one_shot.v1"
MATRIX_REPORT_FILENAME = "matrix-report.json"
CELL_REPORT_FILENAME = "cell-report.json"
CELL_STDOUT_FILENAME = "cell-stdout.log"
CELL_STDERR_FILENAME = "cell-stderr.log"
PARTIAL_SUFFIX = ".partial"

#: The harness redirection names the cell environment carries (plan §1). The
#: harness module that defines ``CHARTREUX_HOME`` imports the whole app-server
#: chain, so the name is pinned here for the parent's environment builder; the
#: child's egress guard re-asserts the redirection it installs.
CONFIG_HOME_ENV = "CHARTREUX_HOME"
BYTECODE_ENV = "PYTHONDONTWRITEBYTECODE"

CELL_CONFIG_DIRNAME = "config"
CELL_SCRATCH_DIRNAME = "scratch"
CELL_RUN_DIRNAME = "run"
CELL_SESSIONS_DIRNAME = "sessions"
EGRESS_DIRNAME = "egress"

#: The profiler readiness handshake (WP7b): a cell publishes its own pid and a
#: phase line before the workload, so an attach names a process that provably
#: started rather than one guessed from outside.
PROFILE_PID_FILENAME = "profile.pid"
PROFILE_PHASE_FILENAME = "profile.phase"
PROFILE_PHASE_READY = "ready"
PROFILE_PHASE_DONE = "done"
PROFILE_PHASES: tuple[str, ...] = (PROFILE_PHASE_READY, PROFILE_PHASE_DONE)
HANDSHAKE_DEADLINE_S = 10.0
HANDSHAKE_POLL_INTERVAL_S = 0.01
HANDSHAKE_READY = "ready"
HANDSHAKE_TIMEOUT = "timeout"
HANDSHAKE_MALFORMED = "malformed"
HANDSHAKE_PID_MISMATCH = "pid_mismatch"
HANDSHAKE_PROCESS_EXITED = "process_exited"

PYSPY_EXECUTABLE = "py-spy"
PROFILER_OUTPUT_FILENAME = "profile.speedscope.json"
PROFILER_NOT_ATTACHED = "not_attached"
PROFILER_AVAILABLE = "available"
PROFILER_UNAVAILABLE = "unavailable"
PROFILER_ATTACHED = "attached"
PROFILER_FAILED = "failed"
PROFILER_STOP_GRACE_S = 5.0

TERMINATE_GRACE_S = 5.0

HEADLESS_ADAPTER = "headless"
PILOT_ADAPTER = "pilot"
ADAPTERS: tuple[str, ...] = (HEADLESS_ADAPTER, PILOT_ADAPTER)

#: Plan §6's per-run campaign prices, used as the steady-state allowance of the
#: cell that reports that campaign. A price is a budget, never an eligibility
#: prediction.
CAMPAIGN_PRICE_S: dict[str, float] = {"A1": 12.0, "B1": 12.0, "D1": 20.0}
SETUP_ALLOWANCE_S = 8.0
PROFILER_ALLOWANCE_S = 6.0
KILL_DEADLINE_FACTOR = 2
#: Plan §1: a matrix-class fixture's own timeout floor.
MATRIX_CLASS_FLOOR_S = 24.0

#: The r12 planning numbers this matrix's own budget replaces (plan §8).
PLANNING_REFERENCE_S: dict[str, float] = {
    "r12_matrix_s": 931.0,
    "r12_timeout_s": 1100.0,
}

SYNTHETIC_CELL_ID = "synthetic-contract-cell"
SYNTHETIC_CELL_BUDGET_S = 1.0
SYNTHETIC_SAMPLES_MS: tuple[float, ...] = (0.25, 0.5)

CELL_OK = "ok"
CELL_FAILED = "failed"
CELL_HALTED = "halted"

#: The only inherited variables a cell may see: process plumbing, nothing that
#: can carry a credential or point at the repository's own configuration.
_PASSTHROUGH_ENV: tuple[str, ...] = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TERM",
    "TMPDIR",
    "TZ",
    "USER",
    "VIRTUAL_ENV",
)
_CREDENTIAL_MARKERS: tuple[str, ...] = (
    "API_KEY",
    "APIKEY",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)
_OUTPUT_TAIL_CHARS = 400


class MatrixError(RuntimeError):
    """A matrix cell or its environment violates the runner's contract."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"profile matrix rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class MatrixCell:
    """One point of the profiling matrix: a workload, an adapter, a run mode.

    ``campaign_id`` is the WP6 row the cell reports, so every emitted run record
    is bound to a gated campaign identity rather than an invented one. Matrix
    runs are observation-only: their records record the cell's own measured
    samples, and the closure note says so, so a matrix run can never be read as
    a campaign baseline.
    """

    cell_id: str
    workload_cell: str
    adapter: str
    mode: str
    campaign_id: str
    synthetic: bool = False
    budget_override_s: float | None = None

    @property
    def budget_s(self) -> float:
        """Return the cell's planning budget in seconds."""
        if self.budget_override_s is not None:
            return self.budget_override_s
        price = CAMPAIGN_PRICE_S.get(self.campaign_id)
        if price is None:
            raise MatrixError(
                "campaign_not_priced", f"{self.campaign_id} has no campaign price"
            )
        allowance = SETUP_ALLOWANCE_S + price
        if self.mode == MODE_PROFILE:
            allowance += PROFILER_ALLOWANCE_S
        return allowance

    @property
    def kill_deadline_s(self) -> float:
        """Return the invocation deadline: the cell budget times the factor."""
        return KILL_DEADLINE_FACTOR * self.budget_s

    def record(self) -> dict[str, Any]:
        """Return the cell's manifest record."""
        return {
            "cell_id": self.cell_id,
            "workload_cell": self.workload_cell,
            "adapter": self.adapter,
            "mode": self.mode,
            "campaign_id": self.campaign_id,
            "synthetic": self.synthetic,
            "budget_s": round(self.budget_s, 3),
            "kill_deadline_s": round(self.kill_deadline_s, 3),
        }


#: The expanded matrix: every landed workload definition on every adapter path
#: that can drive it, in both run modes. The Pilot path is a single-root
#: transcript, so only the fresh workload reaches it.
MATRIX_CELLS: tuple[MatrixCell, ...] = (
    MatrixCell(
        "measure-headless-fresh",
        CELL_FRESH_HEADLESS,
        HEADLESS_ADAPTER,
        MODE_MEASURE,
        "A1",
    ),
    MatrixCell(
        "profile-headless-fresh",
        CELL_FRESH_HEADLESS,
        HEADLESS_ADAPTER,
        MODE_PROFILE,
        "A1",
    ),
    MatrixCell(
        "measure-headless-loaded",
        CELL_LOADED_HEADLESS,
        HEADLESS_ADAPTER,
        MODE_MEASURE,
        "B1",
    ),
    MatrixCell(
        "profile-headless-loaded",
        CELL_LOADED_HEADLESS,
        HEADLESS_ADAPTER,
        MODE_PROFILE,
        "B1",
    ),
    MatrixCell(
        "measure-pilot-fresh", CELL_FRESH_HEADLESS, PILOT_ADAPTER, MODE_MEASURE, "D1"
    ),
    MatrixCell(
        "profile-pilot-fresh", CELL_FRESH_HEADLESS, PILOT_ADAPTER, MODE_PROFILE, "D1"
    ),
)

#: The contract node's cell: a synthetic workload with no fixture, no adapter
#: and no campaign reach, used to drive the isolation machinery in-suite.
SYNTHETIC_CELL = MatrixCell(
    SYNTHETIC_CELL_ID,
    CELL_FRESH_HEADLESS,
    HEADLESS_ADAPTER,
    MODE_PROFILE,
    "A1",
    synthetic=True,
    budget_override_s=SYNTHETIC_CELL_BUDGET_S,
)

#: Every cell the runner can emit, matrix cells and the synthetic one.
CELLS_BY_ID: dict[str, MatrixCell] = {cell.cell_id: cell for cell in MATRIX_CELLS}
CELLS_BY_ID[SYNTHETIC_CELL.cell_id] = SYNTHETIC_CELL


def matrix_cell(cell_id: str) -> MatrixCell:
    """Return the cell with ``cell_id``, raising for an unknown id."""
    cell = CELLS_BY_ID.get(cell_id)
    if cell is None:
        raise MatrixError("unknown_cell", f"{cell_id!r} is not a matrix cell")
    return cell


def matrix_manifest() -> dict[str, Any]:
    """Return the matrix manifest and its repriced budget."""
    per_cell = {cell.cell_id: round(cell.budget_s, 3) for cell in MATRIX_CELLS}
    return {
        "schema": MATRIX_SCHEMA,
        "metric_version": METRIC_VERSION,
        "axes": {
            "workload_cells": [CELL_FRESH_HEADLESS, CELL_LOADED_HEADLESS],
            "adapters": list(ADAPTERS),
            "modes": list(MODES),
            "cells": len(MATRIX_CELLS),
        },
        "scope": {
            "included": (
                "every landed workload definition on every adapter path that can"
                " drive it, in both run modes"
            ),
            "excluded": {
                "ablation_brackets": (
                    "declared and recorded by WP5a/WP5b; an ablation record is not"
                    " a cell record"
                ),
                "injected_delay_points": (
                    "owned by X1's canonical cell; each landed workload definition"
                    " pins its own injection"
                ),
            },
        },
        "budget": {
            "setup_allowance_s": SETUP_ALLOWANCE_S,
            "profiler_allowance_s": PROFILER_ALLOWANCE_S,
            "campaign_prices_s": dict(CAMPAIGN_PRICE_S),
            "price_basis": (
                "plan §6 per-run campaign prices plus the setup and profiler"
                " allowances; a budget, never an eligibility prediction"
            ),
            "kill_deadline_rule": f"{KILL_DEADLINE_FACTOR}x the cell budget",
            "per_cell_s": per_cell,
            "total_estimate_s": round(sum(per_cell.values()), 3),
            "total_kill_deadline_s": round(
                sum(cell.kill_deadline_s for cell in MATRIX_CELLS), 3
            ),
            "expanded_estimate_s": expanded_estimate_s(),
            "one_shot_timeout_s": one_shot_timeout_s(),
            "one_shot_rule": (
                "the expanded matrix runs once; the one-shot deadline is never"
                " below the expanded estimate, and every cell keeps its own kill"
                " deadline inside it — a kill halts, discards the partial output"
                " and escalates, and the matrix is never rerun"
            ),
            "superseded_planning_reference_s": dict(PLANNING_REFERENCE_S),
            "note": (
                "the r12 numbers are planning estimates, revalidated by WP7b's"
                " one-shot matrix run; this table is the basis it is compared to"
            ),
        },
        "cells": [cell.record() for cell in MATRIX_CELLS],
    }


def credential_names(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Return every variable that names a credential-shaped secret."""
    return tuple(
        sorted(
            name
            for name in environ
            if any(marker in name.upper() for marker in _CREDENTIAL_MARKERS)
        )
    )


def strip_credentials() -> dict[str, str]:
    """Remove every credential-shaped variable, returning what was removed.

    The harness guard strips a fixed list of provider names; this strip is by
    shape, so a credential the list does not know about (a second key under the
    same provider, for instance) cannot survive into a matrix run or its cells.
    """
    removed = {name: os.environ.pop(name) for name in credential_names(os.environ)}
    return removed


def restore_credentials(removed: Mapping[str, str]) -> None:
    """Put back what :func:`strip_credentials` removed."""
    os.environ.update(removed)


def cell_root(run_root: Path, cell: MatrixCell) -> Path:
    """Return the cell's own directory under the matrix run directory."""
    root = Path(run_root) / cell.cell_id
    if root.parent != Path(run_root):
        raise MatrixError(
            "cell_id_escapes_run_dir", f"{cell.cell_id!r} leaves {run_root}"
        )
    return root


def cell_run_dir(run_root: Path, cell: MatrixCell) -> Path:
    """Return the ``CHARTREUX_PERF_RUN_DIR`` of one cell."""
    return cell_root(run_root, cell) / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME


def cell_report_path(run_root: Path, cell: MatrixCell) -> Path:
    """Return the ``CHARTREUX_PERF_REPORT`` of one cell."""
    return cell_run_dir(run_root, cell) / CELL_REPORT_FILENAME


def prepare_cell_dirs(root: Path) -> Path:
    """Create the cell's config, scratch, run and session directories."""
    for directory in (
        root / CELL_CONFIG_DIRNAME,
        root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME,
        root / CELL_SCRATCH_DIRNAME / CELL_SESSIONS_DIRNAME,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return root


def cell_environment(
    cell: MatrixCell, run_root: Path, *, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build the fresh environment one cell subprocess runs under.

    The environment is built from an allowlist, so nothing credential-shaped can
    be inherited, and every path it names is derived from the cell's own
    directory. The egress guard in the child re-asserts the outcome.
    """
    source = os.environ if base is None else base
    root = cell_root(run_root, cell)
    environment = {name: source[name] for name in _PASSTHROUGH_ENV if name in source}
    environment[BYTECODE_ENV] = "1"
    environment[CONFIG_HOME_ENV] = str(root / CELL_CONFIG_DIRNAME)
    environment[SCRATCHPAD_ENV] = str(root / CELL_SCRATCH_DIRNAME)
    environment[RUN_DIR_ENV] = str(cell_run_dir(run_root, cell))
    environment[REPORT_ENV] = str(cell_report_path(run_root, cell))
    repository = str(repository_root())
    inherited = source.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        repository if not inherited else f"{repository}{os.pathsep}{inherited}"
    )
    return environment


def assert_cell_environment(environment: Mapping[str, str]) -> PerfPaths:
    """Validate one cell environment against the runner's path guard.

    The environment must disable bytecode writing, carry no credential-shaped
    variable, and resolve to a run directory and report inside the boundary the
    WP1 guard enforces.
    """
    if environment.get(BYTECODE_ENV) != "1":
        raise MatrixError(
            "bytecode_enabled", "a cell must never write bytecode into the tree"
        )
    present = credential_names(environment)
    if present:
        raise MatrixError("credentials_present", f"{list(present)}")
    try:
        paths = resolve_perf_paths(environment)
    except PerfPathError as error:
        raise MatrixError("cell_path_rejected", error.detail) from error
    if paths.run_dir is None or paths.report is None:
        raise MatrixError(
            "cell_paths_unset", "a cell needs a run directory and a report"
        )
    return paths


def discard_cell_output(root: Path) -> bool:
    """Remove a killed cell's unadoptable output; returns whether anything went.

    A killed cell's report is partial and its readiness handshake names a process
    that never finished, so neither is adoptable evidence. The diagnostic logs
    stay — they are what the escalation is read from.
    """
    discarded = False
    run_dir = root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME
    patterns = (
        f"{CELL_REPORT_FILENAME}*",
        f"{PROFILE_PID_FILENAME}*",
        f"{PROFILE_PHASE_FILENAME}*",
    )
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            if path.is_file():
                path.unlink(missing_ok=True)
                discarded = True
    return discarded


def _write_text(path: Path, text: str) -> None:
    """Write a text file atomically, so a killed writer leaves no partial line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write a document atomically, so a killed writer leaves no partial file."""
    _write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")


def _cell_log_paths(root: Path) -> tuple[Path, Path]:
    """Return the cell's stdout and stderr logs (escalation evidence)."""
    run_dir = root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME
    return run_dir / CELL_STDOUT_FILENAME, run_dir / CELL_STDERR_FILENAME


def _log_tail(path: Path) -> str:
    """Return one log's tail, bounded so a chatty cell cannot flood a report."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _OUTPUT_TAIL_CHARS * 8))
            return handle.read().decode("utf-8", "replace")[-_OUTPUT_TAIL_CHARS:]
    except OSError:
        return ""


def _output_tail(root: Path) -> str:
    """Return the cell's combined output tail, read from the logs it wrote."""
    stdout_path, stderr_path = _cell_log_paths(root)
    return f"{_log_tail(stdout_path)}{_log_tail(stderr_path)}"


def _run_start_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id(cell: MatrixCell) -> str:
    return f"{cell.cell_id}-{cell.campaign_id}"


# --------------------------------------------------------------------------
# Profiler readiness handshake
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileHandshake:
    """One cell's published readiness: which process, which phase, which cell."""

    pid: int
    phase: str
    cell_id: str
    mode: str

    def line(self) -> str:
        """Return the handshake as its single phase line."""
        return f"phase={self.phase} pid={self.pid} cell={self.cell_id} mode={self.mode}"

    def record(self) -> dict[str, Any]:
        """Return the handshake's report record."""
        return {
            "pid": self.pid,
            "phase": self.phase,
            "cell_id": self.cell_id,
            "mode": self.mode,
        }


_HANDSHAKE_FIELDS: tuple[str, ...] = ("phase", "pid", "cell", "mode")


def parse_phase_line(line: str) -> ProfileHandshake:
    """Parse one phase line, refusing anything that is not exactly the contract."""
    fields: dict[str, str] = {}
    for token in line.split():
        name, separator, value = token.partition("=")
        if not separator or not name or not value:
            raise MatrixError("handshake_malformed", f"{line!r}")
        fields[name] = value
    if set(fields) != set(_HANDSHAKE_FIELDS) or not fields["pid"].isdigit():
        raise MatrixError("handshake_malformed", f"{line!r}")
    if fields["phase"] not in PROFILE_PHASES:
        raise MatrixError("handshake_malformed", f"unknown phase in {line!r}")
    return ProfileHandshake(
        pid=int(fields["pid"]),
        phase=fields["phase"],
        cell_id=fields["cell"],
        mode=fields["mode"],
    )


def publish_handshake(
    directory: Path,
    cell: MatrixCell,
    *,
    phase: str = PROFILE_PHASE_READY,
    pid: int | None = None,
) -> ProfileHandshake:
    """Publish the cell's pid and phase line, then echo the line to stdout.

    The pid file and the phase file are both written atomically, so a reader never
    sees a half-published handshake.
    """
    if phase not in PROFILE_PHASES:
        raise MatrixError("handshake_malformed", f"unknown phase {phase!r}")
    handshake = ProfileHandshake(
        pid=os.getpid() if pid is None else pid,
        phase=phase,
        cell_id=cell.cell_id,
        mode=cell.mode,
    )
    _write_text(Path(directory) / PROFILE_PID_FILENAME, f"{handshake.pid}\n")
    _write_text(Path(directory) / PROFILE_PHASE_FILENAME, handshake.line() + "\n")
    print(handshake.line(), flush=True)
    return handshake


def read_handshake(directory: Path) -> ProfileHandshake | None:
    """Read a published handshake, or ``None`` when nothing was published yet."""
    root = Path(directory)
    try:
        text = (root / PROFILE_PHASE_FILENAME).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise MatrixError("handshake_unreadable", f"{error}") from error
    handshake = parse_phase_line(text.strip())
    try:
        pid_text = (root / PROFILE_PID_FILENAME).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise MatrixError(
            "handshake_malformed", f"the pid file is unreadable: {error}"
        ) from error
    if pid_text != str(handshake.pid):
        raise MatrixError(
            "handshake_malformed",
            f"pid file {pid_text!r} disagrees with the phase line {handshake.pid}",
        )
    return handshake


def clear_handshake(directory: Path) -> None:
    """Remove a previous handshake, so a fresh cell can never adopt a stale one."""
    for name in (PROFILE_PID_FILENAME, PROFILE_PHASE_FILENAME):
        for path in Path(directory).glob(f"{name}*"):
            if path.is_file():
                path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class HandshakeWait:
    """The outcome of waiting for a cell's readiness handshake."""

    status: str
    handshake: ProfileHandshake | None
    waited_s: float
    reason: str | None = None

    @property
    def ready(self) -> bool:
        """Return whether the cell published a valid, correctly owned handshake."""
        return self.status == HANDSHAKE_READY

    def record(self) -> dict[str, Any]:
        """Return the wait's report record."""
        return {
            "status": self.status,
            "waited_s": round(self.waited_s, 3),
            "reason": self.reason,
            "handshake": None if self.handshake is None else self.handshake.record(),
        }


def wait_for_handshake(
    directory: Path,
    *,
    expected_pid: int | None = None,
    deadline_s: float = HANDSHAKE_DEADLINE_S,
    is_alive: Callable[[], bool] | None = None,
    poll_interval_s: float = HANDSHAKE_POLL_INTERVAL_S,
) -> HandshakeWait:
    """Wait for a cell's readiness handshake inside ``deadline_s``.

    A handshake naming a different process than the launched one is refused
    rather than adopted, and a process that is already gone is reported as such
    instead of being waited out to the deadline.
    """
    started = perf_counter()
    while True:
        try:
            handshake = read_handshake(directory)
        except MatrixError as error:
            return HandshakeWait(
                HANDSHAKE_MALFORMED, None, perf_counter() - started, error.detail
            )
        if handshake is not None:
            if expected_pid is not None and handshake.pid != expected_pid:
                return HandshakeWait(
                    HANDSHAKE_PID_MISMATCH,
                    handshake,
                    perf_counter() - started,
                    f"published pid {handshake.pid} is not the launched process"
                    f" {expected_pid}",
                )
            return HandshakeWait(
                HANDSHAKE_READY, handshake, perf_counter() - started, None
            )
        if is_alive is not None and not is_alive():
            return HandshakeWait(
                HANDSHAKE_PROCESS_EXITED,
                None,
                perf_counter() - started,
                "the cell exited before publishing its handshake",
            )
        waited = perf_counter() - started
        if waited >= deadline_s:
            return HandshakeWait(
                HANDSHAKE_TIMEOUT, None, waited, f"no handshake within {deadline_s:g}s"
            )
        sleep(poll_interval_s)


# --------------------------------------------------------------------------
# Profiler attach (py-spy): observational, never outcome-changing
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfilerRun:
    """One profiler decision: the availability probe and any attached process."""

    status: str
    executable: str | None = None
    output: str | None = None
    process: subprocess.Popen[bytes] | None = None
    reason: str | None = None

    def record(self) -> dict[str, Any]:
        """Return the profiler's report record."""
        return {
            "status": self.status,
            "executable": self.executable,
            "output": self.output,
            "reason": self.reason,
        }


def pyspy_path(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the py-spy executable this environment provides, if any."""
    source = os.environ if environ is None else environ
    return shutil.which(PYSPY_EXECUTABLE, path=source.get("PATH"))


def profiler_status(environ: Mapping[str, str] | None = None) -> ProfilerRun:
    """Probe profiler availability; absence is an honest unavailable branch."""
    executable = pyspy_path(environ)
    if executable is None:
        return ProfilerRun(
            PROFILER_UNAVAILABLE, reason="py-spy is not installed in this environment"
        )
    return ProfilerRun(PROFILER_AVAILABLE, executable=executable)


def start_profiler(
    cell: MatrixCell, handshake: ProfileHandshake | None, run_dir: Path
) -> ProfilerRun:
    """Attach py-spy to a ready cell, or record why no profile was taken.

    Profiling is observational (Appendix B ``run_eligibility.observer_mode``): an
    absent or failing profiler is recorded and never allowed to change the cell's
    outcome.
    """
    if cell.mode != MODE_PROFILE:
        return ProfilerRun(
            PROFILER_NOT_ATTACHED, reason=f"{cell.mode} cell: no profiler is attached"
        )
    if handshake is None:
        return ProfilerRun(
            PROFILER_NOT_ATTACHED, reason="the cell published no readiness handshake"
        )
    available = profiler_status()
    if available.status != PROFILER_AVAILABLE or available.executable is None:
        return available
    output = Path(run_dir) / PROFILER_OUTPUT_FILENAME
    command = [
        available.executable,
        "record",
        "--pid",
        str(handshake.pid),
        "--output",
        str(output),
        "--format",
        "speedscope",
        "--nonblocking",
        "--rate",
        "100",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        return ProfilerRun(
            PROFILER_FAILED,
            executable=available.executable,
            output=str(output),
            reason=f"py-spy launch failed: {error}",
        )
    return ProfilerRun(
        PROFILER_ATTACHED,
        executable=available.executable,
        output=str(output),
        process=process,
    )


def stop_profiler(run: ProfilerRun | None) -> dict[str, Any] | None:
    """Stop an attached profiler and record what it actually did."""
    if run is None:
        return None
    if run.process is None:
        return run.record()
    process = run.process
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=PROFILER_STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    returncode = process.returncode
    if returncode in (0, -signal.SIGINT):
        return ProfilerRun(
            PROFILER_ATTACHED, executable=run.executable, output=run.output
        ).record()
    return ProfilerRun(
        PROFILER_FAILED,
        executable=run.executable,
        output=run.output,
        reason=f"py-spy exited with status {returncode}",
    ).record()


@dataclass(frozen=True, slots=True)
class CellOutcome:
    """What one cell invocation produced, honestly and once."""

    cell_id: str
    status: str
    attempts: int
    wall_s: float
    exit_code: int | None = None
    run_count: int = 0
    report_path: str | None = None
    partial_discarded: bool = False
    halt_reason: str | None = None
    output_tail: str = ""
    records: tuple[dict[str, Any], ...] = ()
    handshake: dict[str, Any] | None = None
    profiler: dict[str, Any] | None = None

    def record(self) -> dict[str, Any]:
        """Return the outcome's report record (the records stay separate)."""
        return {
            "cell_id": self.cell_id,
            "status": self.status,
            "attempts": self.attempts,
            "wall_s": round(self.wall_s, 3),
            "exit_code": self.exit_code,
            "run_count": self.run_count,
            "report_path": self.report_path,
            "partial_discarded": self.partial_discarded,
            "halt_reason": self.halt_reason,
            "output_tail": self.output_tail,
            "handshake": None if self.handshake is None else dict(self.handshake),
            "profiler": None if self.profiler is None else dict(self.profiler),
        }


def _terminate(process: subprocess.Popen[str]) -> None:
    """Stop a cell process and reap it; a killed cell is never left running."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _wait_for_exit(process: subprocess.Popen[str], remaining_s: float) -> int | None:
    """Wait for the cell to exit; ``None`` means it outlived its deadline."""
    try:
        return process.wait(timeout=remaining_s)
    except subprocess.TimeoutExpired:
        return None


def _await_handshake(
    process: subprocess.Popen[str], run_dir: Path, *, deadline_s: float, started: float
) -> tuple[HandshakeWait, bool]:
    """Wait for the cell's handshake inside both its window and the deadline.

    Returns the wait and whether the invocation deadline — rather than the
    handshake window — was the binding constraint: a cell starved of its whole
    deadline is killed by that deadline, and its halt says so.
    """
    remaining = deadline_s - (perf_counter() - started)
    window = min(HANDSHAKE_DEADLINE_S, remaining) if remaining > 0 else 0.0
    wait = wait_for_handshake(
        run_dir,
        expected_pid=process.pid,
        deadline_s=max(window, 0.0),
        is_alive=lambda: process.poll() is None,
    )
    return wait, window == remaining


def run_cell(
    cell: MatrixCell,
    *,
    run_root: Path,
    timeout_s: float | None = None,
    base: Mapping[str, str] | None = None,
) -> CellOutcome:
    """Run one cell as a fresh subprocess and adopt its report if it validates.

    A cell is attempted exactly once. The child publishes its readiness handshake
    before the workload, so a profiler attach names a process that provably
    started; a handshake that never appears, or that names another process, halts
    the cell. An invocation that outlives its deadline is killed and its partial
    output discarded: ``CELL_HALTED`` is a halt, never a retry.
    """
    root = prepare_cell_dirs(cell_root(Path(run_root), cell))
    environment = cell_environment(cell, Path(run_root), base=base)
    paths = assert_cell_environment(environment)
    deadline = cell.kill_deadline_s if timeout_s is None else timeout_s
    if deadline <= 0:
        raise MatrixError("deadline_not_positive", f"{deadline!r}")
    run_dir = root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME
    clear_handshake(run_dir)
    command = [
        sys.executable,
        "-m",
        "tests.perf._runtime_profile",
        "--cell",
        cell.cell_id,
    ]
    stdout_path, stderr_path = _cell_log_paths(root)
    started = perf_counter()
    handshake_record: dict[str, Any] | None = None
    profiler_record: dict[str, Any] | None = None
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=str(repository_root()),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        try:
            wait, deadline_bound = _await_handshake(
                process, run_dir, deadline_s=deadline, started=started
            )
            handshake_record = wait.record()
            if not wait.ready:
                _terminate(process)
                return CellOutcome(
                    cell_id=cell.cell_id,
                    status=CELL_HALTED,
                    attempts=1,
                    wall_s=perf_counter() - started,
                    report_path=str(paths.report),
                    partial_discarded=discard_cell_output(root),
                    halt_reason=(
                        "invocation_timeout"
                        if deadline_bound and wait.status == HANDSHAKE_TIMEOUT
                        else f"handshake_{wait.status}"
                    ),
                    output_tail=_output_tail(root),
                    handshake=handshake_record,
                )
            profiler = start_profiler(cell, wait.handshake, run_dir)
            remaining = deadline - (perf_counter() - started)
            returncode = None if remaining <= 0 else _wait_for_exit(process, remaining)
            profiler_record = stop_profiler(profiler)
            if returncode is None:
                _terminate(process)
                return CellOutcome(
                    cell_id=cell.cell_id,
                    status=CELL_HALTED,
                    attempts=1,
                    wall_s=perf_counter() - started,
                    report_path=str(paths.report),
                    partial_discarded=discard_cell_output(root),
                    halt_reason="invocation_timeout",
                    output_tail=_output_tail(root),
                    handshake=handshake_record,
                    profiler=profiler_record,
                )
        finally:
            if process.poll() is None:
                _terminate(process)
    wall_s = perf_counter() - started
    output = _output_tail(root)
    if returncode != 0:
        return CellOutcome(
            cell_id=cell.cell_id,
            status=CELL_FAILED,
            attempts=1,
            wall_s=wall_s,
            exit_code=returncode,
            report_path=str(paths.report),
            halt_reason="cell_exit_nonzero",
            output_tail=output,
            handshake=handshake_record,
            profiler=profiler_record,
        )
    assert paths.report is not None
    try:
        document = json.loads(paths.report.read_text(encoding="utf-8"))
        validate_report(document)
    except (OSError, json.JSONDecodeError, ReportSchemaError) as error:
        return CellOutcome(
            cell_id=cell.cell_id,
            status=CELL_FAILED,
            attempts=1,
            wall_s=wall_s,
            exit_code=returncode,
            report_path=str(paths.report),
            halt_reason="report_invalid",
            output_tail=f"{output}{type(error).__name__}: {error}",
            handshake=handshake_record,
            profiler=profiler_record,
        )
    records = tuple(dict(record) for record in document["runs"])
    return CellOutcome(
        cell_id=cell.cell_id,
        status=CELL_OK,
        attempts=1,
        wall_s=wall_s,
        exit_code=returncode,
        run_count=len(records),
        report_path=str(paths.report),
        records=records,
        handshake=handshake_record,
        profiler=profiler_record,
    )


@dataclass(frozen=True, slots=True)
class MatrixOutcome:
    """The cells a matrix invocation attempted and whether it halted."""

    cells: tuple[CellOutcome, ...]
    halted: bool
    egress: dict[str, Any] = field(default_factory=dict)
    halt_reason: str | None = None
    wall_s: float = 0.0

    def record(self) -> dict[str, Any]:
        """Return the run's report record."""
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "wall_s": round(self.wall_s, 3),
            "attempted": len(self.cells),
            "egress": dict(self.egress),
            "cells": [cell.record() for cell in self.cells],
        }


def _install_matrix_guard(run_root: Path) -> Any:
    """Install and assert the runner's own fail-closed egress policy."""
    from tests.perf._runtime_scenario import EgressGuard

    guard = EgressGuard(Path(run_root) / EGRESS_DIRNAME)
    guard.install()
    guard.assert_closed()
    return guard


def run_matrix(
    cells: Sequence[MatrixCell] | None = None,
    *,
    run_root: Path,
    base: Mapping[str, str] | None = None,
    overall_timeout_s: float | None = None,
) -> MatrixOutcome:
    """Run the selected matrix cells in order, stopping at the first halt.

    The runner installs and asserts the same egress policy each cell process
    installs for itself, so a bare-script invocation carries the socket block,
    the credential strip and the redirection too (plan §1). A halted cell ends
    the matrix: the remaining cells are not attempted, and the halted cell is
    not repeated (§1's no-rerun rule). A failed cell ends it too — an incomplete
    matrix is reported, never patched over.

    ``overall_timeout_s`` is the one-shot deadline. Every cell still keeps its
    own kill deadline inside it, so one slow cell cannot eat another's budget;
    when the matrix deadline itself is what bounds a cell, the halt is recorded
    as the one-shot timeout.
    """
    selected = MATRIX_CELLS if cells is None else tuple(cells)
    guard = _install_matrix_guard(Path(run_root))
    removed = strip_credentials()
    egress: dict[str, Any] = {}
    outcomes: list[CellOutcome] = []
    halted = False
    halt_reason: str | None = None
    started = perf_counter()
    try:
        for cell in selected:
            allowance: float | None = None
            if overall_timeout_s is not None:
                allowance = overall_timeout_s - (perf_counter() - started)
                if allowance <= 0:
                    halted = True
                    halt_reason = "one_shot_timeout"
                    break
            deadline = (
                cell.kill_deadline_s
                if allowance is None
                else min(cell.kill_deadline_s, allowance)
            )
            outcome = run_cell(cell, run_root=run_root, timeout_s=deadline, base=base)
            outcomes.append(outcome)
            if outcome.status != CELL_OK:
                halted = outcome.status == CELL_HALTED
                halt_reason = outcome.halt_reason
                truncated = allowance is not None and allowance < cell.kill_deadline_s
                if truncated and outcome.halt_reason == "invocation_timeout":
                    halt_reason = "one_shot_timeout"
                break
        guard.assert_closed()
        egress = _runner_egress(guard, Path(run_root))
    finally:
        guard.restore()
        restore_credentials(removed)
    return MatrixOutcome(
        tuple(outcomes),
        halted=halted,
        egress=egress,
        halt_reason=halt_reason,
        wall_s=perf_counter() - started,
    )


def write_matrix_report(
    outcome: MatrixOutcome, *, path: Path, measured_wall_s: float | None = None
) -> dict[str, Any]:
    """Write the matrix report: adopted run records plus the cell outcomes.

    ``measured_wall_s`` records the one-shot run's own wall and the revalidation
    it feeds; a per-cell report write leaves it unset.
    """
    records = [record for cell in outcome.cells for record in cell.records]
    document = build_report(records)
    document["matrix"] = matrix_manifest()
    document["cells"] = [cell.record() for cell in outcome.cells]
    document["halted"] = outcome.halted
    document["halt_reason"] = outcome.halt_reason
    document["egress"] = dict(outcome.egress)
    if measured_wall_s is not None:
        document["one_shot"] = one_shot_record(outcome, measured_wall_s)
    validate_report(document)
    _write_json(Path(path), document)
    return document


# --------------------------------------------------------------------------
# The one-shot run: the full expanded matrix, exactly once
# --------------------------------------------------------------------------


def expanded_estimate_s(cells: Sequence[MatrixCell] | None = None) -> float:
    """Return the expanded matrix's planning estimate: its cells' budgets."""
    selected = MATRIX_CELLS if cells is None else tuple(cells)
    return round(sum(cell.budget_s for cell in selected), 3)


def one_shot_timeout_s(cells: Sequence[MatrixCell] | None = None) -> float:
    """Return the one-shot deadline for the full expanded matrix.

    Plan §1: the deadline is never shorter than the expanded estimate, and every
    cell keeps its own kill deadline inside it, so the sum of the cell kill
    deadlines is the control the runner actually holds.
    """
    selected = MATRIX_CELLS if cells is None else tuple(cells)
    deadlines = sum(cell.kill_deadline_s for cell in selected)
    return round(max(deadlines, expanded_estimate_s(selected)), 3)


def revalidation_record(
    measured_wall_s: float, estimate_s: float, *, completed: bool = True
) -> dict[str, Any]:
    """Compare a measured matrix wall against the repriced estimate and r12.

    Both numbers are carried: the repriced estimate is the basis the runner
    declared, and r12's 931 s / 1,100 s are the planning reference this run
    replaces. Neither is inherited silently. ``completed`` marks whether the
    measured wall is a whole-matrix wall: a halted run's partial wall is recorded
    but is not a revalidation of the matrix it never finished.
    """
    r12_matrix_s = PLANNING_REFERENCE_S["r12_matrix_s"]
    r12_timeout_s = PLANNING_REFERENCE_S["r12_timeout_s"]
    return {
        "comparable": completed,
        "measured_wall_s": round(measured_wall_s, 3),
        "repriced_estimate_s": round(estimate_s, 3),
        "measured_vs_repriced_ratio": (
            round(measured_wall_s / estimate_s, 3) if estimate_s > 0 else None
        ),
        "within_repriced_estimate": measured_wall_s <= estimate_s,
        "r12_matrix_s": r12_matrix_s,
        "r12_timeout_s": r12_timeout_s,
        "measured_vs_r12_matrix_ratio": round(measured_wall_s / r12_matrix_s, 3),
        "measured_vs_r12_timeout_ratio": round(measured_wall_s / r12_timeout_s, 3),
        "within_r12_matrix": measured_wall_s <= r12_matrix_s,
        "within_r12_timeout": measured_wall_s <= r12_timeout_s,
        "basis": (
            "the repriced estimate is the matrix's own sum-of-cell-budgets basis;"
            " r12's 931 s / 1,100 s remain the superseded planning reference"
        ),
        "note": (
            "the measured wall is the whole matrix's"
            if completed
            else "the run halted: this wall is a partial wall, not a whole-matrix"
            " revalidation"
        ),
    }


def _profiler_summary(outcome: MatrixOutcome) -> dict[str, Any]:
    """Summarize the profiler decisions of one matrix run, without elevating."""
    statuses: dict[str, int] = {}
    reasons: list[str] = []
    for cell in outcome.cells:
        if cell.profiler is None:
            continue
        status = str(cell.profiler.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        reason = cell.profiler.get("reason")
        if reason and reason not in reasons:
            reasons.append(reason)
    return {"statuses": statuses, "reasons": sorted(reasons), "elevated": False}


def one_shot_record(outcome: MatrixOutcome, measured_wall_s: float) -> dict[str, Any]:
    """Return the one-shot section of the matrix report.

    An incomplete matrix records its escalation here: halt, discarded partial
    output, no rerun. Nothing about a halted run is presented as measured.
    """
    estimate = expanded_estimate_s()
    return {
        "schema": ONE_SHOT_SCHEMA,
        "cells_planned": len(MATRIX_CELLS),
        "cells_attempted": len(outcome.cells),
        "cells_succeeded": sum(1 for cell in outcome.cells if cell.status == CELL_OK),
        "expanded_estimate_s": estimate,
        "one_shot_timeout_s": one_shot_timeout_s(),
        "measured_wall_s": round(measured_wall_s, 3),
        "halted": outcome.halted,
        "halt_reason": outcome.halt_reason,
        "attempts_per_cell": 1,
        "rerun": False,
        "profiler": _profiler_summary(outcome),
        "revalidation": revalidation_record(
            measured_wall_s,
            estimate,
            completed=not outcome.halted and len(outcome.cells) == len(MATRIX_CELLS),
        ),
        "escalation": (
            None
            if not outcome.halted
            else {
                "action": "halt-and-escalate",
                "reason": outcome.halt_reason,
                "halted_cell": (
                    None if not outcome.cells else outcome.cells[-1].cell_id
                ),
                "partial_output": "discarded",
                "rerun": False,
            }
        ),
    }


def _one_shot_deadline(declared: float | None) -> float:
    """Return the one-shot deadline, refusing a declaration below the estimate."""
    estimate = expanded_estimate_s()
    if declared is None:
        return one_shot_timeout_s()
    if declared < estimate:
        raise MatrixError(
            "one_shot_timeout_below_estimate",
            f"{declared:g}s is shorter than the expanded estimate {estimate:g}s",
        )
    return declared


# --------------------------------------------------------------------------
# Cell side: one fresh process emits one report
# --------------------------------------------------------------------------


def _install_guard(paths: PerfPaths) -> Any:
    """Install and assert the harness egress guard in this cell process."""
    from tests.perf._runtime_scenario import EgressGuard

    if paths.scratchpad_root is None:
        raise MatrixError("cell_paths_unset", "the cell declares no scratchpad")
    guard = EgressGuard(paths.scratchpad_root.parent)
    guard.install(scratch_root=paths.scratchpad_root)
    guard.assert_closed()
    return guard


def _egress_evidence(guard: Any) -> dict[str, Any]:
    """Return the properties every guarded process reports about itself."""
    return {
        "guard": "tests.perf._runtime_scenario.EgressGuard",
        "asserted": True,
        "credentials_present": list(credential_names(os.environ)),
        "bytecode_disabled": os.environ.get(BYTECODE_ENV) == "1",
        "config_home": str(guard.config_home),
        "scratch_home": str(guard.scratch_home),
    }


def _runner_egress(guard: Any, run_root: Path) -> dict[str, Any]:
    return {**_egress_evidence(guard), "run_root": str(run_root)}


def _egress_record(paths: PerfPaths, guard: Any) -> dict[str, Any]:
    return {
        **_egress_evidence(guard),
        "run_dir": str(paths.run_dir),
        "report": str(paths.report),
        "repo_root": str(paths.repo_root),
    }


def emit_cell(cell_id: str) -> int:
    """Emit one cell's report from the current process (the child entry point).

    The handshake is published before the workload — the parent's attach window
    opens on it — and closed at ``done`` once the report is on disk, so the phase
    line records the progress the cell actually made.
    """
    cell = matrix_cell(cell_id)
    paths = current_perf_paths()
    run_dir = paths.run_dir
    report = paths.report
    if run_dir is None or report is None:
        raise MatrixError(
            "cell_paths_unset", f"{cell_id} needs a run directory and a report"
        )
    guard = _install_guard(paths)
    removed = strip_credentials()
    try:
        publish_handshake(run_dir, cell)
        document = build_report(_cell_records(cell, paths))
        document["cell"] = cell.record()
        document["egress"] = _egress_record(paths, guard)
        guard.assert_closed()
        _write_json(report, document)
        publish_handshake(run_dir, cell, phase=PROFILE_PHASE_DONE)
    finally:
        restore_credentials(removed)
        guard.restore()
    return 0


def _synthetic_records(cell: MatrixCell) -> list[dict[str, Any]]:
    """Emit the contract node's synthetic observation, never campaign data."""
    return [
        build_run_record(
            cell.campaign_id,
            run_id=_run_id(cell),
            run_start_utc=_run_start_utc(),
            measured_ops=len(SYNTHETIC_SAMPLES_MS),
            warmup_ops=0,
            samples=list(SYNTHETIC_SAMPLES_MS),
            mode=cell.mode,
            closure_status=None,
            closure_note=(
                "synthetic contract cell: observation-only, never a campaign baseline"
            ),
        )
    ]


def _cell_records(cell: MatrixCell, paths: PerfPaths) -> list[dict[str, Any]]:
    if cell.synthetic:
        return _synthetic_records(cell)
    if cell.adapter == PILOT_ADAPTER:
        return asyncio.run(_pilot_records(cell))
    return asyncio.run(_headless_records(cell, paths))


def _turn_overhead(turns: Sequence[Any]) -> list[float]:
    """Return the measured application overhead of each turn."""
    return [turn.wall_ms - turn.injected_delay_ms for turn in turns]


_HEADLESS_SERIES: dict[str, Callable[[Sequence[Any]], list[float]]] = {
    "A1": _turn_overhead,
    "B1": _turn_overhead,
}


def _observation_note(cell: MatrixCell) -> str:
    return f"matrix {cell.mode} cell: observation-only, never a campaign baseline"


def _headless_record(cell: MatrixCell, scenario: Any) -> dict[str, Any]:
    series = _HEADLESS_SERIES.get(cell.campaign_id)
    if series is None:
        raise MatrixError(
            "campaign_unsupported", f"{cell.campaign_id} has no headless matrix series"
        )
    warmup = scenario.spec.warmup_turns
    measured = [turn for turn in scenario.turns if turn.turn >= warmup]
    samples = series(measured)
    return build_run_record(
        cell.campaign_id,
        run_id=_run_id(cell),
        run_start_utc=_run_start_utc(),
        measured_ops=len(samples),
        warmup_ops=len(scenario.turns) - len(measured),
        samples=samples,
        mode=cell.mode,
        closure_status=None,
        closure_note=_observation_note(cell),
        fixture_hash=scenario.spec.fixture_hash,
    )


async def _headless_records(cell: MatrixCell, paths: PerfPaths) -> list[dict[str, Any]]:
    from tests.perf._runtime_scenario import FULL_SCALES, build_headless_scenario

    run_dir = paths.run_dir
    if run_dir is None:
        raise MatrixError("cell_paths_unset", "the headless cell has no run directory")
    scenario = await build_headless_scenario(
        cell.workload_cell,
        scale=FULL_SCALES[cell.workload_cell],
        session_root=run_dir / CELL_SESSIONS_DIRNAME,
    )
    try:
        await scenario.run_workload()
        record = _headless_record(cell, scenario)
    finally:
        await scenario.close()
    scenario.assert_cleanup()
    return [record]


async def _pilot_records(cell: MatrixCell) -> list[dict[str, Any]]:
    from tests.perf._runtime_pilot import PilotScenario
    from tests.perf._runtime_scenario import FULL_SCALES

    scenario = PilotScenario(
        cell=cell.workload_cell, scale=FULL_SCALES[CELL_FRESH_HEADLESS]
    )
    await scenario.run_workload()
    d1 = scenario.d1_record()
    coverage = d1["chunks_coverage"]
    samples = list(d1["samples_ms"])
    return [
        build_run_record(
            cell.campaign_id,
            run_id=_run_id(cell),
            run_start_utc=_run_start_utc(),
            measured_ops=len(samples),
            warmup_ops=int(coverage["warmup_flushes"]),
            samples=samples,
            unit_counts={"render_flushes": len(samples), "turns": len(scenario.turns)},
            mode=cell.mode,
            closure_status=None,
            closure_note=_observation_note(cell),
            fixture_hash=scenario.spec.fixture_hash,
        )
    ]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runtime-profile-matrix",
        description="Run the stage-2 runtime profiling matrix, one fresh cell each.",
    )
    parser.add_argument("--cell", help="emit one cell's report (child mode)")
    parser.add_argument(
        "--run-dir", help=f"matrix run directory (defaults to {RUN_DIR_ENV})"
    )
    parser.add_argument(
        "--report",
        help=f"matrix report path (defaults to the run directory's {MATRIX_REPORT_FILENAME})",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the matrix manifest and exit"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help=(
            "one-shot deadline in seconds; never below the expanded estimate"
            f" ({expanded_estimate_s():g}s), defaults to the sum of the cell kill"
            f" deadlines ({one_shot_timeout_s():g}s)"
        ),
    )
    return parser


def _declared_run_root(args: argparse.Namespace) -> Path:
    if args.run_dir:
        declaration: dict[str, str] = {RUN_DIR_ENV: str(args.run_dir)}
        scratchpad = os.environ.get(SCRATCHPAD_ENV)
        if scratchpad:
            declaration[SCRATCHPAD_ENV] = scratchpad
        try:
            paths = resolve_perf_paths(declaration)
        except PerfPathError as error:
            raise MatrixError("run_dir_rejected", error.detail) from error
    else:
        paths = current_perf_paths()
    if paths.run_dir is None:
        raise MatrixError("run_dir_unset", f"a matrix run needs {RUN_DIR_ENV}")
    return paths.run_dir


def _matrix_report_path(args: argparse.Namespace, run_root: Path) -> Path:
    declaration = {
        RUN_DIR_ENV: str(run_root),
        REPORT_ENV: str(args.report or run_root / MATRIX_REPORT_FILENAME),
    }
    try:
        paths = resolve_perf_paths(declaration)
    except PerfPathError as error:
        raise MatrixError("report_rejected", error.detail) from error
    assert paths.report is not None
    return paths.report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the matrix CLI; returns the process exit status.

    Without ``--list`` or ``--cell`` the CLI performs the one-shot run: the full
    expanded matrix, exactly once, inside a deadline that is never shorter than
    the expanded estimate. A halt ends the run, the report carries the escalation
    and the measured wall beside both planning references, and the exit status is
    non-zero — the matrix is never rerun here.
    """
    args = _parser().parse_args(argv)
    if args.list:
        print(json.dumps(matrix_manifest(), indent=2, sort_keys=True))
        return 0
    if args.cell:
        return emit_cell(args.cell)
    one_shot_deadline = _one_shot_deadline(args.timeout)
    run_root = _declared_run_root(args)
    report = _matrix_report_path(args, run_root)
    started = perf_counter()
    outcome = run_matrix(run_root=run_root, overall_timeout_s=one_shot_deadline)
    measured_wall_s = perf_counter() - started
    document = write_matrix_report(
        outcome, path=report, measured_wall_s=measured_wall_s
    )
    print(
        json.dumps(
            {
                **outcome.record(),
                "report": str(report),
                "runs": len(document["runs"]),
                "one_shot": document["one_shot"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        0
        if not outcome.halted
        and len(outcome.cells) == len(MATRIX_CELLS)
        and all(cell.status == CELL_OK for cell in outcome.cells)
        else 1
    )


__all__ = [
    "ADAPTERS",
    "BYTECODE_ENV",
    "CAMPAIGN_PRICE_S",
    "CELLS_BY_ID",
    "CELL_CONFIG_DIRNAME",
    "CELL_FAILED",
    "CELL_HALTED",
    "CELL_OK",
    "CELL_REPORT_FILENAME",
    "CELL_RUN_DIRNAME",
    "CELL_SCRATCH_DIRNAME",
    "CELL_SESSIONS_DIRNAME",
    "CELL_STDERR_FILENAME",
    "CELL_STDOUT_FILENAME",
    "CONFIG_HOME_ENV",
    "HANDSHAKE_DEADLINE_S",
    "HANDSHAKE_MALFORMED",
    "HANDSHAKE_PID_MISMATCH",
    "HANDSHAKE_PROCESS_EXITED",
    "HANDSHAKE_READY",
    "HANDSHAKE_TIMEOUT",
    "HEADLESS_ADAPTER",
    "KILL_DEADLINE_FACTOR",
    "MATRIX_CELLS",
    "MATRIX_CLASS_FLOOR_S",
    "MATRIX_REPORT_FILENAME",
    "MATRIX_SCHEMA",
    "ONE_SHOT_SCHEMA",
    "PARTIAL_SUFFIX",
    "PILOT_ADAPTER",
    "PLANNING_REFERENCE_S",
    "PROFILER_ALLOWANCE_S",
    "PROFILER_ATTACHED",
    "PROFILER_AVAILABLE",
    "PROFILER_FAILED",
    "PROFILER_NOT_ATTACHED",
    "PROFILER_OUTPUT_FILENAME",
    "PROFILER_UNAVAILABLE",
    "PROFILE_PHASES",
    "PROFILE_PHASE_DONE",
    "PROFILE_PHASE_FILENAME",
    "PROFILE_PHASE_READY",
    "PROFILE_PID_FILENAME",
    "PYSPY_EXECUTABLE",
    "SETUP_ALLOWANCE_S",
    "SYNTHETIC_CELL",
    "SYNTHETIC_CELL_BUDGET_S",
    "SYNTHETIC_CELL_ID",
    "SYNTHETIC_SAMPLES_MS",
    "CellOutcome",
    "HandshakeWait",
    "MatrixCell",
    "MatrixError",
    "MatrixOutcome",
    "ProfileHandshake",
    "ProfilerRun",
    "assert_cell_environment",
    "cell_environment",
    "cell_report_path",
    "cell_root",
    "cell_run_dir",
    "clear_handshake",
    "credential_names",
    "discard_cell_output",
    "emit_cell",
    "expanded_estimate_s",
    "main",
    "matrix_cell",
    "matrix_manifest",
    "one_shot_record",
    "one_shot_timeout_s",
    "parse_phase_line",
    "prepare_cell_dirs",
    "profiler_status",
    "publish_handshake",
    "pyspy_path",
    "read_handshake",
    "revalidation_record",
    "run_cell",
    "run_matrix",
    "start_profiler",
    "stop_profiler",
    "wait_for_handshake",
    "write_matrix_report",
]

if __name__ == "__main__":
    raise SystemExit(main())
