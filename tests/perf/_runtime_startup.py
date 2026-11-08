"""Cold-start measurement harness for the v0.7.5 stage-2 gate (X4).

Cold-start phases are OBSERVATION-ONLY (plan §4): a cold start happens once per
process, so there is no repeatable operation unit a threshold could be built
on. Everything here records a phase with its reason and ``gated: False``
instead of producing a campaign row. CLI peak memory is the same kind of
one-observation-per-process quantity and belongs to X6b's external hyperfine
contract, not to this harness.

Two halves:

* **``-X importtime``.** :func:`parse_importtime` reads CPython's import-time
  report (self and cumulative microseconds per imported module, indented by
  import depth) and :func:`importtime_summary` ranks it. The format is
  ``import time: <self us> | <cumulative us> | <indent><module>``, where a
  module's indentation is ``2 * depth + 1`` spaces; a line that claims to be an
  import-time line and does not parse is rejected, never skipped silently.
* **Startup path.** :data:`STARTUP_PATH_PHASES` is the observed path from
  interpreter launch through package import and config load to CLI boot, the
  session/resume picker and the first boot frame (design v3 :98). The
  production anchor is ``chartreux.cli.process_start.PROCESS_START_MONOTONIC``,
  captured as early as the CLI entrypoint can capture it, so a cold-start cell
  reports its phases against the same origin the application itself uses.

The in-suite contract node drives ONE tiny synthetic probe in a fresh
interpreter; the real cold-start cells (config load, CLI boot, picker, boot
frame) are external, exactly as the matrix and soak cells are. The probe
installs and asserts its own fail-closed egress policy — socket block,
credential strip, config/scratch redirection, preserved
``PYTHONDONTWRITEBYTECODE`` — because the pytest fixtures never reach a
subprocess. A probe that cannot prove one of those properties fails the
harness instead of degrading it, and a probe that outlives its deadline is a
halt with its partial output discarded: there is no retry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter_ns
from typing import Any

from tests.perf._runtime_metrics import (
    REPORT_ENV,
    RUN_DIR_ENV,
    SCRATCHPAD_ENV,
    repository_root,
)
from tests.perf._runtime_profile import BYTECODE_ENV, CONFIG_HOME_ENV, credential_names

STARTUP_SCHEMA = "chartreux.runtime_startup.v1"
PROBE_SCHEMA = "chartreux.runtime_startup_probe.v1"

#: Plan §4's recorded reason for every phase this harness reports.
OBSERVATION_ONLY_REASON = (
    "cold-start phases are observation-only: one observation per process, so"
    " there is no repeatable operation unit to gate (plan §4)"
)
NOT_DRIVEN_REASON = (
    "not driven in-suite: the contract node runs only a tiny synthetic startup"
    " probe; the real cold-start phase is an external cell"
)
ONE_OBSERVATION_RULE = "one observation per process per phase"

PROBE_FILENAME = "startup_probe.py"
PROBE_TARGET = "chartreux"
PROBE_CONFIG_DIRNAME = "config"
PROBE_SCRATCH_DIRNAME = "scratch"
START_TIME_ENV = "CHARTREUX_STARTUP_T0_NS"
REPO_ROOT_ENV = "CHARTREUX_STARTUP_REPO_ROOT"
PROBE_TIMEOUT_S = 1.5

INTERPRETER_BOOT = "interpreter_boot"
APP_IMPORT = "app_import"
CONFIG_LOAD = "config_load"
CLI_BOOT = "cli_boot"
SESSION_PICKER = "session_picker"
BOOT = "boot"
PROBE_READY = "probe_ready"

STAGE_PROBE = "probe"
STAGE_IMPORTTIME = "importtime"
STAGE_EXTERNAL = "external_cell"

_IMPORTTIME_PREFIX = "import time:"
_IMPORTTIME_HEADER = "import time: self [us] | cumulative | imported package"
_IMPORTTIME_PATTERN = re.compile(
    r"\Aimport time:\s*(?P<self>\d+)\s*\|\s*(?P<cumulative>\d+)\s*\|(?P<module>.*)\Z"
)
_TARGET_PATTERN = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_ANCHOR_TOLERANCE_MS = 1.0
_OUTPUT_TAIL_CHARS = 400

#: The only inherited variables the probe may see: process plumbing, nothing
#: that can carry a credential or point at the repository's own configuration.
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


class StartupError(RuntimeError):
    """A startup observation violates the harness contract."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"startup observation rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------
# The observed cold-start path
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StartupPhaseSpec:
    """One phase of the observed cold-start path."""

    name: str
    description: str
    source: str
    driven_in_suite: bool

    def record(self) -> dict[str, Any]:
        return {
            "phase": self.name,
            "description": self.description,
            "source": self.source,
            "driven_in_suite": self.driven_in_suite,
            "gated": False,
            "reason": OBSERVATION_ONLY_REASON,
        }


STARTUP_PATH_PHASES: tuple[StartupPhaseSpec, ...] = (
    StartupPhaseSpec(
        INTERPRETER_BOOT, "interpreter launch to probe entry", STAGE_PROBE, True
    ),
    StartupPhaseSpec(
        APP_IMPORT,
        "package import, cross-checked by -X importtime",
        STAGE_IMPORTTIME,
        True,
    ),
    StartupPhaseSpec(
        CONFIG_LOAD, "config resolution from CHARTREUX_HOME", STAGE_EXTERNAL, False
    ),
    StartupPhaseSpec(CLI_BOOT, "CLI entrypoint construction", STAGE_EXTERNAL, False),
    StartupPhaseSpec(
        SESSION_PICKER, "session/resume picker construction", STAGE_EXTERNAL, False
    ),
    StartupPhaseSpec(BOOT, "boot screen and first frame", STAGE_EXTERNAL, False),
)

PROBE_PHASE_NAMES: tuple[str, ...] = (INTERPRETER_BOOT, APP_IMPORT, PROBE_READY)

_KNOWN_PHASES: tuple[str, ...] = (
    *(spec.name for spec in STARTUP_PATH_PHASES),
    PROBE_READY,
)


def phase_plan_record() -> dict[str, Any]:
    """Return the observed cold-start path and its observation-only basis."""
    return {
        "schema": STARTUP_SCHEMA,
        "gated": False,
        "observation_only": True,
        "reason": OBSERVATION_ONLY_REASON,
        "one_observation_rule": ONE_OBSERVATION_RULE,
        "probe_phases": list(PROBE_PHASE_NAMES),
        "phases": [spec.record() for spec in STARTUP_PATH_PHASES],
    }


@dataclass(frozen=True, slots=True)
class StartupPhaseObservation:
    """One process's measurement of one phase, never a gated sample."""

    phase: str
    wall_ms: float
    cumulative_ms: float | None = None
    synthetic: bool = True
    detail: str | None = None

    def record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "wall_ms": round(self.wall_ms, 3),
            "cumulative_ms": (
                None if self.cumulative_ms is None else round(self.cumulative_ms, 3)
            ),
            "synthetic": self.synthetic,
            "detail": self.detail,
            "gated": False,
            "reason": OBSERVATION_ONLY_REASON,
        }


# --------------------------------------------------------------------------
# -X importtime
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImportTimeEntry:
    """One module's import cost as CPython reported it."""

    module: str
    self_us: int
    cumulative_us: int
    depth: int

    def record(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "self_us": self.self_us,
            "cumulative_us": self.cumulative_us,
            "depth": self.depth,
        }


@dataclass(frozen=True, slots=True)
class ImportTimeParse:
    """A parsed import-time report and the lines it did not claim."""

    entries: tuple[ImportTimeEntry, ...]
    ignored_lines: tuple[str, ...]


def _module_and_depth(field: str) -> tuple[str, int]:
    """Split one import-time module field into its name and import depth."""
    stripped = field.rstrip()
    module = stripped.lstrip(" ")
    indent = len(stripped) - len(module)
    if not module or indent < 1 or indent % 2 == 0:
        raise StartupError(
            "importtime_malformed", f"unexpected module indentation: {field!r}"
        )
    return module, (indent - 1) // 2


def parse_importtime(text: str) -> ImportTimeParse:
    """Parse CPython's ``-X importtime`` report.

    Non-import-time lines are preserved as ``ignored_lines``. A line that
    claims to be an import-time line but does not parse raises: the harness
    owns this format and never guesses at a truncated or corrupted report.
    """
    entries: list[ImportTimeEntry] = []
    ignored: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if not line.startswith(_IMPORTTIME_PREFIX):
            ignored.append(line)
            continue
        if line.strip() == _IMPORTTIME_HEADER:
            ignored.append(line)
            continue
        match = _IMPORTTIME_PATTERN.match(line)
        if match is None:
            raise StartupError("importtime_malformed", line)
        module, depth = _module_and_depth(match.group("module"))
        entries.append(
            ImportTimeEntry(
                module=module,
                self_us=int(match.group("self")),
                cumulative_us=int(match.group("cumulative")),
                depth=depth,
            )
        )
    return ImportTimeParse(entries=tuple(entries), ignored_lines=tuple(ignored))


def _ranked(
    entries: Sequence[ImportTimeEntry], attribute: str
) -> list[ImportTimeEntry]:
    """Rank entries by one cost column, deterministically on ties."""
    return sorted(
        entries,
        key=lambda entry: (
            -getattr(entry, attribute),
            entry.module,
            entry.depth,
            entry.cumulative_us,
        ),
    )


def importtime_summary(
    parsed: ImportTimeParse, *, target: str | None = None, limit: int = 10
) -> dict[str, Any]:
    """Summarize a parsed report: totals, target cost, and both rankings.

    A module may appear several times (a failed import attempt is retried under
    another path), so the target's cost is the largest cumulative cost of its
    occurrences: the successful import of the top-level package carries its
    whole subtree, while a failed attempt does not.
    """
    if limit < 1:
        raise StartupError("importtime_limit_invalid", f"{limit!r}")
    entries = parsed.entries
    top_level = [entry for entry in entries if entry.depth == 0]
    record: dict[str, Any] = {
        "entry_count": len(entries),
        "ignored_line_count": len(parsed.ignored_lines),
        "total_self_us": sum(entry.self_us for entry in entries),
        "total_cumulative_us": sum(entry.cumulative_us for entry in top_level),
        "max_depth": max((entry.depth for entry in entries), default=0),
        "top_self": [entry.record() for entry in _ranked(entries, "self_us")[:limit]],
        "top_cumulative": [
            entry.record() for entry in _ranked(entries, "cumulative_us")[:limit]
        ],
        "target": None,
    }
    if target is not None:
        occurrences = [entry for entry in entries if entry.module == target]
        if occurrences:
            record["target"] = {
                "module": target,
                "occurrences": len(occurrences),
                "max_self_us": max(entry.self_us for entry in occurrences),
                "max_cumulative_us": max(entry.cumulative_us for entry in occurrences),
            }
    return record


def importtime_command(target: str, *, executable: str | None = None) -> list[str]:
    """Build the interpreter invocation a cold-start cell runs for ``target``."""
    if _TARGET_PATTERN.match(target) is None:
        raise StartupError(
            "importtime_target_invalid", f"{target!r} is not a dotted module name"
        )
    return [
        sys.executable if executable is None else executable,
        "-X",
        "importtime",
        "-c",
        f"import {target}",
    ]


# --------------------------------------------------------------------------
# Probe environment (fail-closed egress, no measurement paths)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeEnvironment:
    """The validated environment one startup probe process runs under."""

    config_home: Path
    scratch_home: Path
    repo_root: Path
    environment: dict[str, str]


def prepare_probe_root(root: Path) -> Path:
    """Create the probe's config and scratch homes under ``root``."""
    for name in (PROBE_CONFIG_DIRNAME, PROBE_SCRATCH_DIRNAME):
        (Path(root) / name).mkdir(parents=True, exist_ok=True)
    return Path(root)


def _is_at_or_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _require_directory_outside_repo(raw: str | None, label: str, repo: Path) -> Path:
    """Return one declared home, or raise: fail closed on every escape."""
    if raw is None or not raw.strip():
        raise StartupError(f"{label}_unset", f"{label} is not set")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise StartupError(f"{label}_relative", f"{label} must be absolute: {raw}")
    home = candidate.resolve()
    if home != candidate:
        raise StartupError(
            f"{label}_not_canonical", f"{label} traverses a symlink: {raw}"
        )
    if not home.is_dir():
        raise StartupError(f"{label}_missing", f"{label} is not a directory: {home}")
    if _is_at_or_within(home, repo):
        raise StartupError(
            f"{label}_inside_repository", f"{label} is at or under {repo}: {home}"
        )
    if home in repo.parents:
        raise StartupError(
            f"{label}_ancestor_of_repo_root", f"{label} is an ancestor of {repo}"
        )
    return home


def probe_environment(
    probe_root: Path, *, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build the fresh environment one startup probe runs under.

    The environment is built from an allowlist, so nothing credential-shaped is
    inherited; every path it names is derived from the probe's own directory,
    and the measurement paths (``CHARTREUX_PERF_RUN_DIR`` /
    ``CHARTREUX_PERF_REPORT``) are deliberately absent: a cold-start probe
    writes no report and never becomes measurement output.
    """
    source = os.environ if base is None else base
    root = Path(probe_root)
    repository = str(repository_root())
    environment = {name: source[name] for name in _PASSTHROUGH_ENV if name in source}
    environment[BYTECODE_ENV] = "1"
    environment[CONFIG_HOME_ENV] = str(root / PROBE_CONFIG_DIRNAME)
    environment[SCRATCHPAD_ENV] = str(root / PROBE_SCRATCH_DIRNAME)
    environment[REPO_ROOT_ENV] = repository
    environment.pop(RUN_DIR_ENV, None)
    environment.pop(REPORT_ENV, None)
    inherited = source.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        repository if not inherited else f"{repository}{os.pathsep}{inherited}"
    )
    for name in credential_names(environment):
        del environment[name]
    return environment


def assert_probe_environment(
    environment: Mapping[str, str], *, repo_root: Path | None = None
) -> ProbeEnvironment:
    """Validate a probe environment, returning its resolved homes."""
    repo = repository_root() if repo_root is None else Path(repo_root)
    repo = repo.resolve()
    if environment.get(BYTECODE_ENV) != "1":
        raise StartupError(
            "bytecode_enabled", "a probe must never write bytecode into the tree"
        )
    present = credential_names(environment)
    if present:
        raise StartupError("credentials_present", f"{list(present)}")
    declared_repo = environment.get(REPO_ROOT_ENV)
    if declared_repo is None or Path(declared_repo).resolve() != repo:
        raise StartupError(
            "repo_root_mismatch", f"the probe must carry {repo}, not {declared_repo!r}"
        )
    for name in (RUN_DIR_ENV, REPORT_ENV):
        if name in environment:
            raise StartupError(
                "probe_paths_present",
                f"a cold-start probe carries no measurement path ({name})",
            )
    config_home = _require_directory_outside_repo(
        environment.get(CONFIG_HOME_ENV), "config_home", repo
    )
    scratch_home = _require_directory_outside_repo(
        environment.get(SCRATCHPAD_ENV), "scratch_home", repo
    )
    return ProbeEnvironment(
        config_home=config_home,
        scratch_home=scratch_home,
        repo_root=repo,
        environment=dict(environment),
    )


# --------------------------------------------------------------------------
# The probe process
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeProcess:
    """One probe invocation's raw outcome."""

    wall_ms: float
    returncode: int
    stdout: str
    stderr: str
    command: tuple[str, ...]
    probe_path: Path

    def record(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "wall_ms": round(self.wall_ms, 3),
            "command": list(self.command),
            "probe_path": str(self.probe_path),
        }


def probe_command(probe_path: Path, *, executable: str | None = None) -> list[str]:
    """Build the ``-X importtime`` invocation of the probe script."""
    return [
        sys.executable if executable is None else executable,
        "-X",
        "importtime",
        str(Path(probe_path)),
    ]


def write_probe(root: Path) -> Path:
    """Write the probe script under the probe root and return its path."""
    path = Path(root) / PROBE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROBE_SOURCE, encoding="utf-8")
    return path


def spawn_probe(
    probe_path: Path,
    environment: Mapping[str, str],
    *,
    executable: str | None = None,
    timeout_s: float = PROBE_TIMEOUT_S,
    cwd: Path | None = None,
) -> ProbeProcess:
    """Run the probe once. A timeout is a halt with no rerun and no output."""
    if timeout_s <= 0:
        raise StartupError("probe_timeout_not_positive", f"{timeout_s!r}")
    command = probe_command(probe_path, executable=executable)
    started = perf_counter_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=str(repository_root() if cwd is None else cwd),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise StartupError(
            "probe_timeout",
            f"the probe outlived {timeout_s:g}s; its partial output is discarded"
            " and the observation is not retried",
        ) from error
    wall_ms = (perf_counter_ns() - started) / _NANOSECONDS_PER_MILLISECOND
    return ProbeProcess(
        wall_ms=wall_ms,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=tuple(command),
        probe_path=Path(probe_path),
    )


def _output_tail(text: str) -> str:
    tail = text.strip()
    return tail[-_OUTPUT_TAIL_CHARS:]


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    """Return one finite non-negative number, or raise."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise StartupError(
            "probe_output_invalid", f"{label} is not a number: {value!r}"
        )
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise StartupError(
            "probe_output_invalid", f"{label} is out of range: {number!r}"
        )
    return number


def _probe_phases(document: Mapping[str, Any]) -> tuple[StartupPhaseObservation, ...]:
    """Validate the probe's phase list and return it as observations."""
    raw_phases = document.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise StartupError("probe_output_invalid", "the probe reported no phase")
    observations: list[StartupPhaseObservation] = []
    previous = -1.0
    for index, raw in enumerate(raw_phases):
        if not isinstance(raw, dict):
            raise StartupError(
                "probe_output_invalid", f"phase {index} is not an object"
            )
        name = raw.get("phase")
        if not isinstance(name, str) or name not in PROBE_PHASE_NAMES:
            raise StartupError("probe_output_invalid", f"unknown probe phase: {name!r}")
        at_ms = _number(raw.get("at_ms"), f"{name}.at_ms")
        wall_ms = _number(raw.get("wall_ms"), f"{name}.wall_ms")
        if at_ms < previous:
            raise StartupError("probe_output_invalid", f"{name} is out of order")
        previous = at_ms
        observations.append(
            StartupPhaseObservation(
                phase=name, wall_ms=wall_ms, cumulative_ms=at_ms, synthetic=True
            )
        )
    if [observation.phase for observation in observations] != list(PROBE_PHASE_NAMES):
        raise StartupError(
            "probe_output_invalid",
            f"the probe phases are {[o.phase for o in observations]},"
            f" expected {list(PROBE_PHASE_NAMES)}",
        )
    return tuple(observations)


def _probe_egress(document: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Validate the probe's own egress evidence; a gap fails the harness."""
    egress = document.get("egress")
    if not isinstance(egress, dict):
        raise StartupError("probe_output_invalid", "egress evidence is missing")
    for flag in ("asserted", "socket_block", "bytecode_disabled"):
        if egress.get(flag) is not True:
            raise StartupError("probe_egress_unproven", f"{flag} is not true")
    present = egress.get("credentials_present")
    if not isinstance(present, list) or present:
        raise StartupError("probe_egress_unproven", f"credentials present: {present!r}")
    for label in ("config_home", "scratch_home"):
        raw = egress.get(label)
        if not isinstance(raw, str) or not raw:
            raise StartupError("probe_output_invalid", f"{label} is missing")
        home = Path(raw).resolve()
        if not home.is_dir() or _is_at_or_within(home, repo) or home in repo.parents:
            raise StartupError(
                "probe_egress_unproven",
                f"{label} is not disjoint from the repository: {home}",
            )
    declared_repo = egress.get("repo_root")
    if not isinstance(declared_repo, str) or Path(declared_repo).resolve() != repo:
        raise StartupError(
            "probe_output_invalid", "the probe reported another repository root"
        )
    return {**egress}


def _probe_anchor(
    document: Mapping[str, Any], phases: Sequence[StartupPhaseObservation]
) -> dict[str, Any]:
    """Validate the production anchor against the import phase it lands in."""
    anchor = document.get("process_anchor")
    if not isinstance(anchor, dict):
        raise StartupError("probe_output_invalid", "process_anchor is missing")
    monotonic_s = _number(anchor.get("monotonic_s"), "process_anchor.monotonic_s")
    if monotonic_s <= 0.0:
        raise StartupError("probe_output_invalid", "process_anchor.monotonic_s is zero")
    delta_ms = _number(anchor.get("delta_ms"), "process_anchor.delta_ms")
    import_phase = next(phase for phase in phases if phase.phase == APP_IMPORT)
    start = import_phase.cumulative_ms or 0.0
    lower = start - _ANCHOR_TOLERANCE_MS
    upper = start + import_phase.wall_ms + _ANCHOR_TOLERANCE_MS
    if not lower <= delta_ms <= upper:
        raise StartupError(
            "probe_anchor_mismatch",
            f"the production anchor lands at {delta_ms:.3f} ms, outside the"
            f" app_import phase [{start:.3f}, {upper:.3f}]",
        )
    return {**anchor}


def _probe_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    probe = document.get("probe")
    if not isinstance(probe, dict):
        raise StartupError("probe_output_invalid", "probe metadata is missing")
    entry_ns = probe.get("entry_ns")
    if not isinstance(entry_ns, int) or isinstance(entry_ns, bool) or entry_ns <= 0:
        raise StartupError("probe_output_invalid", f"entry_ns is invalid: {entry_ns!r}")
    _number(probe.get("total_ms"), "probe.total_ms")
    return {**probe}


@dataclass(frozen=True, slots=True)
class StartupProbeResult:
    """One validated synthetic cold-start observation, never gated data."""

    process: ProbeProcess
    phases: tuple[StartupPhaseObservation, ...]
    importtime: ImportTimeParse
    egress: dict[str, Any]
    anchor: dict[str, Any]
    probe: dict[str, Any]
    target_summary: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return build_startup_record(
            self.phases,
            importtime=self.importtime,
            target=PROBE_TARGET,
            egress=self.egress,
            probe={**self.process.record(), **self.probe},
            anchor=self.anchor,
        )


def run_startup_probe(
    root: Path, *, executable: str | None = None, timeout_s: float = PROBE_TIMEOUT_S
) -> StartupProbeResult:
    """Drive the tiny synthetic cold-start probe in a fresh interpreter.

    The probe root must be an existing canonical directory outside the
    repository; its config and scratch homes are created under it. The probe is
    spawned once: a non-zero exit, a timed-out invocation, unparseable output or
    unproven egress all raise, and nothing is retried.
    """
    candidate = Path(root)
    if not candidate.is_absolute():
        raise StartupError(
            "probe_root_relative", f"the probe root must be absolute: {root}"
        )
    resolved = candidate.resolve()
    if resolved != candidate:
        raise StartupError(
            "probe_root_not_canonical", f"the probe root traverses a symlink: {root}"
        )
    repo = repository_root()
    if _is_at_or_within(resolved, repo):
        raise StartupError(
            "probe_root_under_repo_root", f"the probe root is at or under {repo}"
        )
    if resolved in repo.parents:
        raise StartupError(
            "probe_root_ancestor_of_repo_root",
            f"the probe root is an ancestor of {repo}",
        )
    prepare_probe_root(resolved)
    environment = probe_environment(resolved)
    assert_probe_environment(environment, repo_root=repo)
    probe_path = write_probe(resolved)
    environment[START_TIME_ENV] = str(perf_counter_ns())
    process = spawn_probe(
        probe_path, environment, executable=executable, timeout_s=timeout_s
    )
    if process.returncode != 0:
        raise StartupError(
            "probe_failed",
            f"the probe exited {process.returncode}: {_output_tail(process.stderr)}",
        )
    try:
        document = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise StartupError(
            "probe_output_invalid", f"stdout is not JSON: {error}"
        ) from error
    if not isinstance(document, dict) or document.get("schema") != PROBE_SCHEMA:
        raise StartupError("probe_output_invalid", f"unexpected document: {document!r}")
    phases = _probe_phases(document)
    egress = _probe_egress(document, repo)
    anchor = _probe_anchor(document, phases)
    probe = _probe_metadata(document)
    importtime = parse_importtime(process.stderr)
    if not importtime.entries:
        raise StartupError(
            "importtime_missing", "the probe produced no -X importtime report"
        )
    target_summary = importtime_summary(importtime, target=PROBE_TARGET)["target"]
    if target_summary is None:
        raise StartupError(
            "importtime_target_missing",
            f"the probe's report never imported {PROBE_TARGET}",
        )
    return StartupProbeResult(
        process=process,
        phases=phases,
        importtime=importtime,
        egress=egress,
        anchor=anchor,
        probe=probe,
        target_summary=target_summary,
    )


# --------------------------------------------------------------------------
# The observation record
# --------------------------------------------------------------------------


def build_startup_record(
    phases: Sequence[StartupPhaseObservation],
    *,
    importtime: ImportTimeParse | None = None,
    target: str | None = None,
    egress: Mapping[str, Any] | None = None,
    probe: Mapping[str, Any] | None = None,
    anchor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one observation-only cold-start record.

    ``phases`` may carry external-cell phases (config load, CLI boot, the
    session picker, the boot frame) as well as the probe's own — the record
    never gates any of them. A phase observed twice in one process is refused:
    one observation per process is the rule that makes these phases
    observation-only in the first place. Path phases with no observation are
    reported as ``not_driven`` with their reason, never fabricated.
    """
    seen: set[str] = set()
    ordered: list[StartupPhaseObservation] = []
    for observation in phases:
        if observation.phase not in _KNOWN_PHASES:
            raise StartupError("unknown_phase", f"{observation.phase!r}")
        if observation.phase in seen:
            raise StartupError(
                "duplicate_phase_observation",
                f"{observation.phase} ({ONE_OBSERVATION_RULE})",
            )
        if not math.isfinite(observation.wall_ms) or observation.wall_ms < 0.0:
            raise StartupError(
                "phase_wall_invalid", f"{observation.phase}: {observation.wall_ms!r}"
            )
        if observation.cumulative_ms is not None and (
            not math.isfinite(observation.cumulative_ms)
            or observation.cumulative_ms < 0.0
        ):
            raise StartupError(
                "phase_cumulative_invalid",
                f"{observation.phase}: {observation.cumulative_ms!r}",
            )
        seen.add(observation.phase)
        ordered.append(observation)
    order = {name: index for index, name in enumerate(_KNOWN_PHASES)}
    ordered.sort(key=lambda observation: order[observation.phase])
    not_driven = [
        {"phase": spec.name, "gated": False, "reason": NOT_DRIVEN_REASON}
        for spec in STARTUP_PATH_PHASES
        if spec.name not in seen
    ]
    return {
        "schema": STARTUP_SCHEMA,
        "gated": False,
        "observation_only": True,
        "reason": OBSERVATION_ONLY_REASON,
        "one_observation_per_process": True,
        "processes": 1,
        "phases": [observation.record() for observation in ordered],
        "not_driven": not_driven,
        "importtime": (
            None
            if importtime is None
            else importtime_summary(importtime, target=target)
        ),
        "egress": None if egress is None else dict(egress),
        "anchor": None if anchor is None else dict(anchor),
        "probe": None if probe is None else dict(probe),
    }


# --------------------------------------------------------------------------
# The child script (self-contained: the pytest fixtures never reach it)
# --------------------------------------------------------------------------

PROBE_SOURCE = """# Tiny synthetic cold-start probe (X4 contract node, test side only).
# Fail-closed egress is installed and asserted here because no pytest fixture
# reaches a subprocess. One observation per process, never gated data.
import json
import os
from pathlib import Path
import socket
import time

MARKER = "startup probe egress blocked"
CREDENTIAL_MARKERS = ("API_KEY", "APIKEY", "CREDENTIAL", "PASSWORD", "SECRET", "TOKEN")
NANOSECONDS_PER_MILLISECOND = 1000000.0


def install_guard():
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(sock, address):
        raise OSError(MARKER)

    def blocked_connect_ex(sock, address):
        raise OSError(MARKER)

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect_ex
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            probe.connect(("192.0.2.1", 80))
        except OSError as error:
            if MARKER not in str(error):
                raise AssertionError(
                    "non-loopback connect failed for another reason: %s" % error
                )
        else:
            raise AssertionError("a non-loopback connect was not blocked")
    finally:
        probe.close()
    return original_connect is not None and original_connect_ex is not None


def assert_environment():
    present = sorted(
        name
        for name in os.environ
        if any(marker in name.upper() for marker in CREDENTIAL_MARKERS)
    )
    if present:
        raise AssertionError("provider credentials leaked into the probe: %s" % present)
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise AssertionError("bytecode writing is not disabled")
    repo = Path(os.environ["CHARTREUX_STARTUP_REPO_ROOT"]).resolve()
    homes = {
        name: Path(os.environ[name]).resolve()
        for name in ("CHARTREUX_HOME", "CHARTREUX_PERF_SCRATCHPAD")
    }
    for name, home in homes.items():
        if not home.is_dir():
            raise AssertionError("%s is not a directory: %s" % (name, home))
        if home == repo or repo in home.parents or home in repo.parents:
            raise AssertionError(
                "%s is not disjoint from the repository: %s" % (name, home)
            )
    return repo, homes


def main():
    started = time.perf_counter_ns()
    t0 = int(os.environ["CHARTREUX_STARTUP_T0_NS"])
    phases = [
        {
            "phase": "interpreter_boot",
            "at_ms": (started - t0) / NANOSECONDS_PER_MILLISECOND,
            "wall_ms": (started - t0) / NANOSECONDS_PER_MILLISECOND,
        }
    ]
    repo, homes = assert_environment()
    socket_block = install_guard()
    import_started = time.perf_counter_ns()
    import chartreux  # noqa: F401 - the package import is the observed phase
    import chartreux.cli.process_start
    import_done = time.perf_counter_ns()
    phases.append(
        {
            "phase": "app_import",
            "at_ms": (import_started - t0) / NANOSECONDS_PER_MILLISECOND,
            "wall_ms": (import_done - import_started) / NANOSECONDS_PER_MILLISECOND,
        }
    )
    ready = time.perf_counter_ns()
    phases.append(
        {
            "phase": "probe_ready",
            "at_ms": (ready - t0) / NANOSECONDS_PER_MILLISECOND,
            "wall_ms": 0.0,
        }
    )
    anchor = chartreux.cli.process_start.PROCESS_START_MONOTONIC
    document = {
        "schema": "chartreux.runtime_startup_probe.v1",
        "phases": phases,
        "egress": {
            "asserted": True,
            "socket_block": socket_block,
            "credentials_present": [],
            "bytecode_disabled": os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
            "config_home": str(homes["CHARTREUX_HOME"]),
            "scratch_home": str(homes["CHARTREUX_PERF_SCRATCHPAD"]),
            "repo_root": str(repo),
        },
        "process_anchor": {
            "monotonic_s": anchor,
            "delta_ms": (
                anchor * 1000.0 - t0 / NANOSECONDS_PER_MILLISECOND
            ),
        },
        "probe": {
            "entry_ns": started,
            "total_ms": (ready - t0) / NANOSECONDS_PER_MILLISECOND,
        },
    }
    print(json.dumps(document))


main()
"""
