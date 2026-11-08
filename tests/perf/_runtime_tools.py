"""X6a tooling contracts: pyinstrument, memray and pytest-memray.

The three tools answer different questions and are all observation-only: none of
them feeds a gated campaign row, and none of them may be read as a latency
measurement. Each tool therefore lands with its own contract here — the quantity
it measures, its unit, the artifact a run leaves behind, its invocation and what
its numbers may and may not be used for.

THE CO-RUN RULE (plan §4, "allocation instrumentation must not contaminate
latency"): allocation and sampling tooling never co-runs with a latency gate. A
run carries either gated latency measurement or one of these tools, never both,
because the instrument changes the wall clock it would otherwise be asked to
certify. :func:`plan_tool_run` enforces the rule rather than restating it.

X6b adds the one external tool: hyperfine times a whole CLI process from the
outside and the measured process records its own peak resident memory. Both
quantities are observation-only (plan §4: CLI peak memory is one observation
per process). :func:`hyperfine_contract` pins the quantity, the offline config,
the peak rule and the refusal to co-run with a latency gate or with an
instrumented tool; :func:`plan_hyperfine_run`, :func:`write_cli_wrapper`,
:func:`parse_hyperfine_export` and :func:`collect_peak_observations` build and
read such a cell without ever running a benchmark in-suite.

Tool absence is an honest unavailable branch: :func:`tool_availability` records
the missing module or executable with its reason, and the run is skipped. An
absent tool is never elevated into a pass or a failure.

A tool process runs under the WP3b egress guard: non-loopback sockets are
refused, credentials are stripped by allowlist construction, the config and temp
paths point under the run directory, and bytecode writing stays disabled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from tests.perf._runtime_metrics import (
    REPORT_ENV,
    RUN_DIR_ENV,
    PerfPathError,
    PerfPaths,
    repository_root,
    resolve_perf_paths,
)

SCHEMA_ID = "chartreux.runtime_tools.v1"

TOOL_MEMRAY = "memray"
TOOL_PYINSTRUMENT = "pyinstrument"
TOOL_PYTEST_MEMRAY = "pytest-memray"
TOOL_NAMES: tuple[str, ...] = (TOOL_MEMRAY, TOOL_PYINSTRUMENT, TOOL_PYTEST_MEMRAY)

KIND_ALLOCATION_TRACER = "allocation-tracer"
KIND_SAMPLING_PROFILER = "sampling-profiler"
KIND_ALLOCATION_PLUGIN = "allocation-test-plugin"

STATUS_AVAILABLE = "available"
STATUS_UNAVAILABLE = "unavailable"

#: Plan §4: allocation instrumentation must never contaminate latency.
LATENCY_CO_RUN_RULE = (
    "allocation and sampling tooling never co-runs with a latency gate: a run "
    "carries either gated latency measurement or one of these tools, never both"
)
ABSENCE_RULE = (
    "an absent tool is an honest unavailable branch: the tool run is skipped "
    "with its reason recorded and is never elevated to a pass or a failure"
)
EGRESS_RULE = (
    "a tool process runs under the WP3b EgressGuard: non-loopback sockets are "
    "refused, credentials are stripped, config and temp paths point under the "
    "run directory, and bytecode writing is disabled"
)
COVERAGE_NOTE = (
    "tool runs are observation-only infrastructure: they gate no registry row "
    "and so carry no chunks-coverage floor; a matrix cell that runs a tool "
    "reports the WP7a/WP6 cell record"
)

CONFIG_HOME_ENV = "CHARTREUX_HOME"
BYTECODE_ENV = "PYTHONDONTWRITEBYTECODE"
TMPDIR_ENV = "TMPDIR"

TOOL_CONFIG_DIRNAME = "config"
TOOL_TMP_DIRNAME = "tmp"
TOOL_RUN_DIRNAME = "run"
TOOL_REPORT_FILENAME = "tool-report.json"

#: The only inherited variables a tool process may see: process plumbing.
PASSTHROUGH_ENV: tuple[str, ...] = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TERM",
    "TZ",
    "USER",
    "VIRTUAL_ENV",
)
CREDENTIAL_MARKERS: tuple[str, ...] = (
    "API_KEY",
    "APIKEY",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


class ToolContractError(RuntimeError):
    """A tool selection, environment or artifact violates the measured contract."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"perf tooling rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class MeasurementTool:
    """One profiling tool and the measurement contract a run of it honours.

    ``invocation`` is the tool's own argument shape (verified against the
    installed CLI, not inferred): the artifact placeholders are filled per run.
    ``contaminates_latency`` is True for every tool here — each one observes by
    slowing what it observes — which is why the co-run rule exists.
    """

    name: str
    module: str
    entry_point: str | None
    kind: str
    quantity: str
    unit: str
    artifact: str
    invocation: tuple[str, ...]
    provisioning: str
    notes: str
    contaminates_latency: bool = True

    def record(self) -> dict[str, Any]:
        """Return the tool's committed contract record."""
        return {
            "tool": self.name,
            "module": self.module,
            "entry_point": self.entry_point,
            "kind": self.kind,
            "quantity": self.quantity,
            "unit": self.unit,
            "artifact": self.artifact,
            "invocation": list(self.invocation),
            "observation_only": True,
            "contaminates_latency": self.contaminates_latency,
            "provisioning": self.provisioning,
            "notes": self.notes,
        }


#: The tool contracts. Each ``invocation`` was checked against the tool's own
#: ``--help`` surface on 2026-11-08 (memray 1.20.0, pyinstrument 5.1.3,
#: pytest-memray 1.11.0; the latter rides the pytest CLI, not a script).
TOOLS: tuple[MeasurementTool, ...] = (
    MeasurementTool(
        name=TOOL_MEMRAY,
        module="memray",
        entry_point="memray",
        kind=KIND_ALLOCATION_TRACER,
        quantity="peak heap allocation per allocation site of one tracked process",
        unit="bytes",
        artifact="a memray capture (.bin) under the run directory, read back "
        "with `memray stats`",
        invocation=("run", "--output", "<capture.bin>", "--force", "<script>"),
        provisioning="declared dev dependency in pyproject.toml",
        notes=(
            "every allocation is traced, so the tracked process runs far slower "
            "than the workload it began as: the capture is an allocation "
            "profile, never a latency sample. --aggregate stores only aggregate "
            "stats, --trace-python-allocators adds pymalloc records and "
            "--follow-fork extends tracking to forked children"
        ),
    ),
    MeasurementTool(
        name=TOOL_PYINSTRUMENT,
        module="pyinstrument",
        entry_point="pyinstrument",
        kind=KIND_SAMPLING_PROFILER,
        quantity="sampled wall-clock time per Python call stack",
        unit="ms of sampled time",
        artifact="a JSON profile (-r json -o <report.json>) under the run directory",
        invocation=(
            "-r",
            "json",
            "-o",
            "<report.json>",
            "-i",
            "<interval>",
            "<script>",
        ),
        provisioning=(
            "external executable, not a declared dependency: the repository's "
            "retired-dev-root guard in tests/test_dependency_declarations.py "
            "rejects the pyinstrument dependency, so a machine supplies it"
        ),
        notes=(
            "the sampler resolves stacks statistically at its -i interval "
            "(0.001 s by default) and adds per-sample overhead, so its numbers "
            "are a profile of where time goes, never a gated latency value"
        ),
    ),
    MeasurementTool(
        name=TOOL_PYTEST_MEMRAY,
        module="pytest_memray",
        entry_point=None,
        kind=KIND_ALLOCATION_PLUGIN,
        quantity="peak allocated bytes per pytest item and the session summary",
        unit="bytes",
        artifact="a memray capture per item under --memray-bin-path, plus the "
        "plugin's summary in the pytest report",
        invocation=("--memray", "--memray-bin-path=<dir>", "--most-allocations=<n>"),
        provisioning="declared dev dependency in pyproject.toml",
        notes=(
            "the plugin traces inside the pytest process it runs in; enabling "
            "--memray in a latency-gated suite would measure the tracer, which "
            "is exactly what the co-run rule forbids. limit_memory markers turn "
            "an allocation budget into a test assertion"
        ),
    ),
)

_TOOLS_BY_NAME: dict[str, MeasurementTool] = {tool.name: tool for tool in TOOLS}


def measurement_tool(name: str) -> MeasurementTool:
    """Return one tool's contract, or refuse an unknown tool name."""
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        raise ToolContractError(
            "unknown_tool", f"{name!r} is not one of {list(TOOL_NAMES)}"
        )
    return tool


def contracts_document() -> dict[str, Any]:
    """Return the committed contracts document: every tool and the run rules."""
    return {
        "schema": SCHEMA_ID,
        "tools": [tool.record() for tool in TOOLS],
        "rules": {
            "latency_co_run": LATENCY_CO_RUN_RULE,
            "absence": ABSENCE_RULE,
            "egress": EGRESS_RULE,
            "chunks_coverage": COVERAGE_NOTE,
        },
    }


@dataclass(frozen=True, slots=True)
class ToolAvailability:
    """Whether one tool can run here, and why not when it cannot."""

    tool: str
    status: str
    module_present: bool
    executable: str | None
    reason: str | None

    def record(self) -> dict[str, Any]:
        """Return the availability record; ``elevated`` is always False."""
        return {
            "tool": self.tool,
            "status": self.status,
            "module_present": self.module_present,
            "executable": self.executable,
            "reason": self.reason,
            "elevated": False,
        }


def module_installed(tool: MeasurementTool) -> bool:
    """Return whether the tool's module is importable in this environment."""
    try:
        return importlib.util.find_spec(tool.module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def tool_executable(
    tool: MeasurementTool, environ: Mapping[str, str] | None = None
) -> str | None:
    """Return the tool's console script on ``PATH``, or None."""
    if tool.entry_point is None:
        return None
    path = None if environ is None else environ.get("PATH", "")
    return shutil.which(tool.entry_point, path=path)


def tool_availability(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
    installed: bool | None = None,
) -> ToolAvailability:
    """Probe one tool's availability without ever elevating its absence."""
    tool = measurement_tool(name)
    present = module_installed(tool) if installed is None else installed
    executable = tool_executable(tool, environ)
    if not present:
        return ToolAvailability(
            tool=name,
            status=STATUS_UNAVAILABLE,
            module_present=False,
            executable=executable,
            reason=f"the {tool.module} module is not installed in this environment",
        )
    if tool.entry_point is not None and executable is None:
        return ToolAvailability(
            tool=name,
            status=STATUS_UNAVAILABLE,
            module_present=True,
            executable=None,
            reason=f"the {tool.entry_point} executable is not on PATH",
        )
    return ToolAvailability(
        tool=name,
        status=STATUS_AVAILABLE,
        module_present=True,
        executable=executable,
        reason=None,
    )


def plan_tool_run(
    names: Sequence[str],
    *,
    measures_latency: bool,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one tool run's record, refusing a latency-gated selection.

    A run that carries a latency gate may not carry any of these tools: the
    instrument would become part of the measurement (the co-run rule). An
    unavailable tool is recorded as unavailable and left out of ``runnable``.
    """
    if not names:
        raise ToolContractError("no_tools", "a tool run names at least one tool")
    selected: list[MeasurementTool] = []
    for name in names:
        tool = measurement_tool(name)
        if tool in selected:
            raise ToolContractError("duplicate_tool", f"{name!r} is selected twice")
        selected.append(tool)
    if measures_latency:
        blocking = [tool.name for tool in selected if tool.contaminates_latency]
        if blocking:
            raise ToolContractError(
                "allocation_with_latency_gate",
                f"{blocking} cannot co-run with a latency gate: {LATENCY_CO_RUN_RULE}",
            )
    availability = [tool_availability(tool.name, environ=environ) for tool in selected]
    runnable = [item.tool for item in availability if item.status == STATUS_AVAILABLE]
    unavailable = [
        item.tool for item in availability if item.status != STATUS_AVAILABLE
    ]
    return {
        "schema": SCHEMA_ID,
        "measures_latency": measures_latency,
        "tools": [tool.record() for tool in selected],
        "availability": [item.record() for item in availability],
        "runnable": runnable,
        "unavailable": unavailable,
        "elevated": False,
    }


def credential_names(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Return every variable in ``environ`` whose name is credential-shaped."""
    return tuple(
        sorted(
            name
            for name in environ
            if any(marker in name.upper() for marker in CREDENTIAL_MARKERS)
        )
    )


def prepare_tool_dirs(root: Path) -> Path:
    """Create a tool run's config, temp and run directories."""
    for directory in (
        Path(root) / TOOL_CONFIG_DIRNAME,
        Path(root) / TOOL_TMP_DIRNAME,
        Path(root) / TOOL_RUN_DIRNAME,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return Path(root)


def tool_environment(
    root: Path, *, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build the fresh environment one tool process runs under.

    The environment is built from an allowlist, so nothing credential-shaped is
    inherited, and every path it names derives from the tool run's own
    directory. The egress guard re-asserts the outcome inside the process.
    """
    source = os.environ if base is None else base
    environment = {name: source[name] for name in PASSTHROUGH_ENV if name in source}
    environment[BYTECODE_ENV] = "1"
    environment[CONFIG_HOME_ENV] = str(Path(root) / TOOL_CONFIG_DIRNAME)
    environment[TMPDIR_ENV] = str(Path(root) / TOOL_TMP_DIRNAME)
    environment[RUN_DIR_ENV] = str(Path(root) / TOOL_RUN_DIRNAME)
    environment[REPORT_ENV] = str(Path(root) / TOOL_RUN_DIRNAME / TOOL_REPORT_FILENAME)
    repository = str(repository_root())
    inherited = source.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        repository if not inherited else f"{repository}{os.pathsep}{inherited}"
    )
    return environment


def assert_tool_environment(environment: Mapping[str, str]) -> PerfPaths:
    """Validate one tool environment against the WP1 path guard.

    The environment must disable bytecode writing, carry no credential-shaped
    variable, and resolve to a run directory and report inside the boundary the
    WP1 guard enforces.
    """
    if environment.get(BYTECODE_ENV) != "1":
        raise ToolContractError(
            "bytecode_enabled", "a tool run must never write bytecode into the tree"
        )
    present = credential_names(environment)
    if present:
        raise ToolContractError("credentials_present", f"{list(present)}")
    try:
        paths = resolve_perf_paths(environment)
    except PerfPathError as error:
        raise ToolContractError("tool_path_rejected", error.detail) from error
    if paths.run_dir is None or paths.report is None:
        raise ToolContractError(
            "tool_paths_unset", "a tool run needs a run directory and a report"
        )
    return paths


def install_tool_guard(paths: PerfPaths) -> Any:
    """Install and assert the WP3b egress guard for the calling tool process."""
    from tests.perf._runtime_scenario import EgressGuard

    if paths.scratchpad_root is None:
        raise ToolContractError(
            "tool_paths_unset", "the tool run declares no scratchpad"
        )
    guard = EgressGuard(paths.scratchpad_root.parent)
    guard.install(scratch_root=paths.scratchpad_root)
    guard.assert_closed()
    return guard


def guard_evidence(guard: Any) -> dict[str, Any]:
    """Return the properties every guarded tool process reports about itself."""
    return {
        "guard": "tests.perf._runtime_scenario.EgressGuard",
        "asserted": True,
        "bytecode_disabled": os.environ.get(BYTECODE_ENV) == "1",
        "config_home": str(guard.config_home),
        "scratch_home": str(guard.scratch_home),
    }


# --------------------------------------------------------------------------
# X6b: the external hyperfine contract
# --------------------------------------------------------------------------

TOOL_HYPERFINE = "hyperfine"
HYPERFINE_SCHEMA = "chartreux.runtime_hyperfine.v1"
HYPERFINE_CLI_SCHEMA = "chartreux.runtime_hyperfine_cli.v1"

HYPERFINE_KIND = "external-process-timer"
HYPERFINE_QUANTITY = (
    "end-to-end wall time of one whole CLI process, timed externally by "
    "hyperfine from launch to exit"
)
HYPERFINE_UNIT = "s"
HYPERFINE_PEAK_QUANTITY = "peak resident memory of one measured CLI process"
HYPERFINE_PEAK_UNIT = "bytes"
HYPERFINE_ARTIFACT = (
    "a hyperfine JSON export (--export-json <results.json>) under the run directory"
)
HYPERFINE_PEAK_ARTIFACT = (
    "one JSON observation per measured CLI process under the run directory"
)
HYPERFINE_PROVISIONING = (
    "external executable, not a declared dependency: the machine supplies hyperfine"
)
HYPERFINE_INVOCATION: tuple[str, ...] = (
    "--warmup",
    "<warmup>",
    "--runs",
    "<runs>",
    "--shell=none",
    "--export-json",
    "<results.json>",
    "<cli-wrapper>",
)

HYPERFINE_LATENCY_RULE = (
    "a hyperfine cell is observation-only: it times a whole CLI process, never a "
    "turn, so it never carries a latency gate and never co-runs with an "
    "instrumented tool that would become part of the process it times"
)
OFFLINE_CONFIG_RULE = (
    "the measured CLI runs against a fabricated offline config: CHARTREUX_HOME, "
    "the scratch home and TMPDIR all point under the hyperfine run directory, no "
    "provider credential survives the environment allowlist, and the process "
    "asserts its own non-loopback socket block before the CLI is imported"
)
PEAK_OBSERVATION_RULE = (
    "CLI peak memory is a one-observation-per-process quantity: each measured "
    "process records one ru_maxrss sample and no within-run percentile is formed "
    "from it"
)

HYPERFINE_CLI_ARGV: tuple[str, ...] = ("--version",)
HYPERFINE_WARMUP = 1
HYPERFINE_RUNS = 5
HYPERFINE_MIN_RUNS = 3
HYPERFINE_MAX_RUNS = 10
HYPERFINE_COMMAND_NAME = "chartreux-cli"
HYPERFINE_EXPORT_FILENAME = "hyperfine.json"
HYPERFINE_WRAPPER_FILENAME = "cli_wrapper.py"
HYPERFINE_PEAK_GLOB = "cli-*.json"
HYPERFINE_SETTINGS_PREFIX = "HYPERFINE_"

CLI_COLOR_ENV = "NO_COLOR"
CLI_TERM = "dumb"


@dataclass(frozen=True, slots=True)
class HyperfineAvailability:
    """Whether the external hyperfine executable can run here, and why not."""

    status: str
    executable: str | None
    version: str | None
    reason: str | None

    def record(self) -> dict[str, Any]:
        """Return the availability record; ``elevated`` is always False."""
        return {
            "tool": TOOL_HYPERFINE,
            "status": self.status,
            "executable": self.executable,
            "version": self.version,
            "reason": self.reason,
            "elevated": False,
        }


def hyperfine_executable(environ: Mapping[str, str] | None = None) -> str | None:
    """Return hyperfine's executable on ``PATH``, or None."""
    path = None if environ is None else environ.get("PATH", "")
    return shutil.which(TOOL_HYPERFINE, path=path)


def hyperfine_availability(
    *,
    environ: Mapping[str, str] | None = None,
    executable: str | None = None,
    version: str | None = None,
) -> HyperfineAvailability:
    """Probe hyperfine's availability without ever elevating its absence."""
    resolved = hyperfine_executable(environ) if executable is None else executable
    if resolved is None:
        return HyperfineAvailability(
            status=STATUS_UNAVAILABLE,
            executable=None,
            version=None,
            reason=f"the {TOOL_HYPERFINE} executable is not on PATH",
        )
    return HyperfineAvailability(
        status=STATUS_AVAILABLE, executable=str(resolved), version=version, reason=None
    )


def hyperfine_contract() -> dict[str, Any]:
    """Return the committed external hyperfine contract."""
    return {
        "schema": HYPERFINE_SCHEMA,
        "tool": TOOL_HYPERFINE,
        "kind": HYPERFINE_KIND,
        "quantity": HYPERFINE_QUANTITY,
        "unit": HYPERFINE_UNIT,
        "artifact": HYPERFINE_ARTIFACT,
        "peak_quantity": HYPERFINE_PEAK_QUANTITY,
        "peak_unit": HYPERFINE_PEAK_UNIT,
        "peak_artifact": HYPERFINE_PEAK_ARTIFACT,
        "invocation": list(HYPERFINE_INVOCATION),
        "provisioning": HYPERFINE_PROVISIONING,
        "workload": {
            "kind": "cli-end-to-end",
            "argv": list(HYPERFINE_CLI_ARGV),
            "offline_config": True,
        },
        "repetitions": {
            "warmup": HYPERFINE_WARMUP,
            "runs": HYPERFINE_RUNS,
            "min_runs": HYPERFINE_MIN_RUNS,
            "max_runs": HYPERFINE_MAX_RUNS,
        },
        "observation_only": True,
        "gated": False,
        "contaminates_latency": False,
        "statistic": None,
        "rules": {
            "latency_co_run": LATENCY_CO_RUN_RULE,
            "absence": ABSENCE_RULE,
            "hyperfine_latency": HYPERFINE_LATENCY_RULE,
            "offline_config": OFFLINE_CONFIG_RULE,
            "peak_observation": PEAK_OBSERVATION_RULE,
        },
    }


def hyperfine_environment(
    root: Path, *, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build the offline environment one measured CLI process inherits.

    The environment is the tool allowlist (so nothing credential-shaped and no
    inherited ``HYPERFINE_*`` setting survives) with every path derived from the
    hyperfine run directory, plus a colourless terminal so the measured CLI does
    not emit terminal-dependent output.
    """
    environment = tool_environment(root, base=base)
    environment[CLI_COLOR_ENV] = "1"
    environment["TERM"] = CLI_TERM
    return environment


def assert_hyperfine_environment(environment: Mapping[str, str]) -> PerfPaths:
    """Validate the offline environment one measured CLI process runs under."""
    inherited = sorted(
        name for name in environment if name.startswith(HYPERFINE_SETTINGS_PREFIX)
    )
    if inherited:
        raise ToolContractError(
            "hyperfine_settings_inherited",
            f"{inherited} would configure the run from the environment: the "
            "declared invocation is the only configuration",
        )
    return assert_tool_environment(environment)


def hyperfine_wrapper_path(root: Path) -> Path:
    """Return the generated CLI wrapper's path under the hyperfine run root."""
    return Path(root) / TOOL_RUN_DIRNAME / HYPERFINE_WRAPPER_FILENAME


def _validated_cli_argv(cli_argv: Sequence[str]) -> list[str]:
    argv = list(cli_argv)
    if not argv:
        raise ToolContractError(
            "empty_cli_command", "the measured CLI command names at least one argument"
        )
    for token in argv:
        if not isinstance(token, str) or not token or "\n" in token:
            raise ToolContractError(
                "invalid_cli_argument", f"{token!r} is not a usable argument"
            )
    return argv


def _require_whitespace_free(paths: Sequence[Path], *, option: str) -> None:
    for path in paths:
        if any(character.isspace() for character in str(path)):
            raise ToolContractError(
                "command_path_whitespace",
                f"{option} splits the measured command on whitespace, so this "
                f"path is unusable: {path}",
            )


def write_cli_wrapper(
    root: Path,
    cli_argv: Sequence[str] = HYPERFINE_CLI_ARGV,
    *,
    python_executable: str | None = None,
) -> Path:
    """Write the guarded CLI wrapper hyperfine times, and return its path.

    The wrapper *is* the measured process: it installs and asserts its own
    fail-closed egress policy, runs the CLI in that same process (so its
    ``ru_maxrss`` is the CLI's peak), writes exactly one observation for itself
    and exits with the CLI's status. Its shebang names the project interpreter,
    so the measured command is a single whitespace-free token.
    """
    argv = _validated_cli_argv(cli_argv)
    path = hyperfine_wrapper_path(root)
    interpreter = sys.executable if python_executable is None else python_executable
    _require_whitespace_free([path, Path(interpreter)], option="hyperfine --shell=none")
    path.parent.mkdir(parents=True, exist_ok=True)
    source = HYPERFINE_WRAPPER_SOURCE
    for placeholder, value in (
        ("__CLI_ARGV__", json.dumps(argv)),
        ("__OUTPUT_DIR__", json.dumps(str(path.parent))),
        ("__REPO_ROOT__", json.dumps(str(repository_root()))),
    ):
        source = source.replace(placeholder, value)
    path.write_text(f"#!{interpreter}\n{source}", encoding="utf-8")
    path.chmod(0o755)
    return path


def hyperfine_command(
    wrapper: Path,
    export_json: Path,
    *,
    executable: str = TOOL_HYPERFINE,
    warmup: int = HYPERFINE_WARMUP,
    runs: int = HYPERFINE_RUNS,
) -> list[str]:
    """Build the hyperfine invocation that times the guarded CLI wrapper.

    ``--shell=none`` is hyperfine's no-shell form: the measured command is a
    single token, split on whitespace, which is why the wrapper path must stay
    whitespace-free.
    """
    if warmup < 0:
        raise ToolContractError(
            "invalid_repetitions", f"warmup must be >= 0, not {warmup}"
        )
    if runs < 1:
        raise ToolContractError("invalid_repetitions", f"runs must be >= 1, not {runs}")
    measured = Path(wrapper)
    _require_whitespace_free([measured], option="hyperfine --shell=none")
    return [
        executable,
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--shell=none",
        "--export-json",
        str(Path(export_json)),
        "--command-name",
        HYPERFINE_COMMAND_NAME,
        str(measured),
    ]


def plan_hyperfine_run(
    root: Path,
    *,
    cli_argv: Sequence[str] = HYPERFINE_CLI_ARGV,
    availability: HyperfineAvailability | None = None,
    measures_latency: bool = False,
    instrumented_tools: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    warmup: int = HYPERFINE_WARMUP,
    runs: int = HYPERFINE_RUNS,
) -> dict[str, Any]:
    """Build one hyperfine cell's record, refusing a contaminated selection.

    A latency-gated run may not carry hyperfine (it would certify a turn with a
    process-level timer), and hyperfine may not carry an instrumented tool (the
    instrument would become part of the process being timed). An unavailable
    hyperfine is recorded as unavailable and left out of ``runnable``.
    """
    argv = _validated_cli_argv(cli_argv)
    if measures_latency:
        raise ToolContractError(
            "hyperfine_with_latency_gate",
            f"a latency-gated run cannot carry hyperfine: {HYPERFINE_LATENCY_RULE}",
        )
    blocking = [measurement_tool(name).name for name in instrumented_tools]
    if blocking:
        raise ToolContractError(
            "hyperfine_with_instrumented_tool",
            f"{blocking} would instrument the process hyperfine times: "
            f"{HYPERFINE_LATENCY_RULE}",
        )
    resolved = (
        hyperfine_availability(environ=environ)
        if availability is None
        else availability
    )
    run_root = Path(root)
    run_dir = run_root / TOOL_RUN_DIRNAME
    wrapper = hyperfine_wrapper_path(run_root)
    export_json = run_dir / HYPERFINE_EXPORT_FILENAME
    command = hyperfine_command(
        wrapper,
        export_json,
        executable=resolved.executable or TOOL_HYPERFINE,
        warmup=warmup,
        runs=runs,
    )
    return {
        "schema": HYPERFINE_SCHEMA,
        "tool": TOOL_HYPERFINE,
        "kind": HYPERFINE_KIND,
        "status": resolved.status,
        "executable": resolved.executable,
        "version": resolved.version,
        "reason": resolved.reason,
        "runnable": resolved.status == STATUS_AVAILABLE,
        "elevated": False,
        "observation_only": True,
        "gated": False,
        "statistic": None,
        "measures_latency": measures_latency,
        "instrumented_tools": blocking,
        "workload": {
            "kind": "cli-end-to-end",
            "argv": argv,
            "offline_config": True,
            "environment": {
                "config_home": str(run_root / TOOL_CONFIG_DIRNAME),
                "scratch_home": str(run_root / TOOL_TMP_DIRNAME),
                "tmpdir": str(run_root / TOOL_TMP_DIRNAME),
                "run_dir": str(run_dir),
            },
        },
        "command": command,
        "measured": str(wrapper),
        "artifacts": {
            "wrapper": str(wrapper),
            "export_json": str(export_json),
            "observation_dir": str(run_dir),
            "observation_pattern": HYPERFINE_PEAK_GLOB,
        },
        "peak": {
            "quantity": HYPERFINE_PEAK_QUANTITY,
            "unit": HYPERFINE_PEAK_UNIT,
            "observations_per_process": 1,
            "statistic": None,
            "gated": False,
        },
    }


@dataclass(frozen=True, slots=True)
class HyperfineResult:
    """One measured command's external timing summary."""

    command: str
    mean_s: float
    median_s: float
    min_s: float
    max_s: float
    stddev_s: float
    times_s: tuple[float, ...]
    exit_codes: tuple[int, ...]

    def record(self) -> dict[str, Any]:
        """Return the result record; it names no gate and no percentile."""
        return {
            "command": self.command,
            "unit": HYPERFINE_UNIT,
            "mean_s": self.mean_s,
            "median_s": self.median_s,
            "min_s": self.min_s,
            "max_s": self.max_s,
            "stddev_s": self.stddev_s,
            "times_s": list(self.times_s),
            "exit_codes": list(self.exit_codes),
            "gated": False,
        }


def _hyperfine_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ToolContractError(
            "hyperfine_export_malformed", f"{label} is not a number: {value!r}"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ToolContractError(
            "hyperfine_export_malformed",
            f"{label} is not a finite non-negative number: {value!r}",
        )
    return number


def parse_hyperfine_export(document: Mapping[str, Any]) -> tuple[HyperfineResult, ...]:
    """Read hyperfine's JSON export, refusing anything that is not a sample.

    A command that failed during measurement is not a slow run: a non-zero exit
    code is refused rather than averaged into the timings, and a summary that
    lies outside its own min/max is rejected instead of trusted.
    """
    results = document.get("results")
    if not isinstance(results, list) or not results:
        raise ToolContractError(
            "hyperfine_export_malformed", "the export carries no results list"
        )
    parsed: list[HyperfineResult] = []
    for entry in results:
        if not isinstance(entry, Mapping):
            raise ToolContractError(
                "hyperfine_export_malformed", f"a result is not a mapping: {entry!r}"
            )
        command = entry.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolContractError(
                "hyperfine_export_malformed", f"a result names no command: {command!r}"
            )
        numbers = {
            name: _hyperfine_number(entry.get(name), name)
            for name in ("mean", "median", "min", "max", "stddev")
        }
        raw_times = entry.get("times")
        if not isinstance(raw_times, list) or not raw_times:
            raise ToolContractError(
                "hyperfine_export_malformed", f"{command!r} carries no measured times"
            )
        times = tuple(_hyperfine_number(value, "times") for value in raw_times)
        raw_codes = entry.get("exit_codes", [])
        if not isinstance(raw_codes, list) or not all(
            isinstance(code, int) and not isinstance(code, bool) for code in raw_codes
        ):
            raise ToolContractError(
                "hyperfine_export_malformed",
                f"{command!r} carries malformed exit codes: {raw_codes!r}",
            )
        exit_codes = tuple(int(code) for code in raw_codes)
        if exit_codes and len(exit_codes) != len(times):
            raise ToolContractError(
                "hyperfine_export_malformed",
                f"{command!r} reports {len(exit_codes)} exit codes for "
                f"{len(times)} measured runs",
            )
        failed = sorted({code for code in exit_codes if code != 0})
        if failed:
            raise ToolContractError(
                "cli_exit_failure",
                f"{command!r} exited {failed} during measurement: a failed CLI "
                "run is not a timing sample",
            )
        low = numbers["min"]
        high = numbers["max"]
        if not (low <= numbers["mean"] <= high) or not (
            low <= numbers["median"] <= high
        ):
            raise ToolContractError(
                "hyperfine_export_malformed",
                f"{command!r} summary lies outside its own min/max [{low}, {high}]",
            )
        tolerance = max(1e-9, 1e-9 * high)
        if any(time < low - tolerance or time > high + tolerance for time in times):
            raise ToolContractError(
                "hyperfine_export_malformed",
                f"{command!r} has a measured time outside its own min/max",
            )
        parsed.append(
            HyperfineResult(
                command=command,
                mean_s=numbers["mean"],
                median_s=numbers["median"],
                min_s=low,
                max_s=high,
                stddev_s=numbers["stddev"],
                times_s=times,
                exit_codes=exit_codes,
            )
        )
    return tuple(parsed)


def parse_cli_observation(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one measured process's observation; never aggregate it."""
    for name in ("p50", "p95", "percentile", "mean", "median", "statistic"):
        if name in document:
            raise ToolContractError(
                "peak_observation_aggregated",
                f"{name!r} aggregates a one-observation-per-process quantity: "
                f"{PEAK_OBSERVATION_RULE}",
            )
    if document.get("schema") != HYPERFINE_CLI_SCHEMA:
        raise ToolContractError(
            "peak_observation_malformed",
            f"unexpected observation schema {document.get('schema')!r}",
        )
    pid = document.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ToolContractError(
            "peak_observation_malformed", f"the observation names no pid: {pid!r}"
        )
    peak = document.get("peak")
    if not isinstance(peak, Mapping) or peak.get("unit") != HYPERFINE_PEAK_UNIT:
        raise ToolContractError(
            "peak_observation_malformed",
            f"the observation names no byte-valued peak: {peak!r}",
        )
    peak_bytes = peak.get("bytes")
    if (
        not isinstance(peak_bytes, int)
        or isinstance(peak_bytes, bool)
        or peak_bytes <= 0
    ):
        raise ToolContractError(
            "peak_observation_malformed",
            f"the peak is not a positive byte count: {peak_bytes!r}",
        )
    exit_code = document.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ToolContractError(
            "peak_observation_malformed", "the observation names no exit code"
        )
    if exit_code != 0:
        raise ToolContractError(
            "cli_exit_failure",
            f"the measured CLI exited {exit_code}: a failed run is no observation",
        )
    argv = document.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(token, str) for token in argv):
        raise ToolContractError(
            "peak_observation_malformed", "the observation carries a malformed argv"
        )
    egress = document.get("egress")
    if not isinstance(egress, Mapping):
        raise ToolContractError(
            "cli_egress_unasserted", "the measured process reports no egress assertion"
        )
    for flag in ("asserted", "socket_block", "bytecode_disabled"):
        if egress.get(flag) is not True:
            raise ToolContractError(
                "cli_egress_unasserted", f"the measured process did not assert {flag}"
            )
    credentials = egress.get("credentials_present")
    if not isinstance(credentials, list) or credentials:
        raise ToolContractError(
            "credentials_present",
            f"the measured process reports credentials: {credentials!r}",
        )
    repo = repository_root()
    declared = egress.get("repo_root")
    if not isinstance(declared, str) or Path(declared).resolve() != repo:
        raise ToolContractError(
            "cli_repo_root_mismatch",
            f"the measured process was not run against {repo}: {declared!r}",
        )
    homes: dict[str, str] = {}
    for name in ("config_home", "tmp_home"):
        raw = egress.get(name)
        if not isinstance(raw, str) or not raw:
            raise ToolContractError(
                "cli_egress_unasserted", f"the measured process did not report {name}"
            )
        home = Path(raw)
        resolved = home.resolve()
        if not home.is_absolute():
            raise ToolContractError(
                "cli_egress_unasserted", f"{name} is not absolute: {home}"
            )
        if resolved == repo or repo in resolved.parents or resolved in repo.parents:
            raise ToolContractError(
                "cli_egress_unasserted",
                f"{name} is not disjoint from the repository: {resolved}",
            )
        homes[name] = raw
    return {
        "schema": HYPERFINE_CLI_SCHEMA,
        "pid": pid,
        "peak_bytes": peak_bytes,
        "unit": HYPERFINE_PEAK_UNIT,
        "exit_code": exit_code,
        "argv": list(argv),
        "egress": {
            "asserted": True,
            "socket_block": True,
            "bytecode_disabled": True,
            "credentials_present": [],
            "config_home": homes["config_home"],
            "tmp_home": homes["tmp_home"],
            "repo_root": str(repo),
        },
        "gated": False,
        "statistic": None,
    }


def collect_peak_observations(run_dir: Path) -> tuple[dict[str, Any], ...]:
    """Return one validated observation per measured process under ``run_dir``.

    The samples are returned as they were observed: no percentile, mean or
    "worst process" summary is formed from them (the plan's one-observation-
    per-process rule).
    """
    files = sorted(Path(run_dir).glob(HYPERFINE_PEAK_GLOB))
    if not files:
        raise ToolContractError(
            "peak_observation_missing",
            f"no per-process CLI observation under {run_dir}",
        )
    observations: list[dict[str, Any]] = []
    for path in files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ToolContractError(
                "peak_observation_malformed",
                f"{path.name} is not readable JSON: {error}",
            ) from error
        if not isinstance(document, dict):
            raise ToolContractError(
                "peak_observation_malformed", f"{path.name} is not a JSON object"
            )
        observations.append(parse_cli_observation(document))
    return tuple(observations)


HYPERFINE_WRAPPER_SOURCE = """# Guarded CLI wrapper measured by hyperfine (X6b, generated per run).
# The measured process IS this process: the CLI runs in it, so its ru_maxrss is
# the CLI's peak and exactly one observation is written for this one process.
# No pytest fixture reaches this process, so fail-closed egress is installed and
# asserted here before the CLI is imported.
import json
import os
from pathlib import Path
import resource
import socket
import sys

SCHEMA = "chartreux.runtime_hyperfine_cli.v1"
MARKER = "hyperfine CLI egress blocked"
CREDENTIAL_MARKERS = ("API_KEY", "APIKEY", "CREDENTIAL", "PASSWORD", "SECRET", "TOKEN")
CLI_ARGV = __CLI_ARGV__
OUTPUT_DIR = Path(__OUTPUT_DIR__)
REPO_ROOT = Path(__REPO_ROOT__).resolve()
KILOBYTE = 1024


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
        raise AssertionError("credentials leaked into the CLI process: %s" % present)
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise AssertionError("bytecode writing is not disabled")
    homes = {}
    for name in ("CHARTREUX_HOME", "TMPDIR"):
        raw = os.environ.get(name)
        if not raw:
            raise AssertionError("%s is not declared" % name)
        home = Path(raw).resolve()
        if not home.is_dir():
            raise AssertionError("%s is not a directory: %s" % (name, home))
        if home == REPO_ROOT or REPO_ROOT in home.parents or home in REPO_ROOT.parents:
            raise AssertionError(
                "%s is not disjoint from the repository: %s" % (name, home)
            )
        homes[name] = home
    return homes


def run_cli():
    sys.argv = ["chartreux"] + list(CLI_ARGV)
    from chartreux.cli.entrypoint import main as cli_main

    try:
        cli_main()
    except SystemExit as error:
        code = error.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        sys.stderr.write("%s\\n" % code)
        return 1
    return 0


def peak_bytes():
    if sys.platform != "linux":
        raise AssertionError("CLI peak memory is only observed on Linux")
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * KILOBYTE


def main():
    homes = assert_environment()
    socket_block = install_guard()
    exit_code = run_cli()
    document = {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "argv": list(CLI_ARGV),
        "exit_code": exit_code,
        "peak": {"bytes": peak_bytes(), "unit": "bytes"},
        "egress": {
            "asserted": True,
            "socket_block": socket_block,
            "credentials_present": [],
            "bytecode_disabled": os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
            "config_home": str(homes["CHARTREUX_HOME"]),
            "tmp_home": str(homes["TMPDIR"]),
            "repo_root": str(REPO_ROOT),
        },
    }
    output = OUTPUT_DIR / ("cli-%d.json" % os.getpid())
    output.write_text(json.dumps(document), encoding="utf-8")
    raise SystemExit(exit_code)


main()
"""
