"""X6a/X6b contract nodes: the perf tooling contracts.

The X6a node pins the three Python tool contracts (memray, the external
pyinstrument, pytest-memray), the co-run rule that keeps allocation and sampling
instrumentation out of latency-gated runs, the honest unavailable branch for an
absent tool, the isolated environment a tool process runs under, and the WP3b
egress guard that environment is handed to. The X6b node pins the external
hyperfine contract: CLI end-to-end timing plus the one-observation-per-process
peak memory of the measured CLI, against a fabricated offline config.

Nothing here runs a real profile or a benchmark: the tools are absent from the
project environment by design (and pyinstrument is not a declared dependency),
so the nodes exercise the availability, planning, environment, parse and guard
contracts and stay inside their in-suite windows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import pytest

from tests.perf._runtime_metrics import (
    REPORT_ENV,
    RUN_DIR_ENV,
    PerfPaths,
    repository_root,
)
from tests.perf._runtime_tools import (
    BYTECODE_ENV,
    CONFIG_HOME_ENV,
    HYPERFINE_CLI_ARGV,
    HYPERFINE_CLI_SCHEMA,
    HYPERFINE_EXPORT_FILENAME,
    HYPERFINE_KIND,
    HYPERFINE_RUNS,
    HYPERFINE_SCHEMA,
    HYPERFINE_WARMUP,
    HYPERFINE_WRAPPER_FILENAME,
    LATENCY_CO_RUN_RULE,
    PASSTHROUGH_ENV,
    SCHEMA_ID,
    STATUS_AVAILABLE,
    STATUS_UNAVAILABLE,
    TOOL_HYPERFINE,
    TOOL_MEMRAY,
    TOOL_NAMES,
    TOOL_PYINSTRUMENT,
    TOOL_PYTEST_MEMRAY,
    TOOL_REPORT_FILENAME,
    TOOLS,
    ToolContractError,
    assert_hyperfine_environment,
    assert_tool_environment,
    collect_peak_observations,
    contracts_document,
    guard_evidence,
    hyperfine_availability,
    hyperfine_contract,
    hyperfine_environment,
    hyperfine_wrapper_path,
    install_tool_guard,
    measurement_tool,
    parse_cli_observation,
    parse_hyperfine_export,
    plan_hyperfine_run,
    plan_tool_run,
    prepare_tool_dirs,
    tool_availability,
    tool_environment,
    write_cli_wrapper,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="the runtime tooling contracts are Linux-only machinery",
    ),
]

# Plan §1 in-suite envelope: a 1s budget (target <= 0.5s measured) inside a 2s
# hard window, and the timeout marker is 2x the budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_TOOLS_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_FAKE_CREDENTIALS = ("MISTRAL_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_TOKEN")
_BYTECODE_BEFORE = os.environ.get(BYTECODE_ENV)


def _assert_contracts() -> None:
    document = contracts_document()
    assert document["schema"] == SCHEMA_ID
    assert document["tools"] == [tool.record() for tool in TOOLS]
    assert [record["tool"] for record in document["tools"]] == list(TOOL_NAMES)

    for record in document["tools"]:
        # Every tool states what it measures, in what unit, leaving what artifact,
        # and is observation-only: none of these feeds a gated campaign row.
        assert record["kind"]
        assert record["quantity"]
        assert record["unit"]
        assert record["artifact"]
        assert record["provisioning"]
        assert record["notes"]
        assert record["observation_only"] is True
        assert record["contaminates_latency"] is True
        assert record["invocation"]
        assert all(isinstance(token, str) and token for token in record["invocation"])

        # The declared invocation is the tool's own CLI surface, not a guess.
        invocation = record["invocation"]
        if record["tool"] == TOOL_MEMRAY:
            assert invocation[:2] == ["run", "--output"]
            assert "--force" in invocation
        elif record["tool"] == TOOL_PYINSTRUMENT:
            assert invocation[:4] == ["-r", "json", "-o", "<report.json>"]
            assert "-i" in invocation
        else:
            assert invocation[0] == "--memray"
            assert any(token.startswith("--memray-bin-path") for token in invocation)

    rules = document["rules"]
    assert rules["latency_co_run"] == LATENCY_CO_RUN_RULE
    assert "never co-runs" in rules["latency_co_run"]
    assert "never elevated" in rules["absence"]
    assert "EgressGuard" in rules["egress"]
    assert "no chunks-coverage floor" in rules["chunks_coverage"]

    with pytest.raises(ToolContractError) as unknown:
        measurement_tool("valgrind")
    assert unknown.value.reason == "unknown_tool"


def _assert_plan_rules() -> None:
    # The co-run rule is enforced, not merely documented: every tool here
    # observes by slowing what it observes, so none may carry a latency gate.
    for name in TOOL_NAMES:
        with pytest.raises(ToolContractError) as refused:
            plan_tool_run([name], measures_latency=True)
        assert refused.value.reason == "allocation_with_latency_gate"
        assert name in refused.value.detail

    # The same tool is legal in a run that gates no latency.
    plan = plan_tool_run([TOOL_MEMRAY], measures_latency=False)
    assert plan["schema"] == SCHEMA_ID
    assert plan["measures_latency"] is False
    assert plan["elevated"] is False
    assert [record["tool"] for record in plan["tools"]] == [TOOL_MEMRAY]
    recorded = [record["tool"] for record in plan["availability"]]
    assert recorded == [TOOL_MEMRAY]
    assert {record["status"] for record in plan["availability"]} <= {
        STATUS_AVAILABLE,
        STATUS_UNAVAILABLE,
    }
    # A tool that is not installed is skipped and reported, never failed.
    assert set(plan["runnable"]) | set(plan["unavailable"]) == {TOOL_MEMRAY}
    assert not set(plan["runnable"]) & set(plan["unavailable"])

    with pytest.raises(ToolContractError) as empty:
        plan_tool_run([], measures_latency=False)
    assert empty.value.reason == "no_tools"
    with pytest.raises(ToolContractError) as duplicate:
        plan_tool_run([TOOL_MEMRAY, TOOL_MEMRAY], measures_latency=False)
    assert duplicate.value.reason == "duplicate_tool"
    with pytest.raises(ToolContractError) as unknown:
        plan_tool_run(["valgrind"], measures_latency=False)
    assert unknown.value.reason == "unknown_tool"


def _assert_availability() -> None:
    for name in TOOL_NAMES:
        availability = tool_availability(name)
        record = availability.record()
        assert record["tool"] == name
        assert record["status"] in (STATUS_AVAILABLE, STATUS_UNAVAILABLE)
        # Absence is an honest branch: recorded, never elevated.
        assert record["elevated"] is False
        if availability.status == STATUS_UNAVAILABLE:
            assert availability.reason, name
        else:
            assert availability.reason is None
            assert availability.module_present is True

    # A module that is not installed is unavailable with its reason, whatever
    # PATH says; a present module whose console script is missing on PATH is
    # unavailable too, and the two reasons are distinct.
    missing_module = tool_availability(
        TOOL_MEMRAY, environ={"PATH": "/nonexistent"}, installed=False
    )
    assert missing_module.status == STATUS_UNAVAILABLE
    assert missing_module.module_present is False
    assert missing_module.reason is not None
    assert "memray module is not installed" in missing_module.reason

    missing_script = tool_availability(
        TOOL_MEMRAY, environ={"PATH": "/nonexistent"}, installed=True
    )
    assert missing_script.status == STATUS_UNAVAILABLE
    assert missing_script.module_present is True
    assert missing_script.executable is None
    assert missing_script.reason is not None
    assert "memray executable is not on PATH" in missing_script.reason

    # pytest-memray rides the pytest CLI, so it has no script of its own and its
    # availability turns only on the module.
    plugin = measurement_tool(TOOL_PYTEST_MEMRAY)
    assert plugin.entry_point is None
    plugin_absent = tool_availability(
        TOOL_PYTEST_MEMRAY, environ={"PATH": "/nonexistent"}, installed=False
    )
    assert plugin_absent.status == STATUS_UNAVAILABLE
    plugin_present = tool_availability(
        TOOL_PYTEST_MEMRAY, environ={"PATH": "/nonexistent"}, installed=True
    )
    assert plugin_present.status == STATUS_AVAILABLE
    assert plugin_present.executable is None


def _fabricated_root(tmp_path: Path, name: str) -> Path:
    """Return a tool run root inside a scratchpad-shaped fabricated root."""
    scratchpad = tmp_path / name / "chartreux-scratchpad-fabricated"
    return prepare_tool_dirs(scratchpad / "tools" / "memray")


def _assert_environment(tmp_path: Path) -> None:
    root = _fabricated_root(tmp_path, "environment")
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    base = {
        "PATH": "/usr/bin",
        "HOME": "/home/perf",
        "PYTHONPATH": "/inherited",
        "TERM": "xterm",
        **{name: "not-a-real-key" for name in _FAKE_CREDENTIALS},
        RUN_DIR_ENV: str(decoy),
        REPORT_ENV: str(decoy / "report.json"),
    }
    environment = tool_environment(root, base=base)
    assert environment[BYTECODE_ENV] == "1"
    assert environment[CONFIG_HOME_ENV] == str(root / "config")
    assert environment["TMPDIR"] == str(root / "tmp")
    assert environment[RUN_DIR_ENV] == str(root / "run")
    assert environment[REPORT_ENV] == str(root / "run" / TOOL_REPORT_FILENAME)
    # Nothing credential-shaped and nothing from the inherited run survives.
    for name in _FAKE_CREDENTIALS:
        assert name not in environment
    assert decoy.name not in environment[RUN_DIR_ENV]
    assert decoy.name not in environment[REPORT_ENV]
    # Plumbing is inherited, and the repository stays importable first.
    for name in PASSTHROUGH_ENV:
        if name in base:
            assert environment[name] == base[name]
    assert environment["PYTHONPATH"].startswith(str(repository_root()))
    assert environment["PYTHONPATH"].endswith("/inherited")

    paths = assert_tool_environment(environment)
    assert paths.run_dir == root / "run"
    assert paths.report == root / "run" / TOOL_REPORT_FILENAME

    # A relative report is refused by the WP1 guard, never accepted here.
    relative = dict(environment)
    relative[REPORT_ENV] = "tool-report.json"
    _assert_tool_env_rejected(relative, "tool_path_rejected")

    # A run directory inside the repository is refused in both directions.
    inside_repo = dict(environment)
    inside_repo[RUN_DIR_ENV] = str(repository_root())
    _assert_tool_env_rejected(inside_repo, "tool_path_rejected")

    bytecode = dict(environment)
    bytecode[BYTECODE_ENV] = "0"
    _assert_tool_env_rejected(bytecode, "bytecode_enabled")

    leaked = dict(environment)
    leaked["ANTHROPIC_API_KEY"] = "not-a-real-key"
    _assert_tool_env_rejected(leaked, "credentials_present")

    unset = {name: value for name, value in environment.items() if name != REPORT_ENV}
    _assert_tool_env_rejected(unset, "tool_paths_unset")


def _assert_tool_env_rejected(environment: dict[str, str], reason: str) -> None:
    """Assert one environment is refused with the named reason."""
    with pytest.raises(ToolContractError) as rejected:
        assert_tool_environment(environment)
    assert rejected.value.reason == reason


def _assert_guard(tmp_path: Path) -> None:
    root = _fabricated_root(tmp_path, "guard")
    paths: PerfPaths = assert_tool_environment(tool_environment(root))
    guard = install_tool_guard(paths)
    scratchpad = paths.scratchpad_root
    assert scratchpad is not None
    try:
        # The guard is installed and its own probe says the policy holds.
        guard.assert_closed()
        evidence = guard_evidence(guard)
        assert evidence["guard"] == "tests.perf._runtime_scenario.EgressGuard"
        assert evidence["asserted"] is True
        assert evidence["bytecode_disabled"] is True
        assert Path(evidence["config_home"]) == scratchpad.parent / "config"
        assert Path(evidence["scratch_home"]) == scratchpad
        assert not Path(evidence["config_home"]).is_relative_to(repository_root())
    finally:
        guard.restore()
    # Restoring is not decoration: the process gets its own bytecode policy back.
    assert os.environ.get(BYTECODE_ENV) == _BYTECODE_BEFORE


@pytest.mark.timeout(_TOOLS_TIMEOUT_S)
def test_runtime_tools_contract(tmp_path: Path) -> None:
    """The tool contracts, the co-run rule and the isolated tool environment."""
    started = perf_counter()
    _assert_contracts()
    _assert_plan_rules()
    _assert_availability()
    _assert_environment(tmp_path)
    _assert_guard(tmp_path)
    assert perf_counter() - started <= _CONTRACT_WINDOW_S


# --------------------------------------------------------------------------
# X6b: the external hyperfine contract
# --------------------------------------------------------------------------

# Plan §1 in-suite envelope: a 1s budget (target <= 0.5s measured) inside a 2s
# hard window, and the timeout marker is 2x the budget. Nothing here runs a
# benchmark: the node takes the honest-unavailable branch on a machine without
# hyperfine and exercises the plan, parse, offline-environment and refusal
# contracts on synthetic records.
_HYPERFINE_BUDGET_S = 1.0
_HYPERFINE_WINDOW_S = 2.0
_HYPERFINE_TIMEOUT_S = 2 * _HYPERFINE_BUDGET_S


def _fake_hyperfine_dir(tmp_path: Path, name: str) -> Path:
    """Return a fabricated PATH directory holding an executable hyperfine."""
    directory = tmp_path / name / "bin"
    directory.mkdir(parents=True)
    executable = directory / TOOL_HYPERFINE
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return directory


def _assert_hyperfine_contract() -> None:
    contract = hyperfine_contract()
    assert contract["schema"] == HYPERFINE_SCHEMA
    assert contract["tool"] == TOOL_HYPERFINE
    assert contract["kind"] == HYPERFINE_KIND
    # Two observation-only quantities, each with its unit and its artifact.
    assert contract["unit"] == "s"
    assert contract["peak_unit"] == "bytes"
    assert contract["quantity"]
    assert contract["peak_quantity"]
    assert contract["artifact"]
    assert contract["peak_artifact"]
    assert contract["provisioning"]
    assert contract["workload"]["kind"] == "cli-end-to-end"
    assert contract["workload"]["argv"] == list(HYPERFINE_CLI_ARGV)
    assert contract["workload"]["offline_config"] is True
    # No gate, no latency contamination, and no aggregated statistic.
    assert contract["observation_only"] is True
    assert contract["gated"] is False
    assert contract["contaminates_latency"] is False
    assert contract["statistic"] is None
    assert contract["repetitions"]["warmup"] < contract["repetitions"]["runs"]
    assert contract["repetitions"]["runs"] >= contract["repetitions"]["min_runs"]

    # The declared invocation is hyperfine's own CLI surface, not a guess.
    invocation = contract["invocation"]
    assert invocation[:2] == ["--warmup", "<warmup>"]
    assert "--runs" in invocation
    assert "--shell=none" in invocation
    assert "--export-json" in invocation

    rules = contract["rules"]
    assert rules["latency_co_run"] == LATENCY_CO_RUN_RULE
    assert "never co-runs" in rules["latency_co_run"]
    assert "never elevated" in rules["absence"]
    assert "observation-only" in rules["hyperfine_latency"]
    assert "fabricated offline config" in rules["offline_config"]
    assert "one-observation-per-process" in rules["peak_observation"]


def _assert_hyperfine_availability(tmp_path: Path) -> None:
    # The real verdict on this machine is recorded, never elevated: absent
    # hyperfine is an honest unavailable branch, not a failure.
    verdict = hyperfine_availability()
    record = verdict.record()
    assert record["tool"] == TOOL_HYPERFINE
    assert record["status"] in (STATUS_AVAILABLE, STATUS_UNAVAILABLE)
    assert record["elevated"] is False
    if verdict.status == STATUS_UNAVAILABLE:
        assert verdict.executable is None
        assert verdict.reason is not None
        assert "not on PATH" in verdict.reason
    else:
        assert verdict.executable is not None
        assert verdict.reason is None

    missing = hyperfine_availability(environ={"PATH": "/nonexistent"})
    assert missing.status == STATUS_UNAVAILABLE
    assert missing.executable is None
    assert missing.reason == "the hyperfine executable is not on PATH"

    present = hyperfine_availability(
        environ={"PATH": str(_fake_hyperfine_dir(tmp_path, "availability"))}
    )
    assert present.status == STATUS_AVAILABLE
    assert present.reason is None
    assert present.executable is not None
    assert Path(present.executable).name == TOOL_HYPERFINE
    assert present.record()["elevated"] is False


def _assert_hyperfine_plans(tmp_path: Path) -> None:
    root = prepare_tool_dirs(
        tmp_path / "plan" / "chartreux-scratchpad-fabricated" / "tools" / "hyperfine"
    )
    absent = hyperfine_availability(environ={"PATH": "/nonexistent"})
    plan = plan_hyperfine_run(root, availability=absent)
    assert plan["schema"] == HYPERFINE_SCHEMA
    assert plan["status"] == STATUS_UNAVAILABLE
    assert plan["runnable"] is False
    assert plan["elevated"] is False
    assert plan["observation_only"] is True
    assert plan["gated"] is False
    assert plan["statistic"] is None
    assert plan["reason"] == "the hyperfine executable is not on PATH"
    assert plan["workload"]["argv"] == list(HYPERFINE_CLI_ARGV)
    assert plan["workload"]["offline_config"] is True
    environment = plan["workload"]["environment"]
    assert environment["config_home"] == str(root / "config")
    assert environment["scratch_home"] == str(root / "tmp")
    assert environment["tmpdir"] == str(root / "tmp")
    assert environment["run_dir"] == str(root / "run")
    assert plan["artifacts"]["wrapper"] == str(
        root / "run" / HYPERFINE_WRAPPER_FILENAME
    )
    assert plan["artifacts"]["export_json"] == str(
        root / "run" / HYPERFINE_EXPORT_FILENAME
    )
    assert plan["measured"] == plan["artifacts"]["wrapper"]
    # One observation per process, and no statistic invented over it.
    assert plan["peak"]["observations_per_process"] == 1
    assert plan["peak"]["unit"] == "bytes"
    assert plan["peak"]["statistic"] is None
    assert plan["peak"]["gated"] is False

    # The declared command shape is complete whether or not the tool is here.
    command = plan["command"]
    assert "--shell=none" in command
    assert int(command[command.index("--warmup") + 1]) == HYPERFINE_WARMUP
    assert int(command[command.index("--runs") + 1]) == HYPERFINE_RUNS
    assert (
        command[command.index("--export-json") + 1]
        == (plan["artifacts"]["export_json"])
    )
    assert "--command-name" in command
    assert command[-1] == plan["artifacts"]["wrapper"]

    # An available tool is runnable, and the executable is the resolved one.
    fake_path = {"PATH": str(_fake_hyperfine_dir(tmp_path, "plans"))}
    available = plan_hyperfine_run(
        tmp_path / "plan-available",
        cli_argv=("--help",),
        environ=fake_path,
        warmup=0,
        runs=3,
    )
    assert available["status"] == STATUS_AVAILABLE
    assert available["runnable"] is True
    assert available["reason"] is None
    assert available["executable"] is not None
    assert available["command"][0] == available["executable"]
    assert available["workload"]["argv"] == ["--help"]
    assert int(available["command"][available["command"].index("--runs") + 1]) == 3

    # A latency gate and an instrumented tool are both refused in this cell.
    with pytest.raises(ToolContractError) as latency:
        plan_hyperfine_run(root, availability=absent, measures_latency=True)
    assert latency.value.reason == "hyperfine_with_latency_gate"
    with pytest.raises(ToolContractError) as instrumented:
        plan_hyperfine_run(root, availability=absent, instrumented_tools=[TOOL_MEMRAY])
    assert instrumented.value.reason == "hyperfine_with_instrumented_tool"
    assert TOOL_MEMRAY in instrumented.value.detail
    with pytest.raises(ToolContractError) as unknown:
        plan_hyperfine_run(root, availability=absent, instrumented_tools=["valgrind"])
    assert unknown.value.reason == "unknown_tool"
    with pytest.raises(ToolContractError) as empty:
        plan_hyperfine_run(root, availability=absent, cli_argv=())
    assert empty.value.reason == "empty_cli_command"
    with pytest.raises(ToolContractError) as bad_runs:
        plan_hyperfine_run(root, availability=absent, runs=0)
    assert bad_runs.value.reason == "invalid_repetitions"
    with pytest.raises(ToolContractError) as bad_warmup:
        plan_hyperfine_run(root, availability=absent, warmup=-1)
    assert bad_warmup.value.reason == "invalid_repetitions"


def _assert_hyperfine_environment(tmp_path: Path) -> None:
    root = prepare_tool_dirs(
        tmp_path / "env" / "chartreux-scratchpad-fabricated" / "tools" / "hyperfine"
    )
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    base = {
        "PATH": "/usr/bin",
        "HOME": "/home/perf",
        "TERM": "xterm-256color",
        "PYTHONPATH": "/inherited",
        "HYPERFINE_WARMUP": "99",
        **{name: "not-a-real-key" for name in _FAKE_CREDENTIALS},
        RUN_DIR_ENV: str(decoy),
        REPORT_ENV: str(decoy / "report.json"),
    }
    environment = hyperfine_environment(root, base=base)
    assert environment[BYTECODE_ENV] == "1"
    assert environment[CONFIG_HOME_ENV] == str(root / "config")
    assert environment["TMPDIR"] == str(root / "tmp")
    assert environment[RUN_DIR_ENV] == str(root / "run")
    assert environment["TERM"] == "dumb"
    assert environment["NO_COLOR"] == "1"
    # The offline config: no credential, no inherited hyperfine setting, no
    # inherited measurement path.
    for name in _FAKE_CREDENTIALS:
        assert name not in environment
    assert not any(name.startswith("HYPERFINE_") for name in environment)
    assert decoy.name not in environment[RUN_DIR_ENV]
    assert decoy.name not in environment[REPORT_ENV]
    assert environment["PYTHONPATH"].startswith(str(repository_root()))

    # The WP1 path guard accepts the offline environment.
    paths = assert_hyperfine_environment(environment)
    assert paths.run_dir == root / "run"
    assert paths.report == root / "run" / TOOL_REPORT_FILENAME

    inherited = dict(environment)
    inherited["HYPERFINE_WARMUP"] = "99"
    with pytest.raises(ToolContractError) as refused:
        assert_hyperfine_environment(inherited)
    assert refused.value.reason == "hyperfine_settings_inherited"

    leaked = dict(environment)
    leaked["MISTRAL_API_KEY"] = "not-a-real-key"
    _assert_tool_env_rejected(leaked, "credentials_present")

    bytecode = dict(environment)
    bytecode[BYTECODE_ENV] = "0"
    _assert_tool_env_rejected(bytecode, "bytecode_enabled")


def _assert_hyperfine_wrapper(tmp_path: Path) -> None:
    root = (
        tmp_path / "wrapper" / "chartreux-scratchpad-fabricated" / "tools" / "hyperfine"
    )
    path = write_cli_wrapper(root, ("--version",))
    assert path == hyperfine_wrapper_path(root)
    assert os.access(path, os.X_OK)
    source = path.read_text(encoding="utf-8")
    # The measured process asserts its own egress and records its own peak.
    assert source.startswith("#!")
    assert "--version" in source
    assert "install_guard" in source
    assert "192.0.2.1" in source
    assert "PYTHONDONTWRITEBYTECODE" in source
    assert "ru_maxrss" in source
    assert "chartreux.cli.entrypoint" in source
    assert str(root / "run") in source
    # Well-formed, and never executed here: only an external cell runs the CLI.
    compile(source, str(path), "exec")

    with pytest.raises(ToolContractError) as empty:
        write_cli_wrapper(root, ())
    assert empty.value.reason == "empty_cli_command"

    # --shell=none splits the measured command on whitespace, so a path that
    # would be split is refused instead of measured as a broken command.
    with pytest.raises(ToolContractError) as spaced:
        write_cli_wrapper(tmp_path / "space dir" / "hyperfine", ("--version",))
    assert spaced.value.reason == "command_path_whitespace"


def _assert_hyperfine_export() -> None:
    document = {
        "results": [
            {
                "command": "/scratch/run/cli_wrapper.py",
                "mean": 0.25,
                "stddev": 0.01,
                "median": 0.24,
                "user": 0.2,
                "system": 0.03,
                "min": 0.23,
                "max": 0.28,
                "times": [0.23, 0.24, 0.28],
                "exit_codes": [0, 0, 0],
                "parameters": {},
            }
        ]
    }
    results = parse_hyperfine_export(document)
    assert len(results) == 1
    record = results[0].record()
    assert record["command"] == "/scratch/run/cli_wrapper.py"
    assert record["unit"] == "s"
    assert record["mean_s"] == 0.25
    assert record["median_s"] == 0.24
    assert record["min_s"] == 0.23
    assert record["max_s"] == 0.28
    assert record["times_s"] == [0.23, 0.24, 0.28]
    assert record["exit_codes"] == [0, 0, 0]
    assert record["gated"] is False

    def entry(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "command": "cli",
            "mean": 1.0,
            "median": 1.0,
            "min": 1.0,
            "max": 1.0,
            "stddev": 0.0,
            "times": [1.0],
            "exit_codes": [0],
        }
        base.update(overrides)
        return base

    for payload, reason in (
        ({}, "hyperfine_export_malformed"),
        ({"results": []}, "hyperfine_export_malformed"),
        ({"results": ["cli"]}, "hyperfine_export_malformed"),
        ({"results": [entry(command="")]}, "hyperfine_export_malformed"),
        ({"results": [entry(mean="fast")]}, "hyperfine_export_malformed"),
        ({"results": [entry(mean=-1.0)]}, "hyperfine_export_malformed"),
        ({"results": [entry(times=[])]}, "hyperfine_export_malformed"),
        ({"results": [entry(exit_codes=[0, 0])]}, "hyperfine_export_malformed"),
        ({"results": [entry(exit_codes=[2])]}, "cli_exit_failure"),
        ({"results": [entry(mean=5.0)]}, "hyperfine_export_malformed"),
        ({"results": [entry(times=[1.0, 7.0])]}, "hyperfine_export_malformed"),
    ):
        with pytest.raises(ToolContractError) as rejected:
            parse_hyperfine_export(payload)
        assert rejected.value.reason == reason, payload


def _assert_peak_observations(tmp_path: Path) -> None:
    run_dir = tmp_path / "peak" / "run"
    run_dir.mkdir(parents=True)
    egress: dict[str, Any] = {
        "asserted": True,
        "socket_block": True,
        "bytecode_disabled": True,
        "credentials_present": [],
        "config_home": str(tmp_path / "peak" / "config"),
        "tmp_home": str(tmp_path / "peak" / "tmp"),
        "repo_root": str(repository_root()),
    }
    first: dict[str, Any] = {
        "schema": HYPERFINE_CLI_SCHEMA,
        "pid": 4242,
        "argv": ["--version"],
        "exit_code": 0,
        "peak": {"bytes": 12345678, "unit": "bytes"},
        "egress": egress,
    }
    second = {**first, "pid": 4343, "peak": {"bytes": 12000000, "unit": "bytes"}}
    (run_dir / "cli-4242.json").write_text(json.dumps(first), encoding="utf-8")
    (run_dir / "cli-4343.json").write_text(json.dumps(second), encoding="utf-8")

    observations = collect_peak_observations(run_dir)
    assert [record["pid"] for record in observations] == [4242, 4343]
    assert [record["peak_bytes"] for record in observations] == [12345678, 12000000]
    assert all(record["unit"] == "bytes" for record in observations)
    assert all(record["gated"] is False for record in observations)
    assert all(record["statistic"] is None for record in observations)
    # Two processes give two observations; no mean, worst or percentile.
    assert "peak_bytes" in observations[0]
    assert "mean" not in observations[0]

    aggregated = {**first, "p95": 12345678}
    with pytest.raises(ToolContractError) as refused:
        parse_cli_observation(aggregated)
    assert refused.value.reason == "peak_observation_aggregated"

    unguarded = {**first, "egress": {**egress, "socket_block": False}}
    with pytest.raises(ToolContractError) as unguarded_error:
        parse_cli_observation(unguarded)
    assert unguarded_error.value.reason == "cli_egress_unasserted"

    in_repo = {**first, "egress": {**egress, "config_home": str(repository_root())}}
    with pytest.raises(ToolContractError) as escaped:
        parse_cli_observation(in_repo)
    assert escaped.value.reason == "cli_egress_unasserted"

    broken_peak = {**first, "peak": {"bytes": 0, "unit": "bytes"}}
    with pytest.raises(ToolContractError) as malformed:
        parse_cli_observation(broken_peak)
    assert malformed.value.reason == "peak_observation_malformed"

    failed_run = {**first, "exit_code": 2}
    with pytest.raises(ToolContractError) as failed:
        parse_cli_observation(failed_run)
    assert failed.value.reason == "cli_exit_failure"

    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(ToolContractError) as missing:
        collect_peak_observations(empty)
    assert missing.value.reason == "peak_observation_missing"


@pytest.mark.timeout(_HYPERFINE_TIMEOUT_S)
def test_runtime_hyperfine_contract(tmp_path: Path) -> None:
    """The external hyperfine contract: CLI e2e, peak memory, offline config."""
    started = perf_counter()
    _assert_hyperfine_contract()
    _assert_hyperfine_availability(tmp_path)
    _assert_hyperfine_plans(tmp_path)
    _assert_hyperfine_environment(tmp_path)
    _assert_hyperfine_wrapper(tmp_path)
    _assert_hyperfine_export()
    _assert_peak_observations(tmp_path)
    assert perf_counter() - started <= _HYPERFINE_WINDOW_S
