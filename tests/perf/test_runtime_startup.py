"""X4 contract node: the cold-start (-X importtime) measurement harness.

Cold-start phases are observation-only (plan §4): one observation per process,
fully reported with its reason and never gated. This node pins the observed
path — interpreter launch, package import, config load, CLI boot, the
session/resume picker and the boot frame (design v3 :98) — the ``-X importtime``
report parser and its rankings, and the fail-closed egress policy the probe
installs for itself, because no pytest fixture reaches a subprocess.

Only ONE tiny synthetic probe process is driven in-suite; the real cold-start
cells are external, exactly as the matrix and soak cells are. The node stays
inside plan §1's in-suite envelope: a 1 s budget (measured target <= 0.5 s)
inside a 2 s hard window, with its own timeout marker at 2x the budget.
"""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter, perf_counter_ns

import pytest

from tests.perf._runtime_metrics import (
    REPORT_ENV,
    RUN_DIR_ENV,
    SCRATCHPAD_ENV,
    repository_root,
)
from tests.perf._runtime_profile import BYTECODE_ENV, CONFIG_HOME_ENV
from tests.perf._runtime_startup import (
    APP_IMPORT,
    BOOT,
    CLI_BOOT,
    CONFIG_LOAD,
    INTERPRETER_BOOT,
    NOT_DRIVEN_REASON,
    OBSERVATION_ONLY_REASON,
    ONE_OBSERVATION_RULE,
    PROBE_PHASE_NAMES,
    PROBE_TARGET,
    SESSION_PICKER,
    STAGE_EXTERNAL,
    STAGE_IMPORTTIME,
    STAGE_PROBE,
    START_TIME_ENV,
    STARTUP_PATH_PHASES,
    STARTUP_SCHEMA,
    StartupError,
    StartupPhaseObservation,
    assert_probe_environment,
    build_startup_record,
    importtime_command,
    importtime_summary,
    parse_importtime,
    phase_plan_record,
    prepare_probe_root,
    probe_environment,
    run_startup_probe,
    spawn_probe,
    write_probe,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="the cold-start probe measures a Linux process anchor",
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_STARTUP_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

#: The probe is spawned three times at most (one observation, two negative
#: controls for the child's own fail-closed assertions), each answering in a
#: few tens of milliseconds on a fresh interpreter.
_PROBE_TIMEOUT_S = 1.0
_PROBE_WALL_LIMIT_MS = 500.0

_IMPORTTIME_HEADER = "import time: self [us] | cumulative | imported package"
_IMPORTTIME_SAMPLE = "\n".join((
    _IMPORTTIME_HEADER,
    "import time:        45 |         45 |   _io",
    "import time:       120 |       1650 | chartreux",
    "import time:        60 |         60 |   chartreux.cli",
    "import time:        90 |        240 | chartreux.cli.process_start",
    "noise from another writer",
    "import time:        30 |         30 |   sitecustomize",
))
_FAKE_CREDENTIALS = ("MISTRAL_API_KEY", "OPENAI_API_KEY")


def _fabricated_probe_root(tmp_path: Path, name: str) -> Path:
    """Return a canonical probe root outside the repository tree."""
    root = tmp_path / name / "startup"
    root.mkdir(parents=True)
    return root


def _assert_phase_plan() -> None:
    plan = phase_plan_record()
    assert plan["schema"] == STARTUP_SCHEMA
    assert plan["gated"] is False
    assert plan["observation_only"] is True
    assert plan["reason"] == OBSERVATION_ONLY_REASON
    assert plan["one_observation_rule"] == ONE_OBSERVATION_RULE
    assert plan["probe_phases"] == list(PROBE_PHASE_NAMES)

    names = [spec.name for spec in STARTUP_PATH_PHASES]
    assert names == [record["phase"] for record in plan["phases"]]
    # The observed path, in order, ends in the picker and the boot frame.
    assert tuple(names) == (
        INTERPRETER_BOOT,
        APP_IMPORT,
        CONFIG_LOAD,
        CLI_BOOT,
        SESSION_PICKER,
        BOOT,
    )
    assert {SESSION_PICKER, BOOT} <= set(names)
    sources = {record["phase"]: record["source"] for record in plan["phases"]}
    assert set(sources.values()) == {STAGE_PROBE, STAGE_IMPORTTIME, STAGE_EXTERNAL}
    # Every phase is observation-only: reported, reasoned, never gated.
    assert all(record["gated"] is False for record in plan["phases"])
    assert all(record["reason"] == OBSERVATION_ONLY_REASON for record in plan["phases"])
    # Only the probe's own phases are driven in-suite; the rest are external.
    driven = [record["phase"] for record in plan["phases"] if record["driven_in_suite"]]
    assert driven == [INTERPRETER_BOOT, APP_IMPORT]


def _assert_full_path_record() -> None:
    phases = [
        StartupPhaseObservation(INTERPRETER_BOOT, 1.0, 1.0),
        StartupPhaseObservation(APP_IMPORT, 2.0, 3.0),
        StartupPhaseObservation(CONFIG_LOAD, 0.5, 3.5, synthetic=False),
        StartupPhaseObservation(CLI_BOOT, 0.7, 4.2, synthetic=False),
        StartupPhaseObservation(SESSION_PICKER, 0.3, 4.5, synthetic=False),
        StartupPhaseObservation(BOOT, 1.2, 5.7, synthetic=False),
    ]
    record = build_startup_record(phases)
    assert record["schema"] == STARTUP_SCHEMA
    assert record["gated"] is False
    assert record["one_observation_per_process"] is True
    assert record["processes"] == 1
    # An external cell reports the picker and boot phases as observations; the
    # record still never gates them.
    assert record["not_driven"] == []
    assert [phase["phase"] for phase in record["phases"]] == [
        spec.name for spec in STARTUP_PATH_PHASES
    ]
    assert [phase["synthetic"] for phase in record["phases"]] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert all(phase["gated"] is False for phase in record["phases"])

    # One observation per process per phase is the rule, and it is enforced.
    with pytest.raises(StartupError) as duplicate:
        build_startup_record([phases[0], phases[0]])
    assert duplicate.value.reason == "duplicate_phase_observation"

    with pytest.raises(StartupError) as unknown:
        build_startup_record([StartupPhaseObservation("no-such-phase", 1.0)])
    assert unknown.value.reason == "unknown_phase"

    with pytest.raises(StartupError) as negative:
        build_startup_record([StartupPhaseObservation(INTERPRETER_BOOT, -1.0)])
    assert negative.value.reason == "phase_wall_invalid"

    # Undriven path phases are reported with their recorded reason, never
    # invented, and the probe's own phases carry the observation-only reason.
    partial = build_startup_record([
        StartupPhaseObservation(INTERPRETER_BOOT, 1.0, 1.0)
    ])
    assert {entry["phase"] for entry in partial["not_driven"]} == {
        APP_IMPORT,
        CONFIG_LOAD,
        CLI_BOOT,
        SESSION_PICKER,
        BOOT,
    }
    assert all(entry["reason"] == NOT_DRIVEN_REASON for entry in partial["not_driven"])
    assert all(entry["gated"] is False for entry in partial["not_driven"])


def _assert_importtime() -> None:
    parsed = parse_importtime(_IMPORTTIME_SAMPLE)
    assert [entry.module for entry in parsed.entries] == [
        "_io",
        "chartreux",
        "chartreux.cli",
        "chartreux.cli.process_start",
        "sitecustomize",
    ]
    # Indentation is 2 * depth + 1 spaces in CPython's report.
    assert [entry.depth for entry in parsed.entries] == [1, 0, 1, 0, 1]
    assert [entry.self_us for entry in parsed.entries] == [45, 120, 60, 90, 30]
    assert [entry.cumulative_us for entry in parsed.entries] == [45, 1650, 60, 240, 30]
    # The header and a foreign line are preserved, never parsed as entries.
    assert parsed.ignored_lines == (_IMPORTTIME_HEADER, "noise from another writer")

    summary = importtime_summary(parsed, target=PROBE_TARGET)
    assert summary["entry_count"] == len(parsed.entries) == 5
    assert summary["ignored_line_count"] == len(parsed.ignored_lines) == 2
    assert summary["total_self_us"] == 345
    # Top-level cumulative costs carry their whole subtree.
    assert summary["total_cumulative_us"] == 1650 + 240
    assert summary["max_depth"] == 1
    assert [entry["module"] for entry in summary["top_self"]][0] == "chartreux"
    assert summary["top_cumulative"][0]["module"] == "chartreux"
    assert summary["top_cumulative"][0]["cumulative_us"] == 1650
    assert summary["target"] == {
        "module": PROBE_TARGET,
        "occurrences": 1,
        "max_self_us": 120,
        "max_cumulative_us": 1650,
    }
    assert importtime_summary(parsed, limit=2)["top_cumulative"] == [
        {"module": "chartreux", "self_us": 120, "cumulative_us": 1650, "depth": 0},
        {
            "module": "chartreux.cli.process_start",
            "self_us": 90,
            "cumulative_us": 240,
            "depth": 0,
        },
    ]
    # A module with no entry is absent, not fabricated.
    assert importtime_summary(parsed, target="chartreux.engine")["target"] is None
    with pytest.raises(StartupError) as bad_limit:
        importtime_summary(parsed, limit=0)
    assert bad_limit.value.reason == "importtime_limit_invalid"

    # A repeated module is an import attempt seen twice; the largest cumulative
    # cost is the successful one and the report says how many it saw.
    repeated = parse_importtime(
        "import time:        10 |         10 | chartreux\n"
        "import time:        90 |        900 | chartreux\n"
    )
    assert importtime_summary(repeated, target=PROBE_TARGET)["target"] == {
        "module": PROBE_TARGET,
        "occurrences": 2,
        "max_self_us": 90,
        "max_cumulative_us": 900,
    }

    # An empty report is an empty parse, not an error; a malformed import-time
    # line is an error, not a skipped line.
    empty = parse_importtime("")
    assert empty.entries == ()
    assert importtime_summary(empty)["top_self"] == []
    with pytest.raises(StartupError) as malformed:
        parse_importtime("import time: 12 | 34")
    assert malformed.value.reason == "importtime_malformed"
    with pytest.raises(StartupError) as bad_indent:
        parse_importtime("import time: 12 | 34 |    even-indent")
    assert bad_indent.value.reason == "importtime_malformed"

    command = importtime_command(PROBE_TARGET)
    assert command == [
        sys.executable,
        "-X",
        "importtime",
        "-c",
        f"import {PROBE_TARGET}",
    ]
    assert importtime_command("chartreux.cli", executable="/usr/bin/python3") == [
        "/usr/bin/python3",
        "-X",
        "importtime",
        "-c",
        "import chartreux.cli",
    ]
    for invalid in ("chartreux; import os", "chartreux.cli-x", " chartreux", ""):
        with pytest.raises(StartupError) as target_error:
            importtime_command(invalid)
        assert target_error.value.reason == "importtime_target_invalid"


def _assert_probe_environment(tmp_path: Path) -> None:
    root = _fabricated_probe_root(tmp_path, "environment")
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
    environment = probe_environment(root, base=base)
    assert environment[BYTECODE_ENV] == "1"
    assert environment[CONFIG_HOME_ENV] == str(root / "config")
    assert environment[SCRATCHPAD_ENV] == str(root / "scratch")
    # A cold-start probe carries no measurement path of its own, and the base
    # environment's decoy paths do not survive either.
    assert RUN_DIR_ENV not in environment
    assert REPORT_ENV not in environment
    assert all(str(decoy) not in value for value in environment.values())
    # Credentials are stripped by construction, never merely blanked.
    for name in _FAKE_CREDENTIALS:
        assert name not in environment
    assert environment["PATH"] == "/usr/bin"
    assert environment["PYTHONPATH"].startswith(str(repository_root()))
    assert environment["PYTHONPATH"].endswith("/inherited")

    prepare_probe_root(root)
    paths = assert_probe_environment(environment)
    assert paths.config_home == (root / "config").resolve()
    assert paths.scratch_home == (root / "scratch").resolve()
    assert paths.repo_root == repository_root()
    assert not paths.config_home.is_relative_to(repository_root())

    def rejected(mutate: dict[str, str], reason: str) -> None:
        broken = dict(environment)
        broken.update(mutate)
        with pytest.raises(StartupError) as error:
            assert_probe_environment(broken)
        assert error.value.reason == reason

    rejected({BYTECODE_ENV: "0"}, "bytecode_enabled")
    rejected({"ANTHROPIC_API_KEY": "not-a-real-key"}, "credentials_present")
    rejected({"CHARTREUX_STARTUP_REPO_ROOT": "/tmp"}, "repo_root_mismatch")
    rejected({RUN_DIR_ENV: str(decoy)}, "probe_paths_present")
    rejected(
        {CONFIG_HOME_ENV: str(repository_root() / "tests")},
        "config_home_inside_repository",
    )
    rejected({CONFIG_HOME_ENV: "config"}, "config_home_relative")
    rejected({CONFIG_HOME_ENV: str(root / "nowhere")}, "config_home_missing")
    rejected(
        {CONFIG_HOME_ENV: str(root / "config" / ".." / "config")},
        "config_home_not_canonical",
    )
    rejected({SCRATCHPAD_ENV: str(repository_root())}, "scratch_home_inside_repository")


def _assert_probe_root_rejected() -> None:
    with pytest.raises(StartupError) as inside:
        run_startup_probe(repository_root() / "tests")
    assert inside.value.reason == "probe_root_under_repo_root"
    with pytest.raises(StartupError) as relative:
        run_startup_probe(Path("startup-probe"))
    assert relative.value.reason == "probe_root_relative"


def _assert_probe(tmp_path: Path) -> None:
    root = _fabricated_probe_root(tmp_path, "probe")
    result = run_startup_probe(root, timeout_s=_PROBE_TIMEOUT_S)
    process = result.process
    assert process.returncode == 0
    assert process.wall_ms > 0.0
    assert process.wall_ms < _PROBE_WALL_LIMIT_MS
    assert "-X" in process.command
    assert "importtime" in process.command

    # The probe's own phases, in order, observed once per process.
    assert [observation.phase for observation in result.phases] == list(
        PROBE_PHASE_NAMES
    )
    assert len({observation.phase for observation in result.phases}) == len(
        result.phases
    )
    boot, import_phase, ready = result.phases
    assert boot.wall_ms > 0.0
    assert boot.cumulative_ms is not None and boot.cumulative_ms >= boot.wall_ms
    assert import_phase.wall_ms >= 0.0
    assert import_phase.cumulative_ms is not None
    assert ready.wall_ms == 0.0

    # -X importtime is parsed out of the child's own stderr, and the target
    # package is in it.
    assert result.importtime.entries
    assert result.target_summary["module"] == PROBE_TARGET
    assert result.target_summary["max_cumulative_us"] > 0

    # The child proved its own fail-closed egress policy.
    egress = result.egress
    assert egress["asserted"] is True
    assert egress["socket_block"] is True
    assert egress["credentials_present"] == []
    assert egress["bytecode_disabled"] is True
    assert Path(egress["config_home"]) == (root / "config").resolve()
    assert Path(egress["scratch_home"]) == (root / "scratch").resolve()
    assert Path(egress["repo_root"]) == repository_root()
    assert not Path(egress["config_home"]).is_relative_to(repository_root())
    assert not list(root.rglob("*.pyc"))

    # The production anchor lands inside the measured import phase.
    anchor = result.anchor
    assert anchor["monotonic_s"] > 0.0
    phase_start = import_phase.cumulative_ms or 0.0
    assert phase_start <= anchor["delta_ms"] <= phase_start + import_phase.wall_ms + 1.0

    record = result.record()
    assert record["schema"] == STARTUP_SCHEMA
    assert record["gated"] is False
    assert record["observation_only"] is True
    assert record["reason"] == OBSERVATION_ONLY_REASON
    assert record["one_observation_per_process"] is True
    assert record["processes"] == 1
    assert [phase["phase"] for phase in record["phases"]] == list(PROBE_PHASE_NAMES)
    assert all(phase["gated"] is False for phase in record["phases"])
    assert all(phase["reason"] == OBSERVATION_ONLY_REASON for phase in record["phases"])
    assert {entry["phase"] for entry in record["not_driven"]} == {
        CONFIG_LOAD,
        CLI_BOOT,
        SESSION_PICKER,
        BOOT,
    }
    assert all(entry["reason"] == NOT_DRIVEN_REASON for entry in record["not_driven"])
    assert record["importtime"]["target"]["module"] == PROBE_TARGET
    assert record["probe"]["returncode"] == 0


def _assert_child_guard(tmp_path: Path) -> None:
    root = _fabricated_probe_root(tmp_path, "guard")
    prepare_probe_root(root)
    probe_path = write_probe(root)
    environment = probe_environment(root)
    environment[START_TIME_ENV] = str(perf_counter_ns())

    first = spawn_probe(probe_path, environment, timeout_s=_PROBE_TIMEOUT_S)
    assert first.returncode == 0

    # The child's own assertions are the policy that a bare-script invocation
    # relies on, so each one is proven to fire: a probe that could write
    # bytecode or see a credential fails rather than degrading.
    bytecode = dict(environment)
    bytecode[BYTECODE_ENV] = "0"
    failed = spawn_probe(probe_path, bytecode, timeout_s=_PROBE_TIMEOUT_S)
    assert failed.returncode != 0
    assert "bytecode writing is not disabled" in failed.stderr

    leaked = dict(environment)
    leaked["MISTRAL_API_KEY"] = "not-a-real-key"
    credential_failure = spawn_probe(probe_path, leaked, timeout_s=_PROBE_TIMEOUT_S)
    assert credential_failure.returncode != 0
    assert "provider credentials leaked" in credential_failure.stderr


@pytest.mark.timeout(_STARTUP_TIMEOUT_S)
def test_runtime_startup_contract(tmp_path: Path) -> None:
    """The cold-start harness reports import cost and startup phases, ungated."""
    started = perf_counter()
    _assert_phase_plan()
    _assert_full_path_record()
    _assert_importtime()
    _assert_probe_root_rejected()
    _assert_probe_environment(tmp_path)
    _assert_probe(tmp_path)
    _assert_child_guard(tmp_path)
    assert perf_counter() - started <= _CONTRACT_WINDOW_S
