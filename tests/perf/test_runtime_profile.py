"""WP7a/WP7b contract nodes: the runtime profiling matrix runner.

The first node pins the expanded matrix — its axes, its per-cell budgets and the
repricing rule that replaces r12's planning numbers — and drives the isolation
machinery end to end on one synthetic cell: a fresh interpreter, a purpose-built
environment, the harness egress guard installed and asserted inside the child, a
WP6-schema report adopted only because it validates, and a killed cell that
halts the matrix with its partial output discarded and no second attempt.

The second node pins the profiler readiness handshake: the cell's pid and phase
line, the parent's bounded wait, the refusal of a handshake naming another
process, and py-spy's honest ``unavailable`` branch. It drives only the
handshake — never a real profiler dump — so both nodes stay inside the in-suite
contract-node window (plan §1).

Nothing here runs the real matrix: the only cell driven is synthetic, so the
one-shot run stays an external WP7b artifact.
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
    SCRATCHPAD_ENV,
    repository_root,
)
from tests.perf._runtime_profile import (
    ADAPTERS,
    BYTECODE_ENV,
    CAMPAIGN_PRICE_S,
    CELL_CONFIG_DIRNAME,
    CELL_HALTED,
    CELL_OK,
    CELL_REPORT_FILENAME,
    CELL_RUN_DIRNAME,
    CELL_SCRATCH_DIRNAME,
    CONFIG_HOME_ENV,
    EGRESS_DIRNAME,
    HANDSHAKE_MALFORMED,
    HANDSHAKE_PID_MISMATCH,
    HANDSHAKE_PROCESS_EXITED,
    HANDSHAKE_READY,
    HANDSHAKE_TIMEOUT,
    KILL_DEADLINE_FACTOR,
    MATRIX_CELLS,
    MATRIX_CLASS_FLOOR_S,
    MATRIX_REPORT_FILENAME,
    MATRIX_SCHEMA,
    ONE_SHOT_SCHEMA,
    PLANNING_REFERENCE_S,
    PROFILE_PHASE_DONE,
    PROFILE_PHASE_FILENAME,
    PROFILE_PHASE_READY,
    PROFILE_PHASES,
    PROFILE_PID_FILENAME,
    PROFILER_ALLOWANCE_S,
    PROFILER_ATTACHED,
    PROFILER_AVAILABLE,
    PROFILER_FAILED,
    PROFILER_NOT_ATTACHED,
    PROFILER_UNAVAILABLE,
    SETUP_ALLOWANCE_S,
    SYNTHETIC_CELL,
    SYNTHETIC_SAMPLES_MS,
    MatrixError,
    MatrixOutcome,
    assert_cell_environment,
    cell_environment,
    credential_names,
    discard_cell_output,
    expanded_estimate_s,
    matrix_cell,
    matrix_manifest,
    one_shot_record,
    one_shot_timeout_s,
    parse_phase_line,
    prepare_cell_dirs,
    profiler_status,
    publish_handshake,
    pyspy_path,
    read_handshake,
    revalidation_record,
    run_cell,
    run_matrix,
    start_profiler,
    wait_for_handshake,
    write_matrix_report,
)
from tests.perf._runtime_report import (
    CELL_FRESH_HEADLESS,
    CELL_LOADED_HEADLESS,
    METRIC_VERSION,
    MODE_MEASURE,
    MODE_PROFILE,
    MODES,
    ROW_SPECS,
    SCHEMA_ID,
    validate_report,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="the runtime profile matrix is Linux-only machinery",
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_PROFILE_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

#: The handshake node is filesystem-only, so its own window is the same shape.
_HANDSHAKE_BUDGET_S = 1.0
_HANDSHAKE_TIMEOUT_S = 2 * _HANDSHAKE_BUDGET_S
_HANDSHAKE_WAIT_S = 0.5
_HANDSHAKE_MISS_S = 0.05

#: The halt case is deliberately starved of wall: the cell is killed while it is
#: still starting, which is exactly what the no-rerun rule has to survive.
_HALT_DEADLINE_S = 0.05

_FAKE_CREDENTIALS = ("MISTRAL_API_KEY", "OPENAI_API_KEY")


def _fabricated_run_root(tmp_path: Path) -> Path:
    """Return a run directory inside a scratchpad-shaped fabricated root."""
    scratchpad = tmp_path / "chartreux-scratchpad-fabricated"
    run_root = scratchpad / "matrix"
    run_root.mkdir(parents=True)
    return run_root


def _assert_matrix_manifest() -> None:
    manifest = matrix_manifest()
    assert manifest["schema"] == MATRIX_SCHEMA
    assert manifest["metric_version"] == METRIC_VERSION
    cells = [cell.record() for cell in MATRIX_CELLS]
    assert manifest["cells"] == cells

    identifiers = [record["cell_id"] for record in cells]
    assert len(set(identifiers)) == len(identifiers)
    assert {record["adapter"] for record in cells} == set(ADAPTERS)
    assert {record["mode"] for record in cells} == set(MODES)
    assert {record["workload_cell"] for record in cells} == {
        CELL_FRESH_HEADLESS,
        CELL_LOADED_HEADLESS,
    }
    # Both modes on every adapter/workload point: the expansion is the axis
    # product, not a chosen subset.
    points = {(record["workload_cell"], record["adapter"]) for record in cells}
    for point in points:
        modes = {
            record["mode"]
            for record in cells
            if (record["workload_cell"], record["adapter"]) == point
        }
        assert modes == set(MODES)

    budget = manifest["budget"]
    assert budget["per_cell_s"] == {
        record["cell_id"]: record["budget_s"] for record in cells
    }
    assert budget["total_estimate_s"] == round(
        sum(record["budget_s"] for record in cells), 3
    )
    assert budget["total_kill_deadline_s"] == round(
        sum(record["kill_deadline_s"] for record in cells), 3
    )
    assert budget["superseded_planning_reference_s"] == dict(PLANNING_REFERENCE_S)
    # The one-shot deadline is the matrix's own control, and plan §1 forbids it
    # from being shorter than the expanded estimate it prices.
    assert budget["expanded_estimate_s"] == budget["total_estimate_s"]
    assert budget["one_shot_timeout_s"] >= budget["expanded_estimate_s"]
    assert budget["one_shot_timeout_s"] == budget["total_kill_deadline_s"]
    assert "never rerun" in budget["one_shot_rule"]

    for cell in MATRIX_CELLS:
        assert matrix_cell(cell.cell_id) == cell
        # A matrix cell reports a committed WP6 row, never an invented id, and
        # its budget is the row's own §6 price plus the stated allowances.
        assert cell.campaign_id in ROW_SPECS
        assert cell.campaign_id in CAMPAIGN_PRICE_S
        expected = SETUP_ALLOWANCE_S + CAMPAIGN_PRICE_S[cell.campaign_id]
        if cell.mode == MODE_PROFILE:
            expected += PROFILER_ALLOWANCE_S
        assert cell.budget_s == expected
        assert cell.kill_deadline_s == KILL_DEADLINE_FACTOR * cell.budget_s
        # Plan §1's matrix-class timeout floor applies to every real cell.
        assert cell.kill_deadline_s >= MATRIX_CLASS_FLOOR_S

    with pytest.raises(MatrixError) as unknown:
        matrix_cell("no-such-cell")
    assert unknown.value.reason == "unknown_cell"


def _assert_cell_environment(tmp_path: Path) -> None:
    run_root = _fabricated_run_root(tmp_path / "environment")
    cell = MATRIX_CELLS[1]
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
    environment = cell_environment(cell, run_root, base=base)
    root = run_root / cell.cell_id
    assert environment[BYTECODE_ENV] == "1"
    assert environment[CONFIG_HOME_ENV] == str(root / CELL_CONFIG_DIRNAME)
    assert environment[SCRATCHPAD_ENV] == str(root / CELL_SCRATCH_DIRNAME)
    assert environment[RUN_DIR_ENV] == str(
        root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME
    )
    assert environment[REPORT_ENV] == str(
        root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME / CELL_REPORT_FILENAME
    )
    # The base environment's own measurement paths must not survive.
    assert decoy.name not in environment[RUN_DIR_ENV]
    assert decoy.name not in environment[REPORT_ENV]
    # Credentials are stripped by construction and never merely blanked.
    for name in _FAKE_CREDENTIALS:
        assert name not in environment
    assert credential_names(environment) == ()
    assert environment["PATH"] == "/usr/bin"
    assert environment["PYTHONPATH"].startswith(str(repository_root()))
    assert environment["PYTHONPATH"].endswith("/inherited")

    prepare_cell_dirs(root)
    paths = assert_cell_environment(environment)
    assert paths.run_dir == root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME
    assert paths.report == (
        root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME / CELL_REPORT_FILENAME
    )

    # A relative report is rejected by the WP1 guard, not accepted here.
    relative = dict(environment)
    relative[REPORT_ENV] = "report.json"
    with pytest.raises(MatrixError) as relative_error:
        assert_cell_environment(relative)
    assert relative_error.value.reason == "cell_path_rejected"

    escaped = dict(environment)
    escaped[RUN_DIR_ENV] = str(root / "elsewhere")
    (root / "elsewhere").mkdir()
    with pytest.raises(MatrixError) as escaped_error:
        assert_cell_environment(escaped)
    assert escaped_error.value.reason == "cell_path_rejected"

    inside_repo = dict(environment)
    inside_repo[RUN_DIR_ENV] = str(repository_root() / "tests")
    with pytest.raises(MatrixError) as repo_error:
        assert_cell_environment(inside_repo)
    assert repo_error.value.reason == "cell_path_rejected"

    bytecode = dict(environment)
    bytecode[BYTECODE_ENV] = "0"
    with pytest.raises(MatrixError) as bytecode_error:
        assert_cell_environment(bytecode)
    assert bytecode_error.value.reason == "bytecode_enabled"

    leaked = dict(environment)
    leaked["ANTHROPIC_API_KEY"] = "not-a-real-key"
    with pytest.raises(MatrixError) as leaked_error:
        assert_cell_environment(leaked)
    assert leaked_error.value.reason == "credentials_present"

    unset = {name: value for name, value in environment.items() if name != REPORT_ENV}
    with pytest.raises(MatrixError) as unset_error:
        assert_cell_environment(unset)
    assert unset_error.value.reason == "cell_paths_unset"


def _assert_halt_path(tmp_path: Path) -> None:
    run_root = _fabricated_run_root(tmp_path / "halt")
    root = prepare_cell_dirs(run_root / SYNTHETIC_CELL.cell_id)
    partial = root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME / CELL_REPORT_FILENAME
    partial.write_text('{"runs": [', encoding="utf-8")

    outcome = run_cell(SYNTHETIC_CELL, run_root=run_root, timeout_s=_HALT_DEADLINE_S)
    assert outcome.status == CELL_HALTED
    assert outcome.halt_reason == "invocation_timeout"
    # One attempt: a timeout is a halt, never a retry.
    assert outcome.attempts == 1
    assert outcome.records == ()
    assert outcome.partial_discarded is True
    assert not partial.exists()
    assert not partial.with_name(partial.name + ".partial").exists()
    # The cell was killed by its own deadline, not by something else.
    assert outcome.wall_s < SYNTHETIC_CELL.kill_deadline_s


def _assert_emitting_cell(tmp_path: Path) -> None:
    run_root = _fabricated_run_root(tmp_path / "emit")
    outcome = run_matrix(cells=(SYNTHETIC_CELL,), run_root=run_root)
    assert outcome.halted is False
    assert len(outcome.cells) == 1
    cell = outcome.cells[0]
    assert cell.status == CELL_OK, cell.output_tail
    assert cell.attempts == 1
    assert cell.exit_code == 0
    assert cell.run_count == 1
    assert cell.records and len(cell.records) == 1

    assert cell.report_path is not None
    document = json.loads(Path(cell.report_path).read_text(encoding="utf-8"))
    validate_report(document)
    assert document["schema"] == SCHEMA_ID
    assert document["metric_version"] == METRIC_VERSION
    assert len(document["runs"]) == 1
    record = document["runs"][0]
    assert record["mode"] == SYNTHETIC_CELL.mode == MODE_PROFILE
    assert record["campaign_id"] == SYNTHETIC_CELL.campaign_id
    coverage = record["chunks_coverage"]
    assert coverage["measured_ops"] == len(SYNTHETIC_SAMPLES_MS)
    assert coverage["unit_counts"] == {}
    assert coverage["complete"] is True
    assert coverage["missing"] == []
    # The synthetic observation is below every floor and says so.
    assert record["eligibility"]["eligible"] is False
    assert "ops_floor_measured" in record["eligibility"]["reasons"]

    assert document["cell"]["cell_id"] == SYNTHETIC_CELL.cell_id
    assert document["cell"]["synthetic"] is True
    egress = document["egress"]
    assert egress["asserted"] is True
    assert egress["bytecode_disabled"] is True
    assert egress["credentials_present"] == []
    emitted_root = run_root / SYNTHETIC_CELL.cell_id
    assert (
        Path(egress["run_dir"])
        == emitted_root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME
    )
    assert Path(egress["config_home"]) == emitted_root / CELL_CONFIG_DIRNAME
    assert Path(egress["scratch_home"]) == emitted_root / CELL_SCRATCH_DIRNAME
    assert not Path(egress["config_home"]).is_relative_to(repository_root())

    # The child published its readiness handshake and the parent adopted it only
    # because it named the process the runner launched (WP7b).
    assert cell.handshake is not None
    assert cell.handshake["status"] == HANDSHAKE_READY
    published = cell.handshake["handshake"]
    assert isinstance(published, dict)
    assert published["phase"] in PROFILE_PHASES
    assert published["cell_id"] == SYNTHETIC_CELL.cell_id
    assert published["mode"] == SYNTHETIC_CELL.mode
    assert isinstance(published["pid"], int)
    assert published["pid"] > 0
    emitted_run_dir = emitted_root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME
    assert (emitted_run_dir / PROFILE_PID_FILENAME).read_text().strip() == str(
        published["pid"]
    )
    phase_line = (emitted_run_dir / PROFILE_PHASE_FILENAME).read_text().strip()
    assert parse_phase_line(phase_line).pid == published["pid"]
    assert parse_phase_line(phase_line).cell_id == SYNTHETIC_CELL.cell_id
    assert parse_phase_line(phase_line).phase in PROFILE_PHASES
    # Profiling is observational: an absent py-spy is recorded, never elevated,
    # and the cell still passes.
    assert cell.profiler is not None
    assert cell.profiler["status"] in {
        PROFILER_AVAILABLE,
        PROFILER_UNAVAILABLE,
        PROFILER_ATTACHED,
        PROFILER_FAILED,
    }

    # The matrix report adopts exactly the validated records, and carries the
    # cell outcomes and the repriced manifest beside them.
    report_path = run_root / MATRIX_REPORT_FILENAME
    matrix = write_matrix_report(outcome, path=report_path)
    validate_report(matrix)
    assert _read_document(report_path) == matrix
    assert len(matrix["runs"]) == 1
    assert matrix["cells"] == [cell.record()]
    assert matrix["halted"] is False
    assert matrix["matrix"]["schema"] == MATRIX_SCHEMA
    # The runner process ran under the same egress policy as the cells.
    runner = matrix["egress"]
    assert runner["asserted"] is True
    assert runner["credentials_present"] == []
    assert runner["bytecode_disabled"] is True
    assert Path(runner["config_home"]) == (run_root / EGRESS_DIRNAME / "config")
    assert Path(runner["run_root"]) == run_root


def _read_document(path: Path) -> dict[str, Any]:
    """Read back a written report, so the writer is checked and not assumed."""
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from tests.perf._runtime_profile import main

    assert main(["--list"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == matrix_manifest()

    with pytest.raises(MatrixError) as unknown:
        main(["--cell", "no-such-cell"])
    assert unknown.value.reason == "unknown_cell"

    # A run directory inside the repository is refused before any cell runs.
    with pytest.raises(MatrixError) as refused:
        main(["--run-dir", str(repository_root() / "tests")])
    assert refused.value.reason == "run_dir_rejected"

    # A one-shot deadline shorter than the expanded estimate is refused outright,
    # before a single cell is launched (plan §1).
    with pytest.raises(MatrixError) as too_short:
        main(["--timeout", "1"])
    assert too_short.value.reason == "one_shot_timeout_below_estimate"


def _assert_one_shot_contract(tmp_path: Path) -> None:
    estimate = expanded_estimate_s()
    assert estimate == round(sum(cell.budget_s for cell in MATRIX_CELLS), 3)
    deadline = one_shot_timeout_s()
    # Plan §1: the deadline is never below the expanded estimate, and it is the
    # runner's own control over the cells it launches.
    assert deadline >= estimate
    assert deadline == round(sum(cell.kill_deadline_s for cell in MATRIX_CELLS), 3)

    measured = 120.0
    record = revalidation_record(measured, estimate)
    assert record["measured_wall_s"] == measured
    assert record["repriced_estimate_s"] == estimate
    assert record["measured_vs_repriced_ratio"] == round(measured / estimate, 3)
    assert record["within_repriced_estimate"] is (measured <= estimate)
    # Both r12 numbers are carried, and neither is silently inherited.
    assert record["r12_matrix_s"] == PLANNING_REFERENCE_S["r12_matrix_s"] == 931.0
    assert record["r12_timeout_s"] == PLANNING_REFERENCE_S["r12_timeout_s"] == 1100.0
    assert record["measured_vs_r12_matrix_ratio"] == round(measured / 931.0, 3)
    assert record["measured_vs_r12_timeout_ratio"] == round(measured / 1100.0, 3)
    assert record["within_r12_matrix"] is (measured <= 931.0)
    assert record["within_r12_timeout"] is True
    assert record["comparable"] is True
    # A halted run's wall is recorded but is never a whole-matrix revalidation.
    partial = revalidation_record(measured, estimate, completed=False)
    assert partial["comparable"] is False
    assert "halted" in partial["note"]

    # A complete run records one attempt per cell and no rerun; a halted one
    # escalates and still claims nothing it did not measure.
    complete = one_shot_record(MatrixOutcome(cells=(), halted=False), measured)
    assert complete["schema"] == ONE_SHOT_SCHEMA
    assert complete["cells_planned"] == len(MATRIX_CELLS)
    assert complete["cells_attempted"] == 0
    assert complete["attempts_per_cell"] == 1
    assert complete["rerun"] is False
    assert complete["escalation"] is None
    assert complete["profiler"]["elevated"] is False
    # Fewer cells than planned is not a whole-matrix wall, halted or not.
    assert complete["revalidation"]["comparable"] is False
    halted = one_shot_record(
        MatrixOutcome(cells=(), halted=True, halt_reason="one_shot_timeout"), 5.0
    )
    assert halted["escalation"] == {
        "action": "halt-and-escalate",
        "reason": "one_shot_timeout",
        "halted_cell": None,
        "partial_output": "discarded",
        "rerun": False,
    }
    assert halted["rerun"] is False
    assert halted["revalidation"]["comparable"] is False

    # A matrix report written with a measured wall carries that one-shot section
    # and revalidates against the repriced estimate.
    run_root = _fabricated_run_root(tmp_path / "one-shot")
    report_path = run_root / MATRIX_REPORT_FILENAME
    written = write_matrix_report(
        MatrixOutcome(cells=(), halted=False),
        path=report_path,
        measured_wall_s=measured,
    )
    assert written["one_shot"]["revalidation"]["measured_wall_s"] == measured
    assert written["one_shot"]["expanded_estimate_s"] == estimate
    assert _read_document(report_path) == written


@pytest.mark.timeout(_PROFILE_TIMEOUT_S)
def test_profile_runner_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The matrix runner prices its cells and isolates every one of them."""
    started = perf_counter()
    _assert_matrix_manifest()
    _assert_cell_environment(tmp_path)
    _assert_cli(capsys)
    _assert_one_shot_contract(tmp_path)
    _assert_halt_path(tmp_path)
    _assert_emitting_cell(tmp_path)
    assert perf_counter() - started <= _CONTRACT_WINDOW_S


def _cell_run_root(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """Return a fabricated cell's root and its run directory."""
    root = prepare_cell_dirs(
        _fabricated_run_root(tmp_path / name) / SYNTHETIC_CELL.cell_id
    )
    return root, root / CELL_SCRATCH_DIRNAME / CELL_RUN_DIRNAME


def _assert_handshake_round_trip(tmp_path: Path) -> None:
    _, run_dir = _cell_run_root(tmp_path, "round-trip")
    assert read_handshake(run_dir) is None

    published = publish_handshake(run_dir, SYNTHETIC_CELL)
    assert published.pid == os.getpid()
    assert published.phase == PROFILE_PHASE_READY
    assert published.cell_id == SYNTHETIC_CELL.cell_id
    assert published.mode == SYNTHETIC_CELL.mode

    # The pid file and the phase line are the handshake's two halves, and the
    # phase line alone names everything an attach needs.
    pid_text = (run_dir / PROFILE_PID_FILENAME).read_text(encoding="utf-8")
    phase_text = (run_dir / PROFILE_PHASE_FILENAME).read_text(encoding="utf-8")
    assert pid_text.strip() == str(os.getpid())
    assert phase_text.strip() == published.line()
    assert parse_phase_line(phase_text.strip()) == published
    assert read_handshake(run_dir) == published

    # A later phase is published in place; only the contract's phases are legal.
    assert publish_handshake(run_dir, SYNTHETIC_CELL, phase=PROFILE_PHASE_DONE)
    done = read_handshake(run_dir)
    assert done is not None
    assert done.phase == PROFILE_PHASE_DONE
    with pytest.raises(MatrixError) as unknown_phase:
        publish_handshake(run_dir, SYNTHETIC_CELL, phase="half-done")
    assert unknown_phase.value.reason == "handshake_malformed"

    # The probe is environment-driven, so an empty PATH resolves to absence.
    assert pyspy_path({"PATH": "/nonexistent"}) is None


def _assert_profiler_availability(tmp_path: Path) -> None:
    # py-spy absence is an honest unavailable branch: never an error, never an
    # elevated outcome, and never a changed cell result.
    status = profiler_status()
    assert status.status in (PROFILER_AVAILABLE, PROFILER_UNAVAILABLE)
    if status.status == PROFILER_UNAVAILABLE:
        assert pyspy_path() is None
        assert status.executable is None
        assert status.reason is not None
        assert "py-spy" in status.reason
    else:
        assert status.executable is not None
        assert pyspy_path() == status.executable
    assert status.record()["status"] == status.status

    # A measured cell never attaches a profiler, and the record says so.
    measure_cell = next(cell for cell in MATRIX_CELLS if cell.mode == MODE_MEASURE)
    decision = start_profiler(measure_cell, None, tmp_path)
    assert decision.status == PROFILER_NOT_ATTACHED
    assert decision.process is None
    assert decision.reason is not None


def _assert_handshake_waits(tmp_path: Path) -> None:
    _, run_dir = _cell_run_root(tmp_path, "waits")
    published = publish_handshake(run_dir, SYNTHETIC_CELL)

    ready = wait_for_handshake(
        run_dir, expected_pid=os.getpid(), deadline_s=_HANDSHAKE_WAIT_S
    )
    assert ready.status == HANDSHAKE_READY
    assert ready.ready is True
    assert ready.handshake == published
    assert ready.reason is None

    # A handshake naming another process is refused, never adopted: an attach
    # must target the process the runner launched.
    foreign = wait_for_handshake(
        run_dir, expected_pid=os.getpid() + 1, deadline_s=_HANDSHAKE_WAIT_S
    )
    assert foreign.status == HANDSHAKE_PID_MISMATCH
    assert foreign.ready is False
    assert foreign.reason is not None
    assert str(os.getpid()) in foreign.reason

    # A window that expires is a halt: the caller kills, discards and escalates.
    # The wait itself never retries and never invents a handshake.
    empty = tmp_path / "no-handshake"
    empty.mkdir()
    assert read_handshake(empty) is None
    expired = wait_for_handshake(
        empty, expected_pid=os.getpid(), deadline_s=_HANDSHAKE_MISS_S
    )
    assert expired.status == HANDSHAKE_TIMEOUT
    assert expired.ready is False
    assert expired.handshake is None
    assert expired.waited_s >= _HANDSHAKE_MISS_S
    assert expired.waited_s < _HANDSHAKE_WAIT_S

    # A process that is already gone is reported as such, not waited out.
    exited = wait_for_handshake(
        empty,
        expected_pid=os.getpid(),
        deadline_s=_HANDSHAKE_WAIT_S,
        is_alive=lambda: False,
    )
    assert exited.status == HANDSHAKE_PROCESS_EXITED
    assert exited.waited_s < _HANDSHAKE_WAIT_S

    # A malformed publication is refused, not repaired; so is a pid file that
    # disagrees with its own phase line.
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / PROFILE_PHASE_FILENAME).write_text(
        "phase=ready pid=not-a-pid cell=x mode=profile\n", encoding="utf-8"
    )
    (malformed / PROFILE_PID_FILENAME).write_text("1\n", encoding="utf-8")
    assert wait_for_handshake(malformed, deadline_s=_HANDSHAKE_WAIT_S).status == (
        HANDSHAKE_MALFORMED
    )

    contradictory = tmp_path / "contradictory"
    contradictory.mkdir()
    (contradictory / PROFILE_PHASE_FILENAME).write_text(
        f"phase=ready pid={os.getpid()} cell=x mode=profile\n", encoding="utf-8"
    )
    (contradictory / PROFILE_PID_FILENAME).write_text("1\n", encoding="utf-8")
    disagreeing = wait_for_handshake(contradictory, deadline_s=_HANDSHAKE_WAIT_S)
    assert disagreeing.status == HANDSHAKE_MALFORMED
    assert disagreeing.reason is not None


def _assert_discarded_handshake(tmp_path: Path) -> None:
    root, run_dir = _cell_run_root(tmp_path, "discard")
    publish_handshake(run_dir, SYNTHETIC_CELL)
    partial = run_dir / CELL_REPORT_FILENAME
    partial.write_text('{"runs": [', encoding="utf-8")

    # A killed cell's handshake names a process that never finished, so it is
    # discarded with the rest of that cell's unadoptable output.
    assert discard_cell_output(root) is True
    assert not (run_dir / PROFILE_PID_FILENAME).exists()
    assert not (run_dir / PROFILE_PHASE_FILENAME).exists()
    assert not partial.exists()
    assert read_handshake(run_dir) is None
    # Nothing left to discard is a false report, not an error.
    assert discard_cell_output(root) is False


@pytest.mark.timeout(_HANDSHAKE_TIMEOUT_S)
def test_pyspy_handshake_contract(tmp_path: Path) -> None:
    """The profiler readiness handshake, with an honest no-profiler branch."""
    started = perf_counter()
    _assert_handshake_round_trip(tmp_path)
    _assert_profiler_availability(tmp_path)
    _assert_handshake_waits(tmp_path)
    _assert_discarded_handshake(tmp_path)
    assert perf_counter() - started <= _CONTRACT_WINDOW_S
