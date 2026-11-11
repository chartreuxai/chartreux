"""WP6 [6C]: child crash containment and ``agents/*`` reconciliation.

The three [5C] crash points — before the start marker, after it, and after the
terminal result record — are produced by real SIGKILLed child processes. Each
trace reconciles them with the production coordinator surface and runs the same
oracle in both topologies (in-process component vs spawned, egress-blocked
child), so the recovery semantics cannot depend on the process boundary.

The oracles assert design invariant 5: a completed result without a root receipt
is surfaced as unconsumed (never re-executed), a started run with no terminal
record is outcome-unknown (never replayed), and never-executed is distinct from
both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chartreux.app_server._subagent_recovery import AGENT_TRANSCRIPT_RECONCILE_METHOD
from tests.differential import ledger
from tests.differential.crash_oracle import SCENARIOS, ChildCrashDriver
from tests.differential.replay import (
    COMPONENT_CHILD_CRASH,
    digest,
    in_process_component,
    normalize_frames,
    subprocess_component,
)

IN_PROCESS = "in_process"
SPAWNED = "spawned"
TOPOLOGIES = (IN_PROCESS, SPAWNED)
_CRASH_TIMEOUT = 120.0


def _host(topology: str):
    if topology == IN_PROCESS:
        return in_process_component(
            component=COMPONENT_CHILD_CRASH, timeout=_CRASH_TIMEOUT
        )
    return subprocess_component(component=COMPONENT_CHILD_CRASH, timeout=_CRASH_TIMEOUT)


def _both(base: Path, ops: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Run one crash trace in both topologies and return the common stream."""
    runs: dict[str, tuple[dict[str, Any], ...]] = {}
    for topology in TOPOLOGIES:
        run_base = base / topology
        run_base.mkdir(parents=True, exist_ok=True)
        with _host(topology) as channel:
            driver = ChildCrashDriver(channel)
            ready = driver.initialize({"base": str(run_base)})
            assert ready["op"] == "init"
            runs[topology] = driver.run(ops)
    normalized = {topology: normalize_frames(runs[topology]) for topology in TOPOLOGIES}
    # Each topology used its own base directory; the oracle scrubbed it.
    assert normalized[IN_PROCESS] == normalized[SPAWNED]
    assert digest(normalized[IN_PROCESS]) == digest(normalized[SPAWNED])
    ledger.record(ledger.FAMILY_CHILD_CRASH, {"ops": ops}, normalized[IN_PROCESS])
    return normalized[IN_PROCESS]


def _rows(frames: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    return {
        str(frame["crash"]["scenario"]): frame["crash"]
        for frame in frames
        if "crash" in frame
    }


@pytest.mark.timeout(300)
def test_child_crash_points_are_distinguished_and_never_replayed(
    tmp_path: Path,
) -> None:
    frames = _both(
        tmp_path, [{"kind": "crash", "scenario": scenario} for scenario in SCENARIOS]
    )
    rows = _rows(frames)
    never = rows["before_start"]
    started = rows["after_start"]
    completed = rows["after_result"]

    # Death before the start marker: never executed, nothing to replay.
    assert never["outcome"] == "never_executed"
    assert never["markers"] == {"start": False, "result": False}
    assert never["run_id"] is None and never["result_id"] is None
    assert never["unconsumed"] is False
    assert "never executed" in str(never["denial"])
    assert "Nothing was replayed" in str(never["denial"])

    # Death after the start marker: outcome unknown, never replayed.
    assert started["outcome"] == "started_outcome_unknown"
    assert started["markers"] == {"start": True, "result": False}
    assert started["run_id"] == "run-after_start"
    assert started["result_id"] is None
    assert "outcome is unknown" in str(started["denial"])
    assert "will not be replayed" in str(started["denial"])

    # Death after the terminal save: the result is durable and unconsumed.
    assert completed["outcome"] == "completed"
    assert completed["markers"] == {"start": True, "result": True}
    assert completed["run_id"] == "run-after_result"
    assert completed["result_id"] == "result-run-after_result"
    assert str(completed["digest"]).startswith("sha256:")
    assert completed["unconsumed"] is True
    assert completed["has_unconsumed_result"] is True
    assert completed["denial"] is None

    for row in (never, started, completed):
        assert row["replay_instruction"] is False
        # The injected death is real: the child did not exit on its own.
        assert row["returncode"] != 0
        assert row["coordinator"]["method"] == AGENT_TRANSCRIPT_RECONCILE_METHOD
        assert str(row["transcript_dir"]).startswith("<base>/")


@pytest.mark.timeout(300)
def test_root_resume_receipts_results_and_restarts_fresh(tmp_path: Path) -> None:
    ops: list[dict[str, Any]] = [
        {"kind": "crash", "scenario": scenario} for scenario in SCENARIOS
    ]
    ops.extend(({"kind": "rescan"}, {"kind": "restart"}))
    frames = _both(tmp_path, ops)
    rows = _rows(frames)

    rescan = frames[3]["rescan"]
    scans = rescan["scans"]
    assert [scan["outcome"] for scan in scans] == [
        "completed",
        "started_outcome_unknown",
        "never_executed",
    ]
    # The root recorded the receipt: the same result is not re-delivered.
    assert scans[0]["result_id"] == "result-run-after_result"
    assert scans[0]["unconsumed"] is False
    assert rescan["consumed_result_ids"] == ["result-run-after_result"]
    assert scans[0]["denial"] is None
    assert "outcome is unknown" in str(scans[1]["denial"])
    assert "never executed" in str(scans[2]["denial"])

    restart = frames[4]["restart"]
    fresh = restart["run"]
    completed = rows["after_result"]
    assert fresh["outcome"] == "completed"
    assert fresh["run_id"] == "run-restart"
    assert fresh["result_id"] == "result-run-restart"
    assert fresh["transcript_dir"] != completed["transcript_dir"]

    # The crashed transcripts are untouched: no resume, no re-execution.
    unchanged = {
        str(row["scenario"]): row for row in restart["crashed_transcripts_unchanged"]
    }
    assert [row["outcome"] for row in unchanged.values()] == [
        "never_executed",
        "started_outcome_unknown",
        "completed",
    ]
    assert unchanged["after_result"]["digest"] == completed["digest"]
    assert unchanged["after_result"]["result_id"] == completed["result_id"]
    assert unchanged["after_start"]["run_id"] == rows["after_start"]["run_id"]
    assert unchanged["before_start"]["run_id"] is None
    for row in unchanged.values():
        assert row["replay_instruction"] is False
        assert str(row["transcript_dir"]).startswith("<base>/")
