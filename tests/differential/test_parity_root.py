"""WP6 [6B]: no-loss root parity and the root/approval/cancellation oracles.

Every trace here runs the same root-side component twice — once with the process
boundary removed and once inside a spawned, egress-blocked child — and compares
the recorded frame streams. The component itself is the production decision
surface (permission resolution, the agent-side request broker's arbitration and
unattended expiry, the revision-checked root save); see
``tests/differential/root_oracle.py``.

What the comparison proves, per invariant 1 and 3 of the design's differential
set: no frame the root side emits is lost or reordered across the process
boundary, the topology does not change a permission, revocation, save-before-
response, or competing-writer outcome, and approval arbitration (decision,
unattended expiry, cancellation) settles identically on both sides.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chartreux.core.agent_loop._root_grants import (
    ROOT_GRANT_DECLINED_FEEDBACK,
    ROOT_GRANT_SESSION_LABEL,
)
from tests.differential import egress, ledger
from tests.differential.policy_store import read_revision
from tests.differential.replay import (
    COMPONENT_ROOT_ORACLE,
    REPLAY_ROOT,
    digest,
    in_process_component,
    normalize_frames,
    subprocess_component,
)
from tests.differential.root_oracle import (
    EXPIRY_TIMEOUT_SECONDS,
    GRANTED_DIR,
    INNER_FILE,
    OTHER_DIR,
    PROJECT_DIR,
    RootOracleDriver,
    RootOracleOp,
    RootOracleScript,
    RootOracleTurn,
    build_root_oracle_script,
)

IN_PROCESS = "in_process"
SPAWNED = "spawned"
TOPOLOGIES = (IN_PROCESS, SPAWNED)
_ORACLE_TIMEOUT = 30.0
_COMPETING_TIMEOUT = 30


def _prepare(base: Path) -> tuple[Path, Path, Path]:
    project = base / PROJECT_DIR
    for directory in (project, base / GRANTED_DIR, base / OTHER_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / INNER_FILE).write_text("scenario\n", encoding="utf-8")
    return project, base / GRANTED_DIR, base / "config.toml"


def _host(topology: str):
    if topology == IN_PROCESS:
        return in_process_component(
            component=COMPONENT_ROOT_ORACLE, timeout=_ORACLE_TIMEOUT
        )
    return subprocess_component(
        component=COMPONENT_ROOT_ORACLE, timeout=_ORACLE_TIMEOUT
    )


def _run(
    topology: str, base: Path, script: RootOracleScript
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    """Run one scripted trace through one topology and return its frame stream."""
    project, _granted, config = _prepare(base)
    with _host(topology) as channel:
        driver = RootOracleDriver(channel)
        ready = driver.initialize(base=base, project=project, config=config)
        outcome = driver.run(script)
    return (ready, *outcome.frames), outcome.texts


def _run_both(
    tmp_path: Path, script: RootOracleScript
) -> dict[str, tuple[tuple[dict[str, object], ...], tuple[str, ...]]]:
    return {
        topology: _run(topology, tmp_path / topology, script) for topology in TOPOLOGIES
    }


def _of_kind(
    frames: tuple[dict[str, object], ...], kind: str
) -> list[dict[str, object]]:
    return [frame for frame in frames if frame.get("type") == kind]


def _script_key(script: RootOracleScript) -> dict[str, object]:
    """The mode-independent script identity a ledger entry is keyed by."""
    return {
        "seed": script.seed,
        "turns": [
            {
                "prompt": turn.prompt,
                "ops": [
                    {"kind": op.kind, "subject": op.subject, "detail": op.detail}
                    for op in turn.ops
                ],
            }
            for turn in script.turns
        ],
    }


def _assert_topologies_agree(
    runs: dict[str, tuple[tuple[dict[str, object], ...], tuple[str, ...]]],
    script: RootOracleScript,
) -> tuple[dict[str, object], ...]:
    """Compare both topologies and return the (identical) normalized stream."""
    texts = {topology: runs[topology][1] for topology in TOPOLOGIES}
    assert texts[IN_PROCESS] == texts[SPAWNED]
    assert texts[IN_PROCESS] == tuple(turn.prompt for turn in script.turns)
    in_process = runs[IN_PROCESS][0]
    spawned = runs[SPAWNED][0]
    # The adapter really removed a process boundary: run identities differ.
    assert in_process[0]["run"] != spawned[0]["run"]
    normalized_in_process = normalize_frames(in_process)
    normalized_spawned = normalize_frames(spawned)
    assert normalized_in_process == normalized_spawned
    assert digest(normalized_in_process) == digest(normalized_spawned)
    # A gap, duplicate, or reordering shows up as a non-contiguous sequence.
    assert [frame["sequence"] for frame in in_process] == list(range(len(in_process)))
    assert [frame["sequence"] for frame in spawned] == list(range(len(spawned)))
    ledger.record(ledger.FAMILY_ROOT, _script_key(script), normalized_in_process)
    return normalized_in_process


@pytest.mark.timeout(120)
def test_seeded_root_trace_is_identical_across_the_process_boundary(
    tmp_path: Path, replay_seed: int
) -> None:
    script = build_root_oracle_script(replay_seed)
    normalized = _assert_topologies_agree(_run_both(tmp_path, script), script)

    kinds = {frame["type"] for frame in normalized}
    # The seeded cycle must actually exercise the [6B] scope.
    assert {
        "grant",
        "revoke",
        "permission",
        "save",
        "approval_open",
        "approval_answer",
        "approval_expiry",
        "approval_cancel",
        "approval_keep_waiting",
        "settlement",
    } <= kinds
    decisions = {
        frame.get("decision") for frame in _of_kind(normalized, "approval_answer")
    }
    assert {"session", "project", "deny"} <= decisions
    assert _of_kind(normalized, "approval_expiry")[-1]["decision"] == "deny"


@pytest.mark.timeout(120)
def test_high_volume_root_trace_loses_no_frames(
    tmp_path: Path, replay_seed: int
) -> None:
    script = build_root_oracle_script(replay_seed, turns=6, words=40)
    normalized = _assert_topologies_agree(_run_both(tmp_path, script), script)

    expected_deltas = sum(len(turn.prompt.split(" ")) for turn in script.turns)
    deltas = _of_kind(normalized, "delta")
    assert len(deltas) == expected_deltas > 200
    assert [frame["index"] for frame in deltas[: expected_deltas // 6]] == list(
        range(expected_deltas // 6)
    )
    assert len(_of_kind(normalized, "turn_end")) == len(script.turns)


@pytest.mark.timeout(120)
def test_permissions_and_revocation_match_in_both_topologies(tmp_path: Path) -> None:
    granted_file = f"{GRANTED_DIR}/{INNER_FILE}"
    script = RootOracleScript(
        seed=0,
        turns=(
            RootOracleTurn(
                "grant and read",
                (
                    RootOracleOp("permission", granted_file),
                    RootOracleOp("grant", GRANTED_DIR),
                ),
            ),
            RootOracleTurn(
                "read and revoke",
                (
                    RootOracleOp("permission", granted_file),
                    RootOracleOp("revoke", GRANTED_DIR),
                ),
            ),
            RootOracleTurn(
                "read and regrant",
                (
                    RootOracleOp("permission", granted_file),
                    RootOracleOp("grant", GRANTED_DIR),
                ),
            ),
            RootOracleTurn("read", (RootOracleOp("permission", granted_file),)),
        ),
    )
    normalized = _assert_topologies_agree(_run_both(tmp_path, script), script)

    permissions = _of_kind(normalized, "permission")
    assert [(frame["permission"], frame["denial_kind"]) for frame in permissions] == [
        ("never", "out_of_root"),
        ("always", None),
        ("never", "out_of_root"),
        ("always", None),
    ]
    grants = _of_kind(normalized, "grant")
    assert [frame["fresh"] for frame in grants] == [True, True]
    revokes = _of_kind(normalized, "revoke")
    assert [frame["authorized"] for frame in revokes] == [[]]


@pytest.mark.timeout(120)
def test_save_before_response_ordering_holds_across_the_boundary(
    tmp_path: Path,
) -> None:
    script = RootOracleScript(
        seed=0,
        turns=(
            RootOracleTurn(
                "grant for this project",
                (RootOracleOp("approval-grant-project", GRANTED_DIR),),
            ),
            RootOracleTurn("save", (RootOracleOp("save", GRANTED_DIR, "fresh"),)),
            RootOracleTurn("stale save", (RootOracleOp("save", GRANTED_DIR, "stale"),)),
        ),
    )
    normalized = _assert_topologies_agree(_run_both(tmp_path, script), script)

    answers = _of_kind(normalized, "approval_answer")
    saves = _of_kind(normalized, "save")
    notes = _of_kind(normalized, "note")
    responses = _of_kind(normalized, "response")
    assert answers[0]["decision"] == "project"
    assert answers[0]["applied"] is True
    # The session grant is applied, then persisted, then the turn answers:
    # a save failure can never un-authorize an already-approved session root.
    assert normalized.index(answers[0]) < normalized.index(saves[0])
    assert normalized.index(saves[0]) < normalized.index(notes[0])
    assert normalized.index(notes[0]) < normalized.index(responses[0])
    assert saves[0]["persistence"] == "saved" and saves[0]["error"] is None
    assert saves[1]["persistence"] == "saved" and saves[1]["error"] is None
    # The stale save is arbitrated as a conflict and still answers the turn.
    assert saves[2]["persistence"] == "not_saved"
    assert saves[2]["error"] == "conflict"
    assert "is saved in your user config" in str(notes[0]["text"])
    assert len(responses) == len(script.turns)
    assert normalized.index(saves[2]) < normalized.index(responses[2])
    authorized = answers[0]["authorized"]
    assert isinstance(authorized, list)
    assert GRANTED_DIR in authorized


@pytest.mark.timeout(120)
def test_approval_expiry_and_cancellation_match_in_both_topologies(
    tmp_path: Path,
) -> None:
    other_file = f"{OTHER_DIR}/{INNER_FILE}"
    script = RootOracleScript(
        seed=0,
        turns=(
            RootOracleTurn(
                "approve for this session",
                (
                    RootOracleOp("approval-grant-session", GRANTED_DIR),
                    RootOracleOp("permission", f"{GRANTED_DIR}/{INNER_FILE}"),
                ),
            ),
            RootOracleTurn(
                "deny",
                (
                    RootOracleOp("approval-deny", OTHER_DIR),
                    RootOracleOp("permission", other_file),
                ),
            ),
            RootOracleTurn(
                "arbitrate", (RootOracleOp("approval-duplicate", GRANTED_DIR),)
            ),
            RootOracleTurn(
                "let it expire",
                (
                    RootOracleOp("approval-expire", OTHER_DIR),
                    RootOracleOp("permission", other_file),
                ),
            ),
            RootOracleTurn(
                "keep waiting", (RootOracleOp("approval-keep-waiting", OTHER_DIR),)
            ),
            RootOracleTurn("cancel", (RootOracleOp("approval-cancel", OTHER_DIR),)),
        ),
    )
    normalized = _assert_topologies_agree(_run_both(tmp_path, script), script)

    opens = _of_kind(normalized, "approval_open")
    expired_opens = [frame for frame in opens if frame["mode"] == "expire"]
    approving_opens = [
        frame for frame in opens if frame["origin"] == "runtime_approval"
    ]
    # Runtime approvals are permanently expiry-ineligible; the defaultable
    # model question arms the configured deadline.
    assert all(
        frame["armed"] is False and frame["expiry_seconds"] is None
        for frame in approving_opens
    )
    assert expired_opens[0]["armed"] is True
    assert expired_opens[0]["expiry_seconds"] == EXPIRY_TIMEOUT_SECONDS

    answers = _of_kind(normalized, "approval_answer")
    granted = answers[0]
    assert granted["outcome"] == "accepted"
    assert granted["decision"] == "session"
    assert granted["applied"] is True
    assert granted["waiter_closed"] is True
    # Publication precedes the unblocked waiter in the recorded stream.
    settled = [
        frame
        for frame in _of_kind(normalized, "settlement")
        if frame["request"] == granted["request"]
    ]
    assert [frame["settlement_kind"] for frame in settled] == ["answered"]
    assert normalized.index(settled[0]) < normalized.index(granted)

    denied = answers[1]
    assert denied["decision"] == "deny"
    assert denied["applied"] is False
    assert denied["feedback"] == ROOT_GRANT_DECLINED_FEEDBACK

    arbitration = _of_kind(normalized, "approval_arbitration")[0]
    assert arbitration["accepted"] == "accepted"
    assert arbitration["duplicate"] == "duplicate"
    assert arbitration["conflict"] == "conflict"

    expiry = _of_kind(normalized, "approval_expiry")[0]
    assert expiry["disposition"] == "unanswered"
    assert expiry["reason"] == "unanswered_timeout"
    assert expiry["timeout_seconds"] == EXPIRY_TIMEOUT_SECONDS
    assert expiry["elapsed_seconds"] == 120.0
    assert expiry["deferred"] == 1
    # Expiry never approves anything.
    assert expiry["decision"] == "deny"
    assert _of_kind(normalized, "approval_late_answer")[0]["outcome"] == "expired"

    keep_waiting = _of_kind(normalized, "approval_keep_waiting")[0]
    assert keep_waiting["first"] == "kept"
    assert keep_waiting["answered"] == "accepted"

    cancelled = _of_kind(normalized, "approval_cancel")[0]
    assert cancelled["closed"] is True
    assert cancelled["late_answer"] == "closed"
    cancelled_settlement = [
        frame
        for frame in _of_kind(normalized, "settlement")
        if frame["request"] == cancelled["request"]
    ]
    assert [
        (frame["settlement_kind"], frame["was_open"]) for frame in cancelled_settlement
    ] == [("cancelled", True)]

    permissions = _of_kind(normalized, "permission")
    assert permissions[0]["permission"] == "always"
    # Denied and expired/cancelled roots never became authorized.
    assert all(frame["permission"] == "never" for frame in permissions[1:])
    assert ROOT_GRANT_SESSION_LABEL not in str(normalized)


def _run_competing_writer(base: Path, *, revision: str) -> dict[str, object]:
    """Persist a competing root grant from a second, egress-blocked process."""
    environment = os.environ.copy()
    environment.update(egress.child_environment(egress.MODE_ALL))
    with egress.bootstrap_directory() as guard_dir:
        entries = [str(guard_dir), str(REPLAY_ROOT)]
        inherited = environment.get("PYTHONPATH")
        if inherited:
            entries.append(inherited)
        environment["PYTHONPATH"] = os.pathsep.join(entries)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.differential.policy_writer",
                str(base / "config.toml"),
                str(base / PROJECT_DIR),
                str(base / GRANTED_DIR),
                revision,
            ],
            cwd=str(REPLAY_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=_COMPETING_TIMEOUT,
        )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.timeout(120)
def test_competing_root_writers_arbitrate_identically_in_both_topologies(
    tmp_path: Path,
) -> None:
    script = RootOracleScript(
        seed=0,
        turns=(
            RootOracleTurn(
                "stale save after competition",
                (RootOracleOp("save", GRANTED_DIR, "stale"),),
            ),
            RootOracleTurn("fresh save", (RootOracleOp("save", GRANTED_DIR, "fresh"),)),
        ),
    )
    runs: dict[str, tuple[tuple[dict[str, object], ...], tuple[str, ...]]] = {}
    competitions: dict[str, dict[str, object]] = {}
    for topology in TOPOLOGIES:
        base = tmp_path / topology
        project, _granted, config = _prepare(base)
        captured = asyncio.run(read_revision(config))
        with _host(topology) as channel:
            driver = RootOracleDriver(channel)
            ready = driver.initialize(base=base, project=project, config=config)
            competitions[topology] = _run_competing_writer(base, revision=captured)
            outcome = driver.run(script)
        runs[topology] = ((ready, *outcome.frames), outcome.texts)

    normalized = _assert_topologies_agree(runs, script)

    for topology in TOPOLOGIES:
        assert competitions[topology] == {
            "target": "user",
            "persistence": "saved",
            "application": "unchanged",
            "error": None,
        }
    saves = _of_kind(normalized, "save")
    # The competitor's revision won; the oracle's captured revision is stale,
    # and a later decision-time read succeeds again.
    assert [(frame["basis"], frame["error"]) for frame in saves] == [
        ("initial", "conflict"),
        ("current", None),
    ]
    assert saves[0]["persistence"] == "not_saved"
    assert saves[1]["persistence"] == "saved"
