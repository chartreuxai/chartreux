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
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any, cast

import httpx
import pytest
import tomli_w

from chartreux.app_server._runtime import HarnessProcess, create_harness_server
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._session_backend_port import SessionBackendError
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.host import AppServerHost
from chartreux.app_server.mcp_catalog import AgentLoopMCPCatalogRoot
from chartreux.app_server.models import (
    PublicTurnStatus,
    SessionTextContentBlock,
    TurnUserInputEntry,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    MCPCatalogMutationResponse,
    MCPReadParams,
    MCPReadResponse,
    MCPRefreshParams,
    MCPToggleParams,
    ProtocolErrorCode,
    RootRecoveryAdmitParams,
    RootRecoveryAdmitResponse,
    RootRecoveryCancelParams,
    RootRecoveryCancelResponse,
    RootRecoveryScheduleParams,
    RootRecoveryScheduleResponse,
    RuntimeReadParams,
    RuntimeReadResponse,
    SessionOptions,
    SessionResumeParams,
    SessionResumeResponse,
    TurnEnqueueParams,
)
from chartreux.app_server.root_agent_process import RootFailureSignature
from chartreux.app_server.session import AppServerSession, AppServerTurnError
from chartreux.app_server.transport import IN_PROCESS_TRANSPORT, memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._root_grants import (
    ROOT_GRANT_DECLINED_FEEDBACK,
    ROOT_GRANT_SESSION_LABEL,
)
from chartreux.core.config import build_default_orchestrator
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.model_catalog.availability import AvailabilityRegistry
from tests.app_server.backend_contract.conftest import (
    BackendContractConnection,
    connect_backend_contract_stdio_host,
    install_memory_root_agent,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.differential import egress, ledger, regressions
from tests.differential.policy_store import read_revision
from tests.differential.replay import (
    COMPONENT_ROOT_ORACLE,
    REPLAY_ROOT,
    TRANSPORT_MODES,
    digest,
    in_process_component,
    normalize_frames,
    subprocess_component,
    transport_mode,
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
from tests.mock.utils import mock_llm_chunk
from tests.recovery.conftest import FakeClock
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

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


# ---------------------------------------------------------------------------
# C1g: the recovery/root/{schedule,admit,cancel} wire carriers — the added
# §6.3 concurrent-recovery-admission case, run on both transport selections.
# ---------------------------------------------------------------------------

_STDIO_MODE = "stdio"
_MODES = TRANSPORT_MODES
_COOLDOWN_ADVANCE_SECONDS = 40.0


def _recovery_timeout_fault() -> BackendError:
    """A streaming timeout fault with its httpx cause (classifies TIMEOUT)."""
    fault = BackendError(
        provider="test-first",
        endpoint="https://first.invalid/chat",
        status=None,
        reason="read timed out",
        headers=None,
        body_text=None,
        parsed_error="Network error",
        model="first",
        payload_summary=PayloadSummary(
            model="first",
            message_count=1,
            approx_chars=2,
            temperature=0.0,
            has_tools=False,
            tool_choice=None,
        ),
    )
    fault.__cause__ = httpx.ReadTimeout("read timed out")
    return fault


async def _recovery_carrier_session(
    mode: str, monkeypatch: pytest.MonkeyPatch, project: Path
) -> tuple[BackendContractConnection, AppServerSession, Any, FakeClock]:
    """One app-server session on ``mode`` whose root loop fails its first turn.

    The stdio leg runs the root agent over a memory wire (the spawned-stdio
    legs land with C5c's PTY gate); the in_process leg keeps its harness
    process reachable so the session's loop can be addressed owner-side. The
    shared availability registry runs on a manual clock so the failure-armed
    deployment cooldown can be advanced past deterministically before the
    admission recheck.
    """
    capabilities = ClientCapabilities(callback_kinds=["user_input"])
    options = SessionOptions(cwd=str(project))
    clock = FakeClock()
    if mode == _STDIO_MODE:
        loop = build_test_agent_loop(
            cwd=project,
            # No gateway retry budget: the scripted timeout is terminal on
            # the first attempt, so the turn fails without real backoff.
            config=build_test_vibe_config(api_retry_max_elapsed_time=0.0),
            backend=FakeBackend(exception_to_raise=_recovery_timeout_fault()),
            enable_streaming=True,
        )
        loop.config_orchestrator._availability_registry = AvailabilityRegistry(
            clock=clock
        )
        install_memory_root_agent(monkeypatch, loop)
        connection = await connect_backend_contract_stdio_host(
            session_options=options, capabilities=capabilities
        )
        session = await connection.host.open_session()
        return connection, session, loop, clock
    process = HarnessProcess()
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(
        server_transport, transport_kind=IN_PROCESS_TRANSPORT, process=process
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    host = await AppServerHost.connect(
        client,
        client_info=ClientInfo(name="parity-root-recovery", version="0"),
        capabilities=capabilities,
        session_options=options,
        client_factory=harness.connect_client,
    )
    connection = BackendContractConnection(client=client, host=host)
    session = await host.open_session()
    loop = next(
        candidate
        for candidate in process._accounting_loops
        if candidate.session_id == session.session_id
    )
    loop._backend_lifetime.replace(
        FakeBackend(exception_to_raise=_recovery_timeout_fault()),
        whole_turn_active=False,
    )
    # No gateway retry budget (the harness cwd is untrusted, so a project
    # config file would not load): the scripted timeout is terminal on the
    # first attempt, so the turn fails without real backoff.
    orchestrator = loop.config_orchestrator
    monkeypatch.setattr(
        orchestrator,
        "_snapshot",
        replace(
            orchestrator._snapshot,
            config=orchestrator.config.model_copy(
                update={"api_retry_max_elapsed_time": 0.0}
            ),
        ),
    )
    orchestrator._availability_registry = AvailabilityRegistry(clock=clock)
    return connection, session, loop, clock


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_concurrent_recovery_admission_single_flights_identically_in_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The added §6.3 concurrent-recovery-admission case: the outcome set per mode.

    One terminally failed root turn (a streaming timeout) must schedule on
    both transports — on stdio that proves the typed terminal projection
    crossed the agent wire on the turn/ended frame and the loop-owned
    evidence bundle crossed on ``agent/recovery/evidence`` — and two
    CONCURRENT admissions carrying the same idempotency key must
    single-flight to exactly one "admitted" plus one "already_admitted",
    never two admissions and never an auto-replayed request. An unknown
    episode cancels to ``cancelled=False`` on both. The trace records through
    the C1k regression registry: the checked-in baseline is asserted, and
    ``VIBE_TEST_DIFFERENTIAL_REFRESH=1`` rewrites it.
    """
    script = {"case": regressions.CONCURRENT_RECOVERY, "modes": list(_MODES)}
    frames: list[dict[str, object]] = []
    outcomes: dict[str, tuple[object, ...]] = {}
    for mode in _MODES:
        with transport_mode(mode):
            project = tmp_path / f"recovery-project-{mode}"
            project.mkdir(parents=True, exist_ok=True)
            connection, session, loop, clock = await _recovery_carrier_session(
                mode, monkeypatch, project
            )
            try:
                client = connection.client
                session_id = session.session_id

                # One terminally failed root turn; the terminal turn id it
                # names is the schedule's evidence key on both transports.
                with pytest.raises(AppServerTurnError) as turn_error:
                    async for _event in session.act("hi"):
                        pass
                terminal_turn_id = turn_error.value.turn_id
                assert terminal_turn_id is not None
                latest = session.state.latest_turn
                assert latest is not None and latest.id == terminal_turn_id
                assert latest.status is PublicTurnStatus.FAILED

                schedule = RootRecoveryScheduleResponse.model_validate(
                    await client.request(
                        "recovery/root/schedule",
                        RootRecoveryScheduleParams(
                            session_id=session_id,
                            terminal_turn_id=terminal_turn_id,
                            solicited=True,
                            input_generation=0,
                        ),
                    )
                )
                assert schedule.decision == "scheduled", schedule.model_dump(
                    mode="json"
                )
                assert schedule.episode_id is not None
                assert schedule.idempotency_key is not None

                # Advance past the failure-armed deployment cooldown before
                # the admission recheck (the controller's timer does this).
                clock.advance(_COOLDOWN_ADVANCE_SECONDS)

                admit_params = RootRecoveryAdmitParams(
                    session_id=session_id,
                    episode_id=schedule.episode_id,
                    idempotency_key=schedule.idempotency_key,
                    terminal_turn_id=terminal_turn_id,
                    input_generation=0,
                )
                raw_admits = await asyncio.gather(*[
                    client.request("recovery/root/admit", admit_params)
                    for _ in range(2)
                ])
                admits = [
                    RootRecoveryAdmitResponse.model_validate(raw) for raw in raw_admits
                ]
                admit_decisions = tuple(sorted(admit.decision for admit in admits))
                assert admit_decisions == ("admitted", "already_admitted")

                cancel_unknown = RootRecoveryCancelResponse.model_validate(
                    await client.request(
                        "recovery/root/cancel",
                        RootRecoveryCancelParams(
                            session_id=session_id,
                            episode_id="no-such-episode",
                            reason="differential-node",
                        ),
                    )
                )
                assert cancel_unknown.cancelled is False

                # The exact eligible class may differ by capture layer (the
                # exception envelope crosses one more boundary on stdio);
                # the invariant is that BOTH topologies classified the
                # failure as transient-eligible.
                outcomes[mode] = (
                    schedule.decision,
                    schedule.reason,
                    schedule.attempt,
                    schedule.limit,
                    schedule.eligible_class is not None,
                    admit_decisions,
                    tuple(sorted(admit.attempt for admit in admits)),
                    cancel_unknown.cancelled,
                )
                frames.extend([
                    {
                        "mode": mode,
                        "op": "schedule",
                        "decision": schedule.decision,
                        "reason": schedule.reason,
                        "attempt": schedule.attempt,
                        "limit": schedule.limit,
                        "eligible_class": schedule.eligible_class,
                        "affected_deployment": schedule.affected_deployment,
                    },
                    {
                        "mode": mode,
                        "op": "concurrent-admit",
                        "decisions": list(admit_decisions),
                        "attempts": sorted(admit.attempt for admit in admits),
                    },
                    {
                        "mode": mode,
                        "op": "cancel-unknown-episode",
                        "cancelled": cancel_unknown.cancelled,
                    },
                ])
            finally:
                await session.close()
                await connection.host.close()
                if mode == _STDIO_MODE:
                    await loop.aclose()

    assert outcomes[_STDIO_MODE] == outcomes[IN_PROCESS]
    spec = regressions.case(regressions.CONCURRENT_RECOVERY)
    regressions.record_case(spec, script, frames)


_MCP_OAUTH_SERVER = "oauth"


def _write_mcp_catalog_config(config_dir: Path) -> None:
    """One disabled OAuth server in the user config both modes load."""
    config_path = config_dir / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config["mcp_servers"] = [
        {
            "name": _MCP_OAUTH_SERVER,
            "transport": "streamable-http",
            "url": "http://127.0.0.1:9/mcp",
            "auth": {"type": "oauth", "scopes": []},
            "disabled": True,
        }
    ]
    config_path.write_text(tomli_w.dumps(config), encoding="utf-8")


async def _mcp_carrier_session(
    mode: str, monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> tuple[BackendContractConnection, AppServerSession, Any]:
    """One app-server session on ``mode`` owning one disabled OAuth server.

    The stdio leg runs the root agent over a memory wire (the spawned-stdio
    legs land with C5c's PTY gate) with a loop built on a real orchestrator
    over the config dir; the in_process leg loads the same config dir through
    its harness process. Both legs therefore own the same catalog shape and
    answer ``mcp_catalog/*`` from their own owner.
    """
    _write_mcp_catalog_config(config_dir)
    capabilities = ClientCapabilities()
    options = SessionOptions()
    if mode == _STDIO_MODE:
        loop = AgentLoop(
            config_orchestrator=await build_default_orchestrator(),
            backend=FakeBackend(),
            enable_streaming=False,
            mcp_registry=FakeMCPRegistry(),
        )
        install_memory_root_agent(monkeypatch, loop)
        connection = await connect_backend_contract_stdio_host(
            session_options=options, capabilities=capabilities
        )
        session = await connection.host.open_session()
        return connection, session, loop
    process = HarnessProcess()
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(
        server_transport, transport_kind=IN_PROCESS_TRANSPORT, process=process
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    host = await AppServerHost.connect(
        client,
        client_info=ClientInfo(name="parity-mcp", version="0"),
        capabilities=capabilities,
        session_options=options,
        client_factory=harness.connect_client,
    )
    connection = BackendContractConnection(client=client, host=host)
    session = await host.open_session()
    return connection, session, process


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_mcp_partial_convergence_is_identical_in_both_modes(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    """The added §6.3 mcp-partial-convergence case: the outcome set per mode.

    A fenced owner-side convergence failure during ``mcp_catalog/refresh``
    must leave the same partial public state on both transports -- on stdio
    that proves the catalog projection and the convergence-error receipt
    crossed the agent wire with the typed owner key -- and enabling the OAuth
    server afterwards must gate the authRequired handshake owner-side and
    project ``needs_auth`` identically on both legs. The trace records through
    the C1k regression registry: the checked-in baseline is asserted, and
    ``VIBE_TEST_DIFFERENTIAL_REFRESH=1`` rewrites it.
    """
    script = {"case": regressions.MCP_PARTIAL_CONVERGENCE, "modes": list(_MODES)}
    frames: list[dict[str, object]] = []
    outcomes: dict[str, tuple[object, ...]] = {}
    for mode in _MODES:
        with transport_mode(mode):
            connection, session, owner = await _mcp_carrier_session(
                mode, monkeypatch, config_dir
            )
            try:
                client = connection.client
                session_id = session.session_id

                read = MCPReadResponse.model_validate(
                    await client.request(
                        "mcp_catalog/read", MCPReadParams(session_id=session_id)
                    )
                )
                initial = dict(read.mcp.statuses)

                # Fence the owner-side convergence leg: the refresh must fail
                # typed on both transports, never silently half-applied.
                control_type = (
                    AgentLoopMCPCatalogRoot
                    if mode == _STDIO_MODE
                    else SessionBackendImpl
                )
                original_reconfigure = control_type.reconfigure_mcp

                async def fail_convergence(*args: object, **kwargs: object) -> None:
                    del args, kwargs
                    raise SessionBackendError(
                        ProtocolErrorCode.INTERNAL_ERROR,
                        "injected backend convergence failure",
                    )

                monkeypatch.setattr(control_type, "reconfigure_mcp", fail_convergence)
                with pytest.raises(AppServerResponseError) as exc_info:
                    await client.request(
                        "mcp_catalog/refresh", MCPRefreshParams(session_id=session_id)
                    )
                error_code = exc_info.value.error.code
                monkeypatch.setattr(
                    control_type, "reconfigure_mcp", original_reconfigure
                )

                partial = MCPReadResponse.model_validate(
                    await client.request(
                        "mcp_catalog/read", MCPReadParams(session_id=session_id)
                    )
                )
                partial_statuses = dict(partial.mcp.statuses)
                partial_errors = {
                    name: "did not converge" in error
                    for name, error in partial.mcp.discovery_errors.items()
                }

                # Enable the OAuth server: the owner-side gate marks the source
                # needs_auth and the relayed handshake projects identically.
                toggled = MCPCatalogMutationResponse.model_validate(
                    await client.request(
                        "mcp_catalog/toggle",
                        MCPToggleParams(
                            session_id=session_id,
                            name=_MCP_OAUTH_SERVER,
                            disabled=False,
                        ),
                    )
                )
                assert toggled.runtime is not None
                enabled_statuses = dict(toggled.runtime.mcp.statuses)

                outcomes[mode] = (
                    tuple(sorted(initial.items())),
                    error_code,
                    tuple(sorted(partial_statuses.items())),
                    tuple(sorted(partial_errors.items())),
                    tuple(sorted(enabled_statuses.items())),
                )
                frames.extend([
                    {
                        "mode": mode,
                        "op": "read-initial",
                        "statuses": dict(sorted(initial.items())),
                    },
                    {
                        "mode": mode,
                        "op": "refresh-fenced",
                        "error_code": str(error_code),
                    },
                    {
                        "mode": mode,
                        "op": "read-partial",
                        "statuses": dict(sorted(partial_statuses.items())),
                        "convergence_errors": {
                            name: flag for name, flag in sorted(partial_errors.items())
                        },
                    },
                    {
                        "mode": mode,
                        "op": "toggle-oauth-enabled",
                        "statuses": dict(sorted(enabled_statuses.items())),
                    },
                ])
            finally:
                await session.close()
                await connection.host.close()
                if mode == _STDIO_MODE:
                    await owner.aclose()
                else:
                    await owner.close()

    assert outcomes[_STDIO_MODE] == outcomes[IN_PROCESS]
    spec = regressions.case(regressions.MCP_PARTIAL_CONVERGENCE)
    regressions.record_case(spec, script, frames)


# ---------------------------------------------------------------------------
# C2a: runtime/read's fresh agent-side snapshot carrier and the shared
# agent-sequence cursor — the added §6.3 stale-snapshot case, run on both
# transport selections.
# ---------------------------------------------------------------------------


async def _runtime_read_carrier_session(
    mode: str, monkeypatch: pytest.MonkeyPatch, project: Path
) -> tuple[BackendContractConnection, AppServerSession, Any, Any, Any]:
    """One app-server session on ``mode`` whose root can be declared dead.

    The stdio leg runs the root agent over a memory wire (the spawned-stdio
    legs land with C5c's PTY gate) and keeps the harness's backend handle so
    the node can assert the adoption cursor and declare the root death; the
    in_process leg keeps its harness process reachable so the session's loop
    can be addressed owner-side. Returns the connection, the session, the
    loop, the stdio backend (``None`` on the in_process leg), and the
    owner to close.
    """
    capabilities = ClientCapabilities()
    options = SessionOptions(cwd=str(project))
    if mode == _STDIO_MODE:
        loop = build_test_agent_loop(
            cwd=project, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        install_memory_root_agent(monkeypatch, loop)
        client_transport, server_transport = memory_transport_pair()
        harness = await create_harness_server(server_transport, transport_kind="stdio")
        client = AppServerClient(client_transport, run_peer=harness.serve)
        host = await AppServerHost.connect(
            client,
            client_info=ClientInfo(name="parity-stale-snapshot", version="0"),
            capabilities=capabilities,
            session_options=options,
            client_factory=harness.connect_client,
        )
        connection = BackendContractConnection(client=client, host=host)
        session = await host.open_session()
        backend = cast(Any, harness._server._root)
        return connection, session, loop, backend, loop
    process = HarnessProcess()
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(
        server_transport, transport_kind=IN_PROCESS_TRANSPORT, process=process
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    host = await AppServerHost.connect(
        client,
        client_info=ClientInfo(name="parity-stale-snapshot", version="0"),
        capabilities=capabilities,
        session_options=options,
        client_factory=harness.connect_client,
    )
    connection = BackendContractConnection(client=client, host=host)
    session = await host.open_session()
    loop = next(
        candidate
        for candidate in process._accounting_loops
        if candidate.session_id == session.session_id
    )
    # The harness loop otherwise owns a real backend; the turn this node runs
    # is answered by the fake one, exactly as the C1g node's in_process leg.
    # The attempt path resolves the backend through the injected-backend
    # getter (``backend_for_attempt``), so the fake must be injected there;
    # the lifetime replace keeps the borrow/``loop.backend`` views consistent.
    fake_backend = FakeBackend([[mock_llm_chunk(content="ok")]])
    loop._backend_lifetime.replace(fake_backend, whole_turn_active=False)
    loop._injected_backend = fake_backend
    return connection, session, loop, None, process


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_stale_snapshot_read_is_fresh_and_monotone_in_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The added §6.3 stale-snapshot case: the outcome set per mode.

    ``runtime/read`` must answer with a FRESH agent-side snapshot on every
    read on both transports — on stdio that proves the read crossed the agent
    wire as ``agent/session/snapshot`` instead of serving the coordinator's
    cached open-time copy — and the read is adopted with monotone ordering
    against the events already relayed (the shared agent-sequence cursor, keyed
    to the agent's own event sequence, supersedes the coordinator watermark
    for that ordering). After a declared root death the stdio read answers
    the death receipt, never a stale ``ready=True`` from the dead generation.
    The trace records through the C1k regression registry: the checked-in
    baseline is asserted, and ``VIBE_TEST_DIFFERENTIAL_REFRESH=1`` rewrites
    it.
    """
    script = {"case": regressions.STALE_SNAPSHOT, "modes": list(_MODES)}
    frames: list[dict[str, object]] = []
    outcomes: dict[str, tuple[object, ...]] = {}
    for mode in _MODES:
        with transport_mode(mode):
            project = tmp_path / f"stale-project-{mode}"
            project.mkdir(parents=True, exist_ok=True)
            (
                connection,
                session,
                loop,
                backend,
                owner,
            ) = await _runtime_read_carrier_session(mode, monkeypatch, project)
            try:
                client = connection.client
                session_id = session.session_id

                read_before = RuntimeReadResponse.model_validate(
                    await client.request(
                        "runtime/read", RuntimeReadParams(session_id=session_id)
                    )
                )

                # One turn relays agent events, each stamped with the shared
                # agent-sequence cursor.
                async for _event in session.act("hi"):
                    pass

                read_after = RuntimeReadResponse.model_validate(
                    await client.request(
                        "runtime/read", RuntimeReadParams(session_id=session_id)
                    )
                )
                # Fresh per read: the post-turn read reflects the turn the
                # open-time cached snapshot could not.
                fresh = read_after.runtime.stats.steps > read_before.runtime.stats.steps

                if mode == _STDIO_MODE:
                    assert backend is not None
                    # Monotone adoption: the adopted cursor is never older
                    # than the events already relayed.
                    relayed = backend.process.last_agent_sequence
                    monotone = (
                        backend._adopted_agent_sequence
                        >= backend.process.last_agent_sequence
                    )
                    # After a declared root death the read answers the death
                    # receipt, never a stale ready=True snapshot.
                    backend.process._declare_death(
                        "the root agent process exited",
                        signature=RootFailureSignature.DEATH_RECEIPT,
                    )
                    with pytest.raises(AppServerResponseError) as exc_info:
                        await client.request(
                            "runtime/read", RuntimeReadParams(session_id=session_id)
                        )
                    death_receipt = "death-receipt" in str(exc_info.value)
                    death_code = str(exc_info.value.error.code)
                    outcomes[mode] = (
                        read_before.ready,
                        read_after.ready,
                        fresh,
                        relayed > 0,
                        monotone,
                        death_receipt,
                        death_code,
                    )
                    frames.extend([
                        {
                            "mode": mode,
                            "op": "read-before-turn",
                            "ready": read_before.ready,
                        },
                        {
                            "mode": mode,
                            "op": "read-after-turn",
                            "ready": read_after.ready,
                            "fresh": fresh,
                        },
                        {
                            "mode": mode,
                            "op": "adoption-cursor",
                            "relayed_events": relayed > 0,
                            "monotone": monotone,
                        },
                        {
                            "mode": mode,
                            "op": "read-after-death",
                            "death_receipt": death_receipt,
                            "code": death_code,
                        },
                    ])
                else:
                    outcomes[mode] = (read_before.ready, read_after.ready, fresh)
                    frames.extend([
                        {
                            "mode": mode,
                            "op": "read-before-turn",
                            "ready": read_before.ready,
                        },
                        {
                            "mode": mode,
                            "op": "read-after-turn",
                            "ready": read_after.ready,
                            "fresh": fresh,
                        },
                    ])
            finally:
                await session.close()
                await connection.host.close()
                if mode == _STDIO_MODE:
                    await owner.aclose()
                else:
                    await owner.close()

    # The shared read behavior is identical on both transports: after the turn
    # the read is ready and fresh (steps advanced past the pre-turn read) on
    # both. The pre-turn ready flag is the one known per-transport nuance: the
    # stdio root is live from open, while the in_process loop's deferred init
    # has not completed before its first turn. The death leg is the stdio
    # root's own lifecycle receipt.
    assert outcomes[_STDIO_MODE][1:3] == outcomes[IN_PROCESS][1:3] == (True, True)
    assert outcomes[_STDIO_MODE][0] is True
    assert outcomes[IN_PROCESS][0] is False
    assert outcomes[_STDIO_MODE][3:] == (True, True, True, "internal_error")
    spec = regressions.case(regressions.STALE_SNAPSHOT)
    regressions.record_case(spec, script, frames)


# ---------------------------------------------------------------------------
# C2b: the death/resume/memorial safety unit — the added §6.3 root-death and
# resume cases, run on both transport selections. The stdio legs run the
# root agent over a memory wire (the spawned-stdio legs land with C5c's PTY
# gate); the in_process legs run the harness process in-process.
# ---------------------------------------------------------------------------


async def _c2b_parity_session(
    mode: str, monkeypatch: pytest.MonkeyPatch, project: Path
) -> tuple[BackendContractConnection, AppServerSession, Any, Any, Any, Any]:
    """One app-server session on ``mode`` whose root can be declared dead.

    Same shape as :func:`_runtime_read_carrier_session` but also returns the
    stdio harness, so the resume node can assert the replacement generation
    the server activated. Returns the connection, the session, the loop, the
    stdio backend (``None`` on the in_process leg), the owner to close, and
    the harness (``None`` on the in_process leg).
    """
    capabilities = ClientCapabilities()
    options = SessionOptions(cwd=str(project))
    if mode == _STDIO_MODE:
        loop = build_test_agent_loop(
            cwd=project,
            backend=FakeBackend([[mock_llm_chunk(content="ok")] for _ in range(4)]),
        )
        install_memory_root_agent(monkeypatch, loop)
        client_transport, server_transport = memory_transport_pair()
        harness = await create_harness_server(server_transport, transport_kind="stdio")
        client = AppServerClient(client_transport, run_peer=harness.serve)
        host = await AppServerHost.connect(
            client,
            client_info=ClientInfo(name="parity-c2b", version="0"),
            capabilities=capabilities,
            session_options=options,
            client_factory=harness.connect_client,
        )
        connection = BackendContractConnection(client=client, host=host)
        session = await host.open_session()
        backend = cast(Any, harness._server._root)
        return connection, session, loop, backend, loop, harness
    process = HarnessProcess()
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(
        server_transport, transport_kind=IN_PROCESS_TRANSPORT, process=process
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    host = await AppServerHost.connect(
        client,
        client_info=ClientInfo(name="parity-c2b", version="0"),
        capabilities=capabilities,
        session_options=options,
        client_factory=harness.connect_client,
    )
    connection = BackendContractConnection(client=client, host=host)
    session = await host.open_session()
    loop = next(
        candidate
        for candidate in process._accounting_loops
        if candidate.session_id == session.session_id
    )
    fake_backend = FakeBackend([[mock_llm_chunk(content="ok")] for _ in range(4)])
    loop._backend_lifetime.replace(fake_backend, whole_turn_active=False)
    loop._injected_backend = fake_backend
    return connection, session, loop, None, process, harness


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_root_death_receipt_fences_the_old_generation_in_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The added §6.3 root-death case: the outcome set per mode.

    After a confirmed root death the stdio generation answers every request
    with the typed death receipt — which discloses the terminal disconnect,
    the possible loss of in-flight work, and the explicit session/resume path
    — and publishes no old-generation events. The in_process leg records the
    same live read invariants; the death leg is the stdio root's own lifecycle
    receipt. The trace records through the C1k regression registry: the
    checked-in baseline is asserted, and ``VIBE_TEST_DIFFERENTIAL_REFRESH=1``
    rewrites it.
    """
    script = {"case": regressions.ROOT_DEATH, "modes": list(_MODES)}
    frames: list[dict[str, object]] = []
    outcomes: dict[str, tuple[object, ...]] = {}
    for mode in _MODES:
        with transport_mode(mode):
            project = tmp_path / f"death-project-{mode}"
            project.mkdir(parents=True, exist_ok=True)
            (
                connection,
                session,
                loop,
                backend,
                owner,
                harness,
            ) = await _c2b_parity_session(mode, monkeypatch, project)
            try:
                client = connection.client
                session_id = session.session_id
                async for _event in session.act("hi"):
                    pass
                read_after = RuntimeReadResponse.model_validate(
                    await client.request(
                        "runtime/read", RuntimeReadParams(session_id=session_id)
                    )
                )
                if mode == _STDIO_MODE:
                    assert backend is not None
                    backend.process._declare_death(
                        "the root agent process exited",
                        signature=RootFailureSignature.DEATH_RECEIPT,
                    )
                    with pytest.raises(AppServerResponseError) as exc_info:
                        await client.request(
                            "runtime/read", RuntimeReadParams(session_id=session_id)
                        )
                    death_message = str(exc_info.value)
                    death_receipt = "death-receipt" in death_message
                    death_code = str(exc_info.value.error.code)
                    disclosure = (
                        "in-flight work" in death_message
                        and "session/resume" in death_message
                        and "never replayed" in death_message
                    )
                    guard_fenced = False
                    try:
                        backend.guard_request()
                    except SessionBackendError as exc:
                        guard_fenced = "death-receipt" in str(exc)
                    publish_fenced = (
                        await backend.publish_callback(cast(Any, None)) is False
                    )
                    outcomes[mode] = (
                        read_after.ready,
                        death_receipt,
                        death_code,
                        disclosure,
                        guard_fenced,
                        publish_fenced,
                    )
                    frames.extend([
                        {
                            "mode": mode,
                            "op": "read-after-turn",
                            "ready": read_after.ready,
                        },
                        {
                            "mode": mode,
                            "op": "read-after-death",
                            "death_receipt": death_receipt,
                            "code": death_code,
                            "disclosure": disclosure,
                        },
                        {
                            "mode": mode,
                            "op": "guard-and-publish-fence",
                            "guard_fenced": guard_fenced,
                            "publish_fenced": publish_fenced,
                        },
                    ])
                else:
                    outcomes[mode] = (read_after.ready,)
                    frames.append({
                        "mode": mode,
                        "op": "read-after-turn",
                        "ready": read_after.ready,
                    })
            finally:
                await session.close()
                await connection.host.close()
                if mode == _STDIO_MODE:
                    await owner.aclose()
                else:
                    await owner.close()

    # The shared read behavior is identical on both transports; the death leg
    # is the stdio root's own lifecycle receipt.
    assert outcomes[_STDIO_MODE][0] == outcomes[IN_PROCESS][0] is True
    assert outcomes[_STDIO_MODE][1:] == (True, "internal_error", True, True, True)
    spec = regressions.case(regressions.ROOT_DEATH)
    regressions.record_case(spec, script, frames)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_resume_after_death_respawns_restores_and_surfaces_memorial_in_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The added §6.3 resume case: the outcome set per mode.

    After a confirmed root death the explicit ``session/resume`` retires the
    dead generation, spawns and handshakes one replacement over the wire, and
    restores the durable session under the same id; the resume receipt carries
    the retired generation's typed death receipt and its coordinator-owned
    queue/shell records as memorial state, never re-admitted for execution on
    the new generation, and the old generation is fenced. On in_process the
    same resume is the re-attach arm (no death receipt, empty memorial) and
    the session stays usable on both transports. The trace records through
    the C1k regression registry: the checked-in baseline is asserted, and
    ``VIBE_TEST_DIFFERENTIAL_REFRESH=1`` rewrites it.
    """
    script = {"case": regressions.RESUME, "modes": list(_MODES)}
    frames: list[dict[str, object]] = []
    outcomes: dict[str, tuple[object, ...]] = {}
    for mode in _MODES:
        with transport_mode(mode):
            project = tmp_path / f"resume-project-{mode}"
            project.mkdir(parents=True, exist_ok=True)
            (
                connection,
                session,
                loop,
                backend,
                owner,
                harness,
            ) = await _c2b_parity_session(mode, monkeypatch, project)
            try:
                client = connection.client
                session_id = session.session_id
                async for _event in session.act("before death"):
                    pass
                if mode == _STDIO_MODE:
                    assert backend is not None
                    backend._turn_queue.enqueue(
                        TurnEnqueueParams(
                            session_id=session_id,
                            entries=[
                                TurnUserInputEntry(
                                    content=[
                                        SessionTextContentBlock(text="queued prompt")
                                    ]
                                )
                            ],
                        )
                    )
                    backend.process._declare_death(
                        "the root agent process exited",
                        signature=RootFailureSignature.DEATH_RECEIPT,
                    )
                response = SessionResumeResponse.model_validate(
                    await client.request(
                        "session/resume",
                        SessionResumeParams(session_id=session_id).model_dump(
                            mode="json", by_alias=True
                        ),
                    )
                )
                # The durable session is restored under the same id.
                assert response.state.session.id == session_id
                if mode == _STDIO_MODE:
                    # The resumed generation is usable: the read on the new
                    # generation answers ready. (A further turn on the memory
                    # wire would hit the shared loop's closed backend
                    # lifetime — a production respawn builds a fresh loop in
                    # the new process; the in_process leg below runs the
                    # turn.)
                    pass
                else:
                    # The resumed session is usable: one more turn completes.
                    async for _event in session.act("after resume"):
                        pass
                read = RuntimeReadResponse.model_validate(
                    await client.request(
                        "runtime/read", RuntimeReadParams(session_id=session_id)
                    )
                )
                usable = read.ready is True
                if mode == _STDIO_MODE:
                    assert harness is not None
                    death_receipt = (
                        response.death_receipt is not None
                        and "death-receipt" in response.death_receipt
                    )
                    memorial = [(item.kind, item.text) for item in response.memorial]
                    new_backend = cast(Any, harness._server._root)
                    readmitted = new_backend._turn_queue.state.items != []
                    old_fenced = False
                    try:
                        backend.guard_request()
                    except SessionBackendError as exc:
                        old_fenced = "death-receipt" in str(exc)
                    outcomes[mode] = (
                        usable,
                        death_receipt,
                        tuple(memorial),
                        readmitted,
                        old_fenced,
                        new_backend is not backend,
                    )
                    frames.extend([
                        {
                            "mode": mode,
                            "op": "resume-after-death",
                            "restored_id": response.state.session.id == session_id,
                            "death_receipt": death_receipt,
                            "memorial": memorial,
                            "readmitted": readmitted,
                            "old_fenced": old_fenced,
                            "respawned": new_backend is not backend,
                        },
                        {"mode": mode, "op": "turn-after-resume", "usable": usable},
                    ])
                else:
                    outcomes[mode] = (
                        usable,
                        response.death_receipt is None,
                        response.memorial == [],
                    )
                    frames.extend([
                        {
                            "mode": mode,
                            "op": "resume-reattach",
                            "restored_id": response.state.session.id == session_id,
                            "death_receipt": response.death_receipt,
                            "memorial": [
                                (item.kind, item.text) for item in response.memorial
                            ],
                        },
                        {"mode": mode, "op": "turn-after-resume", "usable": usable},
                    ])
            finally:
                await session.close()
                await connection.host.close()
                if mode == _STDIO_MODE:
                    await owner.aclose()
                else:
                    await owner.close()

    # The shared resume behavior is identical on both transports: the durable
    # session is restored under the same id and stays usable. The death
    # receipt, memorial and respawn legs are the stdio root's own lifecycle.
    assert outcomes[_STDIO_MODE][0] == outcomes[IN_PROCESS][0] is True
    assert outcomes[IN_PROCESS][1:] == (True, True)
    assert outcomes[_STDIO_MODE][1:] == (
        True,
        (("queued_turn", "queued prompt"),),
        False,
        True,
        True,
    )
    spec = regressions.case(regressions.RESUME)
    regressions.record_case(spec, script, frames)
