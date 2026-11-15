"""WP6 [6D]: ``-p`` stdout/stderr/exit parity — the legacy-representative oracle.

Design invariant 1 defines "old" as the campaign's recorded baseline and makes
in_process-vs-stdio parity with ``-p`` stdout/stderr/exit parity the legacy-
representative oracle (design.md:522-524). This module runs the real programmatic
route — the installed CLI entrypoint with ``-p``, a live fake provider on
loopback, and the [6A]-blocked transport — under both transport selections and
compares what a caller observes:

* stdout, stderr, and the exit status are identical across modes;
* the provider received the same seeded request sequence in both modes, so the
  parity is not just an equal rendering of different work;
* continuation (``-c``) and the programmatic denial of a dangerous tool call
  settle identically, including the exit status of a failing turn.

Two run-scoped values are normalized before comparison, and only those: the
per-run scratchpad the harness mints (both the legacy ``chartreux-scratchpad-*``
name and the managed ``chartreux-managed-<uid>/alloc-*`` allocation), the
caller's own directory tree (each mode writes its own config home under it), and
wall-clock stamps the request carries (a tool call's ``started_at``). Everything
else — the model text, the denial, the exit status, the request messages — is
compared exactly.

The programmatic route composes the app-server in this process for both
selectors today (``LocalHarness``), so these tests pin the selection-neutrality
the route must keep: the stdio selection may not change what ``-p`` emits, and
a later wiring of the multi-process route has to preserve this parity.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest

from chartreux.app_server._runtime import HarnessProcess, create_harness_server
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.host import AppServerHost
from chartreux.app_server.protocol import (
    ClientCapabilities,
    ClientInfo,
    DangerousCommandAuthorityReadParams,
    DangerousCommandAuthorityReadResponse,
    DangerousCommandRevokeParams,
    DangerousCommandRevokeResponse,
    SessionOptions,
    SessionReadParams,
    SessionReadResponse,
    SessionStopParams,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import (
    IN_PROCESS_TRANSPORT,
    TRANSPORT_ENV_VAR,
    memory_transport_pair,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.dangerous_commands import RepositoryScope
from tests.app_server.backend_contract.conftest import (
    BackendContractConnection,
    connect_backend_contract_stdio_host,
    install_memory_root_agent,
)
from tests.conftest import build_test_agent_loop
from tests.differential import regressions
from tests.differential.replay import (
    TRANSPORT_MODES,
    build_replay_script,
    transport_mode,
)
from tests.e2e.agent_loop_characterization.support import (
    assistant_text_chunks,
    single_tool_call_chunks,
)
from tests.e2e.common import write_e2e_config
from tests.e2e.mock_server import StreamingMockServer

STDIO = "stdio"
IN_PROCESS = "in_process"
MODES = TRANSPORT_MODES
DEFAULT_SEED = 20_261_110
_RUN_TIMEOUT = 180
_DENIED_ARGUMENTS = {"command": "git clean -fd"}
_SCRATCHPAD = re.compile(r"chartreux-scratchpad-[A-Za-z0-9_-]+")
_MANAGED_ALLOCATION = re.compile(r"chartreux-managed-\d+/alloc-[0-9a-f]+")
_APPROX_CHARS = re.compile(r'"approx_chars":\d+')
_TIMESTAMP = re.compile(r'"(started_at|completed_at|created_at|updated_at)":\d+')


def _cli() -> Path:
    """The chartreux entrypoint next to the interpreter running the tests."""
    executable = Path(sys.executable).with_name("chartreux")
    assert executable.exists(), executable
    return executable


def _environment(home: Path, mode: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CHARTREUX_HOME"] = str(home)
    environment["MISTRAL_API_KEY"] = "fake-key"
    environment["CHARTREUX_TEST_DISABLE_KEYRING"] = "1"
    environment["CHARTREUX_TEST_DISABLE_AUTO_TITLE"] = "1"
    environment["TERM"] = "xterm-256color"
    environment[TRANSPORT_ENV_VAR] = mode
    return environment


def _prepare(
    tmp_path: Path, mode: str, server: StreamingMockServer
) -> tuple[Path, Path]:
    home = tmp_path / f"home-{mode}"
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    write_e2e_config(home, server.api_base)
    return home, workdir


def _normalize(text: str, roots: Iterable[Path]) -> str:
    for root in roots:
        text = text.replace(str(root), "<root>")
    text = _SCRATCHPAD.sub("<scratchpad>", text)
    text = _MANAGED_ALLOCATION.sub("<scratchpad>", text)
    text = _TIMESTAMP.sub(r'"\1":<timestamp>', text)
    return _APPROX_CHARS.sub('"approx_chars":<n>', text)


def test_normalizer_flattens_the_legacy_scratchpad_name() -> None:
    assert _normalize("/tmp/chartreux-scratchpad-s42/turn.txt", ()) == (
        "/tmp/<scratchpad>/turn.txt"
    )


def test_normalizer_flattens_the_managed_allocation_inside_its_parent() -> None:
    expected = "/tmp/<scratchpad>/turn.txt"
    allocation = "/tmp/chartreux-managed-1000/alloc-9c1d3e5f7a02/turn.txt"
    other_rand = "/tmp/chartreux-managed-1000/alloc-4b8c0d2e6f1a/turn.txt"
    other_uid = "/tmp/chartreux-managed-1001/alloc-9c1d3e5f7a02/turn.txt"
    assert _normalize(allocation, ()) == expected
    assert _normalize(other_rand, ()) == expected
    assert _normalize(other_uid, ()) == expected


def test_normalizer_keeps_allocation_text_outside_the_managed_parent() -> None:
    bare = "alloc-9c1d3e5f7a02"
    foreign_parent = "/tmp/other-root/alloc-9c1d3e5f7a02"
    non_numeric_uid = "/tmp/chartreux-managed-acme/alloc-9c1d3e5f7a02"
    assert _normalize(bare, ()) == bare
    assert _normalize(foreign_parent, ()) == foreign_parent
    assert _normalize(non_numeric_uid, ()) == non_numeric_uid


def test_normalizer_keeps_meaningful_request_differences_visible() -> None:
    denied = {
        "command": "git clean -fd",
        "scratchpad": "/tmp/chartreux-managed-1000/alloc-9c1d3e5f7a02",
    }
    allowed = {
        "command": "git status",
        "scratchpad": "/tmp/chartreux-managed-1000/alloc-4b8c0d2e6f1a",
    }
    first = _normalize(json.dumps(denied, sort_keys=True), ())
    second = _normalize(json.dumps(allowed, sort_keys=True), ())
    assert "<scratchpad>" in first
    assert "<scratchpad>" in second
    assert first != second


def _run(
    server: StreamingMockServer, *, tmp_path: Path, mode: str, arguments: list[str]
) -> tuple[tuple[str, str, int], list[str], list[dict[str, Any]]]:
    """Run one invocation; return its observable, normalized and raw requests."""
    home, workdir = _prepare(tmp_path, mode, server)
    started = len(server.requests)
    completed = subprocess.run(
        [str(_cli()), *arguments],
        cwd=str(workdir),
        env=_environment(home, mode),
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT,
        check=False,
    )
    raw_requests = [dict(request) for request in server.requests[started:]]
    requests = [
        _normalize(
            json.dumps(request, sort_keys=True, separators=(",", ":")), (tmp_path,)
        )
        for request in raw_requests
    ]
    observable = (
        _normalize(completed.stdout, (tmp_path,)),
        _normalize(completed.stderr, (tmp_path,)),
        completed.returncode,
    )
    return observable, requests, raw_requests


def _single_turn_prompt() -> str:
    return build_replay_script(DEFAULT_SEED).turns[0]


@pytest.mark.timeout(300)
def test_single_turn_parity_includes_the_provider_request_sequence(
    tmp_path: Path,
) -> None:
    server = StreamingMockServer()
    server.start()
    try:
        runs = {
            mode: _run(
                server,
                tmp_path=tmp_path,
                mode=mode,
                arguments=["-p", _single_turn_prompt()],
            )
            for mode in MODES
        }
    finally:
        server.stop()

    assert runs[STDIO][0] == runs[IN_PROCESS][0]
    stdout, stderr, returncode = runs[STDIO][0]
    assert returncode == 0
    assert stderr == ""
    assert "Hello from mock server" in stdout
    # The same seeded work reached the provider: not an equal rendering of
    # different turns.
    assert runs[STDIO][1] == runs[IN_PROCESS][1]
    assert len(runs[STDIO][1]) == 1
    assert _single_turn_prompt() in runs[STDIO][1][0]


@pytest.mark.timeout(300)
def test_continuation_parity_holds_across_modes(tmp_path: Path) -> None:
    script = build_replay_script(DEFAULT_SEED)
    server = StreamingMockServer()
    server.start()
    try:
        runs: dict[str, list[tuple[str, str, int]]] = {}
        continuations: dict[str, list[str]] = {}
        for mode in MODES:
            first = _run(
                server, tmp_path=tmp_path, mode=mode, arguments=["-p", script.turns[0]]
            )[0]
            second, second_requests, _raw = _run(
                server,
                tmp_path=tmp_path,
                mode=mode,
                arguments=["-c", "-p", script.turns[1]],
            )
            assert second[2] == 0
            runs[mode] = [first, second]
            continuations[mode] = second_requests
    finally:
        server.stop()

    assert runs[STDIO] == runs[IN_PROCESS]
    # The continuation really carried the second seeded prompt, in both modes.
    assert continuations[STDIO] == continuations[IN_PROCESS]
    assert script.turns[1] in continuations[STDIO][0]
    # ... and the first turn's prompt is in the continued conversation.
    assert script.turns[0] in continuations[STDIO][0]


@pytest.mark.timeout(300)
def test_dangerous_tool_denial_is_identical_in_both_modes(tmp_path: Path) -> None:
    def chunk_factory(index: int, payload: Any) -> list[dict[str, object]]:
        # Scope the tool call to the run, not to the server's request counter:
        # each mode drives its own conversation from the same first turn.
        messages = payload.get("messages") or []
        if any(message.get("role") == "tool" for message in messages):
            return assistant_text_chunks("settled after the denial")
        return single_tool_call_chunks(
            call_id="call-1", tool_name="bash", arguments=_DENIED_ARGUMENTS
        )

    server = StreamingMockServer(chunk_factory=chunk_factory)
    server.start()
    try:
        runs = {
            mode: _run(
                server,
                tmp_path=tmp_path,
                mode=mode,
                arguments=["-p", _single_turn_prompt()],
            )
            for mode in MODES
        }
    finally:
        server.stop()

    assert runs[STDIO][0] == runs[IN_PROCESS][0]
    stdout, _stderr, returncode = runs[STDIO][0]
    assert returncode == 0
    assert "settled after the denial" in stdout
    # Both modes ran the tool turn and reported the refusal back to the model.
    assert runs[STDIO][1] == runs[IN_PROCESS][1]
    assert len(runs[STDIO][2]) == 2
    tool_results = [
        message
        for request in runs[STDIO][2]
        for message in request["messages"]
        if message.get("role") == "tool"
    ]
    assert len(tool_results) == 1
    assert "denied" in str(tool_results[0]["content"]).lower()


@pytest.mark.timeout(300)
def test_failed_provider_turn_exits_identically_in_both_modes(tmp_path: Path) -> None:
    server = StreamingMockServer(error_status=500)
    server.start()
    try:
        runs = {
            mode: _run(
                server,
                tmp_path=tmp_path,
                mode=mode,
                arguments=["-p", _single_turn_prompt()],
            )
            for mode in MODES
        }
    finally:
        server.stop()

    assert runs[STDIO][0] == runs[IN_PROCESS][0]
    stdout, stderr, returncode = runs[STDIO][0]
    # A failed turn is a failed command: never a silent success.
    assert returncode != 0
    assert stdout == ""
    assert "LLM backend error" in stderr


def _make_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


async def _race_session(
    mode: str, monkeypatch: pytest.MonkeyPatch, project: Path
) -> tuple[BackendContractConnection, AppServerSession, AgentLoop]:
    """One app-server session on ``mode`` plus the loop owning its grant store.

    The stdio leg runs the root agent over a memory wire (the spawned-stdio
    legs land with C5c's PTY gate); the in_process leg keeps its harness
    process reachable so the session's loop can be addressed owner-side.
    """
    capabilities = ClientCapabilities(callback_kinds=["user_input"])
    options = SessionOptions(cwd=str(project))
    if mode == STDIO:
        loop = build_test_agent_loop(cwd=project)
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
        client_info=ClientInfo(name="p-parity-revocation-race", version="0"),
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
    return connection, session, loop


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_revocation_race_reports_the_cas_conflict_outcome_set_in_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The added §6.3 revocation-race case: the CAS outcome set, per mode.

    A read observes the session's live authority generation; a concurrent
    approval then lands in the process that owns the store and advances the
    generation; the revoke naming the observed (now superseded) generation
    must answer the defined conflict outcome -- reporting the current
    generation and leaving the saved side untouched, never a false ack --
    and only the re-read generation revokes. Both transport selections must
    produce the identical outcome sequence. The trace records through the
    C1k regression registry: the checked-in baseline is asserted, and
    ``VIBE_TEST_DIFFERENTIAL_REFRESH=1`` rewrites it.
    """
    project = _make_git_repo(tmp_path / "race-project")
    script = {"case": regressions.REVOCATION_RACE, "modes": list(MODES)}
    frames: list[dict[str, object]] = []
    outcomes: dict[str, tuple[object, ...]] = {}
    for mode in MODES:
        with transport_mode(mode):
            connection, session, loop = await _race_session(mode, monkeypatch, project)
            try:
                client = connection.client
                session_id = session.session_id

                async def _read(
                    client: AppServerClient = client, session_id: str = session_id
                ) -> DangerousCommandAuthorityReadResponse:
                    return DangerousCommandAuthorityReadResponse.model_validate(
                        await client.request(
                            "policy/dangerous_commands/read",
                            DangerousCommandAuthorityReadParams(session_id=session_id),
                        )
                    )

                async def _revoke(
                    expected_live_generation: int,
                    client: AppServerClient = client,
                    session_id: str = session_id,
                ) -> DangerousCommandRevokeResponse:
                    return DangerousCommandRevokeResponse.model_validate(
                        await client.request(
                            "policy/dangerous_commands/revoke",
                            DangerousCommandRevokeParams(
                                session_id=session_id,
                                operation_id="git.push",
                                expected_live_generation=expected_live_generation,
                                expected_disk_revision=None,
                                user_initiated=True,
                            ),
                        )
                    )

                read = await _read()
                assert read.project == str(project)
                assert read.live_grants == []
                observed = read.live_authority_generation

                # The concurrent approval lands owner-side: the store's
                # generation advances past the one the read observed.
                assert loop._dangerous_command_grants.grant_session({
                    (
                        "git.push",
                        RepositoryScope(git_dir=project / ".git", work_tree=project),
                    )
                })
                current = loop._dangerous_command_grants.authority_generation
                assert current == observed + 1

                stale = await _revoke(observed)
                assert stale.live.status == "conflict"
                assert stale.live.generation == current
                assert stale.saved.status == "skipped"
                assert stale.live_authority_generation == current
                assert stale.warning is None

                # No false ack: the live grant survives the stale revoke.
                reread = await _read()
                assert reread.live_authority_generation == current
                assert [grant.operation_id for grant in reread.live_grants] == [
                    "git.push"
                ]

                applied = await _revoke(reread.live_authority_generation)
                assert applied.live.status == "removed"
                assert applied.saved.status == "skipped"

                final = await _read()
                assert [grant.operation_id for grant in final.live_grants] == []

                outcomes[mode] = (
                    observed,
                    current,
                    stale.live.status,
                    stale.live.generation,
                    stale.saved.status,
                    applied.live.status,
                    [grant.operation_id for grant in reread.live_grants],
                    [grant.operation_id for grant in final.live_grants],
                )
                frames.extend([
                    {"mode": mode, "op": "read", "generation": observed, "live": []},
                    {
                        "mode": mode,
                        "op": "revoke-stale",
                        "expected": observed,
                        "live_status": stale.live.status,
                        "live_generation": stale.live.generation,
                        "saved_status": stale.saved.status,
                        "live_authority_generation": stale.live_authority_generation,
                    },
                    {
                        "mode": mode,
                        "op": "read-after-stale",
                        "generation": reread.live_authority_generation,
                        "live": [grant.operation_id for grant in reread.live_grants],
                    },
                    {
                        "mode": mode,
                        "op": "revoke-current",
                        "expected": reread.live_authority_generation,
                        "live_status": applied.live.status,
                        "saved_status": applied.saved.status,
                    },
                    {
                        "mode": mode,
                        "op": "read-final",
                        "live": [grant.operation_id for grant in final.live_grants],
                    },
                ])
            finally:
                await session.close()
                await connection.host.close()
                if mode == STDIO:
                    await loop.aclose()

    assert outcomes[STDIO] == outcomes[IN_PROCESS]
    spec = regressions.case(regressions.REVOCATION_RACE)
    regressions.record_case(spec, script, frames)


async def _c2c_stop_session(
    mode: str, monkeypatch: pytest.MonkeyPatch, project: Path
) -> tuple[BackendContractConnection, AppServerSession, Any, Any, Any]:
    """One app-server session on ``mode`` plus its harness server and owner.

    The stdio leg runs the root agent over a memory wire (the spawned-stdio
    legs land with C5c's PTY gate); the in_process leg keeps its harness
    process reachable. Returns the connection, the session, the loop, the
    harness (whose ``_server`` the stop gate patches), and the teardown owner.
    """
    capabilities = ClientCapabilities(callback_kinds=["user_input"])
    options = SessionOptions(cwd=str(project))
    client_transport, server_transport = memory_transport_pair()
    if mode == STDIO:
        loop = build_test_agent_loop(cwd=project)
        install_memory_root_agent(monkeypatch, loop)
        harness = await create_harness_server(server_transport, transport_kind="stdio")
        owner: Any = loop
    else:
        process = HarnessProcess()
        harness = await create_harness_server(
            server_transport, transport_kind=IN_PROCESS_TRANSPORT, process=process
        )
        owner = process
    client = AppServerClient(client_transport, run_peer=harness.serve)
    host = await AppServerHost.connect(
        client,
        client_info=ClientInfo(name="p-parity-mutation-timeout", version="0"),
        capabilities=capabilities,
        session_options=options,
        client_factory=harness.connect_client,
    )
    connection = BackendContractConnection(client=client, host=host)
    session = await host.open_session()
    loop = next(
        candidate
        for candidate in ([owner] if mode == STDIO else owner._accounting_loops)
        if candidate.session_id == session.session_id
    )
    return connection, session, loop, harness, owner


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_mutation_timeout_stop_reconciles_without_resend_in_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The added §6.3 mutation-timeout case: the outcome set, per mode.

    A lifecycle mutation (``session/stop``) whose reply outlives the client's
    timeout is an UNKNOWN outcome: it is surfaced as a timeout, never
    suppressed and never resent -- the client reconciles by asking (a read,
    then the resume/reopen path), and the server-side stop still settles. Both
    transport selections must produce the identical outcome sequence. The
    trace records through the C1k regression registry: the checked-in
    baseline is asserted, and ``VIBE_TEST_DIFFERENTIAL_REFRESH=1`` rewrites it.
    """
    script = {"case": regressions.MUTATION_TIMEOUT, "modes": list(MODES)}
    frames: list[dict[str, object]] = []
    outcomes: dict[str, tuple[object, ...]] = {}
    for mode in MODES:
        with transport_mode(mode):
            project = tmp_path / f"stop-project-{mode}"
            project.mkdir(parents=True, exist_ok=True)
            connection, session, loop, harness, owner = await _c2c_stop_session(
                mode, monkeypatch, project
            )
            try:
                client = connection.client
                session_id = session.session_id
                server = harness._server
                gate = asyncio.Event()
                original_run_stop = server._run_stop

                async def gated_run_stop(
                    raw_params: dict[str, Any],
                    _gate: asyncio.Event = gate,
                    _original: Any = original_run_stop,
                ) -> Any:
                    await _gate.wait()
                    return await _original(raw_params)

                monkeypatch.setattr(server, "_run_stop", gated_run_stop)
                # The client's stop outlives its short timeout: the outcome is
                # unknown, surfaced as a timeout -- never a blind resend.
                with pytest.raises(TimeoutError):
                    await client.request(
                        "session/stop",
                        SessionStopParams(session_id=session_id),
                        timeout=0.2,
                    )
                frames.append({"mode": mode, "op": "stop", "outcome": "timeout"})
                # Reconcile with a READ while the stop is still in flight: the
                # client asks for the state instead of resending the mutation.
                read = SessionReadResponse.model_validate(
                    await client.request(
                        "session/read", SessionReadParams(session_id=session_id)
                    )
                )
                reconciled = read.state.session.id == session_id
                frames.append({
                    "mode": mode,
                    "op": "reconcile-read",
                    "reconciled": reconciled,
                })
                gate.set()
                for _ in range(200):
                    if server._shutdown_complete:
                        break
                    await asyncio.sleep(0.01)
                settled = server._shutdown_complete and server._root is None
                frames.append({"mode": mode, "op": "settled", "settled": settled})
                outcomes[mode] = ("timeout", reconciled, settled)
            finally:
                await session.close()
                await connection.host.close()
                if mode == STDIO:
                    await loop.aclose()
                else:
                    await owner.close()

    assert outcomes[STDIO] == outcomes[IN_PROCESS] == ("timeout", True, True)
    spec = regressions.case(regressions.MUTATION_TIMEOUT)
    regressions.record_case(spec, script, frames)
