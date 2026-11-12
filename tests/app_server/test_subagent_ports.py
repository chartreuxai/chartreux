"""Spawn + completion-source ports and their adapters (WP5 [5A])."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import pytest

from chartreux.app_server.agent_wire import (
    AGENT_RECEIVING_NOTIFICATIONS,
    AGENT_RECEIVING_REQUESTS,
    AGENT_TURN_ENDED_METHOD,
    HOST_SUBAGENT_ENDED_METHOD,
    HOST_SUBAGENT_EVENT_METHOD,
    HOST_TOOL_IO_METHOD,
    SUBAGENT_OPEN_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    ToolIOCapabilities,
    agent_request_id,
    decode_host_run_end,
    decode_host_run_frame,
    decode_host_service_offers,
    decode_shell_command_request,
    decode_tool_io_request,
    decode_tool_io_result,
    encode_host_run_end,
    encode_host_run_frame,
    encode_host_service_offers,
    encode_shell_command_request,
    encode_tool_io_request,
    encode_tool_io_result,
)
from chartreux.app_server.child_agent import CHILD_AGENT_MODULE
from chartreux.app_server.child_agent_process import (
    ProcessSpawnedSubagent,
    ProcessSubagentSpawn,
    child_agent_command,
    multi_process_child_selected,
)
from chartreux.app_server.process_bootstrap import (
    PROCESS_PROTOCOL_VERSION,
    ReadyFrame,
    SpawnedProcess,
)
from chartreux.app_server.root_agent_process import RootAgentProcess
from chartreux.app_server.subagent_ports import (
    CHILD_FAILURE_FRAME_KEY,
    ChildFailureEvidence,
    ChildTeardownOutcome,
    ChildTeardownReport,
    ChildTeardownRequest,
    InProcessSubagentSpawn,
    SubagentCompletion,
    SubagentFrame,
    SubagentSpawnDeniedError,
    SubagentSpawnSpec,
    decode_child_failure,
    failure_decision,
    spawn_spec_from_wire,
    spawn_spec_to_wire,
    subagent_completion_frame,
    subagent_spawn_port,
)
from chartreux.app_server.transport import TRANSPORT_ENV_VAR
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.events import (
    AssistantEvent,
    BaseEvent,
    ToolStreamEvent,
    UserMessageEvent,
)
from chartreux.core.llm.exceptions import IncompleteStreamError
from chartreux.core.recovery_models import (
    CallEvidence,
    attach_call_evidence,
    build_failure_envelope,
)
from chartreux.core.recovery_observation import observe_exception
from chartreux.core.subagents import TaskResult
from chartreux.core.tools.io_port import ShellCommandRequest, ShellCommandResult
from chartreux.utils.io import BoundedReadResult, ReadSafeResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.differential import egress
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CHILD_SCRIPT = """\
import asyncio

from chartreux.app_server.child_agent import serve_child_agent
from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class _Factory:
    def __init__(self, loop):
        self._loop = loop

    async def open(self, params):
        return self._loop

    async def close(self):
        pass


init_harness_files_manager("user", "project")
config = build_test_vibe_config(
    session_logging=SessionLoggingConfig(enabled=True, save_dir={save_dir!r})
)
loop = build_test_agent_loop(
    config=config, backend=FakeBackend([[mock_llm_chunk(content="done")]])
)
asyncio.run(
    serve_child_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""


def _child_command(tmp_path: Path) -> list[str]:
    script = _CHILD_SCRIPT.format(save_dir=str(tmp_path / "sessions"))
    return [sys.executable, "-c", script]


def _child_env(guard_dir: Path, *, selector: str = "stdio") -> dict[str, str]:
    env = dict(os.environ)
    env.update(egress.child_environment(egress.MODE_ALL))
    env[TRANSPORT_ENV_VAR] = selector
    paths = [str(_REPO_ROOT), str(guard_dir)]
    inherited = env.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class _LoopFactory:
    """In-process child factory: one test loop per spawn."""

    def __init__(self, loop) -> None:
        self._loop = loop
        self.specs: list[SubagentSpawnSpec] = []

    async def open(self, spec: SubagentSpawnSpec):
        self.specs.append(spec)
        return self._loop


def _loop(tmp_path: Path, backend: FakeBackend | None = None):
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        )
    )
    return build_test_agent_loop(
        config=config,
        backend=backend or FakeBackend([[mock_llm_chunk(content="done")]]),
    )


def _spec() -> SubagentSpawnSpec:
    return SubagentSpawnSpec(
        agent_id="agent-1",
        parent_session_id="root-session-1",
        session_dir="sessions/root-session-1/agents/agent-1",
        payload={"task_summary": "do the thing"},
    )


def _turn_request() -> ExecutionTurnRequest:
    return ExecutionTurnRequest(
        message="hello", turn_options=AgentTurnOptions(turn_id="turn-1")
    )


async def _collect(source, request: ExecutionTurnRequest) -> list[SubagentFrame]:
    frames: list[SubagentFrame] = []
    async with aclosing(source.stream(request)) as stream:
        async for frame in stream:
            frames.append(frame)
    return frames


def test_transport_selection_picks_the_spawn_adapter() -> None:
    assert multi_process_child_selected({}) is False
    assert multi_process_child_selected({TRANSPORT_ENV_VAR: "in_process"}) is False
    assert multi_process_child_selected({TRANSPORT_ENV_VAR: ""}) is False
    assert multi_process_child_selected({TRANSPORT_ENV_VAR: "stdio"}) is True
    assert multi_process_child_selected({TRANSPORT_ENV_VAR: "multi_process"}) is True

    class _AnyFactory:
        async def open(self, spec: SubagentSpawnSpec):
            raise AssertionError("must not be called")

    assert isinstance(
        subagent_spawn_port({}, child_factory=_AnyFactory()), InProcessSubagentSpawn
    )
    assert isinstance(
        subagent_spawn_port(
            {TRANSPORT_ENV_VAR: "in_process"}, child_factory=_AnyFactory()
        ),
        InProcessSubagentSpawn,
    )
    assert isinstance(
        subagent_spawn_port({TRANSPORT_ENV_VAR: "stdio"}, child_factory=_AnyFactory()),
        ProcessSubagentSpawn,
    )
    assert child_agent_command() == [sys.executable, "-m", CHILD_AGENT_MODULE]


def test_spawn_spec_wire_roundtrip() -> None:
    spec = _spec()
    assert spawn_spec_from_wire(spawn_spec_to_wire(spec)) == spec
    assert spawn_spec_from_wire({"agent_id": "agent-2"}) == SubagentSpawnSpec(
        agent_id="agent-2"
    )
    with pytest.raises(ValueError, match="no agent id"):
        spawn_spec_from_wire({"payload": {}})
    with pytest.raises(ValueError, match="no agent id"):
        spawn_spec_from_wire({"agent_id": ""})


def test_spawn_spec_carries_the_publication_attempt_across_the_wire() -> None:
    spec = SubagentSpawnSpec(agent_id="agent-3", attempt_id="attempt-child-9")
    wire = spawn_spec_to_wire(spec)
    assert wire["attempt_id"] == "attempt-child-9"
    assert spawn_spec_from_wire(wire) == spec
    # A spawn minted without an attempt keeps the field absent, so a child
    # created outside a publication attempt records none.
    assert spawn_spec_from_wire({"agent_id": "agent-2"}).attempt_id is None


def test_child_teardown_request_and_report_round_trip() -> None:
    request = ChildTeardownRequest(
        attempt_id="attempt-child-9", agent_id="coder", wait_seconds=0.5
    )
    assert ChildTeardownRequest.from_wire(request.to_wire()) == request
    assert (
        ChildTeardownRequest.from_wire({"attempt_id": "attempt-child-9"}).reason
        == "publication-teardown"
    )

    report = ChildTeardownReport(
        attempt_id="attempt-child-9",
        outcome=ChildTeardownOutcome.TOMBSTONED,
        agent_id="coder",
        session_id="child-session-9",
        child_dir="sessions/root-session-1/agents/coder_1",
        retracted=True,
        detail="the persisted records were retracted",
    )
    assert ChildTeardownReport.from_wire(report.to_wire()) == report
    assert report.settled is True


def test_child_teardown_wire_rejects_malformed_payloads() -> None:
    with pytest.raises(ValueError, match="no attempt id"):
        ChildTeardownRequest.from_wire({})
    with pytest.raises(ValueError, match="no known outcome"):
        # A peer cannot report an outcome this end does not know: a claimed
        # rollback outside the four established classes never parses.
        ChildTeardownReport.from_wire({
            "attempt_id": "attempt-child-9",
            "outcome": "rolled-back",
        })
    with pytest.raises(ValueError, match="no attempt id"):
        ChildTeardownReport.from_wire({"outcome": ChildTeardownOutcome.REAPED.value})
    inconclusive = ChildTeardownReport(
        attempt_id="attempt-child-9", outcome=ChildTeardownOutcome.INCONCLUSIVE
    )
    assert inconclusive.settled is False
    assert (
        ChildTeardownReport(
            attempt_id="attempt-child-9", outcome=ChildTeardownOutcome.ATTEMPT_UNKNOWN
        ).settled
        is False
    )


@pytest.mark.asyncio
async def test_in_process_adapter_streams_events_then_completion(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path)
    factory = _LoopFactory(loop)
    port = subagent_spawn_port({TRANSPORT_ENV_VAR: "in_process"}, child_factory=factory)
    assert isinstance(port, InProcessSubagentSpawn)
    spawned = await port.spawn(_spec())
    try:
        assert spawned.agent_id == "agent-1"
        assert factory.specs == [_spec()]
        frames = await _collect(spawned.completion_source, _turn_request())
    finally:
        await spawned.close()

    assert isinstance(frames[0], UserMessageEvent)
    assert frames[0].content == "hello"
    assert [event.content for event in frames if isinstance(event, AssistantEvent)] == [
        "done"
    ]
    completion = frames[-1]
    assert isinstance(completion, SubagentCompletion)
    assert completion.outcome == TURN_OUTCOME_COMPLETED
    assert completion.error is None
    assert all(isinstance(frame, BaseEvent) for frame in frames[:-1])


@pytest.mark.asyncio
async def test_in_process_adapter_carries_a_failed_run_as_the_completion_frame(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path, backend=FakeBackend(exception_to_raise=RuntimeError("boom")))
    port = InProcessSubagentSpawn(_LoopFactory(loop))
    spawned = await port.spawn(_spec())
    try:
        frames = await _collect(spawned.completion_source, _turn_request())
    finally:
        await spawned.close()

    completion = frames[-1]
    assert isinstance(completion, SubagentCompletion)
    assert completion.outcome == TURN_OUTCOME_FAILED
    assert completion.error is not None and "boom" in completion.error


@pytest.mark.asyncio
async def test_process_adapter_refuses_to_spawn_under_in_process(
    tmp_path: Path,
) -> None:
    port = ProcessSubagentSpawn(
        env={TRANSPORT_ENV_VAR: "in_process"}, command=_child_command(tmp_path)
    )
    with pytest.raises(SubagentSpawnDeniedError, match="in-process adapter"):
        await port.spawn(_spec())


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_process_adapter_spawns_a_child_and_streams_its_run(
    tmp_path: Path,
) -> None:
    with egress.bootstrap_directory() as guard_dir:
        port = ProcessSubagentSpawn(
            env=_child_env(guard_dir),
            cwd=str(_REPO_ROOT),
            command=_child_command(tmp_path),
            ready_timeout=30,
            request_timeout=60,
        )
        spawned = await port.spawn(_spec())
        assert isinstance(spawned, ProcessSpawnedSubagent)
        try:
            assert spawned.agent_id == "agent-1"
            frames = await _collect(spawned.completion_source, _turn_request())
        finally:
            await spawned.close()

    assert isinstance(frames[0], UserMessageEvent)
    assert frames[0].content == "hello"
    assert [event.content for event in frames if isinstance(event, AssistantEvent)] == [
        "done"
    ]
    completion = frames[-1]
    assert isinstance(completion, SubagentCompletion)
    assert completion.outcome == TURN_OUTCOME_COMPLETED
    assert spawned.process.returncode is not None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_child_entrypoint_speaks_the_subagent_open_frame(tmp_path: Path) -> None:
    """The child server answers subagent/open with the spec's agent id."""
    from chartreux.app_server.process_bootstrap import spawn_process

    script = _CHILD_SCRIPT.format(save_dir=str(tmp_path / "sessions"))
    with egress.bootstrap_directory() as guard_dir:
        env = _child_env(guard_dir)
        spawned = await spawn_process(
            [sys.executable, "-c", script],
            env=env,
            cwd=str(_REPO_ROOT),
            ready_timeout=30,
        )
        try:
            transport = spawned.transport
            await transport.send({
                "jsonrpc": "2.0",
                "id": "1",
                "method": SUBAGENT_OPEN_METHOD,
                "params": spawn_spec_to_wire(_spec()),
            })
            response: dict[str, Any] | None = None
            async for message in transport.messages():
                if message.get("id") == "1":
                    response = message
                    break
            assert response is not None
            result = response.get("result")
            assert isinstance(result, dict)
            assert result.get("agent_id") == "agent-1"
            assert isinstance(result.get("session_id"), str)
        finally:
            await spawned.transport.close()


# ---------------------------------------------------------------------------
# op-9: the host-service channel's wire surface (the pre-rider subset).
#
# These assert what travels when the agent asks the coordinator for the two
# host REQUEST ports a turn needs. The spawned end-to-end turn lives in
# test_root_process.py; here the codecs and the coordinator-end refusal are
# pinned without a process.
# ---------------------------------------------------------------------------


class _HostToolIO:
    """A minimal host tool I/O port: the capability flags are all it advertises."""

    supports_read = True
    supports_write = False
    supports_terminal = True


def test_host_service_offers_announce_the_ports_a_turn_may_call() -> None:
    offers = decode_host_service_offers({})
    assert offers.subagent_runner is False
    assert offers.tool_io is None

    payload = encode_host_service_offers(
        subagent_runner=object(), tool_io=_HostToolIO()
    )
    offers = decode_host_service_offers({"host_services": payload})
    assert offers.subagent_runner is True
    assert offers.tool_io == ToolIOCapabilities(
        supports_read=True, supports_write=False, supports_terminal=True
    )


def test_tool_io_requests_and_results_round_trip() -> None:
    request = encode_tool_io_request(
        "read_lines", {"path": "a.txt", "start_line": 2, "limit": 3, "max_bytes": 64}
    )
    op, arguments = decode_tool_io_request(request)
    assert op == "read_lines"
    assert arguments["path"] == "a.txt"

    bounded = BoundedReadResult(
        lines=["two", "three"], total_lines=9, was_truncated=True
    )
    assert decode_tool_io_result(op, encode_tool_io_result(op, bounded)) == bounded

    text = ReadSafeResult(text="hello", encoding="utf-8", newline="\n")
    assert (
        decode_tool_io_result("read_text", encode_tool_io_result("read_text", text))
        == text
    )

    shell = ShellCommandResult(stdout="out", stderr="err", returncode=2)
    assert (
        decode_tool_io_result("run_shell", encode_tool_io_result("run_shell", shell))
        == shell
    )
    assert (
        decode_tool_io_result("write_text", encode_tool_io_result("write_text", None))
        is None
    )

    shell_request = ShellCommandRequest(
        session_id="session-1",
        tool_call_id="call-1",
        command="ls",
        cwd=Path("/tmp"),
        timeout=5.0,
        max_output_bytes=128,
        args=["-l"],
        env={"LANG": "C"},
    )
    decoded = decode_shell_command_request(encode_shell_command_request(shell_request))
    assert decoded.command == "ls"
    assert decoded.args == ["-l"]
    assert decoded.env == {"LANG": "C"}
    assert str(decoded.cwd) == "/tmp"


def test_host_subagent_run_frames_round_trip() -> None:
    event = ToolStreamEvent(tool_name="task", message="working", tool_call_id="call-1")
    assert decode_host_run_frame(encode_host_run_frame(event)) == event

    result = TaskResult(response="done", turns_used=2, completed=True)
    assert decode_host_run_frame(encode_host_run_frame(result)) == result

    assert decode_host_run_end(encode_host_run_end(result=result)) == (result, None)
    assert decode_host_run_end(encode_host_run_end()) == (None, None)
    assert decode_host_run_end(encode_host_run_end(error="boom")) == (None, "boom")


class _FramePipe:
    """A transport double: the test feeds frames and reads what the end sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            frame = await self._incoming.get()
            if frame is None:
                return
            yield frame

    def feed(self, frame: dict[str, Any]) -> None:
        self._incoming.put_nowait(frame)

    async def close(self) -> None:
        self._incoming.put_nowait(None)


class _ProcessDouble:
    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_host_tool_io_request_needs_a_port_offered_with_the_turn() -> None:
    pipe = _FramePipe()
    process = RootAgentProcess(
        SpawnedProcess(
            process=cast("subprocess.Popen[bytes]", _ProcessDouble()),
            transport=cast("Any", pipe),
            ready=ReadyFrame(protocol_version=PROCESS_PROTOCOL_VERSION),
        ),
        heartbeat_interval=30.0,
    )
    try:
        pipe.feed({
            "jsonrpc": "2.0",
            "id": agent_request_id(1),
            "method": HOST_TOOL_IO_METHOD,
            "params": {"op": "read_text", "path": "a.txt"},
        })
        await _until(lambda: pipe.sent)

        reply = pipe.sent[0]
        assert reply["id"] == agent_request_id(1)
        # No turn is in flight, so there is no port to serve: typed, not a hang.
        assert "UnsupportedProcessTurnRequest" in reply["error"]["code"]
        assert "no host tool I/O port" in reply["error"]["message"]
        assert process.death is None
    finally:
        await process.close()


def test_the_host_run_answers_are_notifications_never_requests() -> None:
    # A host subagent run's frames answer the agent's request: they carry no id
    # and live in the notification table, never in the request table.
    assert HOST_SUBAGENT_EVENT_METHOD in AGENT_RECEIVING_NOTIFICATIONS
    assert HOST_SUBAGENT_ENDED_METHOD in AGENT_RECEIVING_NOTIFICATIONS
    assert HOST_SUBAGENT_EVENT_METHOD not in AGENT_RECEIVING_REQUESTS
    assert HOST_SUBAGENT_ENDED_METHOD not in AGENT_RECEIVING_REQUESTS


def _child_failure_evidence() -> ChildFailureEvidence:
    """A derived envelope plus its inputs: the terminal frame's failure block."""
    error = IncompleteStreamError("prov", "wire")
    attach_call_evidence(
        error,
        CallEvidence(
            logical_call_id="call-1",
            started_at=0.0,
            budget_seconds=60.0,
            response_started=True,
            semantic_output_published=False,
            transport_attempts=1,
            elapsed_seconds=0.5,
            budget_exhausted=False,
        ),
    )
    observation = observe_exception(error, phase="turn", turn_id="turn-1")
    envelope = build_failure_envelope(
        observation=observation,
        agent_id="agent-1",
        run_id="turn-1",
        turn_id="turn-1",
        tool_calls_present=True,
    )
    return ChildFailureEvidence(
        envelope=envelope,
        derivation_inputs={
            "agent_id": "agent-1",
            "run_id": "turn-1",
            "tool_calls_present": True,
            "tool_calls_pending": False,
            "stop_reason": None,
            "observation": {"phase": "turn", "call_evidence": None},
        },
    )


def test_the_completion_frame_carries_the_child_failure_evidence() -> None:
    evidence = _child_failure_evidence()
    frame = subagent_completion_frame(
        "agent-1",
        4,
        SubagentCompletion(outcome=TURN_OUTCOME_FAILED, error="boom", failure=evidence),
    )

    assert frame["method"] == AGENT_TURN_ENDED_METHOD
    block = frame["params"][CHILD_FAILURE_FRAME_KEY]
    assert block["derivation_inputs"]["tool_calls_present"] is True
    decoded = decode_child_failure(block)
    assert decoded is not None
    assert failure_decision(decoded.envelope) == failure_decision(evidence.envelope)
    assert "tool_effects_settled" in decoded.envelope.recovery.reason_codes

    # A completed run's terminal frame carries no failure block at all.
    completed = subagent_completion_frame(
        "agent-1", 4, SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED)
    )
    assert CHILD_FAILURE_FRAME_KEY not in completed["params"]
