"""The root agent boots in its own process and serves the execution port."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import aclosing, asynccontextmanager, suppress
from datetime import UTC, datetime
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server import (
    _runtime as runtime,
    agent_reconfigure,
    process_session_backend as psb_module,
    root_agent,
)
from chartreux.app_server._connection_protocol import ClientInfo
from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._projection import project_session_log
from chartreux.app_server._runtime import close_agent_loop
from chartreux.app_server._session_backend_port import SessionBackendError
from chartreux.app_server._session_model import override_active_model
from chartreux.app_server._session_runtime_impl import ProcessRootReconfigure
from chartreux.app_server.agent_admission import (
    OP_OUTCOME_ABORTED,
    OP_OUTCOME_COMPLETED,
    OP_STATE_ACTIVE,
    OP_STATE_TERMINAL,
)
from chartreux.app_server.agent_snapshot import (
    ADOPTION_ADOPTED,
    ADOPTION_NONE,
    MARKER_STALE,
    MarkerVerdict,
    build_session_snapshot,
)
from chartreux.app_server.agent_wire import (
    AGENT_AGENTS_INVENTORY_READ_METHOD,
    AGENT_COMPACT_METHOD,
    AGENT_NOTICE_METHOD,
    AGENT_OP_ADMISSION_METHOD,
    AGENT_OP_CANCEL_METHOD,
    AGENT_OPEN_METHOD,
    AGENT_SESSION_FORK_METHOD,
    AGENT_SESSION_HISTORY_CLEAR_METHOD,
    AGENT_SESSION_RECONFIGURE_METHOD,
    AGENT_SESSION_RELOCATE_METHOD,
    AGENT_SESSION_RENAME_METHOD,
    AGENT_SESSION_REPLACE_METHOD,
    AGENT_SESSION_REWIND_METHOD,
    AGENT_SESSION_REWIND_READ_METHOD,
    AGENT_SESSION_SNAPSHOT_METHOD,
    CLEAR_STAGE_ALREADY_CLEAR,
    CLEAR_STAGE_HISTORY,
    CLEAR_STAGE_MODEL_OVERRIDE,
    COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED,
    COMPACT_SETTLEMENT_PUBLISHED_SETTLED,
    COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED,
    RECONFIGURE_CONFLICT_ERROR,
    RECONFIGURE_MODE_CONFIG_WRITE,
    RECONFIGURE_MODE_PLAN_HANDOFF,
    RECONFIGURE_MODE_SETTINGS,
    RECONFIGURE_OUTCOME_ACCEPTED,
    RECONFIGURE_OUTCOME_APPLIED,
    RECONFIGURE_OUTCOME_CLEARED,
    RECONFIGURE_OUTCOME_CLEARED_WITH_FOLLOWUP_FAILURE,
    UnsupportedAgentRequestError,
    coordinator_request_id,
    decode_compact_reply,
    decode_manual_shell_context,
    decode_open_accounting,
    decode_reconfigure_reply,
    decode_replace_reply,
    decode_session_snapshot,
    encode_manual_shell_context,
    encode_notice,
    encode_open_accounting,
    encode_open_request,
    encode_reconfigure_request,
    encode_replace_request,
    encode_turn_request,
    required_agent_capabilities,
)
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.models import (
    SessionTextContentBlock,
    TextContentBlock,
    TurnUserInputEntry,
)
from chartreux.app_server.process_bootstrap import (
    ADVERTISED_CAPABILITIES,
    AGENT_WIRE_CAPABILITY_LIVE_AGENT_INVENTORY,
    AGENT_WIRE_CAPABILITY_PLAN_HANDOFF_ROTATION,
    PROCESS_PROTOCOL_VERSION,
    ProtocolVersionMismatchError,
    ReadyFrame,
    ReadyFrameMissingError,
    SpawnedProcess,
    encode_ready_frame,
    parse_ready_frame,
)
from chartreux.app_server.process_session_backend import (
    DEFAULT_REQUEST_TIMEOUT as PSB_REQUEST_TIMEOUT,
)
from chartreux.app_server.protocol import (
    AgentConfig,
    AgentsInventoryReadParams,
    AppServerResponseError,
    ConfigWriteOpWire,
    ConfigWriteParams,
    ContextInjectParams,
    ProtocolErrorCode,
    RuntimeReadParams,
    SessionOptions,
    SessionPlanHandoffParams,
    SessionResumeParams,
    SessionResumeResponse,
    SessionTurnsListParams,
    SessionTurnsListResponse,
    TurnEnqueueParams,
    TurnSteerParams,
)
from chartreux.app_server.registry_manifest import RegistryManifest
from chartreux.app_server.root_agent_process import (
    COMPACT_DEADLINE,
    COMPACT_OBSERVATION_IN_FLIGHT,
    COMPACT_OBSERVATION_SETTLEMENT_UNKNOWN,
    DEFAULT_REQUEST_TIMEOUT as ROOT_PORT_REQUEST_TIMEOUT,
    REPLACE_DEADLINE,
    CompactSettlementRecord,
    RootAgentCapabilitySkewError,
    RootAgentProcess,
    RootAgentReconfigureConflict,
    RootAgentReplaceConflict,
    RootAgentRequestError,
    RootAgentSelectionError,
    RootAgentSessionNotFound,
    RootFailureSignature,
    multi_process_root_selected,
    spawn_root_agent_process,
)
from chartreux.app_server.transport import (
    TRANSPORT_ENV_VAR,
    JsonRpcTransport,
    memory_transport_pair,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._execution_port import (
    ExecutionTurnRequest,
    TurnOrderingConflict,
)
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import ModelConfig
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.events import (
    AssistantEvent,
    BaseEvent,
    SessionTitleUpdatedEvent,
    ToolStreamEvent,
    UserInputRequestEvent,
    UserMessageEvent,
)
from chartreux.core.llm_models import LLMMessage, ManualShellContext, Role
from chartreux.core.plan_episode import (
    STATE_ACCEPTED,
    STATE_DRAFT,
    PlanEpisodeSidecar,
    accepted_snapshot_path,
    create_sidecar,
    ensure_episode_fence,
    provenance_snapshot_path,
    read_sidecar,
    sha256_bytes,
)
from chartreux.core.session.session_index import MESSAGES_FILENAME
from chartreux.core.session.session_lease import SessionBusyError, SessionLease
from chartreux.core.session_types import PlanModeRecord, PlanScanDomain
from chartreux.core.subagents import TaskArgs, TaskResult
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.io_port import ShellCommandRequest, ShellCommandResult
from chartreux.core.usage import (
    UsageOutcome,
    UsagePrices,
    UsageRecord,
    UsageState,
    UsageWriteDisposition,
    UsageWriteResult,
)
from chartreux.questions import UserAnswer, UserQuestionResult
from chartreux.utils.io import BoundedReadResult, ReadSafeResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.differential import egress
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CHILD_SCRIPT = """\
import asyncio

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
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
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""


def _child_command(tmp_path: Path) -> list[str]:
    script = _CHILD_SCRIPT.format(save_dir=str(tmp_path / "sessions"))
    return [sys.executable, "-c", script]


def _child_env(guard_dir: Path, *, selector: str = "multi_process") -> dict[str, str]:
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


async def _spawn(tmp_path: Path, guard_dir: Path, **kwargs: Any) -> RootAgentProcess:
    return await spawn_root_agent_process(
        session_options=AgentConfig(),
        client_info=ClientInfo(name="tests", version="1.0"),
        env=_child_env(guard_dir),
        cwd=str(_REPO_ROOT),
        command=_child_command(tmp_path),
        ready_timeout=30,
        request_timeout=60,
        **kwargs,
    )


async def _collect(
    process: RootAgentProcess, request: ExecutionTurnRequest
) -> list[BaseEvent]:
    events: list[BaseEvent] = []
    async with aclosing(process.run_turn(request)) as stream:
        async for event in stream:
            events.append(event)
    return events


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_root_agent_process_streams_a_turn_in_production_order(
    tmp_path: Path,
) -> None:
    with egress.bootstrap_directory() as guard_dir:
        process = await _spawn(tmp_path, guard_dir)
        try:
            assert process.session_id
            events = await _collect(
                process,
                ExecutionTurnRequest(
                    message="hello", turn_options=AgentTurnOptions(turn_id="turn-1")
                ),
            )
        finally:
            await process.close()

    assert isinstance(events[0], UserMessageEvent)
    assert events[0].content == "hello"
    assert [event.content for event in events if isinstance(event, AssistantEvent)] == [
        "done"
    ]
    assert process.process.returncode is not None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_second_turn_is_refused_while_one_is_in_flight(tmp_path: Path) -> None:
    with egress.bootstrap_directory() as guard_dir:
        process = await _spawn(tmp_path, guard_dir)
        try:
            first = process.run_turn(
                ExecutionTurnRequest(
                    message="first", turn_options=AgentTurnOptions(turn_id="turn-1")
                )
            )
            stream: AsyncIterator[BaseEvent] = first
            await stream.__anext__()
            with pytest.raises(TurnOrderingConflict):
                await process.run_turn(
                    ExecutionTurnRequest(
                        message="second",
                        turn_options=AgentTurnOptions(turn_id="turn-2"),
                    )
                ).__anext__()
            await first.aclose()
        finally:
            await process.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_root_agent_pages_are_routed_through_the_registry_manifest(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    manifest = RegistryManifest(session_dir)
    manifest.initialize()

    with egress.bootstrap_directory() as guard_dir:
        process = await _spawn(tmp_path, guard_dir, manifest=manifest)
        agent_id = process.session_id
        assert agent_id is not None
        try:
            await _collect(
                process,
                ExecutionTurnRequest(
                    message="hello", turn_options=AgentTurnOptions(turn_id="turn-1")
                ),
            )
            assert manifest.is_live(agent_id)
            page = await manifest.read_live_page(agent_id)
        finally:
            await process.close()

    assert page.state == "available"
    assert page.entries is not None
    assert [
        entry.display_text
        for entry in page.entries
        if entry.title == "Assistant response"
    ] == ["done"]
    assert not manifest.is_live(agent_id)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_agent_process_saves_the_transcript_it_projects(
    tmp_path: Path,
) -> None:
    with egress.bootstrap_directory() as guard_dir:
        process = await _spawn(tmp_path, guard_dir)
        try:
            await _collect(
                process,
                ExecutionTurnRequest(
                    message="hello", turn_options=AgentTurnOptions(turn_id="turn-1")
                ),
            )
        finally:
            await process.close()

    transcripts = list((tmp_path / "sessions").rglob(MESSAGES_FILENAME))
    assert transcripts
    assert "done" in transcripts[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_run_that_selects_in_process_never_spawns(tmp_path: Path) -> None:
    assert not multi_process_root_selected({})
    assert not multi_process_root_selected({TRANSPORT_ENV_VAR: "in_process"})
    assert not multi_process_root_selected({TRANSPORT_ENV_VAR: ""})
    assert multi_process_root_selected({TRANSPORT_ENV_VAR: "multi_process"})

    env = dict(os.environ)
    env[TRANSPORT_ENV_VAR] = "in_process"
    with pytest.raises(RootAgentSelectionError):
        await spawn_root_agent_process(
            session_options=AgentConfig(),
            client_info=ClientInfo(name="tests", version="1.0"),
            env=env,
            command=_child_command(tmp_path),
        )


def test_main_loads_local_sources_and_announces_when_spawned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    served: dict[str, Any] = {}

    async def fake_serve(**kwargs: Any) -> None:
        calls.append("serve")
        served.update(kwargs)

    monkeypatch.setattr(root_agent, "serve_root_agent", fake_serve)
    monkeypatch.setattr(
        root_agent, "init_harness_files_manager", lambda *_args: calls.append("files")
    )
    monkeypatch.setattr(
        root_agent,
        "init_file_logging",
        lambda *_args, **_kwargs: calls.append("logging"),
    )
    monkeypatch.setattr(root_agent, "HarnessRootLoopFactory", lambda: "factory")
    monkeypatch.setattr(root_agent, "bootstrap_requested", lambda: True)
    monkeypatch.setattr(
        "chartreux.core.config.load_dotenv_values", lambda: calls.append("dotenv")
    )

    root_agent.main()

    assert calls == ["files", "logging", "dotenv", "serve"]
    assert served["announce_ready"] is True
    assert served["factory"] == "factory"


# ---------------------------------------------------------------------------
# op-9: the host-service channel, over a really spawned pipe.
#
# The agent process is the first agent-origin requester: the two host REQUEST
# ports a turn needs (the subagent runner and the client tool I/O port) stay in
# the coordinator, and the agent asks for them across the wire. The scripted
# loop below drives both ports inside its turn, then waits for a user-input
# callback, so one spawned turn exercises host requests AND callbacks.
# ---------------------------------------------------------------------------

_HOST_SERVICES_SCRIPT = """\
import asyncio
from pathlib import Path

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.events import AssistantEvent, UserInputRequestEvent
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.io_port import ShellCommandRequest
from chartreux.questions import QuestionChoice, UserQuestion, UserQuestionRequest


class _Broker:
    def __init__(self, port):
        self._port = port

    async def settle_answer(self, request_id, result):
        self._port.answer(request_id, result)
        return "answered"

    async def settle_rejection(self, request_id, error):
        self._port.answer(request_id, error)
        return "refused"


class _Port:
    def __init__(self):
        self._answers = dict()

    async def run_turn(self, request):
        text = await request.tool_io.read_text(Path({text_path!r}))
        yield AssistantEvent(content="read:" + text.text.strip())
        shell = await request.tool_io.run_shell(
            ShellCommandRequest(
                session_id="session-1",
                tool_call_id="call-1",
                command="printf",
                args=["shell-ok"],
                cwd=Path("."),
                timeout=5.0,
                max_output_bytes=4096,
            )
        )
        yield AssistantEvent(content="shell:" + shell.stdout.strip())
        frames = []
        async for frame in request.subagent_runner.run(
            TaskArgs(task="child work", identity="coder"),
            InvokeContext(tool_call_id="call-2", session_id="session-1"),
        ):
            frames.append(frame)
        yield AssistantEvent(content="subagent:" + frames[-1].response)
        future = asyncio.get_running_loop().create_future()
        self._answers["ask-1"] = future
        yield UserInputRequestEvent(
            request_id="ask-1",
            tool_call_id="call-3",
            args=UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="Proceed?",
                        options=[QuestionChoice(label="Yes"), QuestionChoice(label="No")],
                    )
                ]
            ),
        )
        answer = await future
        yield AssistantEvent(content="answer:" + answer.answers[0].answer)

    def answer(self, request_id, value):
        future = self._answers.get(request_id)
        if future is not None and not future.done():
            future.set_result(value)


class _Loop:
    def __init__(self):
        self.session_id = "root-session"
        port = _Port()
        self.execution_port = port
        self.request_broker = _Broker(port)

    async def aclose(self):
        pass


class _Factory:
    async def open(self, params):
        return _Loop()

    async def close(self):
        pass


init_harness_files_manager("user", "project")
asyncio.run(serve_root_agent(factory=_Factory(), announce_ready=bootstrap_requested()))
"""


class _HostToolIO:
    """The coordinator-side tool I/O port; every call is a wire round trip."""

    supports_read = True
    supports_write = True
    supports_terminal = True

    def __init__(self) -> None:
        self.shell_requests: list[ShellCommandRequest] = []
        self.writes: list[tuple[Path, str]] = []

    async def read_lines(
        self, path: Path, *, start_line: int, limit: int, max_bytes: int
    ) -> BoundedReadResult:
        raise AssertionError("this turn reads whole text, never line windows")

    async def read_text(self, path: Path) -> ReadSafeResult:
        return ReadSafeResult(path.name, "utf-8", "\n")

    async def write_text(
        self,
        path: Path,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        self.writes.append((path, content))

    async def run_shell(self, request: ShellCommandRequest) -> ShellCommandResult:
        self.shell_requests.append(request)
        return ShellCommandResult(stdout="shell-ok\n", stderr="", returncode=0)


class _HostSubagentRunner:
    """The coordinator-side subagent runner: one event, then the result."""

    def __init__(self) -> None:
        self.calls: list[tuple[TaskArgs, InvokeContext]] = []

    def run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        return self._run(args, ctx)

    async def _run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        self.calls.append((args, ctx))
        yield ToolStreamEvent(
            tool_name="task", message="working", tool_call_id="child-call"
        )
        yield TaskResult(response="child-done", turns_used=1, completed=True)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_host_ports_and_user_input_callbacks_cross_the_spawned_pipe(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "host-file.txt"
    text_path.write_text("ignored: the host serves the read\n", encoding="utf-8")
    script = _HOST_SERVICES_SCRIPT.format(text_path=str(text_path))
    tool_io = _HostToolIO()
    runner = _HostSubagentRunner()

    with egress.bootstrap_directory() as guard_dir:
        process = await spawn_root_agent_process(
            session_options=AgentConfig(),
            client_info=ClientInfo(name="tests", version="1.0"),
            env=_child_env(guard_dir),
            cwd=str(_REPO_ROOT),
            command=[sys.executable, "-c", script],
            ready_timeout=30,
            request_timeout=60,
        )
        try:
            events, answer = await _turn_with_callback(process, runner, tool_io)
        finally:
            await process.close()

    contents = [event.content for event in events if isinstance(event, AssistantEvent)]
    # The tool I/O port served both operations from the coordinator.
    assert contents[0] == "read:host-file.txt"
    assert contents[1] == "shell:shell-ok"
    # The subagent runner's result streamed back as the run's frames.
    assert contents[2] == "subagent:child-done"
    # The user-input callback settled agent-side and finished the turn.
    assert contents[3] == "answer:Yes"
    assert answer == "answered"

    assert tool_io.shell_requests
    shell_request = tool_io.shell_requests[0]
    assert shell_request.command == "printf"
    assert shell_request.args == ["shell-ok"]
    assert shell_request.session_id == "session-1"
    assert shell_request.tool_call_id == "call-1"

    assert len(runner.calls) == 1
    args, context = runner.calls[0]
    assert args.task == "child work"
    assert args.identity == "coder"
    # The portable context fields crossed; the coordinator rebuilt the rest.
    assert context.tool_call_id == "call-2"
    assert context.session_id == "session-1"
    assert context.session_dir is None


async def _turn_with_callback(
    process: RootAgentProcess, runner: _HostSubagentRunner, tool_io: _HostToolIO
) -> tuple[list[BaseEvent], Any]:
    """Drive one turn, answering its user-input request through the wire."""
    events: list[BaseEvent] = []
    answer: Any = None
    async with aclosing(
        process.run_turn(
            ExecutionTurnRequest(
                message="hello",
                subagent_runner=runner,
                tool_io=tool_io,
                turn_options=AgentTurnOptions(turn_id="turn-1"),
            )
        )
    ) as stream:
        async for event in stream:
            events.append(event)
            if isinstance(event, UserInputRequestEvent):
                answer = await process.settle_user_input_answer(
                    event.request_id,
                    UserQuestionResult(
                        answers=[UserAnswer(question="Proceed?", answer="Yes")]
                    ),
                )
    return events, answer


# ---------------------------------------------------------------------------
# op 1/2/3/7: interrupt, steering, injection and out-of-band notices.
#
# The scripted loop below must do what the real loop does for each seam: a turn
# that stays in flight until it is cancelled, a waiting-only predicate the test
# sets, the injection primitive WITH its commit callbacks (commit evidence is
# read from them, so a fake that skipped them would prove nothing), and an
# out-of-band event source.
# ---------------------------------------------------------------------------

_MANUAL_RECORD = (
    "Manual `!` command result from the user.\n\nCommand: `sleep 999`\n"
    "Cwd: /workspace\nStatus: timed_out"
)

_TURN_CONTROL_SCRIPT = '''\
import asyncio

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.events import (
    AssistantEvent,
    SessionTitleUpdatedEvent,
    ToolStreamEvent,
    UserMessageEvent,
)


def expect_manual_shell(context):
    """Fail the request unless the record crossed the pipe intact."""
    assert context.status == {shell_status!r}, context.status
    assert context.exit_code == {shell_exit!r}, context.exit_code
    assert context.command == {shell_command!r}, context.command
    assert context.output_text == {shell_output!r}, context.output_text


class _SessionLogger:
    session_dir = None


class _Port:
    async def run_turn(self, request):
        yield AssistantEvent(content="started")
        await asyncio.Event().wait()


class _Loop:
    def __init__(self):
        self.session_id = "root-session"
        self.session_logger = _SessionLogger()
        self.injections = []
        self.steering_cancellations = []
        self.execution_port = _Port()
        self.notices = asyncio.Queue()
        self.notices.put_nowait(
            SessionTitleUpdatedEvent(title="Titled", session_id="root-session")
        )

    async def out_of_band_events(self):
        while True:
            yield await self.notices.get()

    def is_waiting_only(self, turn_id):
        return {waiting_only!r}

    def cancel_outstanding_waits_for_steering(self, turn_id):
        self.steering_cancellations.append(turn_id)
        return ()

    async def inject_user_context(
        self,
        content,
        *,
        as_message=False,
        inject_implicit=False,
        images=None,
        input_text=None,
        resources=None,
        client_message_id=None,
        manual_shell=None,
        before_commit=None,
        on_commit=None,
        on_injected_event=None,
        on_context_ready=None,
    ):
        if before_commit is not None:
            before_commit()
        events = []
        if as_message:
            if manual_shell is not None:
                expect_manual_shell(manual_shell)
            self.injections.append(content)
            count = len(self.injections)
            event = UserMessageEvent(
                content=input_text if input_text is not None else content,
                message_id=client_message_id or "generated-" + str(count),
            )
            events.append(event)
            if on_commit is not None:
                on_commit(event)
            if inject_implicit:
                expanded = ToolStreamEvent(
                    tool_name="skill",
                    message="expanded-" + str(count),
                    tool_call_id="skill-" + str(count),
                )
                events.append(expanded)
                if on_injected_event is not None:
                    on_injected_event(expanded)
        else:
            self.injections.append(content)
        if on_context_ready is not None:
            on_context_ready()
        return events

    async def aclose(self):
        pass


class _Factory:
    async def open(self, params):
        return _Loop()

    async def close(self):
        pass


init_harness_files_manager("user", "project")
asyncio.run(serve_root_agent(factory=_Factory(), announce_ready=bootstrap_requested()))
'''


def _manual_shell() -> ManualShellContext:
    """A timed-out record: a status with no exit code, which the wire must keep."""
    return ManualShellContext(
        operation_id="op-1",
        command="sleep 999",
        cwd="/workspace",
        status="timed_out",
        exit_code=None,
        timed_out=True,
        created_at=1700000000000,
        duration_ms=12.0,
    )


async def _spawn_turn_control(
    tmp_path: Path, guard_dir: Path, *, waiting_only: bool
) -> RootAgentProcess:
    shell = _manual_shell()
    script = _TURN_CONTROL_SCRIPT.format(
        waiting_only=waiting_only,
        shell_status=shell.status,
        shell_exit=shell.exit_code,
        shell_command=shell.command,
        shell_output=shell.output_text,
    )
    return await spawn_root_agent_process(
        session_options=AgentConfig(),
        client_info=ClientInfo(name="tests", version="1.0"),
        env=_child_env(guard_dir),
        cwd=str(_REPO_ROOT),
        command=[sys.executable, "-c", script],
        ready_timeout=30,
        request_timeout=60,
    )


def _steer_params(
    *, key: str, require_waiting_only: bool, text: str
) -> TurnSteerParams:
    return TurnSteerParams(
        session_id="root-session",
        expected_turn_id="turn-1",
        message=[TextContentBlock(text=text)],
        idempotency_key=key,
        require_waiting_only=require_waiting_only,
        inject_invoked_skill=True,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_interrupted_turn_settles_interrupted_over_the_pipe(
    tmp_path: Path,
) -> None:
    with egress.bootstrap_directory() as guard_dir:
        process = await _spawn_turn_control(tmp_path, guard_dir, waiting_only=False)
        try:
            stream = process.run_turn(
                ExecutionTurnRequest(
                    message="hello", turn_options=AgentTurnOptions(turn_id="turn-1")
                )
            )
            async with aclosing(stream):
                started = await stream.__anext__()
                assert isinstance(started, AssistantEvent)
                assert started.content == "started"

                # The turn is in flight but not waiting, so conditional
                # steering is refused typed by the agent, not by the queue.
                with pytest.raises(RootAgentRequestError) as refused:
                    await process.steer(
                        _steer_params(
                            key="steer:blocked",
                            require_waiting_only=True,
                            text="too early",
                        )
                    )
                assert refused.value.code == "SteeringConflictError"
                assert "not waiting-only" in refused.value.message

                await process.interrupt_turn("turn-1")
                with pytest.raises(asyncio.CancelledError):
                    await stream.__anext__()

                # The interrupt's own outcome rides the turn's terminal frame.
                terminal = process.turn_terminal("turn-1")
        finally:
            await process.close()

    assert terminal is not None
    assert terminal["outcome"] == "interrupted"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_steering_commits_onto_the_turn_and_replays_idempotently(
    tmp_path: Path,
) -> None:
    with egress.bootstrap_directory() as guard_dir:
        process = await _spawn_turn_control(tmp_path, guard_dir, waiting_only=True)
        try:
            async with aclosing(process.out_of_band_events()) as notices:
                notice = await asyncio.wait_for(notices.__anext__(), 10)
            assert isinstance(notice, SessionTitleUpdatedEvent)
            assert notice.title == "Titled"

            stream = process.run_turn(
                ExecutionTurnRequest(
                    message="hello", turn_options=AgentTurnOptions(turn_id="turn-1")
                )
            )
            async with aclosing(stream):
                assert isinstance(await stream.__anext__(), AssistantEvent)

                params = _steer_params(
                    key="steer:one", require_waiting_only=True, text="steering"
                )
                settlement = await process.steer(params)
                assert settlement.committed
                assert settlement.events == 2

                # The committed delivery and its expansion ride the turn's own
                # event stream, in the order the loop produced them.
                committed = await stream.__anext__()
                assert isinstance(committed, UserMessageEvent)
                assert committed.content == "steering"
                expanded = await stream.__anext__()
                assert isinstance(expanded, ToolStreamEvent)
                assert expanded.message == "expanded-1"

                # The same key and payload settle again without a second commit.
                replay = await process.steer(params)
                assert replay.committed
                assert replay.events == 2

                # So the next delivery is the second injection, not a third.
                second = await process.steer(
                    _steer_params(
                        key="steer:two", require_waiting_only=True, text="more"
                    )
                )
                assert second.committed
                following = await stream.__anext__()
                assert isinstance(following, UserMessageEvent)
                assert following.content == "more"
                assert isinstance(await stream.__anext__(), ToolStreamEvent)

                await process.interrupt_turn("turn-1")
                with pytest.raises(asyncio.CancelledError):
                    await stream.__anext__()
        finally:
            await process.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_injection_carries_its_manual_shell_record_the_record_text_is_input(
    tmp_path: Path,
) -> None:
    with egress.bootstrap_directory() as guard_dir:
        process = await _spawn_turn_control(tmp_path, guard_dir, waiting_only=False)
        try:
            events = await process.inject(
                ContextInjectParams(
                    session_id="root-session",
                    input=[TextContentBlock(text=_MANUAL_RECORD)],
                    as_message=True,
                    client_user_message_id="record-1",
                ),
                manual_shell=_manual_shell(),
            )
        finally:
            await process.close()

    # The record text is the ordinary user message; the scripted loop refused
    # the delivery unless the manual-shell context arrived intact beside it.
    assert [type(event).__name__ for event in events] == ["UserMessageEvent"]
    assert isinstance(events[0], UserMessageEvent)
    assert events[0].content == _MANUAL_RECORD
    assert events[0].message_id == "record-1"


def test_the_manual_shell_record_round_trips_with_its_optional_exit() -> None:
    failed = ManualShellContext(
        operation_id="op-2",
        command="false",
        cwd="/workspace",
        stdout="",
        stderr="",
        output_text="",
        status="failed",
        exit_code=7,
        created_at=1700000000000,
    )
    for context in (_manual_shell(), failed):
        assert decode_manual_shell_context(encode_manual_shell_context(context)) == (
            context
        )

    # A record written before the status existed still derives its outcome.
    legacy = decode_manual_shell_context({
        "operation_id": "op-3",
        "command": "false",
        "cwd": "/workspace",
        "exit_code": 3,
        "created_at": 1,
    })
    assert legacy.status == "failed"

    with pytest.raises(UnsupportedAgentRequestError):
        decode_manual_shell_context({"status": "melted", "created_at": 1})
    with pytest.raises(UnsupportedAgentRequestError):
        decode_manual_shell_context({"status": "ok", "exit_code": "7", "created_at": 1})


def test_the_turn_request_carries_the_manual_record_as_ordinary_input() -> None:
    # Item-5 B.4: the manual ! turn is an ordinary turn whose input text is the
    # record, so the turn request needs no manual-shell field of its own.
    payload = encode_turn_request(
        ExecutionTurnRequest(
            message=_MANUAL_RECORD,
            client_message_id="record-1",
            turn_options=AgentTurnOptions(turn_id="turn-1"),
        )
    )

    assert payload["message"] == _MANUAL_RECORD
    assert payload["client_message_id"] == "record-1"
    assert "manual_shell" not in payload
    assert "manual_shell" not in payload["turn_options"]


# ---------------------------------------------------------------------------
# Item 1 ops 4/6 (S9-C1, S9-C2, S9-C4): the open reply, the session snapshot
# and long-operation marker admission.
#
# An open that resumes a stored session is a long operation: the coordinator
# mints an absolute monotonic deadline at send, the agent admits it over a
# live fence, records and announces the coordinator's OWN request id, and
# answers that id -- with the reply, or with a typed abort at the deadline.
# The snapshot read answers with state built at reply-write time, and the
# jobs directory is published by the open reply, never by the ready frame.
# ---------------------------------------------------------------------------


class _LoopFactory:
    """The in-process agent factory the server opens its one loop through."""

    def __init__(self, loop: AgentLoop, *, delay: float = 0.0) -> None:
        self._loop = loop
        self._delay = delay
        self.params: list[dict[str, Any]] = []
        # A set event blocks the open body forever: the deadline must abort it.
        self.blocked: asyncio.Event | None = None

    async def open(self, params: Mapping[str, Any]) -> AgentLoop:
        self.params.append(dict(params))
        if self.blocked is not None:
            await self.blocked.wait()
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._loop

    async def close(self) -> None:
        return None


class _AgentPeer:
    """The coordinator end of a raw JSON-RPC pipe to one in-process agent."""

    def __init__(self, transport: JsonRpcTransport) -> None:
        self._transport = transport
        self._replies: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self.notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for frame in self._transport.messages():
            request_id = frame.get("id")
            future = (
                self._replies.get(request_id) if isinstance(request_id, str) else None
            )
            if future is not None:
                if not future.done():
                    future.set_result(frame)
                continue
            await self.notifications.put(frame)

    async def send(self, method: str, params: dict[str, Any]) -> str:
        request_id = coordinator_request_id(next(self._ids))
        self._replies[request_id] = asyncio.get_running_loop().create_future()
        await self._transport.send({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        })
        return request_id

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send one id-less notification: no reply exists to settle."""
        await self._transport.send({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })

    async def reply(self, request_id: str, *, timeout: float = 15.0) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(self._replies[request_id], timeout)
        finally:
            self._replies.pop(request_id, None)

    async def frame(
        self, method: str, params: dict[str, Any], *, timeout: float = 15.0
    ) -> dict[str, Any]:
        return await self.reply(await self.send(method, params), timeout=timeout)

    async def call(
        self, method: str, params: dict[str, Any], *, timeout: float = 15.0
    ) -> dict[str, Any]:
        frame = await self.frame(method, params, timeout=timeout)
        error = frame.get("error")
        assert error is None, error
        return frame["result"]

    async def close(self) -> None:
        self._reader.cancel()
        with suppress(BaseException):
            await self._reader
        await self._transport.close()


def _serve_in_process(factory: Any) -> tuple[_AgentPeer, asyncio.Task[None]]:
    client_transport, server_transport = memory_transport_pair()
    server = root_agent.RootAgentServer(server_transport, factory=factory)
    return _AgentPeer(client_transport), asyncio.create_task(server.serve())


def _open_params(*, deadline: float | None = None) -> dict[str, Any]:
    params = encode_open_request(
        session_options=AgentConfig(),
        client_info=ClientInfo(name="tests", version="1.0"),
    )
    if deadline is not None:
        params["deadline"] = deadline
    return params


async def _persisted_test_loop(tmp_path: Path) -> AgentLoop:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        )
    )
    loop = build_test_agent_loop(config=config)
    await loop.persist_empty_session()
    return loop


@pytest.mark.asyncio
async def test_the_open_reply_and_the_snapshot_carry_the_agent_side_state(
    tmp_path: Path,
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    session_dir = loop.session_logger.session_dir
    assert session_dir is not None
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        opened = await peer.call(AGENT_OPEN_METHOD, _open_params())
        snapshot = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
        log = project_session_log(loop)
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    # The open reply publishes the directories the agent process owns.
    assert opened["session_id"] == loop.session_id
    assert opened["session_dir"] == str(session_dir)
    assert opened["jobs_dir"] == str(session_dir / "jobs")
    assert opened["accounting"]["adoption"] == ADOPTION_NONE
    assert opened["accounting"]["settlements"] == []

    # The snapshot answers with state built from the live loop, not a copy.
    assert snapshot.state.session.id == loop.session_id
    assert snapshot.incarnation_id
    assert snapshot.constructed_at > 0
    assert snapshot.log == log
    assert snapshot.session_dir == str(session_dir)
    assert snapshot.jobs_dir == str(session_dir / "jobs")
    assert snapshot.markers == ()


@pytest.mark.asyncio
async def test_the_deferred_session_siblings_name_the_in_process_escape(
    tmp_path: Path,
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        refusals = [
            await peer.frame(method, {})
            for method in (
                AGENT_SESSION_REWIND_READ_METHOD,
                AGENT_SESSION_REWIND_METHOD,
                AGENT_SESSION_HISTORY_CLEAR_METHOD,
            )
        ]
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    for refusal in refusals:
        error = refusal["error"]
        assert error["code"] == UnsupportedAgentRequestError.__name__
        assert f"{TRANSPORT_ENV_VAR}=in_process" in error["message"]


@pytest.mark.asyncio
async def test_a_long_open_is_announced_by_request_id_then_settles_terminal(
    tmp_path: Path,
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    factory = _LoopFactory(loop, delay=0.05)
    peer, serving = _serve_in_process(factory)
    deadline = time.monotonic() + 60.0
    try:
        request_id = await peer.send(AGENT_OPEN_METHOD, _open_params(deadline=deadline))
        # The admission is announced before the body finishes, so it is queued
        # by the time the reply's own frame arrives.
        announcement = await asyncio.wait_for(peer.notifications.get(), 15)
        reply = await peer.reply(request_id)
        snapshot = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    assert announcement["method"] == AGENT_OP_ADMISSION_METHOD
    admitted = announcement["params"]
    assert admitted["request_id"] == request_id
    assert admitted["method"] == AGENT_OPEN_METHOD
    assert admitted["state"] == "active"
    assert admitted["deadline"] == deadline
    assert admitted["outcome"] is None
    op_id = admitted["op_id"]
    assert op_id

    assert reply.get("error") is None
    assert reply["result"]["session_id"] == loop.session_id
    # The resume parameters the coordinator minted crossed unchanged.
    assert factory.params[0]["deadline"] == deadline

    # S9-C1: the record carries the coordinator's own request id, so the
    # snapshot correlates without any attribution logic.
    record = snapshot.marker_for(request_id)
    assert record is not None
    assert record.op_id == op_id
    assert record.state == OP_STATE_TERMINAL
    assert record.outcome == OP_OUTCOME_COMPLETED
    assert record.deadline == deadline


@pytest.mark.asyncio
async def test_the_deadline_aborts_a_blocked_open_and_frees_the_fence(
    tmp_path: Path,
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    factory = _LoopFactory(loop)
    factory.blocked = asyncio.Event()
    peer, serving = _serve_in_process(factory)
    try:
        aborted_id = await peer.send(
            AGENT_OPEN_METHOD, _open_params(deadline=time.monotonic() + 0.15)
        )
        aborted = await peer.reply(aborted_id)
        # The fence is free again: a second long open is admitted, not refused.
        factory.blocked = None
        admitted_id = await peer.send(
            AGENT_OPEN_METHOD, _open_params(deadline=time.monotonic() + 60.0)
        )
        completed = await peer.reply(admitted_id)
        snapshot = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    # S9-C2: the abort completes the ORIGINAL request id typed.
    error = aborted["error"]
    assert error["code"] == "OperationAbortedError"
    assert "aborted at its absolute deadline" in error["message"]
    assert completed.get("error") is None

    aborted_record = snapshot.marker_for(aborted_id)
    assert aborted_record is not None
    assert aborted_record.state == OP_STATE_TERMINAL
    assert aborted_record.outcome == OP_OUTCOME_ABORTED
    completed_record = snapshot.marker_for(admitted_id)
    assert completed_record is not None
    assert completed_record.state == OP_STATE_TERMINAL
    assert completed_record.outcome == OP_OUTCOME_COMPLETED
    assert completed_record.op_id != aborted_record.op_id


@pytest.mark.asyncio
async def test_a_resumed_root_open_holds_the_lease_and_records_its_accounting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path), session_prefix="session"
        )
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    session_id = saved.session_id
    await saved.aclose()

    process = runtime.HarnessProcess()
    blueprint = Mock()
    blueprint.config = config
    blueprint.config_orchestrator = FakeConfigOrchestrator(config)
    blueprint.harness_files = saved.harness_files
    blueprint.cwd = tmp_path
    blueprint.build.side_effect = lambda **kwargs: build_test_agent_loop(
        config=config, **kwargs
    )
    monkeypatch.setattr(
        process, "build_root_blueprint", AsyncMock(return_value=blueprint)
    )
    request = runtime.RootOpenRequest(
        options=SessionOptions(cwd=str(tmp_path)),
        client_info=ClientInfo(name="tests", version="1.0"),
        session_id=session_id,
    )

    root = await process.open_root(request)
    try:
        assert root.session_id == session_id
        accounting = process.last_open_accounting
        assert accounting is not None
        assert accounting.root_session_id == session_id
        assert accounting.adoption == ADOPTION_NONE
        assert accounting.settlements == ()
        # The resumed session's lease belongs to the root that adopted it, so a
        # competing acquisition is refused typed.
        with pytest.raises(SessionBusyError):
            SessionLease(tmp_path, session_id).acquire()
    finally:
        await close_agent_loop(root)


@pytest.mark.asyncio
async def test_a_fresh_root_open_carries_the_accounting_it_adopted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = build_test_vibe_config(session_logging=SessionLoggingConfig(enabled=False))
    process = runtime.HarnessProcess()
    monkeypatch.setattr(
        runtime,
        "build_default_orchestrator",
        AsyncMock(return_value=FakeConfigOrchestrator(config)),
    )
    monkeypatch.setattr(
        process, "_build_mcp_registry_impl", AsyncMock(return_value=FakeMCPRegistry())
    )
    real_agent_loop = runtime.AgentLoop

    def build_loop(**kwargs: Any) -> AgentLoop:
        return real_agent_loop(backend=FakeBackend(), **kwargs)

    monkeypatch.setattr(runtime, "AgentLoop", build_loop)
    context, _owner = await process.allocate_startup_accounting(tmp_path)
    record = UsageRecord(
        **context.attribution(config.get_active_model()).model_dump(),
        record_id="usage-fixture-1",
        occurred_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        prices_usd_per_million=UsagePrices(),
        known_cost_usd=0.0,
        has_unknown_cost=True,
        outcome=UsageOutcome.FAILED,
        usage_state=UsageState.MISSING,
    )
    settlement = UsageWriteResult(disposition=UsageWriteDisposition.FAILED)
    context.on_early_settlement(record, settlement)

    root = await process.open_root(
        runtime.RootOpenRequest(
            options=SessionOptions(cwd=str(tmp_path), headless=True),
            client_info=ClientInfo(name="tests", version="1.0"),
            startup_accounting=context,
        )
    )
    try:
        accounting = process.last_open_accounting
        assert accounting is not None
        assert accounting.root_session_id == context.identity.root_session_id
        assert accounting.adoption == ADOPTION_ADOPTED
        # A cold ledger scan cannot recover a warning from a failed early
        # append: the content-free settlement crosses as a record.
        assert accounting.settlements == ((record, settlement),)
        assert accounting.warnings == ()
        assert context.state == "adopted"
        carried = decode_open_accounting(
            json.loads(json.dumps(encode_open_accounting(accounting)))
        )
        assert carried == accounting
    finally:
        await close_agent_loop(root)
        await process.close()


_OPEN_REPLY_SCRIPT = """\
import asyncio
import json
from pathlib import Path

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config


class _Factory:
    def __init__(self, loop, params_path):
        self._loop = loop
        self._params_path = Path(params_path)

    async def open(self, params):
        self._params_path.write_text(json.dumps(dict(params)), encoding="utf-8")
        await self._loop.persist_empty_session()
        return self._loop

    async def close(self):
        pass


init_harness_files_manager("user", "project")
config = build_test_vibe_config(
    session_logging=SessionLoggingConfig(enabled=True, save_dir={save_dir!r})
)
loop = build_test_agent_loop(config=config)
asyncio.run(
    serve_root_agent(
        factory=_Factory(loop, {params_path!r}), announce_ready=bootstrap_requested()
    )
)
"""


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_resume_parameters_and_open_reply_cross_the_spawned_pipe(
    tmp_path: Path,
) -> None:
    params_path = tmp_path / "open-params.json"
    script = _OPEN_REPLY_SCRIPT.format(
        save_dir=str(tmp_path / "sessions"), params_path=str(params_path)
    )
    with egress.bootstrap_directory() as guard_dir:
        process = await spawn_root_agent_process(
            session_options=AgentConfig(),
            client_info=ClientInfo(name="tests", version="1.0"),
            env=_child_env(guard_dir),
            cwd=str(_REPO_ROOT),
            command=[sys.executable, "-c", script],
            session_id="session-fixture",
            ready_timeout=30,
            request_timeout=60,
        )
        try:
            open_request_id = process.last_request_id
            assert open_request_id is not None
            snapshot = await process.snapshot()
            log = await process.read_session_log()
        finally:
            await process.close()

    params = json.loads(params_path.read_text(encoding="utf-8"))
    deadline = params["deadline"]
    assert params["session_id"] == "session-fixture"
    assert "continue_latest" not in params
    assert isinstance(deadline, float)
    assert deadline > time.monotonic()

    # The reply published the directories the agent process created for itself.
    assert process.session_dir is not None
    assert Path(process.session_dir).is_dir()
    assert process.jobs_dir == str(Path(process.session_dir, "jobs"))
    accounting = process.open_accounting
    assert accounting is not None
    assert accounting.adoption == ADOPTION_NONE

    # S9-C1: the admission is keyed by the coordinator's own request id.
    assert open_request_id.startswith("coordinator:")
    operation = process.admitted_operation(open_request_id)
    assert operation is not None
    assert operation.request_id == open_request_id
    assert operation.method == AGENT_OPEN_METHOD
    assert operation.deadline == deadline
    record = snapshot.marker_for(open_request_id)
    assert record is not None
    assert record.state == OP_STATE_TERMINAL
    assert record.outcome == OP_OUTCOME_COMPLETED
    assert snapshot.session_dir == process.session_dir
    assert log.path == process.session_dir


# ---------------------------------------------------------------------------
# Op 5: compaction's two-sided deadline and minter-split settlement.
#
# The agent mints the publication class -- published-and-settled,
# published-but-unsettled, cancelled-unpublished -- and answers its request id
# with that receipt, including when a cancel or the deadline aborted the run. A
# coordinator timeout is the coordinator's own observation, settled by ONE
# marker re-check and resolved by any later snapshot. A compaction is never
# retried, and the cancel is a notification with no reply of its own.
# ---------------------------------------------------------------------------


class _LiveProcess:
    """A process handle the test keeps alive: the pipe under it is a memory pair."""

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class _SilentTransport:
    """A transport whose message stream never ends, for a handle with no agent."""

    def __init__(self) -> None:
        self._never = asyncio.Event()

    async def send(self, message: dict[str, Any]) -> None:
        pass

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        await self._never.wait()
        if False:  # pragma: no cover - an async generator must contain a yield
            yield {}

    async def close(self) -> None:
        self._never.set()


class _LossyTransport:
    """A transport double that drops the frames a predicate names."""

    def __init__(
        self, inner: JsonRpcTransport, drop: Callable[[dict[str, Any], bool], bool]
    ) -> None:
        self._inner = inner
        self._drop = drop
        self.sent: list[dict[str, Any]] = []

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(dict(message))
        if self._drop(message, True):
            return
        await self._inner.send(message)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        async for frame in self._inner.messages():
            if not self._drop(frame, False):
                yield frame

    async def close(self) -> None:
        await self._inner.close()

    def requests(self, method: str) -> list[dict[str, Any]]:
        """Every request frame this end sent for one method."""
        return [
            frame
            for frame in self.sent
            if frame.get("method") == method and "id" in frame
        ]


def _drop_replies(
    *,
    compact: int = 0,
    snapshot: int = 0,
    replace: int = 0,
    reconfigure: int = 0,
    request: bool = False,
) -> Callable[[dict[str, Any], bool], bool]:
    """A drop predicate that loses the first N replies, or the compact request."""
    method_of: dict[object, str] = {}
    remaining = {
        AGENT_COMPACT_METHOD: compact,
        AGENT_SESSION_SNAPSHOT_METHOD: snapshot,
        AGENT_SESSION_REPLACE_METHOD: replace,
        AGENT_SESSION_RECONFIGURE_METHOD: reconfigure,
    }

    def drop(frame: dict[str, Any], outbound: bool) -> bool:
        method = frame.get("method")
        if outbound:
            if request and method == AGENT_COMPACT_METHOD:
                return True
            if isinstance(method, str) and "id" in frame:
                method_of[frame["id"]] = method
            return False
        named = method_of.get(frame.get("id"))
        if named is not None and remaining.get(named, 0) > 0:
            remaining[named] -= 1
            return True
        return False

    return drop


@asynccontextmanager
async def _coordinated_agent(
    tmp_path: Path,
    *,
    compact_deadline: float = 5.0,
    compact_reply_grace: float = 0.05,
    replace_deadline: float = 5.0,
    replace_reply_grace: float = 0.05,
    handoff_deadline: float = 5.0,
    handoff_reply_grace: float = 0.05,
    request_timeout: float = 30.0,
    drop: Callable[[dict[str, Any], bool], bool] | None = None,
    rebind: Callable[..., Any] | None = None,
    loop: AgentLoop | None = None,
    capabilities: frozenset[str] = ADVERTISED_CAPABILITIES,
) -> AsyncIterator[tuple[RootAgentProcess, _LossyTransport, AgentLoop]]:
    """A coordinator handle and its agent, wired over one in-memory pair.

    ``capabilities`` is what the child's ready frame announces; pass an empty
    set to simulate an older same-version peer that lacks the additive
    agent-wire capabilities.
    """
    opened = loop if loop is not None else await _persisted_test_loop(tmp_path)
    client, server = memory_transport_pair()
    transport = _LossyTransport(client, drop or (lambda frame, outbound: False))
    factory = _LoopFactory(opened)
    if rebind is not None:
        factory.replace = rebind  # type: ignore[attr-defined]
    serving = asyncio.create_task(
        root_agent.RootAgentServer(server, factory=factory).serve()
    )
    spawned = SpawnedProcess(
        process=cast("subprocess.Popen[bytes]", _LiveProcess()),
        transport=cast("Any", transport),
        ready=ReadyFrame(
            protocol_version=PROCESS_PROTOCOL_VERSION, capabilities=capabilities
        ),
    )
    process = RootAgentProcess(
        spawned,
        heartbeat_interval=30.0,
        request_timeout=request_timeout,
        compact_deadline=compact_deadline,
        compact_reply_grace=compact_reply_grace,
        replace_deadline=replace_deadline,
        replace_reply_grace=replace_reply_grace,
        handoff_deadline=handoff_deadline,
        handoff_reply_grace=handoff_reply_grace,
    )
    try:
        await process.open(
            session_options=AgentConfig(),
            client_info=ClientInfo(name="tests", version="1.0"),
        )
        yield process, transport, opened
    finally:
        await process.close()
        with suppress(BaseException):
            await asyncio.wait_for(serving, timeout=10)


async def _resolved_settlement(
    process: RootAgentProcess, request_id: str, *, timeout: float = 10.0
) -> CompactSettlementRecord:
    """Read snapshots until the settlement on this request id is terminal."""
    limit = time.monotonic() + timeout
    while time.monotonic() < limit:
        await process.snapshot(include_history=False)
        record = process.compaction_settlement(request_id)
        if record is not None and not record.pending:
            return record
        await asyncio.sleep(0.01)
    raise AssertionError(f"the settlement on {request_id} never resolved")


def _compact_params(
    *, deadline: float | None, instructions: str = ""
) -> dict[str, Any]:
    params: dict[str, Any] = {"extra_instructions": instructions}
    if deadline is not None:
        params["deadline"] = deadline
    return params


@pytest.mark.asyncio
async def test_a_compaction_answers_with_its_publication_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    seen: list[tuple[str, bool]] = []

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        seen.append((extra, manual))
        return "SUMMARY"

    monkeypatch.setattr(loop, "compact", compact)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    deadline = time.monotonic() + 60.0
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        result = await peer.call(
            AGENT_COMPACT_METHOD,
            _compact_params(deadline=deadline, instructions="focus"),
        )
        admission = await asyncio.wait_for(peer.notifications.get(), 15)
        snapshot = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    receipt = decode_compact_reply(result)
    assert seen == [("focus", True)]
    assert receipt.settlement == COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    assert receipt.summary == "SUMMARY"
    assert receipt.detail == ""
    # The identity handoff travels with the receipt: the session it leaves
    # behind, its state, and its log projection.
    assert receipt.session_id == loop.session_id
    assert receipt.state is not None and receipt.state.session.id == loop.session_id
    assert receipt.session_log == project_session_log(loop)

    assert admission["method"] == AGENT_OP_ADMISSION_METHOD
    assert admission["params"]["method"] == AGENT_COMPACT_METHOD
    record = snapshot.marker_for(admission["params"]["request_id"])
    assert record is not None
    assert record.state == OP_STATE_TERMINAL
    assert record.outcome == COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    assert record.deadline == deadline


@pytest.mark.asyncio
async def test_a_cancel_before_publication_settles_cancelled_unpublished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    started = asyncio.Event()
    never = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        loop.compaction_outcome.published = False
        started.set()
        await never.wait()
        return "unreachable"

    monkeypatch.setattr(loop, "compact", compact)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        request_id = await peer.send(
            AGENT_COMPACT_METHOD, _compact_params(deadline=time.monotonic() + 60.0)
        )
        admission = await asyncio.wait_for(peer.notifications.get(), 15)
        op_id = admission["params"]["op_id"]
        await asyncio.wait_for(started.wait(), 5)
        # The cancel is an id-less, idempotent notification: no reply exists,
        # so the operation's own settlement is the only outcome there is.
        await peer.notify(AGENT_OP_CANCEL_METHOD, {"op_id": op_id})
        cancelled = await peer.reply(request_id)
        snapshot = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
        record = snapshot.marker_for(request_id)
        # A second cancel for the settled operation is a no-op, and the fence
        # is free again.
        await peer.notify(AGENT_OP_CANCEL_METHOD, {"op_id": op_id})
        never.set()
        second = decode_compact_reply(
            await peer.call(
                AGENT_COMPACT_METHOD, _compact_params(deadline=time.monotonic() + 60.0)
            )
        )
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    # The operation's own settlement is the receipt on its request id.
    assert cancelled.get("error") is None
    receipt = decode_compact_reply(cancelled["result"])
    assert receipt.settlement == COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
    assert "cancelled" in receipt.detail
    assert receipt.summary is None
    assert record is not None
    assert record.state == OP_STATE_TERMINAL
    assert record.outcome == COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
    assert second.settlement == COMPACT_SETTLEMENT_PUBLISHED_SETTLED


@pytest.mark.asyncio
async def test_a_cancel_after_publication_reports_the_published_class(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    published = asyncio.Event()
    never = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        loop.compaction_outcome.published = True
        published.set()
        await never.wait()
        return "unreachable"

    monkeypatch.setattr(loop, "compact", compact)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        request_id = await peer.send(
            AGENT_COMPACT_METHOD, _compact_params(deadline=time.monotonic() + 60.0)
        )
        admission = await asyncio.wait_for(peer.notifications.get(), 15)
        await asyncio.wait_for(published.wait(), 5)
        await peer.notify(
            AGENT_OP_CANCEL_METHOD, {"op_id": admission["params"]["op_id"]}
        )
        cancelled = await peer.reply(request_id)
        snapshot = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
        record = snapshot.marker_for(request_id)
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    # A published boundary is never reported as an unpublished cancel.
    assert cancelled.get("error") is None
    receipt = decode_compact_reply(cancelled["result"])
    assert receipt.settlement == COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED
    assert receipt.published is True
    assert "cancelled" in receipt.detail
    assert record is not None and record.outcome == receipt.settlement


@pytest.mark.asyncio
async def test_the_deadline_aborts_a_wedged_compaction_into_its_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    started = asyncio.Event()
    never = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        loop.compaction_outcome.published = False
        started.set()
        await never.wait()
        return "unreachable"

    monkeypatch.setattr(loop, "compact", compact)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        request_id = await peer.send(
            AGENT_COMPACT_METHOD, _compact_params(deadline=time.monotonic() + 0.15)
        )
        await asyncio.wait_for(started.wait(), 5)
        aborted = await peer.reply(request_id)
        snapshot = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
        record = snapshot.marker_for(request_id)
        # The abort freed the fence: a live compaction is admitted afterwards.
        never.set()
        second = decode_compact_reply(
            await peer.call(
                AGENT_COMPACT_METHOD, _compact_params(deadline=time.monotonic() + 60.0)
            )
        )
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    # The abort answers the request id with its settlement receipt, and names
    # which interruption it was.
    assert aborted.get("error") is None
    receipt = decode_compact_reply(aborted["result"])
    assert receipt.settlement == COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
    assert "absolute deadline" in receipt.detail
    assert record is not None and record.outcome == receipt.settlement
    assert second.settlement == COMPACT_SETTLEMENT_PUBLISHED_SETTLED


@pytest.mark.asyncio
async def test_a_non_cooperative_body_invents_no_settlement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        loop.compaction_outcome.published = False
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The pinned residual: a body that refuses the abort point cannot
            # be settled by the deadline, so nothing may be claimed for it.
            await release.wait()
        return "LATE"

    monkeypatch.setattr(loop, "compact", compact)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        request_id = await peer.send(
            AGENT_COMPACT_METHOD, _compact_params(deadline=time.monotonic() + 0.1)
        )
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.sleep(0.25)
        held = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
        # The fence is still held: the residual is surfaced, not hidden.
        conflict = await peer.frame(
            AGENT_COMPACT_METHOD, _compact_params(deadline=time.monotonic() + 60.0)
        )
        release.set()
        late = await peer.reply(request_id)
        settled = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    active = held.marker_for(request_id)
    assert active is not None
    assert active.state == OP_STATE_ACTIVE
    assert active.outcome is None
    assert conflict["error"]["code"] == "OperationConflictError"
    # The body finished on its own terms, so its own class settles then -- and
    # only then.
    assert late.get("error") is None
    assert decode_compact_reply(late["result"]).settlement == (
        COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    )
    terminal = settled.marker_for(request_id)
    assert terminal is not None
    assert terminal.outcome == COMPACT_SETTLEMENT_PUBLISHED_SETTLED


@pytest.mark.asyncio
async def test_a_compaction_past_its_deadline_never_starts_its_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    calls: list[str] = []

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        calls.append(extra)
        return "SUMMARY"

    monkeypatch.setattr(loop, "compact", compact)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        expired = await peer.frame(
            AGENT_COMPACT_METHOD, _compact_params(deadline=time.monotonic() - 1.0)
        )
        undated = await peer.frame(AGENT_COMPACT_METHOD, _compact_params(deadline=None))
        snapshot = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    assert expired["error"]["code"] == "OperationExpiredError"
    assert "deadline" in expired["error"]["message"]
    assert undated["error"]["code"] == UnsupportedAgentRequestError.__name__
    assert calls == []
    assert snapshot.markers == ()


@pytest.mark.asyncio
async def test_the_coordinator_settles_a_delivered_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def compact(extra: str = "", *, manual: bool = False) -> str:
        return "SUMMARY"

    async with _coordinated_agent(tmp_path, compact_deadline=5.0) as (
        process,
        transport,
        loop,
    ):
        monkeypatch.setattr(loop, "compact", compact)
        sent_at = time.monotonic()
        record = await process.compact("focus")

    assert COMPACT_DEADLINE == 900.0
    assert record.settlement == COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    assert record.pending is False
    assert record.reply is not None and record.reply.summary == "SUMMARY"
    assert process.compaction_settlement(record.request_id) == record
    assert process.pending_compactions() == ()

    (request,) = transport.requests(AGENT_COMPACT_METHOD)
    # The deadline is absolute and minted on the monotonic clock at send.
    assert 0.0 < request["params"]["deadline"] - sent_at <= 5.5
    assert request["params"]["extra_instructions"] == "focus"


@pytest.mark.asyncio
async def test_a_lost_receipt_is_settled_by_the_terminal_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with _coordinated_agent(
        tmp_path,
        compact_deadline=0.1,
        compact_reply_grace=0.4,
        drop=_drop_replies(compact=1),
    ) as (process, transport, loop):

        async def compact(extra: str = "", *, manual: bool = False) -> str:
            loop.compaction_outcome.published = True
            await asyncio.Event().wait()
            return "unreachable"

        monkeypatch.setattr(loop, "compact", compact)
        record = await process.compact()
        consumed = await process.snapshot(include_history=False)

    # ONE request, never retried: the timeout is a settlement, not a re-send.
    assert len(transport.requests(AGENT_COMPACT_METHOD)) == 1
    # The abort landed inside the skew-plus-lateness window, and its TERMINAL
    # record reports the published class instead of a false unpublished one.
    assert consumed.marker_for(record.request_id) is not None
    assert record.settlement == COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED
    assert record.outcome == COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED
    assert record.pending is False


@pytest.mark.asyncio
async def test_an_active_marker_settles_in_flight_and_resolves_later(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    async with _coordinated_agent(
        tmp_path,
        compact_deadline=0.15,
        compact_reply_grace=0.05,
        drop=_drop_replies(compact=1),
    ) as (process, transport, loop):

        async def compact(extra: str = "", *, manual: bool = False) -> str:
            loop.compaction_outcome.published = False
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            return "SUMMARY"

        monkeypatch.setattr(loop, "compact", compact)
        record = await process.compact()
        provisional = process.pending_compactions()
        release.set()
        resolved = await _resolved_settlement(process, record.request_id)
        operation = process.admitted_operation(record.request_id)
    assert len(transport.requests(AGENT_COMPACT_METHOD)) == 1
    # ACTIVE evidence never claims publication: the label is provisional and
    # the op id it names is the one the admission notification minted.
    assert record.settlement == COMPACT_OBSERVATION_IN_FLIGHT
    assert record.pending is True
    assert record.published is False
    assert operation is not None and record.op_id == operation.op_id
    assert provisional == (record,)
    assert resolved.settlement == COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    assert resolved.pending is False


@pytest.mark.asyncio
async def test_a_cancel_is_pending_until_the_operation_settles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = asyncio.Event()
    async with _coordinated_agent(
        tmp_path,
        compact_deadline=0.15,
        compact_reply_grace=0.05,
        drop=_drop_replies(compact=1),
    ) as (process, transport, loop):

        async def compact(extra: str = "", *, manual: bool = False) -> str:
            loop.compaction_outcome.published = False
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # The deadline cannot settle this body; the cancel does.
                await asyncio.Event().wait()
            return "unreachable"

        monkeypatch.setattr(loop, "compact", compact)
        record = await process.compact()
        await asyncio.wait_for(started.wait(), 5)
        assert record.op_id is not None
        await process.cancel_operation(record.op_id)
        pending = process.pending_compactions()
        unresolved = process.compaction_settlement(record.request_id)
        # A cancel that names no live operation is an idempotent no-op.
        await process.cancel_operation("inc-unknown:7")
        resolved = await _resolved_settlement(process, record.request_id)

    # The cancel is a notification: nothing id-bearing carries it.
    assert [
        frame
        for frame in transport.sent
        if frame.get("method") == AGENT_OP_CANCEL_METHOD and "id" in frame
    ] == []
    assert len(pending) == 1
    assert pending[0].cancel_pending is True
    assert unresolved is not None and unresolved.cancel_pending is True
    # The settlement ends the cancel-pending window, with the class the
    # publication evidence supports.
    assert resolved.settlement == COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
    assert resolved.cancel_pending is False
    assert process.pending_compactions() == ()


@pytest.mark.asyncio
async def test_a_never_admitted_compaction_is_proven_unpublished(
    tmp_path: Path,
) -> None:
    async with _coordinated_agent(
        tmp_path,
        compact_deadline=0.1,
        compact_reply_grace=0.02,
        drop=_drop_replies(request=True),
    ) as (process, transport, loop):
        record = await process.compact()

    # The request never reached the agent, so absence from the post-deadline
    # payload is the proof -- and the transport-class signature is recorded so
    # the user can tell the cases apart.
    assert len(transport.requests(AGENT_COMPACT_METHOD)) == 1
    assert record.settlement == COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
    assert "never admitted" in record.detail
    assert record.pending is False
    assert any(
        "no admission notification" in item.reason for item in process.diagnostics
    )


@pytest.mark.asyncio
async def test_a_failed_recheck_settles_unknown_and_a_later_read_clears_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with _coordinated_agent(
        tmp_path,
        compact_deadline=0.1,
        compact_reply_grace=0.02,
        request_timeout=0.1,
        drop=_drop_replies(compact=1, snapshot=1),
    ) as (process, transport, loop):

        async def compact(extra: str = "", *, manual: bool = False) -> str:
            return "SUMMARY"

        monkeypatch.setattr(loop, "compact", compact)
        unknown = await process.compact()
        cleared = await _resolved_settlement(process, unknown.request_id)

    assert len(transport.requests(AGENT_COMPACT_METHOD)) == 1
    assert unknown.settlement == COMPACT_OBSERVATION_SETTLEMENT_UNKNOWN
    assert unknown.pending is True
    assert "re-check failed" in unknown.detail
    # Any later successful snapshot read clears it.
    assert cleared.settlement == COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    assert cleared.pending is False


@pytest.mark.asyncio
async def test_a_stale_payload_is_never_absence_evidence() -> None:
    spawned = SpawnedProcess(
        process=cast("subprocess.Popen[bytes]", _LiveProcess()),
        transport=cast("Any", _SilentTransport()),
        ready=ReadyFrame(
            protocol_version=PROCESS_PROTOCOL_VERSION,
            capabilities=ADVERTISED_CAPABILITIES,
        ),
    )
    process = RootAgentProcess(spawned, heartbeat_interval=30.0)
    try:
        verdict = MarkerVerdict(
            state=MARKER_STALE,
            detail="the payload was constructed before the request's deadline",
        )
        # The verdict mapping is what decides: a payload that predates the
        # deadline proves nothing, so the settlement stays open and unknown.
        record = process._settlement_from_verdict("coordinator:1", verdict)
    finally:
        await process.close()

    assert record.settlement == COMPACT_OBSERVATION_SETTLEMENT_UNKNOWN
    assert record.pending is True
    assert record.published is False


def test_the_compaction_deadline_is_the_sized_wire_budget() -> None:
    # S4-W4: the long-op wire deadline is sized to the 720s in-process budget.
    assert COMPACT_DEADLINE == 900.0
    assert COMPACT_DEADLINE > 720.0


# ---------------------------------------------------------------------------
# Op 10: the session-identity family. A replace rebinds the session the agent
# owns (a long operation with an absolute deadline) and seeds the new
# incarnation's marker table with its own TERMINAL record; a rename applies one
# manual title where the metadata lives; and the operations this release does
# not carry -- fork, relocate and the rewind pair -- answer typed and name the
# in_process escape. The coordinator's own unresolved state refuses a replace
# BEFORE any frame is sent, and a lost reply is settled by ONE marker re-check.
# ---------------------------------------------------------------------------


def _replacement_params(
    *, deadline: float, session_id: str = "session-target"
) -> dict[str, Any]:
    return encode_replace_request(
        session_id=session_id, continue_latest=False, deadline=deadline
    )


@pytest.mark.asyncio
async def test_the_deferred_identity_siblings_name_the_in_process_escape(
    tmp_path: Path,
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        refusals = [
            await peer.frame(method, {})
            for method in (
                AGENT_SESSION_FORK_METHOD,
                AGENT_SESSION_RELOCATE_METHOD,
                AGENT_SESSION_REWIND_READ_METHOD,
                AGENT_SESSION_REWIND_METHOD,
            )
        ]
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    for refusal in refusals:
        error = refusal["error"]
        assert error["code"] == UnsupportedAgentRequestError.__name__
        assert f"{TRANSPORT_ENV_VAR}=in_process" in error["message"]


@pytest.mark.asyncio
async def test_a_replace_rebinds_the_session_and_seeds_its_marker(
    tmp_path: Path,
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    factory = _LoopFactory(loop)
    rebound: list[tuple[str | None, bool]] = []

    async def replace(
        target: AgentLoop, *, session_id: str | None, continue_latest: bool = False
    ) -> None:
        rebound.append((session_id, continue_latest))
        target.session_id = "session-target"

    factory.replace = replace  # type: ignore[attr-defined]
    peer, serving = _serve_in_process(factory)
    deadline = time.monotonic() + 60.0
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        before = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
        request_id = await peer.send(
            AGENT_SESSION_REPLACE_METHOD, _replacement_params(deadline=deadline)
        )
        result = (await peer.reply(request_id))["result"]
        after = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    reply = decode_replace_reply(result)
    assert rebound == [("session-target", False)]
    assert reply.snapshot.state.session.id == "session-target"
    # The replacement mints a fresh incarnation whose table holds EXACTLY its
    # own TERMINAL record, and that seed keeps the op id the old incarnation
    # minted -- which is what makes the one-shot re-check find its evidence.
    assert reply.snapshot.incarnation_id != before.incarnation_id
    assert [marker.request_id for marker in reply.snapshot.markers] == [request_id]
    seed = reply.snapshot.markers[0]
    assert seed.state == OP_STATE_TERMINAL
    assert seed.outcome == OP_OUTCOME_COMPLETED
    assert seed.op_id.startswith(f"{before.incarnation_id}:")
    # Records are never pruned: the seed is still there on the next read, and
    # the new incarnation's own op ids come from its own counter.
    assert after.incarnation_id == reply.snapshot.incarnation_id
    assert after.markers == reply.snapshot.markers


@pytest.mark.asyncio
async def test_a_rename_applies_the_manual_title_where_the_metadata_lives(
    tmp_path: Path,
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        result = await peer.call(AGENT_SESSION_RENAME_METHOD, {"title": "Wire rename"})
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    assert result["title"] == "Wire rename"
    assert loop.session_logger.title == "Wire rename"
    assert loop.session_logger.title_source == "manual"


@pytest.mark.asyncio
async def test_a_replace_is_refused_before_send_while_a_settlement_is_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = asyncio.Event()
    async with _coordinated_agent(
        tmp_path,
        compact_deadline=0.15,
        compact_reply_grace=0.05,
        drop=_drop_replies(compact=1),
    ) as (process, transport, loop):

        async def compact(extra: str = "", *, manual: bool = False) -> str:
            loop.compaction_outcome.published = False
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            return "SUMMARY"

        monkeypatch.setattr(loop, "compact", compact)
        record = await process.compact()
        # The re-check saw an ACTIVE record: the settlement is provisionally
        # in flight and the replacement must refuse before it sends anything.
        assert record.pending is True
        with pytest.raises(RootAgentReplaceConflict, match="still unsettled"):
            await process.replace(session_id="session-target")
        release.set()
        await _resolved_settlement(process, record.request_id)

    assert transport.requests(AGENT_SESSION_REPLACE_METHOD) == []


@pytest.mark.asyncio
async def test_a_second_replace_is_refused_before_send_while_one_is_unresolved(
    tmp_path: Path,
) -> None:
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def replace(
        target: AgentLoop, *, session_id: str | None, continue_latest: bool = False
    ) -> None:
        entered.set()
        await gate.wait()
        target.session_id = session_id or target.session_id

    async with _coordinated_agent(tmp_path, rebind=replace) as (process, transport, _):
        first = asyncio.create_task(process.replace(session_id="session-target"))
        await asyncio.wait_for(entered.wait(), timeout=10)
        # The first replacement has not resolved, so the second never leaves
        # this end: its seed evidence would race the first one's.
        with pytest.raises(RootAgentReplaceConflict, match="still unresolved"):
            await process.replace(session_id="session-other")
        gate.set()
        reply = await asyncio.wait_for(first, timeout=10)

    assert len(transport.requests(AGENT_SESSION_REPLACE_METHOD)) == 1
    assert reply.snapshot.state.session.id == "session-target"


@pytest.mark.asyncio
async def test_a_lost_replace_reply_is_settled_by_its_seeded_marker(
    tmp_path: Path,
) -> None:
    async def replace(
        target: AgentLoop, *, session_id: str | None, continue_latest: bool = False
    ) -> None:
        target.session_id = session_id or target.session_id

    async with _coordinated_agent(
        tmp_path,
        replace_deadline=0.15,
        replace_reply_grace=0.05,
        drop=_drop_replies(replace=1),
        rebind=replace,
    ) as (process, transport, _):
        reply = await process.replace(session_id="session-target")

    # ONE re-check, never a resend: the seed proves the rebind landed, and the
    # payload the re-check read is the receipt.
    assert len(transport.requests(AGENT_SESSION_REPLACE_METHOD)) == 1
    assert reply.snapshot.state.session.id == "session-target"
    assert reply.accounting is None
    assert process.session_id == "session-target"


@pytest.mark.asyncio
async def test_a_replace_answers_a_typed_not_found(tmp_path: Path) -> None:
    async def replace(
        target: AgentLoop, *, session_id: str | None, continue_latest: bool = False
    ) -> None:
        raise root_agent.AgentSessionNotFoundError(f"Session not found: {session_id}")

    async with _coordinated_agent(tmp_path, rebind=replace) as (process, _, _):
        with pytest.raises(RootAgentSessionNotFound, match="Session not found"):
            await process.replace(session_id="session-missing")


def test_the_replacement_deadline_is_the_sized_wire_budget() -> None:
    # S4-W4: transcript load and MCP registry construction, agent-side.
    assert REPLACE_DEADLINE == 300.0


# ---------------------------------------------------------------------------
# Item 1 op 10 (package 22): the reconfigure modes.
#
# One reconfigure request carries ONE invocation mode. A short mode mutates the
# session in place and the agent refuses it typed while a long operation holds
# the fence. The clear acks success only after checking, against LIVE runtime
# state, that the history and the session model pin are gone, and its receipt
# names exactly the steps this call completed so a retry re-enters and finishes
# the rest. The plan handoff is the full consent transaction admitted as a long
# operation: its receipt carries the fence outcome, the episode it committed,
# and the accounting the coordinator replays.
# ---------------------------------------------------------------------------

_WIRE_EPISODE_ID = "ep-wire0001"
_WIRE_DOCUMENT = b"# Plan\r\n\r\n## Summary\r\nCarry these exact bytes.\r\n"


async def _layered_logging_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, pin: bool = False
) -> AgentLoop:
    """A persisted loop on a real orchestrator with a session overrides layer.

    The verbatim fake holds no overrides layer, so ``override_active_model``
    would never see a session pin; the clear drives its pin removal from that
    LIVE state, so it needs the real layered orchestrator to land.
    """
    save_dir = tmp_path / "layered-sessions"
    config = build_test_vibe_config(
        active_model="alpha",
        models=[
            ModelConfig(name="alpha", provider="mistral", alias="alpha"),
            ModelConfig(name="beta", provider="mistral", alias="beta"),
        ],
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(save_dir)),
    )

    async def load(
        cfg: ChartreuxConfigSchema,
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        user_path = tmp_path / "layered-user.toml"
        user_path.write_text('theme = "dark"\n')
        user = UserConfigLayer(path=user_path)
        launch = OverridesLayer(
            name="launch",
            data={
                "active_model": "alpha",
                "session_logging": cfg.session_logging.model_dump(mode="json"),
            },
        )
        session_data: dict[str, Any] = {"active_model": "alpha"} if pin else {}
        return await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[user, launch, OverridesLayer(data=session_data)],
            default_layer_resolver=lambda: user,
            catalog_snapshot=cfg.catalog_snapshot,
        )

    monkeypatch.setattr("tests.conftest._load_orchestrator", load)
    loop = build_test_agent_loop(config=config, session_id="initial")
    await loop.persist_empty_session()
    return loop


def _wire_episode(tmp_path: Path) -> Path:
    """A plan episode directory whose draft carries a usable plan document."""
    directory = tmp_path / "plan-wire-episode"
    directory.mkdir()
    ensure_episode_fence(directory)
    create_sidecar(
        directory,
        PlanEpisodeSidecar(
            episode_id=_WIRE_EPISODE_ID, draft_path=directory / "plan.md"
        ),
    )
    (directory / "plan.md").write_bytes(_WIRE_DOCUMENT)
    return directory


def _wire_plan_record(directory: Path, save_dir: Path) -> PlanModeRecord:
    """The planning session's own record for the episode under test."""
    return PlanModeRecord(
        plan_path=str(directory / "plan.md"),
        theme="Carry these exact bytes",
        planner_model="alpha",
        planner_thinking="medium",
        episode_id=_WIRE_EPISODE_ID,
        sidecar_path=str(directory / ".chartreux-plan-episode.json"),
        draft_path=str(directory / "plan.md"),
        state=STATE_DRAFT,
        scan_domain=PlanScanDomain(save_dir=str(save_dir), session_prefix="session"),
    )


def _wire_handoff_params(directory: Path, session_id: str) -> SessionPlanHandoffParams:
    return SessionPlanHandoffParams(
        session_id=session_id,
        executor_model="beta",
        executor_thinking="medium",
        executor_identity="orchestrator",
        plan_path=str(directory / "plan.md"),
        episode_id=_WIRE_EPISODE_ID,
        snapshot_path=str(accepted_snapshot_path(directory, _WIRE_EPISODE_ID)),
        snapshot_sha256=sha256_bytes(_WIRE_DOCUMENT),
        snapshot_bytes_b64=base64.b64encode(_WIRE_DOCUMENT).decode("ascii"),
    )


@pytest.mark.asyncio
async def test_a_settings_reconfigure_applies_the_bounds_and_acks(
    tmp_path: Path,
) -> None:
    async with _coordinated_agent(tmp_path) as (process, _transport, loop):
        reply = await process.update_settings(max_turns=13, max_tokens=987654)

    assert reply.mode == RECONFIGURE_MODE_SETTINGS
    assert reply.outcome == RECONFIGURE_OUTCOME_APPLIED
    assert (reply.max_turns, reply.max_tokens) == (13, 987654)
    # A bounds update changes no session identity, so there is nothing to adopt.
    assert reply.snapshot is None
    assert loop._max_turns == 13
    assert loop._max_tokens == 987654


@pytest.mark.asyncio
async def test_a_short_mode_is_refused_before_send_while_a_long_operation_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        entered.set()
        await gate.wait()
        return "SUMMARY"

    async with _coordinated_agent(tmp_path) as (process, transport, loop):
        monkeypatch.setattr(loop, "compact", compact)
        running = asyncio.create_task(process.compact())
        await asyncio.wait_for(entered.wait(), timeout=10)
        with pytest.raises(RootAgentReconfigureConflict, match="is in flight"):
            await process.clear_history()
        # Refused on THIS end: no reconfigure frame was minted for the fence at
        # the far end to refuse.
        assert transport.requests(AGENT_SESSION_RECONFIGURE_METHOD) == []
        gate.set()
        await asyncio.wait_for(running, timeout=10)
        reply = await process.clear_history()

    assert reply.outcome == RECONFIGURE_OUTCOME_CLEARED
    assert reply.stage == CLEAR_STAGE_ALREADY_CLEAR
    assert len(transport.requests(AGENT_SESSION_RECONFIGURE_METHOD)) == 1


@pytest.mark.asyncio
async def test_the_agent_refuses_a_short_mode_while_its_fence_is_held(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _persisted_test_loop(tmp_path)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        entered.set()
        await gate.wait()
        return "SUMMARY"

    monkeypatch.setattr(loop, "compact", compact)
    peer, serving = _serve_in_process(_LoopFactory(loop))
    try:
        opened = await peer.call(AGENT_OPEN_METHOD, _open_params())
        session_id = cast(str, opened["session_id"])
        compact_id = await peer.send(
            AGENT_COMPACT_METHOD, _compact_params(deadline=time.monotonic() + 60.0)
        )
        admission = await asyncio.wait_for(peer.notifications.get(), 15)
        await asyncio.wait_for(entered.wait(), 15)
        settings = encode_reconfigure_request(
            mode=RECONFIGURE_MODE_SETTINGS, session_id=session_id, max_turns=3
        )
        refused = await peer.frame(AGENT_SESSION_RECONFIGURE_METHOD, settings)
        gate.set()
        assert (await peer.reply(compact_id)).get("error") is None
        accepted = await peer.call(AGENT_SESSION_RECONFIGURE_METHOD, settings)
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    # The refusal names the operation holding the fence, typed, and nothing ran.
    error = refused["error"]
    assert error["code"] == RECONFIGURE_CONFLICT_ERROR
    assert admission["params"]["op_id"] in error["message"]
    assert "is in flight" in error["message"]

    reply = decode_reconfigure_reply(accepted)
    assert reply.outcome == RECONFIGURE_OUTCOME_APPLIED
    assert loop._max_turns == 3


@pytest.mark.asyncio
async def test_a_partial_clear_names_its_stage_and_a_retry_finishes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_logging_loop(monkeypatch, tmp_path, pin=True)
    orch = loop.config_orchestrator
    assert override_active_model(orch) == "alpha"
    loop.messages.append(LLMMessage(role=Role.user, content="hello"))

    real_clear_override = agent_reconfigure.clear_session_active_model_override
    attempts: list[str] = []

    async def _refuse_once(*args: Any, **kwargs: Any) -> list[BaseException]:
        attempts.append("clear-override")
        if len(attempts) == 1:
            return [RuntimeError("the overrides layer refused the removal")]
        return await real_clear_override(*args, **kwargs)

    monkeypatch.setattr(
        agent_reconfigure, "clear_session_active_model_override", _refuse_once
    )

    async with _coordinated_agent(tmp_path, loop=loop) as (process, _transport, _):
        partial = await process.clear_history()
        # The history IS gone and the receipt says exactly that: reporting a
        # plain failure would invite a retry of the destructive step that ran.
        assert partial.outcome == RECONFIGURE_OUTCOME_CLEARED_WITH_FOLLOWUP_FAILURE
        assert partial.stage == CLEAR_STAGE_HISTORY
        assert "the overrides layer refused the removal" in partial.detail
        assert not [
            message for message in loop.messages if message.role is not Role.system
        ]

        # The retry re-enters and drives the pin removal from LIVE state, so it
        # finishes the step the metadata rewrite could not.
        retried = await process.clear_history()
        assert retried.outcome == RECONFIGURE_OUTCOME_CLEARED
        assert retried.stage == CLEAR_STAGE_MODEL_OVERRIDE
        assert override_active_model(orch) is None

        # Nothing is left to clear, and the terminal check says so instead of
        # re-running the destructive step.
        already = await process.clear_history()
        assert already.outcome == RECONFIGURE_OUTCOME_CLEARED
        assert already.stage == CLEAR_STAGE_ALREADY_CLEAR

    assert attempts == ["clear-override", "clear-override"]


@pytest.mark.asyncio
async def test_a_reconfigure_handoff_runs_the_fence_and_answers_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = _wire_episode(tmp_path)
    loop = await _layered_logging_loop(monkeypatch, tmp_path)
    await loop.session_logger.persist_plan_mode_durable(
        _wire_plan_record(directory, tmp_path / "layered-sessions")
    )
    params = _wire_handoff_params(directory, loop.session_id)
    source_session_id = loop.session_id
    peer, serving = _serve_in_process(_LoopFactory(loop))
    deadline = time.monotonic() + 60.0
    try:
        await peer.call(AGENT_OPEN_METHOD, _open_params())
        request_id = await peer.send(
            AGENT_SESSION_RECONFIGURE_METHOD,
            encode_reconfigure_request(
                mode=RECONFIGURE_MODE_PLAN_HANDOFF, deadline=deadline, handoff=params
            ),
        )
        admission = await asyncio.wait_for(peer.notifications.get(), 15)
        result = (await peer.reply(request_id))["result"]
        snapshot = decode_session_snapshot(
            await peer.call(AGENT_SESSION_SNAPSHOT_METHOD, {})
        )
    finally:
        await peer.close()
        with suppress(BaseException):
            await serving

    # The admission is announced under the coordinator's own request id before
    # the reply is written, so a lost answer is still correlatable.
    assert admission["method"] == AGENT_OP_ADMISSION_METHOD
    assert admission["params"]["request_id"] == request_id
    assert admission["params"]["method"] == AGENT_SESSION_RECONFIGURE_METHOD
    assert admission["params"]["state"] == "active"

    reply = decode_reconfigure_reply(result)
    assert reply.outcome == RECONFIGURE_OUTCOME_ACCEPTED
    assert reply.episode_id == _WIRE_EPISODE_ID
    assert reply.attempt_generation == 1
    assert reply.accounting is not None
    assert reply.snapshot is not None

    record = snapshot.marker_for(request_id)
    assert record is not None
    assert record.state == OP_STATE_TERMINAL
    assert record.outcome == OP_OUTCOME_COMPLETED
    assert record.deadline == deadline

    # C5 landed: the episode is accepted, the executor is bound, and the live
    # snapshot carries the attempt's provenance name.
    sidecar = read_sidecar(directory)
    assert sidecar is not None
    assert sidecar.state == STATE_ACCEPTED
    assert sidecar.executor_session_id == loop.session_id
    assert loop.session_id != source_session_id
    assert (
        provenance_snapshot_path(directory, _WIRE_EPISODE_ID, 1).read_bytes()
        == _WIRE_DOCUMENT
    )
    assert not accepted_snapshot_path(directory, _WIRE_EPISODE_ID).exists()

    # The consented executor selection is what the live session now holds.
    assert override_active_model(loop.config_orchestrator) == "beta"
    assert loop._bound_identity == "orchestrator"


@pytest.mark.asyncio
async def test_a_lost_handoff_reply_is_settled_by_one_marker_recheck(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = _wire_episode(tmp_path)
    loop = await _layered_logging_loop(monkeypatch, tmp_path)
    await loop.session_logger.persist_plan_mode_durable(
        _wire_plan_record(directory, tmp_path / "layered-sessions")
    )
    params = _wire_handoff_params(directory, loop.session_id)

    async with _coordinated_agent(
        tmp_path,
        handoff_deadline=2.0,
        handoff_reply_grace=0.1,
        drop=_drop_replies(reconfigure=1),
        loop=loop,
    ) as (process, transport, _):
        reply = await process.plan_handoff(params)

    # ONE re-check, never a resend: the operation's own TERMINAL record proves
    # the fence reached its flip, and that payload is the receipt.
    assert len(transport.requests(AGENT_SESSION_RECONFIGURE_METHOD)) == 1
    assert reply.outcome == RECONFIGURE_OUTCOME_ACCEPTED
    assert reply.snapshot is not None
    # The re-check claims only what the payload proves; the reply it could not
    # read is not invented.
    assert reply.episode_id is None
    sidecar = read_sidecar(directory)
    assert sidecar is not None
    assert sidecar.state == STATE_ACCEPTED
    assert process.session_id == loop.session_id


class _ClearRecorder:
    """The process leg behind the coordinator's reconfigure leg."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def clear_history(self) -> Any:
        self.calls.append("clear")
        return "cleared-reply"


@pytest.mark.asyncio
async def test_a_refused_plan_settlement_ends_the_clear_before_any_frame() -> None:
    process = _ClearRecorder()

    async def settlement() -> str:
        return "the plan episode could not be cancelled"

    leg = ProcessRootReconfigure(
        process=cast("RootAgentProcess", process), plan_settlement=settlement
    )

    with pytest.raises(RequestFailure) as caught:
        await leg.clear_history()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    assert "Clear refused" in str(caught.value)
    assert "the plan episode could not be cancelled" in str(caught.value)
    assert "Nothing was cleared" in str(caught.value)
    assert "keeps its state" in str(caught.value)
    # The coordinator-owned settlement ran first, and its refusal means the
    # clear frame is never sent.
    assert process.calls == []


@pytest.mark.asyncio
async def test_an_allowed_plan_settlement_lets_the_clear_frame_through() -> None:
    process = _ClearRecorder()

    async def settlement() -> None:
        return None

    leg = ProcessRootReconfigure(
        process=cast("RootAgentProcess", process), plan_settlement=settlement
    )

    assert await leg.clear_history() == "cleared-reply"
    assert process.calls == ["clear"]


# ---------------------------------------------------------------------------
# Item 1 op 10 mode (d), package 23: the config write as a long operation.
#
# The candidate crosses the pipe, the agent runs the shipped prepare/commit
# transaction against it, and the reply carries the SHIPPED result wholesale --
# persistence x application, the rejected flag, the failure details, the
# revision, the saved values and the launch-metadata outcome -- plus the state
# the reply leaves behind. The transaction is bounded by the coordinator's
# absolute deadline, and its receipt is never collapsed into a smaller class.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_config_write_reconfigure_carries_its_shipped_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_logging_loop(monkeypatch, tmp_path)

    async with _coordinated_agent(tmp_path, loop=loop) as (process, transport, _):
        reply = await process.write_config(
            ConfigWriteParams(
                session_id=loop.session_id,
                ops=[ConfigWriteOpWire(op="set", path="/theme", value="github-dark")],
            )
        )

    receipt = reply.config_write
    assert reply.mode == RECONFIGURE_MODE_CONFIG_WRITE
    assert receipt is not None
    # The outcome IS the shipped runtime application dimension.
    assert reply.outcome == receipt.application == "applied"
    assert (receipt.persistence, receipt.application) == ("not_saved", "applied")
    assert receipt.rejected is False
    assert receipt.failures == []
    # A session write persists nothing, so the shipped response carries no
    # saved values -- and says so in the persistence dimension, not by hiding it.
    assert receipt.saved_values == {}
    assert receipt.fields[0].value == "github-dark"
    assert receipt.launch_metadata_persisted is True
    assert receipt.runtime.config.theme == "github-dark"
    assert loop.config.theme == "github-dark"
    # The reply carries the state it leaves behind, which a session write does
    # not rotate but the coordinator still adopts.
    assert reply.snapshot is not None
    assert reply.snapshot.state.session.id == loop.session_id
    assert reply.accounting is not None

    # ONE frame, sent as a long operation: its absolute deadline rides the
    # request, and the candidate crosses without the reload flag.
    frames = transport.requests(AGENT_SESSION_RECONFIGURE_METHOD)
    assert len(frames) == 1
    params = frames[0]["params"]
    assert params["mode"] == RECONFIGURE_MODE_CONFIG_WRITE
    assert isinstance(params["deadline"], float)
    assert "reload_runtime" not in params["config_write"]


def test_the_config_write_deadline_is_the_sized_wire_budget() -> None:
    from chartreux.app_server.root_agent_process import CONFIG_WRITE_DEADLINE

    assert CONFIG_WRITE_DEADLINE == 60.0


# ---------------------------------------------------------------------------
# Package A7a: handshake capability advertisement, version skew, the shared
# 30s ordinary-request default, and ready-read semantics on the stdio route.
# ---------------------------------------------------------------------------


def test_the_ready_frame_advertises_the_agent_wire_capabilities() -> None:
    frame = json.loads(encode_ready_frame().decode())
    assert frame["capabilities"] == sorted(ADVERTISED_CAPABILITIES)
    ready = parse_ready_frame(encode_ready_frame())
    assert ready.protocol_version == PROCESS_PROTOCOL_VERSION
    assert ready.capabilities == ADVERTISED_CAPABILITIES
    assert AGENT_WIRE_CAPABILITY_PLAN_HANDOFF_ROTATION in ready.capabilities


def test_an_older_same_version_child_announces_no_capabilities() -> None:
    raw = json.dumps({
        "chartreuxBootstrap": "ready",
        "protocolVersion": PROCESS_PROTOCOL_VERSION,
    })
    ready = parse_ready_frame(raw)
    assert ready.protocol_version == PROCESS_PROTOCOL_VERSION
    assert ready.capabilities == frozenset()


@pytest.mark.parametrize(
    "capabilities",
    ["plan-handoff-rotation", [1], [""], [None], {"plan-handoff-rotation": True}],
)
def test_a_malformed_capability_advertisement_is_a_missing_frame(
    capabilities: Any,
) -> None:
    raw = json.dumps({
        "chartreuxBootstrap": "ready",
        "protocolVersion": PROCESS_PROTOCOL_VERSION,
        "capabilities": capabilities,
    })
    with pytest.raises(ReadyFrameMissingError):
        parse_ready_frame(raw)


def test_a_different_protocol_version_is_a_typed_restart_error() -> None:
    with pytest.raises(ProtocolVersionMismatchError) as older:
        parse_ready_frame(encode_ready_frame(1))
    assert older.value.expected == PROCESS_PROTOCOL_VERSION
    assert older.value.received == 1
    assert "restart both ends on the same chartreux version" in str(older.value)
    assert "no traffic was sent" in str(older.value)

    with pytest.raises(ProtocolVersionMismatchError) as newer:
        parse_ready_frame(encode_ready_frame(3))
    assert newer.value.received == 3


def test_the_plan_handoff_reconfigure_is_the_only_capability_gated_request() -> None:
    assert required_agent_capabilities(
        AGENT_SESSION_RECONFIGURE_METHOD,
        encode_reconfigure_request(mode=RECONFIGURE_MODE_PLAN_HANDOFF, deadline=1.0),
    ) == frozenset({AGENT_WIRE_CAPABILITY_PLAN_HANDOFF_ROTATION})
    assert (
        required_agent_capabilities(
            AGENT_SESSION_RECONFIGURE_METHOD,
            encode_reconfigure_request(mode=RECONFIGURE_MODE_SETTINGS),
        )
        == frozenset()
    )
    assert required_agent_capabilities(AGENT_SESSION_RENAME_METHOD, {}) == frozenset()


def test_the_stdio_request_default_matches_the_shared_30_second_bound() -> None:
    # The recorded baseline was a private 60s on the stdio route; the shared
    # ordinary-request default is 30s on every leg of the route.
    assert PSB_REQUEST_TIMEOUT == 30.0
    assert ROOT_PORT_REQUEST_TIMEOUT == 30.0
    assert PSB_REQUEST_TIMEOUT == ROOT_PORT_REQUEST_TIMEOUT
    assert PSB_REQUEST_TIMEOUT != 60.0


@pytest.mark.asyncio
async def test_a_child_without_the_rotation_capability_is_a_typed_skew_not_a_send(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = _wire_episode(tmp_path)
    loop = await _layered_logging_loop(monkeypatch, tmp_path)
    await loop.session_logger.persist_plan_mode_durable(
        _wire_plan_record(directory, tmp_path / "layered-sessions")
    )
    params = _wire_handoff_params(directory, loop.session_id)

    async with _coordinated_agent(tmp_path, loop=loop, capabilities=frozenset()) as (
        process,
        transport,
        _loop,
    ):
        assert process.advertised_capabilities == frozenset()
        with pytest.raises(RootAgentCapabilitySkewError) as exc_info:
            await process.plan_handoff(params)
        error = exc_info.value
        assert error.method == AGENT_SESSION_RECONFIGURE_METHOD
        assert error.capabilities == (AGENT_WIRE_CAPABILITY_PLAN_HANDOFF_ROTATION,)
        assert AGENT_WIRE_CAPABILITY_PLAN_HANDOFF_ROTATION in str(error)
        assert "restart both ends on the same chartreux version" in str(error)
        assert "was not sent" in str(error)
        # No frame crossed the pipe: the skew is refused before any send, so
        # there is no stale fallback and nothing to replay.
        assert transport.requests(AGENT_SESSION_RECONFIGURE_METHOD) == []
        # The handle still serves requests that need no additive capability.
        snapshot = await process.snapshot(include_history=False)
        assert snapshot.state.session.id == loop.session_id


@pytest.mark.asyncio
async def test_ready_read_answers_ready_only_while_the_root_is_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.app_server.backend_contract.conftest import install_memory_root_agent
    from tests.stubs.app_server import attach_test_app_server_session

    loop = build_test_agent_loop()
    install_memory_root_agent(monkeypatch, loop)
    client_transport, server_transport = memory_transport_pair()
    harness = await runtime.create_harness_server(
        server_transport, transport_kind="stdio"
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    try:
        backend = cast(Any, harness._server._root)
        session_id = session.session_id
        # While the current generation is known live and open, the open
        # handshake established readiness: the read is immediately ready.
        read = await backend.dispatch_extension(
            "session/ready/read", {"sessionId": session_id}
        )
        assert read.response.ready is True
        wait = await backend.dispatch_extension(
            "session/ready/wait", {"sessionId": session_id}
        )
        assert wait.response.ready is True

        # After a declared death the read answers the death receipt, never a
        # stale ready from the dead generation.
        backend.process._declare_death(
            "the root agent process exited",
            signature=RootFailureSignature.DEATH_RECEIPT,
        )
        for method in ("session/ready/read", "session/ready/wait"):
            with pytest.raises(SessionBackendError) as exc_info:
                await backend.dispatch_extension(method, {"sessionId": session_id})
            assert "death-receipt" in str(exc_info.value)
    finally:
        with suppress(BaseException):
            await session.close()
        with suppress(BaseException):
            await client.close()
        with suppress(BaseException):
            await harness.close()
        await loop.aclose()


async def _runtime_read_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, Any, Any]:
    """One stdio harness over a memory-wire root; returns the closeable parts."""
    from tests.app_server.backend_contract.conftest import install_memory_root_agent
    from tests.stubs.app_server import attach_test_app_server_session

    loop = build_test_agent_loop()
    install_memory_root_agent(monkeypatch, loop)
    client_transport, server_transport = memory_transport_pair()
    harness = await runtime.create_harness_server(
        server_transport, transport_kind="stdio"
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    return harness, client, session, loop


@pytest.mark.asyncio
async def test_runtime_read_serves_a_fresh_agent_snapshot_while_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2a: runtime/read asks the agent for a fresh snapshot on every read.

    The answer is the agent's live view at reply-write time, not the
    coordinator's cached open-time copy: an agent-side change after the open
    is visible on the very next read, and the read is adopted with the shared
    agent-sequence cursor never older than the events already relayed.
    """
    harness, client, session, loop = await _runtime_read_harness(monkeypatch)
    try:
        backend = cast(Any, harness._server._root)
        session_id = session.session_id
        first = await backend.dispatch_extension(
            "runtime/read", {"sessionId": session_id}
        )
        assert first.response.ready is True
        # The adopted cursor is never older than the relayed events.
        assert backend._adopted_agent_sequence >= backend.process.last_agent_sequence

        # An agent-side change after the open: the next read must see it, so
        # the read is fresh per call rather than the cached open snapshot.
        monkeypatch.setattr(
            loop.tool_manager, "pop_mcp_errors", lambda: {"fresh-marker": "boom"}
        )
        second = await backend.dispatch_extension(
            "runtime/read", {"sessionId": session_id}
        )
        assert second.response.ready is True
        assert second.response.runtime.mcp.discovery_errors == {"fresh-marker": "boom"}
        assert second.response.runtime != first.response.runtime
        assert backend._adopted_agent_sequence >= backend.process.last_agent_sequence
    finally:
        with suppress(BaseException):
            await session.close()
        with suppress(BaseException):
            await client.close()
        with suppress(BaseException):
            await harness.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_runtime_read_after_root_death_answers_the_death_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2a: after a declared root death the read answers the death receipt.

    The stale-snapshot case: a dead root must not keep serving the cached
    runtime with ``ready=True``; the fresh-snapshot request fails typed and
    the death receipt propagates instead.
    """
    harness, client, session, loop = await _runtime_read_harness(monkeypatch)
    try:
        backend = cast(Any, harness._server._root)
        session_id = session.session_id
        live = await backend.dispatch_extension(
            "runtime/read", {"sessionId": session_id}
        )
        assert live.response.ready is True

        backend.process._declare_death(
            "the root agent process exited",
            signature=RootFailureSignature.DEATH_RECEIPT,
        )
        with pytest.raises(SessionBackendError) as exc_info:
            await backend.dispatch_extension("runtime/read", {"sessionId": session_id})
        assert "death-receipt" in str(exc_info.value)
    finally:
        with suppress(BaseException):
            await session.close()
        with suppress(BaseException):
            await client.close()
        with suppress(BaseException):
            await harness.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_agent_inventory_read_after_root_death_answers_the_death_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C6b: after a declared root death the inventory read answers the receipt.

    The live-inventory carrier must not keep serving a coordinator-side
    view of the children once the owner is gone: the fresh agent-wire
    request fails typed through the process failure, and the death receipt
    propagates exactly like the runtime/read arm's.
    """
    harness, client, session, loop = await _runtime_read_harness(monkeypatch)
    try:
        backend = cast("Any", harness._server._root)
        session_id = session.session_id
        live = await backend.dispatch_extension("agents/inventory/read", {})
        assert live.response.session_id == session_id
        assert live.response.revision == 0
        assert live.response.agents == []

        backend.process._declare_death(
            "the root agent process exited",
            signature=RootFailureSignature.DEATH_RECEIPT,
        )
        with pytest.raises(SessionBackendError) as exc_info:
            await backend.dispatch_extension("agents/inventory/read", {})
        assert "death-receipt" in str(exc_info.value)
    finally:
        with suppress(BaseException):
            await session.close()
        with suppress(BaseException):
            await client.close()
        with suppress(BaseException):
            await harness.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_session_turns_list_serves_the_settled_turn_over_a_spawned_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C5c: the spawned topology keeps the settled turn on its page.

    A really spawned root agent is attached through the stdio harness -- no
    memory-wire stand-in for the child -- and one turn is driven to
    completion. The agent builds no turn projection of its own, so the
    coordinator's records are the only copy: the page names the settled turn
    both before and after the runtime/read that adopts a fresh agent state.
    """
    from chartreux.app_server import process_session_backend as psb_module
    from chartreux.app_server.transport import STDIO_TRANSPORT
    from tests.app_server import executed_receipts
    from tests.stubs.app_server import attach_test_app_server_session

    monkeypatch.setenv(TRANSPORT_ENV_VAR, STDIO_TRANSPORT)
    executed_receipts.stamp_transport(STDIO_TRANSPORT)
    real_spawn = psb_module.spawn_root_agent_process
    command = _child_command(tmp_path)

    async def scripted_spawn(**kwargs: Any) -> RootAgentProcess:
        # The command override is the only change: the spawn, the pipe, the
        # handshake and every frame are the production ones.
        kwargs["command"] = command
        kwargs["cwd"] = str(_REPO_ROOT)
        return await real_spawn(**kwargs)

    monkeypatch.setattr(psb_module, "spawn_root_agent_process", scripted_spawn)
    client_transport, server_transport = memory_transport_pair()
    harness = await runtime.create_harness_server(
        server_transport, transport_kind=STDIO_TRANSPORT
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    try:
        _ = [event async for event in session.act("hello")]
        session_id = session.session_id

        settled = SessionTurnsListResponse.model_validate(
            await client.request(
                "session/turns/list", SessionTurnsListParams(session_id=session_id)
            )
        )
        assert [turn.status for turn in settled.items] == ["completed"]

        read = await client.request(
            "runtime/read", RuntimeReadParams(session_id=session_id)
        )
        assert read["ready"] is True

        # The read adopted a fresh agent-built state: the turn the spawned
        # root actually ran is still the page's own.
        after_read = SessionTurnsListResponse.model_validate(
            await client.request(
                "session/turns/list", SessionTurnsListParams(session_id=session_id)
            )
        )
        assert [turn.id for turn in after_read.items] == [
            turn.id for turn in settled.items
        ]
        assert [turn.status for turn in after_read.items] == ["completed"]
    finally:
        with suppress(BaseException):
            await session.close()
        with suppress(BaseException):
            await client.close()
        with suppress(BaseException):
            await harness.close()


@pytest.mark.asyncio
async def test_a_child_without_the_inventory_capability_is_a_typed_skew_not_a_send(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C6b: the inventory carrier is an additive capability, refused pre-send.

    An older same-version peer that did not announce the
    ``live-agent-inventory`` capability is a version skew: the coordinator
    answers typed with a restart instruction before any frame crosses, so
    there is no stale fallback for the read and nothing to replay.
    """
    async with _coordinated_agent(tmp_path, capabilities=frozenset()) as (
        process,
        transport,
        loop,
    ):
        assert process.advertised_capabilities == frozenset()
        with pytest.raises(RootAgentCapabilitySkewError) as exc_info:
            await process.read_agent_inventory(
                loop.session_id, AgentsInventoryReadParams()
            )
        error = exc_info.value
        assert error.method == AGENT_AGENTS_INVENTORY_READ_METHOD
        assert error.capabilities == (AGENT_WIRE_CAPABILITY_LIVE_AGENT_INVENTORY,)
        assert AGENT_WIRE_CAPABILITY_LIVE_AGENT_INVENTORY in str(error)
        assert "restart both ends on the same chartreux version" in str(error)
        assert "was not sent" in str(error)
        # No frame crossed the pipe: the skew is refused before any send, so
        # there is no stale fallback and nothing to replay.
        assert transport.requests(AGENT_AGENTS_INVENTORY_READ_METHOD) == []
        # The handle still serves requests that need no additive capability.
        snapshot = await process.snapshot(include_history=False)
        assert snapshot.state.session.id == loop.session_id


def test_the_ready_frame_advertises_the_live_inventory_capability() -> None:
    """C6b: this build's ready frame answers the inventory carrier it serves.

    The capability map routes the carrier to its advertised capability, and
    the advertised set carries it, so same-version peers of this build are
    never refused on the read.
    """
    assert required_agent_capabilities(
        AGENT_AGENTS_INVENTORY_READ_METHOD, {"session_id": "s"}
    ) == frozenset({AGENT_WIRE_CAPABILITY_LIVE_AGENT_INVENTORY})
    assert AGENT_WIRE_CAPABILITY_LIVE_AGENT_INVENTORY in ADVERTISED_CAPABILITIES
    assert (
        AGENT_WIRE_CAPABILITY_LIVE_AGENT_INVENTORY
        in parse_ready_frame(encode_ready_frame()).capabilities
    )


@pytest.mark.asyncio
async def test_read_snapshot_adoption_is_monotone_against_relayed_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2a: adoption keys on the shared agent-sequence cursor, monotone.

    The coordinator's watermark numbers its own publications, so it cannot
    order a snapshot against relayed events; a snapshot whose agent-sequence
    cursor is older than an event already relayed (or than the cursor already
    adopted) is never adopted, so the adopted state can never present older
    than events already relayed.
    """
    harness, client, session, loop = await _runtime_read_harness(monkeypatch)
    try:
        backend = cast(Any, harness._server._root)
        process = backend.process
        # One relayed event frame raises the shared cursor on the coordinator.
        process._deliver_notification(
            AGENT_NOTICE_METHOD,
            {
                "event": encode_notice(
                    SessionTitleUpdatedEvent(
                        session_id=loop.session_id, title="relayed"
                    )
                ),
                "sequence": 10,
            },
        )
        assert process.last_agent_sequence == 10

        projection_before = backend._projection
        # Older than the relayed event: skipped, the projection stands.
        stale = build_session_snapshot(
            loop, markers=(), incarnation_id="inc", agent_sequence=5
        )
        assert backend._adopt_read_snapshot(stale) is False
        assert backend._projection is projection_before
        assert backend._adopted_agent_sequence == 0

        # At the relayed cursor: adopted.
        fresh = build_session_snapshot(
            loop, markers=(), incarnation_id="inc", agent_sequence=10
        )
        assert backend._adopt_read_snapshot(fresh) is True
        assert backend._adopted_agent_sequence == 10
        assert backend._projection is not projection_before

        # Monotone: a later snapshot older than the adopted cursor is skipped.
        older = build_session_snapshot(
            loop, markers=(), incarnation_id="inc", agent_sequence=9
        )
        assert backend._adopt_read_snapshot(older) is False
        assert backend._adopted_agent_sequence == 10
    finally:
        with suppress(BaseException):
            await session.close()
        with suppress(BaseException):
            await client.close()
        with suppress(BaseException):
            await harness.close()
        await loop.aclose()


# ---------------------------------------------------------------------------
# C2b: the death/resume/memorial safety unit — death receipts on both
# transports, explicit session/resume respawn+handshake+restore,
# old-generation fencing, memorial queue/shell state in the resume receipt,
# no mutation replay, unknown-outcome reconcile, disconnect-terminal
# disclosure. The stdio leg runs the root agent over a memory wire
# (install_memory_root_agent); spawned-stdio legs land with C5c's PTY gate.
# ---------------------------------------------------------------------------


def _c2b_resume_params(session_id: str) -> dict[str, Any]:
    return SessionResumeParams(session_id=session_id).model_dump(
        mode="json", by_alias=True
    )


async def _c2b_close(harness: Any, client: Any, session: Any, loop: Any) -> None:
    with suppress(BaseException):
        await session.close()
    with suppress(BaseException):
        await client.close()
    with suppress(BaseException):
        await harness.close()
    await loop.aclose()


@pytest.mark.asyncio
async def test_death_receipt_fences_every_request_and_discloses_the_terminal_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2b: after confirmed root death every request fails with the receipt.

    The server's per-request guard and every session-addressed carrier arm
    answer the typed death receipt, and the receipt discloses that the stdio
    disconnect is terminal, in-flight work can be lost, and the explicit
    session/resume is the recovery path. Old-generation events publish
    nothing any more.
    """
    harness, client, session, loop = await _runtime_read_harness(monkeypatch)
    try:
        backend = cast(Any, harness._server._root)
        session_id = session.session_id
        backend.process._declare_death(
            "the root agent process exited",
            signature=RootFailureSignature.DEATH_RECEIPT,
        )
        assert backend.is_dead
        with pytest.raises(SessionBackendError) as guard_exc:
            backend.guard_request()
        guard_message = str(guard_exc.value)
        assert "death-receipt" in guard_message
        assert "in-flight work" in guard_message
        assert "session/resume" in guard_message
        assert "never replayed" in guard_message
        with pytest.raises(SessionBackendError) as arm_exc:
            await backend.dispatch_extension(
                "session/ready/read", {"sessionId": session_id}
            )
        assert "death-receipt" in str(arm_exc.value)
        # Old-generation fence: the dead generation publishes no events.
        assert await backend.publish_callback(cast(Any, Mock())) is False
    finally:
        await _c2b_close(harness, client, session, loop)


@pytest.mark.asyncio
async def test_session_resume_after_root_death_respawns_and_surfaces_the_memorial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2b: an explicit resume retires the dead backend and respawns once.

    The resume receipt carries the retired generation's typed death receipt and
    its coordinator-owned queue/shell records as memorial state; the
    replacement generation handshakes, restores the durable session, and never
    re-admits the memorial for execution. A second resume is the idempotent
    reconcile: same live backend, no second spawn, no mutation replay.
    """
    harness, client, session, loop = await _runtime_read_harness(monkeypatch)
    try:
        backend = cast(Any, harness._server._root)
        session_id = session.session_id
        # The doomed generation's coordinator-durable state: one queued turn
        # and one retained shell record.
        backend._turn_queue.enqueue(
            TurnEnqueueParams(
                session_id=session_id,
                entries=[
                    TurnUserInputEntry(
                        content=[SessionTextContentBlock(text="queued prompt")]
                    )
                ],
            )
        )
        backend._shell_retained["entry-1"] = psb_module._PromotedShellRecord(
            session_id=session_id,
            entry_id="entry-1",
            text="ls -la",
            context=_manual_shell(),
            committed=True,
        )
        backend.process._declare_death(
            "the root agent process exited",
            signature=RootFailureSignature.DEATH_RECEIPT,
        )
        old_process = backend.process
        spawns = 0
        real_spawn = psb_module.spawn_root_agent_process

        async def counting_spawn(**kwargs: Any) -> Any:
            nonlocal spawns
            spawns += 1
            return await real_spawn(**kwargs)

        monkeypatch.setattr(psb_module, "spawn_root_agent_process", counting_spawn)

        response = SessionResumeResponse.model_validate(
            await client.request("session/resume", _c2b_resume_params(session_id))
        )
        # The receipt surfaces the retired generation's death and memorial.
        assert response.death_receipt is not None
        assert "death-receipt" in response.death_receipt
        assert {item.kind for item in response.memorial} == {
            "queued_turn",
            "retained_shell_record",
        }
        assert {item.text for item in response.memorial} == {"queued prompt", "ls -la"}
        # Exactly one respawn: the replacement handshakes and serves the same
        # durable session id.
        assert spawns == 1
        assert response.state.session.id == session_id
        new_backend = cast(Any, harness._server._root)
        assert new_backend is not backend
        assert new_backend.process is not old_process
        read = await new_backend.dispatch_extension(
            "runtime/read", {"sessionId": session_id}
        )
        assert read.response.ready is True
        # Memorial state is never re-admitted: the new generation's queue and
        # retained records start empty — the user re-submits from the receipt.
        assert new_backend._turn_queue.state.items == []
        assert not new_backend._shell_retained
        # The old generation is fenced.
        assert backend.is_dead
        with pytest.raises(SessionBackendError) as fence_exc:
            backend.guard_request()
        assert "death-receipt" in str(fence_exc.value)
        # A second resume reconciles idempotently: no respawn, no memorial.
        again = SessionResumeResponse.model_validate(
            await client.request("session/resume", _c2b_resume_params(session_id))
        )
        assert spawns == 1
        assert again.death_receipt is None
        assert again.memorial == []
        assert harness._server._root is new_backend
    finally:
        await _c2b_close(harness, client, session, loop)


@pytest.mark.asyncio
async def test_a_concurrent_resume_is_fenced_never_a_second_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2b: two resumes for one dead generation never spawn two replacements.

    The host fences the second resume with a typed conflict while the first
    respawn is in flight; the in-flight one settles it.
    """
    harness, client, session, loop = await _runtime_read_harness(monkeypatch)
    try:
        backend = cast(Any, harness._server._root)
        session_id = session.session_id
        backend.process._declare_death(
            "the root agent process exited",
            signature=RootFailureSignature.DEATH_RECEIPT,
        )
        host = harness._server._session_backend_host
        real_spawn = psb_module.spawn_root_agent_process
        entered = asyncio.Event()
        release = asyncio.Event()

        async def gated_spawn(**kwargs: Any) -> Any:
            entered.set()
            await release.wait()
            return await real_spawn(**kwargs)

        monkeypatch.setattr(psb_module, "spawn_root_agent_process", gated_spawn)
        params = SessionResumeParams(session_id=session_id)
        first = asyncio.create_task(host.resume(params))
        await entered.wait()
        with pytest.raises(SessionBackendError) as exc_info:
            await host.resume(params)
        error = exc_info.value
        assert error.code is ProtocolErrorCode.CONFLICT
        assert "already in flight" in str(error)
        release.set()
        result = await first
        assert result.death_receipt is not None
        assert "death-receipt" in result.death_receipt
    finally:
        await _c2b_close(harness, client, session, loop)


@pytest.mark.asyncio
async def test_resume_against_the_live_generation_is_an_idempotent_reattach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2b: resume of a still-live generation re-attaches without respawning.

    This mirrors the in-process arm's same-id behavior and is the reconcile
    path after an unknown outcome: resume is not a mutation, so retrying it
    never spawns a second root and never replays anything.
    """
    harness, client, session, loop = await _runtime_read_harness(monkeypatch)
    try:
        backend = cast(Any, harness._server._root)
        session_id = session.session_id
        spawns = 0
        real_spawn = psb_module.spawn_root_agent_process

        async def counting_spawn(**kwargs: Any) -> Any:
            nonlocal spawns
            spawns += 1
            return await real_spawn(**kwargs)

        monkeypatch.setattr(psb_module, "spawn_root_agent_process", counting_spawn)
        response = SessionResumeResponse.model_validate(
            await client.request("session/resume", _c2b_resume_params(session_id))
        )
        assert spawns == 0
        assert response.death_receipt is None
        assert response.memorial == []
        assert response.state.session.id == session_id
        assert harness._server._root is backend
    finally:
        await _c2b_close(harness, client, session, loop)


@pytest.mark.asyncio
async def test_a_failed_respawn_is_an_actionable_failure_and_a_retry_reconciles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2b: an unknown-outcome respawn answers typed and reconciles on retry.

    A spawn failure after the death answers the actionable spawn diagnostic —
    never a fresh empty session — and the dead generation keeps answering the
    death receipt. Retrying session/resume (not a mutation) then spawns the one
    replacement and surfaces the memorial.
    """
    harness, client, session, loop = await _runtime_read_harness(monkeypatch)
    try:
        backend = cast(Any, harness._server._root)
        session_id = session.session_id
        backend._turn_queue.enqueue(
            TurnEnqueueParams(
                session_id=session_id,
                entries=[
                    TurnUserInputEntry(
                        content=[SessionTextContentBlock(text="queued prompt")]
                    )
                ],
            )
        )
        backend.process._declare_death(
            "the root agent process exited",
            signature=RootFailureSignature.DEATH_RECEIPT,
        )
        real_spawn = psb_module.spawn_root_agent_process

        async def failing_spawn(**kwargs: Any) -> Any:
            raise RuntimeError("spawn exploded")

        monkeypatch.setattr(psb_module, "spawn_root_agent_process", failing_spawn)
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("session/resume", _c2b_resume_params(session_id))
        error = exc_info.value.error
        assert error.code is ProtocolErrorCode.INTERNAL_ERROR
        assert "could not start the root agent" in error.message
        # The dead generation still answers the death receipt on reads.
        with pytest.raises(AppServerResponseError) as read_exc:
            await client.request(
                "runtime/read", RuntimeReadParams(session_id=session_id)
            )
        assert "death-receipt" in str(read_exc.value)
        # Reconcile: retry the resume — it is not a mutation, so it is safe.
        monkeypatch.setattr(psb_module, "spawn_root_agent_process", real_spawn)
        response = SessionResumeResponse.model_validate(
            await client.request("session/resume", _c2b_resume_params(session_id))
        )
        assert response.death_receipt is not None
        assert "death-receipt" in response.death_receipt
        assert [item.text for item in response.memorial] == ["queued prompt"]
        assert response.state.session.id == session_id
        assert harness._server._root is not backend
    finally:
        await _c2b_close(harness, client, session, loop)
