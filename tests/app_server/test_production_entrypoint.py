"""The production entrypoint: a stdio session's root lives in its own process.

These tests drive the session through the harness the CLI builds -- the
selection resolved once and materialized into the spawn environment, the
process-backed session backend the app server binds for it, and a real turn
that runs in the spawned root agent. The child is a controlled script running
the shipped ``serve_root_agent`` loop, so no provider is reached and the
egress marker blocks any attempt.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing, suppress
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from chartreux.app_server.local import LocalHarness, LocalHarnessOptions
from chartreux.app_server.process_session_backend import materialized_spawn_env
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    SessionOptions,
)
from chartreux.app_server.transport import (
    STDIO_TRANSPORT,
    TRANSPORT_ENV_VAR,
    TransportSelection,
)
from tests.differential import egress

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
{prologue}asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""

#: A child whose provider fails every turn. The manual `!` effect still runs
#: (a shell run reaches no provider), so the only thing that fails is the record
#: turn the queue promotes -- which is exactly the failure B.2a retains for.
_FAULTY_PROVIDER_CHILD_SCRIPT = """\
import asyncio

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config
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
    config=config,
    backend=FakeBackend(exception_to_raise=RuntimeError("provider unavailable")),
)
asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""

#: A child whose loop records the bounds a settings update applied to it. The
#: marker is the evidence that the frame crossed the pipe and ran agent-side
#: rather than being acknowledged by the coordinator.
_SETTINGS_SPY_PROLOGUE = """\
from pathlib import Path

from chartreux.core.agent_loop import AgentLoop

_applied_bounds = Path({marker!r})


def _record_applied_max_tokens(_self, max_tokens):
    _applied_bounds.write_text(str(max_tokens))


AgentLoop.set_max_tokens = _record_applied_max_tokens
"""

_EXITING_SCRIPT = """\
import os
os._exit(3)
"""

#: A child whose model asks the user one question. The turn really opens a
#: user-input record in the root agent process, so every callback class below
#: crosses the pipe in the direction the process topology makes possible.
_CALLBACK_QUESTION_ARGS = json.dumps({
    "questions": [
        {"question": "Ship it?", "options": [{"label": "Yes"}, {"label": "No"}]}
    ]
})

_CALLBACK_CHILD_SCRIPT = """\
import asyncio

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, ToolCall
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_QUESTION_ARGS = {question_args!r}


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
    config=config,
    backend=FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    ToolCall(
                        id="question-1",
                        index=0,
                        function=FunctionCall(
                            name="ask_user_question", arguments=_QUESTION_ARGS
                        ),
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="done")],
    ]),
    user_input_capability=True,
)
asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""

#: A child whose loop carries an out-of-band title, the way a backgrounded
#: title generation reaches the coordinator: no turn runs, only the notice.
#: A notice names its session, so it waits for one, and it lands after the
#: client has attached, exactly as an asynchronous title would.
_NOTICE_PROLOGUE = """\
import asyncio

from chartreux.core.events import SessionTitleUpdatedEvent


async def _out_of_band_notices():
    while True:
        session_id = getattr(loop, "session_id", None)
        if session_id:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.3)
    yield SessionTitleUpdatedEvent(session_id=session_id, title={title!r})


loop.out_of_band_events = _out_of_band_notices
"""

#: A child that, at the start of one turn, asks the coordinator for the host
#: tool-I/O port over the agent-origin channel -- the op-9 direction. The
#: marker records what that round trip answered, so it is the evidence the
#: request crossed the pipe in the direction no coordinator request could.
_HOST_TOOL_IO_PROLOGUE = """\
from pathlib import Path

from chartreux.app_server.root_agent import RootAgentServer

_marker = Path({marker!r})
_target = {target!r}
_real_start_turn = RootAgentServer._start_turn


def _start_turn_after_host_call(self, params):
    self._track(_call_host_tool_io(self, params))


async def _call_host_tool_io(self, params):
    try:
        payload = await self.call_host(
            "host/tool/io", {{"op": "read_text", "path": _target}}
        )
        result = payload.get("result") or {{}}
        _marker.write_text("read:" + str(result.get("text")))
    except BaseException as exc:
        _marker.write_text("error:" + type(exc).__name__ + ":" + str(exc))
    _real_start_turn(self, params)


RootAgentServer._start_turn = _start_turn_after_host_call
"""


#: A child whose loop is driven by a real layered orchestrator with an owned
#: user source. The shared fake holds a config verbatim and publishes no source
#: provenance, so a policy replacement -- which names the source it replaces --
#: needs the production orchestrator to have one.
_POLICY_CHILD_SCRIPT = """\
import asyncio
from pathlib import Path

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.config.orchestrator import ConfigOrchestrator
from tests.conftest import build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


class _Factory:
    def __init__(self, loop):
        self._loop = loop

    async def open(self, params):
        return self._loop

    async def close(self):
        pass


async def _serve():
    user = UserConfigLayer(path=Path({user_path!r}))
    launch = OverridesLayer(
        data={{"session_logging": _config.session_logging.model_dump(mode="json")}}
    )
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[user, launch],
        default_layer_resolver=lambda: user,
        catalog_snapshot=_config.catalog_snapshot,
    )
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend([[mock_llm_chunk(content="done")]]),
        mcp_registry=FakeMCPRegistry(),
    )
    await serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())


init_harness_files_manager("user", "project")
_config = build_test_vibe_config(
    session_logging=SessionLoggingConfig(enabled=True, save_dir={save_dir!r})
)
Path({user_path!r}).write_text('[tools.read_file]\\ndenylist = ["A"]\\n')
asyncio.run(_serve())
"""


def _child_command(tmp_path: Path, prologue: str = "") -> list[str]:
    return [
        sys.executable,
        "-c",
        _CHILD_SCRIPT.format(save_dir=str(tmp_path / "sessions"), prologue=prologue),
    ]


def _callback_command(tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        _CALLBACK_CHILD_SCRIPT.format(
            save_dir=str(tmp_path / "sessions"), question_args=_CALLBACK_QUESTION_ARGS
        ),
    ]


def _faulty_provider_command(tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        _FAULTY_PROVIDER_CHILD_SCRIPT.format(save_dir=str(tmp_path / "sessions")),
    ]


def _policy_command(tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        _POLICY_CHILD_SCRIPT.format(
            save_dir=str(tmp_path / "sessions"),
            user_path=str(tmp_path / "policy-user.toml"),
        ),
    ]


def _exiting_command() -> list[str]:
    return [sys.executable, "-c", _EXITING_SCRIPT]


def _install_stdio_environment(
    monkeypatch: pytest.MonkeyPatch, guard_dir: Path
) -> None:
    """The environment an interactive stdio run carries into its spawn."""
    monkeypatch.setenv(TRANSPORT_ENV_VAR, STDIO_TRANSPORT)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    paths = [str(_REPO_ROOT), str(guard_dir)]
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    for name, value in egress.child_environment(egress.MODE_ALL).items():
        monkeypatch.setenv(name, value)


def _options(tmp_path: Path) -> LocalHarnessOptions:
    return LocalHarnessOptions(
        client=_client(),
        session_options=SessionOptions(cwd=str(tmp_path), headless=True),
    )


def _install_default_environment(
    monkeypatch: pytest.MonkeyPatch, guard_dir: Path
) -> None:
    """The environment an interactive run carries when nothing names a route."""
    monkeypatch.delenv(TRANSPORT_ENV_VAR, raising=False)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    paths = [str(_REPO_ROOT), str(guard_dir)]
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    for name, value in egress.child_environment(egress.MODE_ALL).items():
        monkeypatch.setenv(name, value)


def _client() -> Any:
    from chartreux.app_server.local import ClientDescriptor

    return ClientDescriptor(
        info=ClientInfo(name="tests", version="1.0"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
    )


async def _collect_turn(session: Any, message: str) -> list[Any]:
    events: list[Any] = []
    async with aclosing(session.act(message)) as stream:
        async for event in stream:
            events.append(event)
    return events


def test_a_stdio_selection_materializes_the_selector_into_the_spawn_env() -> None:
    selection = TransportSelection(
        kind=STDIO_TRANSPORT, selector=STDIO_TRANSPORT, source="config"
    )

    env = materialized_spawn_env(selection, environ={})

    assert env[TRANSPORT_ENV_VAR] == STDIO_TRANSPORT


def test_an_in_process_selection_materializes_nothing() -> None:
    selection = TransportSelection(
        kind="in_process", selector="in_process", source="default"
    )

    env = materialized_spawn_env(selection, environ={"KEEP": "1"})

    assert TRANSPORT_ENV_VAR not in env
    assert env["KEEP"] == "1"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_stdio_session_opens_through_a_spawned_root_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            assert len(backends) == 1
            process = backends[0].process.process
            assert process.pid != os.getpid()
            assert process.poll() is None
            assert session.session_id == backends[0].session_id
            await session.close()
        finally:
            await harness.close()

    assert process.poll() is not None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_turn_runs_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            process = backends[0].process.process
            events = await _collect_turn(session, "hello")
            await session.close()
        finally:
            await harness.close()

    assert events
    assert "done" in _assistant_text(events)
    assert process.poll() is not None


def _user_answer(answer: str) -> Any:
    from chartreux.app_server.models import (
        UserAnswer,
        UserInputCallbackOutput,
        UserQuestionResult,
    )

    return UserInputCallbackOutput(
        result=UserQuestionResult(
            answers=[UserAnswer(question="Ship it?", answer=answer)]
        )
    )


def _callback_entry(session: Any, callback_id: str) -> Any:
    from chartreux.app_server.models import PublicCallbackEntry

    return next(
        entry
        for entry in session.history
        if isinstance(entry, PublicCallbackEntry) and entry.callback_id == callback_id
    )


async def _await_callback_entry(
    session: Any, callback_id: str, predicate: Any, timeout: float = 10.0
) -> Any:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            entry = _callback_entry(session, callback_id)
        except StopIteration:
            await asyncio.sleep(0.05)
            continue
        if predicate(entry):
            return entry
        await asyncio.sleep(0.05)
    return _callback_entry(session, callback_id)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_user_input_callback_is_answered_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.events import CallbackRequested
    from chartreux.app_server.models import AnsweredCallbackState

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _callback_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            callback_ids: list[str] = []
            events: list[Any] = []
            async for event in session.act("ask me"):
                if isinstance(event, CallbackRequested):
                    callback_ids.append(event.callback.callback_id)
                    await session.respond_to_callback(
                        event.callback.callback_id, _user_answer("Yes")
                    )
                events.append(event)
            entry = await _await_callback_entry(
                session,
                callback_ids[0],
                lambda candidate: isinstance(candidate.state, AnsweredCallbackState),
            )
            await session.close()
        finally:
            await harness.close()

    # The request the agent opened reached this client, the answer crossed the
    # pipe to the record that owns it, and the terminal state the agent settled
    # with was published back before its waiter unblocked.
    assert callback_ids
    assert "done" in _assistant_text(events)
    assert isinstance(entry.state, AnsweredCallbackState)
    assert entry.state.output.result.answers[0].answer == "Yes"
    assert session.state.active_callbacks == []
    assert backends[0].open_callbacks() == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_refused_callback_closes_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.events import CallbackRequested
    from chartreux.app_server.models import CancelledCallbackState
    from chartreux.app_server.protocol import CallbackResultError

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _callback_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            callback_ids: list[str] = []
            events: list[Any] = []
            async for event in session.act("ask me"):
                if isinstance(event, CallbackRequested):
                    callback_ids.append(event.callback.callback_id)
                    await session.reject_callback(
                        event.callback.callback_id,
                        CallbackResultError(message="not now"),
                    )
                events.append(event)
            entry = await _await_callback_entry(
                session,
                callback_ids[0],
                lambda candidate: isinstance(candidate.state, CancelledCallbackState),
            )
            await session.close()
        finally:
            await harness.close()

    # A refusal is the agent's arbitration too: the record closes where it
    # lives and the closed entry is what this client ends up projecting.
    assert "done" in _assistant_text(events)
    assert isinstance(entry.state, CancelledCallbackState)
    assert entry.state.reason == "not now"
    assert session.state.active_callbacks == []
    assert backends[0].open_callbacks() == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_kept_waiting_callback_stays_open_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.events import CallbackRequested

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _callback_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            callback_ids: list[str] = []
            kept: list[str] = []
            async for event in session.act("ask me"):
                if isinstance(event, CallbackRequested):
                    callback_ids.append(event.callback.callback_id)
                    kept.append(
                        await session.keep_waiting_callback(event.callback.callback_id)
                    )
                    await session.respond_to_callback(
                        event.callback.callback_id, _user_answer("Yes")
                    )
            await session.close()
        finally:
            await harness.close()

    # The deadline decision is the agent's own: an ineligible request answers
    # "no_deadline" from the process that holds its record, and settling the
    # request afterwards still works.
    assert kept == ["no_deadline"]
    assert callback_ids


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_cancelled_callback_closes_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.events import CallbackRequested
    from chartreux.app_server.models import CancelledCallbackState

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _callback_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            callback_ids: list[str] = []
            events: list[Any] = []
            async for event in session.act("ask me"):
                if isinstance(event, CallbackRequested):
                    callback_ids.append(event.callback.callback_id)
                    await backends[0].process.cancel_user_input(
                        event.callback.callback_id,
                        RuntimeError("cancelled by the test"),
                    )
                events.append(event)
            entry = await _await_callback_entry(
                session,
                callback_ids[0],
                lambda candidate: isinstance(candidate.state, CancelledCallbackState),
            )
            await session.close()
        finally:
            await harness.close()

    # The cancel frame closed the record agent-side, and the closure it
    # published is the terminal state this client's projection carries.
    assert "done" in _assistant_text(events)
    assert isinstance(entry.state, CancelledCallbackState)
    assert entry.state.reason == "cancelled by the test"
    assert backends[0].open_callbacks() == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_relayed_request_is_adopted_and_settled_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.core.events import UserInputOrigin, UserInputRequestEvent
    from chartreux.questions import UserQuestionRequest

    def _relay(process: Any, request_id: str) -> None:
        process.relay_user_input_request(
            UserInputRequestEvent(
                request_id=request_id,
                tool_call_id="host-tool-call",
                origin=UserInputOrigin.RUNTIME_APPROVAL,
                args=UserQuestionRequest.model_validate(
                    json.loads(_CALLBACK_QUESTION_ARGS)
                ),
            )
        )

    async def _until_outcome(probe: Any, expected: Any) -> Any:
        outcome: Any = None
        for _ in range(200):
            outcome = await probe()
            if outcome == expected:
                return outcome
            await asyncio.sleep(0.02)
        return outcome

    async def _flush(process: Any, request_id: str) -> None:
        """Let an id-less frame reach the agent before the next request."""
        for _ in range(10):
            await asyncio.sleep(0)
        assert await process.keep_waiting_user_input(request_id) == "no_deadline"

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _callback_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            process = backends[0].process
            # The relay class: a host-minted request is adopted agent-side,
            # which only the process holding the record can answer for.
            _relay(process, "relayed-answer")
            adopted = await _until_outcome(
                lambda: process.keep_waiting_user_input("relayed-answer"), "no_deadline"
            )
            answered = await process.settle_user_input_answer(
                "relayed-answer", _user_answer("Yes").result
            )
            # The raw resolve class: no arbitration, the waiter is settled, so
            # a later answer finds nothing left to settle.
            _relay(process, "relayed-resolve")
            await _flush(process, "relayed-resolve")
            process.resolve_user_input("relayed-resolve", _user_answer("Yes").result)
            for _ in range(10):
                await asyncio.sleep(0)
            resolved = await process.settle_user_input_answer(
                "relayed-resolve", _user_answer("No").result
            )
            # The raw reject class: the waiter fails without a settlement.
            _relay(process, "relayed-reject")
            await _flush(process, "relayed-reject")
            process.reject_user_input(
                "relayed-reject", RuntimeError("refused host-side")
            )
            for _ in range(10):
                await asyncio.sleep(0)
            rejected = await process.settle_user_input_answer(
                "relayed-reject", _user_answer("Yes").result
            )
            await session.close()
        finally:
            await harness.close()

    # A raw resolve settles the waiter, and the record stays for a late answer
    # to find closed; a raw reject does the same by failing it.
    assert adopted == "no_deadline"
    assert answered == "accepted"
    assert resolved == "closed"
    assert rejected == "closed"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_unspawnable_root_fails_fast_and_names_the_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend, "root_agent_command", _exiting_command
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            with pytest.raises(AppServerResponseError) as caught:
                await harness.start()
        finally:
            await harness.close()

    message = str(caught.value)
    assert "stdio topology could not start" in message
    assert "CHARTREUX_TRANSPORT=in_process" in message


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_failed_handshake_leaves_no_root_agent_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    spawned: list[Any] = []
    real_spawn = process_session_backend.spawn_root_agent_process

    async def _recording_spawn(**kwargs: Any) -> Any:
        process = await real_spawn(**kwargs)
        spawned.append(process)
        return process

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend, "root_agent_command", _exiting_command
        )
        monkeypatch.setattr(
            process_session_backend, "spawn_root_agent_process", _recording_spawn
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            with pytest.raises(AppServerResponseError):
                await harness.start()
        finally:
            await harness.close()

    assert spawned == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_in_process_session_spawns_no_root_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    spawned: list[Any] = []

    async def _recording_spawn(**kwargs: Any) -> Any:
        spawned.append(kwargs)
        raise AssertionError("an in-process session must not spawn")

    monkeypatch.setenv(TRANSPORT_ENV_VAR, "in_process")
    monkeypatch.setattr(
        process_session_backend, "spawn_root_agent_process", _recording_spawn
    )
    harness = LocalHarness(_options(tmp_path))
    try:
        session = await harness.start()
        await session.close()
    finally:
        await harness.close()

    assert spawned == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_injection_crosses_the_pipe_and_projects_its_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            entries = await session.inject_user_context(
                "a note from the client", as_message=True
            )
            await session.close()
        finally:
            await harness.close()

    assert [getattr(added.entry, "role", None) for added in entries] == ["user"]
    assert "a note from the client" in _entry_text(entries)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_steering_delivery_is_arbitrated_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol import ProtocolErrorCode

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            with pytest.raises(AppServerResponseError) as caught:
                await session.inject_user_context(
                    "steer this", expected_turn_id="a-turn-nobody-runs"
                )
            await session.close()
        finally:
            await harness.close()

    # The refusal is the agent's own arbitration, carried back typed: only the
    # process that runs the turn can say which turn it is running.
    assert caught.value.error.code is ProtocolErrorCode.CONFLICT
    assert "not running" in caught.value.error.message


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_settings_update_is_applied_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    applied = tmp_path / "applied-max-tokens"
    prologue = _SETTINGS_SPY_PROLOGUE.format(marker=str(applied))
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path, prologue),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            await session.resources.sessions.update_settings(max_tokens=4242)
            await session.close()
        finally:
            await harness.close()

    # Only the agent's own loop can record the bound, so the marker proves the
    # short mode crossed the pipe instead of being acknowledged here.
    assert applied.read_text() == "4242"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_history_clear_empties_the_transcript_the_agent_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            await _collect_turn(session, "hello")
            assert session.history
            await session.clear_history()
            cleared = list(session.history)
            await session.close()
        finally:
            await harness.close()

    # The clear ran where the transcript lives, and the state it answered with
    # is the one this client projects afterwards.
    assert cleared == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_out_of_turn_notice_reaches_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    prologue = _NOTICE_PROLOGUE.format(title="a-written-title")
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path, prologue),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            title = await _await_title(session, "a-written-title")
            await session.close()
        finally:
            await harness.close()

    # No turn ran: the title reached this client as a notice off the agent's
    # own out-of-band channel, projected as the session's new title.
    assert title == "a-written-title"


async def _await_title(session: Any, expected: str, timeout: float = 10.0) -> str:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        title = session.state.session.title
        if title == expected:
            return title
        await asyncio.sleep(0.05)
    return session.state.session.title


class _ClientFilesystem:
    """The client half of one filesystem read, served where the client lives.

    The declared capability is the read alone, so every terminal op is left
    unimplemented: reaching one would be the coordinator servicing a capability
    this client never offered.
    """

    def __init__(self, content: str) -> None:
        self._content = content
        self.requests: list[Any] = []

    async def read_text_file(self, params: Any) -> Any:
        from chartreux.app_server.protocol import ClientToolReadTextFileResponse

        self.requests.append(params)
        return ClientToolReadTextFileResponse(content=self._content)

    async def write_text_file(self, params: Any) -> Any:
        raise NotImplementedError()

    async def create_terminal(self, params: Any) -> Any:
        raise NotImplementedError()

    async def wait_for_terminal_exit(self, params: Any) -> Any:
        raise NotImplementedError()

    async def terminal_output(self, params: Any) -> Any:
        raise NotImplementedError()

    async def kill_terminal(self, params: Any) -> Any:
        raise NotImplementedError()

    async def release_terminal(self, params: Any) -> Any:
        raise NotImplementedError()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_client_tool_read_is_served_over_the_host_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.local import ClientDescriptor

    handler = _ClientFilesystem("client-side text")
    marker = tmp_path / "host-tool-io"
    target = tmp_path / "client-file.txt"
    prologue = _HOST_TOOL_IO_PROLOGUE.format(marker=str(marker), target=str(target))
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path, prologue),
        )
        harness = LocalHarness(
            LocalHarnessOptions(
                client=ClientDescriptor(
                    info=ClientInfo(name="tests", version="1.0"),
                    capabilities=ClientCapabilities(
                        callback_kinds=["user_input"], client_tools=["filesystem/read"]
                    ),
                ),
                session_options=SessionOptions(cwd=str(tmp_path), headless=True),
                client_tool_handler=handler,
            )
        )
        try:
            session = await harness.start()
            await _collect_turn(session, "hello")
            await session.close()
        finally:
            await harness.close()

    # The agent's read crossed the pipe on its OWN request id (the op-9
    # direction), was served by the coordinator from the turn's host port, and
    # the client that declared the capability answered it.
    assert marker.read_text() == "read:client-side text"
    assert [request.path for request in handler.requests] == [str(target)]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_session_list_is_answered_by_the_root_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            await _collect_turn(session, "hello")
            listed = await session.resources.sessions.list()
            await session.close()
        finally:
            await harness.close()

    # The list is projected from the config the agent opened with, so the
    # session this client just ran a turn in is in it.
    assert session.session_id in [item.id for item in listed]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_config_reload_runs_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            response = await session.resources.config.reload()
            await session.close()
        finally:
            await harness.close()

    # The receipt is the reload leg's own: the applied runtime is the config
    # this session now runs on, published back to the client.
    assert response.runtime.config is not None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_policy_read_comes_from_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            policy = await session.resources.config.read_policy()
            await session.close()
        finally:
            await harness.close()

    # The revision names the session generation the agent itself accepted, so
    # only the process that owns the loop can have produced it.
    assert policy.revision.startswith(f"{session.session_id}:")
    assert isinstance(policy.sources, list)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_policy_replacement_is_published_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol import PolicyToolReplacement, ProtocolErrorCode

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _policy_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            # A saved turn is what persists the session directory the tree is
            # journaled in; the replacement itself is a live-session operation.
            await _collect_turn(session, "hello")
            policy = await session.resources.config.read_policy()
            replaced = await session.resources.config.replace_policy(
                source="user-toml",
                expected_revision=policy.revision,
                tools={"read_file": PolicyToolReplacement(denylist=["A"])},
                user_initiated=True,
            )
            after = await session.resources.config.read_policy()
            with pytest.raises(AppServerResponseError) as caught:
                await session.resources.config.replace_policy(
                    source="user-toml",
                    expected_revision=policy.revision,
                    tools={},
                    user_initiated=True,
                )
            await session.close()
        finally:
            await harness.close()

        journals = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (tmp_path / "sessions").rglob("*.registry.json")
        ]

    # The revision the follow-up read reports is the agent's own tree token,
    # so the change was published in the process that owns the loop.
    assert "user-toml" in policy.sources
    assert after.revision != policy.revision
    assert after.revision == replaced.revision
    # A stale request is the agent's refusal, before its first phase reserved.
    assert caught.value.error.code is ProtocolErrorCode.CONFLICT
    # The transaction is journaled where it ran, retired, carrying the change.
    records = [
        record
        for journal in journals
        for record in journal.get("policy_tree", {}).values()
    ]
    assert [record["phase"] for record in records] == ["retired"]
    assert records[0]["change"]["source"] == "user-toml"
    assert records[0]["change"]["tools"]["read_file"]["denylist"] == ["A"]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_roots_view_and_grant_save_run_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    granted = tmp_path / "granted"
    granted.mkdir()
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _policy_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            roots = await session.resources.config.read_roots()
            assert roots.user_revision is not None
            saved = await session.resources.config.save_root(
                root=str(granted),
                expected_revision=roots.user_revision,
                user_initiated=True,
            )
            after = await session.resources.config.read_roots()
            await session.close()
        finally:
            await harness.close()
        user_layer = (tmp_path / "policy-user.toml").read_text(encoding="utf-8")

    # The view names the agent's own workspace, and the save landed in the
    # user layer the agent's orchestrator holds -- not in a coordinator copy.
    assert roots.project == str(tmp_path)
    assert roots.roots == []
    assert saved.persistence == "saved"
    assert str(granted.resolve()) in user_layer
    # The save persists authority without publishing it to the live roots.
    assert after.roots == []
    assert after.saved_roots == [str(granted.resolve())]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_roots_replacement_is_refused_typed_and_names_the_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol import ProtocolErrorCode

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _policy_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            roots = await session.resources.config.read_roots()
            with pytest.raises(AppServerResponseError) as caught:
                await session.resources.config.replace_roots(
                    roots=[str(tmp_path)],
                    expected_revision=roots.revision,
                    user_initiated=True,
                )
            await session.close()
        finally:
            await harness.close()

    assert caught.value.error.code is ProtocolErrorCode.NOT_IMPLEMENTED
    assert "in_process" in caught.value.error.message


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_plan_record_is_written_and_read_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol import SessionPlanRecord

    record = SessionPlanRecord(
        plan_path=str(tmp_path / "plan.md"),
        planner_model="planner-model",
        planner_thinking="high",
        state="drafting",
    )
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            # A saved turn is what carries the session metadata file; the
            # record write itself stays an in-memory carry until one exists.
            await _collect_turn(session, "hello")
            written = await session.resources.sessions.write_plan_record(record)
            # The resume response carries this read, so it is asked of the
            # process directly here: same wire op, without a second spawn.
            read_back = await backends[0].process.read_plan_record()
            await session.close()
        finally:
            await harness.close()

        # The record is on disk where the agent wrote it, not in a coordinator
        # copy of the session's metadata.
        metadata_files = list((tmp_path / "sessions").rglob("meta.json"))
        persisted = [
            json.loads(path.read_text(encoding="utf-8")).get("plan_mode")
            for path in metadata_files
        ]

    assert written is not None
    assert written.plan_path == record.plan_path
    assert read_back is not None
    assert read_back.plan_path == record.plan_path
    assert read_back.state == "drafting"
    recorded = [entry for entry in persisted if entry is not None]
    assert len(recorded) == 1
    assert recorded[0]["plan_path"] == record.plan_path
    assert recorded[0]["state"] == "drafting"


def test_an_explicit_stdio_selector_outranks_the_config_escape() -> None:
    from chartreux.app_server.transport import resolve_transport_selection

    # The shipped precedence: an explicit selector in the environment is read
    # before the config escape, so a programmatic run that would otherwise stay
    # in-process still resolves (and spawns) stdio.
    selection = resolve_transport_selection(
        multi_process=False,
        environ={TRANSPORT_ENV_VAR: STDIO_TRANSPORT},
        interactive=False,
    )

    assert selection.kind == STDIO_TRANSPORT
    assert selection.source == "environment"
    assert materialized_spawn_env(selection, environ={})[TRANSPORT_ENV_VAR] == (
        STDIO_TRANSPORT
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_programmatic_run_still_spawns_on_an_explicit_stdio_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(
            LocalHarnessOptions(
                client=_client(),
                session_options=SessionOptions(cwd=str(tmp_path), headless=True),
                interactive=False,
            )
        )
        try:
            session = await harness.start()
            process = backends[0].process.process
            await session.close()
        finally:
            await harness.close()

    # The programmatic default is in-process; the explicit selector overrides
    # it at the entrypoint, so a root agent process really exists.
    assert len(backends) == 1
    assert process.pid != os.getpid()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_agent_runs_in_the_cwd_the_client_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    tree = tmp_path / "a-checked-out-tree"
    tree.mkdir()
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(
            LocalHarnessOptions(
                client=_client(),
                session_options=SessionOptions(cwd=str(tree), headless=True),
            )
        )
        try:
            session = await harness.start()
            cwd = session.state.session.cwd
            await session.close()
        finally:
            await harness.close()

    # The cwd is part of the spawn: it is the agent's own project, not a value
    # the coordinator could have stamped on the state from its own options.
    assert cwd == str(tree)
    assert (tmp_path / "sessions").is_dir()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_fork_is_refused_typed_and_names_the_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol import ProtocolErrorCode

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            with pytest.raises(AppServerResponseError) as caught:
                await session.resources.sessions.fork()
            await session.close()
        finally:
            await harness.close()

    assert caught.value.error.code is ProtocolErrorCode.NOT_IMPLEMENTED
    assert "in_process" in caught.value.error.message


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_rewind_read_is_refused_typed_and_names_the_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol import ProtocolErrorCode

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            with pytest.raises(AppServerResponseError) as caught:
                await session.resources.sessions.rewind_preview("an-entry-id")
            await session.close()
        finally:
            await harness.close()

    assert caught.value.error.code is ProtocolErrorCode.NOT_IMPLEMENTED
    assert "in_process" in caught.value.error.message


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_relocation_is_refused_typed_and_names_the_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol import ProtocolErrorCode

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            with pytest.raises(AppServerResponseError) as caught:
                await session.resources.sessions.relocate(str(tmp_path / "elsewhere"))
            await session.close()
        finally:
            await harness.close()

    assert caught.value.error.code is ProtocolErrorCode.NOT_IMPLEMENTED
    assert "in_process" in caught.value.error.message


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_config_write_applies_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol.config import ConfigWriteOpWire

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            response = await session.resources.config.write(
                [ConfigWriteOpWire(op="set", path="/show_thinking_nodes", value=True)],
                reason="entrypoint test",
            )
            await session.close()
        finally:
            await harness.close()

    # The receipt is the SHIPPED transaction result and the runtime it carries
    # is the one the applied change produced in the agent's own config.
    assert response.rejected is False
    assert response.failures == []
    assert response.runtime.config.show_thinking_nodes is True


def test_a_typed_agent_refusal_keeps_its_protocol_code() -> None:
    from chartreux.app_server.process_session_backend import _wire_failure
    from chartreux.app_server.protocol import ProtocolErrorCode
    from chartreux.app_server.root_agent_process import (
        RootAgentDeferredOperation,
        RootAgentReconfigureConflict,
        RootAgentReplaceConflict,
        RootAgentSessionNotFound,
    )

    assert (
        _wire_failure(RootAgentDeferredOperation("session/fork")).code
        is ProtocolErrorCode.NOT_IMPLEMENTED
    )
    assert (
        _wire_failure(
            RootAgentReconfigureConflict("a long operation is in flight")
        ).code
        is ProtocolErrorCode.CONFLICT
    )
    assert (
        _wire_failure(RootAgentReplaceConflict("a settlement is open")).code
        is ProtocolErrorCode.CONFLICT
    )
    assert (
        _wire_failure(RootAgentSessionNotFound("no such session")).code
        is ProtocolErrorCode.NOT_FOUND
    )


def test_an_agent_callback_outcome_keeps_its_protocol_code() -> None:
    from chartreux.app_server._session_backend_port import SessionBackendError
    from chartreux.app_server.process_session_backend import _settle_callback
    from chartreux.app_server.protocol import ProtocolErrorCode

    assert _settle_callback("callback-1", "accepted") == "accepted"
    assert _settle_callback("callback-1", "duplicate") == "duplicate"
    for outcome, code in (
        ("not_found", ProtocolErrorCode.NOT_FOUND),
        ("expired", ProtocolErrorCode.CALLBACK_CLOSED),
        ("closed", ProtocolErrorCode.CALLBACK_CLOSED),
        ("conflict", ProtocolErrorCode.CONFLICT),
    ):
        with pytest.raises(SessionBackendError) as caught:
            _settle_callback("callback-1", outcome)
        assert caught.value.code is code
    # An outcome this release cannot name is never reported as a settlement.
    with pytest.raises(SessionBackendError) as unknown:
        _settle_callback("callback-1", "something-else")
    assert unknown.value.code is ProtocolErrorCode.INTERNAL_ERROR


#: A child whose model asks the user one question and then answers plainly on
#: every later turn. The queue is the coordinator's, so the turns it promotes
#: are ordinary turns the root agent process runs: the transcript entry for the
#: queued prompt is the evidence promotion crossed the pipe.
_QUEUE_CHILD_SCRIPT = """\
import asyncio

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, ToolCall
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_QUESTION_ARGS = {question_args!r}


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
    config=config,
    backend=FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    ToolCall(
                        id="question-1",
                        index=0,
                        function=FunctionCall(
                            name="ask_user_question", arguments=_QUESTION_ARGS
                        ),
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(content="done")],
    ]),
    user_input_capability=True,
)
asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""


def _queue_command(tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        _QUEUE_CHILD_SCRIPT.format(
            save_dir=str(tmp_path / "sessions"), question_args=_CALLBACK_QUESTION_ARGS
        ),
    ]


#: A child whose provider answers every turn, failing the first one on request.
#: A manual `!` record turn is the first turn the queue promotes here, so a
#: fail-first provider is what makes a record be retained with later work behind
#: it -- the ordering a record's re-promotion has to survive.
_SHELL_ORDER_CHILD_SCRIPT = """\
import asyncio

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_FAIL_FIRST = {fail_first!r}


class _ScriptedBackend(FakeBackend):
    \"\"\"Four scripted turns; the first of them fails when the test asks.

The record turn a manual command promotes is the first turn here, so failing
    it once is what leaves a record retained while later work is queued.
    \"\"\"

    def __init__(self):
        super().__init__([[mock_llm_chunk(content="done")]] * 4)
        self._failed = False

    def _fail_first(self):
        if not _FAIL_FIRST or self._failed:
            return
        self._failed = True
        raise RuntimeError("provider unavailable")

    async def complete(self, **kwargs):
        self._fail_first()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs):
        self._fail_first()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


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
loop = build_test_agent_loop(config=config, backend=_ScriptedBackend())
asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""


def _shell_order_command(tmp_path: Path, *, fail_first: bool) -> list[str]:
    return [
        sys.executable,
        "-c",
        _SHELL_ORDER_CHILD_SCRIPT.format(
            save_dir=str(tmp_path / "sessions"), fail_first=fail_first
        ),
    ]


#: A child whose provider fails its first two turns, each after a hold. The
#: held first failure is what lets a prompt be queued while the record turn is
#: still running, so the record is then retained beside a *full* queue -- the
#: pending state the read endpoint must answer and the next failed turn must
#: re-offer. The third turn succeeds: the record's retry completes.
_DELAYED_FAILURE_CHILD_SCRIPT = """\
import asyncio

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_HOLD_SECONDS = {hold!r}


class _ScriptedBackend(FakeBackend):
    def __init__(self):
        super().__init__([[mock_llm_chunk(content="done")]] * 4)
        self._calls = 0

    async def _fail(self):
        self._calls += 1
        if self._calls > 2:
            return
        await asyncio.sleep(_HOLD_SECONDS)
        raise RuntimeError("provider unavailable")

    async def complete(self, **kwargs):
        await self._fail()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs):
        await self._fail()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


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
loop = build_test_agent_loop(config=config, backend=_ScriptedBackend())
asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""


def _delayed_failure_command(tmp_path: Path, *, hold: float) -> list[str]:
    return [
        sys.executable,
        "-c",
        _DELAYED_FAILURE_CHILD_SCRIPT.format(
            save_dir=str(tmp_path / "sessions"), hold=hold
        ),
    ]


def _gate_shell_commit(
    monkeypatch: pytest.MonkeyPatch, backend: Any
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold a promotion pass inside the commit of a record's message.

    The pass has peeked the record item and is committing that message while the
    gate is open, which is the window a queue mutation would race if promotion
    were not serialized with the queue.
    """
    entered = asyncio.Event()
    release = asyncio.Event()
    commit = backend._commit_shell_record

    async def gated_commit(*args: Any, **kwargs: Any) -> None:
        entered.set()
        await release.wait()
        await commit(*args, **kwargs)

    monkeypatch.setattr(backend, "_commit_shell_record", gated_commit)
    return entered, release


async def _wait_for(predicate: Any, timeout: float = 20.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Condition not met in time")


def _queue_texts(queue: Any) -> list[str]:
    from chartreux.app_server.models import SessionTextContentBlock

    return [
        block.text
        for item in queue.items
        for entry in item.entries
        for block in entry.content
        if isinstance(block, SessionTextContentBlock)
    ]


def _history_text(entries: list[Any]) -> str:
    parts: list[str] = []
    for entry in entries:
        for block in getattr(entry, "content", ()) or ():
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


async def _await_promoted_turn(
    session: Any, queue_item_id: str, timeout: float = 30.0
) -> Any:
    from chartreux.app_server.models import PublicTurnStatus

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        turn = session.state.latest_turn
        if (
            turn is not None
            and turn.queue_item_id == queue_item_id
            and turn.status is not PublicTurnStatus.IN_PROGRESS
        ):
            return turn
        await asyncio.sleep(0.05)
    raise AssertionError("The queued turn was never promoted to completion")


def _settled_turns(session: Any) -> list[Any]:
    from chartreux.app_server.models import PublicTurnStatus

    return [
        turn
        for turn in session.state.turns or ()
        if turn.status is not PublicTurnStatus.IN_PROGRESS
    ]


async def _await_queued_turn(
    session: Any, queue_item_id: str, *, expected: int = 1, timeout: float = 30.0
) -> Any:
    """The settled turn a queue item became, once it has settled that often.

    A record that failed and was retained takes its queue item into a second
    turn, so the item can name more than one settled turn: the last one is the
    retry.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        settled = [
            turn
            for turn in _settled_turns(session)
            if turn.queue_item_id == queue_item_id
        ]
        if len(settled) >= expected:
            return settled[-1]
        await asyncio.sleep(0.05)
    raise AssertionError("The queued turn never settled")


def _drain_turn(
    session: Any, message: str
) -> tuple[asyncio.Task[None], list[Any], list[str]]:
    """Consume one turn in the background, recording its events and callbacks."""
    from chartreux.app_server.events import CallbackRequested

    events: list[Any] = []
    callback_ids: list[str] = []

    async def drain() -> None:
        async for event in session.act(message):
            events.append(event)
            if isinstance(event, CallbackRequested):
                callback_ids.append(event.callback.callback_id)

    return asyncio.create_task(drain()), events, callback_ids


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_idle_queued_turn_is_promoted_in_the_root_agent_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.models import PublicTurnStatus

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            queued = await session.enqueue("queued prompt")
            turn = await _await_promoted_turn(session, queued.id)
            queue = await session.read_turn_queue()
            history = _history_text(session.history)
            process = backends[0].process.process
            await session.close()
        finally:
            await harness.close()

    # The item left the FIFO because a real turn took its place, and that turn
    # ran where the transcript is written: the agent process.
    assert turn.status is PublicTurnStatus.COMPLETED
    assert queue.items == []
    assert queue.paused is False
    assert "queued prompt" in history
    assert "done" in history
    assert process.poll() is not None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_queued_turn_is_read_replaced_and_removed_while_a_turn_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _queue_command(tmp_path),
        )
        _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            turn_task, events, callback_ids = _drain_turn(session, "ask me")
            await _wait_for(lambda: bool(callback_ids))

            queued = await session.enqueue("queued prompt")
            queue = await session.read_turn_queue()
            assert [item.id for item in queue.items] == [queued.id]
            assert _queue_texts(queue) == ["queued prompt"]

            replaced = await session.replace_queued_turn(queued.id, "replaced prompt")
            assert replaced is not None
            assert replaced.id == queued.id
            # A replacement keeps the item's own identity and its queue place.
            queue = await session.read_turn_queue()
            assert [item.id for item in queue.items] == [queued.id]
            assert _queue_texts(queue) == ["replaced prompt"]

            assert await session.remove_queued_turn(queued.id) is True
            assert (await session.read_turn_queue()).items == []

            await session.respond_to_callback(callback_ids[0], _user_answer("Yes"))
            await asyncio.wait_for(turn_task, timeout=60)
            history = _history_text(session.history)
            await session.close()
        finally:
            await harness.close()

    # A queued item is a waiting turn while another turn runs: it is readable,
    # editable and removable without ever being promoted, and the removal is
    # what kept its prompt out of the transcript.
    assert "done" in _assistant_text(events)
    assert "replaced prompt" not in history
    assert "queued prompt" not in history


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_interrupted_turn_pauses_the_queue_and_resume_promotes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.models import PublicTurnStatus

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _queue_command(tmp_path),
        )
        _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            turn_task, _, callback_ids = _drain_turn(session, "ask me")
            await _wait_for(lambda: bool(callback_ids))

            queued = await session.enqueue("queued prompt")
            await session.interrupt()
            await _wait_for(lambda: session.turn_queue.paused)
            await asyncio.wait_for(turn_task, timeout=60)

            # The interrupt stopped the running turn; the user's queued work
            # waits instead of starting over the stop.
            paused = await session.read_turn_queue()
            assert paused.paused is True
            assert [item.id for item in paused.items] == [queued.id]

            resumed = await session.resume_turn_queue()
            assert resumed.paused is False
            turn = await _await_promoted_turn(session, queued.id)
            history = _history_text(session.history)
            final_queue = await session.read_turn_queue()
            await session.close()
        finally:
            await harness.close()

    assert turn.status is PublicTurnStatus.COMPLETED
    assert "queued prompt" in history
    assert final_queue.items == []
    assert final_queue.paused is False


#: A child whose manual `!` records are redacted against one synthetic value.
#: The one record builder runs in the agent process, so the policy naming the
#: value is bound there; the value's own spelling is only a fixture.
_SHELL_REDACTION_PROLOGUE = """\
from chartreux.core.tools import secret_redaction as _redaction

_POLICY = _redaction.ScrubPolicy(
    redaction_credentials=(("CHARTREUX_TEST_CREDENTIAL_KEY", {value!r}),)
)
_redaction.current_policy = lambda: _POLICY
"""


async def _run_shell(session: Any, command: str) -> list[Any]:
    """Run one manual ``!`` command through the client's shell resource."""
    events: list[Any] = []
    async with aclosing(session.resources.shell.run(command)) as stream:
        async for event in stream:
            events.append(event)
    return events


async def _await_history_text(session: Any, text: str, timeout: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if text in _history_text(session.history):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"The transcript never carried {text!r}")


async def _await_settled_turn(session: Any, timeout: float = 30.0) -> Any:
    """The latest turn, once it is terminal.

    A turn's last history entry can be projected before its completion frame,
    so a status read taken right after the transcript settles may still see the
    turn in progress.
    """
    from chartreux.app_server.models import PublicTurnStatus

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        turn = session.state.latest_turn
        if turn is not None and turn.status is not PublicTurnStatus.IN_PROGRESS:
            return turn
        await asyncio.sleep(0.05)
    raise AssertionError("The turn never settled")


async def _rebuilt_shell_entries(backends: list[Any], session: Any) -> list[Any]:
    """The shell entries a timeline rebuilt from the agent's transcript shows.

    The transcript lives in the root agent process, so the rebuild is a session
    read: the record's metadata rides the committed message, and that metadata
    is what restores the shell tile beside the record text.
    """
    from chartreux.app_server.protocol import SessionReadParams

    read = await backends[0].read(
        SessionReadParams(session_id=session.state.session.id)
    )
    return [
        entry
        for entry in read.state.history or []
        if getattr(entry, "title", None) == "shell"
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_manual_shell_command_executes_and_promotes_a_user_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.models import CompletedEffectState, PublicTurnStatus

    synthetic_value = "fixture-only-redaction-value"
    marker = tmp_path / "manual-shell-ran"
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(
                tmp_path,
                prologue=_SHELL_REDACTION_PROLOGUE.format(value=synthetic_value),
            ),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            command = f"printf '%s' {synthetic_value} > {marker} && cat {marker}"
            await _run_shell(session, command)
            await _await_history_text(session, "done")
            turn = await _await_settled_turn(session)
            history = _history_text(session.history)
            queue = await session.read_turn_queue()
            rebuilt = await _rebuilt_shell_entries(backends, session)
            await session.close()
        finally:
            await harness.close()

    # The command ran in the agent process, and its record -- the unified
    # redacted text -- is the user turn the queue promoted in its own place.
    assert marker.read_text() == synthetic_value
    assert "Manual `!` command result from the user." in history
    assert "Exit code: 0" in history
    assert "[REDACTED]" in history
    assert synthetic_value not in history
    assert "done" in history
    assert turn is not None
    assert turn.status is PublicTurnStatus.COMPLETED
    assert turn.queue_item_id is not None
    assert queue.items == []
    # Item 5 B.2 metadata carriage: the record's context crossed the wire with
    # the reply and rides the committed message, so a timeline rebuilt from the
    # transcript restores the shell tile beside the record text.
    assert len(rebuilt) == 1
    state = rebuilt[0].state
    assert isinstance(state, CompletedEffectState)
    # The context crossed the wire redacted, exactly as the record text did, so
    # the restored tile shows the same value the record carries.
    assert isinstance(state.output, dict)
    assert state.output["stdout"] == "[REDACTED]"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_failed_manual_shell_command_promotes_its_record_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.models import PublicTurnStatus

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            await _run_shell(session, "sh -c 'exit 3'")
            await _await_history_text(session, "done")
            turn = await _await_settled_turn(session)
            history = _history_text(session.history)
            await session.close()
        finally:
            await harness.close()

    # A nonzero exit is a record, not a refusal: it still becomes the turn.
    assert "Manual `!` command result from the user." in history
    assert "Exit code: 3" in history
    assert turn is not None
    assert turn.status is PublicTurnStatus.COMPLETED
    assert turn.queue_item_id is not None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_provider_failure_retains_the_shell_record_on_stdio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 5 B.2a on the production topology: the record is never lost.

    The command's effect runs where the loop lives and its record turn fails at
    the provider afterwards. The record is then retained as a ready queued user
    turn -- the effect is never rerun to recover it, and the next admission runs
    the retained turn from the message already committed.
    """
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.models import PublicTurnStatus

    runs = tmp_path / "retained-command-runs"
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _faulty_provider_command(tmp_path),
        )
        _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            await _run_shell(session, f"printf ran >> {runs} && echo recorded")
            await _wait_for(
                lambda: (
                    session.state.latest_turn is not None
                    and session.state.latest_turn.status is PublicTurnStatus.FAILED
                    and bool(session.turn_queue.items)
                )
            )
            failed = session.state.latest_turn
            assert failed is not None
            retained = _queue_texts(session.turn_queue)
            committed_before = _history_text(session.history)
            assert runs.read_text() == "ran"

            await session.resume_turn_queue()
            await _wait_for(
                lambda: (
                    session.state.latest_turn is not None
                    and session.state.latest_turn.id != failed.id
                    and session.state.latest_turn.status
                    is not PublicTurnStatus.IN_PROGRESS
                )
            )
            promoted = session.state.latest_turn
            committed_after = _history_text(session.history)
            await session.close()
        finally:
            await harness.close()

    # The effect ran exactly once: retention re-queues the record, never the
    # command, so a failed turn cannot repeat an executed command.
    assert runs.read_text() == "ran"
    # The record is a ready queued user turn, not a lost outcome.
    assert len(retained) == 1
    assert retained[0].startswith("Manual `!` command result from the user.")
    assert "Command: `printf ran" in retained[0]
    assert committed_before.count("Manual `!` command result from the user.") == 1
    # The next admission promoted the retained record as a turn in its own
    # queue place, and the record text was not committed a second time.
    assert promoted is not None
    assert promoted.queue_item_id is not None
    assert promoted.status is PublicTurnStatus.FAILED
    assert committed_after.count("Manual `!` command result from the user.") == 1


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_record_promoted_after_a_later_turn_runs_on_its_own_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-A2 on the production topology: the retried record is the turn's input.

    The record turn fails at the provider, so its record is retained while the
    prompt queued beside it runs. The retained record is then promoted over a
    transcript that prompt already extended, so it must carry the record text as
    its own user input instead of running on "no new input" over that tail.
    """
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.models import PublicTurnStatus

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _shell_order_command(tmp_path, fail_first=True),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            entered, release = _gate_shell_commit(monkeypatch, backends[0])
            shell_task = asyncio.create_task(_run_shell(session, "printf recorded"))
            await _wait_for(lambda: entered.is_set())
            # The record item is still the queue's head, unpromoted: the prompt
            # takes its place behind it, so it runs first when the record fails.
            prompt = await session.enqueue("typed prompt")
            release.set()
            await shell_task

            # Three turns settle: the record turn that failed, the prompt that
            # ran while the record waited, and the retained record's own retry.
            await _wait_for(lambda: len(_settled_turns(session)) == 3)
            promoted = _settled_turns(session)[-1]
            history = _history_text(session.history)
            queue = await session.read_turn_queue()
            await session.close()
        finally:
            await harness.close()

    record = "Manual `!` command result from the user."
    # The record committed once before the failure and, because another turn ran
    # in between, the retained record ran again as its own user input.
    assert history.count(record) == 2
    assert "typed prompt" in history
    assert (
        history.index(record) < history.index("typed prompt") < history.rindex(record)
    )
    assert promoted.queue_item_id not in (None, prompt.id)
    assert promoted.status is PublicTurnStatus.COMPLETED
    assert queue.items == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_retained_record_is_read_and_drained_after_a_failed_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1 + C3 on the production topology.

    The record turn fails while the queue is full, so its record waits beside
    the FIFO. The read endpoint must answer that pending state (the event stream
    already does), and when the prompt that took the last place fails too, that
    terminal must re-offer the waiting record -- a drain can never strand it.
    """
    from chartreux.app_server import process_session_backend
    from chartreux.app_server._turn_queue import TurnQueue
    from chartreux.app_server.models import PublicTurnStatus
    from chartreux.app_server.protocol import (
        SessionTextContentBlock,
        TurnUserInputEntry,
    )

    record = "Manual `!` command result from the user."
    runs = tmp_path / "retained-command-runs"
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _delayed_failure_command(tmp_path, hold=1.5),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            backend = backends[0]
            # One queue place is the condition the retention path needs: the
            # record's turn has popped its item, the prompt takes the only place,
            # and the record then has nowhere to go but beside the FIFO.
            backend._turn_queue = TurnQueue(max_items=1)
            shell_task = asyncio.create_task(
                _run_shell(session, f"printf ran >> {runs} && echo recorded")
            )
            await _wait_for(
                lambda: backend._active_turn is not None and not backend._turn_queue
            )
            await asyncio.wait_for(shell_task, timeout=30)
            prompt = await session.enqueue("typed prompt")
            await _wait_for(lambda: bool(backend._shell_retained))

            # C1: the read answers the pending state the notifications carry.
            queue = await session.read_turn_queue()
            retained_entry_id = next(iter(backend._shell_retained))
            assert [item.id for item in queue.retained] == [retained_entry_id]
            retained_entry = queue.retained[0].entries[0]
            assert isinstance(retained_entry, TurnUserInputEntry)
            assert retained_entry.content == [
                SessionTextContentBlock(
                    text=backend._shell_retained[retained_entry_id].text
                )
            ]
            # The record is beside the FIFO, not in it: a read that returned the
            # raw FIFO would answer an empty, workless queue here.
            assert backend._shell_retained[retained_entry_id].text not in _queue_texts(
                queue
            )

            # C3: the prompt's own turn fails next, and its terminal is the
            # admission that re-offers the record: the record runs, so nothing is
            # left stranded once the queue drains.
            await _wait_for(lambda: len(_settled_turns(session)) >= 3)
            settled = _settled_turns(session)
            history = _history_text(session.history)
            final_queue = await session.read_turn_queue()
            await session.close()
        finally:
            await harness.close()

    # The effect ran exactly once: retention re-queues the record, never the
    # command, and a drain after an intervening failure never reruns it.
    assert runs.read_text() == "ran"
    assert backend._shell_retained == {}
    assert final_queue.retained == []
    assert final_queue.items == []
    assert settled[-1].status is PublicTurnStatus.COMPLETED
    assert settled[-1].queue_item_id not in (None, prompt.id)
    # The record committed once, the failed prompt ran, and the retried record
    # turn carried the record text as its own user input.
    assert history.count(record) == 2
    assert "typed prompt" in history


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_removal_during_the_record_commit_loses_no_queued_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-G1 on the production topology: a removal cannot steal a promotion.

    The promotion pass has peeked the record item and is committing its message
    when a removal for that same item arrives. Removal is serialized with
    promotion, so the pass pops the item it peeked and the prompt behind it
    keeps its place and its work instead of being popped and dropped.
    """
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.models import PublicTurnStatus

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _shell_order_command(tmp_path, fail_first=False),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            reported: list[dict[str, Any]] = []
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(lambda _loop, context: reported.append(context))
            entered, release = _gate_shell_commit(monkeypatch, backends[0])
            shell_task = asyncio.create_task(_run_shell(session, "printf recorded"))
            await _wait_for(lambda: entered.is_set())
            record_item_id = session.turn_queue.items[0].id
            behind = await session.enqueue("queued prompt")
            removal = asyncio.create_task(session.remove_queued_turn(record_item_id))
            # The removal arrives inside the peek, commit and pop window; it
            # waits there instead of changing the head under the pass.
            await asyncio.sleep(0.2)
            assert removal.done() is False
            release.set()
            await shell_task
            await asyncio.wait_for(removal, timeout=30)
            record_turn = await _await_queued_turn(session, record_item_id)
            behind_turn = await _await_queued_turn(session, behind.id)
            history = _history_text(session.history)
            queue = await session.read_turn_queue()
            loop.set_exception_handler(None)
            await session.close()
        finally:
            await harness.close()

    # The pass popped the item it peeked: no queue task failed over the removal.
    assert [
        context
        for context in reported
        if context.get("message") == "Turn queue task failed"
    ] == []
    assert record_turn.status is PublicTurnStatus.COMPLETED
    # The prompt behind the record still ran: no popped item lost its work.
    assert behind_turn.status is PublicTurnStatus.COMPLETED
    assert "queued prompt" in history
    assert queue.items == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_interrupted_queued_shell_command_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    marker = tmp_path / "interrupted-command-ran"
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _queue_command(tmp_path),
        )
        _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            turn_task, _, callback_ids = _drain_turn(session, "ask me")
            await _wait_for(lambda: bool(callback_ids))

            # The epoch is busy, so the command takes its FIFO place as a
            # shell-input kind instead of running under the running turn.
            shell_task = asyncio.create_task(
                _run_shell(session, f"printf ran > {marker}")
            )
            await _wait_for(lambda: bool(session.turn_queue.items))
            held = session.turn_queue
            shell_task.cancel()
            with suppress(asyncio.CancelledError):
                await shell_task
            await _wait_for(lambda: not session.turn_queue.items)

            await session.respond_to_callback(callback_ids[0], _user_answer("Yes"))
            await asyncio.wait_for(turn_task, timeout=60)
            queue = await session.read_turn_queue()
            history = _history_text(session.history)
            await session.close()
        finally:
            await harness.close()

    # The interrupted item never executed: the command ran no process, and no
    # record turn was promoted over the user's stop.
    assert [entry.role for item in held.items for entry in item.entries] == ["shell"]
    assert queue.items == []
    assert not marker.exists()
    assert "Manual `!` command result from the user." not in history


def _record_backends(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the session backends one harness materializes."""
    from chartreux.app_server.process_session_backend import ProcessSessionBackendHost

    recorded: list[Any] = []

    def _recorder(name: str) -> None:
        real = getattr(ProcessSessionBackendHost, name)

        async def _recording(self: Any, params: Any) -> Any:
            result = await real(self, params)
            recorded.append(result.backend)
            return result

        monkeypatch.setattr(ProcessSessionBackendHost, name, _recording)

    # A resume and a continue open a backend exactly as a fresh start does,
    # so the recorder covers every session lifecycle route.
    for name in ("start", "resume", "continue_latest"):
        _recorder(name)
    return recorded


def _assistant_text(events: list[Any]) -> str:
    parts: list[str] = []
    for event in events:
        entry = getattr(event, "entry", None)
        if getattr(entry, "role", None) != "assistant":
            continue
        for block in getattr(entry, "content", ()) or ():
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _entry_text(entries: list[Any]) -> str:
    parts: list[str] = []
    for added in entries:
        for block in getattr(added.entry, "content", ()) or ():
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Compaction settlement: the three agent-minted receipt classes and the
# coordinator-minted timeout, driven through the production entrypoint.
#
# The child is a real spawned root agent whose compaction body is scripted, so
# the settlement record, the marker table and the reply are the production
# ones: only the summarization is a stand-in (the transport is blocked).
# ---------------------------------------------------------------------------

_COMPACT_CHILD_SCRIPT = """\
import asyncio
from pathlib import Path

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config

_BODIES = Path({bodies!r})
_FENCE_ABORT = Path({fence_abort!r})
_MODE = {mode!r}
_LATE_DELAY = {late_delay!r}
_STALL = {stall!r}


class _Factory:
    def __init__(self, loop):
        self._loop = loop

    async def open(self, params):
        return self._loop

    async def close(self):
        pass


async def _compact(extra="", *, manual=False):
    with _BODIES.open("a", encoding="utf-8") as handle:
        handle.write(extra + "\\n")
    if _MODE == "settled":
        return "SUMMARY"
    if _MODE == "published-unsettled":
        loop.compaction_outcome.published = True
        raise RuntimeError("the handoff state could not be read")
    if _MODE == "never-settles":
        await asyncio.Event().wait()
        return "unreachable"
    if _MODE == "late":
        try:
            await asyncio.sleep(_LATE_DELAY)
        except asyncio.CancelledError:
            _FENCE_ABORT.write_text("cancelled at the deadline")
            await asyncio.sleep(_LATE_DELAY)
        return "LATE SUMMARY"
    raise AssertionError(_MODE)


if _STALL:
    import time

    from chartreux.app_server.root_agent import RootAgentServer

    _real_serve_request = RootAgentServer._serve_request

    async def _serve_request(self, request_id, method, message):
        # Hold the agent's loop past the request's deadline before the fence
        # ever sees the request: the admission is late by construction.
        if method == "agent/compact":
            time.sleep(_STALL)
        await _real_serve_request(self, request_id, method, message)

    RootAgentServer._serve_request = _serve_request


init_harness_files_manager("user", "project")
config = build_test_vibe_config(
    session_logging=SessionLoggingConfig(enabled=True, save_dir={save_dir!r})
)
loop = build_test_agent_loop(config=config)
loop.compact = _compact
asyncio.run(serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested()))
"""


def _compact_command(
    tmp_path: Path, mode: str, *, late_delay: float = 0.0, stall: float = 0.0
) -> list[str]:
    return [
        sys.executable,
        "-c",
        _COMPACT_CHILD_SCRIPT.format(
            save_dir=str(tmp_path / "sessions"),
            bodies=str(tmp_path / "compact-bodies.log"),
            fence_abort=str(tmp_path / "fence-abort"),
            mode=mode,
            late_delay=late_delay,
            stall=stall,
        ),
    ]


def _shrink_compaction_budget(
    monkeypatch: pytest.MonkeyPatch, *, deadline: float, grace: float
) -> None:
    """Give the spawned coordinator handle a two-sided budget a test can wait on.

    The deadline the coordinator mints and the deadline the agent arms are the
    SAME value (the request carries the absolute one), so shrinking it widens
    neither side: it only makes the wait observable in a test.
    """
    from chartreux.app_server import process_session_backend

    real_spawn = process_session_backend.spawn_root_agent_process

    async def _spawn(**kwargs: Any) -> Any:
        process = await real_spawn(**kwargs)
        process._compact_deadline = deadline
        process._compact_reply_grace = grace
        return process

    monkeypatch.setattr(process_session_backend, "spawn_root_agent_process", _spawn)


async def _await_compact_marker(
    process: Any, *, state: str, timeout: float = 20.0
) -> Any:
    """Read snapshots until the agent's marker table carries one in this state."""
    from chartreux.app_server.agent_wire import AGENT_COMPACT_METHOD

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        snapshot = await process.snapshot(include_history=False)
        found = [
            marker
            for marker in snapshot.markers
            if marker.method == AGENT_COMPACT_METHOD and marker.state == state
        ]
        if found:
            return found[-1]
        await asyncio.sleep(0.02)
    raise AssertionError(f"no {state} compaction marker appeared in the snapshot")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_compaction_settles_published_and_settled_at_the_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from chartreux.app_server import process_session_backend
    from chartreux.app_server.agent_wire import (
        AGENT_COMPACT_METHOD,
        COMPACT_SETTLEMENT_PUBLISHED_SETTLED,
    )
    from chartreux.app_server.root_agent_process import COMPACT_DEADLINE

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _compact_command(tmp_path, "settled"),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            before = time.monotonic()
            summary = await session.resources.sessions.compact("focus")
            process = backends[0].process
            marker = await _await_compact_marker(process, state="terminal")
            settlement = process.compaction_settlement(marker.request_id)
            await session.close()
        finally:
            await harness.close()

    # The receipt the client reads is the agent's: a summary crossed the pipe
    # from the process that ran the compaction body.
    assert summary == "SUMMARY"
    assert marker.method == AGENT_COMPACT_METHOD
    assert marker.outcome == COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    # The coordinator's absolute deadline is the one the agent admitted over:
    # the value is the sized long-op budget, not a second budget of its own.
    assert (
        before + COMPACT_DEADLINE <= marker.deadline <= before + COMPACT_DEADLINE + 60
    )
    # The one-shot record this end holds agrees with the marker the agent
    # wrote, and it is terminal, so nothing is left to re-check.
    assert settlement is not None
    assert settlement.settlement == COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    assert settlement.pending is False
    assert settlement.request_id == marker.request_id
    assert (tmp_path / "compact-bodies.log").read_text() == "focus\n"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_published_but_unsettled_compaction_keeps_its_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.agent_wire import COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED
    from chartreux.app_server.protocol import ProtocolErrorCode

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _compact_command(tmp_path, "published-unsettled"),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            with pytest.raises(AppServerResponseError) as caught:
                await session.resources.sessions.compact("focus")
            process = backends[0].process
            marker = await _await_compact_marker(process, state="terminal")
            settlement = process.compaction_settlement(marker.request_id)
            await session.close()
        finally:
            await harness.close()

    # A compaction that published its boundary still owes its settlement: the
    # class the agent recorded is the published-but-unsettled one, and the
    # client is refused (no summary exists to hand it) rather than told a
    # summarization never happened.
    assert caught.value.error.code is ProtocolErrorCode.COMPACTION_FAILED
    assert marker.outcome == COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED
    assert settlement is not None
    assert settlement.settlement == COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED
    assert settlement.pending is False


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_cancel_notification_settles_the_compaction_unpublished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.agent_wire import COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
    from chartreux.app_server.protocol import ProtocolErrorCode

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _compact_command(tmp_path, "never-settles"),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            process = backends[0].process
            pending = asyncio.create_task(session.resources.sessions.compact("focus"))
            admitted = await _await_compact_marker(process, state="active")
            # The cancel is an id-less notification: it names the op, carries
            # no request id, and its outcome is the operation's own settlement.
            await process.cancel_operation(admitted.op_id)
            with pytest.raises(AppServerResponseError) as caught:
                await asyncio.wait_for(pending, timeout=30)
            marker = await _await_compact_marker(process, state="terminal")
            settlement = process.compaction_settlement(marker.request_id)
            await session.close()
        finally:
            await harness.close()

    # Nothing was published, so the compaction settles cancelled-unpublished
    # under the request id the coordinator is still waiting on -- there is no
    # cancel reply to read, only the settlement class.
    assert caught.value.error.code is ProtocolErrorCode.COMPACTION_FAILED
    assert marker.request_id == admitted.request_id
    assert marker.outcome == COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
    assert settlement is not None
    assert settlement.settlement == COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
    assert settlement.pending is False
    assert "cancelled the compaction" in settlement.detail
    assert (tmp_path / "compact-bodies.log").read_text() == "focus\n"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_coordinator_timeout_resolves_in_flight_to_the_terminal_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.agent_wire import COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    from chartreux.app_server.protocol import ProtocolErrorCode
    from chartreux.app_server.root_agent_process import COMPACT_OBSERVATION_IN_FLIGHT

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _compact_command(tmp_path, "late", late_delay=0.8),
        )
        _shrink_compaction_budget(monkeypatch, deadline=0.25, grace=0.1)
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            process = backends[0].process
            with pytest.raises(AppServerResponseError) as caught:
                await session.resources.sessions.compact("focus")
            # The re-check ran against a payload constructed after the
            # deadline: the record was ADMITTED (an admission handshake the
            # coordinator recorded), so its ACTIVE marker settles provisionally
            # in flight rather than being claimed as a settled compaction.
            record = await _await_compaction_record(process, "in-flight")
            provisional = record.settlement
            assert provisional == COMPACT_OBSERVATION_IN_FLIGHT
            assert record.pending is True
            # The body finished late; the next consumed payload carries the
            # TERMINAL record and resolves the observation without a resend.
            resolved = await _await_compaction_record(
                process, COMPACT_SETTLEMENT_PUBLISHED_SETTLED
            )
            # The agent enforced its own side of the deadline: the body was
            # cancelled at the fence before its natural completion.
            assert (tmp_path / "fence-abort").read_text() == "cancelled at the deadline"
            # A late reply never re-settles a request: it is recorded instead.
            diagnostics = [item.message() for item in process.diagnostics]
            assert any("answers no request" in message for message in diagnostics)
            # The fence is free again: a second operation is admitted and runs
            # its own body instead of being refused as a conflict.
            with pytest.raises(AppServerResponseError):
                await session.resources.sessions.compact("again")
            bodies = (tmp_path / "compact-bodies.log").read_text().splitlines()
            await session.close()
        finally:
            await harness.close()

    assert caught.value.error.code is ProtocolErrorCode.COMPACTION_FAILED
    assert resolved.pending is False
    assert resolved.settlement == COMPACT_SETTLEMENT_PUBLISHED_SETTLED
    # One body ran per request: the timeout path never resent the operation.
    assert bodies == ["focus", "again"]


async def _await_compaction_record(process: Any, settlement: str) -> Any:
    """Read the coordinator's settlement table until it names this class."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 20.0
    while loop.time() < deadline:
        for record in process._compactions.values():
            if record.settlement == settlement:
                return record
        await process.snapshot(include_history=False)
        await asyncio.sleep(0.02)
    raise AssertionError(f"no compaction settlement reached {settlement!r}")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_request_past_its_absolute_deadline_never_starts_a_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol import ProtocolErrorCode

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _compact_command(tmp_path, "settled", stall=0.2),
        )
        # The deadline is minted at send and travels as the same absolute
        # value the agent admits over; the stalled agent finds it spent.
        _shrink_compaction_budget(monkeypatch, deadline=0.05, grace=0.5)
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            with pytest.raises(AppServerResponseError) as caught:
                await session.resources.sessions.compact("focus")
            process = backends[0].process
            snapshot = await process.snapshot(include_history=False)
            await session.close()
        finally:
            await harness.close()

    # The agent refused the admission typed instead of starting a body it
    # could not finish: the two-sided deadline is enforced at the fence.
    assert caught.value.error.code is ProtocolErrorCode.INTERNAL_ERROR
    assert "deadline" in caught.value.error.message
    assert snapshot.markers == ()
    assert not (tmp_path / "compact-bodies.log").exists()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_marker_table_keeps_every_record_within_one_incarnation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _compact_command(tmp_path, "settled"),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            first = await session.resources.sessions.compact("one")
            second = await session.resources.sessions.compact("two")
            process = backends[0].process
            snapshot = await process.snapshot(include_history=False)
            records = [
                marker for marker in snapshot.markers if marker.state == "terminal"
            ]
            incarnations = {marker.op_id.split(":")[0] for marker in records}
            await session.close()
        finally:
            await harness.close()

    # Records are never pruned within an incarnation, so absence always means
    # never-admitted and both compactions keep their evidence in one read.
    assert (first, second) == ("SUMMARY", "SUMMARY")
    assert len(records) == 2
    assert len(incarnations) == 1
    assert incarnations == {snapshot.incarnation_id}
    assert [record.request_id for record in records] == sorted(
        record.request_id for record in records
    ) or len({record.request_id for record in records}) == 2


# ---------------------------------------------------------------------------
# Resume: a fresh process adopts a stored session through the shipped open
# path (lease, transcript load, accounting), not a live re-attach.
# ---------------------------------------------------------------------------

_RESUME_CHILD_SCRIPT = """\
import asyncio
from pathlib import Path
from unittest.mock import Mock

from chartreux.app_server import _runtime as runtime
from chartreux.app_server.agent_wire import decode_open_request
from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.protocol import AgentConfig
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import (
    get_harness_files_manager,
    init_harness_files_manager,
)
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


class _Factory:
    def __init__(self, process):
        self._process = process
        self.open_accounting = None

    async def open(self, params):
        request = decode_open_request(params)
        loop = await self._process.open_root(
            runtime.RootOpenRequest(
                options=AgentConfig.model_validate(request.agent_config),
                client_info=request.client_info,
                client_capabilities=request.client_capabilities,
                session_id=request.session_id,
                continue_latest=request.continue_latest,
            )
        )
        self.open_accounting = self._process.last_open_accounting
        return loop

    async def replace(self, loop, *, session_id, continue_latest=False):
        # The production rebind: the transcript, the lease and the registry all
        # load in THIS process, and the loop object survives the identity swap.
        from chartreux.app_server.agent_wire import AgentSessionNotFoundError
        from chartreux.app_server._runtime import RuntimeSessionNotFoundError

        try:
            await self._process.rebind_root(
                loop, session_id=session_id, continue_latest=continue_latest
            )
        except RuntimeSessionNotFoundError as exc:
            # The production factory's own mapping: an unknown stored session
            # is a typed not-found on the wire, never a partial rebind.
            raise AgentSessionNotFoundError(str(exc)) from exc
        self.open_accounting = runtime.RootOpenAccounting.from_loop(loop)

    async def close(self):
        await self._process.close()


init_harness_files_manager("user", "project")
config = build_test_vibe_config(
    session_logging=SessionLoggingConfig(enabled=True, save_dir={save_dir!r})
)


async def _build_blueprint(self, options, client_info, client_capabilities):
    blueprint = Mock()
    blueprint.config = config
    blueprint.config_orchestrator = FakeConfigOrchestrator(config)
    blueprint.harness_files = get_harness_files_manager()
    blueprint.cwd = Path({cwd!r})
    blueprint.build.side_effect = lambda **kwargs: build_test_agent_loop(
        config=config,
        backend=FakeBackend([[mock_llm_chunk(content={reply!r})]]),
        **kwargs,
    )
    return blueprint


runtime.HarnessProcess.build_root_blueprint = _build_blueprint
asyncio.run(
    serve_root_agent(
        factory=_Factory(runtime.HarnessProcess()), announce_ready=bootstrap_requested()
    )
)
"""


def _resume_command(
    tmp_path: Path, *, cwd: Path | None = None, reply: str = "resumed"
) -> list[str]:
    return [
        sys.executable,
        "-c",
        _RESUME_CHILD_SCRIPT.format(
            save_dir=str(tmp_path / "sessions"),
            cwd=str(cwd or tmp_path / "sessions"),
            reply=reply,
        ),
    ]


async def _persist_test_session(tmp_path: Path) -> str:
    """Persist one stored session the fresh process will adopt."""
    from chartreux.core.config import SessionLoggingConfig as LoggingConfig
    from tests.conftest import build_test_agent_loop, build_test_vibe_config

    config = build_test_vibe_config(
        session_logging=LoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    )
    loop = build_test_agent_loop(config=config)
    await loop.persist_empty_session()
    session_id = loop.session_id
    return session_id


def _resume_options(tmp_path: Path, session_id: str) -> LocalHarnessOptions:
    from chartreux.app_server.local import ResumeSessionIntent

    return LocalHarnessOptions(
        client=_client(),
        session_options=SessionOptions(cwd=str(tmp_path), headless=True),
        session=ResumeSessionIntent(session_id),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_fresh_process_resumes_a_stored_session_at_the_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    session_id = await _persist_test_session(tmp_path)
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _resume_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_resume_options(tmp_path, session_id))
        try:
            session = await harness.start()
            process = backends[0].process
            assert session.session_id == session_id
            events = await _collect_turn(session, "hello")
            accounting = process.open_accounting
            session_dir = process.session_dir
            jobs_dir = process.jobs_dir
            pid = process.process.pid
            await session.close()
        finally:
            await harness.close()

    # The open reply names the session the AGENT loaded, with the directories
    # and accounting records that open adopted: this is the resume round-trip,
    # not a coordinator-side reconstruction.
    assert "resumed" in _assistant_text(events)
    assert accounting is not None
    assert accounting.root_session_id == session_id
    assert accounting.adoption == "none"
    assert accounting.settlements == ()
    assert session_dir is not None and session_id[:8] in Path(session_dir).name
    assert Path(session_dir).is_dir()
    assert jobs_dir == str(Path(session_dir) / "jobs")
    assert pid != os.getpid()
    # The moved transcript is on disk where the agent process wrote it.
    assert any((tmp_path / "sessions").rglob("meta.json"))


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_transcript_a_resumed_process_wrote_is_read_back_by_the_next_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.local import ResumeSessionIntent

    session_id = await _persist_test_session(tmp_path)
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _resume_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        first = LocalHarness(_resume_options(tmp_path, session_id))
        try:
            session = await first.start()
            await _collect_turn(session, "a stored prompt")
            first_pid = backends[-1].process.process.pid
            await session.close()
        finally:
            await first.close()

        second = LocalHarness(
            LocalHarnessOptions(
                client=_client(),
                session_options=SessionOptions(cwd=str(tmp_path), headless=True),
                session=ResumeSessionIntent(session_id),
            )
        )
        try:
            resumed = await second.start()
            history = _history_text(resumed.history)
            second_pid = backends[-1].process.process.pid
            await resumed.close()
        finally:
            await second.close()

    # The second process is a different root that loaded the first one's saved
    # turn from disk: a fresh-process resume, not a live handle.
    assert len(backends) == 2
    assert second_pid != first_pid
    assert "a stored prompt" in history
    assert "resumed" in history


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_lease_contented_resume_fails_typed_and_names_its_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.protocol import ProtocolErrorCode

    session_id = await _persist_test_session(tmp_path)
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _resume_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        holder = LocalHarness(_resume_options(tmp_path, session_id))
        try:
            session = await holder.start()
            holder_pid = backends[-1].process.process.pid
            contender = LocalHarness(_resume_options(tmp_path, session_id))
            try:
                with pytest.raises(AppServerResponseError) as caught:
                    await contender.start()
            finally:
                await contender.close()
            # The lease record is the holder's, read while it still holds it.
            lease_file = tmp_path / "sessions" / "active" / f"{session_id}.lock"
            lease = json.loads(lease_file.read_text(encoding="utf-8"))
            await session.close()
        finally:
            await holder.close()

    # The second process could not take the lease, so its open failed before
    # any session was adopted -- and the failure names the session the record
    # holds, not an anonymous conflict.
    assert caught.value.error.code is ProtocolErrorCode.INTERNAL_ERROR
    assert "could not start" in caught.value.error.message
    assert session_id in caught.value.error.message
    assert len(backends) == 1
    # The holder is surfaced by the shipped lease record.
    assert lease["session_id"] == session_id
    assert lease["process_id"] == holder_pid
    assert lease["acquired_at"]


# ---------------------------------------------------------------------------
# Identity replacement: the coordinator leg (reap the old cohort, then rebind)
# drives the spawned root agent, which owns the session the rebind swaps.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_replacement_reaps_the_cohort_then_rebinds_the_same_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server._session_runtime_impl import ProcessRootReplace

    target_id = await _persist_test_session(tmp_path)
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _resume_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            process = backends[0].process
            pid = process.process.pid
            original_id = process.session_id
            before = await process.snapshot(include_history=False)
            reaped: list[tuple[str, str]] = []

            async def _reap() -> object:
                # The reap runs while the OLD identity is still bound: a rebind
                # sent first would leave the old cohort unowned.
                reaped.append(("reap", process.session_id))
                return object()

            reply = await ProcessRootReplace(process=process, reap=_reap).replace(
                session_id=target_id
            )
            replace_request = process.last_request_id
            after = await process.snapshot(include_history=False)
            await session.close()
        finally:
            await harness.close()

    # The coordinator reaped the cohort before the rebind was sent, and the
    # agent rebound to the stored session without a second process existing.
    assert reaped == [("reap", original_id)]
    assert reply.snapshot.state.session.id == target_id
    assert process.process.pid == pid
    assert after.state.session.id == target_id
    assert after.incarnation_id != before.incarnation_id
    # The rebind's own TERMINAL record is the one seed the new incarnation
    # carries: the one-shot ambiguous-timeout re-check always finds it, and it
    # keeps the op id its own (old) incarnation minted.
    assert len(after.markers) == 1
    seed = after.markers[0]
    assert seed.state == "terminal"
    assert seed.outcome == "completed"
    assert seed.request_id == replace_request
    assert seed.op_id.split(":")[0] == before.incarnation_id


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_cohort_that_could_not_be_reaped_sends_no_rebind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server._child_runtime_lifecycle import (
        TreeNotReapedError,
        TreeShutdownReport,
        TreeShutdownStatus,
    )
    from chartreux.app_server._dispatch import RequestFailure
    from chartreux.app_server._session_runtime_impl import ProcessRootReplace
    from chartreux.app_server.protocol import ProtocolErrorCode

    target_id = await _persist_test_session(tmp_path)
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _resume_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            process = backends[0].process
            pid = process.process.pid
            original_id = process.session_id
            before = await process.snapshot(include_history=False)

            async def _reap() -> object:
                raise TreeNotReapedError(
                    TreeShutdownReport(status=TreeShutdownStatus.TIMED_OUT)
                )

            with pytest.raises(RequestFailure) as refusal:
                await ProcessRootReplace(process=process, reap=_reap).replace(
                    session_id=target_id
                )
            after = await process.snapshot(include_history=False)
            await session.close()
        finally:
            await harness.close()

    # The refusal is typed and arrives before any frame: the identity, its
    # incarnation and its marker table are exactly what they were.
    assert refusal.value.code is ProtocolErrorCode.CONFLICT
    assert "cohort reaper" in str(refusal.value)
    assert "shutdown" in str(refusal.value)
    assert process.process.pid == pid
    assert after.state.session.id == original_id
    assert after.incarnation_id == before.incarnation_id
    assert after.markers == before.markers


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_replacement_naming_an_absent_session_keeps_the_old_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server._session_runtime_impl import ProcessRootReplace
    from chartreux.app_server.root_agent_process import RootAgentSessionNotFound

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _resume_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            process = backends[0].process
            original_id = process.session_id
            before = await process.snapshot(include_history=False)

            async def _reap() -> object:
                return object()

            with pytest.raises(RootAgentSessionNotFound) as refusal:
                await ProcessRootReplace(process=process, reap=_reap).replace(
                    session_id="a-session-nobody-stored"
                )
            after = await process.snapshot(include_history=False)
            await session.close()
        finally:
            await harness.close()

    # The agent refused inside the rebind and the old identity survived it:
    # the not-found answer is typed, not a half-bound session.
    assert "a-session-nobody-stored" in str(refusal.value)
    assert after.state.session.id == original_id
    assert after.incarnation_id == before.incarnation_id


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_rename_is_applied_where_the_session_metadata_lives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _resume_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            await _collect_turn(session, "hello")
            renamed = await session.resources.sessions.rename("a-manual-title")
            state_title = session.state.session.title
            process = backends[0].process
            session_id = process.session_id
            await session.close()
        finally:
            await harness.close()

        # The title is written by the process that owns the session directory.
        metadata = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (tmp_path / "sessions").rglob("meta.json")
        ]

    assert renamed == "a-manual-title"
    assert state_title == "a-manual-title"
    assert any(item.get("title") == "a-manual-title" for item in metadata)
    assert session_id


#: A child whose factory refuses the session the coordinator asked it to open:
#: readiness already succeeded, so the failure lands inside the open leg.
_REFUSING_OPEN_PROLOGUE = """\
class _RefusingFactory(_Factory):
    async def open(self, params):
        raise RuntimeError("the session state this open names is not loadable")


_Factory = _RefusingFactory
"""


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_open_failure_after_the_handshake_cleans_up_the_partial_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend, root_agent_process

    spawned: list[Any] = []
    real_spawn_process = root_agent_process.spawn_process

    async def _recording_spawn_process(*args: Any, **kwargs: Any) -> Any:
        process = await real_spawn_process(*args, **kwargs)
        spawned.append(process)
        return process

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path, _REFUSING_OPEN_PROLOGUE),
        )
        monkeypatch.setattr(
            root_agent_process, "spawn_process", _recording_spawn_process
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            with pytest.raises(AppServerResponseError) as caught:
                await harness.start()
        finally:
            await harness.close()

    # The handshake succeeded, so a real process existed; the failed open is
    # cleaned up with the same fail-fast diagnostic, and the partial tree is
    # reaped rather than left running and unowned.
    assert backends == []
    assert "could not start the root agent" in caught.value.error.message
    assert "CHARTREUX_TRANSPORT=in_process" in caught.value.error.message
    assert len(spawned) == 1
    assert spawned[0].process.poll() is not None


# ---------------------------------------------------------------------------
# Snapshot reads: the agent computes the recovery-gate view, so the decision
# is reproducible from the payload instead of being re-derived coordinator-side.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_snapshot_read_reproduces_the_recovery_gate_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.core.model_catalog.availability import AvailabilitySnapshot

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            await _collect_turn(session, "hello")
            process = backends[0].process
            first = await process.snapshot()
            second = await process.snapshot()
            log = await session.resources.sessions.read_log()
            history = await session.resources.sessions.get_session_history(
                session.session_id
            )
            await session.close()
        finally:
            await harness.close()

    # The gate view is the AGENT's computed snapshot: the same committed model
    # and the same availability decision come back on a second read, so a
    # recovery-gate decision is derivable, not re-estimated here.
    assert isinstance(first.availability, AvailabilitySnapshot)
    assert first.availability == second.availability
    assert first.committed_model == second.committed_model
    assert first.incarnation_id == second.incarnation_id
    assert second.constructed_at >= first.constructed_at
    assert first.state.session.id == process.session_id
    # The directories and the log projection are part of the same reply, so
    # session/log/read is served from the snapshot rather than a second read.
    assert first.session_dir == second.session_dir
    assert first.jobs_dir == second.jobs_dir
    assert first.log == second.log == log
    assert "hello" in _history_text(history)


# ---------------------------------------------------------------------------
# Worktree cwd: the spawn itself chdirs, so the agent's project is the tree
# the client named and not the coordinator's own working directory.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_spawn_chdirs_into_the_worktree_the_client_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend

    tree = tmp_path / "a-linked-worktree"
    tree.mkdir()
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(
            LocalHarnessOptions(
                client=_client(),
                session_options=SessionOptions(cwd=str(tree), headless=True),
            )
        )
        try:
            session = await harness.start()
            process = backends[0].process
            cwd = session.state.session.cwd
            link = Path(f"/proc/{process.process.pid}/cwd")
            observed = os.readlink(link) if link.exists() else None
            if observed is not None:
                await session.close()
            else:
                pytest.skip("no /proc on this host to observe the spawn cwd")
        finally:
            await harness.close()

    # The kernel's view of the spawned root's cwd is the named tree: the
    # propagation happened at launch, not on the coordinator's own state copy.
    assert cwd == str(tree)
    assert observed == str(tree)


# ---------------------------------------------------------------------------
# host-service turn: the child processes a process-owned root spawns.
#
# The offer the coordinator puts on the turn is what grants a root its child
# capability; the agent then owns the launch (the resolved candidate, the
# working-set handoff, the parent link and the transcript all live in the root
# agent process) and every child is a process of its own. These tests drive
# that with a real spawned root and real spawned children, so the child's pid,
# its transcript and its publication attempt are the evidence.
# ---------------------------------------------------------------------------

#: A child agent process: it opens the spec it was spawned under, persists the
#: attempt id at its first durable write, adopts the handoff, and answers its
#: turns with a fixed reply. Every open records its pid, so the test holds a
#: real process identity for its lifecycle assertions.
_CHILD_AGENT_SCRIPT = """\
import asyncio
import os
from pathlib import Path

from chartreux.app_server._subagent_transcript import (
    child_transcript_logging,
    root_session_dir_from_spec,
)
from chartreux.app_server.child_agent import serve_child_agent
from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.subagent_ports import (
    CHILD_HANDOFF_PAYLOAD_KEY,
    spawn_spec_from_wire,
)
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.llm_models import LLMMessage, Role
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_MARKER = Path({marker!r})
_REPLY = {reply!r}


class _Factory:
    async def open(self, params):
        spec = spawn_spec_from_wire(params)
        root_dir = root_session_dir_from_spec(spec.payload)
        config = build_test_vibe_config(
            session_logging=child_transcript_logging(root_dir, "coder")
        )
        loop = build_test_agent_loop(
            config=config,
            backend=FakeBackend([[mock_llm_chunk(content=_REPLY)]] * 4),
        )
        await loop.persist_empty_session(attempt_id=spec.attempt_id)
        handoff = spec.payload.get(CHILD_HANDOFF_PAYLOAD_KEY)
        if isinstance(handoff, str) and handoff:
            loop.messages.append(
                LLMMessage(role=Role.user, content=handoff, injected=True)
            )
        recorded = _MARKER.read_text().splitlines() if _MARKER.exists() else []
        recorded.append(f"{{os.getpid()}}")
        _MARKER.write_text("\\n".join(recorded) + "\\n")
        return loop

    async def close(self):
        pass


init_harness_files_manager("user", "project")
asyncio.run(
    serve_child_agent(factory=_Factory(), announce_ready=bootstrap_requested())
)
"""

#: A root agent whose model calls the ``task`` tool. The blocker this covers
#: was exactly this turn failing with "Task tool requires a subagent runner in
#: context", so the child's own reply reaching the model through a real tool
#: result is the proof the call crossed into a child run and came back.
_TASK_ROOT_SCRIPT = """\
import asyncio
import sys

from chartreux.app_server import child_agent_process
from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import agent_child_host, serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, ToolCall
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_CHILD_SCRIPT = {child_script!r}
_TASK_ARGS = {task_args!r}

child_agent_process.child_agent_command = lambda: [sys.executable, "-c", _CHILD_SCRIPT]


class _Factory:
    def __init__(self, loop):
        self._loop = loop
        self.child_host = None

    async def open(self, params):
        self.child_host = agent_child_host(self._loop)
        return self._loop

    async def close(self):
        pass


init_harness_files_manager("user", "project")
config = build_test_vibe_config(
    session_logging=SessionLoggingConfig(enabled=True, save_dir={save_dir!r})
)
loop = build_test_agent_loop(
    config=config,
    backend=FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    ToolCall(
                        id="task-1",
                        index=0,
                        function=FunctionCall(name="task", arguments=_TASK_ARGS),
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="done")],
    ]),
)
asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""

#: A root agent that drives the child host's runner through the whole lifecycle
#: in the process that owns it, recording every observed fact. The runner is the
#: one the ``task`` tool receives; spelling the arguments out is what lets the
#: test hold the child's pid and its publication attempt id.
_SEQUENCE_ROOT_SCRIPT = """\
import asyncio
import json
import sys
from pathlib import Path

from chartreux.app_server import child_agent_process
from chartreux.app_server._subagent_launch import ChildPublicationError
from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import (
    RootAgentServer,
    agent_child_host,
    serve_root_agent,
)
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig, SubagentsConfig
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_CHILD_SCRIPT = {child_script!r}
_MARKER = Path({marker!r})

child_agent_process.child_agent_command = lambda: [sys.executable, "-c", _CHILD_SCRIPT]

_state = {{"invocations": 0}}


class _Factory:
    def __init__(self, loop):
        self._loop = loop
        self.child_host = None

    async def open(self, params):
        self.child_host = agent_child_host(self._loop)
        return self._loop

    async def close(self):
        pass


async def _collect(stream):
    last = None
    async for item in stream:
        last = item
    return last


async def _settled(child, timeout=60.0):
    for _ in range(int(timeout / 0.05)):
        if child.result is not None:
            return child.result
        await asyncio.sleep(0.05)
    raise AssertionError("the child run never settled")


def _record(values):
    merged = json.loads(_MARKER.read_text()) if _MARKER.exists() else {{}}
    merged.update(values)
    _MARKER.write_text(json.dumps(merged))


async def _sequence(host):
    try:
        await _run_sequence(host)
    except BaseException as exc:
        import traceback

        _record({{"error": traceback.format_exc()}})


async def _run_sequence(host):
    recorded = {{"budget_before": host.budget.live_processes}}
    # A foreground child: spawned for the run, reaped with its result.
    result = await _collect(
        host.runner.run(
            TaskArgs(task="child work", identity="coder", background=False),
            InvokeContext(tool_call_id="task-foreground"),
        )
    )
    recorded["foreground"] = {{
        "completed": result.completed,
        "response": result.response,
        "turns_used": result.turns_used,
        "agent_id": result.agent_id,
    }}
    recorded["budget_after_foreground"] = host.budget.live_processes
    _record(recorded)
    # A permit a process child cannot consume: the stage is named, nothing was
    # spawned to retract, and the reservation is released -- the cap of one
    # makes a stranded reservation fail the next launch.
    try:
        await _collect(
            host.runner.run(
                TaskArgs(
                    task="forced",
                    identity="coder",
                    background=False,
                    override_cooldown=True,
                ),
                InvokeContext(tool_call_id="task-forced"),
            )
        )
        recorded["permit"] = None
    except ChildPublicationError as exc:
        recorded["permit"] = {{
            "stage": exc.stage.value,
            "retraction": exc.retraction.value if exc.retraction else None,
        }}
    after_refusal = await _collect(
        host.runner.run(
            TaskArgs(task="child work", identity="coder", background=False),
            InvokeContext(tool_call_id="task-after-refusal"),
        )
    )
    recorded["after_refusal"] = after_refusal.completed
    _record(recorded)
    # A retained background child keeps its process; a re-task reuses it.
    ack = await _collect(
        host.runner.run(
            TaskArgs(task="keep", identity="coder", background=True, retain=True),
            InvokeContext(tool_call_id="task-retained"),
        )
    )
    child = host.children()[0]
    recorded["retained"] = {{
        "status": ack.status,
        "launch_outcome": ack.launch_outcome.value if ack.launch_outcome else None,
        "agent_id": ack.agent_id,
        "pid": child.spawned.process.pid,
        "session_id": child.session_id,
        "attempt_id": child.attempt_id,
        "child_dir": child.child_dir,
        "root_dir": child.spawned.root_session_dir,
    }}
    recorded["retained_result"] = (await _settled(child)).response
    reused = await _collect(
        host.runner.run(
            TaskArgs(
                task="again", identity="coder", background=True, agent_id=ack.agent_id
            ),
            InvokeContext(tool_call_id="task-reused"),
        )
    )
    child_again = host.children()[0]
    recorded["reused"] = {{
        "agent_id": reused.agent_id,
        "same_pid": child_again.spawned.process.pid == child.spawned.process.pid,
        "same_attempt": child_again.attempt_id == child.attempt_id,
    }}
    recorded["retained_result_2"] = (await _settled(child_again)).response
    recorded["budget_before_teardown"] = host.budget.live_processes
    _record(recorded)


_real_start_turn = RootAgentServer._start_turn


def _start_turn(self, params):
    _state["invocations"] += 1
    if _state["invocations"] == 1:
        self._track(_sequence(self.child_host))
    elif _state["invocations"] == 2:
        host = self.child_host
        _record({{
            "budget_after_teardown": host.budget.live_processes,
            "children_after_teardown": len(host.children()),
        }})
    _real_start_turn(self, params)


RootAgentServer._start_turn = _start_turn


init_harness_files_manager("user", "project")
config = build_test_vibe_config(
    session_logging=SessionLoggingConfig(enabled=True, save_dir={save_dir!r}),
    subagents=SubagentsConfig(max_running_subagents=1),
)
loop = build_test_agent_loop(
    config=config,
    backend=FakeBackend([
        [mock_llm_chunk(content="one")],
        [mock_llm_chunk(content="two")],
    ]),
)
asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""


def _child_agent_script(marker: Path, reply: str) -> str:
    return _CHILD_AGENT_SCRIPT.format(marker=str(marker), reply=reply)


def _task_command(tmp_path: Path, child_marker: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        _TASK_ROOT_SCRIPT.format(
            save_dir=str(tmp_path / "sessions"),
            child_script=_child_agent_script(child_marker, "child-done"),
            task_args=json.dumps({
                "task": "do the child work",
                "identity": "coder",
                "background": False,
            }),
        ),
    ]


def _sequence_command(tmp_path: Path, marker: Path, child_marker: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        _SEQUENCE_ROOT_SCRIPT.format(
            save_dir=str(tmp_path / "sessions"),
            child_script=_child_agent_script(child_marker, "child-done"),
            marker=str(marker),
        ),
    ]


def _marker_state(marker: Path) -> dict[str, Any]:
    """The sequence's record so far; a write in progress reads as empty."""
    if not marker.exists():
        return {}
    try:
        return json.loads(marker.read_text())
    except ValueError:
        return {}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _session_metadata(save_dir: Path) -> list[tuple[Path, Any]]:
    """Every persisted session metadata record under the root's save dir."""
    from chartreux.core.session_types import SessionMetadata

    return [
        (path, SessionMetadata.model_validate_json(path.read_text()))
        for path in sorted(save_dir.rglob("meta.json"))
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_task_call_runs_a_child_process_through_the_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn's ``task`` call spawns a real child from the root agent process."""
    from chartreux.app_server import process_session_backend

    child_marker = tmp_path / "child-pids"
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _task_command(tmp_path, child_marker),
        )
        _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            events = await _collect_turn(session, "delegate this")
            session_id = session.session_id
            await session.close()
        finally:
            await harness.close()

    text = _assistant_text(events)
    # The turn completed instead of aborting on a runner-less task call, and
    # the model kept going after the child's own reply came back.
    assert "done" in text
    assert "Task tool requires a subagent runner" not in text

    pids = [int(line) for line in child_marker.read_text().split()]
    assert len(pids) == 1
    # A non-retained child exits with the run that produced its result.
    assert not _pid_alive(pids[0])

    # The publication linked the child to the parent transcript under the
    # attempt the spawn carried, and the child wrote its transcript under the
    # root's agents subtree.
    metadata = [
        (path, item)
        for path, item in _session_metadata(tmp_path / "sessions")
        if item.session_id == session_id
    ]
    assert len(metadata) == 1
    metadata_path, record = metadata[0]
    links = record.child_sessions
    assert len(links) == 1
    link = links[0]
    assert link.agent == "coder"
    assert link.tool_call_id == "task-1"
    assert link.attempt_id
    assert link.relative_path is not None
    # The child owns the only writer of its transcript, and the transcript it
    # wrote is where the parent link says it is: under the root's agents/ tree.
    child_dir = metadata_path.parent / link.relative_path
    assert child_dir.is_dir()
    assert child_dir.parent.name == "agents"
    assert any(child_dir.iterdir())


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_retained_child_keeps_its_process_and_its_attempt_tombstones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retained lifecycle and the lost-outcome teardown, across the pipe."""
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.subagent_ports import ChildTeardownRequest

    marker = tmp_path / "sequence.json"
    child_marker = tmp_path / "child-pids"
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _sequence_command(tmp_path, marker, child_marker),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            await _collect_turn(session, "run the sequence")
            await _wait_for(lambda: "budget_before_teardown" in _marker_state(marker))
            observed = _marker_state(marker)
            assert "error" not in observed, observed.get("error")
            retained = observed["retained"]
            # The retained child is a live process of its own when the
            # coordinator asks for its attempt to be settled.
            assert _pid_alive(retained["pid"])
            report = await backends[0].process.teardown_child_attempt(
                ChildTeardownRequest(attempt_id=retained["attempt_id"])
            )
            await _collect_turn(session, "check the budget")
            await _wait_for(lambda: "budget_after_teardown" in _marker_state(marker))
            after = _marker_state(marker)
            await session.close()
        finally:
            await harness.close()

    foreground = observed["foreground"]
    assert foreground["completed"] is True
    assert foreground["response"] == "child-done"
    assert foreground["turns_used"] == 1
    # A foreground result carries no retained handle, and the child's process
    # left with the run: no capacity or budget slot strands.
    assert foreground["agent_id"] is None
    assert observed["budget_before"] == 0
    assert observed["budget_after_foreground"] == 0

    # A process child cannot consume a cooldown-override permit: the aggregate
    # failure names that stage, retracts nothing it never spawned, and the
    # released reservation still admits the next child under a cap of one.
    assert observed["permit"] == {"stage": "permit", "retraction": None}
    assert observed["after_refusal"] is True

    # The retained child was acknowledged as a launched background run, and its
    # process outlived the run that produced its result.
    assert retained["status"] == "launched"
    assert retained["launch_outcome"] == "launched"
    assert observed["retained_result"] == "child-done"
    assert retained["child_dir"] and retained["root_dir"]
    # The re-task reused the same process and the same publication attempt.
    assert observed["reused"]["same_pid"] is True
    assert observed["reused"]["same_attempt"] is True
    assert observed["retained_result_2"] == "child-done"
    assert observed["budget_before_teardown"] == 1

    # The teardown crossed the pipe to the agent that owns the attempt: it
    # cancelled the prepared child, adjudicated the records it persisted and
    # retracted them through the attempt-id bridge.
    assert report.outcome.value == "tombstoned"
    assert report.attempt_id == retained["attempt_id"]
    assert report.retracted is True
    assert not _pid_alive(retained["pid"])
    assert after["budget_after_teardown"] == 0
    assert after["children_after_teardown"] == 0
    assert Path(retained["child_dir"]).exists()
    assert list(Path(retained["root_dir"]).rglob("*tombstone*"))


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_root_death_fails_the_session_and_never_replays_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.root_agent_process import RootFailureSignature

    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            backend = backends[0]
            popen = backend.process.process
            root_pid = popen.pid
            first = await _collect_turn(session, "hello")
            assert "done" in _assistant_text(first)

            # The root dies with the pipe still readable by this end: the
            # liveness watcher owns this case.
            popen.kill()
            popen.wait(timeout=10)
            death = await asyncio.wait_for(backend.process.wait_for_death(), timeout=10)
            pid_after_death = popen.pid
            await session.close()
        finally:
            await harness.close()

    # The session failure names the signature a user must be able to tell
    # apart from the pipe, deadline and idle-wedge cases. Which of the two
    # observers won the race is timing; the signature is not.
    assert death.signature is RootFailureSignature.DEATH_RECEIPT
    assert death.diagnostic.startswith("death-receipt: ")
    assert backend.process.alive is False
    assert any(
        "death-receipt" in diagnostic.message()
        for diagnostic in backend.process.diagnostics
    )
    # No second root was ever spawned and no turn was replayed into one: the
    # pid that died is the only pid this session ever had.
    assert len(backends) == 1
    assert pid_after_death == root_pid
    assert popen.poll() is not None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_job_cleanup_consults_the_directory_the_identity_moved_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import process_session_backend
    from chartreux.app_server._session_runtime_impl import ProcessRootReplace
    from chartreux.core.job_launcher import JobGroupRecord, write_job_group_record

    target_id = await _persist_test_session(tmp_path)
    # A directory this session never published: it stands for the identity the
    # transition abandoned, and cleanup must not consult it.
    abandoned_dir = tmp_path / "abandoned-jobs"
    abandoned_dir.mkdir(parents=True, exist_ok=True)
    write_job_group_record(
        abandoned_dir,
        JobGroupRecord(
            job_id="job-abandoned",
            pid=_reaped_pid(),
            start_time=0,
            session_id="session-abandoned",
        ),
    )
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _resume_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            backend = backends[0]
            process = backend.process

            async def _reap() -> object:
                return object()

            await ProcessRootReplace(process=process, reap=_reap).replace(
                session_id=target_id
            )

            # The transition reply published the directory the session moved
            # to, so cleanup consults that one and never the abandoned one.
            current_dir = backend.tracked_jobs_dir
            assert current_dir is not None
            assert current_dir == Path(str(process.session_dir)) / "jobs"
            assert current_dir != abandoned_dir
            current_dir.mkdir(parents=True, exist_ok=True)
            write_job_group_record(
                current_dir,
                JobGroupRecord(
                    job_id="job-current",
                    pid=_reaped_pid(),
                    start_time=0,
                    session_id=process.session_id or "",
                ),
            )

            outcomes = await backend.settle_tracked_jobs()
            await session.close()
        finally:
            await harness.close()

    assert [outcome.job_id for outcome in outcomes] == ["job-current"]
    # A group with no live member settles as reaped, never as a claimed stop
    # of a group it could not re-establish.
    assert outcomes[0].settlement.reaped is True
    assert (abandoned_dir / "job-abandoned.pgid").exists()
    assert [outcome.job_id for outcome in backend.tracked_job_outcomes] == [
        "job-current"
    ]


def _reaped_pid() -> int:
    """A pid that has already exited, so its group has no live member."""
    import subprocess

    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait(timeout=30)
    return finished.pid


#: A child whose spawn records the selector its environment carried, which is
#: the only evidence that a default-resolved route materialized into the spawn.
_SELECTOR_SPY_PROLOGUE = """\
from pathlib import Path
import os

Path({marker!r}).write_text(os.environ.get("CHARTREUX_TRANSPORT", "unset"))
"""


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_config_only_default_spawns_with_no_selector_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interactive default is a topology, not a variable.

    This is the one entrypoint case that by construction needs the flipped
    default: every other case names its selection, and this one sets nothing.
    """
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.transport import resolve_transport_selection

    selection = resolve_transport_selection(environ={}, interactive=True)
    assert selection.kind == STDIO_TRANSPORT
    assert selection.source == "default"
    # The resolved default materializes the selector the spawned root's own
    # env-keyed guards read; nothing else would tell it which route it is on.
    assert materialized_spawn_env(selection, environ={})[TRANSPORT_ENV_VAR] == (
        STDIO_TRANSPORT
    )
    marker = tmp_path / "selector.txt"
    with egress.bootstrap_directory() as guard_dir:
        _install_default_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(
                tmp_path, prologue=_SELECTOR_SPY_PROLOGUE.format(marker=str(marker))
            ),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(
            LocalHarnessOptions(
                client=_client(),
                session_options=SessionOptions(cwd=str(tmp_path), headless=True),
                interactive=True,
            )
        )
        try:
            session = await harness.start()
            process = backends[0].process.process
            events = await _collect_turn(session, "hello")
            await session.close()
        finally:
            await harness.close()

    assert len(backends) == 1
    assert process.pid != os.getpid()
    assert "done" in _assistant_text(events)
    # The root ran with the materialized selector and exited with the session.
    assert marker.read_text() == STDIO_TRANSPORT
    assert process.poll() is not None
