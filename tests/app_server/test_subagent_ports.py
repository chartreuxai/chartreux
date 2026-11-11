"""Spawn + completion-source ports and their adapters (WP5 [5A])."""

from __future__ import annotations

from contextlib import aclosing
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from chartreux.app_server.agent_wire import (
    SUBAGENT_OPEN_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
)
from chartreux.app_server.child_agent import CHILD_AGENT_MODULE
from chartreux.app_server.child_agent_process import (
    ProcessSpawnedSubagent,
    ProcessSubagentSpawn,
    child_agent_command,
    multi_process_child_selected,
)
from chartreux.app_server.subagent_ports import (
    InProcessSubagentSpawn,
    SubagentCompletion,
    SubagentFrame,
    SubagentSpawnDeniedError,
    SubagentSpawnSpec,
    spawn_spec_from_wire,
    spawn_spec_to_wire,
    subagent_spawn_port,
)
from chartreux.app_server.transport import TRANSPORT_ENV_VAR
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.events import AssistantEvent, BaseEvent, UserMessageEvent
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
