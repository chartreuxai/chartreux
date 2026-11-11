"""The root agent boots in its own process and serves the execution port."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from chartreux.app_server import root_agent
from chartreux.app_server._connection_protocol import ClientInfo
from chartreux.app_server.protocol import AgentConfig
from chartreux.app_server.registry_manifest import RegistryManifest
from chartreux.app_server.root_agent_process import (
    RootAgentProcess,
    RootAgentSelectionError,
    multi_process_root_selected,
    spawn_root_agent_process,
)
from chartreux.app_server.transport import TRANSPORT_ENV_VAR
from chartreux.core.agent_loop._execution_port import (
    ExecutionTurnRequest,
    TurnOrderingConflict,
)
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.events import AssistantEvent, BaseEvent, UserMessageEvent
from chartreux.core.session.session_index import MESSAGES_FILENAME
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
