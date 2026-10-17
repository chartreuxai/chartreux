"""Lifecycle lock, holder claim and release, and busy-conflict tests for the local harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server import _runtime as runtime
from chartreux.app_server._execution import SessionExecutionKind
from chartreux.app_server._session_backend_impl import SessionBackendHostImpl
import chartreux.app_server._sessions as _sessions
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    AppServerResponseError,
    AutoWorktreeInput,
    ClientCapabilities,
    ClientInfo,
    ProtocolErrorCode,
    SessionOptions,
    SessionResumeParams,
    SessionStartParams,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.git.worktree import ManagedWorktree
from chartreux.core.session.session_lease import SessionBusyError, SessionLease
from tests.app_server._local_runtime_support import _init_repo
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import create_legacy_app_server, legacy_backend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


@pytest.mark.asyncio
async def test_first_session_open_holds_the_lifecycle_lock() -> None:
    root = build_test_agent_loop()
    lock_states: list[bool] = []

    async def open_root(_request: runtime.RootOpenRequest) -> AgentLoop:
        lock_states.append(server._lifecycle_lock.locked())
        return root

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    await client.initialize(ClientInfo(name="test", version="1"))
    await client.notify("initialized")
    try:
        await client.request("session/start", SessionStartParams())
        host = server._session_backend_host
        assert isinstance(host, SessionBackendHostImpl)
        original_resume = host._resume
        assert original_resume is not None
        event_task = server._backend_event_task

        async def resume(params: SessionResumeParams):
            lock_states.append(server._lifecycle_lock.locked())
            return await original_resume(params)

        host._resume = resume
        await client.request(
            "session/resume", SessionResumeParams(session_id=root.session_id)
        )

        assert lock_states == [True, True]
        assert server._backend_event_task is event_task
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["config/write", "config/reload"])
async def test_busy_config_mutations_return_conflict(method: str) -> None:
    root = build_test_agent_loop()

    async def open_root(_request: runtime.RootOpenRequest) -> AgentLoop:
        return root

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    await client.initialize(ClientInfo(name="test", version="1"))
    await client.notify("initialized")
    try:
        await client.request("session/start", SessionStartParams())
        backend = legacy_backend(server)
        params: dict[str, Any] = {"sessionId": root.session_id}
        if method == "config/write":
            params["ops"] = []
        with backend.session.execution.reserve(SessionExecutionKind.TURN, "turn-1"):
            with pytest.raises(AppServerResponseError) as exc_info:
                await client.request(method, params)
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_session_stop_releases_the_holder_after_shutting_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    loops: list[AgentLoop] = []
    order: list[str] = []

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        agent_loop = build_test_agent_loop(cwd=Path(request.options.cwd))
        loops.append(agent_loop)
        return agent_loop

    release_holder = ManagedWorktree.release_holder
    close_agent_loop = _sessions.close_agent_loop

    def record_release(self: ManagedWorktree, session_id: str) -> None:
        order.append("holder")
        release_holder(self, session_id)

    async def record_close(agent_loop: AgentLoop) -> None:
        order.append("close")
        await close_agent_loop(agent_loop)

    monkeypatch.setattr(ManagedWorktree, "release_holder", record_release)
    monkeypatch.setattr(_sessions, "close_agent_loop", record_close)

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="idle-client", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(
            cwd=str(tmp_path), worktree=AutoWorktreeInput(prompt="fix the login bug")
        ),
    )
    loops[0].session_logger._persisted = True

    try:
        await session.close()
    finally:
        await client.close()

    # Worktree deletion must remain blocked until all owned jobs and runtime
    # handles are reclaimed, even when the session/stop caller times out.
    assert order == ["close", "holder"]


@pytest.mark.asyncio
async def test_a_failed_holder_release_still_shuts_the_session_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    closed: list[str] = []

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        return build_test_agent_loop(cwd=Path(request.options.cwd))

    close_agent_loop = _sessions.close_agent_loop

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("the holder marker could not be unlinked")

    async def record_close(agent_loop: AgentLoop) -> None:
        closed.append(agent_loop.session_id)
        await close_agent_loop(agent_loop)

    monkeypatch.setattr(ManagedWorktree, "remove_if_unheld", refuse)
    monkeypatch.setattr(_sessions, "close_agent_loop", record_close)

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="idle-client", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(
            cwd=str(tmp_path), worktree=AutoWorktreeInput(prompt="fix the login bug")
        ),
    )

    try:
        await session.close()
    finally:
        await client.close()

    # Releasing the holder is the first step of the close now, so letting it
    # throw would skip every step after it - the MCP clients and terminals
    # session.close() shuts down, and the rollback of a failed create.
    assert closed


@pytest.mark.asyncio
async def test_production_legacy_root_holds_lease_until_runtime_shutdown(
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
        client_info=ClientInfo(name="test", version="1"),
        session_id=session_id,
    )

    root = await process.open_root(request)
    with pytest.raises(SessionBusyError):
        SessionLease(tmp_path, session_id).acquire()

    await root.aclose()
    SessionLease(tmp_path, session_id).acquire().release()
