"""Session listing for continue, continue/resume/rebind, and survives-a-resume tests for the local harness."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from chartreux.app_server import _runtime as runtime
import chartreux.app_server._host as host_module
from chartreux.app_server._projection import project_message_history
from chartreux.app_server._projector import ProjectedUpdate
from chartreux.app_server._turns import TurnController
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.local import LocalHarnessHost
from chartreux.app_server.models import PublicHistoryEntry
from chartreux.app_server.protocol import (
    AutoWorktreeInput,
    ClientCapabilities,
    ClientInfo,
    HistoryEntryUpdatedParams,
    SessionListParams,
    SessionOptions,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.session.resume_sessions import ResumeSessionInfo
from chartreux.core.session.session_loader import SessionLoader
from tests.app_server._local_runtime_support import (
    _configure_local_accounting,
    _init_repo,
)
from tests.app_server.test_usage_attribution import read_ledger
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import create_legacy_app_server
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_local_root_close_retires_accounting_and_resume_reopens_ledger(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_local_accounting(monkeypatch, tmp_path, logging_enabled=True)
    harness = LocalHarnessHost()
    options = runtime.LocalHarnessOptions(
        session_options=SessionOptions(cwd=str(tmp_path), headless=True)
    )
    host = await harness.connect(options)
    resumed_host = None
    try:
        session = await host.open_session()
        root_id = session.session_id
        async for _ in session.act("first"):
            pass
        process = harness._process
        assert process is not None
        writer = process._root_usage_writers[root_id]
        paths = set((config_dir / "usage").rglob("*.jsonl"))
        await session.close()
        assert writer._closed
        assert not process._closed
        assert not process.usage_service._closed
        assert root_id not in process._root_usage_writers
        assert root_id not in process._root_accounting_owners
        assert root_id not in process._accounting_loop_roots
        assert not process._accounting_loops
        assert set((config_dir / "usage").rglob("*.jsonl")) == paths
        assert len(read_ledger(config_dir, root_id)) == 1

        resumed_host = await harness.connect(
            runtime.LocalHarnessOptions(
                session_options=options.session_options,
                session=runtime.ResumeSessionIntent(root_id),
            )
        )
        resumed = await resumed_host.open_session()
        assert resumed.session_id == root_id
        assert process._root_usage_writers[root_id] is not writer
        async for _ in resumed.act("second"):
            pass
        await resumed.close()
        records = read_ledger(config_dir, root_id)
        assert len(records) == 2
        assert all(record.root_session_id == root_id for record in records)
        assert set((config_dir / "usage").rglob("*.jsonl")) == paths
        assert not process._root_usage_writers
        assert not process._accounting_loops
    finally:
        await host.close()
        if resumed_host is not None:
            await resumed_host.close()
        await harness.close()


class ClosingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.closed = True


def test_session_list_falls_back_for_malformed_saved_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    monkeypatch.setattr(
        host_module,
        "list_local_resume_sessions",
        lambda _config, _cwd: [
            ResumeSessionInfo(
                session_id="session-1",
                cwd="/workspace",
                start_time="not-a-timestamp",
                updated_at="",
            )
        ],
    )
    monkeypatch.setattr(host_module, "now_ms", lambda: 123_456)
    monkeypatch.setattr(
        host_module.SessionLoader, "get_first_user_message", lambda *_args: ""
    )

    response = host_module.project_session_list(config, SessionListParams())

    assert response.items[0].created_at == 123_456
    assert response.items[0].updated_at == 123_456


def _stub_session_list_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        host_module,
        "list_local_resume_sessions",
        lambda _config, _cwd: [
            ResumeSessionInfo(
                session_id="newer", cwd="/workspace", updated_at="2026-01-02"
            ),
            ResumeSessionInfo(
                session_id="older", cwd="/workspace", updated_at="2026-01-01"
            ),
        ],
    )
    monkeypatch.setattr(
        host_module.SessionLoader, "get_first_user_message", lambda *_args: ""
    )


def test_session_list_continue_id_prefers_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    _stub_session_list_sources(monkeypatch)
    monkeypatch.setattr(
        host_module.last_session_pointer, "load", lambda _config: "older"
    )

    response = host_module.project_session_list(config, SessionListParams())

    assert response.continue_session_id == "older"


def test_session_list_continue_id_falls_back_to_newest_when_pointer_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    _stub_session_list_sources(monkeypatch)
    monkeypatch.setattr(host_module.last_session_pointer, "load", lambda _config: None)

    response = host_module.project_session_list(config, SessionListParams())

    assert response.continue_session_id == "newer"


def test_session_list_continue_id_ignores_stale_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    _stub_session_list_sources(monkeypatch)
    monkeypatch.setattr(
        host_module.last_session_pointer, "load", lambda _config: "does-not-exist"
    )

    response = host_module.project_session_list(config, SessionListParams())

    assert response.continue_session_id == "newer"


def test_session_list_continue_id_accepts_forked_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    monkeypatch.setattr(
        host_module,
        "list_local_resume_sessions",
        lambda _config, _cwd: [
            ResumeSessionInfo(
                session_id="root", cwd="/workspace", updated_at="2026-01-02"
            ),
            ResumeSessionInfo(
                session_id="fork",
                cwd="/workspace",
                updated_at="2026-01-01",
                parent_session_id="root",
            ),
        ],
    )
    monkeypatch.setattr(
        host_module.SessionLoader, "get_first_user_message", lambda *_args: ""
    )
    monkeypatch.setattr(
        host_module.last_session_pointer, "load", lambda _config: "fork"
    )

    response = host_module.project_session_list(config, SessionListParams())

    assert response.continue_session_id == "fork"


@pytest.mark.asyncio
async def test_a_dropped_notification_still_survives_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    emit = TurnController._emit_projected
    loops: list[AgentLoop] = []
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        )
    )

    async def drop_the_completion(
        self: TurnController, update: ProjectedUpdate
    ) -> None:
        if isinstance(update.params, HistoryEntryUpdatedParams):
            raise RuntimeError("the client went away")
        await emit(self, update)

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        agent_loop = build_test_agent_loop(cwd=Path(request.options.cwd), config=config)
        loops.append(agent_loop)
        return agent_loop

    monkeypatch.setattr(TurnController, "_emit_projected", drop_the_completion)
    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="dropped-resume-client", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(
            cwd=str(tmp_path), worktree=AutoWorktreeInput(prompt="fix the login bug")
        ),
    )
    try:
        # The live entry is cosmetic; the persisted metadata is the only thing a
        # resume rebuilds from. Losing the completion notification must not cost
        # the durable record too, or the worktree disappears from the transcript
        # in exactly the failure this notification path is allowed to have.
        agent_loop = loops[0]
        await agent_loop.persist_empty_session()
        session_dir = agent_loop.session_logger.session_dir
        assert session_dir is not None
    finally:
        await session.close()
        await client.close()

    rebuilt = _rebuild_from_disk(agent_loop.session_id, session_dir)
    effects = [entry for entry in rebuilt if entry.type == "effect"]
    assert len(effects) == 1
    assert effects[0].detail.kind == "worktree"


def _rebuild_from_disk(session_id: str, session_dir: Path) -> list[PublicHistoryEntry]:
    """Rebuild a transcript the way a resume does: from the files, not memory."""
    messages, _ = SessionLoader.load_session(session_dir)
    return project_message_history(
        session_id, messages, SessionLoader.load_metadata(session_dir)
    )


@pytest.mark.asyncio
async def test_the_worktree_entry_survives_a_resume(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    loops: list[AgentLoop] = []
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        )
    )

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        agent_loop = build_test_agent_loop(cwd=Path(request.options.cwd), config=config)
        loops.append(agent_loop)
        return agent_loop

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="resume-client", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(
            cwd=str(tmp_path), worktree=AutoWorktreeInput(prompt="fix the login bug")
        ),
    )
    try:
        # The metadata file does not exist yet when the worktree is recorded, so
        # the field patch has nothing to write into: this full save is what
        # first carries the worktree to disk.
        agent_loop = loops[0]
        await agent_loop.persist_empty_session()
        session_dir = agent_loop.session_logger.session_dir
        assert session_dir is not None
    finally:
        await session.close()
        await client.close()

    # A harness effect living only in the TurnController is gone by here, and
    # the runner reclaims the process a second after every turn - so the round
    # trip through the file is the whole guarantee.
    assert SessionLoader.load_metadata(session_dir).created_worktree is not None

    rebuilt = _rebuild_from_disk(agent_loop.session_id, session_dir)
    effects = [entry for entry in rebuilt if entry.type == "effect"]
    assert len(effects) == 1
    assert effects[0].detail.kind == "worktree"
    assert effects[0].detail.input is not None
    assert effects[0].detail.input.name == "fix-the-login-bug"
    # The worktree exists before anything else the transcript can show.
    assert rebuilt[0] is effects[0]


@pytest.mark.asyncio
async def test_session_continue_opens_the_resumed_root_once_over_json_rpc() -> None:
    resumed = build_test_agent_loop()
    requests: list[runtime.RootOpenRequest] = []

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        requests.append(request)
        return resumed

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    session = await AppServerSession.start(
        AppServerClient(client_transport, run_peer=server.serve),
        client_info=ClientInfo(name="continue-client", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(cwd=str(Path.cwd())),
        continue_session=True,
    )
    try:
        assert session.session_id == resumed.session_id
        assert len(requests) == 1
        assert requests[0].continue_latest is True
        assert requests[0].session_id is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_resume_rebinds_session_in_place_preserving_backend(
    tmp_path: Path,
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    source_backend = ClosingBackend()
    source = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging), backend=source_backend
    )
    source.stats.session_prompt_tokens = 11
    source.stats.session_completion_tokens = 7
    source.stats.context_tokens = 18
    await source.persist_empty_session()

    original_session_id = source.session_id
    await runtime.AgentRuntimeFactory().resume_root(source, original_session_id)

    try:
        # resume_root mutates in-place: same object, same backend, stats preserved.
        assert source.session_id == original_session_id
        assert source.backend is source_backend
        assert source.stats.session_prompt_tokens == 11
        assert source.stats.session_completion_tokens == 7
        assert not source_backend.closed
    finally:
        await source.aclose()


def test_continue_prefers_valid_terminal_session_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    session_path = tmp_path / "session"
    monkeypatch.setattr(runtime.last_session_pointer, "load", lambda _config: "saved")
    monkeypatch.setattr(
        runtime.SessionLoader,
        "find_session_by_id",
        lambda *_args, **_kwargs: session_path,
    )

    source = cast(AgentLoop, SimpleNamespace(config=config))
    assert runtime.AgentRuntimeFactory().resolve_latest(source, Path.cwd()) == "saved"


def test_continue_falls_back_to_latest_session_for_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    session_path = tmp_path / "latest"
    monkeypatch.setattr(runtime.last_session_pointer, "load", lambda _config: "stale")
    monkeypatch.setattr(
        runtime.SessionLoader, "find_session_by_id", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime.SessionLoader,
        "find_latest_session",
        lambda *_args, **_kwargs: session_path,
    )
    monkeypatch.setattr(
        runtime.SessionLoader,
        "load_session",
        lambda _path: ([], {"session_id": "latest-id"}),
    )

    source = cast(AgentLoop, SimpleNamespace(config=config))
    assert (
        runtime.AgentRuntimeFactory().resolve_latest(source, Path.cwd()) == "latest-id"
    )


def test_continue_requires_session_logging() -> None:
    config = build_test_vibe_config(session_logging=SessionLoggingConfig(enabled=False))
    source = cast(AgentLoop, SimpleNamespace(config=config))

    with pytest.raises(
        runtime.RuntimeSessionNotFoundError, match="Session logging is disabled"
    ):
        runtime.AgentRuntimeFactory().resolve_latest(source, Path.cwd())
