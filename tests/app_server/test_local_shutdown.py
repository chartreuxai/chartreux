"""Session stop and close, worktree removal, and accounting process close tests for the local harness."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from chartreux.app_server import _runtime as runtime
from chartreux.app_server._model import validate_wire
from chartreux.app_server._projector import ProjectedUpdate
from chartreux.app_server._turns import TurnController
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.local import LocalHarness
from chartreux.app_server.protocol import (
    AutoWorktreeInput,
    ClientCapabilities,
    ClientInfo,
    HistoryEntryUpdatedParams,
    SessionOptions,
    WorkspaceWorktreeRemoveResponse,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.cli import programmatic
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.git.worktree import SNAPSHOT_REF_PREFIX, WorktreeReleaseOutcome
from tests.app_server._local_runtime_support import (
    _configure_local_accounting,
    _init_repo,
    _linked,
    _open_local_session,
    _prepare_auto,
)
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import create_legacy_app_server, legacy_backend


@pytest.mark.parametrize("fail_output", [False, True])
def test_programmatic_closes_owned_accounting_process(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_output: bool
) -> None:
    _configure_local_accounting(monkeypatch, tmp_path)
    harness = LocalHarness(
        runtime.LocalHarnessOptions(
            session_options=SessionOptions(cwd=str(tmp_path), headless=True)
        )
    )
    monkeypatch.setattr(programmatic, "LocalHarness", lambda _: harness)
    close = harness.close

    async def probe_close() -> None:
        await close()
        process = harness._host._process
        assert process is not None
        assert process._closed
        assert process.usage_service._closed
        assert not process.usage_service._tasks
        assert not process.usage_service._writers
        assert not process._root_usage_writers
        assert not process._root_usage_drains
        assert not process._accounting_loops
        assert not process._startup_producers

    monkeypatch.setattr(harness, "close", probe_close)
    if fail_output:
        monkeypatch.setattr(
            programmatic.ProgrammaticOutput,
            "finalize",
            Mock(side_effect=ValueError("output failed")),
        )
        with pytest.raises(ValueError, match="output failed"):
            programmatic.run_programmatic(harness_options=harness._options, prompt="hi")
    else:
        assert (
            programmatic.run_programmatic(harness_options=harness._options, prompt="hi")
            == "answer"
        )
    assert len(list((config_dir / "usage").rglob("*.jsonl"))) == 1


@pytest.mark.asyncio
async def test_session_stop_removes_a_clean_auto_worktree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    session, client = await _open_local_session(
        "release-client",
        SessionOptions(
            cwd=str(tmp_path), worktree=AutoWorktreeInput(prompt="fix the login bug")
        ),
    )
    worktree_root = Path(session.cwd)
    assert worktree_root.is_dir()

    try:
        await session.close()
    finally:
        await client.close()

    assert not worktree_root.exists()
    assert _linked(tmp_path) == ()
    assert "chartreux/fix-the-login-bug" not in [head.name for head in repo.heads]


@pytest.mark.asyncio
async def test_a_dropped_notification_still_retires_the_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    emit = TurnController._emit_projected

    async def drop_the_completion(
        self: TurnController, update: ProjectedUpdate
    ) -> None:
        # Starting an effect adds an entry; only completing one patches it.
        if isinstance(update.params, HistoryEntryUpdatedParams):
            raise RuntimeError("the client went away")
        await emit(self, update)

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        return build_test_agent_loop(cwd=Path(request.options.cwd))

    monkeypatch.setattr(TurnController, "_emit_projected", drop_the_completion)
    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="dropped-client", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(
            cwd=str(tmp_path), worktree=AutoWorktreeInput(prompt="fix the login bug")
        ),
    )
    try:
        effects = [entry for entry in session.history if entry.type == "effect"]
        assert len(effects) == 1
        assert effects[0].state.status == "completed"
        # Losing the notification costs the live update and nothing else. Still
        # registered, the effect would be served from the live bucket, which
        # sorts after every entry that follows - so the creation would drift to
        # the bottom of the transcript as soon as the user said anything.
        assert legacy_backend(server).session.turns._harness_effects == {}
    finally:
        await session.close()
        await client.close()


@pytest.mark.asyncio
async def test_session_stop_saves_the_work_before_removing_a_worktree(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    session, client = await _open_local_session(
        "dirty-client",
        SessionOptions(
            cwd=str(tmp_path), worktree=AutoWorktreeInput(prompt="fix the login bug")
        ),
    )
    worktree_root = Path(session.cwd)
    (worktree_root / "written-by-the-agent.txt").write_text("work\n")

    try:
        await session.close()
    finally:
        await client.close()

    # Closing the last session is what makes the worktree collectable, and a
    # file the agent left uncommitted must not make it uncollectable for good.
    assert not worktree_root.exists()
    saved = repo.commit(f"{SNAPSHOT_REF_PREFIX}/fix-the-login-bug")
    assert saved.tree["written-by-the-agent.txt"].data_stream.read() == b"work\n"


@pytest.mark.asyncio
async def test_session_stop_keeps_the_worktree_of_a_session_that_ran_a_turn(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    loops: list[AgentLoop] = []

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        agent_loop = build_test_agent_loop(cwd=Path(request.options.cwd))
        loops.append(agent_loop)
        return agent_loop

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
    worktree_root = Path(session.cwd)
    # The runner sends session/stop a second after every turn to reclaim the
    # app-server process; the session stays live and resumable. persisted is
    # what records that a turn ran, and it is the only thing standing between an
    # idle session and losing the worktree it is still working in.
    loops[0].session_logger._persisted = True

    try:
        await session.close()
    finally:
        await client.close()

    assert worktree_root.is_dir()
    assert "chartreux/fix-the-login-bug" in [head.name for head in repo.heads]


@pytest.mark.asyncio
async def test_session_stop_keeps_a_worktree_the_session_did_not_create(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    # Opened in a worktree that already existed. Rolling back a failed create
    # must never reach a worktree this session has no claim on.
    worktree = _prepare_auto(tmp_path)
    session, client = await _open_local_session(
        "adopting-client", SessionOptions(cwd=str(worktree.path))
    )

    try:
        await session.close()
    finally:
        await client.close()

    assert worktree.root.is_dir()


async def _remove_worktree_over_rpc(
    cwd: Path, *, attach_root: Path | None = None
) -> WorkspaceWorktreeRemoveResponse:
    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        return build_test_agent_loop(cwd=Path(request.options.cwd))

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = None
    if attach_root is None:
        await client.initialize(ClientInfo(name="remove-client", version="1"))
        await client.notify("initialized")
    else:
        session = await AppServerSession.start(
            client,
            client_info=ClientInfo(name="attached-remove-client", version="1"),
            capabilities=ClientCapabilities(),
            session_options=SessionOptions(cwd=str(attach_root)),
        )

    try:
        raw = await client.request("workspace/git/worktrees/remove", {"cwd": str(cwd)})
    finally:
        if session is not None:
            await session.close()
        await client.close()
    return validate_wire(WorkspaceWorktreeRemoveResponse, raw)


@pytest.mark.asyncio
async def test_worktree_remove_deletes_a_clean_managed_worktree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    worktree = _prepare_auto(tmp_path)

    response = await _remove_worktree_over_rpc(worktree.root)

    assert response.outcome == WorktreeReleaseOutcome.REMOVED
    assert response.branch_deleted is True
    assert response.root == str(worktree.root)
    assert not worktree.root.exists()
    assert worktree.branch not in [head.name for head in repo.heads]


@pytest.mark.asyncio
async def test_worktree_remove_saves_the_work_it_removes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    worktree = _prepare_auto(tmp_path)
    (worktree.root / "unsaved.txt").write_text("work\n")

    response = await _remove_worktree_over_rpc(worktree.root)

    assert response.outcome == WorktreeReleaseOutcome.REMOVED
    assert not worktree.root.exists()
    saved = repo.commit(f"{SNAPSHOT_REF_PREFIX}/{worktree.name}")
    assert saved.tree["unsaved.txt"].data_stream.read() == b"work\n"


@pytest.mark.asyncio
async def test_worktree_remove_ignores_an_unmanaged_worktree(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    response = await _remove_worktree_over_rpc(tmp_path)

    assert response.outcome == WorktreeReleaseOutcome.KEPT_UNMANAGED
    assert tmp_path.is_dir()


@pytest.mark.asyncio
async def test_worktree_remove_forgets_a_worktree_already_gone(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    worktree = _prepare_auto(tmp_path)
    repo.git.worktree("remove", "--force", str(worktree.root))

    response = await _remove_worktree_over_rpc(worktree.root)

    assert response.outcome == WorktreeReleaseOutcome.NOT_FOUND


@pytest.mark.asyncio
async def test_worktree_remove_is_reachable_while_a_root_is_attached(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    worktree = _prepare_auto(tmp_path)

    response = await _remove_worktree_over_rpc(worktree.root, attach_root=tmp_path)

    assert response.outcome == WorktreeReleaseOutcome.REMOVED
    assert not worktree.root.exists()
