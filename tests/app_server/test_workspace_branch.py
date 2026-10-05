from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import threading

import pytest

from chartreux.app_server import _host
from chartreux.app_server._host import workspace_branch_response
from chartreux.app_server.protocol import WorkspaceBranchReadParams
from chartreux.core.git.repo import GitRepo
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import attach_test_app_server_session, start_test_app_server


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-b", "main")
    _git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "--allow-empty",
        "-m",
        "initial",
    )
    return tmp_path


def test_branch_and_detached_head(repository: Path) -> None:
    params = WorkspaceBranchReadParams(session_id="session", cwd=str(repository))
    response = workspace_branch_response(params)
    assert (response.branch, response.status) == ("main", "branch")
    _git(repository, "checkout", "--detach")
    response = workspace_branch_response(params)
    assert (response.branch, response.status) == (None, "detached")


def test_non_repository(tmp_path: Path) -> None:
    response = workspace_branch_response(
        WorkspaceBranchReadParams(session_id="session", cwd=str(tmp_path))
    )
    assert (response.branch, response.status) == (None, "not_repository")


def test_failed_lookup_closes_repository(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[GitRepo] = []
    original_close = GitRepo.close

    def close(repo: GitRepo) -> None:
        closed.append(repo)
        original_close(repo)

    def fail(repo: GitRepo) -> str | None:
        raise RuntimeError("broken HEAD")

    monkeypatch.setattr(GitRepo, "close", close)
    monkeypatch.setattr(GitRepo, "branch", fail)
    response = workspace_branch_response(
        WorkspaceBranchReadParams(session_id="session", cwd=str(repository))
    )
    assert (response.branch, response.status) == (None, "unknown")
    assert len(closed) == 1


@pytest.mark.asyncio
async def test_cache_refresh_and_off_event_loop(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = start_test_app_server(build_test_agent_loop())
    session = await attach_test_app_server_session(client)
    session.state.session.cwd = str(repository)
    threads: list[int] = []
    original = _host.workspace_branch_response

    def probe(params: WorkspaceBranchReadParams):
        threads.append(threading.get_ident())
        return original(params)

    monkeypatch.setattr(_host, "workspace_branch_response", probe)
    try:
        first = await session.resources.workspace.read_branch()
        assert first is not None and first.branch == "main"
        _git(repository, "checkout", "-b", "changed")
        cached = await session.resources.workspace.read_branch()
        assert cached is not None and cached.branch == "main"
        refreshed = await session.resources.workspace.read_branch(refresh=True)
        assert refreshed is not None and refreshed.branch == "changed"
        assert len(threads) == 2
        assert all(thread != threading.get_ident() for thread in threads)
        session.state.session.cwd = None
        unknown = await session.resources.workspace.read_branch()
        assert unknown is not None and unknown.status == "unknown"
        assert unknown.cwd == ""
        assert len(threads) == 2
        session.state.session.cwd = str(repository)
        reread = await session.resources.workspace.read_branch()
        assert reread is not None and reread.branch == "changed"
        assert len(threads) == 3
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cwd", "session", "refresh"])
async def test_stale_in_flight_result_is_rejected(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    client = start_test_app_server(build_test_agent_loop())
    session = await attach_test_app_server_session(client)
    session.state.session.cwd = str(repository)
    started = threading.Event()
    release = threading.Event()
    original = _host.workspace_branch_response
    calls = 0

    def probe(params: WorkspaceBranchReadParams):
        nonlocal calls
        calls += 1
        response = original(params)
        if calls == 1:
            started.set()
            assert release.wait(timeout=10)
        return response

    monkeypatch.setattr(_host, "workspace_branch_response", probe)
    pending = asyncio.create_task(session.resources.workspace.read_branch())
    try:
        assert await asyncio.to_thread(started.wait, 10)
        if change == "cwd":
            other = tmp_path / "not-a-repo"
            other.mkdir()
            # This directory is inside the repo: use a missing path instead.
            session.state.session.cwd = str(other / "missing")
        elif change == "session":
            previous_id = session.state.session.id
            session.state.session.id = "new-session"
            release.set()
            assert await pending is None
            session.state.session.id = previous_id
        else:
            _git(repository, "checkout", "-b", "new-branch")
        newest = await session.resources.workspace.read_branch(refresh=True)
        assert newest is not None
        release.set()
        assert await pending is None
        assert await session.resources.workspace.read_branch() == newest
        assert calls == 2
    finally:
        release.set()
        await pending
        await session.close()


@pytest.mark.asyncio
async def test_unknown_result_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = start_test_app_server(build_test_agent_loop())
    session = await attach_test_app_server_session(client)
    session.state.session.cwd = str(tmp_path)
    calls = 0

    def fail(base: Path) -> GitRepo:
        raise RuntimeError("Git unavailable")

    original = _host.workspace_branch_response

    def probe(params: WorkspaceBranchReadParams):
        nonlocal calls
        calls += 1
        return original(params)

    monkeypatch.setattr(_host, "workspace_branch_response", probe)
    monkeypatch.setattr(GitRepo, "open", fail)
    try:
        first = await session.resources.workspace.read_branch()
        assert first is not None and first.status == "unknown"
        assert await session.resources.workspace.read_branch() == first
        assert calls == 1
    finally:
        await session.close()
