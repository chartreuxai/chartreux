from __future__ import annotations

import asyncio
from pathlib import Path
import threading

from git import Repo
import pytest

from chartreux.app_server import _worktree_session
from chartreux.app_server._host import worktree_remove_response
from chartreux.app_server._worktree_session import SessionWorktrees
from chartreux.app_server.protocol import SessionOptions
from chartreux.core.git.worktree import (
    ManagedWorktree,
    WorktreeRepository,
    abort_transfer,
    begin_transfer,
    commit_transfer,
)
from chartreux.core.git.worktree.record import (
    HolderKind,
    OwnershipToken,
    release_holder,
)


def _init_repo(root: Path) -> None:
    repo = Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "name", "Tester").release()
    repo.config_writer().set_value("user", "email", "t@example.com").release()
    (root / "file.txt").write_text("hello\n")
    repo.index.add(["file.txt"])
    repo.index.commit("initial")


def _managed_worktree(base: Path, prompt: str = "lifecycle hold") -> Path:
    with WorktreeRepository.open(base) as repository:
        prepared = repository.prepare_auto(prompt=prompt)
    assert prepared.pending_token is not None
    release_holder(prepared.pending_token)
    return prepared.path


@pytest.mark.asyncio
async def test_ordinary_managed_cwd_is_held_while_start_is_pending(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    cwd = _managed_worktree(tmp_path)

    resolution = await SessionWorktrees().resolve_for_start(
        SessionOptions(cwd=str(cwd))
    )
    assert resolution.pending_token is not None
    assert resolution.pending_token.holder_kind is HolderKind.PENDING_ATTACHMENT

    await SessionWorktrees().cleanup(resolution)
    managed = ManagedWorktree.at(cwd)
    assert managed is not None
    assert not managed.holders()


@pytest.mark.asyncio
async def test_cancellation_waits_for_attachment_acquisition_and_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    cwd = _managed_worktree(tmp_path)
    entered = threading.Event()
    finish = threading.Event()
    acquire = _worktree_session.acquire_for_attachment

    def blocked(path: Path):
        entered.set()
        finish.wait(timeout=5)
        return acquire(path)

    monkeypatch.setattr(_worktree_session, "acquire_for_attachment", blocked)
    task = asyncio.create_task(
        SessionWorktrees().resolve_for_start(SessionOptions(cwd=str(cwd)))
    )
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    managed = ManagedWorktree.at(cwd)
    assert managed is not None
    assert not managed.holders()


def test_relocation_transfer_releases_only_the_rolled_back_destination(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    source = _managed_worktree(tmp_path, "source hold")
    target = _managed_worktree(tmp_path, "target hold")
    source_managed = ManagedWorktree.at(source)
    target_managed = ManagedWorktree.at(target)
    assert source_managed is not None
    assert target_managed is not None
    source_token = SessionWorktrees.hold(source, "session")
    assert source_token is not None

    attempt = begin_transfer(source_token, target)
    abort_transfer(attempt)
    assert source_managed.holders() == {"session"}
    assert not target_managed.holders()

    attempt = begin_transfer(source_token, target)
    target_token = commit_transfer(attempt)
    assert not source_managed.holders()
    release_holder(target_token)


def test_host_removal_fails_while_a_concurrent_attachment_is_live(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    cwd = _managed_worktree(tmp_path)
    attached = WorktreeRepository.acquire_for_attachment(cwd)
    assert attached.token is not None

    response = worktree_remove_response(cwd)

    assert response.outcome == "kept_in_use"
    assert cwd.exists()
    release_holder(attached.token)


def test_host_removal_waits_for_an_attachment_published_by_another_thread(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    cwd = _managed_worktree(tmp_path)
    barrier = threading.Barrier(2)
    attached = threading.Event()
    release = threading.Event()
    tokens: list[OwnershipToken] = []

    def attach() -> None:
        barrier.wait(timeout=5)
        acquired = WorktreeRepository.acquire_for_attachment(cwd)
        assert acquired.token is not None
        tokens.append(acquired.token)
        attached.set()
        assert release.wait(timeout=5)

    worker = threading.Thread(target=attach)
    worker.start()
    try:
        barrier.wait(timeout=5)
        assert attached.wait(timeout=5)

        response = worktree_remove_response(cwd)

        assert response.outcome == "kept_in_use"
        assert cwd.exists()
    finally:
        release.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        if tokens:
            release_holder(tokens.pop())
