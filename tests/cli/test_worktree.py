from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from git import Repo
import pytest

from chartreux.cli import entrypoint
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.git.errors import GitError
from chartreux.core.git.worktree import (
    PreparedWorktree,
    WorktreeError,
    WorktreeRepository,
    acquire_for_attachment,
)
from chartreux.core.git.worktree.record import (
    OwnershipToken,
    WorktreeClaim,
    inspect_holders,
    release_holder,
)
from chartreux.core.paths import CHARTREUX_HOME
from chartreux.core.session.session_loader import SessionLoader

# The API is repository-scoped; these keep a test that only cares about the
# resulting worktree down to a single call.


def _prepare(name: str, base: Path, *, branch: str | None = None) -> PreparedWorktree:
    with WorktreeRepository.open(base) as repository:
        return repository.prepare(name, branch=branch)


def _prepare_path(name: str, base: Path) -> Path:
    return _prepare(name, base).path


def _init_repo(workdir: Path) -> Repo:
    repo = Repo.init(workdir, initial_branch="main")
    repo.config_writer().set_value("user", "name", "Tester").release()
    repo.config_writer().set_value("user", "email", "t@example.com").release()
    (workdir / "file.txt").write_text("hello\n")
    repo.index.add(["file.txt"])
    repo.index.commit("initial")
    return repo


@pytest.fixture
def git_repo(tmp_path: Path) -> Repo:
    return _init_repo(tmp_path)


def _expected_worktree_target(repo_root: Path, name: str) -> Path:
    common_git_dir = (repo_root / ".git").resolve()
    repo_hash = hashlib.sha256(str(common_git_dir).encode()).hexdigest()[:12]
    return CHARTREUX_HOME.path / "worktrees" / f"{repo_root.name}-{repo_hash}" / name


def test_creates_worktree_and_branch(git_repo: Repo, tmp_path: Path) -> None:
    target = _prepare_path("feature", tmp_path)

    assert target == _expected_worktree_target(tmp_path, "feature")
    assert target.is_dir()
    assert (target / ".git").is_file()
    assert "feature" in (h.name for h in git_repo.heads)


def test_does_not_touch_git_info_exclude(git_repo: Repo, tmp_path: Path) -> None:
    # Worktrees live under CHARTREUX_HOME, not inside the repo, so preparing one
    # must not append a ".worktrees/" ignore line to .git/info/exclude.
    _prepare_path("feature", tmp_path)

    exclude_file = tmp_path / ".git" / "info" / "exclude"
    exclude = exclude_file.read_text() if exclude_file.exists() else ""
    assert ".worktrees/" not in exclude.splitlines()


def test_namespaces_same_worktree_name_by_repo(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _init_repo(first_root)
    _init_repo(second_root)

    first = _prepare_path("feature", first_root)
    second = _prepare_path("feature", second_root)

    assert first == _expected_worktree_target(first_root, "feature")
    assert second == _expected_worktree_target(second_root, "feature")
    assert first != second


def test_preserves_source_subdirectory(git_repo: Repo, tmp_path: Path) -> None:
    subdir = tmp_path / "pkg"
    subdir.mkdir()
    (subdir / "module.py").write_text("print('hello')\n")
    git_repo.index.add(["pkg/module.py"])
    git_repo.index.commit("add package")

    target = _prepare_path("feature", subdir)

    assert target == _expected_worktree_target(tmp_path, "feature") / "pkg"
    assert target.is_dir()
    assert (target / "module.py").is_file()


def test_linked_worktree_uses_primary_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    linked = _prepare_path("feature", repo_root)
    exclude = repo_root / ".git" / "info" / "exclude"
    exclude.write_text("")

    target = _prepare_path("other", linked)

    assert target == _expected_worktree_target(repo_root, "other")
    assert target.is_dir()
    assert not (linked / ".worktrees").exists()
    assert exclude.read_text() == ""


def test_reuse_existing_worktree(git_repo: Repo, tmp_path: Path) -> None:
    first = _prepare_path("feature", tmp_path)
    second = _prepare_path("feature", tmp_path)

    assert first == second


def test_clean_worktree_cleanup_state_is_clean(git_repo: Repo, tmp_path: Path) -> None:
    worktree = _prepare("feature", tmp_path)

    cleanup_state = worktree.inspect_for_cleanup()

    assert cleanup_state.is_clean is True
    assert cleanup_state.reasons == ()


def test_cleanup_state_detects_untracked_files(git_repo: Repo, tmp_path: Path) -> None:
    worktree = _prepare("feature", tmp_path)
    (worktree.root / "new.txt").write_text("hello\n")

    cleanup_state = worktree.inspect_for_cleanup()

    assert cleanup_state.is_clean is False
    assert cleanup_state.has_untracked_files is True
    assert cleanup_state.reasons == ("untracked files",)


def test_cleanup_state_detects_uncommitted_changes(
    git_repo: Repo, tmp_path: Path
) -> None:
    worktree = _prepare("feature", tmp_path)
    (worktree.root / "file.txt").write_text("changed\n")

    cleanup_state = worktree.inspect_for_cleanup()

    assert cleanup_state.is_clean is False
    assert cleanup_state.has_uncommitted_changes is True
    assert cleanup_state.reasons == ("uncommitted changes",)


def test_cleanup_state_detects_new_commits(git_repo: Repo, tmp_path: Path) -> None:
    worktree = _prepare("feature", tmp_path)
    repo = Repo(worktree.root)
    (worktree.root / "file.txt").write_text("changed\n")
    repo.index.add(["file.txt"])
    repo.index.commit("change")

    cleanup_state = worktree.inspect_for_cleanup()

    assert cleanup_state.is_clean is False
    assert cleanup_state.new_commit_count == 1
    assert cleanup_state.reasons == ("1 commit added during this session",)


def test_reused_worktree_cleanup_starts_from_current_head(
    git_repo: Repo, tmp_path: Path
) -> None:
    worktree = _prepare("feature", tmp_path)
    repo = Repo(worktree.root)
    (worktree.root / "file.txt").write_text("changed\n")
    repo.index.add(["file.txt"])
    repo.index.commit("prior session commit")

    # Reusing the worktree resets base_commit to the worktree's current HEAD, so
    # commits from a previous session are not recounted as new work.
    reused = _prepare("feature", tmp_path)

    assert reused.inspect_for_cleanup().new_commit_count == 0


def test_remove_worktree_deletes_directory_and_branch(
    git_repo: Repo, tmp_path: Path
) -> None:
    worktree = _prepare("feature", tmp_path)

    worktree.remove()

    assert not worktree.root.exists()
    assert "feature" not in (h.name for h in git_repo.heads)


def test_remove_worktree_keeps_branch_when_delete_branch_false(
    git_repo: Repo, tmp_path: Path
) -> None:
    worktree = _prepare("feature", tmp_path)

    worktree.remove(delete_branch=False)

    assert not worktree.root.exists()
    assert "feature" in (h.name for h in git_repo.heads)


def test_attaching_existing_branch_is_not_marked_branch_created(
    git_repo: Repo, tmp_path: Path
) -> None:
    git_repo.create_head("feature")

    worktree = _prepare("feature", tmp_path)

    assert worktree.created is True
    assert worktree.branch_created is False


def test_new_branch_worktree_is_marked_branch_created(
    git_repo: Repo, tmp_path: Path
) -> None:
    worktree = _prepare("feature", tmp_path)

    assert worktree.created is True
    assert worktree.branch_created is True


def test_existing_worktree_on_wrong_branch_raises(
    git_repo: Repo, tmp_path: Path
) -> None:
    target = _prepare_path("feature", tmp_path)
    Repo(target).git.checkout("-b", "other")

    with pytest.raises(WorktreeError, match="expected 'feature'"):
        _prepare_path("feature", tmp_path)


def test_existing_worktree_for_different_repo_raises(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _init_repo(first_root)
    second_repo = _init_repo(second_root)
    target = _expected_worktree_target(first_root, "feature")
    second_repo.git.worktree("add", "-b", "feature", str(target))

    with pytest.raises(WorktreeError, match="different git repository"):
        _prepare_path("feature", first_root)


def test_attaches_existing_branch(git_repo: Repo, tmp_path: Path) -> None:
    git_repo.create_head("feature")

    target = _prepare_path("feature", tmp_path)

    assert target.is_dir()
    assert (target / ".git").is_file()


@pytest.mark.parametrize(
    "name", ["", ".", "..", "nested/name", "nested\\name", "/tmp/outside"]
)
def test_rejects_worktree_names_that_are_not_path_segments(
    git_repo: Repo, tmp_path: Path, name: str
) -> None:
    with pytest.raises(WorktreeError, match="single path segment"):
        _prepare_path(name, tmp_path)


def test_existing_non_worktree_dir_raises(git_repo: Repo, tmp_path: Path) -> None:
    target = _expected_worktree_target(tmp_path, "feature")
    target.mkdir(parents=True)

    with pytest.raises(WorktreeError, match="not a git worktree"):
        _prepare_path("feature", tmp_path)


def test_non_git_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        _prepare_path("feature", tmp_path)


def _write_session(save_dir: Path, name: str, working_directory: Path) -> None:
    session_dir = save_dir / f"session_{name}"
    session_dir.mkdir(parents=True)
    (session_dir / "messages.jsonl").write_text("{}\n")
    (session_dir / "meta.json").write_text(
        json.dumps({"environment": {"working_directory": str(working_directory)}})
    )


def test_worktree_continue_scopes_to_worktree(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    target = _prepare_path("feature", repo_root)

    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    _write_session(save_dir, "main", repo_root)
    config = SessionLoggingConfig(save_dir=str(save_dir))

    assert SessionLoader.find_latest_session(config, working_directory=target) is None
    assert (
        SessionLoader.find_latest_session(config, working_directory=repo_root)
        is not None
    )


def _worktree_args(name: str) -> argparse.Namespace:
    return argparse.Namespace(worktree=name, prompt=None, initial_prompt=None)


def test_cli_acquires_holder_before_chdir_and_releases_on_chdir_failure(
    git_repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    original_chdir = entrypoint.os.chdir

    def fail_chdir(path: Path) -> None:
        if path != tmp_path:
            raise OSError("cannot enter worktree")
        original_chdir(path)

    monkeypatch.setattr(entrypoint.os, "chdir", fail_chdir)

    with pytest.raises(OSError, match="cannot enter worktree"):
        entrypoint._enter_worktree(_worktree_args("feature"))

    claim = WorktreeClaim.locate(_expected_worktree_target(tmp_path, "feature"))
    assert claim is not None
    assert inspect_holders(claim).holders == {}


@pytest.mark.parametrize("source", ["initial_prompt", "piped_stdin"])
def test_programmatic_worktree_run_skips_interactive_cleanup(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    args = argparse.Namespace(prompt=None, initial_prompt=None)
    if source == "initial_prompt":
        args.initial_prompt = "prompt"

    def run_cli(reported_args: argparse.Namespace) -> None:
        reported_args.is_programmatic = True

    cleanup_called = False

    def cleanup(*_args: object) -> None:
        nonlocal cleanup_called
        cleanup_called = True

    monkeypatch.setattr("chartreux.cli.cli.run_cli", run_cli)
    monkeypatch.setattr(entrypoint, "_cleanup_worktree_on_exit", cleanup)

    entrypoint._run_cli_with_worktree_cleanup(
        args,
        cast(PreparedWorktree, SimpleNamespace(created=True)),
        cast(OwnershipToken, SimpleNamespace(consumed=True)),
    )

    assert not cleanup_called


def test_cli_cleanup_keeps_worktree_when_attachment_arrives_during_prompt(
    git_repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = _prepare("feature", tmp_path)
    assert worktree.pending_token is not None
    release_holder(worktree.pending_token)
    (worktree.root / "untracked.txt").write_text("work\n")
    monkeypatch.chdir(worktree.root)
    session, cli_token = entrypoint._enter_worktree(_worktree_args("feature"))
    attached = None

    def approve_after_attachment(*args: object) -> bool:
        nonlocal attached
        attached = acquire_for_attachment(session.root)
        return True

    monkeypatch.setattr(entrypoint, "_prompt_remove_worktree", approve_after_attachment)
    monkeypatch.setattr(entrypoint, "_prompt_delete_attached_branch", lambda _: False)
    entrypoint._cleanup_worktree_on_exit(session, cli_token)

    assert attached is not None and attached.token is not None
    assert session.root.exists()
    release_holder(attached.token)
    if not cli_token.consumed:
        release_holder(cli_token)
