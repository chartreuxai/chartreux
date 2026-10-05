from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re
import threading

from git import Repo
import pytest

import chartreux.core.usage_project as usage_project
from chartreux.core.usage_project import resolve_project_key, resolve_project_key_async


@pytest.fixture
def git_repo(tmp_path: Path) -> Iterator[Repo]:
    with Repo.init(tmp_path / "repo", initial_branch="main") as repo:
        with repo.config_writer() as config:
            config.set_value("user", "name", "Tester")
            config.set_value("user", "email", "t@example.com")
        repo.index.commit("initial")
        yield repo


def test_linked_worktrees_share_common_git_identity(
    git_repo: Repo, tmp_path: Path
) -> None:
    root = Path(git_repo.working_dir)
    first = tmp_path / "first"
    second = tmp_path / "second"
    git_repo.git.worktree("add", "-b", "first", str(first))
    git_repo.git.worktree("add", "--detach", str(second))

    key = resolve_project_key(root)
    assert key.startswith("git-")
    assert resolve_project_key(first) == key
    assert resolve_project_key(second) == key


def test_separate_clones_have_distinct_keys(git_repo: Repo, tmp_path: Path) -> None:
    root = Path(git_repo.working_dir)
    with (
        Repo.clone_from(str(root), tmp_path / "clone-one") as first,
        Repo.clone_from(str(root), tmp_path / "clone-two") as second,
    ):
        assert first.remotes.origin.url == second.remotes.origin.url
        keys = {
            resolve_project_key(root),
            resolve_project_key(Path(first.working_dir)),
            resolve_project_key(Path(second.working_dir)),
        }
        assert len(keys) == 3


@pytest.mark.parametrize("is_git", [False, True])
def test_symlinked_root_has_same_key(
    git_repo: Repo, tmp_path: Path, is_git: bool
) -> None:
    root = Path(git_repo.working_dir) if is_git else tmp_path / "plain"
    root.mkdir(exist_ok=True)
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)

    assert resolve_project_key(link) == resolve_project_key(root)
    assert resolve_project_key(link / ".." / root.name) == resolve_project_key(root)


def test_non_git_workspace_uses_canonical_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    child = root / "child"
    child.mkdir()

    key = resolve_project_key(root)
    assert key.startswith("path-")
    assert resolve_project_key(root / ".") == key
    assert resolve_project_key(child) != key


def test_missing_workspace_has_stable_path_key(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "workspace"
    key = resolve_project_key(missing)
    assert key.startswith("path-")
    assert resolve_project_key(missing) == key
    assert not missing.exists()
    missing.mkdir(parents=True)
    assert resolve_project_key(missing) == key


def test_unavailable_git_falls_back_to_workspace_path(
    git_repo: Repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(git_repo.working_dir)
    assert resolve_project_key(root).startswith("git-")
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("GIT_PYTHON_GIT_EXECUTABLE", raising=False)

    key = resolve_project_key(root)
    assert key.startswith("path-")
    assert resolve_project_key(root) == key


@pytest.mark.parametrize("is_git", [False, True])
def test_key_is_stable_short_and_filesystem_safe(
    git_repo: Repo, tmp_path: Path, is_git: bool
) -> None:
    root = Path(git_repo.working_dir) if is_git else tmp_path
    key = resolve_project_key(root)
    assert re.fullmatch(r"(?:git|path)-[0-9a-f]{32}", key)
    assert resolve_project_key(str(root)) == key
    assert resolve_project_key(root) == key


def test_explicit_root_is_independent_of_child_cwd(
    git_repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(git_repo.working_dir)
    key = resolve_project_key(root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert resolve_project_key(root) == key
    assert resolve_project_key(elsewhere) != key


@pytest.mark.asyncio
async def test_async_resolution_runs_off_event_loop(
    git_repo: Repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(git_repo.working_dir)
    expected = resolve_project_key(root)
    event_loop_thread = threading.get_ident()
    resolver = resolve_project_key

    def checked_resolver(root_workspace: str | Path) -> str:
        assert threading.get_ident() != event_loop_thread
        return resolver(root_workspace)

    monkeypatch.setattr(usage_project, "resolve_project_key", checked_resolver)
    assert await resolve_project_key_async(root) == expected
