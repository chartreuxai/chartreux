"""Tests for the environmental guard against stray tmp-tree repositories.

Every failure-path test builds a synthetic ancestor tree under ``tmp_path``
and injects its roots, so the guard's detection is proven without touching
the real ``/tmp``. The guard's own autouse fixture also runs for these
tests, which doubles as a live probe of the real tmp tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tools._tmp_scope_guard import (
    assert_no_stray_repository,
    guard_roots,
    stray_git_markers,
)


def make_tmp_tree(tmp_path: Path) -> tuple[Path, Path]:
    """A synthetic tmp chain: a ``basetemp`` inside a fake ``/tmp``.

    Returns the fake ``/tmp`` ancestor and the basetemp beneath it.
    """
    tmp = tmp_path / "fake-tmp"
    basetemp = tmp / "pytest-of-user" / "pytest-3"
    basetemp.mkdir(parents=True)
    return tmp, basetemp


def test_clean_chain_reports_no_markers(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    (basetemp / "test_clean0").mkdir()
    assert stray_git_markers(guard_roots(basetemp)) == []


def test_stray_dotgit_directory_at_an_ancestor_is_detected(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    (tmp / ".git").mkdir()
    (tmp / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp / ".git" / "objects").mkdir()
    (tmp / ".git" / "refs").mkdir()
    assert stray_git_markers(guard_roots(basetemp)) == [tmp / ".git"]


def test_stray_gitdir_pointer_file_at_an_ancestor_is_detected(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    (tmp / ".git").write_text("gitdir: /elsewhere/repo/.git\n")
    assert stray_git_markers(guard_roots(basetemp)) == [tmp / ".git"]


def test_stray_bare_repository_at_an_ancestor_is_detected(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    (tmp / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp / "config").write_text("[core]\n\trepositoryformatversion = 0\n")
    (tmp / "objects").mkdir()
    (tmp / "refs").mkdir()
    assert stray_git_markers(guard_roots(basetemp)) == [tmp / "HEAD"]


def test_incomplete_bare_markers_are_not_detected(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    (tmp / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp / "objects").mkdir()
    # No refs/ directory: the resolver itself does not see a repository.
    assert stray_git_markers(guard_roots(basetemp)) == []


def test_the_basetemp_itself_is_checked(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    (basetemp / ".git").mkdir()
    (basetemp / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (basetemp / ".git" / "objects").mkdir()
    (basetemp / ".git" / "refs").mkdir()
    assert stray_git_markers(guard_roots(basetemp)) == [basetemp / ".git"]


def test_leaf_directories_are_not_roots(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    test_dir = basetemp / "test_with_repo0"
    test_dir.mkdir()
    (test_dir / ".git").mkdir()
    (test_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (test_dir / ".git" / "objects").mkdir()
    (test_dir / ".git" / "refs").mkdir()
    # Repositories inside the per-test tmp directories are created on
    # purpose; the guard only inspects the basetemp and its ancestors.
    assert stray_git_markers(guard_roots(basetemp)) == []


def test_guard_roots_spans_basetemp_to_filesystem_root(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    roots = guard_roots(basetemp)
    assert roots[0] == basetemp.resolve()
    assert roots[1] == (basetemp / "..").resolve()
    assert roots[-1] == Path(tmp.anchor)


def test_failure_path_names_the_stray_path_and_the_remedy(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    (tmp / ".git").mkdir()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_no_stray_repository(guard_roots(basetemp))
    message = str(excinfo.value)
    assert str(tmp / ".git") in message
    assert "--basetemp" in message


def test_failure_path_is_silent_on_a_clean_chain(tmp_path: Path) -> None:
    tmp, basetemp = make_tmp_tree(tmp_path)
    assert_no_stray_repository(guard_roots(basetemp))


def test_live_tmp_tree_is_clean(tmp_path_factory) -> None:
    assert_no_stray_repository(guard_roots(tmp_path_factory.getbasetemp()))
