from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config.harness_files import HarnessFilesManager


def _dirs(tmp_path: Path, *names: str) -> list[Path]:
    made = []
    for name in names:
        d = tmp_path / name
        d.mkdir()
        made.append(d.resolve())
    return made


def test_local_extensions_survive_without_plugin_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, project = _dirs(tmp_path, "home", "project")
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    for root in (home, project / ".chartreux"):
        for name in ("tools", "skills", "agents", "plugins"):
            (root / name).mkdir(parents=True)
        (root / "hooks.toml").write_text("")
    manager = HarnessFilesManager(sources=("user",), _additional_dirs=(project,))

    assert manager.user_tools_dirs == [home / "tools"]
    assert home / "skills" in manager.user_skills_dirs
    assert manager.user_agents_dirs == [home / "agents"]
    assert manager.project_tools_dirs == [project / ".chartreux" / "tools"]
    assert manager.project_skills_dirs == [project / ".chartreux" / "skills"]
    assert manager.project_agents_dirs == [project / ".chartreux" / "agents"]
    assert manager.hook_files == [
        project / ".chartreux" / "hooks.toml",
        home / "hooks.toml",
    ]
    assert not hasattr(manager, "user_plugins_dirs")
    assert not hasattr(manager, "project_plugins_dirs")
    assert (home / "plugins").is_dir()
    assert (project / ".chartreux" / "plugins").is_dir()


def test_a_move_replaces_the_departure_root_rather_than_adding_to_it(
    tmp_path: Path,
) -> None:
    repo, worktree = _dirs(tmp_path, "repo", "worktree")
    started = HarnessFilesManager(sources=("project",)).for_session(
        repo, workspace_roots=[repo]
    )

    moved = started.moved_to(worktree)

    assert moved.cwd == worktree
    assert moved.project_roots == [worktree]


def test_a_sequence_of_moves_never_widens_the_root_set(tmp_path: Path) -> None:
    # What D3 forbids. for_session merges, so routing a move through it would
    # leave every directory the session had ever sat in still authorised.
    first, second, third = _dirs(tmp_path, "first", "second", "third")
    manager = HarnessFilesManager(sources=("project",)).for_session(
        first, workspace_roots=[first]
    )

    manager = manager.moved_to(second).moved_to(third)

    assert manager.project_roots == [third]


def test_roots_held_for_other_reasons_survive_a_move(tmp_path: Path) -> None:
    # The attachment cache and --add-dir entries are not the directory the
    # session sits in, so a move has no business revoking them.
    repo, worktree, attachments = _dirs(tmp_path, "repo", "worktree", "attachments")
    started = HarnessFilesManager(sources=("project",)).for_session(
        repo, workspace_roots=[repo, attachments]
    )

    moved = started.moved_to(worktree)

    assert moved.project_roots == [attachments, worktree]


def test_an_opted_in_root_survives_the_session_passing_through_it(
    tmp_path: Path,
) -> None:
    # Leaving a directory revokes it as the session's position, not as a root
    # the user opened. An --add-dir the session moves into and back out of has
    # to still be there afterwards, or the move has quietly taken it away.
    repo, attachments, worktree = _dirs(tmp_path, "repo", "attachments", "worktree")
    started = HarnessFilesManager(sources=("project",)).for_session(
        repo, workspace_roots=[repo, attachments]
    )

    moved = started.moved_to(attachments).moved_to(worktree)

    assert moved.project_roots == [attachments, worktree]


def test_moving_where_the_session_already_sits_is_not_a_widening(
    tmp_path: Path,
) -> None:
    (repo,) = _dirs(tmp_path, "repo")
    started = HarnessFilesManager(sources=("project",)).for_session(
        repo, workspace_roots=[repo]
    )

    assert started.moved_to(repo).project_roots == [repo]
