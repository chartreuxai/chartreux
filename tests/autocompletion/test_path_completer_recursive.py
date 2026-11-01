from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.cli.autocompletion.completers import PathCompleter


@pytest.fixture()
def file_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "chartreux" / "acp").mkdir(parents=True)
    (tmp_path / "chartreux" / "acp" / "entrypoint.py").write_text("")
    (tmp_path / "chartreux" / "acp" / "agent.py").write_text("")
    (tmp_path / "chartreux" / "cli" / "autocompletion").mkdir(parents=True)
    (tmp_path / "chartreux" / "cli" / "autocompletion" / "fuzzy.py").write_text("")
    (tmp_path / "chartreux" / "cli" / "autocompletion" / "completers.py").write_text("")
    (tmp_path / "tests" / "autocompletion").mkdir(parents=True)
    (tmp_path / "tests" / "autocompletion" / "test_fuzzy.py").write_text("")
    (tmp_path / "README.md").write_text("")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_finds_files_recursively_by_filename(file_tree: Path) -> None:
    results = PathCompleter().get_completions("@entryp", cursor_pos=7)

    assert results[0] == "@chartreux/acp/entrypoint.py"


def test_finds_files_recursively_by_partial_path(file_tree: Path) -> None:
    results = PathCompleter().get_completions("@acp/entry", cursor_pos=10)

    assert results[0] == "@chartreux/acp/entrypoint.py"


def test_finds_files_recursively_with_subsequence(file_tree: Path) -> None:
    results = PathCompleter().get_completions("@acp/ent", cursor_pos=9)

    assert results[0] == "@chartreux/acp/entrypoint.py"


def test_finds_multiple_matches_recursively(file_tree: Path) -> None:
    results = PathCompleter().get_completions("@fuzzy", cursor_pos=6)

    vibe_index = results.index("@chartreux/cli/autocompletion/fuzzy.py")
    test_index = results.index("@tests/autocompletion/test_fuzzy.py")
    assert vibe_index < test_index


def test_prioritizes_exact_path_matches(file_tree: Path) -> None:
    results = PathCompleter().get_completions(
        "@chartreux/acp/entrypoint", cursor_pos=20
    )

    assert results[0] == "@chartreux/acp/entrypoint.py"


def test_finds_files_when_pattern_matches_directory_name(file_tree: Path) -> None:
    results = PathCompleter().get_completions("@acp", cursor_pos=4)

    assert results == [
        "@chartreux/acp/",
        "@chartreux/acp/agent.py",
        "@chartreux/acp/entrypoint.py",
        "@tests/autocompletion/",
        "@tests/autocompletion/test_fuzzy.py",
        "@chartreux/cli/autocompletion/",
        "@chartreux/cli/autocompletion/completers.py",
        "@chartreux/cli/autocompletion/fuzzy.py",
    ]


@pytest.fixture()
def workspace_away_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "project" / "workspace"
    (workspace / "chartreux" / "acp").mkdir(parents=True)
    (workspace / "chartreux" / "acp" / "entrypoint.py").write_text("")
    (workspace / "README.md").write_text("")
    sibling = tmp_path / "project" / "sibling"
    (sibling / "nested").mkdir(parents=True)
    (sibling / "note.txt").write_text("")
    cwd = tmp_path / "elsewhere" / "cwd"
    cwd.mkdir(parents=True)
    (cwd / "cwd_marker.txt").write_text("")
    monkeypatch.chdir(cwd)
    return workspace


def test_root_listing_anchors_at_workspace_root_not_cwd(
    workspace_away_from_cwd: Path,
) -> None:
    completer = PathCompleter(workspace_root_getter=lambda: workspace_away_from_cwd)

    assert completer.get_completions("@", cursor_pos=1) == ["@chartreux/", "@README.md"]


def test_recursive_index_anchors_at_workspace_root_not_cwd(
    workspace_away_from_cwd: Path,
) -> None:
    completer = PathCompleter(workspace_root_getter=lambda: workspace_away_from_cwd)

    assert completer.get_completions("@entryp", cursor_pos=7) == [
        "@chartreux/acp/entrypoint.py"
    ]
    assert completer.get_completions("@acp/entry", cursor_pos=10) == [
        "@chartreux/acp/entrypoint.py"
    ]


def test_parent_traversal_anchors_at_workspace_root_not_cwd(
    workspace_away_from_cwd: Path,
) -> None:
    completer = PathCompleter(workspace_root_getter=lambda: workspace_away_from_cwd)

    assert completer.get_completions("@../", cursor_pos=4) == [
        "@../sibling/",
        "@../workspace/",
    ]
    assert completer.get_completions("@../sibling/", cursor_pos=12) == [
        "@../sibling/nested/",
        "@../sibling/note.txt",
    ]
