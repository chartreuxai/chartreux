"""Unit tests for the project-env injection module (P2-b).

The loader reads exactly two dotenv files — the trusted folder's ``.env`` and
``$CHARTREUX_HOME/.env`` — trust-gated, mtime-cached, with no cascading
lookups. Trust decisions come from an in-memory stub manager, and the global
file is steered through a ``CHARTREUX_HOME`` pointing into ``tmp_path``, so
no test touches the real trust store or the real Chartreux home.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chartreux.core import env
from chartreux.core.trusted_folders import TrustedFoldersManager


class StubTrustManager(TrustedFoldersManager):
    """Trust decisions served from memory, never touching the real trust store."""

    def __init__(self, roots: dict[Path, Path]) -> None:
        # Maps a working folder to its trust root; unmapped folders are untrusted.
        self._roots = roots

    def find_trust_root(self, path: Path) -> Path | None:
        return self._roots.get(path)


@pytest.fixture(autouse=True)
def _clean_cache():
    env.reset_cache()
    yield
    env.reset_cache()


@pytest.fixture
def chartreux_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "chartreux-home"
    home.mkdir()
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    return home


def test_load_project_env_merges_exactly_two_files_project_wins(
    tmp_path: Path, chartreux_home: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (chartreux_home / ".env").write_text(
        "GLOBAL_ONLY=global-value\nSHARED=from-global\n", encoding="utf-8"
    )
    (project / ".env").write_text(
        "SHARED=from-project\nPROJECT_ONLY=project-value\n", encoding="utf-8"
    )

    entries = env.load_project_env(project)

    assert entries == {
        "GLOBAL_ONLY": "global-value",
        "SHARED": "from-project",
        "PROJECT_ONLY": "project-value",
    }


def test_load_project_env_skips_absent_files(
    tmp_path: Path, chartreux_home: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("PROJECT_ONLY=project-value\n", encoding="utf-8")

    assert env.load_project_env(project) == {"PROJECT_ONLY": "project-value"}

    empty = tmp_path / "empty"
    empty.mkdir()
    assert env.load_project_env(empty) == {}


def test_project_env_for_cwd_requires_a_trusted_folder(
    tmp_path: Path, chartreux_home: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        "PROJECT_KEY=sk-project-0123456789\n", encoding="utf-8"
    )

    assert env.project_env_for_cwd(project, manager=StubTrustManager({})) == {}


def test_project_env_for_cwd_loads_the_trust_root_env_without_cascading(
    tmp_path: Path, chartreux_home: Path
) -> None:
    root = tmp_path / "root"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (root / ".env").write_text("ROOT_KEY=sk-root-0123456789\n", encoding="utf-8")
    (nested / ".env").write_text("NESTED_KEY=sk-nested-0123456789\n", encoding="utf-8")

    entries = env.project_env_for_cwd(nested, manager=StubTrustManager({nested: root}))

    # The trust root's .env loads; the nested folder's own .env does not (no
    # cascading lookups), and an absent global file contributes nothing.
    assert entries == {"ROOT_KEY": "sk-root-0123456789"}


def test_project_env_re_reads_on_mtime_change_so_rotation_takes_effect(
    tmp_path: Path, chartreux_home: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    env_file = project / ".env"
    env_file.write_text("ROTATING_KEY=sk-first-0123456789\n", encoding="utf-8")
    manager = StubTrustManager({project: project})

    assert env.project_env_for_cwd(project, manager=manager) == {
        "ROTATING_KEY": "sk-first-0123456789"
    }

    env_file.write_text("ROTATING_KEY=sk-second-0123456789\n", encoding="utf-8")
    os.utime(env_file, ns=(0, 2_000_000_000_000_000_000))
    assert env.project_env_for_cwd(project, manager=manager) == {
        "ROTATING_KEY": "sk-second-0123456789"
    }


def test_project_env_picks_up_a_file_created_after_the_first_read(
    tmp_path: Path, chartreux_home: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = StubTrustManager({project: project})

    assert env.project_env_for_cwd(project, manager=manager) == {}

    (project / ".env").write_text("LATE_KEY=sk-late-0123456789\n", encoding="utf-8")
    assert env.project_env_for_cwd(project, manager=manager) == {
        "LATE_KEY": "sk-late-0123456789"
    }


def test_dotenv_file_entries_uses_the_real_parser(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "export EXPORTED=exported-value\n"
        'QUOTED="double quoted value"\n'
        "SINGLE='single quoted value'\n"
        "EMPTY=\n"
        "NOVALUE\n"
        'MULTI="line one\nline two"\n',
        encoding="utf-8",
    )

    assert env.dotenv_file_entries(env_file) == {
        "EXPORTED": "exported-value",
        "QUOTED": "double quoted value",
        "SINGLE": "single quoted value",
        "MULTI": "line one\nline two",
    }


def test_dotenv_file_entries_is_fail_closed_for_absent_paths(tmp_path: Path) -> None:
    assert env.dotenv_file_entries(tmp_path / "absent.env") == {}
    directory = tmp_path / "directory.env"
    directory.mkdir()
    assert env.dotenv_file_entries(directory) == {}


def test_dotenv_file_entries_treats_a_fifo_as_absent(tmp_path: Path) -> None:
    # Only regular files load: a FIFO .env must not block env construction —
    # dotenv_values would hang on a blocking open with no writer.
    fifo = tmp_path / ".env"
    os.mkfifo(fifo)
    try:
        assert env.dotenv_file_entries(fifo) == {}
    finally:
        fifo.unlink()


def test_dotenv_file_entries_returns_a_copy_per_call(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KEY=sk-copy-0123456789\n", encoding="utf-8")

    entries = env.dotenv_file_entries(env_file)
    entries["KEY"] = "mutated"
    entries["INJECTED"] = "value"

    assert env.dotenv_file_entries(env_file) == {"KEY": "sk-copy-0123456789"}


def test_project_env_file_points_at_the_trust_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    manager = StubTrustManager({nested: root})

    assert env.project_env_file(nested, manager=manager) == root / ".env"
    assert env.project_env_file(tmp_path / "elsewhere", manager=manager) is None
