"""Shared test-support surface for the bash tool suite (W10, stage 19h).

The definitions here are the shared module-level surface of
``tests/tools/test_bash.py``, relocated AST-verbatim in the 19h phase-(i)
support-seam commit: the ``bash`` fixture (31 tests request it, spanning
three of the four concerns — the redirection, execution and permissions
clusters), the ``_w11_permission`` helper (41 users across the
permissions, redirection and option-tables clusters) and the
``locate_file`` fixture with its ``_LocateFileBody`` data constant (7
tests request it, spanning the execution and permissions clusters). The
remaining module-level surface is single-concern and stays in
``test_bash.py`` until the phase-(ii) concern moves: the dead
``_hide_standard_git_installs`` helper (zero callers in the file), the
``_FILE_CONTENT_READER_COMMANDS`` table (2 users, permissions), the
``_write_repository_config`` helper (12 users, permissions) and the
``TestDenylistWordBoundary`` class with its ``_make_bash`` factory method
(9 tests, permissions). No test bodies or marks live here; the test
population stays in ``test_bash.py`` until the phase-(ii) concern moves.
Importer: ``test_bash.py`` itself (the seam enumeration found zero
external importers of that file).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.permissions import PermissionContext


@pytest.fixture
def bash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig()
    return Bash(config_getter=lambda: config, state=BaseToolState())


def _w11_permission(command: str, tmp_path: Path) -> PermissionContext:
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert isinstance(result, PermissionContext)
    return result


_LocateFileBody = "alpha\nlit\nbeta\n" + "\n".join(f"line{i}" for i in range(4, 60))


@pytest.fixture
def locate_file(tmp_path) -> str:
    (tmp_path / "lines.txt").write_text(_LocateFileBody + "\n")
    return "lines.txt"
