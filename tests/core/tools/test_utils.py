from __future__ import annotations

from pathlib import Path
import shlex

import pytest

from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    reset_harness_files_manager,
)
from chartreux.core.events import ToolCallEvent
from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.builtins.read_file import ReadFile, ReadFileArgs
from chartreux.core.tools.ui import ToolUIDataAdapter
from chartreux.core.tools.utils import (
    display_file_path,
    is_path_within_workdir,
    resolve_file_tool_permission,
    resolve_path_permission,
    resolve_tool_path,
)
from chartreux.core.workspace import Workspace


def test_absent_path_resolves_to_the_working_directory(tmp_path: Path) -> None:
    assert resolve_tool_path(None, tmp_path) == tmp_path


def test_empty_path_resolves_to_the_working_directory(tmp_path: Path) -> None:
    assert resolve_tool_path("", tmp_path) == tmp_path


def test_relative_path_resolves_against_the_working_directory(tmp_path: Path) -> None:
    assert resolve_tool_path("sub/dir", tmp_path) == tmp_path.resolve() / "sub" / "dir"


def test_absolute_path_is_kept(tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()

    assert resolve_tool_path(str(other), tmp_path / "unused") == other.resolve()


def test_foreign_windows_path_is_never_joined_onto_the_working_directory(
    tmp_path: Path,
) -> None:
    assert resolve_tool_path("C:/Users/acmedev/notes.md", tmp_path) == Path(
        "C:/Users/acmedev/notes.md"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "C:/Users/acmedev/notes.md",
        r"C:\Users\acmedev\notes.md",
        r"\\server\share\notes.md",
        "//server/share/notes.md",
    ],
)
def test_foreign_paths_are_denied_before_posix_canonicalization(
    raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = Workspace.for_session(tmp_path)
    assert not is_path_within_workdir(raw, workspace=workspace)
    resolved = resolve_tool_path(raw, tmp_path)
    assert not is_path_within_workdir(str(resolved), workspace=workspace)
    path_permission = resolve_path_permission(
        raw, cwd=tmp_path, allowlist=["*"], denylist=[]
    )
    assert path_permission is not None
    assert path_permission.permission is ToolPermission.NEVER
    file_permission = resolve_file_tool_permission(
        raw,
        tool_name="read_file",
        allowlist=["*"],
        denylist=[],
        config_permission=ToolPermission.ALWAYS,
        sensitive_patterns=[],
        workspace=workspace,
    )
    assert file_permission is not None
    assert file_permission.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "raw",
    [
        "C:/Users/acmedev/notes.md",
        r"C:\Users\acmedev\notes.md",
        r"\\server\share\notes.md",
        "//server/share/notes.md",
    ],
)
@pytest.mark.parametrize("command", ["cat {path}", "echo hello > {path}"])
def test_captured_bash_denies_foreign_operands_and_redirects(
    raw: str, command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    bash = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    permission = bash.resolve_permission(
        BashArgs(command=command.format(path=shlex.quote(raw)))
    )
    assert permission is not None
    assert permission.permission is ToolPermission.NEVER


@pytest.mark.parametrize("raw", ["sub/notes.md", "C:notes.md", "notes with spaces.md"])
def test_posix_relative_paths_remain_authorized(
    raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = Workspace.for_session(tmp_path)
    assert is_path_within_workdir(raw, workspace=workspace)
    assert resolve_tool_path(raw, tmp_path) == tmp_path / raw
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    bash = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    permission = bash.resolve_permission(BashArgs(command=f"cat {shlex.quote(raw)}"))
    assert permission is not None
    assert permission.permission is ToolPermission.ALWAYS


def test_display_file_path_uses_cwd_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "project"
    target = workdir / "pkg" / "config.py"
    workdir.mkdir()
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    monkeypatch.chdir(workdir)
    reset_harness_files_manager()

    assert display_file_path(str(target)) == "pkg/config.py"


def test_display_file_path_keeps_outside_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "project"
    outside = tmp_path / "outside" / "config.py"
    workdir.mkdir()
    outside.parent.mkdir()
    outside.write_text("x", encoding="utf-8")
    monkeypatch.chdir(workdir)
    reset_harness_files_manager()

    assert display_file_path(str(outside)) == str(outside.resolve())


def test_display_file_path_keeps_repo_sibling_absolute_when_outside_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    workdir = repo / "subdir"
    target = repo / "pkg" / "config.py"
    workdir.mkdir(parents=True)
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    monkeypatch.chdir(workdir)
    reset_harness_files_manager()

    assert display_file_path(str(target)) == str(target.resolve())


def test_display_file_path_resolves_relative_input_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "project"
    target = workdir / "pkg" / "config.py"
    workdir.mkdir()
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    monkeypatch.chdir(workdir)
    reset_harness_files_manager()

    assert display_file_path("pkg/config.py") == "pkg/config.py"


def test_tool_display_uses_session_harness_when_process_cwd_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    workdir = repo / "subdir"
    target = workdir / "pkg" / "config.py"
    home.mkdir()
    workdir.mkdir(parents=True)
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    monkeypatch.chdir(home)
    harness_files = HarnessFilesManager(sources=("user", "project")).for_session(
        workdir
    )
    event = ToolCallEvent(
        tool_call_id="test",
        tool_name="read_file",
        tool_class=ReadFile,
        args=ReadFileArgs(file_path=str(target)),
    )

    presentation = ToolUIDataAdapter(
        ReadFile, harness_files=harness_files
    ).get_call_presentation(event)

    assert presentation.display.summary == "Reading pkg/config.py"
    assert presentation.display.message == "pkg/config.py"
    assert presentation.display.settled_message == "pkg/config.py"


def test_tool_display_keeps_absolute_path_outside_session_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    workdir = tmp_path / "repo" / "subdir"
    target = tmp_path / "repo" / "pkg" / "config.py"
    home.mkdir()
    workdir.mkdir(parents=True)
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    monkeypatch.chdir(home)
    harness_files = HarnessFilesManager(sources=("user", "project")).for_session(
        workdir
    )
    event = ToolCallEvent(
        tool_call_id="test",
        tool_name="read_file",
        tool_class=ReadFile,
        args=ReadFileArgs(file_path=str(target)),
    )

    presentation = ToolUIDataAdapter(
        ReadFile, harness_files=harness_files
    ).get_call_presentation(event)

    assert presentation.display.summary == f"Reading {target.resolve()}"
    assert presentation.display.message == str(target.resolve())
    assert presentation.display.settled_message == str(target.resolve())
