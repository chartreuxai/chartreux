from __future__ import annotations

import ctypes
import os
from pathlib import Path
import struct
import sys

import pytest

from chartreux.core.tools.base import BaseToolState, ToolError, ToolPermission
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.builtins.grep import (
    Grep,
    GrepArgs,
    GrepBackend,
    GrepToolConfig,
)
from chartreux.core.workspace import Workspace
from tests.mock.utils import collect_result


@pytest.mark.parametrize("command", ["cat escape", "printf fixture > escape"])
@pytest.mark.parametrize("in_subdir", [False, True])
def test_bash_bare_symlink_is_denied(tmp_path: Path, command: str, in_subdir: bool):
    project = tmp_path / "project"
    project.mkdir()
    subdir = project / "sub"
    subdir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("fixture\n")
    cwd = subdir if in_subdir else project
    (cwd / "escape").symlink_to(outside)
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=cwd)
    tool.workspace = Workspace(cwd, (project,))
    permission = tool.resolve_permission(BashArgs(command=command))
    assert permission is not None and permission.permission == ToolPermission.NEVER


def test_bash_multiword_sensitive_prefix(tmp_path: Path):
    config = BashToolConfig(
        permission=ToolPermission.ALWAYS, sensitive_patterns=["deploy production"]
    )
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    for command in [
        "deploy production",
        "deploy production now",
        "deploy productionish",
    ]:
        permission = tool.resolve_permission(BashArgs(command=command))
        expected = (
            ToolPermission.ALWAYS if command.endswith("ish") else ToolPermission.NEVER
        )
        assert permission is not None and permission.permission == expected


@pytest.mark.parametrize("authority", ["related", "scratch", "ceiling"])
def test_bash_bare_symlink_obeys_accepted_authority(tmp_path: Path, authority: str):
    project = tmp_path / "project"
    project.mkdir()
    related = tmp_path / "related"
    related.mkdir()
    target = related / "fixture.txt"
    target.write_text("fixture\n")
    (project / "escape").symlink_to(target)
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=project)
    if authority == "scratch":
        tool.scratchpad_dir = related
    else:
        tool.workspace = Workspace(
            project,
            (project, related),
            ceiling=Workspace.for_session(project) if authority == "ceiling" else None,
        )
    permission = tool.resolve_permission(BashArgs(command="cat escape"))
    expected = ToolPermission.NEVER if authority == "ceiling" else ToolPermission.ALWAYS
    assert permission is not None and permission.permission == expected


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Linux native-open instrumentation")
@pytest.mark.parametrize("use_default_ignore", [True, False])
@pytest.mark.parametrize("backend", list(GrepBackend))
@pytest.mark.parametrize("restriction", ["deny", "sensitive", "outside", "ceiling"])
async def test_recursive_grep_never_opens_blocked_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: GrepBackend,
    restriction: str,
    use_default_ignore: bool,
):
    project = tmp_path / "project"
    project.mkdir()
    allowed = project / "allowed.txt"
    allowed.write_text("needle allowed fixture\n")
    blocked = (
        tmp_path if restriction in {"outside", "ceiling"} else project
    ) / "blocked.txt"
    blocked.write_text("needle blocked fixture\n")
    if restriction in {"outside", "ceiling"}:
        (project / "escape").symlink_to(blocked)
    config = GrepToolConfig(
        allowlist=["*"],
        denylist=[str(blocked)] if restriction == "deny" else [],
        sensitive_patterns=["**/blocked.txt"] if restriction == "sensitive" else [],
    )
    workspace = Workspace.for_session(project)
    if restriction == "ceiling":
        workspace = Workspace(project, (project, tmp_path), ceiling=workspace)
    tool = Grep(config_getter=lambda: config, state=BaseToolState(), cwd=project)
    tool.workspace = workspace
    monkeypatch.setattr(tool, "_detect_backend", lambda: backend)
    permission = tool.resolve_permission(GrepArgs(pattern="needle"))
    assert permission is not None and permission.permission == ToolPermission.ALWAYS

    execute = tool._execute_search
    content_commands: list[list[str]] = []

    async def checked_command(cmd: list[str]) -> str:
        if "--files" not in cmd:
            operands = cmd[cmd.index("--") + 1 :]
            assert operands and "-r" not in cmd
            assert all(Path(path).is_file() for path in operands)
            assert all(Path(path).resolve() != blocked.resolve() for path in operands)
            content_commands.append(cmd)
        return await execute(cmd)

    monkeypatch.setattr(tool, "_execute_search", checked_command)
    # Observe successful kernel opens by native child processes (IN_OPEN is not
    # an attempted-syscall trace). Argument checks above independently verify
    # that forbidden candidates never reach either content-search backend.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    assert fd >= 0
    try:
        opened = 0x20  # IN_OPEN
        blocked_watch = libc.inotify_add_watch(fd, os.fsencode(blocked), opened)
        allowed_watch = libc.inotify_add_watch(fd, os.fsencode(allowed), opened)
        assert blocked_watch >= 0 and allowed_watch >= 0
        result = await collect_result(
            tool.run(GrepArgs(pattern="needle", use_default_ignore=use_default_ignore))
        )
        events = os.read(fd, 65536)
        watches: set[int] = set()
        offset = 0
        while offset < len(events):
            watch, _mask, _cookie, length = struct.unpack_from("iIII", events, offset)
            watches.add(watch)
            offset += 16 + length
        assert content_commands
        assert allowed_watch in watches, (
            "positive control must observe the allowed read"
        )
        assert blocked_watch not in watches, (
            "blocked candidate was opened before filtering"
        )
        assert "allowed fixture" in result.matches
        assert "blocked fixture" not in result.matches
    finally:
        os.close(fd)


def test_grep_defensive_filter_applies_denies_and_roots(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    blocked = project / "blocked.txt"
    config = GrepToolConfig(
        allowlist=["*"], denylist=[str(blocked)], sensitive_patterns=[]
    )
    tool = Grep(config_getter=lambda: config, state=BaseToolState(), cwd=project)
    result = tool._parse_output(
        f"{blocked}:1:blocked\n{tmp_path / 'outside.txt'}:1:outside\nallowed.txt:1:allowed",
        100,
    )
    assert result.matches == "allowed.txt:1:allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_default_ignore", [True, False])
@pytest.mark.parametrize("control", [".ignore", ".gitignore", ".git/info/exclude"])
async def test_rg_preserves_authorized_parent_ignores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_default_ignore: bool,
    control: str,
):
    (tmp_path / ".git/info").mkdir(parents=True)
    (tmp_path / control).write_text("ignored.txt\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "ignored.txt").write_text("needle ignored\n")
    (sub / "allowed.txt").write_text("needle allowed\n")
    config = GrepToolConfig()
    tool = Grep(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    monkeypatch.setattr(tool, "_detect_backend", lambda: GrepBackend.RIPGREP)
    result = await collect_result(
        tool.run(
            GrepArgs(
                pattern="needle", path="sub", use_default_ignore=use_default_ignore
            )
        )
    )
    assert "needle allowed" in result.matches
    assert ("needle ignored" in result.matches) is not use_default_ignore


@pytest.mark.asyncio
async def test_rg_denied_subtree_does_not_hide_same_named_allowed_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    denied = tmp_path / "one/shared"
    allowed = tmp_path / "two/shared"
    denied.mkdir(parents=True)
    allowed.mkdir(parents=True)
    (denied / ".ignore").write_text("*\n")
    (denied / "fixture.txt").write_text("needle denied\n")
    (allowed / "fixture.txt").write_text("needle allowed\n")
    config = GrepToolConfig(denylist=[str(denied), str(denied / "*")])
    tool = Grep(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    monkeypatch.setattr(tool, "_detect_backend", lambda: GrepBackend.RIPGREP)
    result = await collect_result(tool.run(GrepArgs(pattern="needle")))
    assert "needle allowed" in result.matches
    assert "needle denied" not in result.matches


@pytest.mark.asyncio
async def test_rg_does_not_resurrect_unlisted_symlink_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "allowed.txt"
    target.write_text("needle allowed\n")
    (tmp_path / "alias.txt").symlink_to(target)
    config = GrepToolConfig()
    tool = Grep(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    monkeypatch.setattr(tool, "_detect_backend", lambda: GrepBackend.RIPGREP)
    result = await collect_result(tool.run(GrepArgs(pattern="needle")))
    assert result.match_count == 1
    assert "alias.txt" not in result.matches


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", list(GrepBackend))
async def test_denied_search_root_never_starts_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: GrepBackend
):
    config = GrepToolConfig(denylist=[str(tmp_path)])
    tool = Grep(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    monkeypatch.setattr(tool, "_detect_backend", lambda: backend)

    async def forbidden_command(cmd: list[str]) -> str:
        pytest.fail(f"Denied root reached backend: {cmd}")

    monkeypatch.setattr(tool, "_execute_search", forbidden_command)
    with pytest.raises(ToolError, match="Search path denied"):
        await collect_result(tool.run(GrepArgs(pattern="needle")))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", list(GrepBackend))
@pytest.mark.parametrize("ignore_name", (".chartreuxignore", ".vibeignore"))
async def test_denied_codeignore_is_checked_before_read_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: GrepBackend,
    ignore_name: str,
):
    control = tmp_path / ignore_name
    control.write_text("*.txt\n")
    (tmp_path / "allowed.txt").write_text("needle allowed\n")
    config = GrepToolConfig(denylist=[str(control)])
    tool = Grep(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    monkeypatch.setattr(tool, "_detect_backend", lambda: backend)

    def forbidden_read(path: Path):
        pytest.fail(f"Denied ignore control reached read helper: {path}")

    monkeypatch.setattr("chartreux.core.tools.builtins.grep.read_safe", forbidden_read)
    result = await collect_result(tool.run(GrepArgs(pattern="needle")))
    assert "needle allowed" in result.matches
