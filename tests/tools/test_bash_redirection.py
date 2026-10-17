"""Redirect-target policy tests for the bash suite (protected git metadata redirect protection across explicit, identity and work-tree layouts, redirected-command syntax denial, authorized-root redirect syntax, safe and unsafe workspace-relative redirects, symlink escape, and nested redirect resolution after a directory change)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.workspace import Workspace
from tests.tools._bash_test_support import _w11_permission, bash as bash


@pytest.mark.parametrize(
    "command",
    [
        "git --git-dir=metadata log > metadata/config",
        "git --git-dir metadata --work-tree=. log > metadata/config",
        "git -C sub --git-dir=../metadata log > metadata/config",
        "git --git-dir=metadata log > shared/config",
        "git --git-dir=metadata -C sub log > sub/metadata/config",
        "cd sub; git --git-dir=../metadata log > ../metadata/config",
        "sh -c 'git --git-dir=metadata log > metadata/config'",
    ],
)
def test_redirection_protects_explicit_git_metadata(bash, tmp_path, command):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "shared").mkdir()
    (tmp_path / "sub").mkdir()
    (tmp_path / "metadata" / "commondir").write_text("../shared")
    context = bash.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.NEVER
    assert "protected Git metadata" in (context.reason or "")
    context = bash.resolve_permission(
        BashArgs(command="git --git-dir=metadata --work-tree=. diff > patch.txt")
    )
    assert context is not None and context.permission == ToolPermission.ALWAYS


@pytest.mark.parametrize("layout", ["symlink", "gitfile", "worktree"])
@pytest.mark.parametrize("target", ["metadata/config", "alias/config", ".git/config"])
@pytest.mark.parametrize("command", ["git log", "echo text", ""])
def test_redirection_protects_git_metadata_identity(
    bash, tmp_path, layout, target, command
):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (tmp_path / "alias").symlink_to(metadata, target_is_directory=True)
    if layout == "symlink":
        (tmp_path / ".git").symlink_to(metadata, target_is_directory=True)
    elif layout == "gitfile":
        (tmp_path / ".git").write_text("gitdir: metadata")
    else:
        worktree = tmp_path / "worktree-data"
        worktree.mkdir()
        (worktree / "commondir").write_text("../metadata")
        (tmp_path / ".git").write_text("gitdir: worktree-data")
    context = bash.resolve_permission(BashArgs(command=f"{command} > {target}"))
    assert context is not None and context.permission == ToolPermission.NEVER
    assert "protected Git metadata" in (context.reason or "")
    context = bash.resolve_permission(BashArgs(command="git diff > patch.txt"))
    assert context is not None and context.permission == ToolPermission.ALWAYS


def test_redirection_protects_work_tree_override_metadata(bash, tmp_path):
    worktree = tmp_path / "other"
    worktree.mkdir()
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (worktree / ".git").symlink_to(metadata, target_is_directory=True)
    context = bash.resolve_permission(
        BashArgs(command="git --work-tree=other log > metadata/config")
    )
    assert context is not None and context.permission == ToolPermission.NEVER
    assert "protected Git metadata" in (context.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "python3 << 'EOF'\nprint(42)\nEOF",
        "python3 - << 'EOF'\nprint(42)\nEOF",
        "python3 <<'PYEOF'\nimport sys\nprint('hello')\nPYEOF",
        "python3 < input.txt",
    ],
)
def test_bash_redirected_commands_are_denied_for_syntax(command):
    bash_tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    result = bash_tool.resolve_permission(BashArgs(command=command))
    assert isinstance(result, PermissionContext)
    assert result.permission is ToolPermission.NEVER
    assert "unsupported shell syntax" in (result.reason or "")


def test_authorized_root_does_not_authorize_redirect_syntax(tmp_path):
    root, accepted = tmp_path / "root", tmp_path / "accepted"
    root.mkdir()
    accepted.mkdir()
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=root)
    tool.workspace = Workspace.for_session(root, authorized_roots=[accepted])
    result = tool.resolve_permission(BashArgs(command=f"echo hi > {accepted}/out"))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert "redirection" in (result.reason or "")


@pytest.mark.parametrize(
    "command", ["git status 2>&1", "cmd > out.txt", "cmd >> out.txt", "cmd 2> err.txt"]
)
def test_safe_workspace_relative_redirects_are_allowed(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "cmd > /etc/passwd",
        "echo hi > -x/../../outside",
        "echo hi > 123",
        "'cd' sub && cmd > ../out",
        "command cd sub && cmd > ../out",
        "(cd sub; cmd > ../out)",
        "cd sub && cmd > ../out",
        "> out.txt",
        "cmd < input.txt",
        "cmd 1>&2",
        "cmd > $OUTPUT",
    ],
)
def test_unsafe_redirect_forms_are_denied(command, tmp_path):
    (tmp_path / "sub").mkdir()
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert result.reason


def test_relative_redirect_symlink_escape_is_denied(tmp_path):
    (tmp_path / "escape").symlink_to(tmp_path.parent / "outside.txt")
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command="cmd > escape"))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert "outside" in (result.reason or "")


@pytest.mark.parametrize(
    ("directory_command", "expected", "reason"),
    [
        # pushd .. leaves the workspace, so the nested redirect target
        # resolves against a directory outside it (occurrence-local: the
        # redirect belongs to printf, which runs after the pushd).
        ("pushd ..", ToolPermission.NEVER, "outside the authorized workspace"),
        # A named/option-bearing popd cannot be statically resolved: the
        # relative redirect target fails closed on the unknown cwd.
        ("popd +1", ToolPermission.NEVER, "not statically known"),
        # A plain popd on an empty stack is a no-op, so the nested redirect
        # target stays in the workspace and the chain is allowed.
        ("popd", ToolPermission.ALWAYS, None),
    ],
)
def test_nested_redirect_after_directory_change_resolution(
    directory_command: str, expected: ToolPermission, reason: str | None, tmp_path: Path
):
    # The order-blind cd+redirect veto is gone (ADR 0021): a nested redirect
    # after a directory change is governed by the tracked working directory
    # at the redirect's occurrence position.
    result = _w11_permission(
        f"bash -c '{directory_command}; printf hi > file'", tmp_path
    )
    assert result.permission is expected
    if reason is not None:
        assert reason in (result.reason or "")
