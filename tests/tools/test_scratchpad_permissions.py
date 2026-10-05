from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.scratchpad import init_scratchpad
from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    _collect_outside_dirs,
)
from chartreux.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileConfig,
    ReadFileState,
)
from chartreux.core.tools.builtins.write_file import (
    WriteFile,
    WriteFileArgs,
    WriteFileConfig,
)
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from chartreux.core.tools.permissions import PermissionContext
from tests.conftest import build_test_vibe_config


@pytest.fixture
def scratchpad():
    path = init_scratchpad("test-session")
    assert path is not None
    return path


class TestFileToolScratchpadPermissions:
    def test_write_file_scratchpad_always_allowed(self, scratchpad):
        tool = WriteFile(
            config_getter=lambda: WriteFileConfig(),
            state=BaseToolState(),
            scratchpad_dir=scratchpad,
        )
        result = tool.resolve_permission(
            WriteFileArgs(file_path=str(scratchpad / "draft.py"), content="x")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_read_scratchpad_always_allowed(self, scratchpad):
        tool = ReadFile(
            config_getter=lambda: ReadFileConfig(),
            state=ReadFileState(),
            scratchpad_dir=scratchpad,
        )
        result = tool.resolve_permission(
            ReadFileArgs(file_path=str(scratchpad / "notes.txt"))
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_scratchpad_sensitive_file_denied(self, scratchpad):
        """Scratchpad never bypasses sensitive pattern checks."""
        tool = WriteFile(
            config_getter=lambda: WriteFileConfig(),
            state=BaseToolState(),
            scratchpad_dir=scratchpad,
        )
        result = tool.resolve_permission(
            WriteFileArgs(file_path=str(scratchpad / ".env"), content="SECRET=x")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_non_scratchpad_outside_dir_denied(self):
        tool = WriteFile(config_getter=lambda: WriteFileConfig(), state=BaseToolState())
        result = tool.resolve_permission(
            WriteFileArgs(file_path="/tmp/not-scratchpad/file.txt", content="x")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason


class TestBashScratchpadPermissions:
    def test_scratchpad_path_not_flagged_as_outside_dir(self, scratchpad):
        dirs = _collect_outside_dirs(
            [f"cat {scratchpad}/file.txt"], scratchpad_dir=scratchpad
        )
        assert len(dirs) == 0

    def test_non_scratchpad_outside_path_still_flagged(self):
        dirs = _collect_outside_dirs(["cat /etc/hosts"])
        assert len(dirs) >= 1

    def test_bash_scratchpad_mkdir_allowed_within_scratchpad(self, scratchpad):
        bash = Bash(
            config_getter=lambda: BashToolConfig(),
            state=BaseToolState(),
            scratchpad_dir=scratchpad,
        )
        result = bash.resolve_permission(BashArgs(command=f"mkdir {scratchpad}/subdir"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS


def scratch_chain(tmp_path: Path, **parent_kwargs):
    config = build_test_vibe_config()
    project = tmp_path / "project"
    project.mkdir()
    # Permission resolution must not create even the granting directory.
    scratch = tmp_path / "scratch"
    parent = ToolManager(
        lambda: config,
        cwd=project,
        scratchpad_dir=scratch,
        accepted_token_getter=lambda: 1,
        **parent_kwargs,
    )
    child = ToolManager(
        lambda: config,
        cwd=project,
        parent_authority_getter=lambda: parent,
        accepted_token_getter=lambda: 1,
    )
    grandchild = ToolManager(
        lambda: config,
        cwd=project,
        parent_authority_getter=lambda: child,
        accepted_token_getter=lambda: 1,
    )
    return parent, child, grandchild, scratch


@pytest.mark.parametrize(
    "name", ["read_file", "read_image", "write_file", "edit", "bash", "grep"]
)
def test_live_scratch_inheritance_and_replacement(tmp_path, monkeypatch, name):
    from chartreux.core.tools.builtins.read_image import ReadImage

    monkeypatch.setattr(
        ReadImage, "is_available", classmethod(lambda cls, config: True)
    )
    parent, child, grandchild, scratch = scratch_chain(tmp_path)
    tools = [manager.get(name) for manager in (parent, child, grandchild)]
    raw = {"content": "x", "old_string": "x", "new_string": "y"}

    def verdict(tool, root):
        path = root / "notes.txt"
        args = tool.validate_arguments({
            **raw,
            "file_path": str(path),
            "path": str(path),
            "pattern": "x",
            "command": f"cat {path}",
        })
        return tool.resolve_permission(args).permission

    assert all(verdict(tool, scratch) == ToolPermission.ALWAYS for tool in tools)
    assert child._scratchpad_dir is None and grandchild._scratchpad_dir is None
    assert not scratch.exists()
    old_token = grandchild._effective_authority_token()
    old_workspace = grandchild.workspace
    grandchild.get_tool_config(name)
    # Eviction must not leave an externally retained parent's tool with old grants.
    parent.reset_all()
    replacement = tmp_path / "replacement"
    parent.set_scratchpad_dir(replacement)
    assert grandchild._effective_authority_token() != old_token
    assert grandchild.workspace is not old_workspace
    assert all(verdict(tool, scratch) == ToolPermission.NEVER for tool in tools)
    assert all(verdict(tool, replacement) == ToolPermission.ALWAYS for tool in tools)
    assert grandchild.get(name) is tools[-1]
    assert not replacement.exists()


@pytest.mark.parametrize("failure", ["missing", "cycle", "retired"])
@pytest.mark.parametrize("name", ["read_file", "bash", "grep"])
def test_scratch_chain_fails_closed(tmp_path, failure, name):
    parent, child, grandchild, scratch = scratch_chain(tmp_path)
    tool = grandchild.get(name)
    if failure == "missing":

        def unavailable():
            raise RuntimeError("parent unavailable")

        child._parent_authority_getter = unavailable
    elif failure == "cycle":
        parent._parent_authority_getter = lambda: grandchild
    else:
        parent._retire_authority()
    with pytest.raises(NoSuchToolError, match="authority unavailable"):
        _ = grandchild.scratchpad_roots
    path = scratch / "notes.txt"
    args = tool.validate_arguments({
        "file_path": str(path),
        "path": str(path),
        "pattern": "x",
        "command": f"cat {path}",
    })
    result = tool.resolve_permission(args)
    assert result is not None and result.permission == ToolPermission.NEVER


@pytest.mark.parametrize("ancestor", [0, 1])
@pytest.mark.parametrize("ceiling", ["never", "denylist", "sensitive", "plan"])
def test_inherited_scratch_never_bypasses_ancestor_ceiling(tmp_path, ancestor, ceiling):
    parent, child, grandchild, scratch = scratch_chain(tmp_path)
    owner = (parent, child)[ancestor]
    config = build_test_vibe_config(
        tools={
            "write_file": {
                "permission": "never" if ceiling == "never" else "always",
                "denylist": ["*/notes.txt"] if ceiling == "denylist" else [],
                "sensitive_patterns": ["**/notes.txt"]
                if ceiling == "sensitive"
                else [],
            }
        }
    )
    owner._config_getter = lambda: config
    if ceiling == "plan":
        owner._inherited_plan_write_scopes = ((tmp_path / "plan.md", None),)
    result = grandchild.get("write_file").resolve_permission(
        WriteFileArgs(file_path=str(scratch / "notes.txt"), content="x")
    )
    assert result is not None
    assert result.permission == ToolPermission.NEVER


@pytest.mark.parametrize(
    "template",
    [
        "uv run cat {root}/notes.txt",
        "uv run python {root}/script.py",
        "uv run --directory {root} cat notes.txt",
        "npx --package fixture cat {root}/notes.txt",
        "pipx run --spec fixture cat {root}/notes.txt",
        "go run {root}/main.go",
        "cargo run --manifest-path {root}/Cargo.toml",
    ],
)
def test_executor_uses_live_inherited_path_authority(
    tmp_path: Path, template: str
) -> None:
    parent, _, grandchild, scratch = scratch_chain(tmp_path)
    tool = grandchild.get("bash")

    def verdict(root: Path) -> ToolPermission:
        result = tool.resolve_permission(BashArgs(command=template.format(root=root)))
        assert result is not None
        return result.permission

    assert verdict(scratch) is ToolPermission.ALWAYS
    replacement = tmp_path / "replacement"
    parent.set_scratchpad_dir(replacement)
    assert verdict(scratch) is ToolPermission.NEVER
    assert verdict(replacement) is ToolPermission.ALWAYS
    assert not scratch.exists() and not replacement.exists()


@pytest.mark.parametrize("template", ["uv run cat {path}", "uv run python {path}"])
def test_executor_inherited_scratch_symlink_escape_denied(
    tmp_path: Path, template: str
) -> None:
    _, _, grandchild, scratch = scratch_chain(tmp_path)
    scratch.mkdir()
    (scratch / "escape.py").symlink_to(tmp_path / "outside.py")
    result = grandchild.get("bash").resolve_permission(
        BashArgs(command=template.format(path=scratch / "escape.py"))
    )
    assert result is not None and result.permission is ToolPermission.NEVER


def test_scratch_grants_obey_creating_ancestors_plan_ceiling(tmp_path):
    parent, child, grandchild, scratch = scratch_chain(tmp_path)
    parent._plan_file_write_scope_getter = lambda: tmp_path / "plan.md"
    borrowed = tmp_path / "child-scratch"
    child.set_scratchpad_dir(borrowed)
    tool = grandchild.get("write_file")
    for root, expected in (
        (scratch, ToolPermission.ALWAYS),
        (borrowed, ToolPermission.NEVER),
    ):
        result = tool.resolve_permission(
            WriteFileArgs(file_path=str(root / "notes.txt"), content="x")
        )
        assert result is not None and result.permission == expected


def test_inherited_scratch_symlink_escape_and_root_retarget_denied(tmp_path):
    _, _, grandchild, scratch = scratch_chain(tmp_path)
    scratch.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (scratch / "escape").symlink_to(outside, target_is_directory=True)
    tool = grandchild.get("read_file")
    for path in (scratch / "escape" / "notes.txt",):
        result = tool.resolve_permission(ReadFileArgs(file_path=str(path)))
        assert result is not None and result.permission == ToolPermission.NEVER
    scratch.rename(tmp_path / "moved")
    scratch.symlink_to(outside, target_is_directory=True)
    result = tool.resolve_permission(ReadFileArgs(file_path=str(scratch / "notes.txt")))
    assert result is not None and result.permission == ToolPermission.NEVER
