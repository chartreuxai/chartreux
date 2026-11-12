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
@pytest.mark.parametrize("ceiling", ["never", "denylist", "sensitive"])
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


# --- Shell policy v0.2 (ADR 0021): bash redirect targets and scratch roots ---


def test_bash_scratchpad_redirect_allowed_and_follows_changed_roots(
    tmp_path: Path,
) -> None:
    (tmp_path / "allowed").mkdir()
    parent, _, grandchild, scratch = scratch_chain(tmp_path / "allowed")
    tool = grandchild.get("bash")

    def verdict(command: str) -> ToolPermission:
        result = tool.resolve_permission(BashArgs(command=command))
        assert result is not None, command
        return result.permission

    # An absolute redirect target inside the inherited scratchpad root is
    # allowed; /etc stays denied.
    assert verdict(f"echo hi > {scratch}/log.txt") is ToolPermission.ALWAYS
    assert verdict("echo hi > /etc/x") is ToolPermission.NEVER

    # Changed roots: the retired scratchpad is no longer authorized, the
    # replacement root is.
    replacement = tmp_path / "allowed" / "replacement"
    parent.set_scratchpad_dir(replacement)
    assert verdict(f"echo hi > {scratch}/log.txt") is ToolPermission.NEVER
    assert verdict(f"echo hi > {replacement}/log.txt") is ToolPermission.ALWAYS
    assert not scratch.exists() and not replacement.exists()


def test_bash_scratchpad_redirect_obeys_parent_tool_ceiling(tmp_path: Path) -> None:
    # A parent tool-level ceiling denies the same redirect target: the parent
    # invocation guard re-resolves the call under the parent's authority.
    (tmp_path / "ceiling").mkdir()
    parent, _, grandchild, scratch = scratch_chain(tmp_path / "ceiling")
    config = build_test_vibe_config(tools={"bash": {"permission": "never"}})
    parent._config_getter = lambda: config
    tool = grandchild.get("bash")
    result = tool.resolve_permission(BashArgs(command=f"echo hi > {scratch}/log.txt"))
    assert result is not None
    assert result.permission is ToolPermission.NEVER


# --- v0.7.8 P4: recursive rm confined to the session scratchpad ---


def scratch_rm_tool(workspace: Path, scratch: Path) -> Bash:
    return Bash(
        config_getter=lambda: BashToolConfig(),
        state=BaseToolState(),
        cwd=workspace,
        scratchpad_dir=scratch,
    )


def scratch_rm_verdict(tool: Bash, command: str) -> PermissionContext:
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    return result


class TestScratchpadRecursiveRmCarveOut:
    def test_contained_recursive_removal_resolves_run_class(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scratch = tmp_path / "scratch"
        (scratch / "work" / "nested").mkdir(parents=True)
        (scratch / "work" / "nested" / "draft.txt").write_text("draft\n")
        tool = scratch_rm_tool(workspace, scratch)
        for command in (
            f"rm -r {scratch}/work",
            f"rm -rf {scratch}/work/nested",
            f"rm -r {scratch}/work {scratch}/work/nested",
            f"rm -r -- {scratch}/work",
            # A literal operand that does not exist yet is still contained.
            f"rm -rf {scratch}/not-created-yet",
        ):
            result = scratch_rm_verdict(tool, command)
            assert result.permission is ToolPermission.ALWAYS, command

    def test_scratchpad_root_itself_is_not_a_deletable_operand(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        tool = scratch_rm_tool(workspace, scratch)
        for command in (f"rm -r {scratch}", f"rm -rf {scratch}/"):
            result = scratch_rm_verdict(tool, command)
            assert result.permission is ToolPermission.NEVER, command
            assert "hard guard" in (result.reason or ""), command

    def test_symlink_escaping_the_root_is_denied_on_the_resolved_path(
        self, tmp_path: Path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep\n")
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "escape").symlink_to(outside, target_is_directory=True)
        (scratch / "escape-file").symlink_to(outside / "keep.txt")
        tool = scratch_rm_tool(workspace, scratch)
        for command in (
            f"rm -r {scratch}/escape",
            f"rm -r {scratch}/escape/keep.txt",
            f"rm -r {scratch}/escape-file",
        ):
            result = scratch_rm_verdict(tool, command)
            assert result.permission is ToolPermission.NEVER, command
            assert "hard guard" in (result.reason or ""), command

    def test_mixed_and_outside_operands_stay_hard_guards(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "build").mkdir()
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        tool = scratch_rm_tool(workspace, scratch)
        for command in (
            f"rm -r {scratch} build",
            "rm -r build",
            "rm -r ..",
            f"rm -r {workspace}",
        ):
            result = scratch_rm_verdict(tool, command)
            assert result.permission is ToolPermission.NEVER, command
            assert "hard guard" in (result.reason or ""), command

    def test_unestablishable_call_footprint_stays_a_hard_guard(self, tmp_path: Path):
        # The scratchpad carve-out skips the git-tree analysis, never the
        # footprint analysis: a statement whose write surface cannot be
        # modeled keeps the recursive removal denied.
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scratch = tmp_path / "scratch"
        (scratch / "work").mkdir(parents=True)
        tool = scratch_rm_tool(workspace, scratch)
        result = scratch_rm_verdict(tool, f"rm -r {scratch}/work; eval ls")
        assert result.permission is ToolPermission.NEVER
        assert "hard guard" in (result.reason or "")

    def test_sensitive_operand_inside_the_scratchpad_stays_denied(self, tmp_path: Path):
        # The scratchpad carve-out is a guardrail admission; the path layer's
        # sensitive-file rule still gates every operand.
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / ".env").write_text("SESSION_TOKEN=not-a-credential\n")
        tool = scratch_rm_tool(workspace, scratch)
        result = scratch_rm_verdict(tool, f"rm -r {scratch}/.env")
        assert result.permission is ToolPermission.NEVER
        assert "Sensitive file access denied" in (result.reason or "")

    def test_without_a_scratchpad_root_the_hard_guard_stands(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        (workspace / "build").mkdir(parents=True)
        tool = Bash(
            config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=workspace
        )
        result = scratch_rm_verdict(tool, "rm -r build")
        assert result.permission is ToolPermission.NEVER
        assert "hard guard" in (result.reason or "")
