from __future__ import annotations

import os
from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins import bash as bash_module
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    _collect_outside_dirs,
)
from chartreux.core.tools.builtins.edit import Edit, EditArgs, EditConfig
from chartreux.core.tools.builtins.grep import (
    Grep,
    GrepArgs,
    GrepResult,
    GrepToolConfig,
)
from chartreux.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileConfig,
    ReadFileResult,
    ReadFileState,
)
from chartreux.core.tools.builtins.web_fetch import (
    WebFetch,
    WebFetchArgs,
    WebFetchConfig,
)
from chartreux.core.tools.builtins.write_file import (
    WriteFile,
    WriteFileArgs,
    WriteFileConfig,
)
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.utils import (
    DEFAULT_SENSITIVE_PATTERNS,
    PathAccess,
    matches_sensitive_pattern,
)
from tests.conftest import build_test_vibe_config


@pytest.mark.parametrize(
    "config_class",
    [
        BashToolConfig,
        EditConfig,
        GrepToolConfig,
        ReadFileConfig,
        WebFetchConfig,
        WriteFileConfig,
    ],
)
def test_removed_ask_permission_rejected_by_tool_config(config_class):
    with pytest.raises(ValidationError, match="'always' or 'never'"):
        config_class.model_validate({"permission": "ask"})


def test_permission_enum_has_no_ask_value():
    assert set(ToolPermission) == {ToolPermission.ALWAYS, ToolPermission.NEVER}
    with pytest.raises(ValueError):
        ToolPermission("ask")


class TestBashGranularPermissions:
    @pytest.mark.parametrize("allowlist", [[], ["*"], ["npm *"], ["/tmp/*"]])
    def test_shell_allowlists_rejected_at_tool_and_session_config(self, allowlist):
        with pytest.raises(ValidationError, match="allowlist was removed"):
            BashToolConfig(allowlist=allowlist)
        session_config = build_test_vibe_config(
            tools={"bash": {"allowlist": allowlist}}
        )
        with pytest.raises(ValidationError, match="allowlist was removed"):
            ToolManager(config_getter=lambda: session_config).get_tool_config("bash")
        before = session_config.model_dump()
        with pytest.raises(ValueError, match="allowlist was removed"):
            session_config.build_tool_allowlist_update("bash", allowlist)
        assert session_config.model_dump() == before
        config = BashToolConfig()
        assert config.allowlist == []
        assert "allowlist" not in config.model_dump()

    def test_historical_bypass_config_is_rejected(self):
        with pytest.raises(ValidationError, match="bypass_tool_permissions.*removed"):
            build_test_vibe_config(bypass_tool_permissions=True)

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.workdir = tmp_path

    def _bash(self, **kwargs):
        config = BashToolConfig(**kwargs)
        return Bash(config_getter=lambda: config, state=BaseToolState())

    def test_allowlisted_command_always(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="git status"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_denylisted_command_never(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="vim file.txt"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_standalone_denylisted_never(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="python"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_standalone_denylisted_with_args_executes_automatically(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="python script.py"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    @pytest.mark.parametrize(
        "command",
        [
            "python3 << 'EOF'\nprint(42)\nEOF",
            "python3 - << 'EOF'\nprint(42)\nEOF",
            "python3 <<'PYEOF'\nimport sys\nprint('hello')\nPYEOF",
            "python3 < input.txt",
        ],
    )
    def test_standalone_denylisted_with_redirect_is_denied(self, command):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command=command))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_unknown_command_executes_without_approval_requirements(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="npm install"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_arity_based_command_executes_without_approval_requirements(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="docker compose up -d"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_multiple_commands_execute_without_approval_requirements(self):
        bash = self._bash()
        result = bash.resolve_permission(
            BashArgs(command="npm install foo && npm install bar")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_cd_outside_path_is_denied_by_path_guard(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="cd /tmp"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    @pytest.mark.parametrize("command", ["mkdir /tmp/test", "cat /etc/passwd"])
    def test_outside_directory_is_denied_before_allowlists(self, command):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command=command))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_in_workdir_no_outside_directory(self):
        bash = self._bash()
        (self.workdir / "subdir").mkdir()
        result = bash.resolve_permission(BashArgs(command="mkdir subdir/child"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_the_boundary_comes_from_the_session_not_the_process(
        self, tmp_path, monkeypatch
    ):
        # Every other test here chdirs the process into the session directory,
        # so a tool reading the boundary off the process rather than off its own
        # workspace still looks right. Under the app server the two differ, and
        # then a file the session owns is judged foreign.
        session = tmp_path / "session"
        (session / "subdir").mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        config = BashToolConfig()
        bash = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=session)

        result = bash.resolve_permission(
            BashArgs(command=f"mkdir {session / 'subdir' / 'child'}")
        )

        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    @pytest.mark.parametrize("command", ["rm -rf /tmp/something"])
    def test_dangerous_commands_are_denied_without_approval_requirements(self, command):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command=command))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_rmdir_in_workdir_executes_automatically(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="rmdir foo"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    @pytest.mark.parametrize(
        "permission", [ToolPermission.ALWAYS, ToolPermission.NEVER]
    )
    def test_sensitive_commands_remain_denied_under_configured_permission(
        self, permission
    ):
        bash = self._bash(permission=permission)
        result = bash.resolve_permission(BashArgs(command="sudo ls"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_allowlisted_relative_traversal_is_denied(self):
        bash = self._bash()
        (self.workdir / "src").mkdir()
        result = bash.resolve_permission(
            BashArgs(command="cat src/../../../etc/passwd")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_allowlisted_in_workdir_subdir_executes_automatically(self):
        bash = self._bash()
        (self.workdir / "foo").mkdir()
        (self.workdir / "foo" / "bar.txt").touch()
        result = bash.resolve_permission(BashArgs(command="cat foo/bar.txt"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_allowlisted_in_workdir_executes_automatically(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="cat README.md"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_mixed_allowlisted_and_not_executes_without_approval_requirements(self):
        bash = self._bash()
        result = bash.resolve_permission(
            BashArgs(command="echo hello && npm install foo")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_empty_command_returns_automatic_execution_context(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command=""))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_chmod_outside_path_is_denied(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="chmod +x /tmp/script.sh"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason


class TestReadGranularPermissions:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.workdir = tmp_path

    def _read(self, **kwargs):
        config = ReadFileConfig(**kwargs)
        return ReadFile(config_getter=lambda: config, state=ReadFileState())

    def test_in_workdir_normal_file_executes_automatically(self):
        (self.workdir / "test.py").touch()
        tool = self._read()
        result = tool.resolve_permission(ReadFileArgs(file_path="test.py"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_outside_workdir_is_denied(self):
        tool = self._read()
        result = tool.resolve_permission(ReadFileArgs(file_path="/tmp/file.txt"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_sensitive_env_file_is_denied(self):
        (self.workdir / ".env").touch()
        tool = self._read()
        result = tool.resolve_permission(ReadFileArgs(file_path=".env"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason and "Sensitive" in result.reason

    def test_sensitive_env_local_file_is_denied(self):
        (self.workdir / ".env.local").touch()
        tool = self._read()
        result = tool.resolve_permission(ReadFileArgs(file_path=".env.local"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_sensitive_outside_is_denied_by_the_first_guard(self):
        tool = self._read()
        result = tool.resolve_permission(ReadFileArgs(file_path="/tmp/.env"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_denylisted_returns_never(self):
        tool = self._read(denylist=["*/secret*"])
        result = tool.resolve_permission(ReadFileArgs(file_path="secret.key"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_allowlisted_returns_always(self):
        tool = self._read(allowlist=["*/README*"])
        result = tool.resolve_permission(
            ReadFileArgs(file_path=str(self.workdir / "README.md"))
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_custom_sensitive_patterns(self):
        (self.workdir / "credentials.json").touch()
        tool = self._read(sensitive_patterns=["*/credentials*"])
        result = tool.resolve_permission(ReadFileArgs(file_path="credentials.json"))
        assert isinstance(result, PermissionContext)

    def test_sensitive_env_file_is_denied_without_a_reusable_scope(self):
        (self.workdir / ".env").touch()
        tool = self._read()
        result = tool.resolve_permission(ReadFileArgs(file_path=".env"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_sensitive_case_insensitive_and_variants_are_denied(self):
        tool = self._read()
        for name in [".ENV", ".Env", ".env~", ".envrc", ".env.LOCAL"]:
            result = tool.resolve_permission(ReadFileArgs(file_path=name))
            assert isinstance(result, PermissionContext), name
            assert result.permission is ToolPermission.NEVER, name
            assert result.reason, name


class TestWriteFileGranularPermissions:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.workdir = tmp_path

    def _write_file(self):
        config = WriteFileConfig()
        return WriteFile(config_getter=lambda: config, state=BaseToolState())

    def test_in_workdir_executes_automatically(self):
        tool = self._write_file()
        result = tool.resolve_permission(
            WriteFileArgs(file_path="test.py", content="x")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_outside_workdir_is_denied(self):
        tool = self._write_file()
        result = tool.resolve_permission(
            WriteFileArgs(file_path="/tmp/file.txt", content="x")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_sensitive_env_file_is_denied(self):
        (self.workdir / ".env").touch()
        tool = self._write_file()
        result = tool.resolve_permission(WriteFileArgs(file_path=".env", content="x"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason


class TestEditGranularPermissions:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def test_outside_workdir_is_denied(self):
        config = EditConfig()
        tool = Edit(config_getter=lambda: config, state=BaseToolState())
        result = tool.resolve_permission(
            EditArgs(file_path="/tmp/file.py", old_string="a", new_string="b")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason


class TestGrepGranularPermissions:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.workdir = tmp_path

    def _grep(self):
        config = GrepToolConfig()
        return Grep(config_getter=lambda: config, state=BaseToolState())

    def test_in_workdir_normal_path_executes_automatically(self):
        tool = self._grep()
        result = tool.resolve_permission(GrepArgs(pattern="foo", path="."))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_outside_workdir_is_denied(self):
        tool = self._grep()
        result = tool.resolve_permission(GrepArgs(pattern="foo", path="/tmp"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    def test_sensitive_env_directory_is_denied(self):
        (self.workdir / ".env").touch()
        tool = self._grep()
        result = tool.resolve_permission(GrepArgs(pattern="foo", path=".env"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason


class TestSensitivePatternMatching:
    @pytest.mark.parametrize(
        "path", ["/wd/.ENV", "/wd/.Env", "/wd/.env", "/wd/sub/.env"]
    )
    def test_case_insensitive_match(self, path):
        assert matches_sensitive_pattern(path, DEFAULT_SENSITIVE_PATTERNS)

    @pytest.mark.parametrize(
        "path",
        [
            "/wd/.env~",
            "/wd/.envrc",
            "/wd/.env.local",
            "/wd/.env.PRODUCTION",
            "/wd/.envrc.local",
            "/wd/.envrc~",
        ],
    )
    def test_variant_names_match(self, path):
        assert matches_sensitive_pattern(path, DEFAULT_SENSITIVE_PATTERNS)

    @pytest.mark.parametrize(
        "path", ["/wd/main.py", "/wd/environment.txt", "/wd/readme.md"]
    )
    def test_non_sensitive_not_matched(self, path):
        assert not matches_sensitive_pattern(path, DEFAULT_SENSITIVE_PATTERNS)


class TestGuardRequirements:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    @pytest.mark.parametrize(
        "command",
        [
            "python",
            "sudo printf fixture",
            "find . -exec printf fixture \\;",
            "cat ../outside.txt",
            "printf hi > ../outside.txt",
        ],
    )
    def test_guard_matrix_denies_accident_prone_commands(self, command):
        bash = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
        result = bash.resolve_permission(BashArgs(command=command))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason

    @pytest.mark.parametrize(
        "command", ["python script.py", "printf fixture", "cat fixture.txt"]
    )
    def test_guard_matrix_executes_safe_commands_automatically(self, command):
        bash = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
        result = bash.resolve_permission(BashArgs(command=command))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    @pytest.mark.parametrize("path", [".env", ".env.production"])
    def test_sensitive_file_guard_denies_without_reusable_scope(self, path):
        read = ReadFile(
            config_getter=lambda: ReadFileConfig(),
            state=ReadFileState(),
            cwd=Path.cwd(),
        )
        result = read.resolve_permission(ReadFileArgs(file_path=path))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert result.reason


class TestWebFetchPermissions:
    def _make_webfetch(self) -> WebFetch:
        return WebFetch(config_getter=lambda: WebFetchConfig(), state=BaseToolState())

    def test_default_executes_automatically(self):
        wf = self._make_webfetch()
        result = wf.resolve_permission(
            WebFetchArgs(url="https://docs.python.org/3/library")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_config_permission_always_honored(self):
        wf = WebFetch(
            config_getter=lambda: WebFetchConfig(permission=ToolPermission.ALWAYS),
            state=BaseToolState(),
        )
        result = wf.resolve_permission(WebFetchArgs(url="https://example.com"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_config_permission_never_honored(self):
        wf = WebFetch(
            config_getter=lambda: WebFetchConfig(permission=ToolPermission.NEVER),
            state=BaseToolState(),
        )
        result = wf.resolve_permission(WebFetchArgs(url="https://example.com"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER


class TestCollectOutsideDirs:
    """Tests for _collect_outside_dirs helper."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.workdir = tmp_path

    def test_relative_path_resolving_outside_workdir(self):
        dirs = _collect_outside_dirs(["cat ../../etc/passwd"])
        # The relative path resolves outside workdir, should collect parent dir
        assert len(dirs) >= 1

    def test_multiple_targets_in_one_command(self):
        dirs = _collect_outside_dirs(["cp /tmp/a /var/b"])
        assert len(dirs) == 2

    @pytest.mark.parametrize(
        "command",
        [
            "grep root /etc/passwd",
            "less /etc/passwd",
            "sha256sum /etc/passwd",
            "od -c /etc/passwd",
            "cut -d: -f1 /etc/passwd",
            "find /etc -name x",
        ],
    )
    def test_read_only_allowlisted_commands_collect_outside_paths(self, command):
        # Read-only commands are auto-allowed, so their outside paths must still
        # be collected — otherwise they read outside the workdir with no prompt.
        assert len(_collect_outside_dirs([command])) >= 1

    def test_read_only_command_in_workdir_not_collected(self):
        (self.workdir / "local.txt").touch()
        assert _collect_outside_dirs(["grep root ./local.txt"]) == set()

    def test_chmod_skips_plus_x_token(self):
        dirs = _collect_outside_dirs(["chmod +x /tmp/script.sh"])
        # +x should be skipped, only /tmp/script.sh should be considered
        assert len(dirs) >= 1
        # Verify no dir was created from the "+x" token
        for d in dirs:
            assert "+x" not in d

    def test_empty_command_list(self):
        assert _collect_outside_dirs([]) == set()

    def test_home_relative_path(self):
        home = os.path.expanduser("~")
        dirs = _collect_outside_dirs(["cat ~/some_file"])
        # ~/some_file resolves to home directory, which is likely outside workdir
        if home != str(self.workdir):
            assert len(dirs) >= 1

    def test_in_workdir_path_not_collected(self):
        (self.workdir / "local_file").touch()
        dirs = _collect_outside_dirs(["cat ./local_file"])
        assert len(dirs) == 0

    def test_traversal_path_without_dot_prefix(self):
        """Paths like src/../../../etc/passwd don't start with . but contain /."""
        (self.workdir / "src").mkdir()
        dirs = _collect_outside_dirs(["cat src/../../../etc/passwd"])
        assert len(dirs) >= 1

    def test_in_workdir_subdir_path_not_collected(self):
        """foo/bar inside workdir should not be flagged."""
        (self.workdir / "foo").mkdir()
        (self.workdir / "foo" / "bar").touch()
        dirs = _collect_outside_dirs(["cat foo/bar"])
        assert len(dirs) == 0

    def test_forward_slash_absolute_path_detected(self):
        """Forward-slash absolute paths must be detected regardless of os.sep.

        Detection keys on "/" (the POSIX-shell separator), so absolute paths
        are not silently skipped.
        """
        dirs = _collect_outside_dirs(["cat /c/Users/victim/secret.txt"])
        assert len(dirs) >= 1

    def test_posix_escaped_space_path_stays_single_token(self, monkeypatch):
        seen_paths: list[str] = []

        def resolve(_self, path: str, _access) -> PermissionContext:
            seen_paths.append(path)
            return PermissionContext(permission=ToolPermission.NEVER)

        monkeypatch.setattr(bash_module.PathAuthority, "resolve", resolve)

        dirs = _collect_outside_dirs([r"cat /outside/foo\ bar"])

        assert seen_paths == ["/outside/foo bar"]
        assert len(dirs) == 1


def _permission(context: PermissionContext | None) -> PermissionContext:
    assert context is not None
    return context


def _instruction_manager(cwd: Path, files=(), *, parent=None, tools=None, cached=True):
    config = build_test_vibe_config(tools=tools or {})
    manager = ToolManager(
        lambda: config,
        cwd=cwd,
        defer_mcp=True,
        accepted_token_getter=(lambda: "accepted") if cached else None,
        parent_authority_getter=(lambda: parent) if parent else None,
    )
    manager.set_instruction_read_files(frozenset(files))
    return manager


@pytest.mark.asyncio
async def test_instruction_exact_file_read_capability_main_and_child(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("instruction marker\n")
    parent = _instruction_manager(workspace, [instructions])
    child = _instruction_manager(workspace, parent=parent)
    roots = parent.workspace.authorized_roots
    for manager in (parent, child):
        tool = manager.get("read_file")
        assert isinstance(tool, ReadFile)
        args = ReadFileArgs(file_path=str(instructions))
        assert (
            _permission(tool.resolve_permission(args)).permission
            == ToolPermission.ALWAYS
        )
        results = [item async for item in tool.run(args)]
        result = results[-1]
        assert isinstance(result, ReadFileResult)
        assert "instruction marker" in result.content
        assert not manager.workspace.allows(instructions)
        assert instructions not in manager.workspace.authorized_roots
        assert instructions.parent not in manager.workspace.authorized_roots
    assert parent.workspace.authorized_roots == roots


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "denial", ["none", "configured", "ancestor", "sensitive", "outside", "symlink"]
)
def test_shared_path_denial_equivalence(tmp_path, cached, denial):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = (workspace if denial != "outside" else tmp_path) / "instructions.txt"
    target.write_text("instructions")
    if denial == "symlink":
        outside = tmp_path / "outside.txt"
        outside.write_text("outside")
        target.unlink()
        target.symlink_to(outside)
    if denial == "sensitive":
        target = workspace / ".env"
        # Pure lexical permission test: never create or read a sensitive file.
    rule = {name: {"denylist": [str(target)]} for name in ("read_file", "grep", "bash")}
    parent = _instruction_manager(
        workspace, tools=rule if denial == "ancestor" else {}, cached=cached
    )
    manager = _instruction_manager(
        workspace,
        tools=rule if denial == "configured" else {},
        parent=parent,
        cached=cached,
    )
    expected = ToolPermission.ALWAYS if denial == "none" else ToolPermission.NEVER
    for name, args in (
        ("read_file", {"file_path": str(target)}),
        ("grep", {"path": str(target), "pattern": "instructions"}),
        ("bash", {"command": f"cat {target}"}),
    ):
        tool = manager.get(name)
        assert (
            _permission(
                tool.resolve_permission(tool.validate_arguments(args))
            ).permission
            == expected
        )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "command,expected",
    [
        ("cat {path}", ToolPermission.ALWAYS),
        ("head {path}", ToolPermission.ALWAYS),
        ("grep instructions {path}", ToolPermission.ALWAYS),
        ("touch {path}", ToolPermission.NEVER),
        ("rm {path}", ToolPermission.NEVER),
        ("tee {path}", ToolPermission.NEVER),
        ("sort -o {path} input", ToolPermission.NEVER),
        ("uniq input {path}", ToolPermission.NEVER),
        ("sed -i s/a/b/ {path}", ToolPermission.NEVER),
        ("echo replacement > {path}", ToolPermission.NEVER),
        ("cat {path} > {path}", ToolPermission.NEVER),
        ("sh -c 'cat {path} > {path}'", ToolPermission.NEVER),
    ],
)
def test_bash_instruction_reads_never_authorize_writes(
    tmp_path, cached, command, expected
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("instructions")
    parent = _instruction_manager(workspace, cached=cached)
    child = _instruction_manager(
        workspace, [instructions], parent=parent, cached=cached
    )
    tool = child.get("bash")
    args = BashArgs(command=command.format(path=instructions))
    assert _permission(tool.resolve_permission(args)).permission == expected
    assert instructions.read_text() == "instructions"


@pytest.mark.parametrize("denial", ["configured", "ancestor", "never"])
def test_bash_instruction_grant_stays_below_denials(tmp_path, denial):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("instructions")
    rules = {"bash": {"denylist": [str(instructions)]}}
    parent = _instruction_manager(
        workspace, tools=rules if denial == "ancestor" else {}
    )
    child = _instruction_manager(
        workspace,
        [instructions],
        parent=parent,
        tools={"bash": {"permission": "never"}}
        if denial == "never"
        else rules
        if denial == "configured"
        else {},
    )
    decision = _permission(
        child.get("bash").resolve_permission(BashArgs(command=f"cat {instructions}"))
    )
    assert decision.permission == ToolPermission.NEVER


def test_standalone_bash_instruction_read_uses_local_shared_authority(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("instructions")
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=workspace)
    tool.instruction_read_files_getter = lambda: frozenset({instructions})
    assert (
        _permission(
            tool.resolve_permission(BashArgs(command=f"cat {instructions}"))
        ).permission
        == ToolPermission.ALWAYS
    )
    assert (
        _permission(
            tool.resolve_permission(BashArgs(command=f"touch {instructions}"))
        ).permission
        == ToolPermission.NEVER
    )
    assert (
        _collect_outside_dirs([f"cat {instructions}"], authority=tool.path_authority)
        == set()
    )


@pytest.mark.parametrize("tool_name", ["write_file", "edit", "read_image"])
def test_instruction_capability_is_read_only_and_tool_name_scoped(
    tmp_path, monkeypatch, tool_name
):
    from chartreux.core.tools.builtins.read_image import ReadImage

    monkeypatch.setattr(
        ReadImage, "is_available", classmethod(lambda cls, config=None: True)
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("instructions")
    manager = _instruction_manager(workspace, [instructions])
    tool = manager.get(tool_name)
    args = tool.validate_arguments({
        "file_path": str(instructions),
        "content": "replacement",
        "old_string": "instructions",
        "new_string": "replacement",
        "command": f"cat {instructions}",
    })
    decision = _permission(tool.resolve_permission(args))
    assert decision.permission == ToolPermission.NEVER
    if tool_name in {"write_file", "edit"}:
        assert decision.reason is not None
        assert decision.reason.startswith(
            "Injected instruction files are readable only"
        )
        tool.plan_file_write_scope_getter = lambda: workspace / "plan.md"
        assert "Plan mode" in (_permission(tool.resolve_permission(args)).reason or "")
        shared = _permission(
            tool.path_authority.resolve(str(instructions), PathAccess.WRITE)
        )
        assert shared.permission == ToolPermission.NEVER and "Plan mode" in (
            shared.reason or ""
        )


@pytest.mark.parametrize("tool_name", ["read_file", "grep"])
@pytest.mark.parametrize(
    "sibling", ["config.toml", "sessions.json", "AGENTS.local.md", "other/AGENTS.md"]
)
def test_instruction_manifest_never_grants_siblings(tmp_path, tool_name, sibling):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("instructions")
    target = tmp_path / sibling
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not injected")
    manager = _instruction_manager(workspace, [instruction])
    tool = manager.get(tool_name)
    args = tool.validate_arguments({
        "file_path": str(target),
        "path": str(target),
        "pattern": ".",
    })
    assert _permission(tool.resolve_permission(args)).permission == ToolPermission.NEVER
    if tool_name == "grep":
        assert (
            _permission(
                tool.resolve_permission(GrepArgs(path=str(tmp_path), pattern="."))
            ).permission
            == ToolPermission.NEVER
        )


@pytest.mark.parametrize("tool_name", ["read_file", "grep"])
@pytest.mark.parametrize("denial", ["permission", "denylist", "sensitive_patterns"])
@pytest.mark.parametrize("inherited", [False, True])
def test_instruction_capability_never_overrides_denials(
    tmp_path, tool_name, denial, inherited
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("instructions")
    rule = "never" if denial == "permission" else [str(instruction)]
    parent = (
        _instruction_manager(workspace, tools={tool_name: {denial: rule}})
        if inherited
        else None
    )
    manager = _instruction_manager(
        workspace,
        [instruction],
        parent=parent,
        tools={} if inherited else {tool_name: {denial: rule}},
    )
    tool = manager.get(tool_name)
    args = tool.validate_arguments({
        "file_path": str(instruction),
        "path": str(instruction),
        "pattern": ".",
    })
    assert _permission(tool.resolve_permission(args)).permission == ToolPermission.NEVER
    if tool_name == "grep":
        assert isinstance(tool, Grep)
        _, frozen, _ = tool._snapshot()
        assert frozen is not None and not frozen(instruction)


@pytest.mark.parametrize("tool_name", ["read_file", "grep"])
def test_instruction_symlink_identity_is_pinned(tmp_path, tool_name):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = tmp_path / "original.md"
    other = tmp_path / "other.md"
    original.write_text("injected")
    other.write_text("not injected")
    link = tmp_path / "AGENTS.md"
    link.symlink_to(original)
    manager = _instruction_manager(workspace, [link.resolve()])
    tool = manager.get(tool_name)
    args = tool.validate_arguments({
        "file_path": str(link),
        "path": str(link),
        "pattern": ".",
    })
    assert (
        _permission(tool.resolve_permission(args)).permission == ToolPermission.ALWAYS
    )
    link.unlink()
    link.symlink_to(other)
    assert _permission(tool.resolve_permission(args)).permission == ToolPermission.NEVER


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["rg", "grep"])
@pytest.mark.parametrize("cached", [False, True])
async def test_child_differing_instruction_manifest_grep(
    tmp_path, monkeypatch, backend, cached
):
    import shutil

    from chartreux.core.tools.builtins.grep import GrepBackend

    if not shutil.which(backend):
        pytest.skip(f"{backend} unavailable")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent_doc = tmp_path / "parent.md"
    child_doc = tmp_path / "child.md"
    parent_doc.write_text("parent marker\n")
    child_doc.write_text("child marker\n")
    parent = _instruction_manager(workspace, [parent_doc], cached=cached)
    child = _instruction_manager(workspace, [child_doc], parent=parent, cached=cached)
    tool = child.get("grep")
    assert isinstance(tool, Grep)
    monkeypatch.setattr(
        tool,
        "_detect_backend",
        lambda: GrepBackend.RIPGREP if backend == "rg" else GrepBackend.GNU_GREP,
    )
    for doc in (parent_doc, child_doc):
        args = GrepArgs(path=str(doc), pattern="marker")
        assert (
            _permission(tool.resolve_permission(args)).permission
            == ToolPermission.ALWAYS
        )
        _, frozen, _ = tool._snapshot()
        if cached:
            assert frozen is not None and frozen(doc)
        else:
            assert frozen is None
        results = [item async for item in tool.run(args)]
        result = results[-1]
        assert isinstance(result, GrepResult)
        assert result.match_count == 1
        assert "marker" in result.matches


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_instruction_grep_rejects_replaced_directory_before_discovery(
    tmp_path, monkeypatch, cached
):
    from chartreux.core.tools.base import ToolError
    from chartreux.core.tools.builtins import grep as grep_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    doc = tmp_path / "AGENTS.md"
    doc.write_text("injected")
    manager = _instruction_manager(workspace, [doc], cached=cached)
    tool = manager.get("grep")
    assert isinstance(tool, Grep)
    doc.unlink()
    doc.mkdir()
    (doc / "secret").write_text("not injected")
    monkeypatch.setattr(
        grep_module.os,
        "walk",
        lambda *a, **k: pytest.fail("directory discovery must not run"),
    )
    assert (
        _permission(
            tool.resolve_permission(GrepArgs(path=str(doc), pattern="."))
        ).permission
        == ToolPermission.NEVER
    )
    with pytest.raises(ToolError, match="Search path denied"):
        _ = [item async for item in tool.run(GrepArgs(path=str(doc), pattern="."))]


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("backend", ["rg", "grep"])
@pytest.mark.parametrize("revoke_owner", ["parent", "child"])
@pytest.mark.parametrize("boundary", ["collection", "spawn", "publication"])
async def test_instruction_manifest_revocation_across_grep_async_boundaries(
    tmp_path, monkeypatch, cached, backend, revoke_owner, boundary
):
    import shutil

    from chartreux.core.tools.base import ToolError
    from chartreux.core.tools.builtins.grep import GrepBackend

    if not shutil.which(backend):
        pytest.skip(f"{backend} unavailable")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    doc = tmp_path / "AGENTS.md"
    doc.write_text("injected marker\n")
    parent = _instruction_manager(
        workspace, [doc] if revoke_owner == "parent" else [], cached=cached
    )
    manager = _instruction_manager(
        workspace,
        [doc] if revoke_owner == "child" else [],
        parent=parent,
        cached=cached,
    )
    owner = parent if revoke_owner == "parent" else manager
    tool = manager.get("grep")
    assert isinstance(tool, Grep)
    monkeypatch.setattr(
        tool,
        "_detect_backend",
        lambda: GrepBackend.RIPGREP if backend == "rg" else GrepBackend.GNU_GREP,
    )
    original_stage = tool._stage
    original_collect = tool._collect_live_paths
    revoked = False

    async def stage(function, *args, **kwargs):
        nonlocal revoked
        result = await original_stage(function, *args, **kwargs)
        name = function.__name__
        if not revoked and (
            boundary == "collection"
            and name in {"_collect_paths", "_collect_live_paths"}
            or boundary == "publication"
            and name in {"_parse_frozen", "_finish_output"}
            or boundary == "spawn"
            and name == "_prepare_batch"
        ):
            owner.set_instruction_read_files(frozenset())
            revoked = True
        return result

    async def collect(*args, **kwargs):
        nonlocal revoked
        paths = await original_collect(*args, **kwargs)
        if boundary == "collection" and not revoked:
            owner.set_instruction_read_files(frozenset())
            revoked = True
        return paths

    monkeypatch.setattr(tool, "_stage", stage)
    monkeypatch.setattr(tool, "_collect_live_paths", collect)
    with pytest.raises(ToolError, match="denied|discarded"):
        _ = [item async for item in tool.run(GrepArgs(path=str(doc), pattern="marker"))]
    assert revoked
