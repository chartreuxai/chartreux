from __future__ import annotations

import os
from pathlib import Path

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
from chartreux.core.tools.builtins.grep import Grep, GrepArgs, GrepToolConfig
from chartreux.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileConfig,
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
from chartreux.core.tools.permissions import PermissionContext, PermissionScope
from chartreux.core.tools.utils import (
    DEFAULT_SENSITIVE_PATTERNS,
    matches_sensitive_pattern,
)


class TestBashGranularPermissions:
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
        assert not result.required_permissions

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
        assert not result.required_permissions

    def test_arity_based_command_executes_without_approval_requirements(self):
        bash = self._bash()
        result = bash.resolve_permission(BashArgs(command="docker compose up -d"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS
        assert not result.required_permissions

    def test_multiple_commands_execute_without_approval_requirements(self):
        bash = self._bash()
        result = bash.resolve_permission(
            BashArgs(command="npm install foo && npm install bar")
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS
        assert not result.required_permissions

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

    @pytest.mark.parametrize("permission", [ToolPermission.ASK, ToolPermission.ALWAYS])
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
        assert not result.required_permissions

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
        assert not result.required_permissions

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
        assert not result.required_permissions

    @pytest.mark.parametrize(
        "command", ["python script.py", "printf fixture", "cat fixture.txt"]
    )
    def test_guard_matrix_executes_safe_commands_automatically(self, command):
        bash = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
        result = bash.resolve_permission(BashArgs(command=command))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS
        assert not result.required_permissions

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
        assert not result.required_permissions


class TestWebFetchPermissions:
    def _make_webfetch(self) -> WebFetch:
        return WebFetch(config_getter=lambda: WebFetchConfig(), state=BaseToolState())

    def test_returns_url_pattern_with_domain(self):
        wf = self._make_webfetch()
        result = wf.resolve_permission(
            WebFetchArgs(url="https://docs.python.org/3/library")
        )
        assert isinstance(result, PermissionContext)
        assert len(result.required_permissions) == 1
        rp = result.required_permissions[0]
        assert rp.scope is PermissionScope.URL_PATTERN
        assert rp.invocation_pattern == "docs.python.org"
        assert rp.session_pattern == "docs.python.org"
        assert "docs.python.org" in rp.label

    def test_http_url(self):
        wf = self._make_webfetch()
        result = wf.resolve_permission(WebFetchArgs(url="http://example.com/page"))
        assert isinstance(result, PermissionContext)
        rp = result.required_permissions[0]
        assert rp.invocation_pattern == "example.com"

    def test_url_without_scheme(self):
        wf = self._make_webfetch()
        result = wf.resolve_permission(WebFetchArgs(url="github.com/anthropics"))
        assert isinstance(result, PermissionContext)
        rp = result.required_permissions[0]
        assert rp.invocation_pattern == "github.com"

    def test_url_with_port(self):
        wf = self._make_webfetch()
        result = wf.resolve_permission(WebFetchArgs(url="http://localhost:8080/api"))
        assert isinstance(result, PermissionContext)
        rp = result.required_permissions[0]
        assert rp.invocation_pattern == "localhost:8080"

    def test_url_without_scheme_with_port(self):
        wf = self._make_webfetch()
        result = wf.resolve_permission(WebFetchArgs(url="example.com:3000/path"))
        assert isinstance(result, PermissionContext)
        rp = result.required_permissions[0]
        assert rp.invocation_pattern == "example.com:3000"

    def test_different_domains_require_distinct_url_patterns(self):
        wf = self._make_webfetch()
        docs = wf.resolve_permission(
            WebFetchArgs(url="https://docs.python.org/3/library")
        )
        evil = wf.resolve_permission(WebFetchArgs(url="https://evil.com"))
        assert isinstance(docs, PermissionContext)
        assert isinstance(evil, PermissionContext)
        assert docs.required_permissions[0].session_pattern == "docs.python.org"
        assert evil.required_permissions[0].session_pattern == "evil.com"
        assert docs.required_permissions[0].session_pattern != (
            evil.required_permissions[0].session_pattern
        )

    def test_same_domain_reports_the_same_url_pattern(self):
        wf = self._make_webfetch()
        first = wf.resolve_permission(WebFetchArgs(url="https://docs.python.org/one"))
        second = wf.resolve_permission(WebFetchArgs(url="https://docs.python.org/two"))
        assert isinstance(first, PermissionContext)
        assert isinstance(second, PermissionContext)
        assert first.required_permissions[0].session_pattern == "docs.python.org"
        assert second.required_permissions[0].session_pattern == "docs.python.org"

    def test_double_slash_url(self):
        wf = self._make_webfetch()
        result = wf.resolve_permission(WebFetchArgs(url="//cdn.example.com/lib.js"))
        assert isinstance(result, PermissionContext)
        rp = result.required_permissions[0]
        assert rp.invocation_pattern == "cdn.example.com"

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

    def test_config_permission_ask_falls_through_to_domain(self):
        wf = WebFetch(
            config_getter=lambda: WebFetchConfig(permission=ToolPermission.ASK),
            state=BaseToolState(),
        )
        result = wf.resolve_permission(WebFetchArgs(url="https://example.com"))
        assert isinstance(result, PermissionContext)
        assert result.required_permissions[0].invocation_pattern == "example.com"


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

        def is_within_workdir(path: str, **_kwargs) -> bool:
            seen_paths.append(path)
            return False

        monkeypatch.setattr(bash_module, "is_path_within_workdir", is_within_workdir)

        dirs = _collect_outside_dirs([r"cat /outside/foo\ bar"])

        assert seen_paths == ["/outside/foo bar"]
        assert len(dirs) == 1
