from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.tools.base import BaseToolState, ToolError, ToolPermission
import chartreux.core.tools.builtins.bash as bash_module
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    CapturedShellResult,
)
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.workspace import Workspace
from tests.mock.utils import collect_result


def test_removed_allowlist_is_rejected() -> None:
    with pytest.raises(ValidationError, match="tools.bash.*allowlist.*removed"):
        BashToolConfig.model_validate({"allowlist": ["echo"]})


@pytest.fixture
def bash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig()
    return Bash(config_getter=lambda: config, state=BaseToolState())


def _hide_standard_git_installs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)


def test_shell_results_keep_a_returncode_alias_for_post_tool_hooks():
    # The verbatim result dump is the `tool_output` payload handed to hooks.
    captured = CapturedShellResult(command="x", exit_code=2).model_dump(mode="json")
    assert captured["exit_code"] == 2
    assert captured["returncode"] == 2


@pytest.mark.asyncio
async def test_runs_echo_successfully(bash):
    result = await collect_result(bash.run(BashArgs(command="echo hello")))

    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_fails_cat_command_with_missing_file(bash):
    with pytest.raises(ToolError) as err:
        await collect_result(bash.run(BashArgs(command="cat missing_file.txt")))

    message = str(err.value)
    assert "Command failed" in message
    assert "Return code: 1" in message
    assert "No such file or directory" in message


@pytest.mark.asyncio
async def test_uses_effective_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig()
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    result = await collect_result(bash_tool.run(BashArgs(command="pwd")))

    assert result.stdout.strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_handles_timeout(bash):
    with pytest.raises(ToolError) as err:
        await collect_result(bash.run(BashArgs(command="sleep 2", timeout=1)))

    assert "Command timed out after 1s" in str(err.value)


@pytest.mark.asyncio
async def test_truncates_output_to_max_bytes(bash):
    config = BashToolConfig(max_output_bytes=5)
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    result = await collect_result(
        bash_tool.run(BashArgs(command="printf 'abcdefghij'"))
    )

    assert result.stdout == "abcde"
    assert result.stderr == ""
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_cat_preserves_accents_from_latin1_encoded_file(bash, tmp_path):
    file = tmp_path / "menu.txt"
    file.write_bytes("café au lait\nthé glacé\n".encode("latin-1"))

    result = await collect_result(bash.run(BashArgs(command=f"cat {file.name}")))

    assert result.exit_code == 0
    assert "\ufffd" not in result.stdout
    assert result.stdout == "café au lait\nthé glacé\n"


@pytest.mark.parametrize("predicate", ["-exec", "-execdir", "-ok", "-okdir"])
def test_find_execution_predicates_force_never(predicate: str):
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(
        BashArgs(command=f"find . {predicate} id \\;")
    )

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "find execution" in (permission.reason or "")
    assert not permission.required_permissions


def test_find_exec_compound_is_denied():
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(
        BashArgs(command='find . -exec id \\; && python3 -c "import os"')
    )

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "find execution" in (permission.reason or "")
    assert not permission.required_permissions


def test_find_execution_predicate_does_not_override_denylist():
    config = BashToolConfig(denylist=["passwd"])
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(
        BashArgs(command="find . -exec id \\; && passwd root")
    )

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "matches denylist pattern 'passwd'" in (permission.reason or "")


def test_legacy_bash_quoted_outside_path_is_denied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    bash_tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ASK),
        state=BaseToolState(),
    )

    permission = bash_tool.resolve_permission(BashArgs(command=f'cat "{outside}"'))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "outside" in (permission.reason or "")
    assert not permission.required_permissions


@pytest.mark.parametrize("command", ["grep root", "find", "od -c"])
def test_bash_readers_deny_outside_paths(command, tmp_path, monkeypatch):
    workdir = tmp_path / "workdir"
    outside = tmp_path / "outside"
    workdir.mkdir()
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    monkeypatch.chdir(workdir)
    bash_tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())

    permission = bash_tool.resolve_permission(
        BashArgs(command=f"{command} {outside_file}")
    )

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "outside" in (permission.reason or "")
    assert not permission.required_permissions


def test_resolve_permission():
    config = BashToolConfig(denylist=["rm"])
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    allowed = bash_tool.resolve_permission(BashArgs(command="echo hi"))
    denylisted = bash_tool.resolve_permission(BashArgs(command="rm -rf /tmp"))
    mixed = bash_tool.resolve_permission(BashArgs(command="pwd && whoami"))
    empty = bash_tool.resolve_permission(BashArgs(command=""))

    assert isinstance(allowed, PermissionContext)
    assert allowed.permission is ToolPermission.ALWAYS
    assert isinstance(denylisted, PermissionContext)
    assert denylisted.permission is ToolPermission.NEVER
    assert isinstance(mixed, PermissionContext)
    assert mixed.permission is ToolPermission.ALWAYS
    assert not mixed.required_permissions
    assert isinstance(empty, PermissionContext)
    assert empty.permission is ToolPermission.ALWAYS


class TestDenylistWordBoundary:
    """Verify denylist matches whole command names, not prefixes."""

    def _make_bash(self, **kwargs) -> Bash:
        config = BashToolConfig(**kwargs)
        return Bash(config_getter=lambda: config, state=BaseToolState())

    def test_vi_blocks_vi_exact(self):
        bash_tool = self._make_bash(denylist=["vi"])
        result = bash_tool.resolve_permission(BashArgs(command="vi"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_vi_blocks_vi_with_args(self):
        bash_tool = self._make_bash(denylist=["vi"])
        result = bash_tool.resolve_permission(BashArgs(command="vi file.txt"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_vi_does_not_block_vibe(self):
        bash_tool = self._make_bash(denylist=["vi"])
        result = bash_tool.resolve_permission(BashArgs(command="vibe -p hello"))
        assert result is None or result.permission is not ToolPermission.NEVER

    def test_multiword_pattern_still_works(self):
        bash_tool = self._make_bash(denylist=["bash -i"])
        result = bash_tool.resolve_permission(BashArgs(command="bash -i"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_multiword_pattern_with_trailing_args(self):
        bash_tool = self._make_bash(denylist=["bash -i"])
        result = bash_tool.resolve_permission(BashArgs(command="bash -i extra"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_multiword_pattern_does_not_match_partial(self):
        bash_tool = self._make_bash(denylist=["bash -i"])
        result = bash_tool.resolve_permission(BashArgs(command="bash -init"))
        assert result is None or result.permission is not ToolPermission.NEVER

    def test_deny_reason_is_set(self):
        bash_tool = self._make_bash(denylist=["vim"])
        result = bash_tool.resolve_permission(BashArgs(command="vim file.txt"))
        assert isinstance(result, PermissionContext)
        assert result.reason is not None
        assert "vim" in result.reason

    def test_standalone_deny_reason_is_set(self):
        bash_tool = self._make_bash(denylist_standalone=["python"])
        result = bash_tool.resolve_permission(BashArgs(command="python"))
        assert isinstance(result, PermissionContext)
        assert result.reason is not None
        assert result.permission is ToolPermission.NEVER
        assert "python" in result.reason
        assert "standalone" in result.reason


def test_new_read_only_commands_are_permitted_by_hard_guards():
    """Test that newly added read-only commands are automatically allowed."""
    config = BashToolConfig()  # Use default config
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    # Test that newly added read-only commands are allowed by default
    test_commands = [
        "grep pattern file.txt",
        "cut -d',' -f1 file.csv",
        "sort file.txt",
        "tr 'a' 'b'",
        "uniq file.txt",
        "basename file.txt",
        "comm file1.txt file2.txt",
        "date",
        "diff file1.txt file2.txt",
        "dirname file.txt",
        "du -sh .",
        "fmt file.txt",
        "fold -w 80 file.txt",
        "join -t',' file1.csv file2.csv",
        "less file.txt",
        "md5sum file.txt",
        "more file.txt",
        "nl file.txt",
        "od -c file.bin",
        "paste file1.txt file2.txt",
        "readlink -f link.txt",
        "sha1sum file.txt",
        "sha256sum file.txt",
        "shasum file.txt",
        "stat file.txt",
        "sum file.txt",
        "tac file.txt",
        "which python",
    ]

    for cmd in test_commands:
        permission = bash_tool.resolve_permission(BashArgs(command=cmd))
        assert isinstance(permission, PermissionContext), (
            f"Permission should be PermissionContext for '{cmd}'"
        )
        assert permission.permission is ToolPermission.ALWAYS, (
            f"Command '{cmd}' should be always allowed"
        )


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
    assert "shell syntax requiring approval" in (result.reason or "")


@pytest.mark.parametrize("permission", [ToolPermission.ASK, ToolPermission.ALWAYS])
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("sudo ls", ToolPermission.NEVER),
        ("deploy production now", ToolPermission.NEVER),
        ("find . -exec id \\;", ToolPermission.NEVER),
        ("find . -execdir id \\;", ToolPermission.NEVER),
        ("find . -ok id \\;", ToolPermission.NEVER),
        ("find . -okdir id \\;", ToolPermission.NEVER),
        ("cat ../outside.txt", ToolPermission.NEVER),
        ("cd .. && pwd", ToolPermission.NEVER),
        ("printf hi > ../outside.txt", ToolPermission.NEVER),
        ("cat < ../outside.txt", ToolPermission.NEVER),
        ("cat escape", ToolPermission.NEVER),
        ("> ../outside.txt", ToolPermission.NEVER),
        ("echo hi >> escape", ToolPermission.NEVER),
        ("echo $(sudo ls)", ToolPermission.NEVER),
        ("python", ToolPermission.NEVER),
        ("python3", ToolPermission.NEVER),
        ("vim notes.txt", ToolPermission.NEVER),
        ("touch notes.txt", ToolPermission.ALWAYS),
        ("python script.py", ToolPermission.ALWAYS),
        ("python3 script.py", ToolPermission.ALWAYS),
        ("printf hi > notes.txt", ToolPermission.ALWAYS),
        ("printf hi >&2", ToolPermission.NEVER),
        ("deploy preview", ToolPermission.ALWAYS),
        ("catalog", ToolPermission.ALWAYS),
    ],
)
def test_posix_accident_guard_table(tmp_path, command, expected, permission):
    (tmp_path / "escape").symlink_to(tmp_path.parent / "outside.txt")
    config = BashToolConfig(
        permission=permission, sensitive_patterns=["sudo", "deploy production"]
    )
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None
    assert result.permission is expected
    assert not result.required_permissions
    if expected is ToolPermission.NEVER:
        assert result.reason


@pytest.mark.parametrize("command", ["cat {path}", "cd {path}"])
def test_posix_accident_guard_accepted_workspace_roots(tmp_path, command):
    root = tmp_path / "project"
    root.mkdir()
    accepted = tmp_path / "accepted"
    accepted.mkdir()
    config = BashToolConfig()
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=root)
    tool.workspace = Workspace.for_session(root, authorized_roots=[accepted])
    result = tool.resolve_permission(BashArgs(command=command.format(path=accepted)))
    assert result is not None
    assert result.permission is ToolPermission.ALWAYS
    tool.workspace = Workspace(
        root, (root, accepted), ceiling=Workspace.for_session(root)
    )
    result = tool.resolve_permission(BashArgs(command=command.format(path=accepted)))
    assert result is not None
    assert result.permission is ToolPermission.NEVER


def test_posix_accident_guard_cwd_and_tool_denial(tmp_path):
    config = BashToolConfig(permission=ToolPermission.NEVER)
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command="echo hi"))
    assert result is not None
    assert result.permission is ToolPermission.NEVER
    config.permission = ToolPermission.ALWAYS
    tool.workspace = Workspace(tmp_path, (tmp_path / "other",))
    result = tool.resolve_permission(BashArgs(command="echo hi"))
    assert result is not None
    assert result.permission is ToolPermission.NEVER


def test_posix_accident_guard_unknown_command_is_allowed(tmp_path: Path):
    config = BashToolConfig()
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command="python3 script.py"))
    assert result is not None
    assert result.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize("command", ["cat secret.txt", "echo hi > secret.txt"])
def test_posix_accident_guard_sensitive_paths_precede_hard_guards(
    tmp_path, monkeypatch, command
):
    # Exercise sensitive policy without creating or opening any sensitive file.
    monkeypatch.setattr(bash_module, "DEFAULT_SENSITIVE_PATTERNS", ["**/secret.txt"])
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None
    assert result.permission is ToolPermission.NEVER
    assert result.reason == "Sensitive file access denied (bash)"


def test_posix_accident_guard_scratch_cannot_expand_parent_ceiling(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config = BashToolConfig()
    tool = Bash(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=root,
        scratchpad_dir=scratch,
    )
    command = BashArgs(command=f"touch {scratch}/out")
    result = tool.resolve_permission(command)
    assert result is not None
    assert result.permission is ToolPermission.ALWAYS
    tool.workspace = Workspace(root, (root,), ceiling=Workspace.for_session(root))
    result = tool.resolve_permission(command)
    assert result is not None
    assert result.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "sort --files0-from=list.txt",
        "sort --files0-from list.txt",
        "sort --files0-from='list file.txt'",
        "sort --files0-from=-",
        "sort --files0-from=inside/list.txt",
        "sort -S -- --files0-from=/dev/null",
    ],
)
def test_sort_files0_from_requires_approval(command, tmp_path):
    list_file = tmp_path / "inside" / "list.txt"
    list_file.parent.mkdir()
    list_file.write_bytes(b"input\0")
    tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
        cwd=tmp_path,
    )

    result = tool.resolve_permission(BashArgs(command=command))

    assert result is not None
    assert result.permission is ToolPermission.NEVER
    assert (result.reason or "").startswith(
        "Command denied: side-effecting options are not permitted:"
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("sort input", ToolPermission.ALWAYS),
        ("sort --compress-program=cat input", ToolPermission.NEVER),
        ("sort --output=out input", ToolPermission.NEVER),
        ("sort --temporary-directory=tmp input", ToolPermission.NEVER),
        ("sort -o out input", ToolPermission.NEVER),
        ("sort -T tmp input", ToolPermission.NEVER),
    ],
)
def test_plain_sort_retains_existing_approval_policy(command, expected, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
        cwd=tmp_path,
    )

    result = tool.resolve_permission(BashArgs(command=command))

    assert result is not None
    assert result.permission is expected


@pytest.mark.parametrize(
    "command",
    [
        "sort -o out in",
        "sort -roout in",
        "find . -delete",
        "date -s now",
        "date -u -s now",
        "date -I -s now",
        "date -R --set=now",
        "date --rfc-email --set=now",
        "less -o log in",
        "tree -aotree.txt .",
        "tree -I -- -o output",
        "git diff --output=out",
        "git show --output=out",
        "git --exec-path=/tmp diff --output=out",
        "git --unknown-flag diff --output=out",
        "git --no-pager diff --output=../outside",
        "git -C . log --output=out",
        "command git diff --output=out",
        "builtin git diff --output=out",
        "git -c alias.pwn='!touch /outside/pwn' pwn",
        "git --config-env=foo=bar status",
        "find -- . -exec id \\;",
    ],
)
def test_side_effecting_options_are_denied(command):
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER
    if command.startswith("find"):
        assert (
            result.reason
            == "Command denied: find execution predicates are not permitted"
        )
    else:
        assert (result.reason or "").startswith(
            "Command denied: side-effecting options are not permitted:"
        )


def test_tree_boolean_options_do_not_hide_output_option():
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())

    for command in ("tree --prune -o output", "tree --matchdirs -o output"):
        result = tool.resolve_permission(BashArgs(command=command))
        assert result is not None and result.permission is ToolPermission.NEVER


def test_tree_boolean_short_option_does_not_hide_paths_or_output(tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )

    for command in ("tree -d /outside", "tree -d -o report ."):
        result = tool.resolve_permission(BashArgs(command=command))
        assert result is not None and result.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "file -p /outside",
        "file -p --files-from=/outside/list",
        "less -s /outside",
        "less -s -o log input",
        "less -J -o log input",
        "less -K -o log input",
        "less --status-column -o log input",
        "less --use-backslash -o result input",
        "less --wordwrap /outside/input",
        "less -a -o log input",
    ],
)
def test_reaudited_boolean_options_do_not_consume_following_tokens(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )

    result = tool.resolve_permission(BashArgs(command=command))

    assert result is not None and result.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "git diff --stat --output=result",
        "git diff --color --output=result",
        "git diff -W --output=result",
        "git diff --word-diff --ext-diff",
        "git diff --abbrev --output=result",
        "git diff --color-moved --output=result",
        "git diff --dirstat --output=result",
        "git diff --find-copies --output=result",
        "git diff --find-renames --output=result",
        "git diff --format --output=result",
        "git log --format --output=result",
        "git show --format --output=result",
        "git log --pretty --output=result",
        "git diff --submodule --output=result",
        "git diff --unified --output=result",
        "git diff -U --output=result",
    ],
)
def test_git_optional_value_and_boolean_options_do_not_hide_dangerous_options(command):
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())

    result = tool.resolve_permission(BashArgs(command=command))

    assert result is not None and result.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("grep --binary --file=/outside/patterns input", ToolPermission.NEVER),
        ("grep --binary -- --file=/outside/patterns", ToolPermission.ALWAYS),
    ],
)
def test_exact_boolean_long_option_precedes_value_option_abbreviation(
    command, expected, tmp_path
):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )

    result = tool.resolve_permission(BashArgs(command=command))

    assert result is not None and result.permission is expected


@pytest.mark.parametrize(
    "command",
    [
        "grep --regexp -- --file=/outside/patterns local.txt",
        "file -F -- --magic-file=/outside/magic local.txt",
    ],
)
def test_consumed_double_dash_does_not_hide_later_options(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )

    result = tool.resolve_permission(BashArgs(command=command))

    assert result is not None and result.permission is ToolPermission.NEVER


def test_grep_pattern_value_is_not_reparsed_as_file_option(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=root)

    result = tool.resolve_permission(
        BashArgs(command="grep -e --file=/outside/patterns local.txt")
    )

    assert result is not None and result.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "tree -X -o report.xml .",
        "less -E -o log input",
        "grep -n --file=/outside/patterns local.txt",
        "file -b --files-from=/outside/list",
        "sort -r --output=result input",
        "find . -print -delete",
        "git --no-pager diff --output=result",
        "date -u --set=now",
        "du -h --files0-from=/outside/list",
        "wc -l --files0-from=/outside/list",
        "diff -u --to-file=/outside/reference local",
    ],
)
def test_policy_boolean_options_do_not_hide_dangerous_options(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("tree -I -- -o report.xml .", ToolPermission.NEVER),
        ("less -P -- -o log input", ToolPermission.ALWAYS),
        ("grep --label -- --file=/outside/patterns local.txt", ToolPermission.NEVER),
        ("file --separator -- --files-from=/outside/list", ToolPermission.NEVER),
        ("sort --key -- --output=result input", ToolPermission.NEVER),
        ("find . -printf -- -delete", ToolPermission.NEVER),
        ("git -C -- diff --output=result", ToolPermission.NEVER),
        ("date --date -- --set=now", ToolPermission.NEVER),
        ("du --exclude -- --files0-from=/outside/list", ToolPermission.NEVER),
        ("wc --files0-from -- --files0-from=/outside/list", ToolPermission.NEVER),
        ("diff --label -- --to-file=/outside/reference a b", ToolPermission.NEVER),
    ],
)
def test_policy_value_options_consume_double_dash_then_resume_scanning(
    command, expected, tmp_path
):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is expected


@pytest.mark.parametrize(
    "command",
    [
        "tree -I -o .",
        "less --color -o input",
        "grep --regexp -f/outside/patterns local.txt",
        "file --separator --files-from=/outside/list local.txt",
        "sort --key --output=result input",
        "find . -printf -delete",
        "date --date --set=now",
        "du --exclude --files0-from=/outside/list",
        "wc --files0-from --files0-from=/outside/list",
        "diff --label --to-file=/outside/reference a b",
    ],
)
def test_policy_option_operands_are_not_reparsed(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "tree --output=result .",
        "less --log-file=log input",
        "grep --file=/outside/patterns local.txt",
        "file --files-from=/outside/list",
        "sort --output=result input",
        "git diff --output=result",
        "date --set=now",
        "du --files0-from=/outside/list",
        "wc --files0-from=/outside/list",
        "diff --to-file=/outside/reference local",
    ],
)
def test_policy_value_option_equals_forms_are_gated(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER


def test_less_value_options_consume_attached_cluster_operands():
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())

    benign = tool.resolve_permission(BashArgs(command="less -pfoo input"))
    dangerous = tool.resolve_permission(BashArgs(command="less -olog input"))

    assert benign is not None and benign.permission is ToolPermission.ALWAYS
    assert dangerous is not None and dangerous.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("less --prompt --log-file=example input", ToolPermission.ALWAYS),
        ("less --prompt=x --log-file=example input", ToolPermission.NEVER),
    ],
)
def test_less_optional_long_operands_consume_separate_tokens(command, expected):
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())

    result = tool.resolve_permission(BashArgs(command=command))

    assert result is not None and result.permission is expected


@pytest.mark.parametrize("option", ["-C.", "-C ."])
def test_git_global_directory_option_accepts_attached_and_separate_values(option):
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())

    result = tool.resolve_permission(BashArgs(command=f"git {option} log -1"))

    assert result is not None and result.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "date -Iseconds",
        "git --version",
        'date -d "tomorrow"',
        "find . -name '*.py'",
        "find . -name -delete -print",
        "sort -- --output=name",
        "grep --file=patterns.txt input",
    ],
)
def test_benign_options_are_allowed(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "template",
    [
        "grep --fil={path} in",
        "grep -if{path} in",
        "grep -e -- --file={path}",
        "file -m{path}:other",
        "du --files0-from={path}",
        "date -uf{path}",
        "date --reference={path}",
        "diff --to={path} in",
        "git --git-dir={path} status",
        "git --work-tree={path} diff",
        "git -C {path} status",
        "git diff --no-index {path} other",
        "git -C . diff --no-index {path} other",
        "tree {path}",
    ],
)
def test_option_paths_outside_workspace_are_denied(template, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside" / "data"
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=root)
    result = tool.resolve_permission(BashArgs(command=template.format(path=outside)))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert "outside" in (result.reason or "")


def test_sensitive_option_operand_is_denied(tmp_path, monkeypatch):
    monkeypatch.setattr(bash_module, "DEFAULT_SENSITIVE_PATTERNS", ["**/secret.txt"])
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command="grep --file=secret.txt input"))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert result.reason == "Sensitive file access denied (bash)"


@pytest.mark.parametrize(
    "command", ["/usr/bin/cat ../outside", "/usr/bin/find . -delete"]
)
def test_absolute_commands_cannot_bypass_basename_rules(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER


@pytest.mark.parametrize("command", ["eval echo ok", "exec -- echo ok"])
def test_eval_and_exec_are_denied(command):
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'EOF'\nhi\nEOF",
        "FOO=bar",
        "echo $(id)",
        "&&",
        "cat =sh",
        "cat ~vault/secret",
        "echo ${(e)payload}",
    ],
)
def test_unmodeled_shell_syntax_is_denied(command):
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER


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
    "command", ["command python", "builtin python", "builtin eval echo ok"]
)
def test_shell_wrappers_cannot_bypass_guardrails(command):
    tool = Bash(
        config_getter=lambda: BashToolConfig(denylist=["rm"]), state=BaseToolState()
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert any(
        keyword in (result.reason or "")
        for keyword in ("standalone", "eval and exec", "denylist")
    )


def test_command_wrapper_preserves_find_argument_boundaries():
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    result = tool.resolve_permission(
        BashArgs(command="command find . -name 'x -name' -delete")
    )
    assert result is not None and result.permission is ToolPermission.NEVER
    assert (
        result.reason == "Command denied: find execution predicates are not permitted"
    )


@pytest.mark.parametrize(
    "command",
    [
        "env rm -rf /",
        'env -S "rm -rf /"',
        "env -C /outside touch file",
        "/usr/bin/env rm -rf /",
        "env VAR=value command",
    ],
)
def test_env_wrappers_are_denied_without_unwrapping(command):
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert result.reason == "Command denied: env wrapper cannot be safely inspected"


@pytest.mark.parametrize("command", ["echo \0", "echo \ud800"])
def test_shell_analysis_failures_are_denied(command):
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert "shell analysis failed" in (result.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "CI=1 npm test",
        "PYTHONPATH=. python -m pytest",
        "PYTHONPATH=./lib python -m pytest",
    ],
)
def test_safe_literal_environment_prefixes_are_allowed(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "PATH=/tmp git status",
        "LESS=-o/tmp/output less README.md",
        "HOME=/tmp git status",
        "PYTHONPATH=/tmp python3 -c pass",
        "PYTHONPATH=../lib python3 -c pass",
        "PYTHONPATH=~/lib python3 -c pass",
        "PYTHONPATH=.:/tmp python3 -c pass",
        "GIT_EXTERNAL_DIFF=cat git diff",
        "LD_PRELOAD=plugin.so cmd",
        "NODE_OPTIONS=--inspect node",
        "CI=$X cmd",
        "lower=value cmd",
    ],
)
def test_unsafe_environment_prefixes_are_denied(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert "environment assignment" in (result.reason or "")


def test_environment_prefix_still_applies_full_command_policy(tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command="CI=1 sort -o out input"))
    assert result is not None and result.permission is ToolPermission.NEVER


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
    "command", ["sort $FLAGS input", 'echo "$PWD"', "cmd $EMPTY", "cmd $MULTIWORD"]
)
def test_general_variable_expansion_stays_denied_with_explanation(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert "unsupported in v0.1" in (result.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "rm -r foo",
        "rm -fr foo",
        "rm --recursive foo",
        "rm -R target",
        "rm --rec target",
        "git reset --hard HEAD",
        "git --git-dir .git reset --hard",
        "git --work-tree . reset --hard",
        "git clean -e -- -fd",
        "git clean -e -n -fd",
        "git -C . reset --hard HEAD",
        "git clean -fd",
        "git clean -f -d",
        "env CI=1 rm -rf build",
        "xargs rm -rf",
        "chmod -R 777 build",
        "chmod -R -- 777 target",
        "chmod -R o=rw target",
        "chmod -R go=w target",
        "chmod -R a=w target",
        "chmod -R o+w build",
        "chown -R user build",
    ],
)
def test_destructive_commands_are_denied(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "git clean -n",
        "git clean -nfd",
        "rm file.txt",
        "rm -- -r",
        "git reset --soft HEAD",
        "git reset -- --hard",
        "git clean -- -f",
        "chmod -R 755 build",
    ],
)
def test_destructive_guard_lookalikes_are_allowed(command, tmp_path):
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.ALWAYS


def test_denylist_precedes_syntax_and_nested_wrapper_denial():
    tool = Bash(
        config_getter=lambda: BashToolConfig(denylist=["denied-marker"]),
        state=BaseToolState(),
    )
    for command in ["denied-marker > out", "eval denied-marker harmless"]:
        result = tool.resolve_permission(BashArgs(command=command))
        assert result is not None and result.permission is ToolPermission.NEVER
        assert "matches denylist pattern" in (result.reason or "")
