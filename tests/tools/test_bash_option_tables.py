"""Option-parsing table tests for the bash suite (sort files0-from, side-effecting options, tree/grep/less/policy boolean and value option tables, git option tables, benign option forms, option paths outside the workspace, the sensitive option operand, and the interpreter option tables)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
import chartreux.core.tools.builtins.bash as bash_module
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from tests.tools._bash_test_support import _w11_permission


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
        "Command denied: unsafe or unmodeled command options are not permitted:"
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
            "Command denied: unsafe or unmodeled command options are not permitted:"
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
    assert result.reason == r"Sensitive file access denied \u0028bash\u0029"


@pytest.mark.parametrize(
    "command",
    [
        "python3 -B -c 'print(1)'",
        "python3 -u -c'print(1)'",
        "node --eval '1'",
        "node -pe '1'",
        "perl -we '1'",
        "ruby -we '1'",
    ],
)
@pytest.mark.parametrize("configured_denial", [False, True])
def test_interpreter_switch_after_flags_policy(
    command: str, tmp_path: Path, configured_denial: bool
):
    config = BashToolConfig()
    if configured_denial:
        config.denylist = ["python -c", "node -e", "perl -e", "ruby -e"]
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None
    expected = ToolPermission.NEVER if configured_denial else ToolPermission.ALWAYS
    assert result.permission is expected
    if configured_denial:
        assert "matches denylist pattern" in (result.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "python3 -W ignore -c 'print(1)'",
        "python -X dev -c 'print(1)'",
        "python -Wignore -c 'print(1)'",
        "node --require=module --eval=1",
        "perl -I lib -e '1'",
        "ruby -I lib -e 'puts 1'",
    ],
)
@pytest.mark.parametrize("configured_denial", [False, True])
def test_interpreter_value_options_before_inline_code_policy(
    command: str, tmp_path: Path, configured_denial: bool
) -> None:
    config = BashToolConfig()
    if configured_denial:
        config.denylist = ["python -c", "node -e", "perl -e", "ruby -e"]
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None
    expected = ToolPermission.NEVER if configured_denial else ToolPermission.ALWAYS
    assert result.permission is expected
    if configured_denial:
        assert "matches denylist pattern" in (result.reason or "")


@pytest.mark.parametrize(
    "command", [".venv/bin/python --version", "python --version", "python3 --version"]
)
def test_interpreter_own_version_flag_resolves(command, tmp_path: Path) -> None:
    tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
        cwd=tmp_path,
    )
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.ALWAYS


def test_unknown_interpreter_option_before_inline_code_is_denied(
    tmp_path: Path,
) -> None:
    result = _w11_permission("python --unknown=value -c 'print(1)'", tmp_path)
    assert result.permission is ToolPermission.NEVER
    assert "unsupported python option" in (result.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "python -u script.py -c argument",
        "python -m module -c argument",
        "python -W ignore::DeprecationWarning script.py",
        "node --require=module script.js --eval=1",
        "perl -I lib script.pl -e argument",
        "ruby -Ilib script.rb -e argument",
    ],
)
def test_interpreter_options_before_script_stay_allowed(
    command: str, tmp_path: Path
) -> None:
    assert _w11_permission(command, tmp_path).permission is ToolPermission.ALWAYS
