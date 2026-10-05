from __future__ import annotations

from pathlib import Path
import shlex

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_diagnostics import (
    MAX_DIAGNOSTIC_LENGTH,
    Diagnostic,
    render_diagnostic,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.secret_redaction import ScrubPolicy, bind_policy


@pytest.mark.parametrize("token", ["-r", "-rf", "-R", "--recursive", "--rec"])
@pytest.mark.parametrize("prefix", ["", "uv run ", "env CI=1 uv run "])
def test_recursive_rm_original_option_and_alternative(
    tmp_path: Path, token: str, prefix: str
) -> None:
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=f"{prefix}rm {token} file"))
    assert result is not None and result.permission == ToolPermission.NEVER
    assert result.reason == (
        f"Command denied: recursive rm option '{token}' in 'rm {token} file'. "
        "This is a hard guard regardless of target. "
        "Use non-recursive rm -- 'file', then rmdir -- 'dir'."
    )


@pytest.mark.parametrize(
    "unsafe",
    [
        "\x1b[31m",
        "\n",
        "\r",
        "\t",
        "\x00",
        "[bold]",
        "<tag>",
        "`code`",
        "\u202e",
        "\u2028",
    ],
)
@pytest.mark.parametrize(
    "code",
    ["path_glob", "denylist", "policy", "command", "analysis", "nested_analysis"],
)
def test_untrusted_fields_cannot_inject(code: str, unsafe: str) -> None:
    rendered = render_diagnostic(Diagnostic(code, unsafe, unsafe, unsafe))
    assert unsafe not in rendered
    assert not any(ord(character) < 32 for character in rendered)
    assert not any(character in rendered for character in "[]<>`")
    assert len(rendered) <= MAX_DIAGNOSTIC_LENGTH


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('echo "don\'t"', r"echo \u0022don\u0027t\u0022"),
        ("echo 'unclosed", r"echo \u0027unclosed"),
    ],
)
def test_command_preview_without_secrets_preserves_literal_quotes(
    monkeypatch: pytest.MonkeyPatch, command: str, expected: str
) -> None:
    monkeypatch.setattr(
        "chartreux.core.tools.builtins._shell_diagnostics.known_secret_values",
        lambda: (),
    )
    rendered = render_diagnostic(Diagnostic("denylist", "echo", command))
    assert rendered == (
        f"Command denied: '{expected}' matches denylist pattern 'echo'. "
        "Do not attempt to run this command."
    )
    assert "REDACTED" not in rendered


def test_unrecoverable_command_preview_with_secrets_remains_conservative() -> None:
    secret = "synthetic-unrelated-secret-0123456789"
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))):
        rendered = render_diagnostic(Diagnostic("denylist", "echo", "echo 'unclosed"))
    assert r"\u005bREDACTED\u005d" in rendered
    assert "unclosed" not in rendered


def test_redaction_precedes_escaping_and_truncation() -> None:
    secret = "synthetic-secret-[markup]\n" + "z" * 400
    policy = ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))
    with bind_policy(policy):
        rendered = render_diagnostic(
            Diagnostic("path_glob", "prefix" + secret, "cat " + secret)
        )
    assert "synthetic-secret" not in rendered
    assert "REDACTED" in rendered
    assert len(rendered) <= MAX_DIAGNOSTIC_LENGTH


def test_glob_feedback_names_candidate_and_segment(tmp_path: Path) -> None:
    token = "./[bold]\n\x1b[31m*"
    command = "cat " + shlex.quote(token)
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission == ToolPermission.NEVER
    assert (
        result.reason
        and "candidate" in result.reason
        and "command segment" in result.reason
    )
    assert "cannot be safely inspected" in result.reason
    assert "\\u001b" in result.reason and "\\u000a" in result.reason
    assert "[bold]" not in result.reason


def test_secret_in_original_command_is_absent_from_permission_feedback(
    tmp_path: Path,
) -> None:
    secret = "synthetic-command-token-secret-0123456789"
    config = BashToolConfig(denylist=["printf"])
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))):
        result = tool.resolve_permission(
            BashArgs(command="printf " + shlex.quote(secret + "[bold]\n\x1b[31m"))
        )
    assert result is not None and result.permission == ToolPermission.NEVER
    assert result.reason and secret not in result.reason
    assert "REDACTED" in result.reason
    assert (
        "[bold]" not in result.reason
        and "\x1b" not in result.reason
        and "\n" not in result.reason
    )


@pytest.mark.parametrize("suffix", ["\x1b[31m", "[bold]", "\n", "`markup`"])
def test_recursive_rm_option_cluster_is_escaped(tmp_path: Path, suffix: str) -> None:
    option = "-rf" + suffix
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(
        BashArgs(command="rm " + shlex.quote(option) + " file")
    )
    assert result is not None and result.permission == ToolPermission.NEVER
    assert result.reason and "recursive rm option '-rf" in result.reason
    assert suffix not in result.reason
    assert "hard guard regardless of target" in result.reason


@pytest.mark.parametrize(
    "prefix", ["", "uv run ", "npx --package fixture ", "env CI=1 "]
)
@pytest.mark.parametrize("nesting", [0, 1, 2])
@pytest.mark.parametrize("denylist", [["printf"], ["bash", "printf"]])
def test_apostrophe_secret_denials_match_direct_form(
    tmp_path: Path, prefix: str, nesting: int, denylist: list[str]
) -> None:
    secret = "synthetic-credential-before'after-0123456789"
    config = BashToolConfig(denylist=denylist)
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))):
        # Double quoting keeps the original credential contiguous in the direct
        # form; wrapper expansion instead serializes it using shlex.join.
        command = f'printf "{secret}"'
        if nesting:
            command = shlex.join(["printf", secret])
            for _ in range(nesting):
                command = shlex.join(["bash", "-c", command])
        direct = tool.resolve_permission(BashArgs(command=command))
        wrapped = tool.resolve_permission(BashArgs(command=prefix + command))
    assert direct is not None and wrapped is not None
    assert direct.permission == wrapped.permission == ToolPermission.NEVER
    assert direct.reason == wrapped.reason
    for result in (direct, wrapped):
        assert result.reason and "REDACTED" in result.reason
        assert secret not in result.reason
        assert "synthetic-credential-before" not in result.reason
        assert "after-0123456789" not in result.reason


@pytest.mark.parametrize("unrecoverable", ["quoting", "depth"])
def test_unrecoverable_nested_source_omits_credential_fragments(
    unrecoverable: str,
) -> None:
    secret = "synthetic-credential-before'after-0123456789"
    source = shlex.join(["printf", secret])
    if unrecoverable == "quoting":
        source += " '"
    else:
        for _ in range(10):
            source = shlex.join(["bash", "-c", source])
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))):
        rendered = render_diagnostic(
            Diagnostic("denylist", "bash", shlex.join(["bash", "-c", source]))
        )
    assert "REDACTED" in rendered
    assert secret not in rendered
    assert "synthetic-credential-before" not in rendered
    assert "after-0123456789" not in rendered


def test_configured_shell_shaped_denylist_pattern_is_redacted(tmp_path: Path) -> None:
    secret = "synthetic-pattern-before'after-0123456789"
    command = shlex.join(["printf", secret])
    config = BashToolConfig(denylist=[command])
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))):
        result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission == ToolPermission.NEVER
    assert result.reason and "matches denylist pattern" in result.reason
    assert "REDACTED" in result.reason
    for fragment in (secret, "synthetic-pattern-before", "after-0123456789"):
        assert fragment not in result.reason


def test_long_fields_are_bounded_without_partial_escape() -> None:
    rendered = render_diagnostic(Diagnostic("denylist", "[" * 10000, "\x1b" * 10000))
    assert len(rendered) <= MAX_DIAGNOSTIC_LENGTH
    assert "..." in rendered
    assert "\\u001..." not in rendered
