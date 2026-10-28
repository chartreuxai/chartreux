from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_diagnostics import (
    MAX_DIAGNOSTIC_LENGTH,
    DenialClass,
    Diagnostic,
    aggregate_denial_class,
    render_diagnostic,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.secret_redaction import ScrubPolicy, bind_policy

# Phrasing the renderer adds only for SYNTAX denials; a POLICY denial must
# never carry it.
_ALTERNATIVE_MARKERS = (
    "Instead,",
    "Instead ",
    "use read_file",
    "use grep",
    "use the edit tool",
    "use write_file",
    "use bash_start",
    "use non-recursive",
    "rmdir",
    "try ",
)


@pytest.mark.parametrize("token", ["-r", "-rf", "-R", "--recursive", "--rec"])
@pytest.mark.parametrize("prefix", ["", "uv run ", "env CI=1 uv run "])
def test_recursive_rm_original_option_is_a_hard_guard_without_alternative(
    tmp_path: Path, token: str, prefix: str
) -> None:
    # Recursive rm is a policy hard guard: the message states the blocked
    # requirement plainly and suggests no alternative deletion method.
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=f"{prefix}rm {token} file"))
    assert result is not None and result.permission == ToolPermission.NEVER
    assert result.reason == (
        f"Command denied: recursive rm option '{token}' in 'rm {token} file'. "
        "This is a hard guard regardless of target."
    )
    assert not any(marker in (result.reason or "") for marker in _ALTERNATIVE_MARKERS)


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
    [
        "path_glob",
        "denylist",
        "policy",
        "command",
        "analysis",
        "nested_analysis",
        "dangerous_command",
        "dangerous_guard",
    ],
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


def test_push_destination_denial_detail_with_apostrophe_renders_informatively() -> None:
    # The shell-aware redactor returns None on the detail's unmatched quote
    # ("the destination's receive-pack and hooks"); the detail field is
    # policy-authored prose, not shell source, so it must render with only
    # actual secret values redacted instead of degrading to [REDACTED].
    secret = "synthetic-unrelated-secret-0123456789"
    detail = (
        "git push: destination 'https://x/y' is a path or URL, not a remote "
        "name: a push to it runs the destination's receive-pack and hooks, "
        "outside the repository the grant covers"
    )
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))):
        rendered = render_diagnostic(Diagnostic("dangerous_guard", detail=detail))
    assert "Command denied: git push: destination" in rendered
    assert r"the destination\u0027s receive-pack and hooks" in rendered
    assert "REDACTED" not in rendered
    assert secret not in rendered


def test_detail_preview_fallback_still_redacts_registered_secret_values() -> None:
    secret = "synthetic-unrelated-secret-0123456789"
    detail = f"the credential {secret} cannot authorize this command"
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))):
        rendered = render_diagnostic(Diagnostic("dangerous_guard", detail=detail))
    assert secret not in rendered
    assert r"\u005bREDACTED\u005d" in rendered


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


def test_dangerous_authority_denial_names_operation_and_consequence(
    tmp_path: Path,
) -> None:
    # Missing dangerous authority is distinguishable from a denylist veto and
    # from a hard guard: it names the operation and its consequence.
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command="git push origin main"))
    assert result is not None and result.permission == ToolPermission.NEVER
    assert result.denial_kind == "dangerous_command"
    assert "requires approval for a dangerous git operation" in (result.reason or "")
    assert "git.push" in (result.reason or "")
    assert "Publish local commits to the remote" in (result.reason or "")


def test_dangerous_authority_denial_redacts_secrets_and_escapes_markup(
    tmp_path: Path,
) -> None:
    secret = "synthetic-dangerous-secret-0123456789"
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))):
        result = tool.resolve_permission(
            BashArgs(command=f"git push {shlex.quote(secret + '[bold]\n\x1b[31m')}")
        )
    assert result is not None and result.permission == ToolPermission.NEVER
    assert result.denial_kind == "dangerous_command"
    assert result.reason and secret not in result.reason
    assert "REDACTED" in result.reason
    assert "[bold]" not in result.reason
    assert "\x1b" not in result.reason and "\n" not in result.reason
    assert len(result.reason) <= MAX_DIAGNOSTIC_LENGTH


def test_denylist_veto_and_hard_guards_are_distinguishable(tmp_path: Path) -> None:
    # An absolute denylist veto and a hard guard never read as a missing
    # dangerous authority, and neither carries a grantable proposal.
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)

    veto = tool.resolve_permission(BashArgs(command="vim file.txt"))
    assert veto is not None and veto.permission == ToolPermission.NEVER
    assert veto.denial_kind is None
    assert veto.dangerous_command_proposal is None
    assert "matches denylist pattern 'vim'" in (veto.reason or "")

    guard = tool.resolve_permission(BashArgs(command="git push --bogus origin main"))
    assert guard is not None and guard.permission == ToolPermission.NEVER
    assert guard.denial_kind is None
    assert guard.dangerous_command_proposal is None
    assert "requires approval" not in (guard.reason or "")
    assert "--bogus" in (guard.reason or "")

    policy_guard = tool.resolve_permission(BashArgs(command="git reset --hard HEAD"))
    assert policy_guard is not None and policy_guard.permission == ToolPermission.NEVER
    assert policy_guard.denial_kind is None
    assert policy_guard.dangerous_command_proposal is None


def test_unlocatable_repository_is_a_hard_guard(tmp_path: Path) -> None:
    # A repository marker that cannot be interpreted fails closed with no
    # grantable proposal.
    (tmp_path / ".git").write_text("not a gitdir pointer")
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command="git push"))
    assert result is not None and result.permission == ToolPermission.NEVER
    assert result.denial_kind is None
    assert result.dangerous_command_proposal is None
    assert "cannot be located" in (result.reason or "")


def test_denial_class_defaults_to_policy_fail_closed() -> None:
    # A diagnostic created without an explicit class is a policy denial:
    # unknown and uncategorized denials name no alternative.
    unclassified = Diagnostic("denylist", "vim", "vim file")
    assert unclassified.denial_class is DenialClass.POLICY
    assert unclassified.alternative is None
    analysis_default = Diagnostic("analysis", detail="anything")
    assert analysis_default.denial_class is DenialClass.POLICY
    assert analysis_default.alternative is None


@pytest.mark.parametrize(
    ("code", "detail", "alternative"),
    [
        (
            "analysis",
            "command substitution",
            "use read_file with offset and limit to read files; "
            "use grep to locate content and line numbers; "
            "use the edit tool to modify files",
        ),
        (
            "nested_analysis",
            "command substitution",
            "use read_file with offset and limit to read files; "
            "use grep to locate content and line numbers; "
            "use the edit tool to modify files",
        ),
        (
            "command",
            "malformed shell quoting",
            "write the command with well-formed shell quoting",
        ),
    ],
)
def test_syntax_denials_name_the_compliant_alternative(
    code: str, detail: str, alternative: str
) -> None:
    diagnostic = Diagnostic(
        code, detail=detail, denial_class=DenialClass.SYNTAX, alternative=alternative
    )
    rendered = render_diagnostic(diagnostic)
    assert diagnostic.denial_class is DenialClass.SYNTAX
    assert rendered.endswith(f"Instead, {alternative}.")
    assert detail in rendered


def test_syntax_aggregate_joins_related_alternatives() -> None:
    first = Diagnostic(
        "analysis",
        detail="command substitution",
        denial_class=DenialClass.SYNTAX,
        alternative="use read_file with offset and limit to read files",
    )
    second = Diagnostic(
        "analysis",
        detail="heredocs are unsupported in v0.1",
        denial_class=DenialClass.SYNTAX,
        alternative="use write_file to create the content",
    )
    aggregate = Diagnostic(
        "analysis",
        related=(first, second),
        denial_class=aggregate_denial_class((first, second)),
    )
    rendered = render_diagnostic(aggregate)
    assert aggregate.denial_class is DenialClass.SYNTAX
    assert "command substitution" in rendered
    assert "heredocs are unsupported in v0.1" in rendered
    assert (
        "Instead, use read_file with offset and limit to read files; "
        "use write_file to create the content." in rendered
    )


@pytest.mark.parametrize(
    ("code", "requirement"),
    [
        ("analysis", "the blocked requirement"),
        ("nested_analysis", "the blocked requirement"),
        ("command", "the blocked requirement"),
        ("policy", "the blocked requirement"),
        ("dangerous_guard", "the blocked requirement"),
        ("text", "the blocked requirement"),
        ("denylist", "matches denylist pattern"),
        ("standalone", "is not allowed as a standalone command"),
        ("dangerous_command", "requires approval for a dangerous git operation"),
        ("recursive_rm", "This is a hard guard regardless of target."),
        ("path_glob", "cannot be safely inspected"),
        ("eval_exec", "eval and exec cannot be safely inspected"),
        ("protected_env", "unsetting a protected shell environment variable"),
        ("sensitive_command", "sensitive command rule"),
        ("git_cwd_unknown", "not statically known"),
        ("tool_denied", "Tool denied: bash"),
        ("cwd_outside", "outside the authorized workspace"),
        ("operand_cwd_unknown", "not statically known"),
        ("path_outside", "outside the authorized workspace"),
        ("metadata_unlocated", "protected Git metadata cannot be located"),
        ("metadata_redirect", "redirection to protected Git metadata"),
        ("metadata_operand", "file operand reaches protected Git metadata"),
        ("metadata_hardlink", "hard link to protected Git metadata"),
        ("metadata_worktree", "work tree resolves inside protected Git metadata"),
        ("byte_budget", "shell analysis budget exceeded"),
    ],
)
def test_policy_denials_state_the_requirement_without_alternative(
    code: str, requirement: str
) -> None:
    # A policy denial states the blocked requirement plainly and suggests
    # nothing: no alternative phrasing of any kind.
    rendered = render_diagnostic(
        Diagnostic(code, detail="the blocked requirement", command_part="part")
    )
    assert requirement in rendered
    assert not any(marker in rendered for marker in _ALTERNATIVE_MARKERS)


def test_aggregate_denial_class_is_fail_closed() -> None:
    syntax = Diagnostic(
        "analysis",
        detail="command substitution",
        denial_class=DenialClass.SYNTAX,
        alternative="use read_file with offset and limit to read files",
    )
    policy = Diagnostic("analysis", detail="dangerous environment assignment (PATH)")
    unknown = Diagnostic("analysis", detail="unsupported syntax (mystery_node)")
    assert aggregate_denial_class((syntax,)) is DenialClass.SYNTAX
    # A mixed construct — unmodeled syntax plus a policy violation — and any
    # unknown or unclassified component classify as POLICY.
    assert aggregate_denial_class((syntax, policy)) is DenialClass.POLICY
    assert aggregate_denial_class((policy, syntax)) is DenialClass.POLICY
    assert aggregate_denial_class((syntax, unknown)) is DenialClass.POLICY
    assert aggregate_denial_class((unknown,)) is DenialClass.POLICY
    assert aggregate_denial_class(()) is DenialClass.POLICY


def test_syntax_alternative_survives_redaction_and_bounds() -> None:
    # The enrichment must not weaken the redaction, escaping and bounding
    # boundary: a secret in the detail is still redacted and the enriched
    # message stays within the diagnostic budget.
    secret = "synthetic-enriched-secret-0123456789"
    policy = ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))
    with bind_policy(policy):
        rendered = render_diagnostic(
            Diagnostic(
                "analysis",
                detail="command substitution in " + secret,
                denial_class=DenialClass.SYNTAX,
                alternative=(
                    "use read_file with offset and limit to read files; "
                    "use grep to locate content and line numbers; "
                    "use the edit tool to modify files"
                ),
            )
        )
    assert secret not in rendered
    assert "REDACTED" in rendered
    assert "Instead," in rendered
    assert "use read_file with offset and limit" in rendered
    assert len(rendered) <= MAX_DIAGNOSTIC_LENGTH


@pytest.mark.parametrize(
    ("command", "requirement", "alternative"),
    [
        (
            "cat $(locate chartreux)",
            "command substitution",
            "use read_file with offset and limit to read files; "
            "use grep to locate content and line numbers; "
            "use the edit tool to modify files",
        ),
        (
            "python3 << 'EOF'\nprint(42)\nEOF",
            "heredocs are unsupported in v0.1",
            "use write_file to create the content, or use read_file with "
            "offset and limit, grep, and the edit tool for existing files",
        ),
        (
            "for i in 1 2; do echo $i; done",
            "for loop",
            "split the work into separate simple commands, or use read_file "
            "with offset and limit, grep, and the edit tool for file work",
        ),
        (
            "echo =ls",
            "Zsh equals expansion",
            "drop the leading equals sign and invoke the command by its "
            "name or absolute path",
        ),
        (
            "echo $HOME",
            "variable expansion",
            "write the literal value directly as the argument",
        ),
        (
            "bash -c 'echo $(x)'",
            "command substitution",
            "use read_file with offset and limit to read files",
        ),
    ],
)
def test_syntax_denials_name_alternative_end_to_end(
    tmp_path: Path, command: str, requirement: str, alternative: str
) -> None:
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission == ToolPermission.NEVER
    reason = result.reason or ""
    assert requirement in reason
    assert "Instead," in reason
    assert alternative in reason
    assert len(reason) <= MAX_DIAGNOSTIC_LENGTH


@pytest.mark.parametrize(
    ("command", "requirements"),
    [
        ("rm -r build", ("This is a hard guard regardless of target.",)),
        ("python3", ("is not allowed as a standalone command",)),
        ("vim file.txt", ("matches denylist pattern 'vim'",)),
        ("eval 'echo hi'", ("eval and exec cannot be safely inspected",)),
        (
            "PYTHONPATH=/abs python x",
            ("dangerous environment assignment", "PYTHONPATH"),
        ),
        ("LD_PRELOAD=x ls", ("dangerous environment assignment", "PRELOAD")),
        ("foo=bar ls", ("an environment assignment with a non-uppercase name",)),
        (
            "unset PATH",
            ("unsetting shell environment variables cannot be safely inspected",),
        ),
        (
            "hash -p /usr/bin/curl harmless",
            ("unsupported command lookup modification", "hash"),
        ),
        ("echo hi > .git/config", ("redirection to protected Git metadata",)),
        ("cd / && pwd", ("outside the authorized workspace",)),
        ("find . -exec id \\;", ("find execution predicates are not permitted",)),
    ],
)
def test_policy_denials_suggest_nothing_end_to_end(
    tmp_path: Path, command: str, requirements: tuple[str, ...]
) -> None:
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission == ToolPermission.NEVER
    reason = result.reason or ""
    # The rendered detail escapes markup characters, so the requirement is
    # asserted on escape-free fragments.
    for requirement in requirements:
        assert requirement in reason
    assert not any(marker in reason for marker in _ALTERNATIVE_MARKERS)
    assert len(reason) <= MAX_DIAGNOSTIC_LENGTH


@pytest.mark.parametrize(
    "command",
    [
        # Unmodeled syntax plus a policy violation in one command.
        "PYTHONPATH=/abs cat $(ls)",
        # A non-literal redirect target expands a variable.
        "echo hi > $TMP/x",
        # A herestring leaves an unrecognized node in the command.
        "cat <<<x",
    ],
)
def test_mixed_constructs_classify_as_policy_end_to_end(
    tmp_path: Path, command: str
) -> None:
    # Fail closed: a mixed construct is a policy denial and names no
    # alternative, even though part of it is unmodeled syntax.
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission == ToolPermission.NEVER
    reason = result.reason or ""
    assert not any(marker in reason for marker in _ALTERNATIVE_MARKERS)
    assert len(reason) <= MAX_DIAGNOSTIC_LENGTH


def test_mixed_syntax_and_dangerous_authority_renders_as_policy(tmp_path: Path) -> None:
    # AM-F: the whole-call denial class is decided before rendering. A call
    # that also lacks a dangerous-command authority carries a policy-class
    # reason, so the aggregate is a policy denial: it names the blocked
    # requirements and suggests no alternative — suggesting one invites the
    # refused action through another tool.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    result = tool.resolve_permission(
        BashArgs(command="git checkout main; echo $(cat AGENTS.md)")
    )
    assert result is not None and result.permission is ToolPermission.NEVER
    reason = result.reason or ""
    assert "command substitution" in reason
    assert "git.checkout" in reason
    assert not any(marker in reason for marker in _ALTERNATIVE_MARKERS)
    assert len(reason) <= MAX_DIAGNOSTIC_LENGTH
    # The authorization outcome is unchanged: still a plain denial with no
    # grantable proposal (the dangerous flow reports only the sole missing
    # authority).
    assert result.denial_kind is None
    assert result.dangerous_command_proposal is None

    # A pure-syntax denial keeps its compliant alternative.
    result = tool.resolve_permission(BashArgs(command="echo $(cat AGENTS.md)"))
    assert result is not None and result.permission is ToolPermission.NEVER
    assert "Instead," in (result.reason or "")


def test_authorization_outcomes_are_unchanged(tmp_path: Path) -> None:
    # The class-aware messages change denial text and typing only: no
    # currently-denied command becomes allowed and vice versa.
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    denied = [
        "rm -r build",
        "cat $(locate chartreux)",
        "python3 << 'EOF'\nprint(42)\nEOF",
        "for i in 1 2; do echo $i; done",
        "echo =ls",
        "echo $HOME",
        "python3",
        "vim file.txt",
        "eval 'echo hi'",
        "PYTHONPATH=/abs python x",
        "unset PATH",
        "echo hi > .git/config",
        "cd / && pwd",
        "find . -exec id \\;",
        "git push origin main",
        "git reset --hard HEAD",
        "bash -c 'echo $(x)'",
    ]
    allowed = [
        "echo hi",
        "ls",
        "git status",
        "echo hi > out.txt",
        "echo hi 2>&1",
        "echo hi > /dev/null",
    ]
    for command in denied:
        result = tool.resolve_permission(BashArgs(command=command))
        assert result is not None and result.permission == ToolPermission.NEVER, command
    for command in allowed:
        result = tool.resolve_permission(BashArgs(command=command))
        assert result is None or result.permission != ToolPermission.NEVER, command


def test_recognized_but_rejected_shapes_keep_their_typed_class(
    tmp_path: Path, monkeypatch
) -> None:
    # A recognized-but-rejected static-expansion shape keeps its proper
    # class: the multi-match locate-then-print variant is an unexecutable
    # construct, so its denial is SYNTAX and names the compliant alternative
    # through the standard renderer.
    from chartreux.core.tools.builtins._shell_permission_analysis import (
        analyze_shell_command,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "lines.txt").write_text("lit\nalpha\nlit\n")
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=tmp_path)
    command = "sed -n \"$(grep -n 'lit' lines.txt | cut -d: -f1),+45p\" lines.txt"
    context = tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.NEVER
    analysis = analyze_shell_command(command)
    assert analysis.approval_diagnostic.denial_class is DenialClass.SYNTAX
    rendered = render_diagnostic(analysis.approval_diagnostic)
    assert "locating multiple lines" in rendered
    assert "Instead," in rendered
    assert "head -1" in rendered
