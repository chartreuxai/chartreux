from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.builtins._shell_diagnostics import (
    DenialClass,
    Diagnostic,
    render_diagnostic,
)
from chartreux.core.tools.builtins._shell_permission_analysis import (
    aggregate_approval_diagnostic,
    analyze_shell_command,
    argv_permission_diagnostics,
    argv_permission_reasons,
)


def test_analysis_diagnostics_retain_original_source() -> None:
    analysis = analyze_shell_command("cat $FIXTURE")
    assert analysis.requires_approval
    diagnostic = next(
        item for item in analysis.diagnostics if item.offending_token == "$FIXTURE"
    )
    assert diagnostic.code == "analysis"
    assert diagnostic.command_part == "cat $FIXTURE"
    assert (
        tuple(item.detail for item in analysis.diagnostics) == analysis.approval_reasons
    )
    assert analysis.approval_diagnostic.related == analysis.diagnostics


@pytest.mark.parametrize(
    "command",
    [
        'uv run python -c "print(1 * 2)"',
        "uv run python -c \"print('[abc]')\"",
        "env CI=1 uv run python -c \"print('[abc]')\"",
    ],
)
def test_executor_payload_and_native_arguments_are_not_package_globs(
    command: str,
) -> None:
    assert not analyze_shell_command(command).requires_approval


@pytest.mark.parametrize(
    "command",
    [
        "uv run pip install fixture[extra]",
        "env CI=1 uv run pip install fixture[extra]",
        "npx fixture[extra]",
        "uv run npx fixture[extra]",
        "cargo run -- [abc]",
        "go run main.go [abc]",
    ],
)
def test_nested_package_operands_keep_original_quote_checks(command: str) -> None:
    assert (
        "unquoted package bracket syntax may expand as a shell glob"
        in analyze_shell_command(command).approval_reasons
    )


@pytest.mark.parametrize(
    "argv,reason",
    [
        (["env", "PATH=/outside", "echo", "hi"], "dangerous environment assignment"),
        (
            ["env", "PYTHONPATH=../outside", "python", "-c", "code"],
            "dangerous environment assignment",
        ),
        (["hash", "-p", "./fake", "git"], "command lookup modification"),
        (["alias", "git=./fake"], "command lookup modification"),
    ],
)
def test_recursive_argv_security_checks_without_shell_reparse(
    argv: list[str], reason: str
) -> None:
    assert any(reason in value for value in argv_permission_reasons(argv))
    assert argv_permission_reasons(["echo", *argv]) == ()


@pytest.mark.parametrize(
    "name",
    [
        "UV_WORKING_DIR",
        "UV_WORKING_DIRECTORY",
        "UV_PROJECT",
        "PIP_LOG",
        "PIP_SOURCE",
        "PIP_SOURCE_DIR",
        "PIP_SOURCE_DIRECTORY",
        "PIP_DEST",
        "PIP_DESTINATION_DIR",
        "PIP_DESTINATION_DIRECTORY",
        "PIP_REQUIREMENTS_FROM_SCRIPT",
        "PIP_REPORT",
        "PIPX_COMPLETION_DIR",
        "PIPX_GLOBAL_COMPLETION_DIR",
        "XDG_STATE_HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SYSTEMDRIVE",
        "UV_PYTHON_DOWNLOADS_JSON_URL",
        "PIP_BUILD_TRACKER",
        "PIP_DOWNLOAD_DIR",
        "PIP_WHEEL_DIR",
        "PIP_REQUIREMENT",
        "PIP_CONSTRAINT",
        "PIP_BUILD_CONSTRAINT",
        "UV_CREDENTIALS_DIR",
        "UV_INSTALL_DIR",
        "UV_UNMANAGED_INSTALL",
        "UV_PYTHON_BIN_DIR",
        "UV_PYTHON_CACHE_DIR",
        "UV_ENV_FILE",
        "UV_BUILD_CONSTRAINT",
        "UV_CONSTRAINT",
        "UV_OVERRIDE",
        "UV_EXCLUDE",
        "PIPX_MAN_DIR",
        "PIPX_SHARED_LIBS",
        "PIPX_GLOBAL_HOME",
        "PIPX_GLOBAL_BIN_DIR",
        "PIPX_GLOBAL_MAN_DIR",
        "POETRY_CACHE_DIR",
        "POETRY_DATA_DIR",
        "POETRY_HOME",
        "POETRY_PYTHON_INSTALLATION_DIR",
        "TRACING_DURATIONS_FILE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "XDG_BIN_HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CONFIG_DIRS",
        "XDG_DATA_HOME",
        "APPDATA",
        "LOCALAPPDATA",
        "USERPROFILE",
        "NPM_CONFIG_USERCONFIG",
        "NPM_CONFIG_GLOBALCONFIG",
        "NPM_CONFIG_LOGS_DIR",
    ],
)
def test_package_destination_environment_is_gated(name: str) -> None:
    analysis = analyze_shell_command(f"{name}=../outside uv sync")
    assert analysis.requires_approval
    assert f"dangerous environment assignment ({name})" in analysis.approval_reasons


@pytest.mark.parametrize("name", ["PIP_DISABLE_PIP_VERSION_CHECK", "UV_NO_SYNC"])
def test_benign_package_environment_is_not_gated(name: str) -> None:
    assert not analyze_shell_command(f"{name}=1 uv sync").requires_approval


def test_simple_command_is_extracted_without_approval() -> None:
    analysis = analyze_shell_command("echo hello")

    assert analysis.command_parts == ("echo hello",)
    assert not analysis.requires_approval


def test_semicolon_separated_commands_are_extracted_separately() -> None:
    analysis = analyze_shell_command("echo a; echo b")

    assert analysis.command_parts == ("echo a", "echo b")
    assert not analysis.requires_approval


def test_line_continuations_are_joined() -> None:
    analysis = analyze_shell_command("echo hi \\\nthere")

    assert analysis.command_parts == ("echo hi there",)
    assert not analysis.requires_approval


@pytest.mark.parametrize("command", ["echo $(whoami)", "echo `whoami`"])
def test_command_substitution_requires_approval(command: str) -> None:
    analysis = analyze_shell_command(command)

    assert analysis.requires_approval
    assert "command substitution" in analysis.approval_reasons


def test_substituted_command_is_extracted_for_inspection() -> None:
    analysis = analyze_shell_command("echo $(whoami)")

    assert "whoami" in analysis.command_parts


def test_process_substitution_requires_approval() -> None:
    analysis = analyze_shell_command("cat <(ls)")

    assert analysis.requires_approval
    assert "process substitution" in analysis.approval_reasons


def test_variable_expansion_requires_approval() -> None:
    analysis = analyze_shell_command("echo $HOME")

    assert analysis.requires_approval
    assert (
        "variable expansion (unsupported in v0.1 because it can change arguments)"
        in analysis.approval_reasons
    )


def test_parameter_expansion_requires_approval() -> None:
    analysis = analyze_shell_command("echo ${HOME}")

    assert analysis.requires_approval
    assert "parameter expansion" in analysis.approval_reasons


def test_arithmetic_expansion_requires_approval() -> None:
    analysis = analyze_shell_command("echo $((1+1))")

    assert analysis.requires_approval
    assert "arithmetic expansion" in analysis.approval_reasons


def test_brace_expansion_requires_approval() -> None:
    analysis = analyze_shell_command("echo {a,b}")

    assert analysis.requires_approval
    assert "brace expansion" in analysis.approval_reasons


def test_ansi_c_string_requires_approval_but_preserves_token() -> None:
    analysis = analyze_shell_command("echo $'x'")

    assert analysis.requires_approval
    assert "ANSI-C quoted arguments" in analysis.approval_reasons
    # The raw token stays visible so guardrails such as find's execution
    # predicate still see it.
    assert analysis.command_parts == ("echo $'x'",)


def test_heredoc_requires_approval() -> None:
    analysis = analyze_shell_command("cat <<EOF\nhi\nEOF")

    assert analysis.requires_approval
    assert "heredocs are unsupported in v0.1" in analysis.approval_reasons


def test_herestring_requires_approval() -> None:
    analysis = analyze_shell_command("cat <<<x")

    assert analysis.requires_approval
    assert "redirection" in analysis.approval_reasons


def test_background_execution_requires_approval() -> None:
    analysis = analyze_shell_command("sleep 1 &")

    assert analysis.requires_approval
    assert "background execution" in analysis.approval_reasons


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("if true; then ls; fi", "if statement"),
        ("for i in 1 2; do echo $i; done", "for loop"),
        ("while true; do sleep 1; done", "while loop"),
        ("case x in y) ls;; esac", "case statement"),
        ("(ls)", "subshell"),
        ("f() { ls; }", "function definition"),
        ("[ -f x ]", "test expression"),
    ],
)
def test_compound_constructs_require_approval(command: str, reason: str) -> None:
    analysis = analyze_shell_command(command)

    assert analysis.requires_approval
    assert reason in analysis.approval_reasons


def test_benign_assignment_prefix_is_allowed() -> None:
    analysis = analyze_shell_command("FOO=bar ls")

    assert analysis.command_parts == ("ls",)
    assert not analysis.requires_approval


@pytest.mark.parametrize(
    "name", ["LD_PRELOAD", "LD_LIBRARY_PATH", "PATH", "HOME", "IFS", "ENV", "BASH_ENV"]
)
def test_dangerous_environment_names_require_approval(name: str) -> None:
    analysis = analyze_shell_command(f"{name}=x ls")

    assert analysis.requires_approval
    assert f"dangerous environment assignment ({name})" in analysis.approval_reasons


@pytest.mark.parametrize("name", ["GIT_EDITOR", "GIT_PAGER", "DYLD_INSERT_LIBRARIES"])
def test_dangerous_environment_prefixes_require_approval(name: str) -> None:
    analysis = analyze_shell_command(f"{name}=x ls")

    assert analysis.requires_approval
    assert f"dangerous environment assignment ({name})" in analysis.approval_reasons


@pytest.mark.parametrize("value", ["/abs/path", "relative:/abs/path", "..", "a/..", ""])
def test_pythonpath_dangerous_entries_require_approval(value: str) -> None:
    analysis = analyze_shell_command(f"PYTHONPATH={value} python x")

    assert analysis.requires_approval
    assert "dangerous environment assignment (PYTHONPATH)" in analysis.approval_reasons


@pytest.mark.parametrize(
    "form",
    [
        "{assignment} npm test",
        "env {assignment} npm test",
        "export {assignment}; npm test",
    ],
)
@pytest.mark.parametrize("name", ["LESSOPEN", "LESSCLOSE", "LESSEDIT", "KSH_ENV"])
def test_executable_environment_forms_are_gated(form: str, name: str) -> None:
    analysis = analyze_shell_command(
        form.format(assignment=f"{name}='|cat /outside/data %s'")
    )
    assert f"dangerous environment assignment ({name})" in analysis.approval_reasons


@pytest.mark.parametrize(
    "form",
    [
        "{assignment} npm test",
        "env {assignment} npm test",
        "export {assignment}; npm test",
    ],
)
@pytest.mark.parametrize("value", [r"\|cat", r"\`cat\`", r"\$\(cat\)"])
def test_unquoted_assignment_execution_syntax_is_gated(form: str, value: str) -> None:
    analysis = analyze_shell_command(form.format(assignment=f"FOO={value}"))
    assert (
        "dangerous environment assignment (FOO): command-execution syntax"
        in analysis.approval_reasons
    )


@pytest.mark.parametrize("form", ["{assignment} npm test", "env {assignment} npm test"])
@pytest.mark.parametrize("value", ["'|cat'", "'`cat`'", "'$(cat)'"])
def test_quoted_non_executable_environment_is_literal(form: str, value: str) -> None:
    analysis = analyze_shell_command(form.format(assignment=f"FOO={value}"))
    assert not analysis.requires_approval


def test_pythonpath_relative_entries_are_allowed() -> None:
    analysis = analyze_shell_command("PYTHONPATH=lib:vendor python x")

    assert not analysis.requires_approval


def test_lowercase_assignment_name_requires_approval() -> None:
    analysis = analyze_shell_command("foo=bar ls")

    assert analysis.requires_approval
    assert (
        "an environment assignment with a non-uppercase name"
        in analysis.approval_reasons
    )


def test_non_literal_assignment_value_requires_approval() -> None:
    analysis = analyze_shell_command("FOO=$BAR ls")

    assert analysis.requires_approval
    assert "a non-literal environment assignment" in analysis.approval_reasons


def test_assignment_only_statement_is_denied() -> None:
    analysis = analyze_shell_command("FOO=bar")

    assert analysis.requires_approval
    assert "assignment-only statements are not permitted" in analysis.approval_reasons


@pytest.mark.parametrize(
    "command",
    [
        "echo hi > out.txt",
        "echo hi >> out.txt",
        "echo hi 2> err.txt",
        "echo hi 2>> err.txt",
        "echo hi 2>&1",
    ],
)
def test_relative_and_stderr_redirects_are_allowed(command: str) -> None:
    analysis = analyze_shell_command(command)

    assert not analysis.requires_approval
    assert analysis.command_parts == ("echo hi <redirect>",)


def test_absolute_redirect_target_requires_approval() -> None:
    analysis = analyze_shell_command("echo hi > /abs/path")

    assert analysis.requires_approval
    assert "an absolute redirection target" in analysis.approval_reasons


def test_non_literal_redirect_target_requires_approval() -> None:
    analysis = analyze_shell_command("echo hi > $TMP/x")

    assert analysis.requires_approval
    assert "a non-literal redirection target" in analysis.approval_reasons


@pytest.mark.parametrize("target", ["-weird", "123"])
def test_invalid_redirect_targets_require_approval(target: str) -> None:
    analysis = analyze_shell_command(f"echo hi > {target}")

    assert analysis.requires_approval
    assert "an invalid redirection path target" in analysis.approval_reasons


@pytest.mark.parametrize(
    "command",
    [
        "echo hi < input.txt",
        "echo hi >&2",
        "echo hi >&file",
        "echo hi &> file",
        "echo hi >| file",
    ],
)
def test_unsupported_redirect_operators_require_approval(command: str) -> None:
    analysis = analyze_shell_command(command)

    assert analysis.requires_approval
    assert (
        "unsupported redirection (only >, >>, 2>, and 2>&1 are allowed)"
        in analysis.approval_reasons
    )


def test_redirect_only_statement_is_denied() -> None:
    analysis = analyze_shell_command("> out.txt")

    assert analysis.requires_approval
    assert "redirect-only statements are not permitted" in analysis.approval_reasons


@pytest.mark.parametrize(
    "command",
    [
        "cd /tmp > log",
        "cd /tmp && echo hi > log",
        "cd sub && echo hi > log",
        "echo hi > log && cd sub",
        "cd sub && echo hi 2>&1 | head",
        "cd sub && echo hi > /dev/null",
    ],
)
def test_redirect_in_cd_chain_is_not_vetoed(command: str) -> None:
    # The order-blind cd+redirect veto is gone (ADR 0021): the analysis no
    # longer denies a chain merely because a cd and a redirect coexist. The
    # redirect target is attributed to its occurrence position instead, and
    # the resolver resolves it against the tracked working directory there.
    analysis = analyze_shell_command(command)

    assert not analysis.requires_approval
    assert not any(
        "redirection in a command chain" in reason
        for reason in analysis.approval_reasons
    )


@pytest.mark.parametrize(
    "command,expected_parts,expected_operands",
    [
        # A redirect attached to the cd command resolves against the pre-cd
        # working directory, so its operand precedes the cd part.
        ("cd sub > log", ("cd sub <redirect>",), (("log",), ())),
        # A redirect after the cd belongs to the later command.
        ("cd sub && echo hi > log", ("cd sub", "echo hi"), ((), ("log",), ())),
        # A redirect before the cd belongs to the earlier command.
        (
            "echo hi > log && cd sub",
            ("echo hi <redirect>", "cd sub"),
            (("log",), (), ()),
        ),
        # A redirect-only statement anchors to the next command.
        ("> log; echo hi", ("echo hi",), (("log",), ())),
        # A redirect-only statement after the last command anchors to the end.
        ("echo hi; > log", ("echo hi",), ((), ("log",))),
        # fd duplication and device sinks are not filesystem targets.
        ("echo hi 2>&1", ("echo hi <redirect>",), ((), ())),
        ("echo hi > /dev/null", ("echo hi <redirect>",), (("/dev/null",), ())),
    ],
)
def test_redirect_operands_are_attributed_to_occurrence_position(
    command: str,
    expected_parts: tuple[str, ...],
    expected_operands: tuple[tuple[str, ...], ...],
) -> None:
    analysis = analyze_shell_command(command)

    assert analysis.command_parts == expected_parts
    assert analysis.redirect_operands == expected_operands
    assert len(analysis.redirect_operands) == len(expected_parts) + 1


@pytest.mark.parametrize("directory_command", ["pushd ..", "popd", "popd +1"])
def test_redirect_in_directory_stack_chain_is_not_vetoed(
    directory_command: str,
) -> None:
    # The stack-chain veto is gone with the cd-chain veto (ADR 0021); the
    # resolver's tracked-cwd machinery governs the redirect target instead.
    analysis = analyze_shell_command(f"{directory_command}; printf hi > file")

    assert not any(
        "redirection in a command chain" in reason
        for reason in analysis.approval_reasons
    )
    assert analysis.command_parts == (directory_command, "printf hi <redirect>")
    assert analysis.redirect_operands == ((), ("file",), ())


@pytest.mark.parametrize("target", ["/dev/null", "/dev/stdout", "/dev/stderr"])
def test_device_sink_redirects_are_not_file_redirects(target: str) -> None:
    # Device sinks never open a file, so they are exempt from the absolute
    # redirect-target rule without any string rewrite (ADR 0021).
    analysis = analyze_shell_command(f"echo hi > {target}")

    assert not analysis.requires_approval
    assert analysis.command_parts == ("echo hi <redirect>",)


def test_absolute_redirect_target_inside_scratchpad_root_is_allowed(tmp_path) -> None:
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    roots = frozenset({scratchpad.resolve()})

    analysis = analyze_shell_command(
        f"echo hi > {scratchpad}/log.txt", scratchpad_roots=roots
    )

    assert not analysis.requires_approval
    assert analysis.command_parts == ("echo hi <redirect>",)


@pytest.mark.parametrize(
    "relative",
    [
        # A sibling directory that merely spells like the scratchpad root.
        "scratch-sibling/log.txt",
        # A path that escapes the root through a parent component.
        "scratch/../outside.log",
        # A symlink inside the root that points outside it.
        "scratch/escape/log.txt",
    ],
)
def test_absolute_redirect_target_outside_scratchpad_roots_is_denied(
    tmp_path, relative: str
) -> None:
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    (scratchpad / "escape").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "scratch-sibling").mkdir()
    roots = frozenset({scratchpad.resolve()})

    analysis = analyze_shell_command(
        f"echo hi > {tmp_path / relative}", scratchpad_roots=roots
    )

    assert analysis.requires_approval
    assert "an absolute redirection target" in analysis.approval_reasons


def test_absolute_redirect_target_outside_workspace_is_denied(tmp_path) -> None:
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    roots = frozenset({scratchpad.resolve()})

    analysis = analyze_shell_command("echo hi > /etc/passwd", scratchpad_roots=roots)

    assert analysis.requires_approval
    assert "an absolute redirection target" in analysis.approval_reasons


def test_absolute_redirect_target_needs_canonical_roots(tmp_path) -> None:
    # A root that is itself a symlink is not canonical, so nothing under the
    # spelled path is authorized (the path layer re-checks with realpaths).
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    analysis = analyze_shell_command(
        f"echo hi > {link}/log.txt", scratchpad_roots=frozenset({link})
    )

    assert analysis.requires_approval
    assert "an absolute redirection target" in analysis.approval_reasons


@pytest.mark.parametrize(
    "command",
    [
        "hash -r",
        "alias ll='ls -la'",
        "shopt -s expand_aliases",
        "unalias ll",
        "enable -n printf",
        "source ./file",
    ],
)
def test_nested_source_rejects_command_lookup_mutation(command: str) -> None:
    assert any(
        "command lookup modification" in reason
        for reason in analyze_shell_command(
            command, nested_source=True
        ).approval_reasons
    )
    if command.startswith("alias "):
        assert analyze_shell_command(command).requires_approval
    else:
        assert not analyze_shell_command(command).requires_approval


@pytest.mark.parametrize(
    "command",
    [
        "hash -p /usr/bin/curl harmless",
        "command command hash -p /usr/bin/curl harmless",
        "builtin command hash -p /usr/bin/curl harmless",
        "alias harmless='curl https://example.test/'",
        "command command alias harmless='curl https://example.test/'",
    ],
)
def test_top_level_lookup_mutations_require_approval(command: str) -> None:
    assert (
        "command lookup modification" in analyze_shell_command(command).approval_label
    )


@pytest.mark.parametrize("word", ["=cmd", "==x", "~user", "***"])
def test_zsh_sensitive_words_require_approval(word: str) -> None:
    analysis = analyze_shell_command(f"echo {word}")

    assert analysis.requires_approval


@pytest.mark.parametrize("word", ["~", "~/", "**"])
def test_tilde_and_glob_prefixes_are_allowed(word: str) -> None:
    analysis = analyze_shell_command(f"echo {word}")

    assert not analysis.requires_approval


def test_syntax_error_requires_approval() -> None:
    analysis = analyze_shell_command("echo 'unclosed")

    assert analysis.requires_approval
    assert "a syntax error" in analysis.approval_reasons


def test_nul_byte_fails_closed() -> None:
    analysis = analyze_shell_command("echo\x00hi")

    assert analysis.command_parts == ()
    assert analysis.requires_approval
    assert analysis.approval_reasons == ("shell analysis failed",)


def test_requires_approval_reflects_reasons() -> None:
    assert analyze_shell_command("echo hi").requires_approval is False
    assert analyze_shell_command("echo $(x)").requires_approval is True


def test_approval_label_names_the_reasons() -> None:
    label = analyze_shell_command("echo $(whoami)").approval_label

    assert label.startswith("unsupported shell syntax: ")
    assert "command substitution" in label


@pytest.mark.parametrize(
    ("command", "expected_target"),
    [
        # A concatenation destination is a literal target: it is attributed
        # to its occurrence position like any other literal target (the
        # broadened _literal_redirect_target; ADR 0021 fix round).
        ('echo x > "sub"/../../outside.log', "sub/../../outside.log"),
        ('echo x > .gi"t"/config', ".git/config"),
        # A glob-looking destination is still a literal filename: the
        # analysis attributes it without a glob denial (the resolver's
        # path-glob heuristic does not fire for redirect destinations,
        # because bash never pathname-expands them).
        ("echo x > f[0]", "f[0]"),
        ('echo x > f"*"', "f*"),
        ("echo x > out[1].log", "out[1].log"),
    ],
)
def test_literal_redirect_targets_include_concatenations_and_glob_lookalikes(
    command: str, expected_target: str
) -> None:
    analysis = analyze_shell_command(command)

    assert not analysis.requires_approval
    assert analysis.command_parts == ("echo x <redirect>",)
    assert analysis.redirect_operands == ((expected_target,), ())


@pytest.mark.parametrize(
    ("command", "expected_part"),
    [
        # Script-loading builtins are ordinary commands to the analysis;
        # their operands are path-inspected by the resolver (ADR 0021 fix
        # round).
        ("source sub/setup.sh", "source sub/setup.sh"),
        (". ./lib.sh", ". ./lib.sh"),
        ("source sub/setup.sh > out", "source sub/setup.sh <redirect>"),
    ],
)
def test_source_commands_are_extracted_as_parts(
    command: str, expected_part: str
) -> None:
    analysis = analyze_shell_command(command)

    assert not analysis.requires_approval
    assert analysis.command_parts == (expected_part,)


@pytest.mark.parametrize(
    ("command", "reason", "alternative"),
    [
        (
            "echo $(whoami)",
            "command substitution",
            "use read_file with offset and limit to read files; "
            "use grep to locate content and line numbers; "
            "use the edit tool to modify files",
        ),
        (
            "cat <<EOF\nhi\nEOF",
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
            "echo =cmd",
            "Zsh equals expansion",
            "drop the leading equals sign and invoke the command by its "
            "name or absolute path",
        ),
        (
            "echo $HOME",
            "variable expansion (unsupported in v0.1 because it can change arguments)",
            "write the literal value directly as the argument",
        ),
        (
            "echo $((1+1))",
            "arithmetic expansion",
            "compute the value and write the literal result as the argument",
        ),
        (
            "echo {a,b}",
            "brace expansion",
            "write each expanded value literally as its own argument",
        ),
        (
            "cat <(ls)",
            "process substitution",
            "write the command output to a file first, then pass the file path",
        ),
        (
            "sleep 1 &",
            "background execution",
            "run the command in the foreground, or use bash_start for a "
            "long-running job",
        ),
        (
            "echo $'x'",
            "ANSI-C quoted arguments",
            "write the argument as an ordinary quoted string",
        ),
        (
            "echo 'unclosed",
            "a syntax error",
            "write the command as well-formed shell syntax",
        ),
        (
            "uv run pip install fixture[extra]",
            "unquoted package bracket syntax may expand as a shell glob",
            "quote the package argument with single quotes so the shell "
            "cannot expand it",
        ),
        (
            "echo hi < input.txt",
            "unsupported redirection (only >, >>, 2>, and 2>&1 are allowed)",
            "read the file with read_file, or pass its content as a literal argument",
        ),
        (
            "echo hi >&2",
            "unsupported redirection (only >, >>, 2>, and 2>&1 are allowed)",
            "use an allowed redirection form instead: plain output, append, "
            "stderr, or combined stderr to stdout",
        ),
        (
            "> out.txt",
            "redirect-only statements are not permitted",
            "attach the redirection to a command, or use write_file or the "
            "edit tool to create or change the file",
        ),
        (
            "FOO=bar",
            "assignment-only statements are not permitted",
            "attach the assignment to a command in the same statement",
        ),
    ],
)
def test_syntax_reasons_carry_class_and_alternative(
    command: str, reason: str, alternative: str
) -> None:
    # A SYNTAX denial is an unmodeled v0.1 construct: the diagnostic carries
    # the SYNTAX class and the compliant alternative its message names.
    analysis = analyze_shell_command(command)
    diagnostic = next(item for item in analysis.diagnostics if item.detail == reason)
    assert diagnostic.denial_class is DenialClass.SYNTAX
    assert diagnostic.alternative == alternative
    assert analysis.approval_diagnostic.denial_class is DenialClass.SYNTAX
    rendered = render_diagnostic(analysis.approval_diagnostic)
    assert "Instead," in rendered
    assert alternative in rendered


@pytest.mark.parametrize(
    ("command", "reason", "fragments"),
    [
        (
            "PYTHONPATH=/abs python x",
            "dangerous environment assignment (PYTHONPATH)",
            ("dangerous environment assignment", "PYTHONPATH"),
        ),
        (
            "LD_PRELOAD=x ls",
            "dangerous environment assignment (LD_PRELOAD)",
            ("dangerous environment assignment", "PRELOAD"),
        ),
        (
            "foo=bar ls",
            "an environment assignment with a non-uppercase name",
            ("an environment assignment with a non-uppercase name",),
        ),
        (
            "echo hi > -weird",
            "an invalid redirection path target",
            ("an invalid redirection path target",),
        ),
        (
            "echo hi > /abs/path",
            "an absolute redirection target",
            ("an absolute redirection target",),
        ),
        (
            "unset PATH",
            "unsetting shell environment variables cannot be safely inspected",
            ("unsetting shell environment variables cannot be safely inspected",),
        ),
        (
            "hash -p /usr/bin/curl harmless",
            "unsupported command lookup modification (hash)",
            ("unsupported command lookup modification", "hash"),
        ),
    ],
)
def test_policy_reasons_carry_policy_class_and_no_alternative(
    command: str, reason: str, fragments: tuple[str, ...]
) -> None:
    # A POLICY denial is a deliberate security decision: the diagnostic
    # carries the POLICY class and no alternative.
    analysis = analyze_shell_command(command)
    diagnostic = next(item for item in analysis.diagnostics if item.detail == reason)
    assert diagnostic.denial_class is DenialClass.POLICY
    assert diagnostic.alternative is None
    assert analysis.approval_diagnostic.denial_class is DenialClass.POLICY
    rendered = render_diagnostic(analysis.approval_diagnostic)
    # The rendered detail escapes markup characters, so the requirement is
    # asserted on escape-free fragments.
    for fragment in fragments:
        assert fragment in rendered
    assert "Instead," not in rendered


def test_mixed_constructs_classify_as_policy() -> None:
    # A command with both unmodeled syntax and a policy violation is a
    # policy denial: it names no alternative.
    analysis = analyze_shell_command("PYTHONPATH=/abs cat $(ls)")
    assert "command substitution" in analysis.approval_reasons
    assert "dangerous environment assignment (PYTHONPATH)" in analysis.approval_reasons
    assert analysis.approval_diagnostic.denial_class is DenialClass.POLICY
    rendered = render_diagnostic(analysis.approval_diagnostic)
    assert "command substitution" in rendered
    assert "dangerous environment assignment" in rendered
    assert "PYTHONPATH" in rendered
    assert "Instead," not in rendered


def test_unknown_and_failed_analysis_classifies_as_policy() -> None:
    # Unknown and uncategorized denials fail closed to POLICY.
    analysis = analyze_shell_command("echo\x00hi")
    assert analysis.approval_reasons == ("shell analysis failed",)
    assert analysis.approval_diagnostic.denial_class is DenialClass.POLICY
    rendered = render_diagnostic(analysis.approval_diagnostic)
    assert rendered == "Command denied: shell analysis failed"
    assert "Instead," not in rendered


def test_diagnostics_keep_their_reasons_class_and_alternative() -> None:
    analysis = analyze_shell_command("echo $(whoami)")
    assert tuple(item.detail for item in analysis.diagnostics) == (
        analysis.approval_reasons
    )
    (diagnostic,) = analysis.diagnostics
    assert diagnostic.detail == "command substitution"
    assert diagnostic.denial_class is DenialClass.SYNTAX
    assert diagnostic.alternative is not None
    assert analysis.approval_diagnostic.related == analysis.diagnostics
    assert analysis.approval_diagnostic.denial_class is DenialClass.SYNTAX


@pytest.mark.parametrize(
    ("argv", "reason"),
    [
        (
            ["env", "PATH=/outside", "echo", "hi"],
            "dangerous environment assignment (PATH)",
        ),
        (
            ["env", "PYTHONPATH=../outside", "python", "-c", "code"],
            "dangerous environment assignment (PYTHONPATH)",
        ),
        (
            ["env", "foo=bar", "echo", "hi"],
            "an environment assignment with a non-uppercase name",
        ),
        (
            ["hash", "-p", "./fake", "git"],
            "unsupported command lookup modification (hash)",
        ),
        (["alias", "git=./fake"], "unsupported command lookup modification (alias)"),
    ],
)
def test_argv_permission_diagnostics_are_policy_class(
    argv: list[str], reason: str
) -> None:
    # Environment-assignment and command-lookup findings are policy
    # decisions: classed POLICY, no alternative, and the string view of the
    # reasons is unchanged.
    diagnostics = argv_permission_diagnostics(argv)
    diagnostic = next(item for item in diagnostics if item.detail == reason)
    assert diagnostic.denial_class is DenialClass.POLICY
    assert diagnostic.alternative is None
    assert reason in argv_permission_reasons(argv)
    assert argv_permission_reasons(["echo", *argv]) == ()


# --- tier-2 static expansion wiring -------------------------------------------

_LocateFile = "lines.txt"


@pytest.fixture
def locate_workspace(tmp_path, monkeypatch) -> None:
    """A cwd holding the file the locate-then-print idiom greps."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / _LocateFile).write_text(
        "alpha\nlit\nbeta\n" + "\n".join(f"line{i}" for i in range(4, 60)) + "\n"
    )


@pytest.mark.parametrize(
    ("command", "expanded"),
    [
        (
            "sed -n \"$(grep -n 'lit' lines.txt | head -1 | cut -d: -f1),+45p\" lines.txt",
            "sed -n 2,+45p lines.txt",
        ),
        (
            "sed -n \"$(grep -n 'lit' lines.txt | head -3 | cut -d: -f1),+45p\" lines.txt",
            "sed -n 2,+45p lines.txt",
        ),
        (
            "sed -n \"$(grep -n 'lit' lines.txt | cut -d: -f1),+45p\" lines.txt",
            "sed -n 2,+45p lines.txt",
        ),
        (
            "sed -n \"$(grep -n 'lit' lines.txt | tail -1 | cut -d: -f1),87p\" lines.txt",
            "sed -n 2,87p lines.txt",
        ),
        (
            "sed -n \"$(grep -n 'lit' lines.txt | head -1 | cut -d: -f1)p\" lines.txt",
            "sed -n 2p lines.txt",
        ),
        # A no-match expansion degrades to the suffix alone: `p` prints the
        # whole file (benign); a `,+Np` suffix is a sed runtime error.
        (
            "sed -n \"$(grep -n 'absent' lines.txt | head -1 | cut -d: -f1)p\" lines.txt",
            "sed -n p lines.txt",
        ),
    ],
)
def test_sed_locate_print_idiom_expands_to_literal_script(
    locate_workspace, command: str, expanded: str
) -> None:
    analysis = analyze_shell_command(command)

    assert not analysis.requires_approval
    assert analysis.command_parts[0] == expanded
    # The whitelisted inner pipeline stages stay visible as command parts so
    # their read operands remain path-checked by the resolver.
    assert any(part.startswith("grep -n '") for part in analysis.command_parts)
    assert "cut -d: -f1" in analysis.command_parts


def test_sed_locate_print_multi_match_is_rejected_as_unexecutable(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / _LocateFile).write_text("lit\nalpha\nlit\n")

    analysis = analyze_shell_command(
        "sed -n \"$(grep -n 'lit' lines.txt | cut -d: -f1),+45p\" lines.txt"
    )

    assert analysis.requires_approval
    diagnostic = next(
        item
        for item in analysis.diagnostics
        if item.detail == "a sed locate-then-print expansion locating multiple lines"
    )
    assert diagnostic.denial_class is DenialClass.SYNTAX
    assert diagnostic.alternative == (
        "narrow the grep pattern or select one located line (for example with "
        "head -1) so the sed script has a single numeric address"
    )
    assert analysis.approval_diagnostic.denial_class is DenialClass.SYNTAX


@pytest.mark.parametrize(
    "command",
    [
        # head without the numeric-extraction stage: raw file content would
        # reach the sed script (probed write vector).
        "sed -n \"$(grep -n 'lit' lines.txt | head -1),+45p\" lines.txt",
        # Nested substitution inside the construct.
        'sed -n "$(grep -n "$(id)" lines.txt | head -1 | cut -d: -f1),+45p" lines.txt',
        # A redirect inside the substitution.
        "sed -n \"$(grep -n 'lit' lines.txt > out.txt | head -1 | cut -d: -f1)p\" lines.txt",
        # A non-literal (unquoted, pattern-shaped) grep argument.
        'sed -n "$(grep -n lit lines.txt | head -1 | cut -d: -f1)p" lines.txt',
        # A prefix assignment keeps the invocation text unrecognizable.
        "X=1 sed -n \"$(grep -n 'lit' lines.txt | head -1 | cut -d: -f1)p\" lines.txt",
    ],
)
def test_sed_locate_print_variants_deny_unchanged(
    locate_workspace, command: str
) -> None:
    analysis = analyze_shell_command(command)

    assert analysis.requires_approval
    assert "command substitution" in analysis.approval_reasons
    assert analysis.approval_diagnostic.denial_class is DenialClass.SYNTAX
    rendered = render_diagnostic(analysis.approval_diagnostic)
    assert "Instead," in rendered


def test_arithmetic_in_sed_script_stays_denied(locate_workspace) -> None:
    analysis = analyze_shell_command('sed -n "$((1+1))p" lines.txt')

    assert analysis.requires_approval
    assert "arithmetic expansion" in analysis.approval_reasons
    assert analysis.approval_diagnostic.denial_class is DenialClass.SYNTAX


@pytest.fixture
def equals_path(tmp_path, monkeypatch) -> str:
    directory = tmp_path / "bin"
    directory.mkdir()
    tool = directory / "mytool"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory))
    return str(directory)


def test_equals_command_position_resolves_to_executable_path(equals_path: str) -> None:
    analysis = analyze_shell_command("=mytool -la")

    assert not analysis.requires_approval
    # The resolved executable's basename is inlined: the policy keys every
    # command surface by name, and zsh's =name runs the same PATH lookup as
    # the plain name, so the authorization is exactly the hand-written form.
    assert analysis.command_parts == ("mytool -la",)


def test_equals_unresolvable_command_keeps_its_denial(equals_path: str) -> None:
    analysis = analyze_shell_command("=notacommand -la")

    assert analysis.requires_approval
    assert analysis.approval_reasons == ("Zsh equals expansion",)
    diagnostic = analysis.diagnostics[0]
    assert diagnostic.denial_class is DenialClass.SYNTAX
    assert diagnostic.alternative == (
        "drop the leading equals sign and invoke the command by its name or "
        "absolute path"
    )


def test_equals_in_argument_position_stays_denied(equals_path: str) -> None:
    analysis = analyze_shell_command("echo =mytool")

    assert analysis.requires_approval
    assert analysis.approval_reasons == ("Zsh equals expansion",)


@pytest.mark.parametrize(
    ("command", "expanded"),
    [
        ("X=5; grep -m $X lit lines.txt", "grep -m 5 lit lines.txt"),
        ("X=lit; grep $X lines.txt", "grep lit lines.txt"),
        ("X=5; cat $X", "cat 5"),
        ("N=2; head -n $((N+1)) lines.txt", "head -n 3 lines.txt"),
        ("grep -m $((2+3)) lit lines.txt", "grep -m 5 lit lines.txt"),
        ("X=5; head -n$X lines.txt", "head -n5 lines.txt"),
        ("X=5; cut -d $X -f1 lines.txt", "cut -d 5 -f1 lines.txt"),
        # Multiple literal assignments bind in one statement.
        ("X=5 Y=3; grep -m $((X+Y)) lit lines.txt", "grep -m 8 lit lines.txt"),
        # A binding persists for later statements of the same invocation
        # (the shell keeps the assignment); a later assignment rebinds.
        (
            "X=5; grep -m $X lit lines.txt; head -n $X lines.txt",
            "grep -m 5 lit lines.txt",
        ),
    ],
)
def test_read_only_positions_inline_the_resolved_literal(
    locate_workspace, command: str, expanded: str
) -> None:
    analysis = analyze_shell_command(command)

    assert not analysis.requires_approval
    assert analysis.command_parts[0] == expanded


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        # Write positions reject everything: every sed position is not
        # read-only, so the expansion stays unmodeled.
        ("N=2; sed -n $((N))p lines.txt", "arithmetic expansion"),
        (
            "X=5; sed -n $X lines.txt",
            "variable expansion (unsupported in v0.1 because it can change arguments)",
        ),
        # A command-prefix assignment never binds same-line expansions:
        # every POSIX shell expands argument words before the command's own
        # assignments take effect (probed in zsh, bash, and sh), so the
        # variable is unknown to the analysis exactly as it is to the shell.
        (
            "X=5 grep -m $X lit lines.txt",
            "variable expansion (unsupported in v0.1 because it can change arguments)",
        ),
        # An assignment guarded by && / || may be skipped at runtime, so it
        # is not a binding source.
        (
            "X=5 && grep -m $X lit lines.txt",
            "variable expansion (unsupported in v0.1 because it can change arguments)",
        ),
        # An unknown variable is not resolvable.
        (
            "grep -m $X lit lines.txt",
            "variable expansion (unsupported in v0.1 because it can change arguments)",
        ),
        # Quoted expansion text is not an unquoted word.
        (
            'X=5; grep -m "$X" lit lines.txt',
            "variable expansion (unsupported in v0.1 because it can change arguments)",
        ),
        # A boolean option before the expansion: the position is ambiguous
        # (option-arg:-c is unclassified), so it fails closed.
        (
            "X=5; grep -c $X lit lines.txt",
            "variable expansion (unsupported in v0.1 because it can change arguments)",
        ),
        # A value with shell metacharacters is inert, not inlinable.
        (
            "X='a;b'; grep -m $X lit lines.txt",
            "variable expansion (unsupported in v0.1 because it can change arguments)",
        ),
    ],
)
def test_unresolvable_or_write_position_expansions_deny_unchanged(
    locate_workspace, command: str, reason: str
) -> None:
    analysis = analyze_shell_command(command)

    assert analysis.requires_approval
    assert reason in analysis.approval_reasons
    assert analysis.approval_diagnostic.denial_class is DenialClass.SYNTAX


def test_for_loop_body_expansion_stays_denied(locate_workspace) -> None:
    analysis = analyze_shell_command(
        "for i in 1; do X=5 grep -m $X lit lines.txt; done"
    )

    assert analysis.requires_approval
    assert "for loop" in analysis.approval_reasons


def test_dangerous_assignment_wins_over_expansion_resolution(locate_workspace) -> None:
    analysis = analyze_shell_command("X=5; PYTHONPATH=/abs grep -m $X lit lines.txt")

    assert analysis.requires_approval
    assert "dangerous environment assignment (PYTHONPATH)" in analysis.approval_reasons
    assert analysis.approval_diagnostic.denial_class is DenialClass.POLICY
    rendered = render_diagnostic(analysis.approval_diagnostic)
    assert "Instead," not in rendered


def test_approval_label_reflects_the_policy_class() -> None:
    syntax = analyze_shell_command("echo $(whoami)").approval_label
    policy = analyze_shell_command("PYTHONPATH=/abs python x").approval_label

    assert syntax.startswith("unsupported shell syntax: ")
    assert policy.startswith("shell policy restriction: ")
    assert "PYTHONPATH" in policy


def test_unrecognized_idiom_variants_are_drift_logged(locate_workspace, caplog) -> None:
    with caplog.at_level(
        "DEBUG", logger="chartreux.core.tools.builtins._shell_permission_analysis"
    ):
        analyze_shell_command(
            "sed -n \"$(grep -n 'lit' lines.txt | head -1),+45p\" lines.txt"
        )
        analyze_shell_command("=notacommand-xyz -la")
        analyze_shell_command("echo $HOME")

    messages = caplog.messages
    assert any("sed locate-then-print" in message for message in messages)
    assert any("zsh equals expansion" in message for message in messages)
    # Generic expansion misses are too common to be tier-3 signal.
    assert not any("$HOME" in message for message in messages)


# --- F1: zsh modifier/subscript suffixes after a bare `$NAME` -------------------


@pytest.mark.parametrize(
    "command",
    [
        # Each resolves to a workspace-looking operand in the policy while
        # zsh re-interprets the suffix (probed) into an out-of-workspace
        # path — the resolver must refuse the whole expansion.
        "X=a; cat $X:s,a,../x,",
        "X=a; cat $X:gs,a,../x,",
        "X=a; cat $X:h",
        "X=a; cat $X[1]",
        "X=a; cat $X:q:s,a,../x,",
        "X=a; cat ${X}:h",
        "X=a; cat $((X)):h",
    ],
)
def test_zsh_modifier_suffix_expansions_deny(locate_workspace, command: str) -> None:
    analysis = analyze_shell_command(command)

    assert analysis.requires_approval
    # The construct keeps its ordinary dynamic-syntax denial (the reason
    # names the node type: variable, parameter, or arithmetic expansion).
    assert analysis.approval_diagnostic.denial_class is DenialClass.SYNTAX
    assert any(
        reason in analysis.approval_reasons
        for reason in (
            "variable expansion (unsupported in v0.1 because it can change arguments)",
            "parameter expansion",
            "arithmetic expansion",
        )
    )


# --- B1: unmodeled reassignment invalidates the binding -------------------------


@pytest.mark.parametrize(
    "command",
    [
        # export/typeset/readonly/declare/local arguments reassign the name
        # in a form the binding collection does not model.
        "X=AGENTS.md; export X=/etc/passwd; cat $X",
        "X=AGENTS.md; typeset X=/etc/passwd; cat $X",
        "X=AGENTS.md; readonly X=/etc/passwd; cat $X",
        "X=AGENTS.md; declare X=/etc/passwd; cat $X",
        "X=AGENTS.md; local X=/etc/passwd; cat $X",
        # A quoted declaration argument is not even a variable_assignment.
        "X=AGENTS.md; export 'X=/etc/passwd'; cat $X",
        # An &&/||-guarded reassignment DOES bind at runtime.
        "X=AGENTS.md; true && X=/etc/passwd; cat $X",
        "X=AGENTS.md; false || X=/etc/passwd; cat $X",
        # A command-prefix assignment is not a proven non-reassignment.
        "X=AGENTS.md; X=/etc/passwd cat /dev/null; cat $X",
        # A subshell assignment is not modeled either (conservative).
        "X=AGENTS.md; (X=/etc/passwd); cat $X",
        # A non-literal value is not a provable reassignment.
        "X=AGENTS.md; X=$Y; cat $X",
    ],
)
def test_unmodeled_reassignment_invalidates_the_binding(
    locate_workspace, command: str
) -> None:
    analysis = analyze_shell_command(command)

    assert analysis.requires_approval
    assert (
        "variable expansion (unsupported in v0.1 because it can change arguments)"
        in analysis.approval_reasons
        or "a non-literal environment assignment" in analysis.approval_reasons
    )


def test_literal_rebinding_resolves_to_the_last_binding(locate_workspace) -> None:
    # Only the LAST top-level literal binding is in force at the use.
    analysis = analyze_shell_command("X=a; X=AGENTS.md; cat $X")

    assert not analysis.requires_approval
    assert analysis.command_parts[0] == "cat AGENTS.md"


def test_bare_export_without_value_keeps_the_binding(locate_workspace) -> None:
    # `export X` carries no value, so the runtime value is unchanged and the
    # binding still resolves.
    analysis = analyze_shell_command("X=AGENTS.md; export X; cat $X")

    assert not analysis.requires_approval
    assert analysis.command_parts[0] == "cat AGENTS.md"


def test_rebinding_after_an_unmodeled_reassignment_resolves(locate_workspace) -> None:
    # A later literal assignment rebinds after the invalidation, exactly as
    # the shell's last assignment wins.
    analysis = analyze_shell_command("X=a; export X=/etc/passwd; X=AGENTS.md; cat $X")

    assert not analysis.requires_approval
    assert analysis.command_parts[0] == "cat AGENTS.md"


# --- H1: whitespace and empty resolved values -----------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # A splitting value is not one inert argv element under every
        # executor (latent today: zsh runs with SH_WORD_SPLIT off).
        'X="AGENTS.md /etc/passwd"; cat $X',
        'X="a\tb"; cat $X',
        # An empty unquoted word is dropped by zsh entirely, so the policy
        # would authorize a different argv than the runtime runs.
        "X=; cat $X",
    ],
)
def test_whitespace_and_empty_values_deny(locate_workspace, command: str) -> None:
    analysis = analyze_shell_command(command)

    assert analysis.requires_approval
    assert (
        "variable expansion (unsupported in v0.1 because it can change arguments)"
        in analysis.approval_reasons
    )


# --- F2: the simulation read is gated to the authorized roots -------------------


def _ban_simulation_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the recognizer never reads the gated source (call after setup)."""

    def _no_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("the simulation must not read this source")

    monkeypatch.setattr(Path, "read_text", _no_read)
    monkeypatch.setattr(Path, "open", _no_read)


def test_out_of_workspace_source_denies_without_being_read(
    locate_workspace, tmp_path, monkeypatch, caplog
) -> None:
    outside = tmp_path.parent / "ame-outside"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("needle\n")
    command = (
        "sed -n \"$(grep -n 'needle' ../ame-outside/secret.txt "
        '| head -1 | cut -d: -f1),+45p" lines.txt'
    )
    _ban_simulation_reads(monkeypatch)

    with caplog.at_level(
        "DEBUG", logger="chartreux.core.tools.builtins._shell_permission_analysis"
    ):
        analysis = analyze_shell_command(command)

    # The pre-check refuses the source before the recognizer runs: the
    # construct keeps its ordinary denial and the file is never read.
    assert analysis.requires_approval
    assert "command substitution" in analysis.approval_reasons
    assert any("sed locate-then-print" in message for message in caplog.messages)


def test_device_source_denies_without_being_read(locate_workspace, monkeypatch) -> None:
    # /dev/stdin on a FIFO would hang the synchronous analysis; the root
    # gate refuses it before the recognizer (and its stat gate) runs.
    _ban_simulation_reads(monkeypatch)
    analysis = analyze_shell_command(
        'sed -n "$(grep -n x /dev/stdin | cut -d: -f1)p" lines.txt'
    )

    assert analysis.requires_approval
    assert "command substitution" in analysis.approval_reasons


def test_in_workspace_oversized_source_denies_without_being_read(
    locate_workspace, tmp_path, monkeypatch
) -> None:
    from chartreux.core.tools.builtins._shell_static_expansion import (
        _SIMULATION_READ_CAP_BYTES,
    )

    big = tmp_path / "big.txt"
    big.write_bytes(b"needle\n" + b"x" * _SIMULATION_READ_CAP_BYTES)
    _ban_simulation_reads(monkeypatch)
    analysis = analyze_shell_command(
        f"sed -n \"$(grep -n 'needle' {big} | head -1 | cut -d: -f1),+45p\" lines.txt"
    )

    assert analysis.requires_approval
    assert "command substitution" in analysis.approval_reasons


def test_in_workspace_binary_source_denies(locate_workspace, tmp_path) -> None:
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"needle\n\x00binary\n")
    analysis = analyze_shell_command(
        f"sed -n \"$(grep -n 'needle' {binary} | head -1 | cut -d: -f1),+45p\" lines.txt"
    )

    assert analysis.requires_approval
    assert "command substitution" in analysis.approval_reasons


def test_workspace_roots_gate_the_simulation_read(locate_workspace, tmp_path) -> None:
    command = (
        "sed -n \"$(grep -n 'lit' lines.txt | head -1 | cut -d: -f1),+45p\" lines.txt"
    )

    # No roots at all: fail closed, the simulation may not read anything.
    analysis = analyze_shell_command(
        command, scratchpad_roots=frozenset(), workspace_roots=frozenset()
    )
    assert analysis.requires_approval
    # The ambient workspace (the analysis-time working directory) allows the
    # in-workspace source, and so do explicit workspace roots.
    assert not analyze_shell_command(command).requires_approval
    assert not analyze_shell_command(
        command, workspace_roots=frozenset({tmp_path.resolve()})
    ).requires_approval
    # A root that does not contain the source denies it.
    assert analyze_shell_command(
        command, workspace_roots=frozenset({tmp_path / "elsewhere"})
    ).requires_approval


# --- W7 support: the whole-call aggregate includes policy-class reasons ---------


def test_aggregate_approval_diagnostic_includes_policy_reasons() -> None:
    analysis = analyze_shell_command("echo $(whoami)")
    assert analysis.approval_diagnostic.denial_class is DenialClass.SYNTAX

    policy_reason = Diagnostic(
        "analysis", detail="a dangerous git operation requiring approval (git.push)"
    )
    aggregate = aggregate_approval_diagnostic(analysis, policy_reason)

    assert aggregate.denial_class is DenialClass.POLICY
    rendered = render_diagnostic(aggregate)
    assert "command substitution" in rendered
    assert "git.push" in rendered
    assert "Instead," not in rendered
    # Without the policy reason the aggregate keeps the syntax class.
    assert aggregate_approval_diagnostic(analysis).denial_class is DenialClass.SYNTAX
