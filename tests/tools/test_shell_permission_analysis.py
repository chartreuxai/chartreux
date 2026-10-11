from __future__ import annotations

import pytest

from chartreux.core.tools.builtins._shell_permission_analysis import (
    analyze_shell_command,
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
