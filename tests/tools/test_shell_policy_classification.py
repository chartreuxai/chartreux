"""General command-classification tests for the shell-command policy suite (rm/chmod/chown/find/xargs/sort/date/less/tree/grep/diff policies, sed, inline interpreter switches, prefix matching, the generic denylist, path candidates and path values, policy dispatch, ported policies, the second-review guardrail cluster, the non-git write-operand cluster, and the v0.2 redirect cluster)."""

from __future__ import annotations

from pathlib import Path
import shlex

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
    inline_interpreter_switch,
    matches_command_prefix,
    path_candidates,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from tests.tools._shell_policy_test_support import bash_tool as bash_tool


@pytest.mark.parametrize("option", ["-r", "-rf", "-fr", "--recursive", "--rec"])
def test_recursive_rm_diagnostic_retains_original_option(option: str) -> None:
    policy = analyze_shell_command_policy(["rm", option, "file"])
    assert policy.requires_approval
    diagnostic = policy.denial_diagnostic(f"rm {option} file")
    assert diagnostic.code == "recursive_rm"
    assert diagnostic.offending_token == option
    assert diagnostic.command_part == f"rm {option} file"
    assert not analyze_shell_command_policy(["rm", "--", option]).requires_approval


@pytest.mark.parametrize(
    "argv",
    [
        ["rm", "-r", "file"],
        ["rm", "--recursive", "file"],
        ["sed", "-f", "script.sed", "file"],
        ["git", "reset", "--hard", "HEAD"],
    ],
)
def test_policy_module_denials_classify_as_policy_without_alternative(
    argv: list[str],
) -> None:
    # Denials created by the command policy layer arrive without an explicit
    # class and fail closed to POLICY: the rendered message states the
    # blocked requirement and names no alternative.
    from chartreux.core.tools.builtins._shell_diagnostics import (
        DenialClass,
        render_diagnostic,
    )

    policy = analyze_shell_command_policy(argv)
    assert policy.requires_approval
    diagnostic = policy.denial_diagnostic(shlex.join(argv))
    assert diagnostic.denial_class is DenialClass.POLICY
    assert diagnostic.alternative is None
    rendered = render_diagnostic(diagnostic)
    assert "Instead," not in rendered
    assert "use read_file" not in rendered
    assert "use grep" not in rendered
    assert "use the edit tool" not in rendered
    assert "use write_file" not in rendered
    assert "rmdir" not in rendered


# --- Attached write-operand extraction (delta review, family 2) ---


@pytest.mark.parametrize(
    "command, expected",
    [
        # cp/mv/install/ln: --target-directory= and the attached short -t<dir>.
        ("cp --target-directory=.git seed.txt", ".git"),
        ("cp -t.git seed.txt", ".git"),
        ("mv --target-directory=.git/hooks f", ".git/hooks"),
        ("install -t.git seed.txt", ".git"),
        ("ln --target-directory=.git seed.txt", ".git"),
        # tar: -C<dir> attached (also in a cluster), -f<archive>, --file=.
        ("tar -C.git -xf evil.tgz", ".git"),
        ("tar -xC.git -f evil.tgz", ".git"),
        ("tar -cf.git/config seed.txt", ".git/config"),
        ("tar --file=.git/config -c seed.txt", ".git/config"),
        ("tar -xf evil.tgz --directory=.git", ".git"),
        # unzip: -d<dir> attached.
        ("unzip -d.git evil.zip", ".git"),
        # patch: --output=/-o<val>, --reject-file=, -d<dir>, --prefix=.
        ("patch --output=.git/config fix.diff", ".git/config"),
        ("patch -o.git/config fix.diff", ".git/config"),
        ("patch --reject-file=.git/rej fix.diff", ".git/rej"),
        ("patch -d.git fix.diff", ".git"),
        ("patch --prefix=.git/ -b fix.diff", ".git/"),
        # openssl: -out= and --out= (single- and double-dash spellings).
        ("openssl enc -out=.git/config -in seed.txt", ".git/config"),
        ("openssl enc --out=.git/config -in seed.txt", ".git/config"),
        ("openssl enc -out .git/config -in seed.txt", ".git/config"),
        # iconv: -o<val> and --output=.
        ("iconv -o.git/config seed.txt", ".git/config"),
        ("iconv --output=.git/config seed.txt", ".git/config"),
        # rsync: the write-directory options and -T<dir>.
        ("rsync --temp-dir=.git a b", ".git"),
        ("rsync -T.git a b", ".git"),
        ("rsync --log-file=.git/log a b", ".git/log"),
        # tee: the attached --output-error=.
        ("tee --output-error=.git/config f", ".git/config"),
        # csplit: --prefix= and the attached short -f<prefix>.
        ("csplit --prefix=.git/cracked seed.txt 1", ".git/cracked"),
        ("csplit -f.git/cracked seed.txt 1", ".git/cracked"),
        # git's raw write operands: archive --output and a writing config
        # --file (both flow through write_path_values).
        ("git archive --output=.git/config HEAD", ".git/config"),
        ("git archive -o.git/config HEAD", ".git/config"),
        ("git config --file=.git/config user.name x", ".git/config"),
        ("git config -f.git/config user.name x", ".git/config"),
    ],
)
def test_attached_write_operands_are_extracted_as_path_candidates(
    command: str, expected: str
) -> None:
    candidates = path_candidates(shlex.split(command), inspect_positional_paths=True)
    assert expected in candidates, command


def test_split_extracts_only_the_prefix_operand() -> None:
    # split's value options (a SIZE, a SEPARATOR) never become candidates;
    # only the trailing PREFIX names the written output files.
    candidates = path_candidates(
        shlex.split("split -b 100 -t / seed.txt out"), inspect_positional_paths=False
    )
    assert candidates == ("out",)
    candidates = path_candidates(
        shlex.split("split -b 100 seed.txt"), inspect_positional_paths=False
    )
    assert candidates == ()


@pytest.mark.parametrize(
    "command",
    [
        'pip install "requests[socks]"',
        r"pip install requests\[socks\]",
        "pip install requests[socks",
        "git log --format=[%h]",
        "git log --format=[%h] -- go",
        'printf "%s\\n" env LESSOPEN=fixture',
        r'nohup p"ip" install requests\[socks\]',
        'env CI=1 p"ip" install requests[socks',
        'pip install -e ".[dev]"',
        'uv pip install -e ".[dev]"',
        'uv add "httpx[http2]"',
        'poetry add "httpx[http2]"',
        "pytest > /dev/null",
        "pytest 2> /dev/stderr",
        "pytest > /dev/stdout",
        "ls src/*.py",
        "ls *.md",
        'env CI=1 pip install "requests[socks]"',
        "env CI=1 npm test",
        "env CI=1 git status",
    ],
)
def test_benign_shell_forms(bash_tool: Bash, command: str) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        'pip install -e "../outside[dev]"',
        'uv pip install -e "/tmp/outside[dev]"',
        'uv add "../outside[dev]"',
        'poetry add "/tmp/outside[dev]"',
        "pytest > /dev/random",
        "pip install requests[socks]",
        "pip install -e .[dev]",
        "uv pip install -e .[dev]",
        "uv add httpx[http2]",
        "poetry add httpx[http2]",
        "python -m pip install requests[socks]",
        "env CI=1 pip install -e .[dev]",
        "nohup pip install -e .[dev]",
        'python -m p"ip" install -e .[dev]',
        'nohup p"ip" install -e .[dev]',
        'env CI=1 p"ip" install -e .[dev]',
        'command nohup p"ip" install -e .[dev]',
        'timeout 1 env CI=1 p"ip" install -e .[dev]',
        "timeout 1 python -m pip install requests[socks]",
        "ls .*",
        "ls ..*",
        "ls src/.*",
        "ls -L .venv/bin/python[^x]",
        "ls [.]?",
        "ls ../*.py",
        "ls /tmp/*.py",
        "env PATH=/tmp npm test",
        "env GIT_CONFIG_GLOBAL=fixture git status",
        "env PYTHONPATH=../outside pytest",
        "env ci=1 npm test",
        "env CI=1 rm -rf src",
    ],
)
def test_benign_shell_form_controls(bash_tool: Bash, command: str) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission is ToolPermission.NEVER


def test_listing_glob_rejects_outside_symlink(bash_tool: Bash, tmp_path: Path) -> None:
    (bash_tool.cwd / "escape.py").symlink_to(bash_tool.cwd.parent / "outside.py")
    context = bash_tool.resolve_permission(BashArgs(command="ls *.py"))
    assert context is not None and context.permission is ToolPermission.NEVER


def test_unquoted_local_extras_cannot_hide_outside_symlink(bash_tool: Bash) -> None:
    (bash_tool.cwd / ".d").symlink_to(bash_tool.cwd.parent / "outside-package")
    context = bash_tool.resolve_permission(BashArgs(command="pip install -e .[dev]"))
    assert context is not None and context.permission is ToolPermission.NEVER
    context = bash_tool.resolve_permission(BashArgs(command='pip install -e ".[dev]"'))
    assert context is not None and context.permission is ToolPermission.ALWAYS


def test_listing_bracket_glob_cannot_hide_outside_symlink(bash_tool: Bash) -> None:
    bin_dir = bash_tool.cwd / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python3").symlink_to(bash_tool.cwd.parent / "outside-python")
    context = bash_tool.resolve_permission(
        BashArgs(command="ls -L .venv/bin/python[^x]")
    )
    assert context is not None and context.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "form",
    [
        "{assignment} less -FX AGENTS.md",
        "env {assignment} less -FX AGENTS.md",
        "export {assignment}; less -FX AGENTS.md",
    ],
)
@pytest.mark.parametrize(
    "name",
    [
        "LESSOPEN",
        "LESSCLOSE",
        "LESSEDIT",
        "KSH_ENV",
        "BASH_ENV",
        "ENV",
        "ZDOTDIR",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "PAGER",
        "GIT_PAGER",
        "PYTHONSTARTUP",
        "RUBYOPT",
        "PERL5OPT",
        "NODE_OPTIONS",
        "SSH_ASKPASS",
        "SSH_ASKPASS_REQUIRE",
        "RUSTC_WRAPPER",
        "RUSTC",
        "CC",
        "CXX",
        "RUSTFLAGS",
        "GOFLAGS",
        "MAKEFLAGS",
        "GNUMAKEFLAGS",
        "PERLLIB",
        "_JAVA_OPTIONS",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "'|cat /outside/data %s'",
        "'`cat /outside/data`'",
        "'$(cat /outside/data)'",
        "fixture",
    ],
)
def test_executable_configuring_environment_is_denied(
    bash_tool: Bash, form: str, name: str, value: str
) -> None:
    command = form.format(assignment=f"{name}={value}")
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission is ToolPermission.NEVER


@pytest.mark.parametrize("prefix", ["", "env ", "nohup env "])
@pytest.mark.parametrize(
    "name",
    [
        "LANG",
        "LC_ALL",
        "CI",
        "TERM",
        "NO_COLOR",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "UV_NO_SYNC",
    ],
)
def test_benign_environment_controls(bash_tool: Bash, prefix: str, name: str) -> None:
    context = bash_tool.resolve_permission(
        BashArgs(command=f"{prefix}{name}=1 npm test")
    )
    assert context is not None and context.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "form",
    [
        "{assignment} npm test",
        "env {assignment} npm test",
        "command env {assignment} npm test",
        "nohup env {assignment} npm test",
        "export {assignment}; npm test",
    ],
)
@pytest.mark.parametrize("value", [r"\|cat", r"\`cat\`", r"\$\(cat\)"])
def test_unquoted_environment_execution_syntax_is_denied(
    bash_tool: Bash, form: str, value: str
) -> None:
    context = bash_tool.resolve_permission(
        BashArgs(command=form.format(assignment=f"FOO={value}"))
    )
    assert context is not None and context.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "options", ["-i", "-i.bak", "-ni", "--in-place", "--in-place=.bak", "--in-p"]
)
def test_sed_in_place_is_a_scoped_mutator(bash_tool: Bash, options: str) -> None:
    tokens = shlex.split(f"sed {options} s/a/b/ input.txt")
    policy = analyze_shell_command_policy(tokens)
    assert policy.mutates_files
    assert not policy.requires_approval
    paths = path_candidates(tokens, inspect_positional_paths=False)
    assert "input.txt" in paths
    assert all(path.startswith("input.txt") for path in paths)
    context = bash_tool.resolve_permission(BashArgs(command=shlex.join(tokens)))
    assert context is not None and context.permission is ToolPermission.ALWAYS
    tokens[-1] = "../outside.txt"
    context = bash_tool.resolve_permission(BashArgs(command=shlex.join(tokens)))
    assert context is not None and context.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "tokens,mutates,paths",
    [
        (["sed", "s/a/b/", "input.txt"], False, ("input.txt",)),
        (["sed", "-e", "s/i/b/", "input.txt"], False, ("input.txt",)),
        (["sed", "-es/i/b/", "input.txt"], False, ("input.txt",)),
        (["sed", "-ie", "s/a/b/", "input.txt"], True, ("input.txte", "input.txt")),
        (["sed", "--", "s/a/b/", "-i"], False, ("-i",)),
    ],
)
def test_sed_options_do_not_reparse_scripts_or_suffixes(
    tokens: list[str], mutates: bool, paths: tuple[str, ...]
) -> None:
    assert analyze_shell_command_policy(tokens).mutates_files is mutates
    assert path_candidates(tokens, inspect_positional_paths=False) == paths


@pytest.mark.parametrize(
    "command",
    [
        "npm install --prefix=/etc pkg",
        "npm install --prefix /etc pkg",
        "pip install --target=/etc pkg",
        "pip install -t/etc pkg",
        "python3 -m pip install --target=/etc pkg",
        "NPM_CONFIG_PREFIX=/etc npm install pkg",
        "PIP_TARGET=/etc pip install pkg",
        "PIP_PREFIX=/etc python -m pip install pkg",
        "PIP_CONFIG_FILE=../pip.conf pip install pkg",
        "export NPM_CONFIG_PREFIX=/etc; npm install pkg",
        "sed 'r ~/.ssh/id_rsa' input.txt",
        "sed 'w /etc/passwd' input.txt",
        "sed -e '1,2r ../secret' input.txt",
        "sed --expression='s/a/b/w /etc/passwd' input.txt",
        "sed '/pattern/{w ../outside' input.txt",
        "sed 'p; R ../outside' input.txt",
        "gsed 'r ../outside' input.txt",
        "gsed -i s/a/b/ ../outside.txt",
        "dd of=../outside.txt",
        "dd if=input of=.env",
        "sed -iv s/a/b/ .en",
        "sed --in-place=v s/a/b/ .en",
        "gsed -iv s/a/b/ .en",
    ],
)
def test_operand_bypass_vectors_are_denied(bash_tool: Bash, command: str) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "pip install package",
        "npm install pkg",
        "npm install --prefix=local pkg",
        "pip install --target=local pkg",
        "python -m pip install package",
        "sed -i s/a/b/ file.txt",
        "sed -i.bak s/a/b/ file.txt",
        "sed 'r local.txt' input.txt",
        "sed 'w local.txt' input.txt",
        "sed 's/a/b/w local.txt' input.txt",
        "sed '/word/r local.txt' input.txt",
        "gsed -i s/a/b/ file.txt",
        "dd if=input of=workspace_out",
        "LANG=C npm install pkg",
        "python script.py -m pip ../argument",
    ],
)
def test_operand_controls_remain_permitted(bash_tool: Bash, command: str) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission is ToolPermission.ALWAYS


def test_sed_backup_and_script_paths_are_extracted() -> None:
    assert path_candidates(
        ["sed", "-iv", "w output.txt", ".en"], inspect_positional_paths=False
    ) == ("output.txt", ".env", ".en")


def test_sed_derived_backup_symlink_is_workspace_scoped(bash_tool: Bash) -> None:
    (bash_tool.cwd / "input.txt.bak").symlink_to(bash_tool.cwd.parent / "outside")
    context = bash_tool.resolve_permission(
        BashArgs(command="sed -i.bak s/a/b/ input.txt")
    )
    assert context is not None and context.permission is ToolPermission.NEVER


def test_sed_filename_trailing_space_is_preserved(bash_tool: Bash) -> None:
    (bash_tool.cwd / "output ").symlink_to(bash_tool.cwd.parent / "outside")
    context = bash_tool.resolve_permission(
        BashArgs(command="sed 'w output ' input.txt")
    )
    assert context is not None and context.permission is ToolPermission.NEVER


def test_sed_backup_directory_suffix_is_denied(bash_tool: Bash) -> None:
    context = bash_tool.resolve_permission(
        BashArgs(command="sed --in-place=../* s/a/b/ input.txt")
    )
    assert context is not None and context.permission is ToolPermission.NEVER


@pytest.mark.parametrize("script", ["42,+45p", "42,87p", "42p", "p", "3,+2p"])
def test_numeric_address_print_scripts_are_allowed(
    bash_tool: Bash, script: str
) -> None:
    # The GNU relative-address joiner `,+N` is the form the statically
    # expanded locate-then-print idiom produces; a hand-written literal
    # script of the same shape is allowed by the normal sed script policy.
    context = bash_tool.resolve_permission(
        BashArgs(command=f"sed -n '{script}' input.txt")
    )
    assert context is not None and context.permission is ToolPermission.ALWAYS


def test_sed_step_address_stays_unmodeled(bash_tool: Bash) -> None:
    context = bash_tool.resolve_permission(BashArgs(command="sed -n '1~2p' input.txt"))
    assert context is not None and context.permission is ToolPermission.NEVER


def test_sed_relative_address_does_not_hide_write_commands(bash_tool: Bash) -> None:
    # The `+` skip is address-scoped in effect only: this scan continues
    # linearly, so a file command after it is still found and denied.
    context = bash_tool.resolve_permission(
        BashArgs(command="sed -n '2,+45p; w ../outside' input.txt")
    )
    assert context is not None and context.permission is ToolPermission.NEVER


def test_sed_locate_print_idiom_is_authorized_end_to_end(bash_tool: Bash) -> None:
    (bash_tool.cwd / "lines.txt").write_text(
        "alpha\nlit\nbeta\n" + "\n".join(f"line{i}" for i in range(4, 60)) + "\n"
    )
    context = bash_tool.resolve_permission(
        BashArgs(
            command=(
                "sed -n \"$(grep -n 'lit' lines.txt | head -1 "
                '| cut -d: -f1),+45p" lines.txt'
            )
        )
    )
    assert context is not None and context.permission is ToolPermission.ALWAYS
    # The inner pipeline's read operand stays path-checked: a source file
    # outside the workspace denies the recognized idiom.
    context = bash_tool.resolve_permission(
        BashArgs(
            command=(
                "sed -n \"$(grep -n 'lit' ../outside.txt | head -1 "
                '| cut -d: -f1),+45p" lines.txt'
            )
        )
    )
    assert context is not None and context.permission is ToolPermission.NEVER


def test_sed_locate_print_multi_match_is_denied_end_to_end(bash_tool: Bash) -> None:
    (bash_tool.cwd / "lines.txt").write_text("lit\nalpha\nlit\n")
    context = bash_tool.resolve_permission(
        BashArgs(
            command=(
                "sed -n \"$(grep -n 'lit' lines.txt | cut -d: -f1),+45p\" lines.txt"
            )
        )
    )
    assert context is not None and context.permission is ToolPermission.NEVER
    # The resolver's guardrail denies the raw (unexpanded) script text first,
    # so end-to-end the multi-match denial carries the sed-script message; the
    # locate-then-print-specific reason is carried by the analysis diagnostics.
    assert "unsupported" in (context.reason or "")
    # Selecting one located line restores the single-address guarantee.
    context = bash_tool.resolve_permission(
        BashArgs(
            command=(
                "sed -n \"$(grep -n 'lit' lines.txt | head -1 "
                '| cut -d: -f1),+45p" lines.txt'
            )
        )
    )
    assert context is not None and context.permission is ToolPermission.ALWAYS


def test_inline_interpreter_switch_scans_only_real_options() -> None:
    assert (
        inline_interpreter_switch(["/usr/bin/python3", "-B", "-uc", "print(1)"])
        == "python3 -c"
    )
    assert inline_interpreter_switch(["node", "--eval=1"]) == "node -e"
    assert (
        inline_interpreter_switch(["node", "--no-warnings", "--eval", "1"]) == "node -e"
    )
    assert inline_interpreter_switch(["ruby", "-we", "puts 1"]) == "ruby -e"
    assert inline_interpreter_switch(["python3", "script.py", "-c"]) is None
    assert inline_interpreter_switch(["python3", "--", "-c"]) is None
    assert inline_interpreter_switch(["echo", "curl"]) is None


@pytest.mark.parametrize(
    "tokens",
    [
        ["python3", "-W", "ignore", "-c", "print(1)"],
        ["python", "-X", "dev", "-c", "print(1)"],
        ["pypy", "-Wignore", "-c", "print(1)"],
        ["pypy3", "-Xdev", "-c", "print(1)"],
        ["node", "--require=module", "--eval=1"],
        ["node", "-r", "module", "-e", "1"],
        ["perl", "-I", "lib", "-e", "1"],
        ["ruby", "-I", "lib", "-e", "puts 1"],
    ],
)
def test_inline_interpreter_options_consume_values(tokens: list[str]) -> None:
    assert (
        inline_interpreter_switch(tokens)
        == f"{tokens[0]} -{'c' if tokens[0].startswith(('python', 'pypy')) else 'e'}"
    )


@pytest.mark.parametrize(
    "tokens",
    [
        ["python", "-u", "script.py", "-c", "arg"],
        ["python", "-m", "module", "-c", "argument"],
        ["python", "-W", "ignore::DeprecationWarning", "script.py"],
        ["python3", "-Xdev", "script.py", "-c", "argument"],
        ["node", "--require=module", "script.js", "--eval=1"],
        ["perl", "-Ilib", "script.pl", "-e", "argument"],
        ["ruby", "-Ilib", "script.rb", "-e", "argument"],
    ],
)
def test_inline_interpreter_options_stop_at_target(tokens: list[str]) -> None:
    assert inline_interpreter_switch(tokens) is None


def test_unknown_interpreter_option_before_inline_code_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported python option"):
        inline_interpreter_switch(["python", "--unknown=value", "-c", "print(1)"])


def test_prefix_match_accepts_extra_arguments() -> None:
    assert matches_command_prefix(["git", "status", "--short"], ["git", "status"])


def test_prefix_match_accepts_exact_command() -> None:
    assert matches_command_prefix(["git", "status"], ["git", "status"])


def test_prefix_match_normalizes_executable_basename_in_command() -> None:
    assert matches_command_prefix(["/usr/bin/git", "status"], ["git", "status"])
    assert matches_command_prefix(["./git", "status"], ["git", "status"])


def test_prefix_match_normalizes_executable_basename_in_pattern() -> None:
    assert matches_command_prefix(["gdb", "-x"], ["/usr/bin/gdb"])


def test_prefix_match_does_not_normalize_argument_tokens() -> None:
    # Only the executable token is basename-normalized; arguments compare literally.
    assert not matches_command_prefix(["git", "/usr/bin/status"], ["git", "status"])


def test_prefix_match_requires_exact_argument_tokens() -> None:
    # Argument tokens must match exactly: "statusx" is not the argument "status".
    assert not matches_command_prefix(["git", "statusx"], ["git", "status"])


def test_prefix_match_rejects_different_executable() -> None:
    assert not matches_command_prefix(["got", "status"], ["git", "status"])
    assert not matches_command_prefix(["gdbx"], ["gdb"])


def test_prefix_match_rejects_command_shorter_than_pattern() -> None:
    assert not matches_command_prefix(["git"], ["git", "status"])


def test_prefix_match_rejects_empty_tokens_or_pattern() -> None:
    assert not matches_command_prefix([], ["x"])
    assert not matches_command_prefix(["x"], [])


def test_prefix_match_short_option_matches_combined_short_options() -> None:
    # getopt parses "-ic" as "-i -c", so the pattern "bash -i" must catch the
    # equivalent combined form of the interactive shell.
    assert matches_command_prefix(["bash", "-ic", "x"], ["bash", "-i"])
    assert matches_command_prefix(["bash", "-ci"], ["bash", "-i"])


def test_prefix_match_short_option_does_not_match_long_options() -> None:
    assert not matches_command_prefix(["bash", "--interactive"], ["bash", "-i"])
    assert not matches_command_prefix(["cmd", "--ignore"], ["cmd", "-i"])


def test_prefix_match_short_option_does_not_match_plain_words() -> None:
    assert not matches_command_prefix(["bash", "interactive"], ["bash", "-i"])
    assert not matches_command_prefix(["bash", "i"], ["bash", "-i"])


def test_prefix_match_combined_flags_outside_any_pattern_pass_through() -> None:
    # Benign combined flags that appear in no denylist pattern are unaffected.
    assert not matches_command_prefix(["ls", "-la"], ["bash", "-i"])
    assert not matches_command_prefix(["grep", "-in", "x"], ["bash", "-i"])


def test_single_token_pattern_matches_command_with_arguments() -> None:
    # A bare-name denylist pattern is a prefix, so it denies every invocation.
    assert matches_command_prefix(["gdb", "-x", "file"], ["gdb"])
    assert matches_command_prefix(["gdb"], ["gdb"])


@pytest.fixture
def shell_resolver(bash_tool: Bash):
    from chartreux.core.tools.builtins._shell_permission_resolver import (
        ShellPermissionResolver,
    )

    return ShellPermissionResolver(
        bash_tool.config, bash_tool.cwd, bash_tool.workspace, bash_tool.path_authority
    )


def test_with_args_denylist_denies_matching_prefix(shell_resolver) -> None:
    assert shell_resolver._find_denylist_match("bash -i") == "bash -i"
    assert shell_resolver._find_denylist_match("bash -i -x") == "bash -i"


def test_with_args_denylist_normalizes_executable_basename(shell_resolver) -> None:
    assert shell_resolver._find_denylist_match("/usr/bin/bash -i") == "bash -i"


def test_with_args_denylist_denies_combined_short_options(shell_resolver) -> None:
    # "bash -ic x" is the interactive shell the "bash -i" pattern blocks.
    assert shell_resolver._find_denylist_match("bash -ic x") == "bash -i"


def test_with_args_denylist_allows_benign_combined_flags(shell_resolver) -> None:
    assert shell_resolver._find_denylist_match("ls -la") is None
    assert shell_resolver._find_denylist_match("grep -in pattern file") is None


def test_with_args_denylist_does_not_deny_bare_command(shell_resolver) -> None:
    # "bash -i" is a prefix pattern: bare "bash" does not carry the -i argument.
    assert shell_resolver._find_denylist_match("bash") is None


def test_standalone_denylist_denies_bare_command(shell_resolver) -> None:
    assert shell_resolver._is_standalone_denylisted("python")


def test_standalone_denylist_normalizes_executable_basename(shell_resolver) -> None:
    assert shell_resolver._is_standalone_denylisted("/usr/bin/python")


def test_standalone_denylist_allows_command_with_arguments(shell_resolver) -> None:
    assert not shell_resolver._is_standalone_denylisted("python script.py")


def test_standalone_denylist_matches_exact_names_only(shell_resolver) -> None:
    # Standalone entries are exact names, not prefixes: python3 is a different binary.
    assert not shell_resolver._is_standalone_denylisted("python3")


def test_guardrail_denies_with_args_denylist_match(shell_resolver) -> None:
    context = shell_resolver._resolve_guardrail_permission(["/usr/bin/bash -i"])

    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "matches denylist pattern" in context.reason


def test_guardrail_denies_standalone_denylist_match(shell_resolver) -> None:
    context = shell_resolver._resolve_guardrail_permission(["python"])

    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "not allowed as a standalone command" in context.reason


def test_guardrail_allows_non_denylisted_commands(shell_resolver) -> None:
    assert shell_resolver._resolve_guardrail_permission(["echo hi"]) is None
    assert shell_resolver._resolve_guardrail_permission(["python script.py"]) is None


def test_option_specific_denylist_pattern_retains_raw_matching(
    tmp_path, monkeypatch
) -> None:
    # Only the command side is canonicalized: a pattern that carries options
    # still matches its raw spelling and nothing else.
    from chartreux.core.tools.builtins._shell_permission_resolver import (
        ShellPermissionResolver,
    )

    monkeypatch.chdir(tmp_path)
    config = BashToolConfig(denylist=["git -C . push"])
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())
    resolver = ShellPermissionResolver(
        bash_tool.config, bash_tool.cwd, bash_tool.workspace, bash_tool.path_authority
    )

    assert resolver._find_denylist_match("git -C . push") == "git -C . push"
    assert resolver._find_denylist_match("git -C sub push") is None
    assert resolver._find_denylist_match("git --no-pager -C . push") is None


def test_sensitive_patterns_stay_raw_prefix_only(tmp_path, monkeypatch) -> None:
    # Normalization applies only to the configurable denylist lookup.
    from chartreux.core.tools.builtins._shell_permission_resolver import (
        ShellPermissionResolver,
    )

    monkeypatch.chdir(tmp_path)
    config = BashToolConfig(sensitive_patterns=["git push"])
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())
    resolver = ShellPermissionResolver(
        bash_tool.config, bash_tool.cwd, bash_tool.workspace, bash_tool.path_authority
    )

    assert resolver._is_sensitive("git push")
    assert not resolver._is_sensitive("git -C . push")


def test_rm_plain_file_needs_no_approval() -> None:
    assert not analyze_shell_command_policy(["rm", "file"]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["rm", "-r", "dir"],
        ["rm", "-R", "dir"],
        ["rm", "-rf", "/"],
        ["rm", "-fr", "dir"],
        ["rm", "--recursive", "dir"],
        ["rm", "--rec", "dir"],
    ],
)
def test_rm_recursive_flags_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_rm_double_dash_stops_option_scanning() -> None:
    # After "--", "-r" is a filename, not a recursive flag.
    assert not analyze_shell_command_policy(["rm", "--", "-r"]).requires_approval


def test_rm_option_after_positional_still_requires_approval() -> None:
    # GNU option permutation: "dir" does not end option scanning.
    assert analyze_shell_command_policy(["rm", "dir", "--recursive"]).requires_approval


def test_chmod_non_recursive_permissive_mode_needs_no_approval() -> None:
    assert not analyze_shell_command_policy(["chmod", "777", "dir"]).requires_approval


@pytest.mark.parametrize(
    "mode", ["777", "0007", "00007", "666", "7", "77", "07", "6", "2"]
)
def test_chmod_recursive_permissive_numeric_mode_requires_approval(mode: str) -> None:
    # GNU chmod zero-pads octal modes on the left (any number of digits up to
    # its 07777 value ceiling), so the last digit is always the "others"
    # class: "chmod -R 00007 dir" is mode 0007.
    assert analyze_shell_command_policy(["chmod", "-R", mode, "dir"]).requires_approval


def test_chmod_recursive_too_large_numeric_mode_needs_no_approval() -> None:
    # Beyond GNU's 07777 ceiling chmod fails at runtime ("mode too large");
    # the policy treats it like any other non-permissive mode.
    assert not analyze_shell_command_policy([
        "chmod",
        "-R",
        "10000",
        "f",
    ]).requires_approval


@pytest.mark.parametrize("mode", ["755", "444", "664", "4", "44", "5", "644"])
def test_chmod_recursive_read_only_numeric_mode_needs_no_approval(mode: str) -> None:
    assert not analyze_shell_command_policy([
        "chmod",
        "-R",
        mode,
        "dir",
    ]).requires_approval


@pytest.mark.parametrize("mode", ["a+w", "o+w", "=rw", "u+w,o+w"])
def test_chmod_recursive_permissive_symbolic_mode_requires_approval(mode: str) -> None:
    assert analyze_shell_command_policy(["chmod", "-R", mode, "dir"]).requires_approval


@pytest.mark.parametrize("mode", ["u+w", "g+w", "ug+w"])
def test_chmod_recursive_owner_or_group_symbolic_mode_needs_no_approval(
    mode: str,
) -> None:
    assert not analyze_shell_command_policy([
        "chmod",
        "-R",
        mode,
        "dir",
    ]).requires_approval


def test_chown_non_recursive_needs_no_approval() -> None:
    assert not analyze_shell_command_policy(["chown", "user", "file"]).requires_approval


def test_chown_recursive_requires_approval() -> None:
    assert analyze_shell_command_policy([
        "chown",
        "-R",
        "user",
        "dir",
    ]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["find", ".", "-delete"],
        ["find", ".", "-exec", "rm", "{}", ";"],
        ["find", ".", "-execdir", "ls"],
        ["find", ".", "-fprint", "/tmp/out"],
        ["find", ".", "-ok", "rm", "{}", ";"],
    ],
)
def test_find_side_effect_predicates_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_find_read_only_predicates_need_no_approval() -> None:
    assert not analyze_shell_command_policy([
        "find",
        ".",
        "-name",
        "x",
    ]).requires_approval
    assert not analyze_shell_command_policy([
        "find",
        ".",
        "-printf",
        "y",
    ]).requires_approval


def test_find_value_predicate_consumes_its_argument() -> None:
    # "-delete" is the pattern argument of -name, not a side-effect predicate.
    assert not analyze_shell_command_policy([
        "find",
        ".",
        "-name",
        "-delete",
    ]).requires_approval
    assert analyze_shell_command_policy([
        "find",
        ".",
        "-name",
        "x",
        "-delete",
    ]).requires_approval


def test_find_newer_variants_consume_their_argument() -> None:
    assert analyze_shell_command_policy([
        "find",
        ".",
        "-newermt",
        "x",
        "-delete",
    ]).requires_approval


def test_find_leading_option_tokens_are_skipped() -> None:
    assert analyze_shell_command_policy([
        "find",
        "-L",
        ".",
        "-name",
        "x",
        "-delete",
    ]).requires_approval
    assert analyze_shell_command_policy([
        "find",
        "-O3",
        ".",
        "-delete",
    ]).requires_approval


def test_xargs_always_requires_approval() -> None:
    assert analyze_shell_command_policy(["xargs", "echo"]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["sort", "-o", "out", "in"],
        ["sort", "--output=out", "in"],
        ["sort", "--compress-program", "gzip"],
        ["sort", "-T", "dir", "in"],
        ["sort", "--files0-from", "list"],
        ["sort", "--temporary-directory", "dir"],
    ],
)
def test_sort_side_effecting_options_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_sort_random_source_is_a_path_value_not_an_approval() -> None:
    policy = analyze_shell_command_policy(["sort", "--random-source", "f", "in"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("f",)


@pytest.mark.parametrize(
    "tokens", [["date", "-s", "2020-01-01"], ["date", "--set", "2020-01-01"]]
)
def test_date_set_requires_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_date_reference_options_are_path_values() -> None:
    policy = analyze_shell_command_policy(["date", "-f", "stampfile"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("stampfile",)

    reference = analyze_shell_command_policy(["date", "-r", "stampfile"])
    assert reference.option_path_values == ("stampfile",)


@pytest.mark.parametrize(
    "tokens",
    [
        ["less", "-o", "log", "file"],
        ["less", "--log-file", "x"],
        ["less", "--LOG-FILE", "x"],
    ],
)
def test_less_log_file_options_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize("tokens", [["tree", "-o", "out"], ["tree", "--output", "out"]])
def test_tree_output_requires_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["tree", "-Ho", "host", "out"],
        ["tree", "-Io", "pat", "out"],
        ["tree", "-Lo", "2", "out"],
        ["tree", "-To", "title", "out"],
        ["tree", "-Po", "pat", "out"],
    ],
)
def test_tree_cluster_output_file_requires_approval(tokens: list[str]) -> None:
    # tree takes a value-taking short's operand from the next argv token and
    # resumes scanning the same cluster, so the "o" in each cluster is the
    # output-file option plain `tree -o` already gates.
    assert analyze_shell_command_policy(tokens).requires_approval


def test_tree_inspects_positional_paths_by_default() -> None:
    policy = analyze_shell_command_policy(["tree", "."])
    assert not policy.requires_approval
    assert policy.inspect_positional_paths
    assert policy.positional_values == (".",)


def test_grep_file_option_value_is_collected() -> None:
    policy = analyze_shell_command_policy(["grep", "-f", "patterns", "file"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("patterns",)

    attached = analyze_shell_command_policy(["grep", "--file=patterns", "x"])
    assert attached.option_path_values == ("patterns",)


@pytest.mark.parametrize(
    "tokens", [["du", "--files0-from", "list"], ["wc", "--files0-from", "list"]]
)
def test_files0_from_option_gates_and_names_its_list_file(tokens: list[str]) -> None:
    # The list file's own path is nominated, but the paths it lists come from
    # file content, which static path analysis cannot see.
    policy = analyze_shell_command_policy(tokens)
    assert policy.requires_approval
    assert policy.option_path_values == ("list",)


def test_file_files_from_splits_colon_separated_lists() -> None:
    policy = analyze_shell_command_policy(["file", "--files-from", "a:b", "x"])
    assert policy.requires_approval
    assert policy.option_path_values == ("a", "b")


def test_diff_from_file_is_a_path_value() -> None:
    policy = analyze_shell_command_policy(["diff", "--from-file", "a", "b"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("a",)


def test_policy_dispatch_normalizes_executable_basename() -> None:
    assert analyze_shell_command_policy(["/bin/rm", "-r", "dir"]).requires_approval


@pytest.mark.parametrize("tokens", [["echo", "hi"], ["frobnicate", "--dangerous"], []])
def test_unknown_commands_have_no_policy(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


def test_path_candidates_include_option_values_without_inspect() -> None:
    assert path_candidates(
        ["grep", "-f", "patterns", "file"], inspect_positional_paths=False
    ) == ("patterns",)


def test_path_candidates_add_positionals_when_inspecting() -> None:
    assert path_candidates(
        ["grep", "-f", "patterns", "file"], inspect_positional_paths=True
    ) == ("patterns", "file")
    assert path_candidates(["cat", "-n", "a"], inspect_positional_paths=True) == ("a",)


def test_path_candidates_treat_tokens_after_double_dash_as_positional() -> None:
    assert path_candidates(["cat", "--", "-n"], inspect_positional_paths=True) == (
        "-n",
    )


def test_path_candidates_skip_chmod_mode_tokens() -> None:
    assert path_candidates(["chmod", "+x", "file"], inspect_positional_paths=True) == (
        "file",
    )


def test_path_candidates_exclude_sort_output_file() -> None:
    # "out" is consumed by -o; only the input file is a path candidate.
    assert path_candidates(
        ["sort", "-o", "out", "in"], inspect_positional_paths=True
    ) == ("in",)


def test_path_candidates_inspect_tree_positionals_via_policy() -> None:
    # tree declares inspect_positional_paths itself, so callers need not opt in.
    assert path_candidates(["tree", "."], inspect_positional_paths=False) == (".",)


def test_path_candidates_include_git_no_index_positionals() -> None:
    assert path_candidates(
        ["git", "diff", "--no-index", "a", "b"], inspect_positional_paths=False
    ) == ("a", "b")


def test_path_candidates_include_git_global_path_values() -> None:
    assert path_candidates(
        ["git", "--git-dir", "/repo", "log"], inspect_positional_paths=False
    ) == ("/repo",)


def test_path_candidates_empty_tokens() -> None:
    assert path_candidates([], inspect_positional_paths=True) == ()


@pytest.mark.parametrize(
    "tokens",
    [
        ["du", "--files0-from", "list"],
        ["du", "--files0-from=list"],
        ["wc", "--files0-from", "list"],
        ["md5sum", "--check", "sums"],
        ["md5sum", "-c", "sums"],
        ["sha1sum", "-c", "sums"],
        ["sha256sum", "-c", "sums"],
        ["shasum", "-c", "sums"],
        ["uniq", "a", "b"],
        ["uniq", "-f", "1", "a", "b"],
        ["less", "+!curl", "file"],
        ["less", "-k", "keys", "file"],
        ["less", "--lesskey-file", "keys"],
        ["less", "-t", "tag"],
        ["less", "--tag-file", "tags"],
        ["less", "--pattern=foo$-k", "file"],
        ["date", "06010203"],
        ["date", "-f", "%Y", "06010203"],
        ["file", "--files-from", "list"],
        ["file", "-f", "list"],
        ["file", "-z", "archive.gz"],
        ["file", "--compile"],
        ["git", "log", "--remerge-diff"],
        ["git", "log", "--show-signature"],
        ["git", "log", "--help"],
        ["git", "log", "-p", "--diff-merges=remerge"],
        ["git", "log", "--diff-merges", "r"],
    ],
)
def test_ported_policies_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["md5sum", "file.txt"],
        ["sha256sum", "file.txt"],
        ["uniq", "file.txt"],
        ["uniq", "-c", "file.txt"],
        ["uniq", "--skip-fields=1", "a"],
        ["less", "file.txt"],
        ["less", "-N", "file.txt"],
        ["less", "+/pattern", "file.txt"],
        ["less", "+G", "file.txt"],
        ["more", "file.txt"],
        ["date"],
        ["date", "+%Y"],
        ["date", "-u"],
        ["date", "-d", "tomorrow"],
        ["date", "-f", "stampfile"],
        ["file", "file.txt"],
        ["file", "-m", "magic"],
        ["git", "log"],
        ["git", "log", "--diff-merges=first-parent"],
    ],
)
def test_ported_policies_allow_benign_forms(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


def test_grep_exclude_from_is_a_path_value() -> None:
    policy = analyze_shell_command_policy([
        "grep",
        "--exclude-from",
        "patterns",
        "file",
    ])
    assert not policy.requires_approval
    assert policy.option_path_values == ("patterns",)


def test_diff_exclude_from_is_a_path_value() -> None:
    policy = analyze_shell_command_policy(["diff", "-X", "patterns", "a", "b"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("patterns",)

    attached = analyze_shell_command_policy([
        "diff",
        "--exclude-from=patterns",
        "a",
        "b",
    ])
    assert attached.option_path_values == ("patterns",)


def test_tree_gitfile_is_a_path_value() -> None:
    policy = analyze_shell_command_policy(["tree", "--gitfile", ".gitignore", "."])
    assert not policy.requires_approval
    assert policy.option_path_values == (".gitignore",)


def test_git_pathspec_from_file_is_a_path_value() -> None:
    policy = analyze_shell_command_policy([
        "git",
        "diff",
        "--pathspec-from-file",
        "list",
    ])
    assert not policy.requires_approval
    assert policy.option_path_values == ("list",)

    short = analyze_shell_command_policy(["git", "log", "-O", "list"])
    assert short.option_path_values == ("list",)


def test_policy_dispatch_normalizes_quoted_command_names() -> None:
    assert analyze_shell_command_policy(["'sort'", "-o", "out", "in"]).requires_approval
    assert analyze_shell_command_policy(["./git", "log"]).inspect_git_repository


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["tar", "-cf-", "--add-file=.env"], ".env"),
        (["dd", "if=.env"], ".env"),
        (["sed", "-n", "p", ".env"], ".env"),
    ],
)
def test_file_operands_are_inspected(tokens: list[str], expected: str) -> None:
    assert expected in path_candidates(tokens, inspect_positional_paths=True)


def test_less_option_resume_has_bounded_depth() -> None:
    assert analyze_shell_command_policy(["less", "-" + "$" * 2100]).requires_approval


@pytest.mark.parametrize(
    "command",
    [
        "sed 's/.*/touch pwned/e' input.txt",
        "sed 's/a/b/ge' input.txt",
        "sed 's/a/b/Z' input.txt",
        "sed -f script.sed input.txt",
        "sed -fscript.sed input.txt",
        "sed --file=script.sed input.txt",
        "sed 's/a/b/w .env' input.txt",
        "python -m uv pip install --target=/etc pkg",
        "python -m poetry add ../outside",
        "python -O -m uv pip install --target=/etc pkg",
        "python -S -m poetry add ../outside",
        "python -b -m pip install ../outside",
        "python -W ignore -m uv pip install -t/outside pkg",
        "python3 --check-hash-based-pycs always -m pip install ../outside",
        "python3 --check-hash-based-pycs=always -m uv pip install -t/outside pkg",
        "uv pip install -t/outside pkg",
        "uv pip install -e../outside",
        "yarn install --cache-folder /tmp/ycache",
        "pnpm install --store-dir /tmp/store",
        "cargo build --target-dir /tmp/target",
        "npm install --cache-dir ~/.npm pkg",
        "PIP_TARGET=/etc pip install pkg",
        "PIP_PREFIX=/etc pip install pkg",
        "PIP_CACHE_DIR=/tmp/pip-cache pip install pkg",
        "UV_PROJECT_ENVIRONMENT=/etc uv sync",
        "UV_CACHE_DIR=/tmp/uv-cache uv sync",
        "PIP_CONFIG_FILE=config pip install pkg",
        "UV_CONFIG_FILE=config uv pip install pkg",
        "UV_TARGET=/etc uv pip install pkg",
        "POETRY_CONFIG_DIR=config poetry install",
        "UV_WORKING_DIR=../outside uv pip install --target local pkg",
        "UV_PROJECT=../outside uv sync",
        "PIP_LOG=../outside pip install pkg",
        "git unknown-command",
    ],
)
def test_second_review_bypass_denials(bash_tool: Bash, command: str) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.NEVER
    assert "require approval" not in (context.reason or "")
    assert "side-effecting options" not in (context.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "sed 's/a/b/2gip' input.txt",
        "sed 's/a/b/IM' input.txt",
        "sed '/a/{s/a/b/g}' input.txt",
        "sed 's/a/b/g # comment' input.txt",
        "sed 's/a/b/w result.txt' input.txt",
        "sed -e 's/a/b/g' input.txt",
        "git diff > patch.txt",
        "git log 2> err.txt",
        "echo ok > output.txt; git status",
        "git log | grep foo > out.txt",
        "git rev-parse HEAD",
        "git ls-files",
        "git describe --tags",
        "git show-ref",
        "git for-each-ref",
        "git check-attr diff file.txt",
        "git check-ignore file.txt",
        "git check-ref-format refs/heads/main",
        "git check-mailmap 'Example <example@example.org>'",
        "git column",
        "git get-tar-commit-id",
        "git var GIT_EDITOR",
        "git ls-tree HEAD",
        "git rev-list HEAD",
        "git cat-file -t HEAD",
        "git count-objects",
        "git diff-files",
        "git diff-index HEAD",
        "git diff-tree HEAD",
        "git merge-base HEAD HEAD",
        "git name-rev HEAD",
        "git verify-pack objects/pack/example.idx",
        "git submodule status",
        "git help log",
        "git version",
        "git archive HEAD",
        "git notes list",
        "pipx install --python /usr/bin/python3 pkg",
        "pipx install --python=../python pkg",
        "npm install --cache-dir cache pkg",
        "pip install --target target pkg",
        "uv pip install -ttarget pkg",
        "uv pip install -e./local",
        "python -m uv pip install --target=target pkg",
        "python -m poetry add ./local",
        "python3 --check-hash-based-pycs always -m pip install ./local",
        "yarn install --cache-folder cache",
        "pnpm install --store-dir store",
        "cargo build --target-dir target",
        "PIP_DISABLE_PIP_VERSION_CHECK=1 pip install pkg",
        "UV_NO_PROGRESS=1 uv pip install pkg",
        "UV_NO_SYNC=1 uv sync",
    ],
)
def test_second_review_benign_forms(bash_tool: Bash, command: str) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.ALWAYS, context


# --- Newly inspected write/mutate commands (delta re-review, round 2) ---
#
# The resolver-layer unmodeled-file matrices (denied and benign) migrated to
# the shared fixtures in tests/tools/_git_apply_matrix.py (stage 12b-3); the
# resolver-layer assertions live in test_git_metadata_guard.py, whose
# matrices were strict supersets of this suite's twins.


# --- Round-3 closures (P10 final enumeration round) ---
#
# The same-family gaps the closure check named: binutils/assembler output
# operands, config-content writers, in-place compression mutators, archive
# creation/extraction destinations, mktemp, the round-3 program-selection
# config keys, and the worktree-move/submodule-add destinations. Grammar
# facts verified against the shipped tools (binutils 2.46, Info-ZIP zip 3.0,
# 7-Zip 26.00, unar 1.10.8, xz-utils 5.8.3, zstd 1.5.7, uutils mktemp 0.8.0,
# OpenSSH scp 10.2p1, GnuPG 2.4.8, libxml2 xmllint 2.15.2).


@pytest.mark.parametrize(
    "tokens,expected",
    [
        # strip: -o names the written result in the separate and attached
        # forms; every positional operand is mutated in place.
        (["strip", "-o", "out", "evil"], ("out", "evil")),
        (["strip", "-ores", "evil"], ("res", "evil")),
        (["strip", "evil"], ("evil",)),
        (["strip", "-R", ".text", "-o", "out", "evil"], ("out", "evil")),
        # objcopy: the positional output operand (and the in-place input).
        (["objcopy", "evil", "out"], ("evil", "out")),
        (["objcopy", "evil"], ("evil",)),
        # as: -o names the written object, separate and attached.
        (["as", "-o", "out.o", "evil.s"], ("out.o",)),
        (["as", "-ores.o", "evil.s"], ("res.o",)),
    ],
)
def test_binutils_output_operands_are_write_operands(
    tokens: list[str], expected: tuple[str, ...]
) -> None:
    policy = analyze_shell_command_policy(tokens)
    assert not policy.requires_approval
    candidates = path_candidates(tokens, inspect_positional_paths=True)
    for value in expected:
        assert value in candidates, (tokens, value)


@pytest.mark.parametrize(
    "tokens,expected",
    [
        # gpg: every -o/--output spelling names the written output file.
        (["gpg", "--decrypt", "-o", "out", "payload.gpg"], ("out",)),
        (["gpg", "--decrypt", "-ores", "payload.gpg"], ("res",)),
        (["gpg", "--decrypt", "--output", "out", "payload.gpg"], ("out",)),
        (["gpg", "--decrypt", "--output=out", "payload.gpg"], ("out",)),
        # xmllint: --output/-o name the written XML result.
        (["xmllint", "--output", "out", "doc.xml"], ("out",)),
        (["xmllint", "-o", "out", "doc.xml"], ("out",)),
        # scp: a local-to-local copy inspects BOTH operands (the ln rule);
        # value-taking options never surface as positional candidates.
        (["scp", "seed.txt", "copied.txt"], ("seed.txt", "copied.txt")),
        (["scp", "-P", "22", "seed.txt", "copied.txt"], ("seed.txt", "copied.txt")),
        (["scp", "-i", "id", "seed.txt", "host:x"], ("seed.txt", "host:x")),
    ],
)
def test_config_content_writer_operands_are_write_operands(
    tokens: list[str], expected: tuple[str, ...]
) -> None:
    policy = analyze_shell_command_policy(tokens)
    assert not policy.requires_approval
    candidates = path_candidates(tokens, inspect_positional_paths=True)
    for value in expected:
        assert value in candidates, (tokens, value)


@pytest.mark.parametrize(
    "tokens,expected",
    [
        # The in-place compression families: every positional operand is
        # replaced or annotated in place.
        (["bzip2", "seed.txt"], ("seed.txt",)),
        (["bunzip2", "seed.txt.bz2"], ("seed.txt.bz2",)),
        (["bzcat", "seed.txt.bz2"], ("seed.txt.bz2",)),
        (["xz", "seed.txt"], ("seed.txt",)),
        (["unxz", "seed.txt.xz"], ("seed.txt.xz",)),
        (["xzcat", "seed.txt.xz"], ("seed.txt.xz",)),
        (["lzma", "seed.txt"], ("seed.txt",)),
        (["unlzma", "seed.txt.lzma"], ("seed.txt.lzma",)),
        (["lzcat", "seed.txt.lzma"], ("seed.txt.lzma",)),
        (["lzmainfo", "seed.txt.lzma"], ("seed.txt.lzma",)),
        # zstd: the positionals mutate in place, and -o/--output-dir-*
        # name written destinations in the separate and `=` spellings.
        (["zstd", "seed.txt"], ("seed.txt",)),
        (["zstdmt", "seed.txt"], ("seed.txt",)),
        (["unzstd", "seed.txt.zst"], ("seed.txt.zst",)),
        (["zstdcat", "seed.txt.zst"], ("seed.txt.zst",)),
        (["zstd", "-o", "out.zst", "seed.txt"], ("out.zst", "seed.txt")),
        (["zstd", "--output-dir-flat=odir", "seed.txt"], ("odir", "seed.txt")),
        (["zstd", "--output-dir-mirror", "odm", "seed.txt"], ("odm", "seed.txt")),
    ],
)
def test_compression_mutator_operands_are_write_operands(
    tokens: list[str], expected: tuple[str, ...]
) -> None:
    policy = analyze_shell_command_policy(tokens)
    assert not policy.requires_approval
    candidates = path_candidates(tokens, inspect_positional_paths=True)
    for value in expected:
        assert value in candidates, (tokens, value)


@pytest.mark.parametrize(
    "tokens,expected",
    [
        # zip: the positional archive operand is created or replaced (the
        # value-taking options are consumed, never candidates).
        (["zip", "out.zip", "seed.txt"], ("out.zip", "seed.txt")),
        (["zip", "-r", "-q", "out.zip", "sub"], ("out.zip", "sub")),
        (["zip", "-P", "secret", "out.zip", "seed.txt"], ("out.zip", "seed.txt")),
        # 7z/7zr/7za: the positional archive operand, plus the attached -o
        # extraction destination (7z rejects the separate spelling, so its
        # value stays a positional candidate the scan still inspects).
        (["7z", "a", "out.7z", "seed.txt"], ("a", "out.7z", "seed.txt")),
        (["7zr", "a", "out.7z", "seed.txt"], ("a", "out.7z", "seed.txt")),
        (["7za", "a", "out.7z", "seed.txt"], ("a", "out.7z", "seed.txt")),
        (["7z", "x", "-ooutdir", "evil.7z"], ("outdir", "x", "evil.7z")),
        (["7z", "x", "-o", "outdir", "evil.7z"], ("outdir", "x", "evil.7z")),
        # unar: the four real -o spellings name the extraction directory.
        (["unar", "-f", "-o", "outdir", "evil.zip"], ("outdir",)),
        (["unar", "-f", "-o=outdir", "evil.zip"], ("outdir",)),
        (["unar", "-f", "-output-directory", "outdir", "evil.zip"], ("outdir",)),
        (["unar", "-f", "-output-directory=outdir", "evil.zip"], ("outdir",)),
    ],
)
def test_archive_destination_operands_are_write_operands(
    tokens: list[str], expected: tuple[str, ...]
) -> None:
    policy = analyze_shell_command_policy(tokens)
    assert not policy.requires_approval
    candidates = path_candidates(tokens, inspect_positional_paths=True)
    for value in expected:
        assert value in candidates, (tokens, value)


@pytest.mark.parametrize(
    "tokens,expected",
    [
        # mktemp: the template positionals name the created files, and
        # -p/--tmpdir name the directory they are created in (the long form
        # takes its directory only attached).
        (["mktemp", "tmpl.XXXXXX"], ("tmpl.XXXXXX",)),
        (["mktemp", "-p", "tmpdir", "tmpl.XXXXXX"], ("tmpdir", "tmpl.XXXXXX")),
        (["mktemp", "-ptmpdir", "tmpl.XXXXXX"], ("tmpdir", "tmpl.XXXXXX")),
        (["mktemp", "--tmpdir=tmpdir", "tmpl.XXXXXX"], ("tmpdir", "tmpl.XXXXXX")),
        (["mktemp", "-d", "-p", "tmpdir"], ("tmpdir",)),
    ],
)
def test_mktemp_creation_operands_are_write_operands(
    tokens: list[str], expected: tuple[str, ...]
) -> None:
    policy = analyze_shell_command_policy(tokens)
    assert not policy.requires_approval
    candidates = path_candidates(tokens, inspect_positional_paths=True)
    for value in expected:
        assert value in candidates, (tokens, value)


# --- Shell policy v0.2 (ADR 0021): occurrence-local cd+redirect chains and
# --- scratchpad absolute redirect targets ---


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "sub").mkdir()
    (root / "sub" / "nested").mkdir()
    (root / "file.txt").write_text("x\n")
    (root / "sub" / "file.txt").write_text("x\n")
    return root


@pytest.fixture
def scratchpad_dir(tmp_path: Path) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return scratch


def _v02_tool(
    cwd: Path,
    scratchpad: Path | None = None,
    *,
    denylist: list[str] | None = None,
    denylist_standalone: list[str] | None = None,
) -> Bash:
    config = BashToolConfig(
        denylist=denylist if denylist is not None else ["bash -i", "curl"],
        denylist_standalone=(
            denylist_standalone if denylist_standalone is not None else ["python"]
        ),
    )
    return Bash(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=cwd,
        scratchpad_dir=scratchpad,
    )


def _v02_verdict(tool: Bash, command: str) -> ToolPermission:
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    return result.permission


def _v02_reason(tool: Bash, command: str) -> str:
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    return result.reason or ""


@pytest.mark.parametrize(
    "command",
    [
        # Compound investigation chains: cd, pipeline, separators (ADR 0021).
        "cd sub && grep x file.txt | head; echo ---; git branch -a",
        "cd sub && grep x file.txt 2>&1 | head; echo ---; git branch -a",
        "cd sub && grep x file.txt > /dev/null",
        "cd sub && grep x file.txt > out.log",
        "cd sub && grep x file.txt >> out.log 2> err.log",
        "cd sub; grep x file.txt > out.log",
        "cd sub && grep x file.txt > out.log; cd sub/nested && grep y file.txt > y.log",
        # A redirect before a cd resolves against the pre-cd working directory.
        "grep x file.txt > out.log && cd sub",
        "grep x file.txt > out.log && pushd sub",
        # A redirect attached to the cd command itself resolves against the
        # pre-cd working directory (bash opens the file before running cd).
        "cd sub > log",
        "pushd sub > log",
        # Nested sources resolve occurrence-locally too.
        "bash -c 'cd sub && grep x file.txt > out.log'",
        "bash -c 'grep x file.txt > out.log && cd sub'",
    ],
)
def test_cd_chains_with_redirects_are_allowed(workspace: Path, command: str) -> None:
    tool = _v02_tool(workspace)
    assert _v02_verdict(tool, command) is ToolPermission.ALWAYS, command


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        # The redirect target itself leaves the workspace.
        (
            "cd sub && grep x file.txt > ../outside.log",
            "outside the authorized workspace",
        ),
        # A cd outside the workspace is still denied, redirect or not.
        ("grep x file.txt > out.log && cd /tmp", "outside the authorized workspace"),
        ("cd /tmp > log", "outside the authorized workspace"),
        ("cd /tmp && grep x file.txt > out.log", "outside the authorized workspace"),
        # Dynamic cd forms stay fail-closed (documented residual): the
        # unresolvable cd makes the relative redirect target's working
        # directory unknown, and the analysis denies the expansion itself.
        ("cd $HOME && grep x file.txt > out.log", "not statically known"),
        ("cd $(pwd) && grep x file.txt > out.log", "not statically known"),
        # A redirect after a cd that cannot be statically resolved fails
        # closed on the unknown working directory.
        ("popd +1 && grep x file.txt > out.log", "not statically known"),
        ("cd - && grep x file.txt > out.log", "not statically known"),
        # Heredocs stay fail-closed (documented residual).
        ("cd sub && cat <<EOF > out.log\nhi\nEOF", "heredocs are unsupported"),
    ],
)
def test_cd_chains_with_redirects_still_fail_closed(
    workspace: Path, command: str, reason: str
) -> None:
    tool = _v02_tool(workspace)
    assert _v02_verdict(tool, command) is ToolPermission.NEVER, command
    assert reason in _v02_reason(tool, command), command


@pytest.mark.parametrize(
    "command",
    [
        "echo hi > {scratch}/log.txt",
        "grep x file.txt > {scratch}/log.txt",
        "grep x file.txt 2> {scratch}/err.log",
        "cd sub && grep x file.txt > {scratch}/log.txt",
        "grep x file.txt > {scratch}/log.txt && cd sub",
        "cd {scratch} && echo hi > log.txt",
        "bash -c 'echo hi > {scratch}/log.txt'",
    ],
)
def test_absolute_scratchpad_redirect_targets_are_allowed(
    workspace: Path, scratchpad_dir: Path, command: str
) -> None:
    tool = _v02_tool(workspace, scratchpad_dir)
    resolved = command.format(scratch=scratchpad_dir)
    assert _v02_verdict(tool, resolved) is ToolPermission.ALWAYS, resolved


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        # Outside workspace and scratchpad.
        ("echo hi > /etc/x", "outside the authorized workspace"),
        # Absolute targets inside the workspace but outside the scratchpad
        # stay denied: only scratchpad targets are authorized (ADR 0021).
        ("echo hi > {workspace}/abs.log", "an absolute redirection target"),
        # A sibling directory that merely spells like the scratchpad root:
        # the path layer denies it as outside workspace+scratchpad, and the
        # analysis would deny it as an absolute target under any roots.
        (
            "echo hi > {workspace}/../scratch-sibling/log.txt",
            "outside the authorized workspace",
        ),
        # A path that escapes the scratchpad root through a parent component.
        ("echo hi > {scratch}/../outside.log", "outside the authorized workspace"),
        # Sensitive patterns still apply inside the scratchpad.
        ("echo hi > {scratch}/.env", "Sensitive file"),
    ],
)
def test_absolute_redirect_targets_outside_scratchpad_are_denied(
    workspace: Path, scratchpad_dir: Path, command: str, reason: str
) -> None:
    (workspace.parent / "scratch-sibling").mkdir()
    tool = _v02_tool(workspace, scratchpad_dir)
    resolved = command.format(workspace=workspace, scratch=scratchpad_dir)
    assert _v02_verdict(tool, resolved) is ToolPermission.NEVER, resolved
    assert reason in _v02_reason(tool, resolved), resolved


def test_absolute_scratchpad_redirect_symlink_escape_is_denied(
    workspace: Path, scratchpad_dir: Path, tmp_path: Path
) -> None:
    # A symlink inside the scratchpad that points outside it is not a
    # scratchpad target: the full realpath is matched, component-wise.
    (scratchpad_dir / "escape").symlink_to(tmp_path, target_is_directory=True)
    tool = _v02_tool(workspace, scratchpad_dir)
    command = f"echo hi > {scratchpad_dir}/escape/log.txt"
    assert _v02_verdict(tool, command) is ToolPermission.NEVER, command
    assert "outside the authorized workspace" in _v02_reason(tool, command)


def test_absolute_scratchpad_redirect_follows_changed_roots(
    workspace: Path, scratchpad_dir: Path, tmp_path: Path
) -> None:
    # The roots are read from the live path authority per resolution.
    tool = _v02_tool(workspace, scratchpad_dir)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    command = f"echo hi > {scratchpad_dir}/log.txt"
    assert _v02_verdict(tool, command) is ToolPermission.ALWAYS

    other = _v02_tool(workspace, replacement)
    assert _v02_verdict(other, command) is ToolPermission.NEVER
    assert (
        _v02_verdict(other, f"echo hi > {replacement}/log.txt") is ToolPermission.ALWAYS
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # The <redirect> marker keeps a redirected standalone-denylisted
        # command runnable (ADR 0021: marker semantics preserved).
        ("python > log.txt", ToolPermission.ALWAYS),
        ("python script.py > log.txt", ToolPermission.ALWAYS),
        ("python", ToolPermission.NEVER),
        # Prefix denylisting is unaffected by the marker.
        ("curl https://x.test/ > log.txt", ToolPermission.NEVER),
        ("curl https://x.test/", ToolPermission.NEVER),
    ],
)
def test_redirect_marker_standalone_and_prefix_semantics_are_preserved(
    workspace: Path, command: str, expected: ToolPermission
) -> None:
    tool = _v02_tool(workspace)
    assert _v02_verdict(tool, command) is expected, command
    if expected is ToolPermission.NEVER:
        reason = _v02_reason(tool, command)
        assert "standalone" in reason or "denylist" in reason, reason


# --- P11 fix round: redirect-target glob exemption, concatenation-target
# --- attribution, and script-loading operand inspection ---


@pytest.mark.parametrize(
    "command",
    [
        # Shells pathname-expand redirect destinations (zsh writes every
        # match; bash overwrites a single match or creates the literal name),
        # so a target with any UNQUOTED glob metacharacter fails closed.
        # C2b restores fully quoted targets as literal WRITE operands below.
        "echo x > f[0]",
        'echo x > "f"[0]',
        'echo x > "sub"/f[0]',
        "echo x > out[1].log",
    ],
)
def test_glob_looking_redirect_targets_fail_closed(
    workspace: Path, command: str
) -> None:
    tool = _v02_tool(workspace)
    assert _v02_verdict(tool, command) is ToolPermission.NEVER, command
    assert "contains shell glob characters" in _v02_reason(tool, command), command


@pytest.mark.parametrize(
    "command",
    [
        # C2b (D1 exit criterion): every glob metacharacter inside a quoted
        # segment is literal data — the shell writes the literal name, so the
        # target resolves as a literal WRITE operand.
        'echo x > f"*"',
        'echo x > "f[0]"',
    ],
)
def test_fully_quoted_redirect_targets_are_literal_operands(
    workspace: Path, command: str
) -> None:
    tool = _v02_tool(workspace)
    assert _v02_verdict(tool, command) is ToolPermission.ALWAYS, command


@pytest.mark.parametrize(
    "command",
    [
        # The path-glob heuristic still fires for command operands, where
        # the shell really globs.
        "cat f[0]",
        "cp f* out",
        "ls f[0]",
    ],
)
def test_glob_command_operands_still_fail_closed(workspace: Path, command: str) -> None:
    tool = _v02_tool(workspace)
    assert _v02_verdict(tool, command) is ToolPermission.NEVER, command
    assert "glob" in _v02_reason(tool, command), command


def test_concatenation_redirect_target_escaping_workspace_is_denied(
    workspace: Path,
) -> None:
    # Pre-v0.2 bypass closed by the broadened _literal_redirect_target: at
    # HEAD only word/string/raw_string destinations were extracted, so a
    # concatenation destination was never path-checked and this wrote
    # outside the workspace.
    tool = _v02_tool(workspace)
    command = 'echo x > "sub"/../../outside.log'
    assert _v02_verdict(tool, command) is ToolPermission.NEVER, command
    assert "outside the authorized workspace" in _v02_reason(tool, command), command


def test_source_operands_are_path_inspected(
    workspace: Path, scratchpad_dir: Path, tmp_path: Path
) -> None:
    # source/. read their operand as shell source, so the script path joins
    # the path-inspected surface as a READ operand: outside scripts are
    # denied, benign in-workspace and scratchpad scripts stay allowed.
    (workspace / "sub" / "setup.sh").write_text("echo setup\n")
    (workspace / "lib.sh").write_text("echo lib\n")
    (scratchpad_dir / "env.sh").write_text("echo env\n")
    (tmp_path / "evil.sh").write_text("echo evil\n")
    tool = _v02_tool(workspace, scratchpad_dir)

    denied = [
        # Absolute outside script, with and without a redirect.
        f"source {tmp_path}/evil.sh",
        f"source {tmp_path}/evil.sh > out",
        # The dot builtin reads the same operand.
        f". {tmp_path}/evil.sh",
        # Relative escape from the workspace.
        "source ../evil.sh",
        # The cd+redirect variant: the sourced path resolves against the
        # tracked cwds (the monotonic union keeps the pre-cd directory a
        # candidate), so the outside script is denied there too.
        "cd sub && source ../evil.sh > out",
    ]
    for command in denied:
        assert _v02_verdict(tool, command) is ToolPermission.NEVER, command
        assert "outside the authorized workspace" in _v02_reason(tool, command), command

    allowed = [
        "source sub/setup.sh",
        f"source {scratchpad_dir}/env.sh",
        ". ./lib.sh",
        "source sub/setup.sh > out",
    ]
    for command in allowed:
        assert _v02_verdict(tool, command) is ToolPermission.ALWAYS, command
