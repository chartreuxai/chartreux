from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

import pytest

from chartreux.core.dangerous_commands import DangerousCommandAuthority
from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    EXECUTOR_REGISTRY,
    GitRepositoryRelocation,
    _package_module_tokens,
    analyze_shell_command_policy,
    canonical_git_command,
    executor_boundary,
    git_cwd_relocations,
    git_repository_config_risk,
    git_work_tree_writing_subcommand,
    inline_interpreter_switch,
    matches_command_prefix,
    path_candidates,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig


@pytest.mark.parametrize("selector", ["/usr/bin/printf", "../outside.sh"])
@pytest.mark.parametrize("prefix", ["uv run ", "pipx run "])
def test_executor_filesystem_selector_matches_direct_denial(
    bash_tool: Bash, prefix: str, selector: str
) -> None:
    direct = bash_tool.resolve_permission(BashArgs(command=selector + " hello"))
    wrapped = bash_tool.resolve_permission(
        BashArgs(command=prefix + selector + " hello")
    )
    assert direct is not None and wrapped is not None
    assert direct.permission == wrapped.permission == ToolPermission.NEVER
    assert direct.reason == wrapped.reason


@pytest.mark.parametrize(
    "command, expected",
    [
        ("uv run ./script-in-workspace.sh", ToolPermission.ALWAYS),
        ("uv run python script.py", ToolPermission.ALWAYS),
        ("uv run python ../outside.py", ToolPermission.NEVER),
    ],
)
def test_executor_scoped_script_paths(
    bash_tool: Bash, command: str, expected: ToolPermission
) -> None:
    result = bash_tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission == expected


@pytest.mark.parametrize(
    "prefix",
    [
        "uv run ",
        "python -m uv run ",
        "python3 -m uv run ",
        "python -m uv run python3 -m uv run ",
    ],
)
@pytest.mark.parametrize(
    "inner, expected",
    [
        ('python -c "print(1 * 2)"', ToolPermission.ALWAYS),
        ("rm -rf file", ToolPermission.NEVER),
        ("python ../outside.py", ToolPermission.NEVER),
        ("/usr/bin/printf hello", ToolPermission.NEVER),
        ("pip install requests[socks]", ToolPermission.NEVER),
        ('pip install "requests[socks]"', ToolPermission.ALWAYS),
    ],
)
def test_module_executor_ownership_matches_direct_form(
    bash_tool: Bash, prefix: str, inner: str, expected: ToolPermission
) -> None:
    result = bash_tool.resolve_permission(BashArgs(command=prefix + inner))
    assert result is not None and result.permission == expected


@pytest.mark.parametrize("option", ["-p", "-P", "-u"])
@pytest.mark.parametrize("suffix", ["", " printf hello"])
def test_legacy_ionice_process_selection_remains_allowed(
    bash_tool: Bash, option: str, suffix: str
) -> None:
    result = bash_tool.resolve_permission(
        BashArgs(command=f"ionice {option} 1{suffix}")
    )
    assert result is not None and result.permission == ToolPermission.ALWAYS


def test_unknown_ionice_option_fails_closed(bash_tool: Bash) -> None:
    result = bash_tool.resolve_permission(
        BashArgs(command="ionice --unknown 1 printf hello")
    )
    assert result is not None and result.permission == ToolPermission.NEVER


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


@pytest.mark.parametrize(
    "command,expected",
    [
        ('uv run python -c "print(1 * 2)"', ()),
        ('python -c "print(1 * 2)"', ()),
        ("python script.py", ("script.py",)),
        ("uv run --directory child python script.py", ("child",)),
        ("uv run -p /usr/bin/python3 echo hi", ()),
        ("uv run --with ./fixture echo hi", ("./fixture",)),
        ("cargo run --bin app --target triple", ()),
        ("npx fixture arg", ("fixture",)),
        ("npx ./fixture arg", ("./fixture",)),
        ("pipx run --spec fixture app ../not-an-executable", ("fixture",)),
        ("go run ./main.go rm -r", ("./main.go", "rm", "-r")),
        (
            "cargo run --manifest-path ./Cargo.toml -- rm -r",
            ("./Cargo.toml", "rm", "-r"),
        ),
    ],
)
def test_executor_path_candidates_respect_argument_ownership(
    command: str, expected: tuple[str, ...]
) -> None:
    assert (
        path_candidates(shlex.split(command), inspect_positional_paths=False)
        == expected
    )


@pytest.mark.parametrize(
    "command",
    [
        "uv run --script script.py",
        "uv run script.py",
        "uv --directory child run --directory other python",
        "go run -C child main.go",
    ],
)
def test_executor_unmodeled_execution_modes_and_directory_ambiguity_deny(
    command: str,
) -> None:
    with pytest.raises(ValueError):
        executor_boundary(shlex.split(command))


@pytest.mark.parametrize(
    "command",
    [
        "env -i echo hi",
        "env echo hi",
        "busybox ls",
        "script -c 'echo hi'",
        "bash -- -c 'echo hi'",
    ],
)
def test_executor_legacy_admission_restrictions_are_registry_owned(
    command: str,
) -> None:
    with pytest.raises(ValueError):
        executor_boundary(shlex.split(command), admission=True)


@pytest.mark.parametrize(
    "command, start, kind, cwd",
    [
        ("uv run python -c code", 2, "argv_after_keyword", ()),
        (
            "uv -qv --directory=child run --no-sync -p3.12 -- python -c code",
            7,
            "argv_after_keyword",
            ("child",),
        ),
        ("uv run --directory child python", 4, "argv_after_keyword", ("child",)),
        ("npx -ypfixture app --unknown arg", 2, "argv_positional", ()),
        ("npx --package=fixture -- app -x", 3, "argv_positional", ()),
        ("npx fixture arg", 1, "argv_positional", ()),
        (
            "pipx -v run --spec=fixture --python python3 app -x",
            6,
            "argv_after_keyword",
            (),
        ),
        ("pipx run --spec fixture app", 4, "argv_after_keyword", ()),
        ("env -iCchild KEY=value python -c code", 3, "argv_positional", ("child",)),
        ("env -- KEY=value python -c code", 3, "argv_positional", ()),
        ("env - KEY=value python -c code", 3, "argv_positional", ()),
        (
            "env --chdir=child --unset HOME KEY=value -- python",
            6,
            "argv_positional",
            ("child",),
        ),
        ("exec -claalias python", 2, "argv_positional", ()),
        ("command -p -- python", 3, "argv_positional", ()),
        ("builtin printf x", 1, "argv_positional", ()),
        ("busybox sh -c code", 1, "argv_positional", ()),
        ("timeout -vk2 --signal=TERM 1 python", 4, "argv_positional", ()),
        ("nice -n5 python", 2, "argv_positional", ()),
        ("nice -5 python", 2, "argv_positional", ()),
        ("stdbuf -oL -e0 python", 3, "argv_positional", ()),
        ("flock -xnw2 lock python", 3, "argv_positional", ()),
        ("setsid -fw python", 2, "argv_positional", ()),
        ("nohup -- python", 2, "argv_positional", ()),
        ("ionice -tc2 -n0 python", 3, "argv_positional", ()),
        ("taskset -ac 0,1 python", 3, "argv_positional", ()),
        ("time -vf%s python", 2, "argv_positional", ()),
    ],
)
def test_executor_argv_boundaries(
    command: str, start: int, kind: str, cwd: tuple[str, ...]
) -> None:
    tokens = shlex.split(command)
    boundary = executor_boundary(tokens)
    assert boundary is not None
    assert boundary.boundary_kind == kind
    assert boundary.inner_command_range == (start, len(tokens))
    assert boundary.consumed_ranges == ((0, start),)
    assert boundary.inline_payload_ranges == ()
    assert boundary.inline_payloads == ()
    assert boundary.module_targets == ()
    assert boundary.native_targets == ()
    assert boundary.cwd_overrides == cwd
    # Inner options are opaque here: their grammar belongs to the inner command.
    assert boundary.inner_command_range is not None
    assert tokens[start:] == tokens[slice(*boundary.inner_command_range)]


@pytest.mark.parametrize(
    "command, kind, payload_range, payload, module, end",
    [
        (
            "python -IWignore -c 'print(\"*.py\")' arg",
            "inline_code_switch",
            (3, 4),
            'print("*.py")',
            (),
            4,
        ),
        ("python3 -BcCODE arg", "inline_code_switch", (1, 2), "CODE", (), 2),
        ("pypy -c CODE", "inline_code_switch", (2, 3), "CODE", (), 3),
        ("pypy3 -cCODE", "inline_code_switch", (1, 2), "CODE", (), 2),
        (
            "node --require=fixture --eval=CODE arg",
            "inline_code_switch",
            (2, 3),
            "CODE",
            (),
            3,
        ),
        ("node -ipCODE", "inline_code_switch", (1, 2), "CODE", (), 2),
        ("perl -wle CODE arg", "inline_code_switch", (2, 3), "CODE", (), 3),
        ("ruby -Ilib -weCODE arg", "inline_code_switch", (2, 3), "CODE", (), 3),
        (
            "bash --noprofile -ic 'echo *.py' name arg",
            "shell_source",
            (3, 4),
            "echo *.py",
            (),
            4,
        ),
        ("sh -c code name", "shell_source", (2, 3), "code", (), 3),
        ("bash +O extglob -c code", "shell_source", (4, 5), "code", (), 5),
        ("python -Wignore -Impip arg", "module_target", None, None, ("pip",), 3),
        (
            "python3 --check-hash-based-pycs=always -m fixture arg",
            "module_target",
            None,
            None,
            ("fixture",),
            4,
        ),
    ],
)
def test_executor_payload_and_module_ranges(
    command: str,
    kind: str,
    payload_range: tuple[int, int] | None,
    payload: str | None,
    module: tuple[str, ...],
    end: int,
) -> None:
    tokens = shlex.split(command)
    boundary = executor_boundary(tokens)
    assert boundary is not None
    assert boundary.boundary_kind == kind
    assert boundary.inner_command_range is None
    assert boundary.consumed_ranges == ((0, end),)
    assert boundary.inline_payload_ranges == (
        () if payload_range is None else (payload_range,)
    )
    assert boundary.inline_payloads == (() if payload is None else (payload,))
    assert boundary.module_targets == module
    assert boundary.cwd_overrides == ()
    assert boundary.program_argument_range == (end, len(tokens))
    if payload_range is not None:
        # A payload is executor-owned even if its syntax resembles a path/glob.
        assert payload_range[1] <= end


@pytest.mark.parametrize(
    "command, targets, args_start, cwd",
    [
        ("go run main.go other.go rm -r x", ("main.go", "other.go"), 4, ()),
        ("go -C child run -tags=fixture ./cmd rm -r x", ("./cmd",), 6, ("child",)),
        ("go run fixture@v1 arg", ("fixture@v1",), 3, ()),
        ("go run -- main.go arg", ("main.go",), 4, ()),
        (
            "cargo run --manifest-path=child/Cargo.toml -pfixture --bin app -- rm -r x",
            ("child/Cargo.toml", "fixture", "app"),
            7,
            (),
        ),
        ("cargo -q run --release", (), 4, ()),
        ("cargo run --example demo -- arg", ("demo",), 5, ()),
    ],
)
def test_executor_native_target_semantics(
    command: str, targets: tuple[str, ...], args_start: int, cwd: tuple[str, ...]
) -> None:
    tokens = shlex.split(command)
    boundary = executor_boundary(tokens)
    assert boundary is not None
    assert boundary.boundary_kind == "none"
    assert boundary.inner_command_range is None
    assert boundary.native_targets == targets
    assert boundary.program_argument_range == (args_start, len(tokens))
    assert boundary.consumed_ranges == ((0, args_start),)
    assert boundary.inline_payload_ranges == ()
    assert boundary.module_targets == ()
    assert boundary.cwd_overrides == cwd


@pytest.mark.parametrize("name", sorted(EXECUTOR_REGISTRY))
@pytest.mark.parametrize("unknown", ["--unknown", "--unknown=value", "-Z"])
def test_executor_unknown_options_cannot_manufacture_a_boundary(
    name: str, unknown: str
) -> None:
    grammar = EXECUTOR_REGISTRY[name]
    if name == "eval":
        return  # eval's entire argument list is literal shell source, not options.
    prefix = [name, *([grammar.keyword] if grammar.keyword else [])]
    with pytest.raises(ValueError):
        executor_boundary([*prefix, unknown, "python", "-c", "CODE"])


@pytest.mark.parametrize(
    "command",
    [
        "uv run --directory",
        "uv run --directory=",
        "uv run --directory -- python",
        "uv run --dir child python",
        "uv run --unknown child python",
        "uv run -qZ python",
        "uv run --no-sync=true python",
        "uv run",
        "npx --package",
        "npx -p",
        "npx --package=",
        "npx --pa fixture app",
        "pipx run --spec",
        "pipx run --spec -- app",
        "pipx run --spec=fixture",
        "go run -tags",
        "go run -tags=",
        "go run -exec python main.go",
        "go run -unknown main.go",
        "go run --tags fixture main.go",
        "go run",
        "cargo run --bin",
        "cargo run --bin=",
        "cargo run app",
        "cargo run --b app",
        "timeout -k",
        "timeout 1",
        "env -u",
        "exec -a",
        "stdbuf -o",
        "python -W",
        "python -c",
        "python -m",
        "python -m -c CODE",
        "python --unknown -c CODE",
        "node --input-type",
        "node --eval=",
        "perl -e CODE -e MORE",
        "ruby -e CODE -e MORE",
        "bash -ic",
        "bash -cCODE",
    ],
)
def test_executor_missing_and_ambiguous_forms_fail_closed(command: str) -> None:
    with pytest.raises(ValueError):
        executor_boundary(shlex.split(command))


@pytest.mark.parametrize(
    "command",
    [
        "python script.py -c code",
        "python -- -c code",
        "node script.js --eval code",
        "bash script.sh -c code",
    ],
)
def test_executor_never_reads_script_arguments_as_selectors(command: str) -> None:
    boundary = executor_boundary(shlex.split(command))
    assert boundary is not None and boundary.boundary_kind == "none"
    assert boundary.inner_command_range is None
    assert boundary.inline_payload_ranges == ()
    assert boundary.module_targets == ()


def test_executor_eval_source_and_non_execution_forms() -> None:
    boundary = executor_boundary(["eval", "echo", "*.py"])
    assert boundary is not None
    assert boundary.boundary_kind == "shell_source"
    assert boundary.inline_payload_ranges == ((1, 3),)
    assert boundary.inline_payloads == ("echo *.py",)
    assert boundary.consumed_ranges == ((0, 3),)
    assert boundary.inner_command_range is None
    assert executor_boundary(["ls", "-l"]) is None
    assert executor_boundary([]) is None
    for tokens in [
        ["uv", "pip", "install", "fixture"],
        ["cargo", "build"],
        ["go", "test"],
        ["pipx", "install", "fixture"],
        ["command", "-v", "python"],
    ]:
        boundary = executor_boundary(tokens)
        assert boundary is not None and boundary.boundary_kind == "none"
        assert boundary.inner_command_range is None
    # Registering execution grammar does not change existing package dispatch.
    assert analyze_shell_command_policy([
        "uv",
        "pip",
        "install",
        "../fixture",
    ]).positional_values == ("pip", "install", "../fixture")
    assert _package_module_tokens(["python", "-Impip", "install"]) == ["pip", "install"]
    assert inline_interpreter_switch(["python", "-BcCODE"]) == "python -c"


@pytest.mark.parametrize(
    "command",
    [
        "npm install",
        "pnpm add",
        "yarn add",
        "pip install",
        "pip3 install",
        "cargo install",
        "go install",
        "uv pip install",
        "python -m pip install",
        "python3 -m pip install",
        "python3 -B -W ignore -m pip install",
        "python3 -mpip install",
        "python3 -umpip install",
        "python3 -um pip install",
        "poetry add",
        "pipx install",
        "bun add",
        "npx",
        "tee",
    ],
)
@pytest.mark.parametrize("operand", ["package", "./package", "../package"])
def test_package_and_tee_operands_are_workspace_scoped(
    bash_tool: Bash, command: str, operand: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=f"{command} {operand}"))
    assert context is not None
    assert context.permission is (
        ToolPermission.NEVER if operand.startswith("../") else ToolPermission.ALWAYS
    )
    outside = bash_tool.cwd.parent / "outside-package"
    context = bash_tool.resolve_permission(
        BashArgs(command=f"{command} {shlex.quote(str(outside))}")
    )
    assert context is not None and context.permission is ToolPermission.NEVER


@pytest.mark.parametrize("command", ["npm install", "pip install", "uv pip install"])
@pytest.mark.parametrize("prefix", ["file:", "file://", "file:///"])
def test_package_url_absolute_paths_are_workspace_scoped(
    bash_tool: Bash, command: str, prefix: str
) -> None:
    for path, expected in [
        (bash_tool.cwd / "package", ToolPermission.ALWAYS),
        (bash_tool.cwd.parent / "outside-package", ToolPermission.NEVER),
        (bash_tool.cwd / ".env", ToolPermission.NEVER),
    ]:
        # file:// + an absolute path yields the conventional file:/// URL.
        url = prefix + (str(path).lstrip("/") if prefix == "file:///" else str(path))
        context = bash_tool.resolve_permission(BashArgs(command=f"{command} {url}"))
        assert context is not None and context.permission is expected


@pytest.mark.parametrize(
    "operand, expected",
    [
        ("file:./package", ToolPermission.ALWAYS),
        ("file:../../outside-package", ToolPermission.NEVER),
        ("file:%2e%2e/outside-package", ToolPermission.NEVER),
        ("fixture@file:../outside-package", ToolPermission.NEVER),
        ("'fixture @ file:../outside-package'", ToolPermission.NEVER),
        ("file://server/package", ToolPermission.NEVER),
        ("file:./package?query", ToolPermission.NEVER),
        ("file:./package#fragment", ToolPermission.NEVER),
        ("file:%00package", ToolPermission.NEVER),
        ("file:%invalid", ToolPermission.NEVER),
        ("git+file:///tmp/package", ToolPermission.NEVER),
        ("file:", ToolPermission.NEVER),
    ],
)
def test_package_url_relative_and_unmodeled_forms(
    bash_tool: Bash, operand: str, expected: ToolPermission
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=f"npm install {operand}"))
    assert context is not None and context.permission is expected


def test_package_url_paths_are_normalized_before_resolution() -> None:
    assert "../package" in path_candidates(
        ["pip", "install", "file:%2e%2e/package"], inspect_positional_paths=False
    )
    assert "../package" in path_candidates(
        ["pip", "install", "--editable=file:../package"], inspect_positional_paths=False
    )


@pytest.mark.parametrize(
    "command",
    [
        "fsck",
        "bundle list-heads",
        "gc --auto",
        "maintenance run",
        "sparse-checkout list",
    ],
)
def test_unmodeled_git_builtins_are_not_described_as_aliases(
    bash_tool: Bash, command: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=f"git {command}"))
    assert context is not None and context.permission is ToolPermission.NEVER
    assert context.reason and "unmodeled git builtin" in context.reason
    assert "denied conservatively" in context.reason
    assert "permission settings cannot override" in context.reason
    assert "aliases" not in context.reason


@pytest.mark.parametrize(
    "command",
    [
        # git resolves repeated booleans in order and accepts --no-
        # spellings, so the final --no-dry-run cancels --dry-run (or -n) and
        # re-enables the destructive clean: the shipped policy layer must
        # require approval for it too, not only the classifier.
        "git clean --force --dry-run --no-dry-run",
        "git clean --force -n --no-dry-run",
        "git clean -f --dry-run --no-dry-run",
        "git clean -fd --no-dry-run",
        "git clean -fd",
        "git clean -f -d",
    ],
)
def test_clean_dry_run_negation_requires_approval(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert policy.requires_approval


@pytest.mark.parametrize(
    "command",
    [
        # Effective last-wins semantics: --no-force cancels --force (git
        # itself refuses to clean), and a trailing -n wins.
        "git clean --force --no-force",
        "git clean -f --no-force",
        "git clean --no-dry-run -n",
        "git clean -n",
        "git clean -fdn",
        "git clean -n -f",
        "git clean",
    ],
)
def test_negated_clean_flags_stay_permitted(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert not policy.requires_approval


@pytest.mark.parametrize(
    "command",
    [
        "git config remote.origin.mirror true",
        "git config --global remote.x.push +HEAD:main",
        "git config set remote.origin.mirror true",
        "git config --unset remote.origin.mirror",
        "git config -e",
        # The remote definition, the program a push runs, url rewrites, and
        # the bare push destination are push-effecting too.
        "git config remote.origin.url https://example.com/x.git",
        "git config remote.origin.pushurl ../evil.git",
        "git config remote.origin.receivepack echo",
        "git config remote.origin.uploadpack echo",
        "git config remote.origin.proxy echo",
        "git config url.https://evil/.pushInsteadOf https://github.com/",
        "git config url.https://evil/.insteadOf https://github.com/",
        "git config branch.main.remote origin",
        "git config branch.main.pushremote ../evil.git",
        "git config --unset remote.origin.url",
        "git config set branch.main.remote origin",
        "git config --remove-section url.https://evil/",
        "git config --rename-section branch.old branch.new",
        # The remote's vcs helper: git remote update executes the configured
        # git-remote-<vcs> helper from PATH.
        "git config remote.origin.vcs evil",
        "git config --unset remote.origin.vcs",
        # The program-execution family: the ssh command (even a --dry-run),
        # the git:// proxy, the ask-pass program, and the credential helpers.
        "git config core.sshCommand evil",
        "git config core.gitProxy evil",
        "git config core.askPass evil",
        "git config set core.gitProxy evil",
        "git config credential.helper evil",
        "git config credential.https://example.com.helper evil",
        "git config --unset credential.helper",
        # An authored include folds a file into configuration resolution,
        # which can set any push-effecting key for a verified remote.
        "git config include.path /tmp/evil.cfg",
        "git config includeIf.gitdir:~/work/.path /tmp/evil.cfg",
        "git config set include.path /tmp/evil.cfg",
        "git config --unset include.path",
        # Section operations reaching the credential and include sections.
        "git config --remove-section credential",
        "git config --remove-section include",
        "git config --rename-section includeIf.gitdir:/x includeIf.gitdir:/y",
    ],
)
def test_push_effecting_config_writes_are_denied(bash_tool: Bash, command: str) -> None:
    # A hard guard: denied with no grantable proposal, so no answer can
    # authorize the write.
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason


def test_config_then_push_compound_is_denied_without_a_proposal(
    bash_tool: Bash,
) -> None:
    # The same-call exploit: writing the config that turns an ordinary push
    # into a mirror push is denied on the config segment, and the whole
    # compound call carries no proposal.
    context = bash_tool.resolve_permission(
        BashArgs(command="git config remote.origin.mirror true && git push origin")
    )
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason
    assert "push-effecting configuration" in context.reason


@pytest.mark.parametrize(
    "command",
    [
        # core.hooksPath selects the directory git runs hooks from; the write
        # is a hard guard in every scope, including --file and relocation.
        "git config core.hooksPath /tmp/h",
        "git config --global core.hooksPath /tmp/h",
        "git config --worktree core.hooksPath /tmp/h",
        "git config --file cfg core.hooksPath /tmp/h",
        "git config set core.hooksPath /tmp/h",
        "git config --unset core.hooksPath",
        "git -C .git config core.hooksPath /tmp/h",
        "git --git-dir=.git config core.hooksPath /tmp/h",
        "env X=1 git config core.hooksPath /tmp/h",
        # core.editor and sequence.editor select the programs git runs for
        # commit/tag/rebase without -m and interactive rebase.
        "git config core.editor evil",
        "git config --global core.editor evil",
        "git config --worktree core.editor evil",
        "git config --file cfg core.editor evil",
        "git config set core.editor evil",
        "git config --unset core.editor",
        "git -C .git config core.editor evil",
        "git --git-dir=.git config core.editor evil",
        "git config sequence.editor evil",
        "git config --global sequence.editor evil",
        "git config --worktree sequence.editor evil",
        "git config --file cfg sequence.editor evil",
        "git config set sequence.editor evil",
        "git config --unset sequence.editor",
        "git -C .git config sequence.editor evil",
        "git --git-dir=.git config sequence.editor evil",
        "env X=1 git config sequence.editor evil",
        # filter.<name>.clean and filter.<name>.process select the programs
        # git runs for files with that filter (git add, git checkout).
        "git config filter.pwn.clean evil",
        "git config --global filter.pwn.clean evil",
        "git config --worktree filter.pwn.clean evil",
        "git config --file cfg filter.pwn.clean evil",
        "git config set filter.pwn.clean evil",
        "git config --add filter.pwn.clean evil",
        "git config --replace-all filter.pwn.clean evil",
        "git config --unset filter.pwn.clean",
        "git config unset filter.pwn.clean",
        "git -C .git config filter.pwn.clean evil",
        "git --git-dir=.git config filter.pwn.clean evil",
        "git config filter.pwn.process evil",
        "git config --file cfg filter.pwn.process evil",
        "git config set filter.pwn.process evil",
        "git config --unset-all filter.pwn.process",
        "git -C .git config filter.pwn.process evil",
        # merge.<name>.driver selects the program git runs for files with
        # that merge driver (git merge).
        "git config merge.ours.driver evil",
        "git config --global merge.ours.driver evil",
        "git config --worktree merge.ours.driver evil",
        "git config --file cfg merge.ours.driver evil",
        "git config set merge.ours.driver evil",
        "git config --unset merge.ours.driver",
        "git -C .git config merge.ours.driver evil",
        "git --git-dir=.git config merge.ours.driver evil",
        # diff.<name>.command/textconv/external select the programs git
        # runs for diffs with that driver (git diff, git log -p).
        "git config diff.external.command evil",
        "git config --global diff.external.command evil",
        "git config --worktree diff.external.command evil",
        "git config --file cfg diff.external.command evil",
        "git config set diff.external.command evil",
        "git config --unset diff.external.command",
        "git -C .git config diff.external.command evil",
        "git --git-dir=.git config diff.external.command evil",
        "git config diff.external.textconv evil",
        "git config --file cfg diff.external.textconv evil",
        "git config --unset-all diff.external.textconv",
        "git config diff.external.external evil",
        "git config set diff.external.external evil",
        "git config --replace-all diff.external.external evil",
        # Section and key names fold case-insensitively; the guard fails
        # closed on any spelling.
        "git config Filter.Pwn.Clean evil",
        "git config MERGE.Ours.Driver evil",
        "git config Diff.External.Command evil",
        "git config diff.EXTERNAL.Textconv evil",
        # Section renames reaching core/sequence (destination names included).
        "git config --rename-section other core",
        "git config --rename-section core.other core",
        "git config --rename-section other sequence",
        "git config rename-section other core",
        "git config --file cfg --rename-section other core",
        "git -C .git config --rename-section other core",
        "git --git-dir=.git config --rename-section other core",
        # The same bypass class for the driver sections: renaming into
        # (or out of) filter/merge/diff lands an agent-authored driver key.
        "git config --rename-section other filter",
        "git config --rename-section filter other",
        "git config --rename-section filter.lfs filter",
        "git config --rename-section other merge",
        "git config --rename-section merge ours2",
        "git config --rename-section other diff",
        "git config --rename-section diff ours",
        "git config --file cfg --rename-section other filter",
        "git -C .git config --rename-section other merge",
        "git --git-dir=.git config --rename-section other diff",
    ],
)
def test_program_selection_config_writes_are_denied(
    bash_tool: Bash, command: str
) -> None:
    # A hard guard: denied with no grantable proposal, so no answer can
    # authorize the write.
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason
    assert "program-selection" in context.reason


@pytest.mark.parametrize(
    "command",
    [
        # Reads of the program-selection keys never write.
        "git config --get core.hooksPath",
        "git config --get core.editor",
        "git config --get sequence.editor",
        "git config core.hooksPath",
        "git config core.editor",
        "git config sequence.editor",
        "git config --get filter.pwn.clean",
        "git config --get filter.pwn.process",
        "git config --get merge.ours.driver",
        "git config --get diff.external.command",
        "git config --get diff.external.textconv",
        "git config --get diff.external.external",
        "git config filter.pwn.clean",
        "git config merge.ours.driver",
        "git config diff.external.command",
        "git config --list",
        # Writes to unrelated keys and sections stay routine, including
        # section removals (a removal cannot move a key into a guarded
        # section) and unrelated renames. The driver sections keep their
        # non-program keys: required/smudge are boolean flags and data
        # spellings, not command strings (a documented residual).
        "git config user.name X",
        "git config core.pager less",
        "git config core.autocrlf true",
        "git config sequence.squashWarning false",
        "git config filter.pwn.required true",
        "git config filter.pwn.smudge cat",
        "git config merge.ours.name ours",
        "git config merge.conflictStyle diff3",
        "git config diff.external.foo bar",
        "git config diff.renameLimit 100",
        "git config --remove-section core.old",
        "git config --remove-section sequence.old",
        "git config --remove-section filter.old",
        "git config --remove-section merge.old",
        "git config --remove-section diff.old",
        "git config --rename-section other.one other.two",
    ],
)
def test_program_selection_config_reads_and_unrelated_writes_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS


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


@pytest.mark.parametrize(
    "command",
    [
        # Read forms do not put the destination in write_path_values: a
        # reading config --file stays an authority-scoped operand.
        "git config --file=.git/config --get user.name",
        "git config --file=.git/config user.name",
        "git config --file=.git/config --list",
    ],
)
def test_git_config_read_file_is_not_a_write_operand(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert policy.write_path_values == ()
    assert ".git/config" in policy.option_path_values


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


def test_git_work_tree_writing_subcommand_detection() -> None:
    assert git_work_tree_writing_subcommand(shlex.split("git merge evil")) == "merge"
    assert (
        git_work_tree_writing_subcommand(shlex.split("git -C .git apply p.diff"))
        == "apply"
    )
    assert (
        git_work_tree_writing_subcommand(
            shlex.split("git --git-dir=.git --work-tree=.git stash pop")
        )
        == "stash"
    )
    # Readers, non-writers, and unmodeled forms carry nothing to check.
    assert git_work_tree_writing_subcommand(shlex.split("git status")) is None
    assert git_work_tree_writing_subcommand(shlex.split("git -C .git status")) is None
    assert git_work_tree_writing_subcommand(shlex.split("git fetch origin")) is None
    assert git_work_tree_writing_subcommand(shlex.split("git log")) is None
    assert git_work_tree_writing_subcommand(shlex.split("git commit -m x")) is None
    assert git_work_tree_writing_subcommand(shlex.split("ls")) is None


# --- Git raw-write operand and work-tree governance (delta review, family 1) ---


def _git_dir_with_bare_markers(root: Path) -> Path:
    """A minimal git dir: discovery finds it as a repository git dir."""
    git_dir = root / ".git"
    git_dir.mkdir(exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text("[core]\n\tbare = false\n")
    (git_dir / "objects").mkdir(exist_ok=True)
    (git_dir / "refs").mkdir(exist_ok=True)
    (root / "seed.txt").write_text("seed\n")
    return root


@pytest.mark.parametrize(
    "command",
    [
        # git archive --output/-o writes the named file directly.
        "git archive --output=.git/config HEAD",
        "git archive --output .git/config HEAD",
        "git archive -o .git/config HEAD",
        "git archive -o.git/config HEAD",
        "git archive --output=.git/hooks/pre-commit HEAD",
        # A writing git config --file/-f names the written file.
        "git config --file=.git/config user.name x",
        "git config --file .git/config user.name x",
        "git config -f.git/config user.name x",
        "git config --file=.git/config set user.name x",
        "git config --file=.git/config --unset user.name",
        # The operand guard fires before the classifier's key guard for a
        # metadata destination, and the key guard still covers the rest.
        "git config --file=.git/config core.hooksPath /tmp/h",
        # Work-tree relocation into the metadata dir.
        "git -C .git apply ../hook.diff",
        "git -C .git merge evil",
        "git --git-dir=.git --work-tree=.git merge evil",
        "git --git-dir=.git --work-tree=.git stash pop",
        "git --git-dir=.git --work-tree=.git checkout main",
        "git --work-tree=.git merge evil",
        "git -C .git/hooks apply hook.diff",
    ],
)
def test_git_raw_write_and_work_tree_bypasses_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert "protected Git metadata" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        # In-workspace destinations and readers stay permitted.
        "git archive --output=bundle.tar HEAD",
        "git archive -obundle.tar HEAD",
        "git config --file=custom.cfg user.name X",
        "git config --file=custom.cfg --get user.name",
        "git config --file=.git/config --get user.name",
        "git -C .git status",
        "git --git-dir=.git log",
        "git --work-tree=. --git-dir=.git diff",
        # In-worktree tree-writing calls and legitimate relocations.
        "git apply fix.diff",
        "git merge evil",
        "git stash pop",
        "git switch main",
        "git fetch origin",
        "git --git-dir=.git --work-tree=wt merge evil",
        "git -C sub apply ../fix.diff",
    ],
)
def test_git_raw_write_and_work_tree_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


def test_include_then_push_compound_is_denied_without_a_proposal(
    bash_tool: Bash,
) -> None:
    # The F-1 exploit shape: the include write folds an authored file that
    # could set remote.<name>.pushurl for a verified remote, so it is
    # denied on its own segment and the whole compound call — the push
    # included — carries no proposal.
    context = bash_tool.resolve_permission(
        BashArgs(command="git config include.path /tmp/evil.cfg && git push origin")
    )
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason
    assert "push-effecting configuration" in context.reason


@pytest.mark.parametrize(
    "command",
    [
        "git config user.name X",
        "git config --global user.name X",
        "git config --get remote.origin.mirror",
        "git config --get remote.origin.url",
        "git config --get branch.main.remote",
        "git config --get url.https://x/.pushInsteadOf",
        "git config --get remote.origin.vcs",
        "git config --get core.sshCommand",
        "git config --get core.gitProxy",
        "git config --get core.askPass",
        "git config --get credential.helper",
        "git config --get credential.https://example.com.helper",
        "git config --get include.path",
        "git config --get includeIf.gitdir:/x.path",
        "git config --list",
        # Reads of the newly guarded keys never write: the bare-key form is
        # git's read form too.
        "git config core.sshCommand",
        "git config credential.helper",
        "git config include.path",
        "git config remote.origin.vcs",
        # push.default writes stay routine: they broaden what a bare push
        # publishes, never force, delete, or mirror.
        "git config push.default simple",
        "git config remote.origin.fetch +refs/heads/*:refs/remotes/origin/*",
    ],
)
def test_config_reads_and_unrelated_writes_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "git remote add origin ../x",
        "git remote set-url origin ../x",
        "git remote rename origin evil",
        "git remote remove origin",
        "git remote rm origin",
    ],
)
def test_git_remote_mutations_are_denied(bash_tool: Bash, command: str) -> None:
    # A git remote mutation redefines where a granted push lands, so it is
    # a hard guard: denied with no grantable proposal.
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason
    assert "redefines where a granted push lands" in context.reason


@pytest.mark.parametrize(
    "command",
    [
        "git remote",
        "git remote -v",
        "git remote get-url origin",
        "git remote show origin",
        "git remote prune origin",
    ],
)
def test_git_remote_reads_stay_permitted(bash_tool: Bash, command: str) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS


def test_a_push_grant_does_not_authorize_the_config_write(tmp_path: Path) -> None:
    # The guard's boundary: an approved ordinary push stays publication,
    # but it never extends to the config write that would make the push a
    # mirror push.
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    # The push destination is a bare name, so the repository must resolve
    # it: a configured remote keeps the ordinary push grantable.
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            "https://example.com/x.git",
        ],
        check=True,
    )
    scope = resolve_repository_scope(repo, GitRepositoryRelocation())
    assert scope is not None
    config = BashToolConfig(denylist=[])
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=repo)
    tool.dangerous_command_authorities_getter = lambda: (
        DangerousCommandAuthority(
            operations=frozenset({"git.push"}), scope=scope, source="runtime_grant"
        ),
    )
    context = tool.resolve_permission(BashArgs(command="git push origin"))
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS
    context = tool.resolve_permission(
        BashArgs(command="git config remote.origin.mirror true && git push origin")
    )
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None


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


def test_unknown_git_subcommand_may_be_an_alias(bash_tool: Bash) -> None:
    context = bash_tool.resolve_permission(BashArgs(command="git fixture-alias"))
    assert context is not None and context.permission is ToolPermission.NEVER
    assert context.reason and "aliases and external helpers" in context.reason


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
def bash_tool(tmp_path, monkeypatch) -> Bash:
    # Explicit entries only: the default denylists are extended elsewhere and
    # must not be pinned by these tests.
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    return Bash(config_getter=lambda: config, state=BaseToolState())


@pytest.fixture
def shell_resolver(bash_tool: Bash):
    from chartreux.core.tools.builtins._shell_permission_resolver import (
        ShellPermissionResolver,
    )

    return ShellPermissionResolver(
        bash_tool.config, bash_tool.cwd, bash_tool.workspace, bash_tool.path_authority
    )


@pytest.fixture
def git_denylist_tool(tmp_path, monkeypatch) -> Bash:
    # Explicit git entries only: the shipped defaults are extended elsewhere and
    # must not be pinned by these tests.
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig(
        denylist=[
            "git push",
            "git checkout",
            "git stash drop",
            "git stash clear",
            "git restore",
            "git switch --discard-changes",
            "git switch -f",
            "git reflog expire",
            "git reflog delete",
        ]
    )
    return Bash(config_getter=lambda: config, state=BaseToolState())


@pytest.fixture
def git_denylist_resolver(git_denylist_tool: Bash):
    from chartreux.core.tools.builtins._shell_permission_resolver import (
        ShellPermissionResolver,
    )

    return ShellPermissionResolver(
        git_denylist_tool.config,
        git_denylist_tool.cwd,
        git_denylist_tool.workspace,
        git_denylist_tool.path_authority,
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


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["git", "push"], ["git", "push"]),
        (["git", "-C", ".", "push"], ["git", "push"]),
        (["git", "-Csubdir", "push"], ["git", "push"]),
        (["git", "-C.", "push"], ["git", "push"]),
        (["git", "--no-pager", "push"], ["git", "push"]),
        (["git", "-C", ".", "--no-pager", "-C", "sub", "push"], ["git", "push"]),
        (["git", "--git-dir", ".", "push"], ["git", "push"]),
        (["git", "--git-dir=.", "push"], ["git", "push"]),
        # The full subcommand suffix is retained, never truncated.
        (["git", "-C", ".", "stash", "drop"], ["git", "stash", "drop"]),
        (["git", "-C", ".", "stash", "clear"], ["git", "stash", "clear"]),
        (["git", "-C", ".", "checkout", "file"], ["git", "checkout", "file"]),
        (["git", "-C", ".", "restore", "file"], ["git", "restore", "file"]),
        (["/usr/bin/git", "-C", ".", "push"], ["/usr/bin/git", "push"]),
        # The walker stops at the subcommand: later tokens are never stripped.
        (["git", "push", "--no-pager"], ["git", "push", "--no-pager"]),
        (["git", "status", "--", "push"], ["git", "status", "--", "push"]),
        (["git", "stash", "list"], ["git", "stash", "list"]),
        # A global option's operand is consumed even when it names a subcommand.
        (["git", "-C", "push", "status"], ["git", "status"]),
    ],
)
def test_canonical_git_command_elides_recognized_global_options(
    tokens: list[str], expected: list[str]
) -> None:
    assert canonical_git_command(tokens) == expected


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "--unknown", "push"],
        ["git", "-c", "alias.pwn=!touch pwn", "push"],
        ["git", "--config-env", "foo=bar", "push"],
        ["git", "--config-env=foo=bar", "push"],
        ["ls", "-la"],
        ["gitx", "-C", ".", "push"],
        [],
    ],
)
def test_canonical_git_command_fails_closed_on_unrecognized_globals(
    tokens: list[str],
) -> None:
    assert canonical_git_command(tokens) is None


@pytest.mark.parametrize(
    "command, pattern",
    [
        ("git -C . push", "git push"),
        ("git -Csubdir push", "git push"),
        ("git --no-pager push", "git push"),
        ("git -C . --no-pager -C sub push", "git push"),
        ("git --git-dir . push", "git push"),
        ("git --git-dir=. push", "git push"),
        ("git -C . checkout file", "git checkout"),
        ("git -C . restore file", "git restore"),
        ("git -C . stash drop", "git stash drop"),
        ("git -C . stash clear", "git stash clear"),
        ("git -C . switch --discard-changes", "git switch --discard-changes"),
        ("git -C . switch -f main", "git switch -f"),
        ("git -C . reflog expire --all", "git reflog expire"),
        ("git -C . reflog delete main", "git reflog delete"),
        ("/usr/bin/git -C . push", "git push"),
    ],
)
def test_denylist_matches_canonical_git_form(
    git_denylist_resolver, command: str, pattern: str
) -> None:
    assert git_denylist_resolver._find_denylist_match(command) == pattern


@pytest.mark.parametrize(
    "command",
    [
        # Unrecognized globals get no normalization; the policy layer denies them.
        "git --unknown push",
        "git -c alias.pwn=!touch pwn push",
        "git --config-env foo=bar push",
        # A global option's operand is consumed, even when it names a subcommand.
        "git -C push status",
        # Tokens after the subcommand are never stripped.
        "git status -- push",
        "git stash -- drop",
        # A shorter subcommand suffix never matches a longer pattern.
        "git stash list",
        "git stash",
        # Only the destructive switch/reflog forms match; plain forms do not.
        "git switch main",
        "git switch -c feature",
        "git stash pop",
        "git reflog",
        "git reflog show",
    ],
)
def test_denylist_canonical_git_form_rejects_non_matching_commands(
    git_denylist_resolver, command: str
) -> None:
    assert git_denylist_resolver._find_denylist_match(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "git --unknown push",
        "git -c alias.pwn=!touch pwn push",
        "git --config-env foo=bar push",
    ],
)
def test_unparsable_git_globals_stay_denied_by_policy_layer(
    git_denylist_resolver, command: str
) -> None:
    # No denylist normalization applies, and the policy layer hard-denies the
    # spelling, so the command never runs.
    assert git_denylist_resolver._find_denylist_match(command) is None
    assert analyze_shell_command_policy(shlex.split(command)).requires_approval


def test_guardrail_denies_command_builtin_wrapped_git_global_form(
    git_denylist_resolver,
) -> None:
    # The "command" builtin is unwrapped before the denylist lookup, so the
    # canonical git form applies to the wrapped command too.
    context = git_denylist_resolver._resolve_guardrail_permission([
        "command git -C . push"
    ])

    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "matches denylist pattern 'git push'" in context.reason


def test_bash_tool_denies_git_global_form_via_denylist(git_denylist_tool) -> None:
    result = git_denylist_tool.resolve_permission(BashArgs(command="git -C . push"))

    assert result is not None
    assert result.permission == ToolPermission.NEVER
    assert "matches denylist pattern 'git push'" in (result.reason or "")


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


def test_git_reset_hard_requires_approval() -> None:
    assert analyze_shell_command_policy(["git", "reset", "--hard"]).requires_approval


def test_git_reset_soft_needs_no_approval() -> None:
    assert not analyze_shell_command_policy([
        "git",
        "reset",
        "--soft",
    ]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "clean", "-f"],
        ["git", "clean", "-fd"],
        ["git", "clean", "-fdx"],
        ["git", "clean", "-f", "-e", "pattern"],
    ],
)
def test_git_clean_force_requires_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "clean", "-n"],
        ["git", "clean", "-n", "-f"],
        ["git", "clean", "-ndx"],
        ["git", "clean", "--dry-run", "-f"],
    ],
)
def test_git_clean_dry_run_needs_no_approval(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


def test_git_global_options_precede_subcommand() -> None:
    assert analyze_shell_command_policy([
        "git",
        "-C",
        "/repo",
        "reset",
        "--hard",
    ]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "diff", "--output", "/tmp/x"],
        ["git", "diff", "--output=/tmp/x"],
        ["git", "show", "--ext-diff"],
        ["git", "show", "--textconv"],
    ],
)
def test_git_diff_output_and_external_tools_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "status"],
        ["git", "log"],
        ["git", "--no-pager", "diff"],
        ["git", "diff", "--no-ext-diff"],
        ["git", "--help"],
        ["git", "--version"],
    ],
)
def test_git_read_only_commands_need_no_approval(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "-c", "core.editor=x", "status"],
        ["git", "--config-env", "x", "status"],
        ["git", "--unknown-global", "status"],
    ],
)
def test_git_config_override_and_unknown_global_fail_closed(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_git_dir_values_are_collected_as_paths() -> None:
    policy = analyze_shell_command_policy(["git", "--git-dir", "/x", "log"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("/x",)

    attached = analyze_shell_command_policy(["git", "--git-dir=/x", "log"])
    assert attached.option_path_values == ("/x",)


def test_git_diff_no_index_inspects_positional_paths() -> None:
    policy = analyze_shell_command_policy(["git", "diff", "--no-index", "a", "b"])
    assert not policy.requires_approval
    assert policy.inspect_positional_paths
    assert policy.positional_values == ("a", "b")


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
        ["git", "diff"],
        ["git", "log"],
        ["git", "show"],
        ["git", "status"],
        ["git", "blame"],
        ["git", "whatchanged"],
    ],
)
def test_git_readers_inspect_the_repository(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).inspect_git_repository


@pytest.mark.parametrize(
    "tokens", [["git", "commit", "-m", "x"], ["git", "init"], ["git"], ["cat", "file"]]
)
def test_non_readers_do_not_inspect_the_repository(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).inspect_git_repository


def _repository_with_config(root: Path, config: str) -> Path:
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(config)
    return root


def test_git_repository_config_risk_names_the_active_vector(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\tpager = ./pager\n")
    risk = git_repository_config_risk(["git", "log"], cwd=root)
    assert risk is not None
    assert risk.endswith("via core.pager")


def test_git_repository_config_risk_is_none_for_clean_repositories(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\trepositoryformatversion = 0\n")
    assert git_repository_config_risk(["git", "status"], cwd=root) is None


def test_git_repository_config_risk_ignores_inactive_values(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\tpager = false\n")
    assert git_repository_config_risk(["git", "log"], cwd=root) is None


def test_git_repository_config_risk_fails_closed_on_unreadable_config(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").mkdir()
    risk = git_repository_config_risk(["git", "log"], cwd=tmp_path)
    assert risk is not None
    assert "unreadable" in risk


def test_git_repository_config_risk_is_none_outside_any_repository(tmp_path) -> None:
    assert git_repository_config_risk(["git", "log"], cwd=tmp_path) is None


def test_git_repository_config_risk_reads_worktree_pointer(tmp_path) -> None:
    git_dir = tmp_path / "real-git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\tpager = ./pager\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {git_dir}")
    risk = git_repository_config_risk(["git", "log"], cwd=worktree)
    assert risk is not None
    assert risk.endswith("via core.pager")


def test_git_repository_config_risk_treats_empty_config_value_as_unset(
    tmp_path,
) -> None:
    # `core.pager =` (empty value) is FALSE to git, not the implicit true of
    # the valueless bare-key form.
    root = _repository_with_config(tmp_path, "[core]\n\tpager =\n")
    assert git_repository_config_risk(["git", "log"], cwd=root) is None


def test_git_repository_config_risk_treats_bare_config_key_as_true(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\tpager\n")
    risk = git_repository_config_risk(["git", "log"], cwd=root)
    assert risk is not None
    assert risk.endswith("via core.pager")


def test_git_repository_config_risk_fails_closed_on_unreadable_commondir(
    tmp_path,
) -> None:
    # A directory where git expects the commondir file makes the read fail;
    # falling back to the worktree gitdir's own (nonexistent) config would
    # hide the active core.pager in the shared config.
    worktree_git_dir = tmp_path / "real-git" / "worktrees" / "wt"
    worktree_git_dir.mkdir(parents=True)
    (tmp_path / "real-git" / "config").write_text("[core]\n\tpager = ./pager\n")
    (worktree_git_dir / "commondir").mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_git_dir}")
    risk = git_repository_config_risk(["git", "log"], cwd=worktree)
    assert risk is not None
    assert "unreadable" in risk


def test_git_repository_config_risk_reads_separate_git_dir_config(tmp_path) -> None:
    # A --separate-git-dir checkout has no commondir; its gitdir's own config
    # is the repository config, so the missing file must not fail closed.
    git_dir = tmp_path / "separate-git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\tpager = ./pager\n")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text(f"gitdir: {git_dir}")
    risk = git_repository_config_risk(["git", "log"], cwd=checkout)
    assert risk is not None
    assert risk.endswith("via core.pager")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            "[include]\n\tpath = ./included\n",
            "via include.path; audit or remove the [include] directive in .git/config",
        ),
        (
            '[includeIf "gitdir:."]\n\tpath = ./included\n',
            "via includeif.path; "
            "audit or remove the [includeIf] directive in .git/config",
        ),
        (
            "[core]\n\tfsmonitor = ./monitor\n",
            "via core.fsmonitor; "
            "audit or remove the core.fsmonitor setting in .git/config",
        ),
        (
            '[filter "lfs"]\n\tclean = ./clean\n',
            "via filter.clean; audit or remove the [filter] command in .git/config",
        ),
        (
            '[filter "lfs"]\n\tprocess = ./filter\n',
            "via filter.process; audit or remove the [filter] command in .git/config",
        ),
    ],
)
def test_git_repository_config_risk_names_remediation_for_broad_vectors(
    tmp_path, config: str, expected: str
) -> None:
    # The resolver is deny-only with no passthrough, so the broadest vectors
    # must tell the reader what to do about them, not just name the key.
    root = _repository_with_config(tmp_path, config)
    risk = git_repository_config_risk(["git", "status"], cwd=root)
    assert risk is not None
    assert risk.endswith(expected)


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
    "command", ["branch", "tag", "grep", "reflog", "stash", "shortlog"]
)
def test_paging_git_subcommands_inspect_repository(command: str) -> None:
    assert analyze_shell_command_policy(["git", command]).inspect_git_repository


def test_git_config_same_line_and_unsupported_syntax_fail_closed(
    tmp_path: Path,
) -> None:
    root = _repository_with_config(tmp_path, '[core] fsmonitor = "./monitor"\n')
    assert "core.fsmonitor" in (
        git_repository_config_risk(["git", "status"], cwd=root) or ""
    )
    (root / ".git" / "config").write_text("[core]\n  nonsense ! syntax\n")
    assert "unreadable" in (
        git_repository_config_risk(["git", "status"], cwd=root) or ""
    )


def test_git_config_invalid_utf8_fails_closed(tmp_path: Path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n")
    (root / ".git" / "config").write_bytes(b"[core]\nfsmonitor = \xff\n")
    assert "unreadable" in (
        git_repository_config_risk(["git", "status"], cwd=root) or ""
    )


def test_no_pager_disables_only_pager_config_vector(tmp_path: Path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n pager = ./pager\n")
    assert git_repository_config_risk(["git", "--no-pager", "log"], cwd=root) is None
    assert git_repository_config_risk(["git", "-P", "log"], cwd=root) is None
    assert git_repository_config_risk(["git", "--no-pager", "-p", "log"], cwd=root)


@pytest.mark.parametrize("globals_", ["", "--no-pager ", "-C . "])
def test_repository_alias_requires_approval(bash_tool: Bash, globals_: str) -> None:
    git_dir = bash_tool.cwd / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[alias]\n planted = !touch payload\n")
    command = f"git {globals_}planted"
    assert analyze_shell_command_policy(shlex.split(command)).requires_approval
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.NEVER
    assert not (bash_tool.cwd / "payload").exists()


@pytest.mark.parametrize("command", ["git status", "git diff", "git stash list"])
@pytest.mark.parametrize("custom", [False, True])
def test_git_index_hooks_are_denied(
    bash_tool: Bash, command: str, custom: bool
) -> None:
    git_dir = bash_tool.cwd / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        "[core]\n hooksPath = custom-hooks\n" if custom else "[core]\n bare = false\n"
    )
    hooks = bash_tool.cwd / "custom-hooks" if custom else git_dir / "hooks"
    hooks.mkdir()
    hook = hooks / "post-index-change"
    hook.write_text("#!/bin/sh\ntouch payload\n")
    hook.chmod(0o755)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.NEVER
    assert (
        "hooksPath" in (context.reason or "")
        if custom
        else "post-index-change" in (context.reason or "")
    )
    assert not (bash_tool.cwd / "payload").exists()


@pytest.mark.parametrize(
    "target", [".git/config", ".git/hooks/post-index-change", "../outside"]
)
@pytest.mark.parametrize("operator", [">", ">>", "2>", "&>"])
def test_git_reader_file_redirections_are_denied(
    bash_tool: Bash, target: str, operator: str
) -> None:
    context = bash_tool.resolve_permission(
        BashArgs(command=f"git log {operator} {target}")
    )
    assert context is not None and context.permission == ToolPermission.NEVER
    assert "require approval" not in (context.reason or "")


@pytest.mark.parametrize(
    "operand", ["/outside/file", "../outside", ".env", ".env.local"]
)
@pytest.mark.parametrize("attached", [False, True])
def test_git_blame_contents_is_workspace_scoped(
    bash_tool: Bash, operand: str, attached: bool
) -> None:
    option = f"--contents={operand}" if attached else f"--contents {operand}"
    command = f"git blame {option} tracked.txt"
    assert operand in path_candidates(
        shlex.split(command), inspect_positional_paths=False
    )
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "git log",
        "git status",
        "git diff",
        "git blame tracked.txt",
        "git blame --contents=tracked.txt tracked.txt",
        "git log 2>&1",
    ],
)
def test_clean_git_readers_remain_permitted(bash_tool: Bash, command: str) -> None:
    git_dir = bash_tool.cwd / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n bare = false\n")
    hooks = git_dir / "hooks"
    hooks.mkdir()
    # Git ignores non-executable hook files, including the shipped samples.
    (hooks / "post-index-change").write_text("#!/bin/sh\nexit 0\n")
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.ALWAYS


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


@pytest.mark.parametrize(
    "command,option",
    [
        ("pip install", "--target"),
        ("pip install", "--prefix"),
        ("pip install", "-t"),
        ("pip install", "-e"),
        ("uv pip install", "-t"),
        ("uv pip install", "-e"),
        ("npm install", "--cache-dir"),
        ("yarn install", "--cache-folder"),
        ("pnpm install", "--store-dir"),
        ("cargo build", "--target-dir"),
        ("uv pip install", "--config-file"),
        ("uv python install", "--install-dir"),
        ("pip install", "--log"),
        ("pip install", "--report"),
        ("pip install", "-r"),
        ("pip install", "-c"),
        ("pip wheel", "-w"),
        ("pip download", "-d"),
        ("uv pip compile", "-o"),
        ("npm install", "--cache"),
        ("npm install", "--userconfig"),
        ("npm install", "--globalconfig"),
        ("npm install", "--logs-dir"),
        ("pip download", "--dest"),
        ("pip download", "--destination-dir"),
        ("pip download", "--destination-directory"),
        ("pip install", "--source"),
        ("pip install", "--source-dir"),
        ("pip install", "--source-directory"),
        ("pip install", "--requirements-from-script"),
        ("npm pack", "--pack-destination"),
        ("pip wheel", "--wheel-dir"),
        ("uv build", "--out-dir"),
        ("uv pip compile", "--output-file"),
        ("poetry build", "--output"),
        ("pip install", "--requirement"),
        ("pip install", "--constraint"),
        ("uv pip install", "--build-constraints"),
        ("uv pip install", "--overrides"),
        ("uv pip install", "--excludes"),
        ("uv run", "--env-file"),
    ],
)
@pytest.mark.parametrize("destination", ["local", "../outside", ".env"])
@pytest.mark.parametrize("attached", [False, True])
def test_package_destinations_are_consistently_scoped(
    bash_tool: Bash, command: str, option: str, destination: str, attached: bool
) -> None:
    separator = "=" if option.startswith("--") else ""
    argument = option + (separator if attached else " ") + destination
    context = bash_tool.resolve_permission(
        BashArgs(command=f"{command} {argument} pkg")
    )
    assert context is not None
    assert context.permission == (
        ToolPermission.ALWAYS if destination == "local" else ToolPermission.NEVER
    )


# --- git apply --directory write roots and the --unsafe-paths hard guard
# --- (delta re-review, round 2) ---


@pytest.mark.parametrize(
    "command",
    [
        # --unsafe-paths disables the path refusal git itself applies
        # outside the work tree, including paths inside .git metadata:
        # a hard guard, whatever else the call carries.
        "git apply --unsafe-paths hook.diff",
        "git apply --unsafe-paths --directory=.git/hooks hook.diff",
        "git apply --directory=.git/hooks --unsafe-paths hook.diff",
        "git apply --unsafe-paths --directory=fixes hook.diff",
        "git apply --unsafe-paths --directory=/abs/path hook.diff",
    ],
)
def test_git_apply_unsafe_paths_is_a_hard_guard(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert policy.requires_approval
    assert policy.denial_reason is not None
    assert "--unsafe-paths" in policy.denial_reason


@pytest.mark.parametrize(
    "command",
    [
        # Benign apply forms carry no hard guard: an in-workspace
        # --directory is an ordinary write operand, and git am's
        # --directory names the mailbox it reads.
        "git apply hook.diff",
        "git apply --directory=fixes hook.diff",
        "git apply --exclude=*.txt --include=*.py hook.diff",
        "git am hook.diff",
        "git am --directory=.git/hooks hook.diff",
    ],
)
def test_git_apply_benign_forms_carry_no_hard_guard(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert not policy.requires_approval


@pytest.mark.parametrize(
    "command, expected",
    [
        # git apply --directory relocates the patch's write root: the value
        # is a write operand like any other, in every spelling.
        ("git apply --directory=.git/hooks hook.diff", (".git/hooks",)),
        ("git apply --directory .git/hooks hook.diff", (".git/hooks",)),
        ("git apply --directory=.git/hooks/ hook.diff", (".git/hooks/",)),
        ("git apply -p0 --directory=.git/hooks hook.diff", (".git/hooks",)),
        ("git apply --directory=/abs/path hook.diff", ("/abs/path",)),
        ("git apply hook.diff", ()),
        # git init creates its positional directory and the git dir its
        # --separate-git-dir names; --template is a read, and --shared
        # takes an optional attached value only, so it never consumes the
        # positional.
        ("git init newdir", ("newdir",)),
        ("git init --shared newdir", ("newdir",)),
        ("git init --template=/usr/share/git-core/templates newdir", ("newdir",)),
        ("git init --separate-git-dir=.git/evil newdir", ("newdir", ".git/evil")),
        ("git init", ()),
        # git worktree add's first positional is the new work tree;
        # --reason/-b take values, --orphan is bare.
        ("git worktree add wt2", ("wt2",)),
        ("git worktree add --lock --reason why wt2", ("wt2",)),
        ("git worktree add -b topic wt2", ("wt2",)),
        ("git worktree add --orphan wt2", ("wt2",)),
        ("git worktree list", ()),
        ("git worktree add", ()),
    ],
)
def test_git_apply_init_and_worktree_destinations_are_write_operands(
    command: str, expected: tuple[str, ...]
) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert policy.write_path_values == expected, command


def test_git_cwd_relocations_models_the_c_chain(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    # No -C (or an empty -C value, a no-op relocation git itself accepts):
    # the tracked cwds resolve unchanged.
    assert git_cwd_relocations(shlex.split("git archive -o b.tar HEAD"), {root}) is None
    assert (
        git_cwd_relocations(shlex.split("git -C '' archive -o b.tar HEAD"), {root})
        is None
    )
    assert (
        git_cwd_relocations(
            shlex.split("git --git-dir=.git archive -o b.tar HEAD"), {root}
        )
        is None
    )
    # -C relocates each tracked cwd through the chain, in turn.
    assert git_cwd_relocations(
        shlex.split("git -C .git archive -o config HEAD"), {root}
    ) == {root / ".git"}
    assert git_cwd_relocations(
        shlex.split("git -C sub -C .. archive -o b.tar HEAD"), {root}
    ) == {root}
    assert git_cwd_relocations(
        shlex.split("git -C /abs archive -o b.tar HEAD"), {root}
    ) == {Path("/abs")}


@pytest.mark.parametrize(
    "command",
    [
        # git's own -C relocation moves the write root into the metadata.
        "git -C .git archive -o config HEAD",
        "git -C .git archive -o hooks/pre-commit HEAD",
        "git -C .git/hooks archive -o pre-commit HEAD",
        "git -C .git archive -o index HEAD",
        "git -C .git config --file=config user.name x",
        "git -C .git config --file=hooks/pre-commit user.name x",
        "git -C .git/hooks config --file=pre-commit user.name x",
        "git -C subdir archive -o ../.git/config HEAD",
        "git -C subdir config --file=../.git/config user.name x",
    ],
)
def test_git_c_relocated_write_operands_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "subdir").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert "protected Git metadata" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        # Legitimate relocations: the write lands outside the metadata.
        "git -C subdir archive -o bundle.tar HEAD",
        "git -C subdir archive -o ../bundle.tar HEAD",
        "git -C subdir config --file=custom.cfg user.name x",
        "git -C subdir config --file=../custom.cfg user.name x",
        "git -C . -C . archive -o bundle.tar HEAD",
        "git -C .git status",
        "git -C .git log",
    ],
)
def test_git_c_relocated_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "subdir").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


@pytest.mark.parametrize(
    "command",
    [
        # --directory relocates the patch's write root into the metadata.
        "git apply --directory=.git/hooks hook.diff",
        "git apply --directory .git/hooks hook.diff",
        "git apply --directory=./.git/hooks hook.diff",
        "git apply --directory=.git/../.git/hooks hook.diff",
        "git apply --directory=.git/hooks/ hook.diff",
        "git apply -p0 --directory=.git/hooks hook.diff",
        "git apply --directory=.git hook.diff",
        "git -C subdir apply --directory=../.git/hooks hook.diff",
        # ...and outside the workspace.
        "git apply --directory=/tmp/outside hook.diff",
        "git apply --directory=../outside hook.diff",
    ],
)
def test_git_apply_directory_write_roots_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "subdir").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    reason = context.reason or ""
    if "outside" in command:
        assert "outside the authorized workspace" in reason, command
    else:
        assert "protected Git metadata" in reason, command


@pytest.mark.parametrize(
    "command",
    [
        "git apply --unsafe-paths hook.diff",
        "git apply --unsafe-paths --directory=.git/hooks hook.diff",
        "git apply --unsafe-paths --directory=fixes hook.diff",
    ],
)
def test_git_apply_unsafe_paths_is_denied_without_a_proposal(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert "--unsafe-paths" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        "git apply hook.diff",
        "git apply --directory=fixes hook.diff",
        "git apply --exclude=*.txt hook.diff",
        "git am hook.diff",
        "git am --directory=.git/hooks hook.diff",
    ],
)
def test_git_apply_benign_forms_stay_permitted(bash_tool: Bash, command: str) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


# --- git init / git worktree add destinations (delta re-review, round 2) ---


@pytest.mark.parametrize(
    "command",
    [
        # Destinations resolving into the metadata.
        "git init .git/evil",
        "git init .git/hooks/evil",
        "git init --separate-git-dir=.git/evil newdir",
        "git init --separate-git-dir .git/evil newdir",
        "git worktree add .git/wt2",
        "git worktree add .git/hooks/wt",
        "git worktree add -b wtb .git/wt2",
        "git worktree add --lock --reason why .git/wt2",
        "git -C subdir init ../.git/evil",
        "git -C subdir worktree add ../.git/wt2",
        # ...and outside the workspace.
        "git init /tmp/outside-init",
        "git worktree add ../outside-wt",
    ],
)
def test_git_init_and_worktree_add_metadata_destinations_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "subdir").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    reason = context.reason or ""
    if "outside" in command:
        assert "outside the authorized workspace" in reason, command
    else:
        assert "protected Git metadata" in reason, command


@pytest.mark.parametrize(
    "command",
    [
        # No positional: git init creates .git in the cwd, which stays
        # permitted; in-workspace destinations stay permitted.
        "git init",
        "git init newdir",
        "git init --shared newdir",
        "git init --shared=group newdir",
        "git init -b trunk newdir",
        "git init --template=/usr/share/git-core/templates newdir",
        "git worktree add wt2",
        "git worktree add -b wtb wt2",
        "git worktree add --orphan wt2",
        "git worktree list",
        # git clone keeps its existing treatment (not a metadata destination).
        "git clone -q . /tmp/x-clone",
    ],
)
def test_git_init_and_worktree_add_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


# --- Newly inspected write/mutate commands (delta re-review, round 2) ---


@pytest.mark.parametrize(
    "command",
    [
        # cpio: the pass-through destination positional, the -D directory
        # copy-in changes into, and the copy-out archive files.
        "cpio -p .git/hooks",
        "cpio --pass-through .git/hooks",
        "cpio -i -D .git",
        "cpio -o -F .git/config",
        "cpio -o -O .git/config",
        # shred and fallocate mutate their positional file in place.
        "shred .git/config",
        "shred -u .git/config",
        "fallocate -l 100 .git/config",
        # ar: the archive operand is created or replaced, and --output
        # names the extraction destination.
        "ar rcs .git/evil.a x.o",
        "ar x --output=.git lib.a",
        # pax/bsdtar: tar-family archive operands and extraction
        # destinations.
        "pax -w -f .git/bundle.tar seed.txt",
        "pax -r -f .git/config",
        "bsdtar -cf .git/bundle.tar seed.txt",
        "bsdtar -C .git -xf evil.tgz",
    ],
)
def test_unmodeled_file_command_metadata_writes_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert "protected Git metadata" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        "cpio -p outdir",
        "cpio -i -D outdir",
        "cpio -o -F out.cpio",
        "cpio -t",
        "shred seed.txt",
        "fallocate -l 100 seed.txt",
        "ar rcs lib.a x.o",
        "ar t lib.a",
        "pax -w -f out.tar seed.txt",
        "bsdtar -cf out.tar seed.txt",
        "bsdtar -C outdir -xf good.tgz",
    ],
)
def test_unmodeled_file_command_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


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


@pytest.mark.parametrize(
    "command",
    [
        # Binutils/assembler output operands into metadata.
        "strip -o .git/hooks/pre-push evil",
        "strip -o .git/config evil",
        "strip -o.git/config evil",
        "strip .git/config",
        "strip -R .text .git/config",
        "objcopy evil .git/hooks/pre-push",
        "objcopy --strip-all evil .git/config",
        "objcopy .git/config",
        "as -o .git/config evil.s",
        "as -o.git/config evil.s",
        # Config-content writers into metadata (gpg/xmllint output, scp
        # local-to-local in both directions).
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt -o .git/config payload.gpg",
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt --output .git/config payload.gpg",
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt --output=.git/config payload.gpg",
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt -o.git/config payload.gpg",
        "xmllint --output .git/config doc.xml",
        "xmllint -o .git/config doc.xml",
        "xmllint --output=.git/config doc.xml",
        "scp seed.txt .git/config",
        "scp .git/config out.txt",
        "scp -r .git host:x",
        # In-place compression mutators.
        "bzip2 .git/config",
        "bunzip2 .git/config.bz2",
        "bzcat .git/config.bz2",
        "xz .git/config",
        "unxz .git/config.xz",
        "xzcat .git/config.xz",
        "zstd .git/config",
        "zstd -q -o .git/config seed.txt",
        "zstd --output-dir-flat=.git seed.txt",
        "zstd --output-dir-mirror .git seed.txt",
        "zstdmt .git/config",
        "unzstd .git/config.zst",
        "lzma .git/config",
        "unlzma .git/config.lzma",
        "lzcat .git/config.lzma",
        "lzmainfo .git/config.lzma",
        # Archive creation/extraction destinations.
        "zip .git/evil.zip seed.txt",
        "zip -r .git/evil2.zip sub",
        "7z a .git/evil.7z seed.txt",
        "7zr a .git/evilr.7z seed.txt",
        "7za a .git/evil.7z seed.txt",
        "7z x -o.git evil.7z",
        "7z x -o .git evil.7z",
        "unar -f -o .git evil.zip",
        "unar -f -o=.git evil.zip",
        "unar -f -output-directory .git evil.zip",
        "unar -f -output-directory=.git evil.zip",
        # mktemp creation operands.
        "mktemp .git/evil.XXXXXX",
        "mktemp -p .git evil.XXXXXX",
        "mktemp --tmpdir=.git evil.XXXXXX",
        # git worktree move / submodule add destinations.
        "git worktree move wt .git/wt2",
        "git worktree move -f wt .git/wt2",
        "git -C sub worktree move ../wt ../.git/wt2",
        "git submodule add ../src .git/evilmod",
        "git submodule add -b main ../src .git/evilmod",
        "git submodule add --separate-git-dir=.git/evilmod ../src dest",
        "git -C sub submodule add ../../src ../.git/evilmod",
        # ...and outside the workspace (the write-operand authority denies
        # them, exactly like the init/worktree-add flips).
        "git worktree move wt ../outside-wt",
        "git submodule add ../src ../outside-mod",
    ],
)
def test_round3_write_operand_metadata_writes_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "sub").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    reason = context.reason or ""
    if "outside" in command:
        assert "outside the authorized workspace" in reason, command
    else:
        assert "protected Git metadata" in reason, command


@pytest.mark.parametrize(
    "command",
    [
        # Binutils/assembler benign forms.
        "strip -o stripped evil",
        "strip evil",
        "strip -R .text -o stripped2 evil",
        # --output abbreviates --output-target in real strip, so the form
        # consumes the operand as a bfdname and errors: no write happens.
        "strip --output .git/config evil",
        "objcopy evil copied",
        "objcopy --strip-all evil copied2",
        "as -o evil.o evil.s",
        # Config-content writers to workspace/scratchpad targets.
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt -o out.txt payload.gpg",
        "xmllint --output out.xml doc.xml",
        "xmllint -o out2.xml doc.xml",
        "scp seed.txt copied.txt",
        "scp -P 22 seed.txt copied2.txt",
        "scp -i fake-id seed.txt copied3.txt",
        # scp remote forms stay outside the local-metadata scope.
        "scp seed.txt host:x",
        "scp host:x local.txt",
        # In-place compression mutators on workspace files.
        "bzip2 seed.txt",
        "bunzip2 seed.txt.bz2",
        "bzcat seed.txt.bz2",
        "xz seed.txt",
        "unxz seed.txt.xz",
        "xzcat seed.txt.xz",
        "zstd seed.txt",
        "zstd -q -o out.zst seed.txt",
        "zstd --output-dir-flat=odir seed.txt",
        "zstd --output-dir-mirror odm seed.txt",
        "zstdmt seed.txt",
        "unzstd seed.txt.zst",
        "lzma seed.txt",
        "unlzma seed.txt.lzma",
        "lzcat seed.txt.lzma",
        "lzmainfo seed.txt.lzma",
        # Archive destinations in the workspace.
        "zip out.zip seed.txt",
        "zip -r out2.zip sub",
        "zip -q -b . out3.zip seed.txt",
        "zip -P secret out4.zip seed.txt",
        "7z a out.7z seed.txt",
        "7zr a outr.7z seed.txt",
        "7za a out5.7z seed.txt",
        "7z x -od7z evil.7z",
        "7z l out.7z",
        # -o= names a directory literally called "=.git" in real 7z (the
        # value is everything after -o), never the metadata directory.
        "7z x -o=.git evil.7z",
        "unar -f -o unarout evil.zip",
        "unar -f -o=unarout2 evil.zip",
        # mktemp creation in the workspace (no operand: the default temp
        # directory stays permitted).
        "mktemp",
        "mktemp tmpl.XXXXXX",
        "mktemp -p . tmpl2.XXXXXX",
        "mktemp --tmpdir=. tmpl3.XXXXXX",
        # git worktree move / submodule add to workspace destinations.
        "git worktree move wt wt3",
        "git worktree move -f wt wt4",
        "git submodule add ../src submod",
        "git submodule add -b main ../src submod2",
        "git submodule list",
        "git worktree list",
    ],
)
def test_round3_write_operand_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "sub").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


@pytest.mark.parametrize(
    "command,expected",
    [
        ("git worktree move wt dest", ("dest",)),
        ("git worktree move -f wt dest", ("dest",)),
        ("git worktree move --force wt dest", ("dest",)),
        ("git worktree add dest", ("dest",)),
        ("git submodule add ../src dest", ("dest",)),
        ("git submodule add -b main --name nm ../src dest", ("dest",)),
        ("git submodule add --separate-git-dir=gd ../src dest", ("dest", "gd")),
        ("git submodule add ../src", ()),
        ("git worktree move wt", ()),
        ("git worktree prune", ()),
        ("git submodule status", ()),
    ],
)
def test_git_worktree_move_and_submodule_add_destinations_are_write_operands(
    command: str, expected: tuple[str, ...]
) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert not policy.requires_approval, command
    assert policy.write_path_values == expected, command


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
        # Bash never pathname-expands a redirect destination, so a literal
        # glob-looking target is a literal filename, not a glob.
        "echo x > f[0]",
        'echo x > f"*"',
        'echo x > "f"[0]',
        'echo x > "sub"/f[0]',
        "echo x > out[1].log",
    ],
)
def test_glob_looking_redirect_targets_are_literal_filenames(
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
