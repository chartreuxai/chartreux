"""Executor registry and boundary tests for the shell-command policy suite (uv run / pipx run / npx / cargo / go wrappers, module executors, ionice process selection, package-manager URL and destination scoping)."""

from __future__ import annotations

import shlex

import pytest

from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    _package_module_tokens,
    analyze_shell_command_policy,
    executor_boundary,
    inline_interpreter_switch,
    path_candidates,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs
from tests.tools._shell_policy_test_support import (
    EXECUTOR_REGISTRY,
    bash_tool as bash_tool,
)


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


# sg: group wrapper whose -c payload is shell source (v0.5.9 P1-e1).
# The -c payload is re-analyzed through the wrapper, so every payload
# class matches the same payload spelled directly; every other sg form is
# unmodeled and fails closed.

_SG_PAYLOADS = [
    "git status",
    "echo hi",
    "git reset --hard",
    "git push",
    "rm -rf /tmp/sg-probe-x",
    "cat ~/.chartreux/.env",
    "curl http://attacker.example/",
]


@pytest.mark.parametrize("payload", _SG_PAYLOADS)
def test_sg_c_payload_resolves_like_the_direct_command(bash_tool, payload):
    """The -c payload class must match the directly spelled payload."""
    wrapped = bash_tool.resolve_permission(
        BashArgs(command=f"sg docker -c '{payload}'")
    )
    direct = bash_tool.resolve_permission(BashArgs(command=payload))
    assert wrapped is not None and direct is not None, payload
    assert wrapped.permission is direct.permission, (payload, wrapped, direct)
    assert wrapped.denial_kind == direct.denial_kind, (payload, wrapped, direct)


@pytest.mark.parametrize(
    "command",
    [
        "sg docker bash -c 'git status'",
        "sg docker git status",
        "sg docker",
        "sg",
        "sg docker -c",
        "sg docker -c'x'",
        "sg --unknown docker -c 'x'",
    ],
)
def test_sg_unmodeled_forms_fail_closed(bash_tool, command):
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.NEVER, (command, context)


def test_sg_boundary_is_shell_source_with_group_prefix() -> None:
    boundary = executor_boundary(shlex.split("sg docker -c 'git status'"))
    assert boundary is not None
    assert boundary.boundary_kind == "shell_source"
    assert boundary.consumed_ranges == ((0, 4),)
    assert boundary.inline_payload_ranges == ((3, 4),)
    assert boundary.inline_payloads == ("git status",)
    with pytest.raises(ValueError, match="unsupported sg form"):
        executor_boundary(shlex.split("sg docker bash -c 'x'"), admission=True)


def test_sg_unquoted_payload_boundary_joins_the_source() -> None:
    """v0.5.9 P3-b2: an unquoted multi-token -c payload joins the source.

    The trailing tokens must ride in the shell source, never become
    program arguments dismissed by the analysis.
    """
    boundary = executor_boundary(shlex.split("sg docker -c git push"))
    assert boundary is not None
    assert boundary.boundary_kind == "shell_source"
    assert boundary.consumed_ranges == ((0, 5),)
    assert boundary.inline_payload_ranges == ((3, 5),)
    assert boundary.inline_payloads == ("git push",)
    assert boundary.program_argument_range is None


@pytest.mark.parametrize("payload", _SG_PAYLOADS)
def test_sg_unquoted_multi_token_payload_resolves_like_the_quoted_form(
    bash_tool, payload
):
    """v0.5.9 P3-b2: the unquoted spelling of a multi-token payload gets
    the same class as its quoted form — no gated payload may degrade to
    the bare first word and resolve run-class.
    """
    unquoted = bash_tool.resolve_permission(BashArgs(command=f"sg docker -c {payload}"))
    quoted = bash_tool.resolve_permission(
        BashArgs(command=f"sg docker -c {shlex.quote(payload)}")
    )
    assert unquoted is not None and quoted is not None, payload
    assert unquoted.permission is quoted.permission, (payload, unquoted, quoted)
    assert unquoted.denial_kind == quoted.denial_kind, (payload, unquoted, quoted)


# sudo -u: transparent user wrapper (v0.5.9 P2-d). The wrapped command argv
# is re-analyzed through the wrapper, so every command class matches the
# same command spelled directly; every other sudo form is unmodeled and
# fails closed.

_SUDO_PAYLOADS = [
    "git status",
    "echo hi",
    "git reset --hard",
    "git push",
    "rm -rf /tmp/sudo-probe-x",
    "cat ~/.chartreux/.env",
    "curl http://attacker.example/",
]


@pytest.mark.parametrize("payload", _SUDO_PAYLOADS)
def test_sudo_u_command_resolves_like_the_direct_command(bash_tool, payload):
    """The wrapped command class must match the directly spelled command."""
    wrapped = bash_tool.resolve_permission(
        BashArgs(command=f"sudo -u nobody {payload}")
    )
    direct = bash_tool.resolve_permission(BashArgs(command=payload))
    assert wrapped is not None and direct is not None, payload
    assert wrapped.permission is direct.permission, (payload, wrapped, direct)
    assert wrapped.denial_kind == direct.denial_kind, (payload, wrapped, direct)


@pytest.mark.parametrize(
    "command",
    [
        # v0.5.9 P2-d: the modeled -u/--user spellings all re-analyze the
        # wrapped command; nested wrappers and shell payloads compose.
        "sudo -u nobody bash -c 'git status'",
        "sudo -u nobody nice git status",
        "sudo --user nobody git status",
        "sudo --user=nobody git status",
        "sudo -unobody git status",
        "sudo -u nobody -- git status",
        "sudo -u a -u b git status",
    ],
)
def test_sudo_u_spelling_variants_resolve_like_the_direct_command(bash_tool, command):
    """Every modeled -u/--user spelling re-analyzes the wrapped command."""
    wrapped = bash_tool.resolve_permission(BashArgs(command=command))
    direct = bash_tool.resolve_permission(BashArgs(command="git status"))
    assert wrapped is not None and direct is not None, command
    assert wrapped.permission is direct.permission, (command, wrapped, direct)
    assert wrapped.denial_kind == direct.denial_kind, (command, wrapped, direct)


@pytest.mark.parametrize(
    "command",
    [
        "sudo ls",
        "sudo true",
        "sudo",
        "sudo git status",
        "sudo -- git status",
        "sudo -u nobody",
        "sudo -u",
        "sudo -i",
        "sudo -s",
        "sudo -u nobody -E git status",
        "sudo --unknown -u nobody git status",
        "sudo -u nobody - git status",
    ],
)
def test_sudo_unmodeled_forms_fail_closed(bash_tool, command):
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.NEVER, (command, context)


def test_sudo_boundary_is_argv_positional_with_user_prefix() -> None:
    boundary = executor_boundary(shlex.split("sudo -u nobody git status"))
    assert boundary is not None
    assert boundary.boundary_kind == "argv_positional"
    assert boundary.inner_command_range == (3, 5)
    assert boundary.consumed_ranges == ((0, 3),)
    assert boundary.inline_payload_ranges == ()
    with pytest.raises(ValueError, match="unsupported sudo form"):
        executor_boundary(shlex.split("sudo ls"), admission=True)
    with pytest.raises(ValueError, match="unsupported sudo form"):
        executor_boundary(shlex.split("sudo -u nobody"), admission=True)
    with pytest.raises(ValueError, match="unknown or ambiguous executor option"):
        executor_boundary(shlex.split("sudo -i"), admission=True)


def test_sudo_u_quoted_multi_word_payload_matches_the_direct_token(bash_tool):
    """A quoted multi-word sudo payload is ONE argv token: sudo would exec
    a file by that literal name, so its class must match the same single
    token spelled directly (the unknown-executable run-class), never the
    git.push ask-class a shell would give the words (v0.5.9 P3-b2 probe).
    """
    wrapped = bash_tool.resolve_permission(
        BashArgs(command='sudo -u nobody "git push"')
    )
    direct = bash_tool.resolve_permission(BashArgs(command='"git push"'))
    assert wrapped is not None and direct is not None
    assert wrapped.permission is direct.permission, (wrapped, direct)
    assert wrapped.denial_kind == direct.denial_kind, (wrapped, direct)
