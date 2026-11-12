"""Executor registry and boundary tests for the shell-command policy suite (uv run / pipx run / npx / cargo / go wrappers, module executors, ionice process selection, package-manager URL and destination scoping)."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import time

import pytest

from chartreux.core.background_jobs import BackgroundJobRegistry
from chartreux.core.completion_checks import CompletionCheckRunner
from chartreux.core.completion_contract import CompletionCheckTemplate
from chartreux.core.system_prompt import _get_os_system_prompt
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    _package_module_tokens,
    analyze_shell_command_policy,
    executor_boundary,
    inline_interpreter_switch,
    path_candidates,
)
from chartreux.core.tools.builtins._shell_permission_analysis import (
    analyze_shell_command,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    _analyze_guardrail_source,
    _has_unquoted_extended_syntax,
    shell_analysis_scope,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs
from chartreux.core.utils import shell as shell_module
from chartreux.core.utils.shell import (
    EXECUTOR_ENV_VAR,
    UnknownShellExecutorError,
    _shell_environment,
    apply_executor_selection,
    current_executor_contract,
    executor_option_contract,
    publish_shell_executor,
    reset_executor_selection,
    resolve_shell_executor,
    shell_executor,
    spawn_shell_command,
)
from tests.tools._shell_policy_test_support import (
    EXECUTOR_REGISTRY,
    bash_tool as bash_tool,
)


@pytest.fixture(autouse=True)
def _isolated_executor_selection() -> object:
    """Keep one test's process-level executor out of the next."""
    reset_executor_selection()
    yield
    reset_executor_selection()


@pytest.fixture
def ambient_shell(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Resolve from $SHELL alone: no shipped bash default in front of it."""
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: None)
    reset_executor_selection()
    return monkeypatch


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


# --- v0.7.1 F1pkg1(b): the immutable executor option contract -----------------
#
# The contract pins, in the EXECUTING process, the executor identity, every
# expansion-relevant option family OFF, the startup files off via the child
# environment, and the locale. Unknown executors, option-changing shell
# invocations and builtins, and nested shell re-entry with an expansion-
# bearing payload fail closed. The recognition pins above stay: the grammar
# still recognizes `bash +O extglob -c code` as a shell_source boundary; the
# DENIAL is layered at resolution (Amendment UU: re-decided, not presupposed
# flips — the payload-boundary recognition and the recursive-rm hard denial
# REMAIN, the new contract gets separate tests).


def test_executor_option_contract_pins_every_known_family() -> None:
    bash_contract = executor_option_contract("/usr/bin/bash")
    assert bash_contract is not None and bash_contract.family == "bash"
    assert {"extglob", "globstar", "nullglob", "dotglob", "failglob", "nocaseglob"} <= (
        bash_contract.pinned_off_options
    )
    zsh_contract = executor_option_contract("/usr/bin/zsh")
    assert zsh_contract is not None and zsh_contract.family == "zsh"
    assert {"extendedglob", "globdots", "kshglob", "nullglob", "caseglob"} <= (
        zsh_contract.pinned_off_options
    )
    assert ("ZDOTDIR", "/dev/null/chartreux-pinned-no-zdotdir") in (
        zsh_contract.startup_env_pins
    )
    posix_contract = executor_option_contract("/bin/sh")
    assert posix_contract is not None and posix_contract.family == "posix"
    # No $SHELL: the kernel default /bin/sh applies at spawn.
    assert executor_option_contract(None) is not None
    # An unknown executor has NO contract: fail closed.
    assert executor_option_contract("/usr/bin/fish") is None
    assert executor_option_contract("/usr/local/bin/mybash") is None


def test_executor_chain_is_config_then_bash_then_shell_then_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1b: the shipped bash default OUTRANKS the ambient ``$SHELL``.

    The documented chain is config (``[tools.bash] executor``) > bash > the
    ambient ``$SHELL`` > the kernel default, in that order.
    """
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    # The shipped default wins over an ambient zsh host.
    assert resolve_shell_executor() == "/usr/bin/bash"
    # An explicit config value wins over both.
    assert resolve_shell_executor("/usr/bin/zsh") == "/usr/bin/zsh"
    # No bash on the host: the ambient $SHELL is the documented fallback.
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: None)
    assert resolve_shell_executor() == "/usr/bin/zsh"
    # Neither names a shell: the kernel default applies, unnamed.
    monkeypatch.delenv("SHELL", raising=False)
    assert resolve_shell_executor() is None


def test_shell_executor_prefers_the_process_override_over_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    assert apply_executor_selection("/usr/bin/zsh") == "/usr/bin/zsh"
    assert shell_executor() == "/usr/bin/zsh"
    assert current_executor_contract() is not None
    assert current_executor_contract().family == "zsh"  # type: ignore[union-attr]
    # The resolution travels to children through the publish helper, never as
    # a process-global environment side effect.
    assert publish_shell_executor({})[EXECUTOR_ENV_VAR] == "/usr/bin/zsh"
    assert EXECUTOR_ENV_VAR not in os.environ
    reset_executor_selection()
    assert shell_executor() == "/usr/bin/bash"


def test_a_child_process_resolves_the_published_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The root agent reads no config: the parent's value is authoritative."""
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    monkeypatch.setenv(EXECUTOR_ENV_VAR, "/usr/bin/zsh")
    assert shell_executor() == "/usr/bin/zsh"
    assert current_executor_contract() is not None
    assert current_executor_contract().family == "zsh"  # type: ignore[union-attr]


def test_publish_shell_executor_fills_only_a_missing_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    assert publish_shell_executor({"A": "1"})[EXECUTOR_ENV_VAR] == "/usr/bin/bash"
    explicit = publish_shell_executor({"A": "1", EXECUTOR_ENV_VAR: "/bin/sh"})
    assert explicit[EXECUTOR_ENV_VAR] == "/bin/sh"
    # Nothing resolves: nothing is published.
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: None)
    monkeypatch.delenv("SHELL", raising=False)
    assert EXECUTOR_ENV_VAR not in publish_shell_executor({})


def test_equals_expansion_rules_are_gated_on_the_executor_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    assert not analyze_shell_command("echo ===X===").requires_approval
    assert not analyze_shell_command("echo =ls").requires_approval
    apply_executor_selection("/usr/bin/zsh")
    assert analyze_shell_command("echo ===X===").approval_reasons == (
        "Zsh equals expansion",
    )
    assert analyze_shell_command("echo =ls").approval_reasons == (
        "Zsh equals expansion",
    )


def test_non_equals_word_checks_stay_under_every_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`~name` and `***` are not equals rules and keep firing under bash."""
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    for executor in ("/usr/bin/bash", "/usr/bin/zsh"):
        apply_executor_selection(executor)
        assert analyze_shell_command("echo ~user").approval_reasons == (
            "shell-specific named-directory expansion",
        ), executor
        assert analyze_shell_command("echo ***").approval_reasons == (
            "Zsh symlink-following glob",
        ), executor


def test_resolver_gates_the_equals_family_on_the_executor(
    bash_tool: Bash, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    admitted = bash_tool.resolve_permission(BashArgs(command="echo === hi"))
    assert admitted is not None and admitted.permission is ToolPermission.ALWAYS
    apply_executor_selection("/usr/bin/zsh")
    denied = bash_tool.resolve_permission(BashArgs(command="echo === hi"))
    assert denied is not None and denied.permission is ToolPermission.NEVER
    assert "Zsh equals expansion" in (denied.reason or "")


def test_analysis_cache_is_keyed_on_the_executor_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    with shell_analysis_scope():
        bash_analysis = _analyze_guardrail_source("echo =ls")
        apply_executor_selection("/usr/bin/zsh")
        zsh_analysis = _analyze_guardrail_source("echo =ls")
        # The second family's entry is REUSED, never the first family's.
        assert _analyze_guardrail_source("echo =ls") is zsh_analysis
    assert not bash_analysis.requires_approval
    assert zsh_analysis.approval_reasons == ("Zsh equals expansion",)


def test_os_line_reports_the_resolved_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    assert _get_os_system_prompt().endswith("with shell `/usr/bin/bash`")
    apply_executor_selection("/usr/bin/zsh")
    assert _get_os_system_prompt().endswith("with shell `/usr/bin/zsh`")


# A shell-text probe whose output names the executing shell: bash sets
# BASH_VERSION in the spawned shell itself, a POSIX shell leaves the word list
# at its default. Passed as shell TEXT (never as argv), so it exercises the
# executor the spawn resolves.
_EXECUTOR_PROBE = "printf '%s' ${BASH_VERSION:-posix}"


@pytest.mark.asyncio
async def test_the_shell_spawn_funnel_uses_the_resolved_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    bash_spawn = await spawn_shell_command(_EXECUTOR_PROBE, cwd=tmp_path)
    bash_output, _ = await bash_spawn.communicate()
    assert bash_output.strip() != b"posix"
    apply_executor_selection("/bin/sh")
    posix_spawn = await spawn_shell_command(_EXECUTOR_PROBE, cwd=tmp_path)
    posix_output, _ = await posix_spawn.communicate()
    assert posix_output.strip() == b"posix"


@pytest.mark.asyncio
async def test_the_manual_shell_spawn_uses_the_resolved_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server._shell import ShellController
    from chartreux.app_server.protocol import ShellRunParams

    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    controller = ShellController(tmp_path)

    def params(operation_id: str) -> ShellRunParams:
        return ShellRunParams(
            session_id="session",
            operation_id=operation_id,
            command=_EXECUTOR_PROBE,
            cwd=str(tmp_path),
        )

    assert (await controller.run(params("op-bash"))).stdout.strip() != "posix"
    apply_executor_selection("/bin/sh")
    assert (await controller.run(params("op-posix"))).stdout.strip() == "posix"


@pytest.mark.asyncio
async def test_the_background_job_spawn_uses_the_resolved_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")
    registry = BackgroundJobRegistry()
    bash_process = await registry._spawn_checked(
        _EXECUTOR_PROBE, cwd=tmp_path, extra_env={}
    )
    bash_output, _ = await bash_process.communicate()
    assert bash_output.strip() != b"posix"
    apply_executor_selection("/bin/sh")
    posix_process = await registry._spawn_checked(
        _EXECUTOR_PROBE, cwd=tmp_path, extra_env={}
    )
    posix_output, _ = await posix_process.communicate()
    assert posix_output.strip() == b"posix"


@pytest.mark.asyncio
async def test_the_completion_check_spawn_uses_the_resolved_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CompletionCheckRunner(cwd=tmp_path, workspace=None, path_authority=None)
    spawned: list[object] = []

    class _FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def create_subprocess_shell(*args: object, **kwargs: object) -> object:
        del args
        spawned.append(kwargs["executable"])
        return _FakeProcess()

    async def spawn_registered_process(spawn: object) -> object:
        await spawn  # type: ignore[misc]
        return _FakeProcess()

    monkeypatch.setattr(
        shell_module.asyncio, "create_subprocess_shell", create_subprocess_shell
    )
    monkeypatch.setattr(
        shell_module, "spawn_registered_process", spawn_registered_process
    )
    monkeypatch.setattr(shell_module, "_locate_bash", lambda: "/usr/bin/bash")

    await runner._spawn(
        CompletionCheckTemplate.PYTEST,
        ("printf", "%s", "hi"),
        tmp_path,
        time.monotonic(),
    )
    apply_executor_selection("/bin/sh")
    await runner._spawn(
        CompletionCheckTemplate.PYTEST,
        ("printf", "%s", "hi"),
        tmp_path,
        time.monotonic(),
    )
    assert spawned == ["/usr/bin/bash", "/bin/sh"]


def test_shell_environment_pins_startup_and_locale(
    ambient_shell: pytest.MonkeyPatch,
) -> None:
    ambient_shell.setenv("SHELL", "/usr/bin/zsh")
    env = _shell_environment({
        "BASH_ENV": "/tmp/hostile",
        "ENV": "/tmp/hostile",
        "ZSHENV": "/tmp/hostile",
        "KSH_ENV": "/tmp/hostile",
        "ZDOTDIR": "/home/user",
        "LC_ALL": "C.UTF-8",
        "LC_COLLATE": "C.UTF-8",
    })
    for name in ("BASH_ENV", "ENV", "ZSHENV", "KSH_ENV", "LC_ALL"):
        assert name not in env, name
    assert env["ZDOTDIR"] == "/dev/null/chartreux-pinned-no-zdotdir"
    assert env["LC_COLLATE"] == "C"
    assert env["LC_CTYPE"] == "C.UTF-8"


@pytest.mark.asyncio
async def test_spawn_shell_command_refuses_an_unknown_executor(
    tmp_path: Path, ambient_shell: pytest.MonkeyPatch
) -> None:
    ambient_shell.setenv("SHELL", "/usr/bin/fish")
    assert shell_executor() == "/usr/bin/fish"
    with pytest.raises(UnknownShellExecutorError, match="no pinned executor"):
        await spawn_shell_command("echo hi", cwd=tmp_path)


@pytest.mark.parametrize(
    "command",
    [
        "bash +O extglob -c 'echo hi'",
        "bash -O extglob -c 'echo hi'",
        "bash +o errexit -c 'echo hi'",
        "env CI=1 bash +o errexit -c 'echo hi'",
        "zsh -o extendedglob -c 'echo hi'",
        "zsh +O extendedglob -c 'echo hi'",
        "bash --posix -c 'echo hi'",
        "bash --rcfile startup -c 'echo hi'",
        "bash -ec 'echo hi'",
    ],
)
def test_shell_invocation_option_changes_fail_closed(bash_tool, command: str) -> None:
    """A shell invocation whose own argv changes an option family or the
    startup behavior is outside the pinned executor option contract.
    """
    result = bash_tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER, command
    assert "pinned executor option contract" in (result.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'echo hi'",
        "sh -c 'echo hi'",
        "zsh -c 'echo hi'",
        "bash --noprofile -c 'echo hi'",
        "bash --norc --noediting -c 'echo hi'",
        "bash -c 'uv run python -c \"print(1)\"'",
    ],
)
def test_plain_and_rc_disabling_shell_invocations_stay_admitted(
    bash_tool, command: str
) -> None:
    result = bash_tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.ALWAYS, command


def test_option_changing_shell_invocation_keeps_the_recursive_rm_reason(
    bash_tool,
) -> None:
    """UU: the recursive-rm hard denial REMAINS — the deferred contract
    denial never preempts an earlier payload guard's own reason.
    """
    result = bash_tool.resolve_permission(
        BashArgs(command="bash +O extglob -c 'uv run rm -r build'")
    )
    assert result is not None and result.permission is ToolPermission.NEVER
    assert result.reason == (
        "Command denied: recursive rm option '-r' in 'rm -r build'. "
        "This is a hard guard outside the clean-tree and session scratchpad "
        "carve-outs."
    )


@pytest.mark.parametrize(
    "command",
    [
        "setopt extendedglob; echo hi",
        "setopt NO_NOMATCH; echo hi",
        "shopt -s extglob; echo hi",
        "set -o extglob; echo hi",
        "set -e; echo hi",
        "emulate -L zsh; echo hi",
        "bash -c 'setopt extendedglob; echo hi'",
        "env CI=1 setopt extendedglob; echo hi",
    ],
)
def test_option_changing_builtins_fail_closed(bash_tool, command: str) -> None:
    result = bash_tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER, command
    assert "pinned executor option contract" in (result.reason or ""), command


def test_set_with_only_positional_arguments_stays_admitted(bash_tool) -> None:
    result = bash_tool.resolve_permission(BashArgs(command="set -- x; echo hi"))
    assert result is not None and result.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'echo *.py'",
        "bash -c 'echo ^foo'",
        "bash -c 'ls foo#'",
        "bash -c 'echo a~b'",
        "bash -c \"bash -c 'echo *.py'\"",
        "env CI=1 bash -c 'echo *.py'",
    ],
)
def test_nested_shell_reentry_with_expansion_bearing_payload_fails_closed(
    bash_tool, command: str
) -> None:
    """A nested shell's payload that pathname-expands (or carries extended-glob
    syntax even without *?[ characters) runs under an inner executor this
    process does not pin: fail closed.
    """
    result = bash_tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER, command
    assert "nested shell re-entry" in (result.reason or ""), command


def test_extended_syntax_detection_is_quote_aware() -> None:
    """Extended syntax is detected even without *?[ characters, but only in
    UNQUOTED text: quoted or escaped markers stay literal data.
    """
    # Unquoted extended-glob operators and zsh EXTENDED_GLOB markers:
    assert _has_unquoted_extended_syntax("echo ^foo")
    assert _has_unquoted_extended_syntax("echo foo#")
    assert _has_unquoted_extended_syntax("echo a~b")
    assert _has_unquoted_extended_syntax("echo @(foo)")
    assert _has_unquoted_extended_syntax("echo +(foo)")
    assert _has_unquoted_extended_syntax("echo !(foo)")
    assert _has_unquoted_extended_syntax("echo *(.)")
    # Quoted, escaped, comment, tilde-home and mid-word caret stay literal:
    assert not _has_unquoted_extended_syntax("echo '^foo'")
    assert not _has_unquoted_extended_syntax('echo "^foo"')
    assert not _has_unquoted_extended_syntax("echo \\^foo")
    assert not _has_unquoted_extended_syntax("echo x^y")
    assert not _has_unquoted_extended_syntax("echo #comment")
    assert not _has_unquoted_extended_syntax("echo ~/home")
    assert not _has_unquoted_extended_syntax("echo plain")


@pytest.mark.parametrize(
    "command", ["ls *", "cat *.py", "echo ^foo", "ls foo#", "bash -c 'echo *.py'"]
)
def test_unknown_executor_fails_closed_for_expansion_bearing_commands(
    bash_tool, ambient_shell: pytest.MonkeyPatch, command: str
) -> None:
    ambient_shell.setenv("SHELL", "/usr/bin/fish")
    result = bash_tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.NEVER, command
    assert "no pinned executor option contract" in (result.reason or ""), command


@pytest.mark.parametrize("command", ["echo hi", "git status", "bash -c 'echo hi'"])
def test_unknown_executor_still_admits_plain_commands(
    bash_tool, ambient_shell: pytest.MonkeyPatch, command: str
) -> None:
    ambient_shell.setenv("SHELL", "/usr/bin/fish")
    result = bash_tool.resolve_permission(BashArgs(command=command))
    assert result is not None and result.permission is ToolPermission.ALWAYS, command


def test_executor_boundary_recognition_of_option_forms_is_retained() -> None:
    """UU: the :259 recognition pin REMAINS — the grammar still parses
    `bash +O extglob -c code` as a shell_source boundary; the contract denial
    is layered at resolution, not in the grammar.
    """
    boundary = executor_boundary(shlex.split("bash +O extglob -c code"))
    assert boundary is not None
    assert boundary.boundary_kind == "shell_source"
    assert boundary.inline_payload_ranges == ((4, 5),)
    assert boundary.inline_payloads == ("code",)
