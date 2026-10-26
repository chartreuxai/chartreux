"""The executor grammar model, registry, boundary engine, and package projection.

The declarative grammar model (``ExecutorGrammar``, with option tables built on
``_shell_argv._OptionTable``), the boundary result type (``ExecutorBoundary``
with the ``BoundaryKind``/``ArgumentRange`` aliases), the legacy inline-switch
table (``_INLINE_SWITCHES``), and ``EXECUTOR_REGISTRY`` — all 38 executor
families — are declared here, together with the boundary engine
(``_executor_option``, ``_module_executor_command``, ``_group_shell_boundary``,
``_env_assignment_operand``, ``_sudo_env_operand``, ``_sudo_user_boundary``,
``executor_boundary``)
and the package-adapter projection (``_PACKAGE_PATH_COMMANDS``,
``_PACKAGE_PATH_OPTIONS``, the package path helpers,
``_short_interpreter_option``, ``inline_interpreter_switch``,
``_package_module_tokens``, ``executor_owned_tokens``,
``package_argument_indexes``, ``is_package_command``,
``_executor_package_policy``). ``_executor_package_policy`` constructs
``ShellCommandPolicy``, imported at module level from ``_shell_policy_types``
(stage 4.1): the type module imports nothing from the family, so the binding
is a plain top-level import and the facade round-trip is retired.
The facade re-exports every name defined here, so import paths and object
identity are unchanged by the split; the ``EXECUTOR_REGISTRY`` dict identity is
a mutation seam (``tests/tools/test_shell_command_policy.py`` imports it from
the facade) and the corpus ``registry`` family pins every grammar asdict.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal
from urllib.parse import unquote, urlsplit

from chartreux.core.tools.builtins._shell_argv import (
    _command_name,
    _OptionTable,
    _ParsedOptions,
    _scan_options,
)
from chartreux.core.tools.builtins._shell_policy_types import ShellCommandPolicy

# Only modeled options may advance the scan: an unknown option might consume
# the next word, hiding an execution target behind it.
_INLINE_SWITCHES = {
    "python": (
        "c",
        frozenset("bBdEhiIOPqRsSuvVx"),
        frozenset("WX"),
        {"--version": False},
    ),
    "python3": (
        "c",
        frozenset("bBdEhiIOPqRsSuvVx"),
        frozenset("WX"),
        {"--version": False},
    ),
    "pypy": (
        "c",
        frozenset("bBdEhiIOPqRsSuvVx"),
        frozenset("WX"),
        {"--version": False},
    ),
    "pypy3": (
        "c",
        frozenset("bBdEhiIOPqRsSuvVx"),
        frozenset("WX"),
        {"--version": False},
    ),
    "node": (
        "e",
        frozenset("ip"),
        frozenset("r"),
        {
            "--require": True,
            "--input-type": True,
            "--no-warnings": False,
            "--trace-warnings": False,
            "--no-deprecation": False,
            "--experimental-repl-await": False,
            "--version": False,
        },
    ),
    "perl": ("e", frozenset("wlnpT"), frozenset("IMmF"), {"--version": False}),
    "ruby": (
        "e",
        frozenset("wdv"),
        frozenset("IrCEFKTW"),
        {"--disable": True, "--enable": True, "--version": False},
    ),
}


_MAX_EXECUTOR_DEPTH = 8

BoundaryKind = Literal[
    "none",
    "shell_source",
    "argv_after_keyword",
    "argv_positional",
    "module_target",
    "inline_code_switch",
]
ArgumentRange = tuple[int, int]


@dataclass(frozen=True)
class ExecutorGrammar:
    """Declarative, bounded argv grammar; not a shell parser or package resolver."""

    options: _OptionTable = _OptionTable()
    flags: frozenset[str] = frozenset()
    boundary_kind: BoundaryKind = "none"
    keyword: str | None = None
    prefix_operands: int = 0
    cwd_options: tuple[str, ...] = ()
    inline_options: tuple[str, ...] = ()
    module_options: tuple[str, ...] = ()
    native_target: Literal["go", "cargo"] | None = None
    assignments: bool = False
    single_dash_long: bool = False
    plus_value_options: frozenset[str] = frozenset()
    admitted_applets: frozenset[str] | None = None
    require_leading_assignment: bool = False
    package_executable: bool = False
    child_execution: bool = True
    reject_execution: str | None = None
    # Legacy adapters intentionally retain their historical recognition rules.
    legacy_inline: (
        tuple[str, frozenset[str], frozenset[str], dict[str, bool]] | None
    ) = None


@dataclass(frozen=True)
class ExecutorBoundary:
    """Token ranges are half-open indexes into the original argv, including argv[0].

    consumed_ranges belong to the executor, never the inner command. Payload
    ranges may overlap consumed ranges (including an attached switch/payload).
    Native targets and program arguments are NOT executable argv: callers must
    not recursively dispatch them. cwd_overrides apply only to child execution.
    Unknown registered syntax raises ValueError, never returns a guessed boundary.
    """

    boundary_kind: BoundaryKind
    inner_command_range: ArgumentRange | None = None
    consumed_ranges: tuple[ArgumentRange, ...] = ()
    inline_payload_ranges: tuple[ArgumentRange, ...] = ()
    module_targets: tuple[str, ...] = ()
    cwd_overrides: tuple[str, ...] = ()
    native_targets: tuple[str, ...] = ()
    program_argument_range: ArgumentRange | None = None
    inline_payloads: tuple[str, ...] = ()
    child_execution: bool = False
    module_command: tuple[str, ...] = ()


EXECUTOR_REGISTRY: dict[str, ExecutorGrammar] = {
    "uv": ExecutorGrammar(
        options=_OptionTable(
            long_values=frozenset({
                "--directory",
                "--project",
                "--python",
                "--with",
                "--with-editable",
                "--with-requirements",
                "--package",
                "--extra",
                "--group",
                "--index",
                "--index-url",
                "--default-index",
                "--cache-dir",
                "--config-file",
                "--exclude",
                "--no-group",
                "--env-file",
            }),
            short_values=frozenset("p"),
            boolean_long=frozenset({
                "--no-sync",
                "--locked",
                "--frozen",
                "--isolated",
                "--no-project",
                "--no-dev",
                "--all-extras",
                "--all-groups",
                "--no-cache",
                "--offline",
                "--quiet",
                "--verbose",
                "--no-config",
                "--active",
            }),
        ),
        flags=frozenset("qv"),
        boundary_kind="argv_after_keyword",
        keyword="run",
        cwd_options=("--directory",),
    ),
    "npx": ExecutorGrammar(
        options=_OptionTable(
            long_values=frozenset({
                "--package",
                "--cache",
                "--registry",
                "--userconfig",
            }),
            short_values=frozenset("p"),
            boolean_long=frozenset({"--yes", "--no", "--no-install", "--quiet"}),
        ),
        flags=frozenset("ynq"),
        boundary_kind="argv_positional",
        package_executable=True,
    ),
    "pipx": ExecutorGrammar(
        options=_OptionTable(
            long_values=frozenset({"--spec", "--python", "--pip-args", "--index-url"}),
            boolean_long=frozenset({
                "--verbose",
                "--quiet",
                "--no-cache",
                "--system-site-packages",
                "--pypackages",
                "--editable",
            }),
        ),
        flags=frozenset("vq"),
        boundary_kind="argv_after_keyword",
        keyword="run",
    ),
    # go run compiles .go files (possibly several) or ONE package target; later
    # operands are program arguments. Neither is an executable argv boundary.
    "go": ExecutorGrammar(
        # Canonical -- names let the shared scanner model Go's single-dash
        # whole-name flags without treating -tags as a short-option cluster.
        options=_OptionTable(
            long_values=frozenset({
                "--C",
                "--asmflags",
                "--buildmode",
                "--compiler",
                "--gccgoflags",
                "--gcflags",
                "--ldflags",
                "--mod",
                "--modfile",
                "--overlay",
                "--p",
                "--pkgdir",
                "--tags",
            }),
            boolean_long=frozenset({
                "--a",
                "--n",
                "--race",
                "--msan",
                "--asan",
                "--trimpath",
                "--v",
                "--work",
                "--x",
                "--buildvcs",
            }),
        ),
        boundary_kind="none",
        keyword="run",
        native_target="go",
        single_dash_long=True,
        cwd_options=("--C",),
    ),
    # cargo run selects a manifest/package/bin/example, or the manifest's default
    # binary. Only operands AFTER -- are program arguments, never executable argv.
    "cargo": ExecutorGrammar(
        options=_OptionTable(
            long_values=frozenset({
                "--manifest-path",
                "--package",
                "--bin",
                "--example",
                "--features",
                "--target",
                "--target-dir",
                "--profile",
                "--jobs",
                "--config",
            }),
            short_values=frozenset("pjF"),
            boolean_long=frozenset({
                "--release",
                "--quiet",
                "--verbose",
                "--locked",
                "--frozen",
                "--offline",
                "--all-features",
                "--no-default-features",
            }),
        ),
        flags=frozenset("qvr"),
        keyword="run",
        native_target="cargo",
    ),
    "script": ExecutorGrammar(
        options=_OptionTable(short_values=frozenset("c")),
        boundary_kind="shell_source",
        inline_options=("-c",),
        reject_execution="unsupported script -c shell wrapper",
    ),
    # `env [OPTION]... [-] [NAME=VALUE]... [COMMAND [ARG]...]` runs the
    # command with the assignments applied: the wrapped command argv is
    # re-analyzed exactly like a directly spelled invocation. env applies
    # every pre-command operand containing `=` — putenv(3) does not
    # validate the name — so the operand scan consumes them all, whatever
    # the name (see `_env_assignment_operand`); admission still requires a
    # strictly valid LEADING assignment, so a leading invalid name fails
    # closed. Every other form (a missing command, an unmodeled option)
    # fails closed.
    "env": ExecutorGrammar(
        options=_OptionTable(
            long_values=frozenset({"--unset", "--chdir"}),
            short_values=frozenset("uC"),
            boolean_long=frozenset({"--ignore-environment"}),
        ),
        flags=frozenset("i"),
        boundary_kind="argv_positional",
        assignments=True,
        require_leading_assignment=True,
        cwd_options=("--chdir", "-C"),
    ),
    # `sg` runs its group session the way the shells run -c: the -c payload
    # is shell source. The group name is a prefix operand, so the scan must
    # skip it before looking for the -c option; every other form (interactive
    # shell, direct command operand, nested shell behind the group) is
    # unmodeled and fails closed.
    "sg": ExecutorGrammar(
        options=_OptionTable(short_values=frozenset("c")),
        boundary_kind="shell_source",
        prefix_operands=1,
        inline_options=("-c",),
    ),
    # `sudo -u <user> <command>` runs the command as the named user: the
    # wrapped command argv is re-analyzed exactly like a directly spelled
    # invocation. Only the -u/--user form is modeled — every other sudo
    # form (a root shell, a shell behind -s/-i, an unmodeled option, a
    # missing command, a root sudo without -u) fails closed. sudo's
    # NAME=value operands are environment settings for the wrapped command
    # (sudo(8) parse_args) and are consumed as such — see
    # `_sudo_user_boundary`.
    "sudo": ExecutorGrammar(
        options=_OptionTable(
            long_values=frozenset({"--user"}), short_values=frozenset("u")
        ),
        boundary_kind="argv_positional",
    ),
    "eval": ExecutorGrammar(boundary_kind="shell_source"),
    "command": ExecutorGrammar(
        flags=frozenset("pVv"), boundary_kind="argv_positional", child_execution=False
    ),
    "builtin": ExecutorGrammar(boundary_kind="argv_positional", child_execution=False),
    "exec": ExecutorGrammar(
        options=_OptionTable(short_values=frozenset("a")),
        flags=frozenset("cl"),
        boundary_kind="argv_positional",
        child_execution=False,
    ),
    "busybox": ExecutorGrammar(
        boundary_kind="argv_positional",
        admitted_applets=frozenset({"sh", "ash", "bash", "dash", "hush"}),
    ),
    # toybox is busybox's multi-call twin: every applet runs through the
    # same launcher, so the same admission rule applies — the shell applets
    # are analyzed as shell sources, and every other applet fails closed
    # (an unmodeled `toybox cp` must never reach the operand guard as an
    # unknown command). toybox ships the sh/ash shells.
    "toybox": ExecutorGrammar(
        boundary_kind="argv_positional", admitted_applets=frozenset({"sh", "ash"})
    ),
}

# These argv wrappers exec their first operand as the command. A
# NAME=value operand therefore becomes the exec'd command name, and the
# exec fails (ENOENT) before any wrapped command runs: fail-safe by exec
# semantics, not by model. Unlike env (assignments=True) and sudo (modeled
# NAME=value operands), these wrappers never apply the operand to the
# wrapped command's environment, so the routine resolution of such forms
# is pinned by the PERMISSION_R7_CASES nice/timeout rows rather than
# remodeled.
for _name, (_values, _flags, _operands) in {
    "timeout": (
        ("--signal", "--kill-after", "-s", "-k"),
        ("--foreground", "--preserve-status", "--verbose", "-v"),
        1,
    ),
    "nice": (("--adjustment", "-n"), (), 0),
    "stdbuf": (("-i", "-o", "-e"), (), 0),
    "flock": (
        ("--wait", "--conflict-exit-code", "-w", "-E"),
        (
            "--shared",
            "--exclusive",
            "--unlock",
            "--nonblock",
            "--close",
            "--no-fork",
            "-s",
            "-x",
            "-u",
            "-n",
            "-o",
            "-F",
        ),
        1,
    ),
    "setsid": ((), ("--ctty", "--fork", "--wait", "-c", "-f", "-w"), 0),
    "nohup": ((), (), 0),
    "ionice": (
        ("--class", "--classdata", "-c", "-n", "-p", "-P", "-u"),
        ("--ignore", "-t"),
        0,
    ),
    "taskset": ((), ("--all-tasks", "--cpu-list", "-a", "-c"), 1),
    "time": (
        ("--format", "--output", "-f", "-o"),
        ("--append", "--portability", "--verbose", "--quiet", "-a", "-p", "-v", "-q"),
        0,
    ),
}.items():
    EXECUTOR_REGISTRY[_name] = ExecutorGrammar(
        options=_OptionTable(
            long_values=frozenset(value for value in _values if value.startswith("--")),
            short_values=frozenset(
                value[1] for value in _values if not value.startswith("--")
            ),
            boolean_long=frozenset(flag for flag in _flags if flag.startswith("--")),
        ),
        flags=frozenset(flag[1] for flag in _flags if not flag.startswith("--")),
        boundary_kind="argv_positional",
        prefix_operands=_operands,
    )

for _name in ("ash", "bash", "sh", "dash", "hush", "zsh", "ksh"):
    EXECUTOR_REGISTRY[_name] = ExecutorGrammar(
        options=_OptionTable(
            long_values=frozenset({"--rcfile", "--init-file"}),
            short_values=frozenset("cOo"),
            boolean_long=frozenset({
                "--noprofile",
                "--norc",
                "--posix",
                "--login",
                "--interactive",
                "--noediting",
            }),
        ),
        flags=frozenset("ilrsuvxenfC"),
        boundary_kind="shell_source",
        inline_options=("-c",),
        plus_value_options=frozenset("oO"),
    )

for _name, _legacy in _INLINE_SWITCHES.items():
    _switch, _flags, _values, _long = _legacy
    _python = _name in {"python", "python3", "pypy", "pypy3"}
    EXECUTOR_REGISTRY[_name] = ExecutorGrammar(
        options=_OptionTable(
            long_values=frozenset(key for key, value in _long.items() if value)
            | (frozenset({"--check-hash-based-pycs"}) if _python else frozenset())
            | (frozenset({"--eval", "--print"}) if _name == "node" else frozenset()),
            short_values=_values
            | frozenset(_switch)
            | (frozenset("m") if _python else frozenset())
            | (frozenset("p") if _name == "node" else frozenset()),
            boolean_long=frozenset(key for key, value in _long.items() if not value),
        ),
        flags=_flags,
        boundary_kind="inline_code_switch",
        inline_options=(f"-{_switch}",)
        + (("--eval", "-p", "--print") if _name == "node" else ()),
        module_options=("-m",) if _python else (),
        legacy_inline=_legacy,
    )


def _executor_option(
    tokens: list[str], index: int, grammar: ExecutorGrammar
) -> tuple[int, _ParsedOptions]:
    """Validate BEFORE _scan_options: its permissive unknown-option handling is
    deliberately unsuitable for determining executable boundaries. No long-option
    abbreviation is assumed across executors. Missing/empty values also fail closed.
    """
    token = tokens[index]
    if token.startswith("+"):
        if token[1:] not in grammar.plus_value_options:
            raise ValueError(f"unknown or ambiguous executor option: {token}")
        token = "-" + token[1:]
    if grammar.single_dash_long:
        if token.startswith("--"):
            raise ValueError(f"unknown executor option: {token}")
        token = "-" + token
    table = grammar.options
    count = 1
    if token.startswith("--"):
        option, separator, value = token.partition("=")
        if option in table.long_values:
            if separator:
                if not value:
                    raise ValueError("empty executor option value")
            else:
                count = 2
        elif option not in table.boolean_long or separator:
            raise ValueError(f"unknown or ambiguous executor option: {token}")
    else:
        for offset, short in enumerate(token[1:]):
            if short in table.short_values:
                count = 2 if offset == len(token) - 2 else 1
                break
            if short not in grammar.flags:
                raise ValueError(f"unknown or ambiguous executor option: {token}")
    if count > 1 and (
        index + 1 >= len(tokens)
        or not tokens[index + 1]
        or tokens[index + 1].startswith("-")
    ):
        raise ValueError(f"missing or ambiguous executor option value: {token}")
    return count, _scan_options([token, *tokens[index + 1 : index + count]], table)


def _module_executor_command(value: str, arguments: list[str]) -> tuple[str, ...]:
    known_modules = EXECUTOR_REGISTRY.keys() | {"pip", "pip3"}
    if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value) is None:
        raise ValueError("ambiguous interpreter module target")
    if value in known_modules:
        return (value, *arguments)
    if value.split(".")[0] in known_modules:
        raise ValueError("unrecognized executor module target")
    return ()


def _group_shell_boundary(
    name: str, tokens: list[str], grammar: ExecutorGrammar
) -> ExecutorBoundary:
    """Boundary for a group wrapper whose -c payload is shell source (``sg``).

    ``sg <group> -c <payload>`` runs the payload through a local shell, so
    the payload is shell source exactly like a shell's own -c. The group
    name is a prefix operand the option scan must skip first; every other
    form (interactive shell, direct command operand, a nested shell behind
    the group) is unmodeled and fails closed. A payload spanning several
    argv tokens is unquoted shell word splitting: the remaining tokens join
    the source (the eval precedent) so a gated payload spelled unquoted is
    analyzed exactly like its quoted form instead of degrading to the bare
    first word with the rest dismissed as program arguments.
    """
    prefix = grammar.prefix_operands
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-") or token == "-":
            if prefix == 0:
                raise ValueError(f"unsupported {name} form")
            prefix -= 1
            index += 1
            continue
        count, parsed = _executor_option(tokens, index, grammar)
        if any(option in grammar.inline_options for option, _ in parsed.values):
            if count == 1:
                raise ValueError("shell -c requires a separate source operand")
            payload = index + 1
            if payload + 1 < len(tokens):
                return ExecutorBoundary(
                    "shell_source",
                    None,
                    ((0, len(tokens)),),
                    ((payload, len(tokens)),),
                    inline_payloads=(" ".join(tokens[payload:]),),
                )
            return ExecutorBoundary(
                "shell_source",
                None,
                ((0, payload + 1),),
                ((payload, payload + 1),),
                program_argument_range=(payload + 1, len(tokens)),
                inline_payloads=(tokens[payload],),
            )
        index += count
    raise ValueError(f"unsupported {name} form")


def _env_assignment_operand(token: str) -> bool:
    """Whether env(1) applies a pre-command operand as an environment setting.

    env's operand scan applies every operand containing ``=`` — putenv(3)
    does not validate the name, so ``env FOO=1 1X=1 printenv 1X`` prints
    the value, an empty name is consumed (with a warning) while the command
    still runs, and a name containing ``/`` is set verbatim — until the
    first operand without ``=``, which env execs as the command. Operands
    starting with ``-`` are excluded here: they keep flowing to option
    parsing, which fails closed on unknown options (GNU env's own scan
    would putenv them, so refusing them only over-gates).
    """
    return "=" in token and not token.startswith("-")


def _sudo_env_operand(token: str) -> bool:
    """Whether sudo(8) applies a command-line operand as an environment setting.

    sudo's parse_args treats an operand containing ``=`` — starting with
    neither ``/``, ``-``, nor ``=`` — as a NAME=value environment setting
    for the wrapped command (its ``is_envar``), interspersed with options.
    """
    return "=" in token and not token.startswith(("/", "-", "="))


def _sudo_user_boundary(
    name: str, tokens: list[str], grammar: ExecutorGrammar
) -> ExecutorBoundary:
    """Boundary for sudo's transparent ``-u <user>`` wrapper form.

    ``sudo -u <user> <command>`` runs the command as the named user, so
    the command argv is re-analyzed exactly like a directly spelled
    invocation. Only the -u/--user form is modeled: a sudo without it (a
    root shell or a root command), an unmodeled option, a shell behind
    -s/-i, and a missing command all fail closed.

    sudo also applies NAME=value operands as environment settings for the
    wrapped command, interspersed with its options, so the scan consumes
    them (the resolver recovers them as the child's assignments) and the
    wrapped command is re-analyzed exactly like a directly spelled
    invocation. After ``--`` sudo execs the next operand as the command
    even when it looks like NAME=value — the exec then fails (ENOENT), so
    consuming assignment-shaped operands there too only over-gates a form
    that never runs: fail-closed either way.
    """
    index = 1
    user_seen = False
    while index < len(tokens):
        token = tokens[index]
        if _sudo_env_operand(token):
            index += 1
            continue
        if token == "--":
            index += 1
            while index < len(tokens) and _sudo_env_operand(tokens[index]):
                index += 1
            break
        if not token.startswith("-") or token == "-":
            break  # The wrapped command starts here.
        count, parsed = _executor_option(tokens, index, grammar)
        if parsed.values_for("-u", "--user"):
            user_seen = True
        index += count
    if not user_seen or index >= len(tokens):
        raise ValueError(f"unsupported {name} form")
    if tokens[index].startswith("-") and tokens[index - 1] != "--":
        raise ValueError(f"unsupported {name} form")
    return ExecutorBoundary(
        "argv_positional",
        (index, len(tokens)),
        ((0, index),),
        child_execution=grammar.child_execution,
    )


def executor_boundary(  # noqa: PLR0911, PLR0912, PLR0914, PLR0915
    tokens: list[str], *, admission: bool = False
) -> ExecutorBoundary | None:
    """Return a registry-owned boundary; admission retains legacy restrictions.

    npx's first positional is its package-derived executable (unless --package
    supplies the package); pipx run's first positional is the app, NOT --spec's
    package operand. uv run executes its first positional as argv. go/cargo never
    expose compiled targets or program arguments as commands. Unsupported syntax
    is intentionally rejected rather than inferred from the next positional.
    """
    if (
        not tokens
        or (grammar := EXECUTOR_REGISTRY.get(_command_name(tokens[0]))) is None
    ):
        return None
    name = _command_name(tokens[0])
    if admission and grammar.reject_execution:
        raise ValueError(grammar.reject_execution)
    if (
        admission
        and grammar.require_leading_assignment
        and (
            not tokens[2:]
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[1], re.DOTALL) is None
        )
    ):
        raise ValueError("unsupported env wrapper form")
    if (
        admission
        and grammar.admitted_applets is not None
        and (not tokens[1:] or tokens[1] not in grammar.admitted_applets)
    ):
        raise ValueError(f"unsupported {name} applet")
    index = 1
    keyword_seen = grammar.keyword is None
    cwd: list[str] = []
    targets: list[str] = []
    process_selection = False
    kind = grammar.boundary_kind
    if name == "eval":
        return ExecutorBoundary(
            kind,
            consumed_ranges=((0, len(tokens)),),
            inline_payload_ranges=((1, len(tokens)),),
            inline_payloads=(" ".join(tokens[1:]),),
        )
    if name == "sudo":
        return _sudo_user_boundary(name, tokens, grammar)
    if grammar.prefix_operands and kind == "shell_source":
        return _group_shell_boundary(name, tokens, grammar)

    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            if admission and kind == "shell_source" and "-c" in tokens[index + 1 :]:
                raise ValueError("unsupported shell -- -c option form")
            index += 1
            break
        if grammar.assignments and token == "-":
            index += 1  # env's legacy ignore-environment selector.
            continue
        if (
            not (
                token.startswith("-")
                or (grammar.plus_value_options and token.startswith("+"))
            )
            or token == "-"
        ):
            if not keyword_seen:
                if token != grammar.keyword:
                    return ExecutorBoundary("none")
                keyword_seen = True
                index += 1
                continue
            if grammar.assignments and _env_assignment_operand(token):
                index += 1
                continue
            break
        if name == "nice" and re.fullmatch(r"-\d+", token):
            index += 1
            continue
        count, parsed = _executor_option(tokens, index, grammar)
        if name == "ionice" and parsed.values_for("-p", "-P", "-u"):
            process_selection = True
        cwd.extend(parsed.values_for(*grammar.cwd_options))
        if len(cwd) > 1:
            raise ValueError("ambiguous repeated executor directory override")
        if (
            grammar.single_dash_long
            and keyword_seen
            and parsed.values_for(*grammar.cwd_options)
        ):
            raise ValueError("go -C must precede run")
        if grammar.native_target == "cargo":
            targets.extend(
                parsed.values_for(
                    "--manifest-path", "--package", "-p", "--bin", "--example"
                )
            )
        selected = next(
            (
                (option, value)
                for option, value in parsed.values
                if option in grammar.inline_options + grammar.module_options
            ),
            None,
        )
        if selected is not None:
            if kind == "shell_source" and count == 1:
                raise ValueError("shell -c requires a separate source operand")
            if (
                name in {"perl", "ruby"}
                and index + count < len(tokens)
                and tokens[index + count].startswith("-")
                and tokens[index + count] != "--"
            ):
                # These interpreters may resume option parsing (including more
                # code switches) after a snippet. Do not label that code argv.
                raise ValueError("unsupported options after inline code")
            option, value = selected
            payload = (index + count - 1, index + count)
            end = index + count
            selected_kind: BoundaryKind = (
                "module_target" if option in grammar.module_options else kind
            )
            return ExecutorBoundary(
                selected_kind,
                None,
                ((0, end),),
                () if selected_kind == "module_target" else (payload,),
                (value,) if selected_kind == "module_target" else (),
                tuple(cwd),
                program_argument_range=(end, len(tokens)),
                inline_payloads=() if selected_kind == "module_target" else (value,),
                module_command=(
                    _module_executor_command(value, tokens[end:])
                    if selected_kind == "module_target"
                    else ()
                ),
            )
        index += count
    if not keyword_seen:
        raise ValueError("missing executor subcommand")
    if grammar.native_target == "cargo":
        if index < len(tokens) and tokens[index - 1] != "--":
            raise ValueError("cargo run program arguments require --")
        return ExecutorBoundary(
            "none",
            consumed_ranges=((0, index),),
            cwd_overrides=tuple(cwd),
            native_targets=tuple(targets),
            program_argument_range=(index, len(tokens)),
        )
    if grammar.native_target == "go":
        if index >= len(tokens):
            raise ValueError("missing go run target")
        end = index + 1
        if tokens[index].endswith(".go"):
            while end < len(tokens) and tokens[end].endswith(".go"):
                end += 1
        return ExecutorBoundary(
            "none",
            consumed_ranges=((0, end),),
            cwd_overrides=tuple(cwd),
            native_targets=tuple(tokens[index:end]),
            program_argument_range=(end, len(tokens)),
        )
    if grammar.assignments:
        # env still consumes assignment operands after its option
        # terminator, whatever the operand's name.
        while index < len(tokens) and _env_assignment_operand(tokens[index]):
            index += 1
    index += grammar.prefix_operands
    if kind in {"inline_code_switch", "shell_source"}:
        # Script operands remain native interpreter targets, not executable argv.
        return ExecutorBoundary(
            "none",
            consumed_ranges=((0, min(index + 1, len(tokens))),),
            cwd_overrides=tuple(cwd),
            native_targets=tuple(tokens[index : index + 1]),
            program_argument_range=(min(index + 1, len(tokens)), len(tokens)),
        )
    if index >= len(tokens):
        if process_selection:
            return ExecutorBoundary("none", consumed_ranges=((0, len(tokens)),))
        raise ValueError("missing executor execution target")
    if tokens[index].startswith("-") and (index == 0 or tokens[index - 1] != "--"):
        raise ValueError("ambiguous executor execution target")
    if name == "uv" and tokens[index].endswith((".py", ".pyw")):
        raise ValueError(
            "unsupported uv implicit script execution; use an explicit interpreter"
        )
    # command -v/-V describes a command rather than executing it.
    if name == "command" and any(
        flag in token[1:]
        for token in tokens[1:index]
        if token.startswith("-") and token != "--"
        for flag in "vV"
    ):
        return ExecutorBoundary("none", consumed_ranges=((0, len(tokens)),))
    return ExecutorBoundary(
        kind,
        (index, len(tokens)),
        ((0, index),),
        cwd_overrides=tuple(cwd),
        child_execution=grammar.child_execution,
    )


_PACKAGE_PATH_COMMANDS = frozenset({
    "bun",
    "cargo",
    "go",
    "npm",
    "npx",
    "pip",
    "pip3",
    "pipx",
    "pnpm",
    "poetry",
    "uv",
    "yarn",
})
_PACKAGE_PATH_OPTIONS = frozenset({
    "--prefix",
    "--target",
    "--root",
    "--cache-dir",
    "--src",
    "--path",
    "--directory",
    "--cwd",
    "--project",
    "--store-dir",
    "--cache-folder",
    "--target-dir",
    "--global-dir",
    "--modules-folder",
    "--virtualenv",
    "--dest",
    "--editable",
    "--config-file",
    "--install-dir",
    "--log",
    "--report",
    "--cache",
    "--userconfig",
    "--globalconfig",
    "--logs-dir",
    "--output",
    "--output-file",
    "--out-dir",
    "--download",
    "--download-dir",
    "--wheel-dir",
    "--requirements",
    "--requirement",
    "--requirements-from-script",
    "--destination-dir",
    "--destination-directory",
    "--source",
    "--source-dir",
    "--source-directory",
    "--pack-destination",
    "--constraints",
    "--constraint",
    "--build-constraints",
    "--build-constraint",
    "--overrides",
    "--excludes",
    "--env-file",
})


def _package_local_path(value: str) -> str:
    """Normalize local package URLs; fail closed on unmodeled URL syntax."""
    # Extras on a local project are requirement syntax, not shell globs.
    local = re.fullmatch(r"(\.{1,2}(?:/[^\[\]]*)?|/[^\[\]]+)\[[\w., -]+\]", value)
    if local is not None:
        value = local[1]
    if "file:" not in value.casefold():
        return value
    # npm name@file: and pip's quoted 'name @ file:' direct references.
    match = re.fullmatch(r"(?:[\w.@/\[\],-]+\s*@\s*)?(file:.*)", value, re.IGNORECASE)
    if match is None:
        raise ValueError("Unmodeled local package URL")
    url = match[1]
    parsed = urlsplit(url)
    if (
        any((parsed.netloc, parsed.query, parsed.fragment, not parsed.path))
        or re.search(r"%(?![0-9a-fA-F]{2})", parsed.path)
        or any(character in url for character in "\\\r\n\t")
    ):
        raise ValueError("Unmodeled local package URL")
    path = unquote(parsed.path, errors="strict")
    if "\x00" in path or path.startswith("//"):
        raise ValueError("Unmodeled local package URL")
    return path


def _package_requirement_path(value: str) -> str | None:
    # Named requirements (including extras and version constraints) are not paths.
    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9._-]+(?:\s*,\s*[A-Za-z0-9._-]+)*\])"
        r"(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+(?:\s*,\s*(?:==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+)*)?",
        value,
    ):
        return None
    return _package_local_path(value)


def _short_interpreter_option(
    token: str, switch: str, flags: frozenset[str], values: frozenset[str]
) -> int:
    """Return -1 for inline code, 0 for unknown, or words consumed."""
    short = token[1:]
    for position, flag in enumerate(short):
        if flag == switch:
            return -1
        if flag in values:
            return 2 if position == len(short) - 1 else 1
        if flag not in flags:
            return 0
    return 1


def inline_interpreter_switch(tokens: list[str]) -> str | None:
    """Recognize inline code before the execution target, not script arguments.

    Unclassified options before an apparent inline selector fail closed rather
    than guessing whether they consume the following word.
    """
    name = _command_name(tokens[0]) if tokens else ""
    grammar = EXECUTOR_REGISTRY.get(name)
    if grammar is None or grammar.legacy_inline is None:
        return None
    switch, flags, values, long_options = grammar.legacy_inline
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--" or (
            name in {"python", "python3", "pypy", "pypy3"} and token.startswith("-m")
        ):
            return None  # Module and later words (or --) are program arguments.
        if name == "node" and (token == "--eval" or token.startswith("--eval=")):
            return "node -e"
        if token.startswith("--"):
            option, separator, value = token.partition("=")
            if option not in long_options:
                break
            takes_value = long_options[option]
            if separator and (not takes_value or not value):
                break
            consumed = 2 if takes_value and not separator else 1
        elif token.startswith("-") and token != "-":
            consumed = _short_interpreter_option(token, switch, flags, values)
            if consumed == -1:
                return f"{name} -{switch}"
        else:
            return None  # Script operand or stdin target.
        if not consumed or index + consumed > len(tokens):
            break
        index += consumed
    if any(
        token.startswith("-")
        and not token.startswith("--")
        and switch in token[1:]
        or (name == "node" and token.startswith("--eval"))
        for token in tokens[index + 1 :]
    ):
        raise ValueError(f"unsupported {name} option before inline code")
    return None


def _package_module_tokens(tokens: list[str]) -> list[str]:
    """Recognize Python's module target, stopping before script arguments."""
    name = _command_name(tokens[0]) if tokens else ""
    if name not in {"python", "python3", "pypy", "pypy3"}:
        return tokens
    grammar = EXECUTOR_REGISTRY[name]
    assert grammar.legacy_inline is not None
    _, flags, values, _ = grammar.legacy_inline
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--" or not token.startswith("-"):
            return tokens
        if token.startswith("--"):
            option, separator, _ = token.partition("=")
            if option == "--check-hash-based-pycs":
                index += 1 if separator else 2
                continue
            break
        consumed = _short_interpreter_option(token, "m", flags, values)
        if consumed == -1:
            offset = token.index("m") + 1
            module = token[offset:] or (
                tokens[index + 1] if index + 1 < len(tokens) else ""
            )
            package = module.removesuffix(".__main__")
            if package in _PACKAGE_PATH_COMMANDS:
                return [package, *tokens[index + (1 if token[offset:] else 2) :]]
            return tokens
        if consumed <= 0:
            return tokens
        index += consumed
    return tokens


def executor_owned_tokens(tokens: list[str]) -> list[str]:
    """Project executor-owned argv; child argv and inline source are opaque here."""
    boundary = executor_boundary(tokens)
    if boundary is None:
        return tokens
    if boundary.module_command and boundary.program_argument_range:
        return tokens[: boundary.program_argument_range[0]]
    end = (
        boundary.inner_command_range[0] if boundary.inner_command_range else len(tokens)
    )
    if (
        boundary.inner_command_range
        and EXECUTOR_REGISTRY[_command_name(tokens[0])].package_executable
    ):
        end += 1  # npx's executable selector is also a package requirement.
    if (
        boundary.program_argument_range
        and boundary.native_targets
        and not EXECUTOR_REGISTRY[_command_name(tokens[0])].native_target
    ):
        end = boundary.program_argument_range[0]
    excluded = {
        index
        for start, stop in boundary.inline_payload_ranges
        for index in range(start, stop)
    }
    grammar = EXECUTOR_REGISTRY[_command_name(tokens[0])]
    if grammar.keyword and boundary.consumed_ranges:
        index = 1
        while index < end and tokens[index].startswith("-"):
            count, _ = _executor_option(tokens, index, grammar)
            index += count
        if index < end and tokens[index] == grammar.keyword:
            excluded.add(index)
    return [token for index, token in enumerate(tokens[:end]) if index not in excluded]


def package_argument_indexes(tokens: list[str], *, depth: int = 0) -> set[int]:
    """Original argv indexes needing quote-aware package-glob checks."""
    if depth > _MAX_EXECUTOR_DEPTH:
        raise ValueError("shell nesting depth limit exceeded")
    boundary = executor_boundary(tokens)
    if boundary and boundary.module_command and boundary.program_argument_range:
        start, _ = boundary.program_argument_range
        return {
            start + index - 1
            for index in package_argument_indexes(
                list(boundary.module_command), depth=depth + 1
            )
        }
    owned = executor_owned_tokens(tokens)
    end = (
        boundary.inner_command_range[0]
        if boundary and boundary.inner_command_range
        else len(tokens)
    )
    if (
        boundary
        and boundary.inner_command_range
        and EXECUTOR_REGISTRY[_command_name(tokens[0])].package_executable
    ):
        end += 1
    indexes = set(range(1, end)) if is_package_command(owned) else set()
    if boundary:
        for start, stop in boundary.inline_payload_ranges:
            indexes.difference_update(range(start, stop))
        if boundary.inner_command_range:
            start, stop = boundary.inner_command_range
            indexes.update(
                start + index
                for index in package_argument_indexes(
                    tokens[start:stop], depth=depth + 1
                )
            )
    return indexes


def is_package_command(tokens: list[str]) -> bool:
    """Identify commands whose requirement syntax needs quote-aware analysis."""
    tokens = _package_module_tokens(tokens)
    return bool(tokens) and _command_name(tokens[0]) in _PACKAGE_PATH_COMMANDS


def _executor_package_policy(
    tokens: list[str], boundary: ExecutorBoundary
) -> ShellCommandPolicy:
    """Inspect package/executor operands with the SAME grammar as expansion.

    Native program argv remains conservatively path-inspected, but never becomes
    executable argv or resumes parsing the compiler's options.
    """
    grammar = EXECUTOR_REGISTRY[_command_name(tokens[0])]
    end = (
        boundary.inner_command_range[0]
        if boundary.inner_command_range
        else boundary.program_argument_range[0]
        if boundary.program_argument_range
        else len(tokens)
    )
    index = 1
    paths: list[str] = []
    operands: list[str] = []
    path_options = _PACKAGE_PATH_OPTIONS | frozenset({
        "--manifest-path",
        "--config",
        "--modfile",
        "--overlay",
        "--pkgdir",
        "--with-requirements",
        "--C",
    })
    if grammar.native_target == "cargo":
        path_options -= {"--target"}  # Rust target triple, not a directory.
    requirement_options = {"--with", "--with-editable", "--package", "--spec"}
    if grammar.package_executable:
        requirement_options.add("-p")
    try:
        while index < end:
            token = tokens[index]
            if token == "--":
                operands.extend(tokens[index + 1 : end])
                break
            if token.startswith("-"):
                count, parsed = _executor_option(tokens, index, grammar)
                paths.extend(
                    _package_local_path(value)
                    for value in parsed.values_for(*path_options)
                )
                operands.extend(parsed.values_for(*requirement_options))
                index += count
            else:
                if token != grammar.keyword:
                    operands.append(token)
                index += 1
        if grammar.package_executable and boundary.inner_command_range:
            operands.append(tokens[boundary.inner_command_range[0]])
        if grammar.native_target and boundary.program_argument_range:
            start, stop = boundary.program_argument_range
            operands.extend(tokens[start:stop])
        positionals = tuple(
            path
            for value in operands
            if (path := _package_requirement_path(value)) is not None
        )
    except ValueError:
        return ShellCommandPolicy(
            requires_approval=True, denial_reason="unmodeled local package URL"
        )
    return ShellCommandPolicy(
        inspect_positional_paths=True,
        option_path_values=tuple(paths),
        positional_values=positionals,
    )
