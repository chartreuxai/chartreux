from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
import stat
from typing import Literal
from urllib.parse import unquote, urlsplit

from chartreux.core.dangerous_commands import (
    DangerousCommandClassification,
    GitRepositoryRelocation,
    GrantableGitCall,
    ModeledGitCall,
    RepositoryScope,
    operations_for_ids,
)
from chartreux.core.tools.builtins._shell_diagnostics import Diagnostic
from chartreux.core.tools.utils import resolve_tool_path

# Length of a single-character short option ("-i") and the minimum length of
# an unambiguous combined short-option cluster ("-ic").
_SHORT_OPTION_LENGTH = 2
_COMBINED_SHORT_OPTION_MIN_LENGTH = 3
# GNU chmod's octal-mode value ceiling; larger values fail as "mode too large".
_CHMOD_MAX_NUMERIC_MODE = 0o7777
_MAX_LESS_RESUMED_OPTIONS = 256


@dataclass(frozen=True)
class ShellCommandPolicy:
    requires_approval: bool = False
    denial_reason: str | None = None
    diagnostic: Diagnostic | None = None

    def denial_diagnostic(self, command_part: str) -> Diagnostic:
        if self.diagnostic is not None:
            return Diagnostic(
                self.diagnostic.code,
                self.diagnostic.offending_token,
                command_part,
                self.diagnostic.detail,
            )
        return Diagnostic(
            "policy",
            command_part=command_part,
            detail=self.denial_reason
            or "unsafe or unmodeled command options are not permitted",
        )

    inspect_positional_paths: bool = False
    # Test-only classification contract: runtime scopes all sed operands alike,
    # including in-place writes; no reader exemption depends on this flag.
    mutates_files: bool = False
    # The command is a Git reader whose repository-owned config may execute
    # code; the permission resolver inspects that config before permitting it.
    inspect_git_repository: bool = False
    option_path_values: tuple[str, ...] = ()
    # Path operands that name files the command WRITES itself (git archive
    # --output, a writing git config --file, git apply --directory, an
    # init/worktree-add/move or submodule-add destination). Unlike
    # option_path_values, these stay checked against protected Git metadata
    # even when the command is git: git's -C/--git-dir/--work-tree
    # selectors locate the repository, but these operands write it.
    write_path_values: tuple[str, ...] = ()
    positional_values: tuple[str, ...] | None = None


@dataclass(frozen=True)
class _OptionTable:
    long_values: frozenset[str] = frozenset()
    short_values: frozenset[str] = frozenset()
    optional_long_values: frozenset[str] = frozenset()
    optional_short_values: frozenset[str] = frozenset()
    boolean_long: frozenset[str] = frozenset()
    optional_long_values_consume_next: bool = False


@dataclass(frozen=True)
class _ParsedOptions:
    options: tuple[str, ...]
    values: tuple[tuple[str, str], ...]
    positionals: tuple[str, ...]

    def values_for(self, *names: str) -> tuple[str, ...]:
        wanted = set(names)
        return tuple(value for option, value in self.values if option in wanted)


def _is_combined_short_options(token: str) -> bool:
    """Whether *token* is unambiguously a cluster of one-character options.

    Only tokens that start with a single dash and carry at least two further
    characters qualify: getopt parses ``-ic`` as ``-i -c``. Long options
    (``--interactive``) and plain words never qualify.
    """
    return (
        token.startswith("-")
        and not token.startswith("--")
        and len(token) >= _COMBINED_SHORT_OPTION_MIN_LENGTH
    )


def _argument_matches(pattern: str, token: str) -> bool:
    """Compare one command argument token against a pattern argument token.

    Tokens compare literally, except that a single-character short-option
    pattern (``-i``) also matches a combined short-option token (``-ic``)
    containing that character: ``bash -ic x`` is the interactive shell the
    ``bash -i`` denylist entry exists to block. Anything that is not
    unambiguously a combined short-option cluster only matches exactly, so
    ``-i`` never matches ``--ignore`` or an arbitrary word.
    """
    if pattern == token:
        return True
    if not (
        len(pattern) == _SHORT_OPTION_LENGTH and pattern[0] == "-" and pattern[1] != "-"
    ):
        return False
    return _is_combined_short_options(token) and pattern[1] in token[1:]


def matches_command_prefix(tokens: list[str], pattern_tokens: list[str]) -> bool:
    """Match a command prefix while normalizing only the executable token."""
    if not tokens or not pattern_tokens or len(tokens) < len(pattern_tokens):
        return False
    executable = tokens[0]
    pattern_executable = pattern_tokens[0]
    if (
        executable != pattern_executable
        and Path(executable).name != Path(pattern_executable).name
    ):
        return False
    return all(
        _argument_matches(pattern, token)
        for token, pattern in zip(tokens[1:], pattern_tokens[1:], strict=False)
    )


def _command_name(token: str) -> str:
    """Normalize a command token to the name policy tables are keyed by."""
    return token.strip("\"'").replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _match_long(token: str, choices: frozenset[str]) -> str | None:
    name = token.partition("=")[0]
    exact = name if name in choices else None
    if exact is not None:
        return exact
    matches = [choice for choice in choices if name != "--" and choice.startswith(name)]
    return matches[0] if len(matches) == 1 else None


def _scan_options(args: list[str], table: _OptionTable) -> _ParsedOptions:
    """Parse option boundaries once, never reparsing a consumed operand.

    Required operands consume the next token even when it is ``--``. Optional
    operands are accepted only in attached/``=`` form, except for commands
    whose native parser consumes a following token for optional operands.
    """
    options: list[str] = []
    values: list[tuple[str, str]] = []
    positionals: list[str] = []
    all_long = table.long_values | table.optional_long_values | table.boolean_long
    index = 0
    options_ended = False
    while index < len(args):
        token = args[index]
        if options_ended:
            positionals.append(token)
            index += 1
            continue
        if token == "--":
            options_ended = True
            index += 1
            continue
        if token.startswith("--"):
            option = _match_long(token, all_long)
            options.append(option or token)
            if option is None:
                index += 1
                continue
            _, separator, attached = token.partition("=")
            if separator:
                values.append((option, attached))
                index += 1
            elif (
                option in table.long_values
                or (
                    option in table.optional_long_values
                    and table.optional_long_values_consume_next
                )
            ) and index + 1 < len(args):
                values.append((option, args[index + 1]))
                index += 2
            else:
                index += 1
            continue
        if token.startswith("-") and token != "-":
            options.append(token)
            chars = token[1:]
            for offset, short in enumerate(chars):
                if short not in table.short_values | table.optional_short_values:
                    continue
                attached = chars[offset + 1 :]
                if attached:
                    values.append((f"-{short}", attached))
                elif short in table.short_values and index + 1 < len(args):
                    values.append((f"-{short}", args[index + 1]))
                    index += 1
                break
            index += 1
            continue
        positionals.append(token)
        index += 1
    return _ParsedOptions(tuple(options), tuple(values), tuple(positionals))


def _has_long(parsed: _ParsedOptions, *names: str) -> bool:
    choices = frozenset(names)
    return any(
        token.startswith("--") and _match_long(token, choices)
        for token in parsed.options
    )


def _has_short(
    parsed: _ParsedOptions, names: frozenset[str], table: _OptionTable
) -> bool:
    value_shorts = table.short_values | table.optional_short_values
    for token in parsed.options:
        if not token.startswith("-") or token.startswith("--"):
            continue
        for short in token[1:]:
            if short in names:
                return True
            if short in value_shorts:
                break
    return False


def _has_short_in_cluster(parsed: _ParsedOptions, names: frozenset[str]) -> bool:
    """Whether a short cluster carries *names* even after value-taking shorts.

    tree's parser takes a value-taking short's operand from the next argv
    token and then resumes scanning the same cluster (verified against tree
    2.3.2), so a flag later in the cluster is still an option. Upstream flags
    any such ``o`` rather than modeling the resume precisely, accepting the
    conservative denial when an attached value merely contains the letter.
    """
    return any(
        token.startswith("-")
        and not token.startswith("--")
        and any(short in names for short in token[1:])
        for token in parsed.options
    )


# Only modeled options may advance the scan: an unknown option might consume
# the next word, hiding an execution target behind it.
_INLINE_SWITCHES = {
    "python": ("c", frozenset("bBdEhiIOPqRsSuvVx"), frozenset("WX"), {}),
    "python3": ("c", frozenset("bBdEhiIOPqRsSuvVx"), frozenset("WX"), {}),
    "pypy": ("c", frozenset("bBdEhiIOPqRsSuvVx"), frozenset("WX"), {}),
    "pypy3": ("c", frozenset("bBdEhiIOPqRsSuvVx"), frozenset("WX"), {}),
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
        },
    ),
    "perl": ("e", frozenset("wlnpT"), frozenset("IMmF"), {}),
    "ruby": (
        "e",
        frozenset("wdv"),
        frozenset("IrCEFKTW"),
        {"--disable": True, "--enable": True},
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
            if grammar.assignments and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*=.*", token, re.DOTALL
            ):
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
        # GNU env still consumes assignments after its option terminator.
        while index < len(tokens) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[index], re.DOTALL
        ):
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


_TABLES = {
    "date": _OptionTable(
        frozenset({"--date", "--file", "--reference", "--set", "--rfc-3339"}),
        frozenset("dfrs"),
        frozenset({"--iso-8601"}),
        frozenset("I"),
    ),
    "diff": _OptionTable(
        frozenset({
            "--changed-group-format",
            "--exclude-from",
            "--from-file",
            "--horizon-lines",
            "--ignore-matching-lines",
            "--label",
            "--line-format",
            "--new-group-format",
            "--new-line-format",
            "--old-group-format",
            "--old-line-format",
            "--show-function-line",
            "--starting-file",
            "--tabsize",
            "--to-file",
            "--unchanged-group-format",
            "--unchanged-line-format",
            "--width",
        }),
        frozenset("FILSWX"),
    ),
    "du": _OptionTable(
        frozenset({
            "--block-size",
            "--exclude",
            "--exclude-from",
            "--files0-from",
            "--max-depth",
            "--threshold",
            "--time-style",
        }),
        frozenset("BdtX"),
        frozenset({"--time"}),
    ),
    "wc": _OptionTable(frozenset({"--files0-from"})),
    "grep": _OptionTable(
        frozenset({
            "--after-context",
            "--before-context",
            "--binary-files",
            "--context",
            "--directories",
            "--exclude",
            "--exclude-dir",
            "--exclude-from",
            "--file",
            "--group-separator",
            "--include",
            "--label",
            "--max-count",
            "--regexp",
        }),
        frozenset("ABCDdefm"),
        boolean_long=frozenset({"--binary"}),
    ),
    "file": _OptionTable(
        frozenset({
            "--exclude",
            "--exclude-quiet",
            "--files-from",
            "--magic-file",
            "--parameter",
            "--separator",
        }),
        frozenset("eFfmP"),
    ),
    "sort": _OptionTable(
        frozenset({
            "--batch-size",
            "--buffer-size",
            "--compress-program",
            "--files0-from",
            "--key",
            "--output",
            "--parallel",
            "--random-source",
            "--sort",
            "--temporary-directory",
            "--field-separator",
        }),
        frozenset("koSTt"),
    ),
    "tree": _OptionTable(
        frozenset({
            "--charset",
            "--filelimit",
            "--gitfile",
            "--output",
            "--sort",
            "--timefmt",
        }),
        frozenset("HILPTo"),
    ),
}


_BSD_DATE_COMPONENT_LENGTHS = frozenset({2, 4, 6, 8, 10, 12})
_BSD_DATE_SECONDS_LENGTH = 2


def _date_short_option_state(token: str) -> tuple[bool, bool, bool]:
    """Return -j seen, -f seen, and whether the next argument is consumed."""
    has_no_set = False
    has_input_format = False
    value_options = frozenset({"d", "f", "r", "v", "z"})
    for index, option in enumerate(token[1:]):
        has_no_set = has_no_set or option == "j"
        has_input_format = has_input_format or option == "f"
        if option == "I":
            break
        if option in value_options:
            return has_no_set, has_input_format, index + 2 == len(token)
    return has_no_set, has_input_format, False


def _matches_bsd_date_setting_operand(value: str) -> bool:
    date_part, separator, seconds = value.partition(".")
    return (
        date_part.isdigit()
        and len(date_part) in _BSD_DATE_COMPONENT_LENGTHS
        and (
            not separator
            or (len(seconds) == _BSD_DATE_SECONDS_LENGTH and seconds.isdigit())
        )
    )


def _date_has_setting_operand(args: list[str]) -> bool:
    """Detect BSD date's positional clock-setting forms.

    GNU date has no valid non-option operand other than an output format, so
    conservatively applying this grammar on every POSIX platform only turns
    otherwise-invalid GNU invocations into denials.
    """
    has_no_set = False
    has_input_format = False
    positional: list[str] = []
    skip_next = False
    options_ended = False
    long_value_options = frozenset({"--date", "--file", "--reference", "--rfc-3339"})

    for token in args:
        if skip_next:
            skip_next = False
            continue
        if options_ended:
            if not token.startswith("+"):
                positional.append(token)
            continue
        if token == "--":
            options_ended = True
            continue
        if token.startswith("+"):
            continue
        if token.startswith("--"):
            option, separator, _value = token.partition("=")
            if not separator and option in long_value_options:
                skip_next = True
            continue
        if token.startswith("-") and token != "-":
            token_no_set, token_input_format, skip_next = _date_short_option_state(
                token
            )
            has_no_set = has_no_set or token_no_set
            has_input_format = has_input_format or token_input_format
            continue
        positional.append(token)

    if has_no_set or not positional:
        return False
    if has_input_format:
        return True

    # BSD's positional setter uses [[[[[cc]yy]mm]dd]HH]MM[.ss]. Avoid denying
    # arbitrary invalid GNU operands merely because they are positional.
    return any(_matches_bsd_date_setting_operand(value) for value in positional)


def _standard_policy(command: str, args: list[str]) -> ShellCommandPolicy:
    table = _TABLES[command]
    parsed = _scan_options(args, table)
    approval_long: tuple[str, ...] = ()
    approval_short = frozenset()
    path_options: tuple[str, ...] = ()
    inspect = False
    if command == "sort":
        approval_long = (
            "--compress-program",
            "--files0-from",
            "--output",
            "--temporary-directory",
        )
        approval_short = frozenset("oT")
        path_options = ("--random-source",)
    elif command == "grep":
        path_options = ("--exclude-from", "--file", "-f")
    elif command == "file":
        approval_long = (
            "--compile",
            "--files-from",
            "--uncompress",
            "--uncompress-noreport",
        )
        approval_short = frozenset("CfzZ")
        path_options = ("--files-from", "-f", "--magic-file", "-m")
    elif command == "du":
        # --files0-from reads a list of paths from file content, which static
        # path analysis cannot see.
        approval_long = ("--files0-from",)
        path_options = ("--files0-from", "--exclude-from", "-X")
    elif command == "wc":
        approval_long = ("--files0-from",)
        path_options = ("--files0-from",)
    elif command == "date":
        approval_long = ("--set",)
        approval_short = frozenset("s")
        path_options = ("--file", "-f", "--reference", "-r")
    elif command == "diff":
        path_options = ("--exclude-from", "-X", "--from-file", "--to-file")
    elif command == "tree":
        approval_long = ("--output",)
        approval_short = frozenset("o")
        path_options = ("--gitfile",)
        inspect = True
    path_values = parsed.values_for(*path_options)
    if command == "file":
        path_values = tuple(
            path
            for option, value in parsed.values
            if option in path_options
            for path in value.split(":")
        )
    if command == "tree":
        # tree resumes short-cluster scanning after a value-taking short, so
        # any "o" in a cluster reaches the output-file option.
        requires_approval = _has_long(parsed, *approval_long) or _has_short_in_cluster(
            parsed, approval_short
        )
    else:
        requires_approval = _has_long(parsed, *approval_long) or _has_short(
            parsed, approval_short, table
        )
    if command == "date":
        requires_approval = requires_approval or _date_has_setting_operand(args)
    return ShellCommandPolicy(
        requires_approval=requires_approval,
        inspect_positional_paths=inspect,
        option_path_values=path_values,
        positional_values=parsed.positionals,
    )


_CHECKSUM_TABLE = _OptionTable(frozenset({"--check"}))


def _checksum_policy(args: list[str]) -> ShellCommandPolicy:
    """md5sum/sha1sum/sha256sum/shasum --check reads paths from file content."""
    parsed = _scan_options(args, _CHECKSUM_TABLE)
    return ShellCommandPolicy(
        requires_approval=_has_long(parsed, "--check")
        or _has_short(parsed, frozenset("c"), _CHECKSUM_TABLE)
    )


def _uniq_policy(args: list[str]) -> ShellCommandPolicy:
    # A second positional operand is uniq's output file, the positional
    # counterpart of sort --output.
    operand_count_with_output = 2
    positional_count = 0
    skip_next = False
    options_ended = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if options_ended:
            positional_count += 1
            continue
        if token == "--":
            options_ended = True
            continue
        if token.startswith("--"):
            option, separator, _value = token.partition("=")
            if not separator and option in {
                "--check-chars",
                "--skip-chars",
                "--skip-fields",
            }:
                skip_next = True
            continue
        is_legacy_plus_option = token.startswith("+") and token[1:].isdigit()
        if (token.startswith("-") and token != "-") or is_legacy_plus_option:
            for index, option in enumerate(token[1:]):
                if option in {"f", "s", "w"}:
                    if index + 2 == len(token):
                        skip_next = True
                    break
            continue
        positional_count += 1
    return ShellCommandPolicy(
        requires_approval=positional_count >= operand_count_with_output
    )


_LESS_LONG_VALUE_OPTIONS = frozenset({
    "--autosave",
    "--buffers",
    "--cmd",
    "--color",
    "--emouse",
    "--end-prompt",
    "--jump-target",
    "--lesskey-content",
    "--lesskey-context",
    "--lesskey-file",
    "--lesskey-src",
    "--log-file",
    "--max-back-scroll",
    "--max-forw-scroll",
    "--pattern",
    "--prompt",
    "--quotes",
    "--rscroll",
    "--shift",
    "--tabs",
    "--tag",
    "--tag-file",
    "--window",
})
_LESS_APPROVAL_LONG_OPTIONS = frozenset({
    "--autosave",
    "--cmd",
    "--lesskey-content",
    "--lesskey-context",
    "--lesskey-file",
    "--lesskey-src",
    "--log-file",
    "--tag",
    "--tag-file",
})
_LESS_SHORT_VALUE_OPTIONS = frozenset('"#DbhjkopPtTxyz')
_LESS_SHORT_NUMERIC_VALUE_OPTIONS = frozenset("#bhjxyz")
_LESS_LONG_NUMERIC_VALUE_OPTIONS = frozenset({
    "--buffers",
    "--header",
    "--jump-target",
    "--line-num-width",
    "--match-shift",
    "--max-back-scroll",
    "--max-forw-scroll",
    "--modelines",
    "--shift",
    "--status-col-width",
    "--tabs",
    "--wheel-lines",
    "--window",
})
_LESS_LONG_STRING_VALUE_OPTIONS = _LESS_LONG_VALUE_OPTIONS | frozenset({
    "--intr",
    "--search-options",
})
_LESS_APPROVAL_SHORT_OPTIONS = frozenset({"k", "o", "O", "t", "T"})


def _matches_long_option(token: str, option: str) -> bool:
    return _match_long(token, frozenset({option})) is not None


def _normalize_less_long_option(token: str) -> str:
    # less accepts ``--+name`` as its long reset/default spelling. String-valued
    # option handlers can still consume their supplied configuration value.
    if token.startswith("--+"):
        token = "--" + token[3:]
    option, separator, value = token.partition("=")
    return option.lower() + separator + value


def _less_startup_requires_approval(token: str) -> bool:
    command = token[2:] if token.startswith("++") else token[1:]
    if command in {"g", "G"} or (
        command and all("0" <= character <= "9" for character in command)
    ):
        return False
    if not (command.startswith(("/", "?")) and command.isprintable()):
        return True
    return _less_string_value_requires_approval(command[1:])


def _less_numeric_value_requires_approval(value: str) -> bool:
    """Check options that less resumes parsing after a numeric value."""
    index = 0
    if (
        len(value) > 1
        and value[0] == "-"
        and ("0" <= value[1] <= "9" or value[1] == ".")
    ):
        index = 1
    while index < len(value) and (
        "0" <= value[index] <= "9" or value[index] in {".", ","}
    ):
        index += 1
    suffix = value[index:]
    if not suffix:
        return False
    if not suffix.startswith(("-", "+")):
        suffix = "-" + suffix
    return _less_policy([suffix]).requires_approval


def _less_string_value_requires_approval(value: str) -> bool:
    """Check the option suffix after less's attached-string ``$`` terminator."""
    _, separator, suffix = value.partition("$")
    if not separator or not suffix:
        return False
    if not suffix.startswith(("-", "+")):
        suffix = "-" + suffix
    return _less_policy([suffix]).requires_approval


def _less_long_option_requires_approval(token: str) -> bool:
    if any(
        _matches_long_option(token, option) for option in _LESS_APPROVAL_LONG_OPTIONS
    ):
        return True
    option, separator, attached_value = token.partition("=")
    if not separator:
        return False
    if any(
        _matches_long_option(option, numeric_option)
        for numeric_option in _LESS_LONG_NUMERIC_VALUE_OPTIONS
    ):
        return _less_numeric_value_requires_approval(attached_value)
    if any(
        _matches_long_option(option, string_option)
        for string_option in _LESS_LONG_STRING_VALUE_OPTIONS
    ):
        return _less_string_value_requires_approval(attached_value)
    # New less releases may add string-valued options. Their attached values
    # share the ``$`` terminator grammar, so fail closed on a risky resumed suffix
    # even before the option name is added to the version-specific table.
    if "$" in attached_value:
        return _less_string_value_requires_approval(attached_value)
    return False


def _less_short_option_action(token: str) -> tuple[bool, bool]:
    """Return whether this token requires approval and consumes the next token."""
    index = 1
    while index < len(token):
        remainder = token[index:]
        if remainder.startswith("--"):
            normalized = _normalize_less_long_option(remainder)
            return _less_long_option_requires_approval(normalized), False

        option = token[index]
        if option == "$":
            index += 1
            continue
        if option == "+":
            return _less_startup_requires_approval(remainder), False
        if "0" <= option <= "9":
            return _less_numeric_value_requires_approval(remainder), False
        if option in _LESS_APPROVAL_SHORT_OPTIONS:
            return True, False
        if option not in _LESS_SHORT_VALUE_OPTIONS:
            index += 1
            continue
        attached_value = token[index + 1 :]
        if option in _LESS_SHORT_NUMERIC_VALUE_OPTIONS:
            requires_approval = bool(
                attached_value and _less_numeric_value_requires_approval(attached_value)
            )
        else:
            requires_approval = bool(
                attached_value and _less_string_value_requires_approval(attached_value)
            )
        return requires_approval, not attached_value
    return False, False


def _less_policy(args: list[str]) -> ShellCommandPolicy:
    """less's option grammar can start shell commands and lesskey programs.

    Ported from upstream v2.25.7: startup commands (``+!cmd``), lesskey files,
    tag files, and option values whose ``$`` terminator resumes option parsing
    are all code-execution or config-loading vectors that token-shape analysis
    cannot model.
    """
    if any(
        not argument.isprintable() or argument.count("$") > _MAX_LESS_RESUMED_OPTIONS
        for argument in args
    ):
        # Resumed options recursively re-enter this parser; bound the work.
        return ShellCommandPolicy(requires_approval=True)

    for argument in args:
        _, separator, resumed = argument.partition("$")
        if not separator or not (resumed := resumed.lstrip()):
            continue
        if not resumed.startswith(("-", "+")):
            resumed = "-" + resumed
        if _less_policy([resumed]).requires_approval:
            return ShellCommandPolicy(requires_approval=True)

    skip_next = False
    option_tokens = [
        (token, argument == "--") for argument in args for token in argument.split()
    ]
    for token, ends_options in option_tokens:
        if token == "--" and ends_options:
            # less ends option scanning at "--" instead of feeding it to a
            # value-taking option, which then errors for lack of a value
            # (verified against less 668); a later "-o" is a filename.
            break
        if skip_next:
            skip_next = False
            continue
        if token.startswith("+") and _less_startup_requires_approval(token):
            return ShellCommandPolicy(requires_approval=True)
        if token.startswith("--"):
            normalized = _normalize_less_long_option(token)
            if _less_long_option_requires_approval(normalized):
                return ShellCommandPolicy(requires_approval=True)
            # Only exact, version-stable spellings may hide the following token.
            # An ambiguous abbreviation or an option unknown to an older less
            # release does not consume its apparent value, which can expose a
            # following -k/--lesskey-* option to less instead.
            if "=" not in normalized and normalized in _LESS_LONG_VALUE_OPTIONS:
                skip_next = True
            continue
        if not token.startswith("-"):
            continue
        requires_approval, skip_next = _less_short_option_action(token)
        if requires_approval:
            return ShellCommandPolicy(requires_approval=True)
    return ShellCommandPolicy()


_FIND_VALUE_PREDICATES = frozenset({
    "-amin",
    "-anewer",
    "-atime",
    "-cmin",
    "-cnewer",
    "-context",
    "-ctime",
    "-fstype",
    "-gid",
    "-group",
    "-ilname",
    "-iname",
    "-inum",
    "-ipath",
    "-iregex",
    "-iwholename",
    "-links",
    "-lname",
    "-maxdepth",
    "-mindepth",
    "-mmin",
    "-mtime",
    "-name",
    "-newer",
    "-path",
    "-perm",
    "-printf",
    "-regex",
    "-regextype",
    "-samefile",
    "-size",
    "-type",
    "-uid",
    "-used",
    "-user",
    "-wholename",
    "-xtype",
})
_FIND_SIDE_EFFECTS = frozenset({
    "-delete",
    "-exec",
    "-execdir",
    "-fls",
    "-fprint",
    "-fprintf",
    "-fprint0",
    "-ok",
    "-okdir",
})


def _find_policy(args: list[str]) -> ShellCommandPolicy:
    index = 0
    while index < len(args) and (
        args[index] in {"-H", "-L", "-P"} or re.fullmatch(r"-O[0-3]", args[index])
    ):
        index += 1
    if index < len(args) and args[index] == "--":
        index += 1
    while index < len(args) and not (
        args[index].startswith("-") or args[index] in {"!", "(", ")"}
    ):
        index += 1
    while index < len(args):
        token = args[index]
        if token in _FIND_SIDE_EFFECTS:
            return ShellCommandPolicy(requires_approval=True)
        if token in _FIND_VALUE_PREDICATES or re.fullmatch(
            r"-newer[aBcmt][aBcmt]", token
        ):
            index += 2
        else:
            index += 1
    return ShellCommandPolicy()


_GIT_GLOBAL_TABLE = _OptionTable(
    frozenset({
        "--config-env",
        "--git-dir",
        "--namespace",
        "--resolve-git-dir",
        "--super-prefix",
        "--work-tree",
    }),
    frozenset("Cc"),
    frozenset({"--exec-path"}),
)
_GIT_BOOLEAN_GLOBALS = frozenset({
    "--bare",
    "--glob-pathspecs",
    "--help",
    "--html-path",
    "--icase-pathspecs",
    "--info-path",
    "--literal-pathspecs",
    "--man-path",
    "--no-advice",
    "--no-optional-locks",
    "--no-pager",
    "--no-replace-objects",
    "--noglob-pathspecs",
    "--paginate",
    "--verbose",
    "--version",
})
_GIT_SUBCOMMAND_TABLE = _OptionTable(
    frozenset({
        "--anchored",
        "--color-moved-ws",
        "--contents",
        "--diff-filter",
        "--diff-algorithm",
        "--diff-merges",
        "--ignore-matching-lines",
        "--inter-hunk-context",
        "--line-prefix",
        "--max-count",
        "--output",
        "--output-indicator-context",
        "--output-indicator-new",
        "--output-indicator-old",
        "--pathspec-from-file",
        "--rotate-to",
        "--skip-to",
        "--src-prefix",
        "--dst-prefix",
        "--stat-count",
        "--stat-graph-width",
        "--stat-name-width",
        "--word-diff-regex",
    }),
    frozenset("GLSO"),
    frozenset({
        "--abbrev",
        "--color",
        "--color-moved",
        "--dirstat",
        "--find-copies",
        "--find-renames",
        "--format",
        "--pretty",
        "--stat",
        "--submodule",
        "--unified",
        "--word-diff",
    }),
    frozenset("U"),
    boolean_long=frozenset({
        "--ext-diff",
        "--help",
        "--remerge-diff",
        "--show-signature",
        "--textconv",
    }),
)


def _options_before_double_dash(args: list[str]) -> list[str]:
    return args[: args.index("--")] if "--" in args else args


def _rm_policy(args: list[str]) -> ShellCommandPolicy:
    for token in _options_before_double_dash(args):
        if (
            token.startswith("--rec")
            and "--recursive".startswith(token)
            or token.startswith("-")
            and not token.startswith("--")
            and any(flag in token[1:] for flag in "rR")
        ):
            return ShellCommandPolicy(
                requires_approval=True,
                diagnostic=Diagnostic("recursive_rm", offending_token=token),
            )
    return ShellCommandPolicy()


def _effective_clean_flags(options: tuple[str, ...]) -> tuple[bool, bool]:
    """Effective (force, dry-run) state of git clean under last-wins semantics.

    git's parse-options resolves repeated boolean options in order and
    accepts a ``--no-`` spelling of every boolean, so the final spelling
    decides: ``--dry-run --no-dry-run`` leaves the clean destructive, and
    ``--force --no-force`` leaves it inert. Presence alone would misread
    both. ``options`` holds resolved option tokens — long options by
    canonical name, short options one character each, so a ``-fdn`` cluster
    contributes ``-f``, ``-d``, and ``-n`` in order.
    """
    force = False
    dry_run = False
    for option in options:
        if option in {"--force", "-f"}:
            force = True
        elif option == "--no-force":
            force = False
        if option in {"--dry-run", "-n"}:
            dry_run = True
        elif option == "--no-dry-run":
            dry_run = False
    return force, dry_run


def _legacy_clean_flags(effective_options: list[str]) -> tuple[bool, bool]:
    """Resolve raw git clean option tokens into effective (force, dry-run).

    The runtime policy layer scans raw tokens, not the classifier's
    resolved options: long spellings map directly, and a short cluster
    contributes its ``f``/``n`` characters in order.
    """
    options: list[str] = []
    for token in effective_options:
        if token in {"--dry-run", "--no-dry-run", "--force", "--no-force"}:
            options.append(token)
        elif token.startswith("-") and not token.startswith("--"):
            options.extend(f"-{short}" for short in token[1:] if short in {"f", "n"})
    return _effective_clean_flags(tuple(options))


def _git_destructive_policy(args: list[str]) -> bool:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        token = args[index]
        if token == "--":
            index += 1
            break
        name = token.partition("=")[0]
        if name in _GIT_BOOLEAN_GLOBALS or token in {"-h", "-p", "-P", "-v"}:
            index += 1
            continue
        if name in _GIT_GLOBAL_TABLE.long_values or token[:2] in {"-C", "-c"}:
            has_attached_value = (
                "=" in token if token.startswith("--") else len(token) > len(token[:2])
            )
            index += 1 if has_attached_value else 2
            continue
        if name in _GIT_GLOBAL_TABLE.optional_long_values:
            index += 1
            continue
        return False
    if index >= len(args):
        return False
    subcommand = args[index]
    options = _options_before_double_dash(args[index + 1 :])
    if subcommand == "reset":
        return "--hard" in options
    if subcommand == "clean":
        effective_options: list[str] = []
        option_index = 0
        subcommand_args = args[index + 1 :]
        while option_index < len(subcommand_args):
            token = subcommand_args[option_index]
            if token == "--":
                break
            effective_options.append(token)
            if token in {"-e", "--exclude"}:
                option_index += 2
            else:
                option_index += 1
        # git resolves repeated boolean options in order and accepts --no-
        # spellings of them, so the last dry-run/force spelling decides:
        # --dry-run --no-dry-run is a destructive clean, --force --no-force
        # is not. Presence alone would misread both.
        force, dry_run = _legacy_clean_flags(effective_options)
        return force and not dry_run
    return False


def _recursive_option(args: list[str]) -> bool:
    return any(
        token == "--recursive"
        or (token.startswith("-") and not token.startswith("--") and "R" in token[1:])
        for token in _options_before_double_dash(args)
    )


def _chmod_policy(args: list[str]) -> ShellCommandPolicy:
    if not _recursive_option(args):
        return ShellCommandPolicy()
    options_ended = False
    mode = ""
    for token in args:
        if token == "--" and not options_ended:
            options_ended = True
            continue
        if not options_ended and token.startswith("-"):
            continue
        mode = token
        break
    # GNU chmod accepts any number of leading-zero-padded octal digits and
    # rejects only values above 07777 ("mode too large"), so the last digit is
    # always the "others" permission class ("00007" means mode 0007).
    numeric_permissive = (
        bool(re.fullmatch(r"[0-7]+", mode))
        and int(mode, 8) <= _CHMOD_MAX_NUMERIC_MODE
        and int(mode[-1]) & 2 != 0
    )
    symbolic_permissive = any(
        (not who or "a" in who or "o" in who) and "w" in permissions
        for who, operator, permissions in re.findall(
            r"(?:^|,)([ugoa]*)([+=-])([^,]*)", mode
        )
        if operator in {"+", "="}
    )
    return ShellCommandPolicy(
        requires_approval=numeric_permissive or symbolic_permissive
    )


def _chown_policy(args: list[str]) -> ShellCommandPolicy:
    return ShellCommandPolicy(requires_approval=_recursive_option(args))


_GIT_READER_SUBCOMMANDS = frozenset({
    "blame",
    "diff",
    "log",
    "show",
    "status",
    "whatchanged",
    "branch",
    "tag",
    "grep",
    "reflog",
    "stash",
    "shortlog",
    "rev-parse",
    "ls-files",
    "ls-tree",
    "describe",
    "show-ref",
    "for-each-ref",
    "rev-list",
    "cat-file",
    "count-objects",
    "diff-files",
    "diff-index",
    "diff-tree",
    "merge-base",
    "name-rev",
    "verify-pack",
    "check-attr",
    "check-ignore",
    "check-ref-format",
    "check-mailmap",
    "column",
    "get-tar-commit-id",
    "var",
})

_GIT_PAGING_SUBCOMMANDS = _GIT_READER_SUBCOMMANDS

# Explicitly recognized non-reader builtins retain their existing policy.
# Other subcommands remain gated, whether builtins or potential external helpers.
_GIT_NONREADER_SUBCOMMANDS = frozenset({
    "add",
    "am",
    "apply",
    "bisect",
    "checkout",
    "cherry-pick",
    "clean",
    "clone",
    "commit",
    "config",
    "fetch",
    "init",
    "merge",
    "mv",
    "pull",
    "push",
    "rebase",
    "remote",
    "reset",
    "restore",
    "revert",
    "rm",
    "switch",
    "worktree",
    "submodule",
    "help",
    "version",
    "archive",
    "notes",
})


# Known builtins without modeled operand/side-effect rules remain gated, but
# are not aliases. Do not treat these as readers (e.g. fsck --lost-found writes).
_GIT_UNMODELED_BUILTINS = frozenset({
    "bundle",
    "fsck",
    "gc",
    "maintenance",
    "reflog",
    "prune",
    "repack",
    "pack-objects",
    "index-pack",
    "unpack-objects",
    "update-index",
    "update-ref",
    "symbolic-ref",
    "hash-object",
    "write-tree",
    "read-tree",
    "commit-tree",
    "mktree",
    "merge-base",
    "merge-file",
    "merge-index",
    "merge-tree",
    "name-rev",
    "describe",
    "fast-export",
    "fast-import",
    "filter-branch",
    "format-patch",
    "send-email",
    "request-pull",
    "shortlog",
    "sparse-checkout",
    "multi-pack-index",
    "pack-refs",
    "verify-pack",
    "verify-commit",
    "verify-tag",
    "count-objects",
    "rerere",
    "replace",
    "credential",
    "credential-cache",
    "credential-store",
    "daemon",
    "http-backend",
    "upload-pack",
    "receive-pack",
    "upload-archive",
    "for-each-repo",
    "bugreport",
    "diagnose",
    "scalar",
    "whatchanged",
})


def _git_global_options(args: list[str]) -> tuple[int, list[str]] | None:
    """Walk git's global options; fail closed on unknown or config overrides.

    Returns the index of the first non-global token together with the global
    path values (-C/--git-dir/--work-tree), or None when the option sequence
    itself must be denied.
    """
    index = 0
    paths: list[str] = []
    while index < len(args) and args[index].startswith("-"):
        token = args[index]
        name = token.partition("=")[0]
        if token in {"-h", "-p", "-P", "-v", "--"} or name in _GIT_BOOLEAN_GLOBALS:
            index += 1
            continue
        parsed = _scan_options(args[index:], _GIT_GLOBAL_TABLE)
        known = (
            _match_long(
                token,
                _GIT_GLOBAL_TABLE.long_values | _GIT_GLOBAL_TABLE.optional_long_values,
            )
            if token.startswith("--")
            else token[:2] in {"-C", "-c"}
        )
        if not known:
            return None
        if name in {"--config-env"} or token.startswith("-c"):
            return None
        value_pairs = parsed.values[:1]
        if value_pairs and value_pairs[0][0] in {"-C", "--git-dir", "--work-tree"}:
            paths.append(value_pairs[0][1])
        consumed_value = bool(value_pairs) and (
            (
                token.startswith("--")
                and "=" not in token
                and not token.startswith("--exec-path")
            )
            or token in {"-C", "-c"}
        )
        index += 2 if consumed_value else 1
    return index, paths


def canonical_git_command(tokens: list[str]) -> list[str] | None:
    """Canonical git command form with recognized global options elided.

    The canonical form keeps the executable and the full subcommand suffix:
    ``git -C . stash drop`` becomes ``git stash drop``. Returns None when the
    executable is not git or the leading global options are unrecognized; the
    policy layer denies those spellings, so declining to normalize is safe.
    """
    if not tokens or Path(tokens[0]).name != "git":
        return None
    globals_result = _git_global_options(tokens[1:])
    if globals_result is None:
        return None
    index, _paths = globals_result
    return [tokens[0], *tokens[1 + index :]]


# git archive's --output/-o operand names the written archive: a raw
# file-write operand of an otherwise exempted subcommand, governed like any
# other write operand.
_GIT_ARCHIVE_OUTPUT_TABLE = _OptionTable(
    long_values=frozenset({"--output"}), short_values=frozenset({"o"})
)


def _git_archive_output_values(args: list[str]) -> tuple[str, ...]:
    """The --output/-o targets of one git archive call, in every spelling."""
    parsed = _scan_options(args, _GIT_ARCHIVE_OUTPUT_TABLE)
    return parsed.values_for("--output", "-o")


# git apply's --directory relocates the patch's write root: every patched
# path (after -p stripping) lands beneath it, so the value is a raw write
# operand of an otherwise exempted subcommand, governed like any other.
_GIT_APPLY_TABLE = _OptionTable(
    long_values=frozenset({"--directory"}), boolean_long=frozenset({"--unsafe-paths"})
)
# --unsafe-paths disables the path refusal git itself applies outside the
# work tree — including paths inside .git metadata, which git otherwise
# rejects as "invalid path" — and no legitimate agent workflow needs it.
_GIT_APPLY_UNSAFE_PATHS_REASON = (
    "git apply --unsafe-paths disables the path refusal git itself "
    "applies outside the work tree, including paths inside protected "
    "Git metadata: no legitimate agent workflow needs it, and no "
    "permission setting can authorize it"
)

# git init's positional operand is the directory the call creates, and
# --separate-git-dir names the git dir it creates; both are write operands.
# --shared takes an optional attached value only, so it never consumes the
# positional directory.
_GIT_INIT_TABLE = _OptionTable(
    long_values=frozenset({
        "--initial-branch",
        "--object-format",
        "--ref-format",
        "--separate-git-dir",
        "--template",
    }),
    short_values=frozenset({"b"}),
    optional_long_values=frozenset({"--shared"}),
    boolean_long=frozenset({"--bare", "--quiet"}),
)

# git worktree add's first positional operand is the new work tree's path;
# -b/-B and --reason take values, every other option is a boolean (--orphan
# included: it selects an unborn branch and takes no value).
_GIT_WORKTREE_ADD_TABLE = _OptionTable(
    long_values=frozenset({"--reason"}),
    short_values=frozenset({"b", "B"}),
    boolean_long=frozenset({
        "--checkout",
        "--detach",
        "--force",
        "--guess-remote",
        "--lock",
        "--orphan",
        "--quiet",
        "--relative-paths",
        "--track",
    }),
)

# git worktree move's grammar is `git worktree move [-f] <worktree>
# <new-path>`: -f/--force is the only option, and the SECOND positional
# names the directory the call relocates the work tree to.
_GIT_WORKTREE_MOVE_TABLE = _OptionTable(boolean_long=frozenset({"--force"}))

# git submodule add's grammar is `git submodule add [options] <repository>
# [<path>]`: the SECOND positional names the directory the call creates
# the submodule work tree at, and --separate-git-dir names the git dir it
# creates for it (the init twin). -b/--branch, --name, --depth,
# --reference, and --separate-git-dir take values; every other option is a
# boolean.
_GIT_SUBMODULE_ADD_TABLE = _OptionTable(
    long_values=frozenset({
        "--branch",
        "--depth",
        "--name",
        "--reference",
        "--separate-git-dir",
    }),
    short_values=frozenset({"b"}),
    boolean_long=frozenset({"--dissociate", "--force", "--progress", "--quiet"}),
)


def _git_destination_operands(  # noqa: PLR0911
    subcommand: str, args: list[str], index: int
) -> tuple[list[str], str | None]:
    """The write operands a git subcommand creates or relocates its writes to.

    ``git apply --directory`` names the patch's write root (every patched
    path lands beneath it after ``-p`` stripping), ``git init``'s positional
    operand and ``--separate-git-dir`` name directories the call creates,
    ``git worktree add``'s first positional names the new work tree,
    ``git worktree move``'s second positional names the directory the call
    relocates the work tree to, and ``git submodule add``'s second
    positional (plus its ``--separate-git-dir`` git dir) names the
    directories the call creates. Returns the write operands and, for
    ``git apply --unsafe-paths``, the hard-guard reason: the flag disables
    the path refusal git itself applies outside the work tree, including
    inside protected metadata.
    """
    rest = args[index + 1 :]
    if subcommand == "apply":
        apply_parsed = _scan_options(rest, _GIT_APPLY_TABLE)
        write_paths = list(apply_parsed.values_for("--directory"))
        if _has_long(apply_parsed, "--unsafe-paths"):
            return write_paths, _GIT_APPLY_UNSAFE_PATHS_REASON
        return write_paths, None
    if subcommand == "init":
        init_parsed = _scan_options(rest, _GIT_INIT_TABLE)
        write_paths = [init_parsed.positionals[0]] if init_parsed.positionals else []
        write_paths.extend(init_parsed.values_for("--separate-git-dir"))
        return write_paths, None
    if subcommand == "worktree" and args[index + 1 : index + 2] == ["add"]:
        add_parsed = _scan_options(args[index + 2 :], _GIT_WORKTREE_ADD_TABLE)
        write_paths = [add_parsed.positionals[0]] if add_parsed.positionals else []
        return write_paths, None
    if subcommand == "worktree" and args[index + 1 : index + 2] == ["move"]:
        move_parsed = _scan_options(args[index + 2 :], _GIT_WORKTREE_MOVE_TABLE)
        # The first positional selects the work tree to move; the second
        # names its new location.
        write_paths = (
            [move_parsed.positionals[1]] if len(move_parsed.positionals) > 1 else []
        )
        return write_paths, None
    if subcommand == "submodule" and args[index + 1 : index + 2] == ["add"]:
        add_parsed = _scan_options(args[index + 2 :], _GIT_SUBMODULE_ADD_TABLE)
        # The first positional is the source repository (a read); the
        # second names the destination the call creates.
        write_paths = (
            [add_parsed.positionals[1]] if len(add_parsed.positionals) > 1 else []
        )
        write_paths.extend(add_parsed.values_for("--separate-git-dir"))
        return write_paths, None
    return [], None


def _git_policy(args: list[str]) -> ShellCommandPolicy:
    if _git_destructive_policy(args):
        return ShellCommandPolicy(requires_approval=True)
    globals_result = _git_global_options(args)
    if globals_result is None:
        return ShellCommandPolicy(requires_approval=True)
    index, paths = globals_result
    if index >= len(args):
        return ShellCommandPolicy(option_path_values=tuple(paths))
    subcommand = args[index]
    if subcommand not in _GIT_READER_SUBCOMMANDS:
        option_paths = list(paths)
        write_paths: list[str] = []
        hard_guard_reason: str | None = None
        if subcommand == "archive":
            # git archive --output/-o writes the named file directly.
            write_paths.extend(_git_archive_output_values(args[index + 1 :]))
        elif subcommand == "config":
            # A writing git config --file/-f names the written file, so it
            # is a write-classified operand; a read names a file git only
            # reads, which stays authority-scoped like a selector. An
            # unscannable form fails closed as a write.
            call = _scan_git_config_call(args[index + 1 :])
            file_values = [
                value for option, value in call.values if option in {"--file", "-f"}
            ]
            if call.error is not None or call.mutates or call.edits:
                write_paths.extend(file_values)
            else:
                option_paths.extend(file_values)
        elif (
            subcommand in {"apply", "init"}
            or (
                subcommand == "worktree"
                and args[index + 1 : index + 2] in (["add"], ["move"])
            )
            or (subcommand == "submodule" and args[index + 1 : index + 2] == ["add"])
        ):
            # Destinations the call creates or relocates its writes to
            # (apply --directory, init/worktree-add/move/submodule-add
            # destinations); the helper also reports the --unsafe-paths
            # hard guard.
            write_paths, hard_guard_reason = _git_destination_operands(
                subcommand, args, index
            )
        denial_reason = (
            f"unmodeled git builtin {subcommand!r}; denied conservatively; permission settings cannot override this guard"
            if subcommand in _GIT_UNMODELED_BUILTINS
            else f"unrecognized git subcommand {subcommand!r}; aliases and external helpers cannot be inspected"
        )
        if hard_guard_reason is not None:
            denial_reason = hard_guard_reason
        return ShellCommandPolicy(
            requires_approval=(
                subcommand not in _GIT_NONREADER_SUBCOMMANDS
                or hard_guard_reason is not None
            ),
            denial_reason=denial_reason,
            option_path_values=tuple(option_paths),
            write_path_values=tuple(write_paths),
        )
    parsed = _scan_options(args[index + 1 :], _GIT_SUBCOMMAND_TABLE)
    # --remerge-diff and --diff-merges=remerge re-run repository-configured
    # merge drivers; --show-signature runs the configured gpg program.
    diff_merge_values = {
        value.casefold() for value in parsed.values_for("--diff-merges")
    }
    requires_approval = (
        _has_long(parsed, "--output")
        or any(
            token
            in {
                "--ext-diff",
                "--help",
                "--remerge-diff",
                "--show-signature",
                "--textconv",
            }
            for token in parsed.options
        )
        or bool(diff_merge_values & {"r", "remerge"})
    )
    option_paths = list(paths)
    option_paths.extend(parsed.values_for("--contents", "--pathspec-from-file"))
    option_paths.extend(parsed.values_for("-O"))
    return ShellCommandPolicy(
        requires_approval=requires_approval,
        inspect_positional_paths=subcommand == "diff"
        and "--no-index" in parsed.options,
        # Plain readers stay permitted for ordinary repositories; the Bash
        # resolver separately inspects repository-owned git config for
        # executable helpers and denies the reader when one is active.
        inspect_git_repository=True,
        option_path_values=tuple(option_paths),
        positional_values=parsed.positionals,
    )


# --- Dangerous-git classifier (dangerous-command permissions) ---
#
# The pure, typed classifier behind the dangerous-command permission flow:
# one whole expanded call is routine, hard-denied, or grantable. It never
# mutates grants, persists nothing, and executes nothing; the runtime
# enforcement above is unchanged. Ambiguous or unmodeled forms fail closed:
# never grantable, and never an ordinary git.push.

_GIT_PUSH_TABLE = _OptionTable(
    frozenset({"--push-option", "--repo"}),
    frozenset("o"),
    frozenset({"--force-with-lease", "--signed"}),
    boolean_long=frozenset({
        "--all",
        "--atomic",
        "--delete",
        "--dry-run",
        "--follow-tags",
        "--force",
        "--force-if-includes",
        "--ipv4",
        "--ipv6",
        "--mirror",
        "--no-atomic",
        "--no-dry-run",
        "--no-force",
        "--no-progress",
        "--no-quiet",
        "--no-set-upstream",
        "--no-thin",
        "--no-verbose",
        "--no-verify",
        "--porcelain",
        "--progress",
        "--prune",
        "--quiet",
        "--set-upstream",
        "--tags",
        "--thin",
        "--verbose",
        "--verify",
    }),
)
_GIT_PUSH_BOOLEAN_SHORTS = frozenset("46fnquv")
_GIT_PUSH_FORCE_OPTIONS = frozenset({
    "--force",
    "--force-if-includes",
    "--force-with-lease",
    "-f",
})
# --receive-pack/--exec select the program git runs for a local remote, so a
# push carrying either is never grantable.
_GIT_PUSH_HARD_GUARD_LONGS = frozenset({"--exec", "--receive-pack"})
# A URL scheme prefix (https://, ssh://, git://, ...) names a destination
# directly, exactly like a path operand does.
_PUSH_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def _push_destination_is_explicit(destination: str) -> bool:
    """Whether a push destination operand is a path or URL, not a remote name.

    A bare name (``origin``, ``upstream``) resolves through the repository's
    configured remotes, which the grant covers; a path or URL names its
    destination directly, outside the source repository the grant scopes.
    A remote name can never contain ``:`` (git's refname validation rejects
    it), so an scp-like ``host:repo.git`` operand is a URL destination, not
    a remote name.
    """
    if "/" in destination or ":" in destination or destination.startswith((".", "~")):
        return True
    return _PUSH_URL_SCHEME.match(destination) is not None


def push_destination_operands(call: ModeledGitCall) -> tuple[str, ...]:
    """The destination operands of a modeled push call.

    The first positional is always the repository, never a refspec, and
    ``--repo`` overrides it, so both name the destination a push runs
    receive-pack and hooks on.
    """
    destinations = list(call.operands[:1])
    destinations.extend(
        value for option, value in call.option_values if option == "--repo"
    )
    return tuple(destinations)


def _push_destination_fail_closed_reason(call: ModeledGitCall) -> str | None:
    """Why a push destination is not a grantable remote name, or None.

    The destination operands name the destination a push runs receive-pack
    and hooks on, so an explicit path or URL destination is never
    grantable. A bare name is only name-shaped: the classifier stays pure
    and cannot inspect the repository, so the resolver verifies the name
    against the repository's configured remotes — git resolves any
    destination that is not a configured remote name as a URL.
    """
    for destination in push_destination_operands(call):
        if _push_destination_is_explicit(destination):
            return (
                f"destination {destination!r} is a path or URL, not a remote "
                "name: a push to it runs the destination's receive-pack and "
                "hooks, outside the repository the grant covers"
            )
    return None


_GIT_SWITCH_TABLE = _OptionTable(
    frozenset({"--create", "--force-create", "--orphan"}),
    frozenset("cC"),
    frozenset({"--recurse-submodules"}),
    boolean_long=frozenset({
        "--detach",
        "--discard-changes",
        "--force",
        "--guess",
        "--merge",
        "--no-create",
        "--no-guess",
        "--no-track",
        "--quiet",
        "--track",
    }),
)
_GIT_SWITCH_BOOLEAN_SHORTS = frozenset("dfgmq")
_GIT_SWITCH_DISCARD_OPTIONS = frozenset({"--discard-changes", "--force", "-f"})

# checkout/restore are scanned leniently: every form is consequential and git
# itself rejects unmodeled options, so unknown options are recorded without
# failing the classification. Only their value-taking options matter, to keep
# consumed operands out of the positional operands.
_GIT_CHECKOUT_TABLE = _OptionTable(
    frozenset({"--conflict", "--orphan", "--pathspec-from-file"}),
    frozenset("bB"),
    frozenset({"--recurse-submodules"}),
)
_GIT_RESTORE_TABLE = _OptionTable(
    frozenset({"--pathspec-from-file", "--source"}), frozenset("s")
)
_GIT_WORKTREE_TABLES = {"checkout": _GIT_CHECKOUT_TABLE, "restore": _GIT_RESTORE_TABLE}

# stash's value-taking options must consume before the sub-subcommand is
# read: `git stash -m drop` is a stash push with message "drop", not a drop.
_GIT_STASH_TABLE = _OptionTable(
    frozenset({"--message", "--pathspec-from-file"}), frozenset("m")
)
_GIT_REFLOG_TABLE = _OptionTable()
_STASH_REFLOG_OPERATION_IDS = {
    ("stash", "drop"): "git.stash.drop",
    ("stash", "clear"): "git.stash.clear",
    ("reflog", "expire"): "git.reflog.expire",
    ("reflog", "delete"): "git.reflog.delete",
}

# reset/clean are scanned strictly: their hard-guard flags accept unambiguous
# abbreviations (`git reset --ha` really hard-resets), so an unrecognized
# option could hide one and must fail closed.
_GIT_RESET_TABLE = _OptionTable(
    frozenset({"--pathspec-from-file"}),
    boolean_long=frozenset({
        "--hard",
        "--keep",
        "--merge",
        "--mixed",
        "--quiet",
        "--soft",
    }),
)
_GIT_RESET_BOOLEAN_SHORTS = frozenset("q")
_GIT_CLEAN_TABLE = _OptionTable(
    frozenset({"--exclude"}),
    frozenset("e"),
    boolean_long=frozenset({
        "--dry-run",
        "--force",
        "--interactive",
        "--no-dry-run",
        "--no-force",
        "--no-interactive",
        "--no-quiet",
        "--quiet",
    }),
)
_GIT_CLEAN_BOOLEAN_SHORTS = frozenset("dfnqxXi")

# git config is a permitted routine command, so the classifier scans it
# leniently: git itself rejects unmodeled options, and the key check below
# looks at every operand, so a misread option consumption can only
# over-block, never miss a write. Value options consume their operand
# exactly as git's parser does (-f/--file, -t/--type, --blob, --default,
# --comment); every other documented option is a boolean.
_GIT_CONFIG_TABLE = _OptionTable(
    frozenset({"--blob", "--comment", "--default", "--file", "--type"}),
    frozenset({"f", "t"}),
    boolean_long=frozenset({
        "--add",
        "--all",
        "--edit",
        "--fixed-value",
        "--get",
        "--get-all",
        "--get-color",
        "--get-colorbool",
        "--get-regexp",
        "--get-urlmatch",
        "--global",
        "--includes",
        "--list",
        "--local",
        "--name-only",
        "--null",
        "--replace-all",
        "--show-names",
        "--show-origin",
        "--show-scope",
        "--system",
        "--unset",
        "--unset-all",
        "--worktree",
    }),
)
_GIT_CONFIG_BOOLEAN_SHORTS = frozenset("elz")
# Read actions report configuration; they never write it.
_GIT_CONFIG_READ_OPTIONS = frozenset({
    "--get",
    "--get-all",
    "--get-color",
    "--get-colorbool",
    "--get-regexp",
    "--get-urlmatch",
    "--list",
    "-l",
})
# Write actions record a value.
_GIT_CONFIG_WRITE_LONGS = frozenset({"--add", "--replace-all"})
# Removal actions only ever remove configuration.
_GIT_CONFIG_UNSET_LONGS = frozenset({"--unset", "--unset-all"})
# Section operations act on section names, not keys.
_GIT_CONFIG_SECTION_LONGS = frozenset({"--rename-section", "--remove-section"})
# The newer subcommand spellings (git config set/get/list/...): the first
# operand is the action word, not a key.
_GIT_CONFIG_SUBCOMMANDS = frozenset({
    "edit",
    "get",
    "list",
    "remove-section",
    "rename-section",
    "set",
    "unset",
})
_GIT_CONFIG_READ_SUBCOMMANDS = frozenset({"get", "list"})
_GIT_CONFIG_MUTATION_SUBCOMMANDS = frozenset({
    "remove-section",
    "rename-section",
    "set",
    "unset",
})

# git remote is scanned leniently: only the sub-subcommand decides the
# classification, and git itself rejects unmodeled options. The global
# grammar is the single boolean -v/--verbose; every other option belongs to
# the sub-subcommand, whose own operands never matter once the form is a
# hard guard.
_GIT_REMOTE_TABLE = _OptionTable(boolean_long=frozenset({"--verbose"}))
_GIT_REMOTE_BOOLEAN_SHORTS = frozenset("v")
# git remote mutations redefine where a granted push lands: add/set-url
# point a bare destination name at an arbitrary URL or path, rename
# retargets an existing remote, and remove (rm) turns a configured bare
# name back into a URL git resolves as a path.
_GIT_REMOTE_MUTATION_SUBCOMMANDS = frozenset({
    "add",
    "set-url",
    "rename",
    "remove",
    "rm",
})

_ROUTINE = DangerousCommandClassification(kind="routine")


@dataclass(frozen=True)
class _OptionTokenScan:
    """Scan result for one classified option token."""

    options: tuple[str, ...] = ()
    values: tuple[tuple[str, str], ...] = ()
    consumed: int = 1
    error: str | None = None


@dataclass(frozen=True)
class _StrictGitOptions:
    """Strictly scanned options of one classified git subcommand."""

    options: tuple[str, ...]
    values: tuple[tuple[str, str], ...]
    operands: tuple[str, ...]
    end_of_options: bool
    error: str | None


@dataclass(frozen=True)
class _GitGlobalScan:
    """Structured walk of git's global options for classification."""

    index: int
    relocation: GitRepositoryRelocation
    end_of_options: bool
    error: str | None


def _scan_required_long_value(
    option: str, token: str, args: list[str], index: int
) -> _OptionTokenScan:
    """Consume one required-value long option, in = or separated form."""
    _, separator, attached = token.partition("=")
    if separator:
        if not attached:
            return _OptionTokenScan(
                error=f"option {option!r} requires a non-empty value"
            )
        return _OptionTokenScan(options=(option,), values=((option, attached),))
    if index + 1 >= len(args):
        return _OptionTokenScan(error=f"option {option!r} requires a value")
    return _OptionTokenScan(
        options=(option,), values=((option, args[index + 1]),), consumed=2
    )


def _scan_long_option_token(
    token: str,
    args: list[str],
    index: int,
    table: _OptionTable,
    *,
    strict: bool,
    hard_guard_longs: frozenset[str],
) -> _OptionTokenScan:
    """Resolve one long option token against a classified subcommand's grammar."""
    name = token.partition("=")[0]
    if hard_guard_longs and _match_long(token, hard_guard_longs) is not None:
        return _OptionTokenScan(error=f"option {name!r} selects the program git runs")
    all_long = table.long_values | table.optional_long_values | table.boolean_long
    option = _match_long(token, all_long)
    if option is None:
        if strict:
            return _OptionTokenScan(error=f"unknown or ambiguous option {name!r}")
        return _OptionTokenScan(options=(name,))
    if option in table.boolean_long and "=" in token and strict:
        return _OptionTokenScan(error=f"option {option!r} takes no value")
    if option in table.long_values:
        return _scan_required_long_value(option, token, args, index)
    # Optional values attach only in = form; an empty attached value is valid
    # (e.g. --force-with-lease=). A boolean option given a value in lenient
    # mode lands here too and is recorded, not consumed.
    values = ((option, token.partition("=")[2]),) if "=" in token else ()
    return _OptionTokenScan(options=(option,), values=values)


def _scan_short_option_token(
    token: str,
    args: list[str],
    index: int,
    table: _OptionTable,
    *,
    boolean_shorts: frozenset[str],
    strict: bool,
) -> _OptionTokenScan:
    """Resolve one short-option token (a cluster) against the grammar."""
    options: list[str] = []
    values: list[tuple[str, str]] = []
    error: str | None = None
    consumed = 1
    chars = token[1:]
    for offset, short in enumerate(chars):
        name = f"-{short}"
        if short in table.short_values:
            attached = chars[offset + 1 :]
            options.append(name)
            if attached:
                values.append((name, attached))
            elif index + 1 < len(args):
                values.append((name, args[index + 1]))
                consumed = 2
            else:
                error = f"option {name!r} requires a value"
            break
        if short in table.optional_short_values:
            attached = chars[offset + 1 :]
            options.append(name)
            if attached:
                values.append((name, attached))
            break
        if short in boolean_shorts:
            options.append(name)
            continue
        if strict:
            error = f"unknown or ambiguous option {name!r}"
            break
        options.append(name)
    return _OptionTokenScan(tuple(options), tuple(values), consumed, error)


def _scan_classified_git_options(
    args: list[str],
    table: _OptionTable,
    *,
    boolean_shorts: frozenset[str] = frozenset(),
    hard_guard_longs: frozenset[str] = frozenset(),
    strict: bool = True,
) -> _StrictGitOptions:
    """Scan one classified git subcommand's arguments.

    Unlike _scan_options, classification must never misread a consumed
    operand as a flag, so every option is resolved against the subcommand's
    modeled grammar. Strict mode (push, switch, reset, clean) fails closed on
    any unknown option, ambiguous abbreviation, value given to a boolean
    option, empty required value, or dangling required value. Lenient mode
    (checkout, restore, stash, reflog) records unknown options without
    consuming them: git itself rejects unmodeled options, and the consequence
    of those subcommands does not depend on their options.
    """
    options: list[str] = []
    values: list[tuple[str, str]] = []
    operands: list[str] = []
    end_of_options = False
    error: str | None = None
    index = 0
    while index < len(args) and error is None:
        token = args[index]
        if end_of_options:
            operands.append(token)
            index += 1
        elif token == "--":
            end_of_options = True
            index += 1
        elif token.startswith("--"):
            step = _scan_long_option_token(
                token,
                args,
                index,
                table,
                strict=strict,
                hard_guard_longs=hard_guard_longs,
            )
            error = step.error
            options.extend(step.options)
            values.extend(step.values)
            index += step.consumed
        elif token.startswith("-") and token != "-":
            step = _scan_short_option_token(
                token, args, index, table, boolean_shorts=boolean_shorts, strict=strict
            )
            error = step.error
            options.extend(step.options)
            values.extend(step.values)
            index += step.consumed
        else:
            operands.append(token)
            index += 1
    return _StrictGitOptions(
        tuple(options), tuple(values), tuple(operands), end_of_options, error
    )


def _scan_git_globals(args: list[str]) -> _GitGlobalScan:
    """Walk git's global options, recording repository relocation operands.

    The recognized grammar mirrors _git_global_options exactly, so the
    classifier agrees with runtime enforcement on which spellings are known.
    Unlike the runtime walker, classification also fails closed on a dangling
    or empty required value and keeps -C/--git-dir/--work-tree operands
    structured.
    """
    index = 0
    cwd_paths: list[str] = []
    git_dir: str | None = None
    work_tree: str | None = None
    end_of_options = False
    error: str | None = None
    while index < len(args) and args[index].startswith("-"):
        token = args[index]
        name = token.partition("=")[0]
        if token in {"-h", "-p", "-P", "-v"} or name in _GIT_BOOLEAN_GLOBALS:
            index += 1
            continue
        if token == "--":
            end_of_options = True
            index += 1
            continue
        parsed = _scan_options(args[index:], _GIT_GLOBAL_TABLE)
        known = (
            _match_long(
                token,
                _GIT_GLOBAL_TABLE.long_values | _GIT_GLOBAL_TABLE.optional_long_values,
            )
            if token.startswith("--")
            else token[:2] in {"-C", "-c"}
        )
        if not known:
            error = f"unknown or ambiguous global option {name!r}"
            break
        if name == "--config-env" or token.startswith("-c"):
            error = f"global option {name!r} overrides git configuration"
            break
        value_pairs = parsed.values[:1]
        if value_pairs and not value_pairs[0][1]:
            error = f"global option {value_pairs[0][0]!r} requires a non-empty value"
            break
        if not value_pairs:
            # A required-value global at the end of the invocation has no operand.
            required = (
                _match_long(token, _GIT_GLOBAL_TABLE.long_values)
                if token.startswith("--")
                else token[:2] == "-C"
            )
            if required is not None:
                error = f"global option {name!r} requires a value"
                break
        elif value_pairs[0][0] == "-C":
            cwd_paths.append(value_pairs[0][1])
        elif value_pairs[0][0] == "--git-dir":
            git_dir = value_pairs[0][1]
        elif value_pairs[0][0] == "--work-tree":
            work_tree = value_pairs[0][1]
        consumed_value = bool(value_pairs) and (
            (
                token.startswith("--")
                and "=" not in token
                and not token.startswith("--exec-path")
            )
            or token in {"-C", "-c"}
        )
        index += 2 if consumed_value else 1
    relocation = GitRepositoryRelocation(
        cwd_paths=tuple(cwd_paths), git_dir=git_dir, work_tree=work_tree
    )
    return _GitGlobalScan(index, relocation, end_of_options, error)


def _hard_denied(reason: str) -> DangerousCommandClassification:
    return DangerousCommandClassification(kind="hard_denied", reason=reason)


def _grantable_classification(
    operation_ids: tuple[str, ...],
    call: ModeledGitCall,
    relocation: GitRepositoryRelocation,
) -> DangerousCommandClassification:
    operations = operations_for_ids(operation_ids)
    consequence = "; ".join(operation.consequence for operation in operations)
    return DangerousCommandClassification(
        kind="grantable",
        grantable=GrantableGitCall(operations, consequence, call, relocation),
    )


def _fail_closed_on_end_of_options(
    classification: DangerousCommandClassification, *, end_of_options: bool
) -> DangerousCommandClassification:
    """An end-of-options marker is never a modeled form of a grantable call."""
    if classification.kind == "grantable" and end_of_options:
        return _hard_denied("end-of-options marker is not a modeled form")
    return classification


def _refspec_fail_closed_reason(refspec: str) -> str | None:
    """Why a push refspec is not a modeled form, or None when it is."""
    if not refspec:
        return "empty refspec"
    if any(char in refspec for char in "*?["):
        return f"glob refspec {refspec!r} is not a modeled form"
    if refspec.count(":") > 1:
        return f"refspec {refspec!r} is not a valid src:dst form"
    _, separator, dst = refspec.partition(":")
    if separator and not dst:
        return f"refspec {refspec!r} has an empty destination"
    body = refspec.removeprefix("+")
    if not body:
        return f"refspec {refspec!r} is not a valid form"
    return None


def _push_operation_ids(*, force: bool, delete: bool, mirror: bool) -> tuple[str, ...]:
    """Operation IDs a push requires; a compound push never inherits git.push."""
    ids = []
    if force:
        ids.append("git.push.force")
    if delete:
        ids.append("git.push.delete")
    if mirror:
        ids.append("git.push.mirror")
    return tuple(ids) if ids else ("git.push",)


def _classify_git_push(
    args: list[str], globals_scan: _GitGlobalScan
) -> DangerousCommandClassification:
    scan = _scan_classified_git_options(
        args,
        _GIT_PUSH_TABLE,
        boolean_shorts=_GIT_PUSH_BOOLEAN_SHORTS,
        hard_guard_longs=_GIT_PUSH_HARD_GUARD_LONGS,
    )
    if scan.error is not None:
        return _hard_denied(f"git push: {scan.error}")
    options = set(scan.options)
    force = bool(options & _GIT_PUSH_FORCE_OPTIONS)
    # --prune removes remote refs without a local counterpart, so it is a
    # deletion mechanism like --delete, never ordinary publication.
    delete = "--delete" in options or "--prune" in options
    mirror = "--mirror" in options
    call = ModeledGitCall("push", scan.options, scan.values, scan.operands)
    # A push to an explicit path or URL destination is never grantable: the
    # grant scopes the source repository, and the push runs the destination's
    # receive-pack and hooks.
    destination_reason = _push_destination_fail_closed_reason(call)
    if destination_reason is not None:
        return _hard_denied(f"git push: {destination_reason}")
    # The first positional is always the repository, never a refspec.
    for refspec in scan.operands[1:]:
        reason = _refspec_fail_closed_reason(refspec)
        if reason is not None:
            return _hard_denied(f"git push: {reason}")
        if refspec.startswith("+"):
            force = True
        if refspec.removeprefix("+").startswith(":"):
            delete = True
    result = _grantable_classification(
        _push_operation_ids(force=force, delete=delete, mirror=mirror),
        call,
        globals_scan.relocation,
    )
    return _fail_closed_on_end_of_options(
        result, end_of_options=globals_scan.end_of_options or scan.end_of_options
    )


def _classify_git_worktree_subcommand(
    subcommand: str,
    operation_id: str,
    table: _OptionTable,
    args: list[str],
    globals_scan: _GitGlobalScan,
) -> DangerousCommandClassification:
    """checkout/restore: every form is consequential, so all forms are grantable."""
    scan = _scan_classified_git_options(args, table, strict=False)
    if scan.error is not None:
        return _hard_denied(f"git {subcommand}: {scan.error}")
    call = ModeledGitCall(subcommand, scan.options, scan.values, scan.operands)
    result = _grantable_classification((operation_id,), call, globals_scan.relocation)
    return _fail_closed_on_end_of_options(
        result, end_of_options=globals_scan.end_of_options or scan.end_of_options
    )


def _classify_git_switch(
    args: list[str], globals_scan: _GitGlobalScan
) -> DangerousCommandClassification:
    scan = _scan_classified_git_options(
        args, _GIT_SWITCH_TABLE, boolean_shorts=_GIT_SWITCH_BOOLEAN_SHORTS
    )
    if scan.error is not None:
        return _hard_denied(f"git switch: {scan.error}")
    if not set(scan.options) & _GIT_SWITCH_DISCARD_OPTIONS:
        return _ROUTINE
    call = ModeledGitCall("switch", scan.options, scan.values, scan.operands)
    result = _grantable_classification(
        ("git.switch.discard",), call, globals_scan.relocation
    )
    return _fail_closed_on_end_of_options(
        result, end_of_options=globals_scan.end_of_options or scan.end_of_options
    )


def _classify_git_stash_or_reflog(
    subcommand: str, args: list[str], globals_scan: _GitGlobalScan
) -> DangerousCommandClassification:
    """stash drop/clear and reflog expire/delete are grantable; other forms are not."""
    table = _GIT_STASH_TABLE if subcommand == "stash" else _GIT_REFLOG_TABLE
    scan = _scan_classified_git_options(args, table, strict=False)
    if scan.error is not None:
        return _hard_denied(f"git {subcommand}: {scan.error}")
    operation_id = _STASH_REFLOG_OPERATION_IDS.get((
        subcommand,
        scan.operands[0] if scan.operands else "",
    ))
    if operation_id is None:
        return _ROUTINE
    call = ModeledGitCall(subcommand, scan.options, scan.values, scan.operands)
    result = _grantable_classification((operation_id,), call, globals_scan.relocation)
    return _fail_closed_on_end_of_options(
        result, end_of_options=globals_scan.end_of_options or scan.end_of_options
    )


def _classify_git_hard_guard_subcommand(
    subcommand: str, args: list[str]
) -> DangerousCommandClassification:
    """reset/clean: destructive forms are hard guards no grant can lift."""
    if subcommand == "reset":
        scan = _scan_classified_git_options(
            args, _GIT_RESET_TABLE, boolean_shorts=_GIT_RESET_BOOLEAN_SHORTS
        )
        dangerous = "--hard" in scan.options
        guard = "git reset --hard"
    else:
        scan = _scan_classified_git_options(
            args, _GIT_CLEAN_TABLE, boolean_shorts=_GIT_CLEAN_BOOLEAN_SHORTS
        )
        force, dry_run = _effective_clean_flags(scan.options)
        dangerous = force and not dry_run
        guard = "forced non-dry-run git clean"
    if scan.error is not None:
        return _hard_denied(f"git {subcommand}: {scan.error}")
    if dangerous:
        return _hard_denied(f"{guard} is a hard guard; no grant can authorize it")
    return _ROUTINE


def _classify_git_remaining(subcommand: str) -> DangerousCommandClassification:
    """Readers and unclassified nonreaders stay routine; the rest are hard guards."""
    if subcommand in _GIT_UNMODELED_BUILTINS:
        return _hard_denied(
            f"unmodeled git builtin {subcommand!r}; denied conservatively"
        )
    if (
        subcommand not in _GIT_READER_SUBCOMMANDS
        and subcommand not in _GIT_NONREADER_SUBCOMMANDS
    ):
        return _hard_denied(
            f"unrecognized git subcommand {subcommand!r}; "
            "aliases and external helpers cannot be inspected"
        )
    return _ROUTINE


# remote.<name> keys whose write redefines where a granted push lands or
# the program a push runs. url.<base>.insteadOf is guarded alongside
# pushInsteadOf because git applies it to the URL of a configured remote
# when pushing, not only when fetching.
_GIT_CONFIG_REMOTE_PUSH_EFFECTING_KEYS = frozenset({
    "mirror",
    "push",
    "url",
    "pushurl",
    "receivepack",
    "uploadpack",
    "proxy",
    "vcs",
})
# branch.<name> keys that select the destination of a bare push.
_GIT_CONFIG_BRANCH_PUSH_EFFECTING_KEYS = frozenset({"remote", "pushremote"})
# url.<base> keys that rewrite the URL a push uses.
_GIT_CONFIG_URL_PUSH_EFFECTING_KEYS = frozenset({"insteadof", "pushinsteadof"})
# include.path and includeIf.<condition>.path author a file git folds into
# configuration resolution, which can set any push-effecting key for a remote
# the resolver verified.
_GIT_CONFIG_INCLUDE_SECTIONS = frozenset({"include", "includeif"})
# core keys that select a program a push runs: the ssh command (even a
# --dry-run), the git:// proxy, and the ask-pass program.
_GIT_CONFIG_CORE_PROGRAM_KEYS = frozenset({"sshcommand", "gitproxy", "askpass"})
# core/sequence keys that select a program git runs on routine subcommands,
# not only on a push: core.hooksPath selects the directory git takes hooks
# from, core.editor the editor git runs for commit/tag/rebase without -m,
# sequence.editor the editor an interactive rebase runs, and core.fsmonitor
# the filesystem-monitor daemon git runs for status-bearing subcommands
# such as add and commit.
_GIT_CONFIG_CORE_PROGRAM_SELECTION_KEYS = frozenset({
    "editor",
    "fsmonitor",
    "hookspath",
})
_GIT_CONFIG_SEQUENCE_PROGRAM_SELECTION_KEYS = frozenset({"editor"})
# The man section selects the program `git help` runs: man.<tool>.path names
# the man viewer's executable for the tool man.viewer selects (git 2.53
# rejects an arbitrary man.viewer command string, so the path key is the
# live vector), and interactive.diffFilter names the pager-style filter
# git runs for `git add -p` diffs.
_GIT_CONFIG_MAN_PROGRAM_SELECTION_KEYS = frozenset({"path", "viewer"})
_GIT_CONFIG_INTERACTIVE_PROGRAM_SELECTION_KEYS = frozenset({"difffilter"})
# The driver sections select programs the same way on routine subcommands:
# filter.<name>.clean and filter.<name>.process run for files with that
# filter (git add, git checkout), merge.<name>.driver runs for files with
# that merge driver (git merge), and diff.<name>.command,
# diff.<name>.textconv, and diff.<name>.external run for diffs with that
# driver (git diff, git log -p).
_GIT_CONFIG_FILTER_PROGRAM_SELECTION_KEYS = frozenset({"clean", "process"})
_GIT_CONFIG_MERGE_PROGRAM_SELECTION_KEYS = frozenset({"driver"})
_GIT_CONFIG_DIFF_PROGRAM_SELECTION_KEYS = frozenset({"command", "external", "textconv"})


def _git_config_key_is_push_effecting(key: str) -> bool:  # noqa: PLR0911
    """Whether one git config key changes where a granted push lands.

    The guarded surface is every key whose write redefines the push
    destination or the program a push runs: ``remote.<name>.url`` and
    ``remote.<name>.pushurl`` define where a push to that remote lands,
    ``remote.<name>.receivepack`` and ``remote.<name>.uploadpack`` select
    the program git runs for the remote (receivepack is the config twin of
    the hard-guarded ``--receive-pack`` option; uploadpack is its
    fetch-side twin, guarded for symmetry), ``remote.<name>.proxy`` selects
    the proxy program git runs, ``remote.<name>.vcs`` selects the
    ``git-remote-<vcs>`` helper that ``git remote update`` executes,
    ``url.<base>.insteadOf`` and ``url.<base>.pushInsteadOf`` rewrite the
    URL a push uses, ``branch.<name>.remote`` and ``branch.<name>.pushremote``
    select the destination of a bare push (path-capable),
    ``remote.<name>.mirror`` mirrors every push to that remote,
    ``remote.<name>.push`` installs the refspecs a refspec-less push uses
    (forced or deleting ones included), and ``remote.pushDefault`` (also
    spelled ``remote.<name>.pushDefault``) selects the remote a bare push
    targets. The program-execution family is guarded the same way:
    ``core.sshCommand`` selects the program git runs for an ssh push (even
    a ``--dry-run``), ``core.gitProxy`` the program it runs for a
    ``git://`` push, ``core.askPass`` the program it runs to ask for a
    credential, and the whole ``credential`` section installs the
    credential-lookup program a push runs (``credential.helper`` and
    ``credential.<url>.helper``). Finally, ``include.path`` and
    ``includeIf.<condition>.path`` author a file git folds into
    configuration resolution, which can set any of these keys for a remote
    the resolver verified. Git folds section and key names
    case-insensitively, so the guard does too; remote, branch, and url-base
    names are case-sensitive, but the guard fails closed on any spelling.
    """
    *sections, name = key.split(".")
    if not sections:
        return False
    folded = name.casefold()
    section = sections[0].casefold()
    if section == "remote":
        if folded in _GIT_CONFIG_REMOTE_PUSH_EFFECTING_KEYS:
            # The name-bearing keys carry the remote name.
            return bool(sections[1:])
        return folded == "pushdefault"
    if section == "branch":
        return folded in _GIT_CONFIG_BRANCH_PUSH_EFFECTING_KEYS and bool(sections[1:])
    if section == "url":
        return folded in _GIT_CONFIG_URL_PUSH_EFFECTING_KEYS and bool(sections[1:])
    if section in _GIT_CONFIG_INCLUDE_SECTIONS:
        # include.path and includeIf.<condition>.path are the only include
        # keys; the whole section is guarded so no spelling slips through.
        return True
    if section == "credential":
        # credential.helper and credential.<url>.helper install a
        # credential-lookup program; the whole section is guarded so no
        # helper spelling slips through.
        return True
    if section == "core":
        return folded in _GIT_CONFIG_CORE_PROGRAM_KEYS
    return False


def _git_config_section_is_push_effecting(section: str) -> bool:
    """Whether one section operand of a section operation is push-effecting.

    A section rename or removal relocates or deletes whole sections, so it
    can set or relocate any key the guarded surface covers: the remote
    definitions, the url rewrites, the branch push destinations, the
    program-execution keys, the credential section, and the include
    directives.
    """
    folded = section.casefold()
    return folded in {
        "remote",
        "url",
        "branch",
        "credential",
        "include",
        "includeif",
    } or folded.startswith((
        "remote.",
        "url.",
        "branch.",
        "credential.",
        "include.",
        "includeif.",
    ))


def _git_config_key_is_program_selection(key: str) -> bool:  # noqa: PLR0911
    """Whether one git config key selects a program git runs on routine subcommands.

    ``core.hooksPath`` selects the directory git runs hooks from, so a granted
    push — or any hook-running subcommand, including readers that refresh the
    index — executes the hooks there; ``core.editor`` selects the program
    ``git commit`` (without ``-m``), ``git tag -a``, and ``git rebase`` run;
    ``sequence.editor`` selects the program an interactive rebase runs; and
    ``core.fsmonitor`` selects the daemon git runs for status-bearing
    subcommands (``git add``, ``git commit``). The driver sections select
    programs the same way on routine subcommands:
    ``filter.<name>.clean`` and ``filter.<name>.process`` run for files with
    that filter (``git add``, ``git checkout``), ``merge.<name>.driver`` runs
    for files with that merge driver (``git merge``), and
    ``diff.<name>.command``, ``diff.<name>.textconv``, and
    ``diff.<name>.external`` run for diffs with that driver (``git diff``,
    ``git log -p``). The ``man`` section selects the program ``git help``
    runs: ``man.<tool>.path`` names the man viewer's executable for the tool
    ``man.viewer`` selects, and ``interactive.diffFilter`` names the filter
    git runs for ``git add -p`` diffs. Git folds section and key names
    case-insensitively, so the guard does too.
    """
    *sections, name = key.split(".")
    if not sections:
        return False
    folded = name.casefold()
    section = sections[0].casefold()
    if section in {"core", "sequence"}:
        keys = (
            _GIT_CONFIG_CORE_PROGRAM_SELECTION_KEYS
            if section == "core"
            else _GIT_CONFIG_SEQUENCE_PROGRAM_SELECTION_KEYS
        )
        return folded in keys
    if section == "filter":
        return folded in _GIT_CONFIG_FILTER_PROGRAM_SELECTION_KEYS and bool(
            sections[1:]
        )
    if section == "merge":
        return folded in _GIT_CONFIG_MERGE_PROGRAM_SELECTION_KEYS and bool(sections[1:])
    if section == "diff":
        return folded in _GIT_CONFIG_DIFF_PROGRAM_SELECTION_KEYS and bool(sections[1:])
    if section == "man":
        # man.viewer selects the viewer tool and man.<tool>.path names its
        # executable; both keys are guarded (a bare man.path or a
        # man.<tool>.viewer spelling is not a real git key, but guarding it
        # fails closed).
        return folded in _GIT_CONFIG_MAN_PROGRAM_SELECTION_KEYS
    if section == "interactive":
        return folded in _GIT_CONFIG_INTERACTIVE_PROGRAM_SELECTION_KEYS
    return False


def _git_config_section_is_program_selection(section: str) -> bool:
    """Whether one section operand of a rename reaches a program-selection section.

    Renaming a section to (or from) ``core``, ``sequence``, ``filter``,
    ``merge``, ``diff``, ``man``, or ``interactive`` moves its keys into or
    out of the sections whose ``hooksPath``/``editor``/``fsmonitor``/driver/
    ``man``/``diffFilter`` keys select the programs git runs, so an
    agent-authored key lands exactly where the key guard blocks writing it:
    guarding the key set alone is bypassable.
    """
    folded = section.casefold()
    return folded in {
        "core",
        "sequence",
        "filter",
        "merge",
        "diff",
        "man",
        "interactive",
    } or folded.startswith((
        "core.",
        "sequence.",
        "filter.",
        "merge.",
        "diff.",
        "man.",
        "interactive.",
    ))


def _git_config_push_effecting_reason(target: str) -> str:
    """Why a push-effecting config write is a hard guard."""
    return (
        f"git config writes {target!r}, push-effecting configuration: "
        "remote.<name>.url and remote.<name>.pushurl define where a push "
        "to that remote lands, remote.<name>.receivepack and "
        "remote.<name>.uploadpack select the program git runs for the "
        "remote (receivepack is the config twin of the hard-guarded "
        "--receive-pack option; uploadpack is its fetch-side twin, guarded "
        "for symmetry), remote.<name>.proxy selects the proxy program git "
        "runs, and remote.<name>.vcs selects the git-remote-<vcs> helper "
        "that git remote update executes (git remote update and prune stay "
        "routine: with vcs and sshCommand writes barred, the programs they "
        "run are user-configured), url.<base>.insteadOf and "
        "url.<base>.pushInsteadOf rewrite the URL a push uses, "
        "branch.<name>.remote and branch.<name>.pushremote select the "
        "destination of a bare push, remote.<name>.mirror mirrors every "
        "push to that remote, remote.<name>.push installs the refspecs a "
        "refspec-less push uses, and remote.pushDefault selects the remote "
        "a bare push targets; core.sshCommand selects the program git runs "
        "for an ssh push (even a --dry-run), core.gitProxy the program it "
        "runs for a git:// push, core.askPass the program it runs to ask "
        "for a credential, and credential.helper and credential.<url>.helper "
        "install the credential-lookup program a push runs; include.path "
        "and includeIf.<condition>.path author a file git folds into "
        "configuration, which can set any of these keys for a remote the "
        "resolver verified, so an approved push would run what the write "
        "selects; no grant can authorize the write. A push to a remote "
        "whose configuration you set yourself is ordinary publication — "
        "your own config and any file it includes define what push means "
        "there — so run this command manually if you intend it."
    )


def _git_config_program_selection_reason(target: str) -> str:
    """Why a program-selection config write is a hard guard."""
    return (
        f"git config writes {target!r}, program-selection configuration: "
        "core.hooksPath selects the directory git runs hooks from, so a "
        "granted push — or any hook-running git command, including readers "
        "that refresh the index — would execute the hooks there, "
        "core.editor selects the program git runs for commit, tag, and "
        "rebase without -m, sequence.editor selects the program an "
        "interactive rebase runs, core.fsmonitor selects the daemon git "
        "runs for status-bearing subcommands (git add, git commit), "
        "filter.<name>.clean and filter.<name>.process select the programs "
        "git runs for files with that filter (git add, git checkout), "
        "merge.<name>.driver selects the program git runs for files with "
        "that merge driver (git merge), and diff.<name>.command, "
        "diff.<name>.textconv, and diff.<name>.external select the "
        "programs git runs for diffs with that driver (git diff, git log "
        "-p); man.<tool>.path selects the man viewer git runs for "
        "git help once man.viewer selects its tool, and "
        "interactive.diffFilter selects the filter git runs for git add -p "
        "diffs; no grant can authorize the write. A repository whose hooks, "
        "editor, fsmonitor daemon, filters, merge drivers, diff drivers, "
        "man viewer, or interactive diff filter you point at a program of "
        "your choosing is not covered by any git approval — run this "
        "command manually if you intend it"
    )


def _git_config_program_section_reason(target: str, *, destination: bool) -> str:
    """Why a section rename reaching a program-selection section is a hard guard.

    The denial names the destination section when the rename's destination
    operand reaches one, and otherwise the matched operand itself, so the
    reason always names the section the rename actually moves keys into or
    out of.
    """
    if destination:
        return (
            f"git config renames a section to {target!r}, a "
            "program-selection section: the rename moves the keys of "
            f"that section into {target}, whose hooksPath/editor/fsmonitor/"
            "filter/merge/diff/man/interactive keys select the programs "
            "git runs on routine subcommands, so an agent-authored key "
            "lands exactly where the key guard blocks writing it; no grant "
            "can authorize the write. Run this command manually if you "
            "intend it"
        )
    return (
        f"git config renames the program-selection section {target!r}: the "
        "rename moves the keys of that section out of the section whose "
        "program-selection keys the key guard blocks, and an agent-authored "
        "key could take their place; no grant can authorize the write. Run "
        "this command manually if you intend it"
    )


def _git_config_rename_guard_reason(operands: list[str]) -> str | None:
    """Why a section rename is a hard guard, naming the section it lands in.

    A rename to (or from) a program-selection section relocates keys into
    or out of the sections whose program-selection keys the key guard
    blocks, so an agent-authored hooksPath/editor or driver key would take
    effect. The destination operand is named when it reaches a guarded
    section; otherwise the matched operand is named.
    """
    rename_operands = list(operands)
    destination = rename_operands[-1] if rename_operands[1:] else None
    if destination is not None and _git_config_section_is_program_selection(
        destination
    ):
        return _git_config_program_section_reason(destination, destination=True)
    target = next(
        (
            operand
            for operand in rename_operands
            if _git_config_section_is_program_selection(operand)
        ),
        None,
    )
    if target is not None:
        return _git_config_program_section_reason(target, destination=False)
    return None


def _git_config_section_operation_guard_reason(
    operands: list[str], *, rename: bool
) -> str | None:
    """Why a section operation (rename or removal) is a hard guard, or None."""
    target = next(
        (
            operand
            for operand in operands
            if _git_config_section_is_push_effecting(operand)
        ),
        None,
    )
    if target is not None:
        return _git_config_push_effecting_reason(target)
    if rename:
        # Only renames are checked here: a removal cannot move a key into
        # a guarded section.
        return _git_config_rename_guard_reason(operands)
    return None


def _git_config_write_guard_reason(
    operands: list[str], *, section_operation: bool, rename: bool
) -> str | None:
    """Why a mutating ``git config`` call is a hard guard, or None when routine.

    The scan already established that the call mutates; this selects the first
    operand that reaches the guarded surface. Section operations check the
    push-effecting sections first and, for renames, the program-selection
    sections (either operand, so a destination section name is covered); key
    writes check the push-effecting keys and then the program-selection keys.
    """
    if section_operation:
        return _git_config_section_operation_guard_reason(operands, rename=rename)
    target = next(
        (operand for operand in operands if _git_config_key_is_push_effecting(operand)),
        None,
    )
    if target is not None:
        return _git_config_push_effecting_reason(target)
    target = next(
        (
            operand
            for operand in operands
            if _git_config_key_is_program_selection(operand)
        ),
        None,
    )
    if target is not None:
        return _git_config_program_selection_reason(target)
    return None


@dataclass(frozen=True)
class _GitConfigCall:
    """One scanned ``git config`` call, shared by classifier and runtime policy.

    Both layers must agree on which calls write: the classifier guards the
    push-effecting and program-selection keys, and the runtime policy governs
    the ``--file`` destination of a writing call. ``operands`` holds the
    positional operands with the subcommand word removed.
    """

    options: frozenset[str]
    operands: tuple[str, ...]
    subcommand: str | None
    values: tuple[tuple[str, str], ...]
    error: str | None
    reads: bool
    edits: bool
    mutates: bool
    section_operation: bool
    renames: bool


def _scan_git_config_call(args: list[str]) -> _GitConfigCall:
    """Scan one git config call into its write-relevant shape.

    The scan is lenient — git itself rejects unmodeled options — and the
    key check looks at every operand, not only the first, so an option
    whose value consumption is misread can only over-block, never miss a
    write. The only scan errors are a dangling or empty required value at
    the end of the invocation, which leaves every earlier --file value
    already extracted.
    """
    scan = _scan_classified_git_options(
        args, _GIT_CONFIG_TABLE, boolean_shorts=_GIT_CONFIG_BOOLEAN_SHORTS, strict=False
    )
    options = set(scan.options)
    operands = list(scan.operands)
    subcommand: str | None = None
    if operands and operands[0] in _GIT_CONFIG_SUBCOMMANDS:
        subcommand = operands.pop(0)
    reads = bool(options & _GIT_CONFIG_READ_OPTIONS) or (
        subcommand in _GIT_CONFIG_READ_SUBCOMMANDS
    )
    edits = bool(options & {"--edit", "-e"}) or subcommand == "edit"
    section_operation = bool(options & _GIT_CONFIG_SECTION_LONGS) or (
        subcommand in {"rename-section", "remove-section"}
    )
    mutates = (
        len(operands) > 1
        or bool(options & _GIT_CONFIG_WRITE_LONGS)
        or bool(options & _GIT_CONFIG_UNSET_LONGS)
        or bool(options & _GIT_CONFIG_SECTION_LONGS)
        or subcommand in _GIT_CONFIG_MUTATION_SUBCOMMANDS
    )
    return _GitConfigCall(
        options=frozenset(options),
        operands=tuple(operands),
        subcommand=subcommand,
        values=scan.values,
        error=scan.error,
        reads=reads,
        edits=edits,
        mutates=mutates,
        section_operation=section_operation,
        renames=subcommand == "rename-section" or "--rename-section" in options,
    )


def _classify_git_config(args: list[str]) -> DangerousCommandClassification:
    """git config: writes to push-effecting or program-selection keys are hard guards.

    ``git config`` is otherwise a permitted routine command, so reads stay
    routine — ``--get``/``--list``/``--get-urlmatch``/... and the ``get`` and
    ``list`` subcommands, including a bare key, which git reads — and so do
    writes to unrelated keys, which cannot change where a granted push
    lands or what program git runs. What must never classify routine is the
    agent redefining the push-effecting surface: a configured
    ``remote.<name>.url``, ``remote.<name>.pushurl``, ``url.<base>.insteadOf``,
    or ``branch.<name>.remote`` points a granted push at a destination of the
    agent's choosing, ``remote.<name>.receivepack`` selects the program the
    push runs, ``remote.<name>.vcs`` selects the helper ``git remote
    update`` executes, ``core.sshCommand``, ``core.gitProxy``,
    ``core.askPass``, and the ``credential`` section select the other
    programs a push runs (the ssh command, the git:// proxy, the ask-pass
    program, and the credential lookup), and ``include.path`` or
    ``includeIf.<condition>.path`` authors a file git folds into
    configuration, which can set any of those keys for a remote the
    resolver verified, while a configured ``remote.<name>.mirror``,
    ``remote.<name>.push`` refspec, or ``remote.pushDefault`` makes an
    ordinary ``git.push`` approval authorize configured forced, deleted, or
    mirrored updates. The program-selection surface is guarded the same way
    in every write scope (local, ``--global``, ``--file``, ``--worktree``,
    and ``-C``/``--git-dir`` relocation): ``core.hooksPath`` selects the
    directory git runs hooks from, ``core.editor`` the program git runs for
    commit/tag/rebase without ``-m``, ``sequence.editor`` the program an
    interactive rebase runs, ``core.fsmonitor`` the daemon git runs for
    status-bearing subcommands (``git add``, ``git commit``),
    ``filter.<name>.clean`` and
    ``filter.<name>.process`` the programs git runs for files with that
    filter (``git add``, ``git checkout``), ``merge.<name>.driver`` the
    program git runs for files with that merge driver (``git merge``), and
    ``diff.<name>.command``, ``diff.<name>.textconv``, and
    ``diff.<name>.external`` the programs git runs for diffs with that
    driver (``git diff``, ``git log -p``), ``man.<tool>.path`` the man
    viewer git runs for ``git help`` once ``man.viewer`` selects its tool,
    and ``interactive.diffFilter`` the filter git runs for ``git add -p``
    diffs — each executes on routine
    subcommands, so every mutation of those keys — set, add, replace-all,
    unset, the ``set``/``unset`` subcommands — is a hard guard with no
    grantable operation. A section rename reaching ``core``, ``sequence``,
    ``filter``, ``merge``, ``diff``, ``man``, or ``interactive`` (either
    operand, so a destination section name is covered) relocates keys into
    the section whose program-selection keys the key guard blocks, and is a
    hard guard too; section removals stay routine for those sections
    because a removal cannot move a key into a guarded section. Editor
    writes (``--edit``/``-e``) are hard guards as well: an editor can set
    any key, so it is never classifiable by key. The boundary is deliberate: a push
    to a remote whose configuration the USER set is ordinary publication,
    because the user's own config — includes included — defines what push
    means there; only the agent's writes are barred.
    """
    call = _scan_git_config_call(args)
    if call.error is not None:
        return _hard_denied(f"git config: {call.error}")
    if call.reads:
        # A read never writes configuration.
        return _ROUTINE
    if call.edits:
        # An editor write can set any key, so it is never classifiable.
        return _hard_denied(
            "git config --edit opens an editor on the config file, which "
            "can set push-effecting configuration (remote.<name>.url, "
            "remote.<name>.pushurl, remote.<name>.receivepack, "
            "remote.<name>.uploadpack, remote.<name>.proxy, "
            "remote.<name>.vcs, url.<base>.insteadOf, "
            "url.<base>.pushInsteadOf, branch.<name>.remote, "
            "branch.<name>.pushremote, remote.<name>.mirror, "
            "remote.<name>.push, remote.pushDefault, core.sshCommand, "
            "core.gitProxy, core.askPass, credential.helper, "
            "credential.<url>.helper, include.path, "
            "includeIf.<condition>.path) or program-selection configuration "
            "(core.hooksPath, core.editor, sequence.editor, "
            "core.fsmonitor, filter.<name>.clean, filter.<name>.process, "
            "merge.<name>.driver, diff.<name>.command, "
            "diff.<name>.textconv, diff.<name>.external, "
            "man.<tool>.path, man.viewer, interactive.diffFilter) that no "
            "grant can authorize; run it manually if you intend it"
        )
    if not call.mutates:
        # A bare key with no action is git's read form; nothing is written.
        return _ROUTINE
    if reason := _git_config_write_guard_reason(
        list(call.operands),
        section_operation=call.section_operation,
        rename=call.renames,
    ):
        return _hard_denied(reason)
    return _ROUTINE


def _classify_git_remote(args: list[str]) -> DangerousCommandClassification:
    """git remote: mutations of the remote definitions are hard guards.

    ``git remote add``/``set-url`` point a bare destination name at an
    arbitrary URL or path, ``rename`` retargets an existing remote, and
    ``remove`` (``rm``) turns a configured bare name back into a URL git
    resolves as a path — each redefines where a granted push lands, so no
    grant can authorize it. Reads (``-v``, ``show``, ``get-url``) and the
    other non-mutating forms stay routine, including ``update`` and
    ``prune``: with ``remote.<name>.vcs`` and ``core.sshCommand`` writes
    barred, the helper and ssh program those forms run are
    user-configured, so they stay in-contract.
    """
    scan = _scan_classified_git_options(
        args, _GIT_REMOTE_TABLE, boolean_shorts=_GIT_REMOTE_BOOLEAN_SHORTS, strict=False
    )
    if scan.error is not None:
        return _hard_denied(f"git remote: {scan.error}")
    if scan.operands and scan.operands[0] in _GIT_REMOTE_MUTATION_SUBCOMMANDS:
        return _hard_denied(
            f"git remote {scan.operands[0]} redefines where a granted push "
            "lands: it defines, repoints, renames, or removes the remote a "
            "push destination resolves through, and a removed remote turns "
            "the bare name into a URL git resolves as a path; no grant can "
            "authorize it — run it manually if you intend it"
        )
    return _ROUTINE


def _dispatch_git_subcommand(
    subcommand: str, args: list[str], globals_scan: _GitGlobalScan
) -> DangerousCommandClassification:
    classification: DangerousCommandClassification
    if subcommand == "push":
        classification = _classify_git_push(args, globals_scan)
    elif subcommand in _GIT_WORKTREE_TABLES:
        classification = _classify_git_worktree_subcommand(
            subcommand,
            f"git.{subcommand}",
            _GIT_WORKTREE_TABLES[subcommand],
            args,
            globals_scan,
        )
    elif subcommand == "switch":
        classification = _classify_git_switch(args, globals_scan)
    elif subcommand in {"stash", "reflog"}:
        classification = _classify_git_stash_or_reflog(subcommand, args, globals_scan)
    elif subcommand in {"reset", "clean"}:
        classification = _classify_git_hard_guard_subcommand(subcommand, args)
    elif subcommand == "config":
        classification = _classify_git_config(args)
    elif subcommand == "remote":
        classification = _classify_git_remote(args)
    else:
        classification = _classify_git_remaining(subcommand)
    return classification


def classify_dangerous_git_command(tokens: list[str]) -> DangerousCommandClassification:
    """Classify one whole expanded call for the dangerous-command flow.

    Pure and typed: the call is ``routine`` (no dangerous-command
    involvement), ``hard_denied`` (denied with no grantable operation), or
    ``grantable`` (denied pending per-operation grants; the classification
    carries the required operations, consequence, modeled operands, and
    repository relocation). Only git invocations are classified; every other
    command is routine and stays governed by the existing policy layer, whose
    enforcement is unchanged. Ambiguous or unmodeled forms fail closed: never
    grantable, never an ordinary git.push. ``git config`` and ``git remote``
    are the routed subcommands that mostly stay routine — they are permitted
    commands — so only writes to push-effecting configuration (remote
    definitions and url rewrites, branch push destinations, mirror/push
    refspecs, push defaults, the programs a push runs, the credential
    section, and the include directives), writes to program-selection
    configuration (``core.hooksPath``, ``core.editor``, ``sequence.editor``,
    ``filter.<name>.clean``, ``filter.<name>.process``,
    ``merge.<name>.driver``, ``diff.<name>.command``,
    ``diff.<name>.textconv``, ``diff.<name>.external``, in every write
    scope, plus section renames reaching ``core``, ``sequence``,
    ``filter``, ``merge``, or ``diff``), and ``git remote`` mutations (add,
    set-url, rename, remove) are hard guards: a push to a remote whose
    configuration the user set is ordinary publication, because the user's
    own config — includes included — defines what push means there, and
    only the agent's writes and remote mutations are barred.
    """
    if not tokens or _command_name(tokens[0]) != "git":
        return _ROUTINE
    globals_scan = _scan_git_globals(tokens[1:])
    if globals_scan.error is not None:
        return _hard_denied(globals_scan.error)
    args = tokens[1:]
    if globals_scan.index >= len(args):
        # No subcommand: bare git prints usage/version and carries nothing to
        # classify.
        return _ROUTINE
    subcommand = args[globals_scan.index]
    return _dispatch_git_subcommand(
        subcommand, args[globals_scan.index + 1 :], globals_scan
    )


# Subcommands the dangerous-command flow routes: malformed or unmodeled forms
# of these fail closed (git config and git remote excepted: they are
# permitted routine commands, so they stay routine unless they mutate the
# push-effecting surface). Every other git invocation — readers, unmodeled
# builtins, unrecognized subcommands, dangling globals without a subcommand —
# carries no dangerous operation and stays governed by the existing policy
# layer, whose enforcement is unchanged.
_DANGEROUS_GIT_SUBCOMMANDS = frozenset({
    "push",
    "checkout",
    "restore",
    "switch",
    "stash",
    "reflog",
    "reset",
    "clean",
    "config",
    "remote",
})


def is_dangerous_git_subcommand(tokens: list[str]) -> bool:
    """Whether one git invocation routes through the dangerous-command flow."""
    canonical = canonical_git_command(tokens)
    return (
        canonical is not None
        and len(canonical) > 1
        and canonical[1] in _DANGEROUS_GIT_SUBCOMMANDS
    )


def resolve_repository_scope(
    cwd: Path, relocation: GitRepositoryRelocation
) -> RepositoryScope | None:
    """Resolve the repository/worktree identity one git invocation operates on.

    Pure filesystem inspection; git is never invoked. ``cwd`` is the statically
    tracked working directory of the invocation and ``relocation`` carries
    git's global -C/--git-dir/--work-tree operands (each -C applies in turn,
    before the override paths are resolved, exactly as git applies them).
    Returns None when a repository marker is present but its identity cannot
    be determined, so callers fail closed. A directory governed by no
    repository yields that directory's nominal scope: git itself refuses such
    a call, so the nominal identity only ever matters for grant scoping.
    """
    current = (cwd if cwd.is_absolute() else Path.cwd() / cwd).resolve()
    for value in relocation.cwd_paths:
        current = resolve_tool_path(value, current)
    work_tree: Path | None = None
    if relocation.work_tree is not None:
        work_tree = resolve_tool_path(relocation.work_tree, current)
    if relocation.git_dir is not None:
        return RepositoryScope(
            git_dir=resolve_tool_path(relocation.git_dir, current), work_tree=work_tree
        )
    scope: RepositoryScope | None = None
    for directory in (current, *current.parents):
        marker = directory / ".git"
        if marker.is_dir():
            scope = RepositoryScope(
                git_dir=marker.resolve(), work_tree=work_tree or directory
            )
            break
        if (
            (directory / "HEAD").is_file()
            and (directory / "config").is_file()
            and (directory / "objects").is_dir()
            and (directory / "refs").is_dir()
        ):
            # A bare repository (or a git dir used directly) has no work tree.
            scope = RepositoryScope(git_dir=directory.resolve(), work_tree=work_tree)
            break
        if not marker.is_file():
            continue
        try:
            marker_value = marker.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        prefix, separator, value = marker_value.partition(":")
        if not separator or prefix.casefold() != "gitdir":
            # Not a usable repository pointer; git itself would refuse it, so
            # the repository identity cannot be determined.
            return None
        git_dir = Path(value.strip()).expanduser()
        if not git_dir.is_absolute():
            git_dir = directory / git_dir
        # A linked worktree's own git dir is its identity: sibling worktrees
        # of one repository never share a scope.
        scope = RepositoryScope(
            git_dir=git_dir.resolve(), work_tree=work_tree or directory
        )
        break
    if scope is None:
        scope = RepositoryScope(
            git_dir=(current / ".git").resolve(), work_tree=work_tree or current
        )
    return scope


# Routine git subcommands that write work-tree files: tracked paths under the
# work tree (checkout/switch/restore/merge/cherry-pick/revert/rebase/am and
# the whole stash subcommand, whose push/pop/apply forms all touch the work
# tree) or patch paths (apply), plus pull, whose merge step writes the work
# tree. A relocation that resolves the work tree inside the git metadata dir
# turns each of them into a metadata writer, so the resolver denies them
# there. fetch never writes the work tree and stays out; reset --keep/--merge,
# git rm/mv, bisect, and submodule stay routine under such a relocation (a
# documented residual).
_GIT_TREE_WRITING_SUBCOMMANDS = frozenset({
    "am",
    "apply",
    "cherry-pick",
    "checkout",
    "merge",
    "pull",
    "rebase",
    "restore",
    "revert",
    "stash",
    "switch",
})

# Position of the subcommand token in a canonical git command.
_GIT_SUBCOMMAND_INDEX = 1


def git_work_tree_writing_subcommand(tokens: list[str]) -> str | None:
    """The subcommand when one git invocation writes work-tree files.

    Returns the subcommand name for the routine tree-writing subcommands, or
    None for every other invocation — readers, unmodeled or unrecognized
    forms (already denied or governed elsewhere), and non-git commands.
    """
    canonical = canonical_git_command(tokens)
    if canonical is None or len(canonical) <= _GIT_SUBCOMMAND_INDEX:
        return None
    subcommand = canonical[1]
    return subcommand if subcommand in _GIT_TREE_WRITING_SUBCOMMANDS else None


def resolve_git_work_tree(tokens: list[str], cwd: Path) -> Path | None:
    """Resolve the work tree one git invocation writes, without invoking git.

    Mirrors git's own resolution order: the ``-C`` paths apply in turn, then
    ``--work-tree``, then ``core.worktree`` from the repository's own config
    (relative values resolve against the git dir, as git resolves them), then
    discovery — the directory holding the ``.git`` marker, or the current
    directory itself when it is a git dir (git then uses the cwd as the work
    tree). Returns None when the work tree cannot be determined, so callers
    fail closed.
    """
    globals_scan = _scan_git_globals(tokens[1:])
    if globals_scan.error is not None:
        return None
    relocation = globals_scan.relocation
    current = (cwd if cwd.is_absolute() else Path.cwd() / cwd).resolve()
    for value in relocation.cwd_paths:
        current = resolve_tool_path(value, current)
    if relocation.work_tree is not None:
        # GIT_WORK_TREE wins over core.worktree and discovery.
        return resolve_tool_path(relocation.work_tree, current)
    scope = resolve_repository_scope(
        current, GitRepositoryRelocation(git_dir=relocation.git_dir)
    )
    if scope is None:
        return None
    # An explicit git dir without --work-tree, or a git dir discovered as the
    # cwd itself, leaves git using the cwd as the work tree.
    work_tree = scope.work_tree if scope.work_tree is not None else current
    config_paths = _git_dir_config_paths(scope.git_dir)
    entries = (
        _parse_git_config_entries(config_paths) if config_paths is not None else None
    )
    if entries is None:
        # An unreadable commondir pointer or an unparsable config: the work
        # tree cannot be verified, so fail closed.
        return None
    configured = next(
        (
            value
            for section, key, value in entries
            if section == "core" and key == "worktree"
        ),
        None,
    )
    if configured is None:
        return work_tree
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = scope.git_dir / path
    return path.resolve()


def git_cwd_relocations(tokens: list[str], cwds: set[Path]) -> set[Path] | None:
    """The directories git's own -C sequence relocates each candidate cwd to.

    Git applies its -C operands in turn — each relative to the directory the
    previous one produced — before the subcommand runs, so every relative
    operand of the subcommand, raw write operands included, resolves against
    the final directory (mirroring how ``resolve_git_work_tree`` models the
    same sequence). Returns None when the invocation carries no -C, so those
    operands resolve against the tracked cwds unchanged. An unscannable global
    sequence never reaches operand resolution — the policy layer denies it
    before the operand guard runs — and an empty -C value is a no-op
    relocation git itself accepts, so the tracked cwds stay correct there too.
    """
    globals_scan = _scan_git_globals(tokens[1:])
    if globals_scan.error is not None or not globals_scan.relocation.cwd_paths:
        return None
    relocated: set[Path] = set()
    for cwd in cwds:
        current = (cwd if cwd.is_absolute() else Path.cwd() / cwd).resolve()
        for value in globals_scan.relocation.cwd_paths:
            current = resolve_tool_path(value, current)
        relocated.add(current)
    return relocated


_GIT_CONFIG_ENTRY = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9-]*)\s*(?:=\s*(?P<value>.*))?$"
)
_FALSE_GIT_CONFIG_VALUES = frozenset({"", "0", "false", "no", "off"})
# The section name is casefolded on parse; the remediation text restores the
# spelling git documents for the directive.
_INCLUDE_DIRECTIVE_NAMES = {"include": "include", "includeif": "includeIf"}


def _git_config_paths(cwd: Path) -> tuple[Path, ...] | None:  # noqa: PLR0911
    """Locate repository-owned config without invoking Git.

    Returns None when a repository is present but its config location cannot be
    determined, so callers fail closed. Returns () when no repository governs
    ``cwd``. Worktrees point at the shared config through ``commondir``; the
    worktree-specific ``config.worktree`` is read alongside it.
    """
    for directory in (cwd, *cwd.parents):
        marker = directory / ".git"
        if marker.is_dir():
            return marker / "config", marker / "config.worktree"
        if (
            (directory / "HEAD").is_file()
            and (directory / "config").is_file()
            and (directory / "objects").is_dir()
            and (directory / "refs").is_dir()
        ):
            return directory / "config", directory / "config.worktree"
        if not marker.is_file():
            continue
        try:
            marker_value = marker.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        prefix, separator, value = marker_value.partition(":")
        if not separator or prefix.casefold() != "gitdir":
            # Not a usable repository pointer; git itself would refuse it.
            return ()
        git_dir = Path(value.strip()).expanduser()
        if not git_dir.is_absolute():
            git_dir = directory / git_dir
        try:
            common_value = (git_dir / "commondir").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            # No commondir: a --separate-git-dir checkout, whose gitdir's own
            # config is the repository config. A linked worktree always has a
            # commondir, and git refuses to operate in one without it.
            return git_dir / "config", git_dir / "config.worktree"
        except (OSError, UnicodeError):
            # An unreadable commondir would fall back to the worktree gitdir's
            # own config, which does not exist in a linked worktree; fail
            # closed like the .git-pointer and config reads.
            return None
        common_dir = Path(common_value)
        if not common_dir.is_absolute():
            common_dir = git_dir / common_dir
        return common_dir / "config", git_dir / "config.worktree"
    return ()


def git_metadata_paths(tokens: list[str], *, cwd: Path) -> tuple[Path, ...] | None:
    """Resolve repository metadata identity without running Git or its helpers.

    Include the ambient repository as well as global-option overrides. An explicit
    git-dir is protected even if it is not yet a complete repository.
    """
    roots = {cwd}
    explicit_dirs: set[Path] = set()
    override_values: list[tuple[str, str]] = []
    if tokens and _command_name(tokens[0]) == "git":
        current = cwd
        index = 1
        while index < len(tokens) and tokens[index].startswith("-"):
            token = tokens[index]
            name, separator, value = token.partition("=")
            if token == "-C" or name in {"--git-dir", "--work-tree"}:
                if not separator:
                    index += 1
                    if index >= len(tokens):
                        return None
                    value = tokens[index]
                path = Path(value).expanduser()
                path = (path if path.is_absolute() else current / path).resolve()
                if token == "-C":
                    current = path
                elif name == "--git-dir":
                    explicit_dirs.add(path)
                    override_values.append((name, value))
                else:
                    roots.add(path)
                    override_values.append((name, value))
            elif token.startswith("-C"):
                current = (current / token[2:]).resolve()
            index += 1
        roots.add(current)
        # Git applies -C before resolving repository override paths, even when
        # the override appeared earlier on the command line.
        for name, value in override_values:
            path = Path(value).expanduser()
            path = (path if path.is_absolute() else current / path).resolve()
            (explicit_dirs if name == "--git-dir" else roots).add(path)
    metadata = set(explicit_dirs)
    for root in roots:
        configs = _git_config_paths(root)
        if configs is None:
            return None
        metadata.update(config.parent.resolve() for config in configs)
    # Follow commondir for explicit git-dir and symlinked .git directories too.
    for git_dir in tuple(metadata):
        try:
            value = (git_dir / "commondir").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError):
            return None
        common = Path(value)
        metadata.add((common if common.is_absolute() else git_dir / common).resolve())
    return tuple(metadata)


def _parse_git_config_entries(
    config_paths: tuple[Path, ...],
) -> tuple[tuple[str, str, str], ...] | None:
    """Parse config file entries without invoking Git.

    None means one of the files exists but could not be read or parsed;
    callers fail closed on that result.
    """
    entries: list[tuple[str, str, str]] = []
    for config_path in config_paths:
        if not config_path.exists():
            continue
        try:
            lines = config_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return None
        section = ""
        for raw_line in lines:
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith("["):
                header, separator, line = line.partition("]")
                if not separator or not header[1:].strip():
                    return None
                section = (
                    header[1:].split(maxsplit=1)[0].split(".", maxsplit=1)[0].casefold()
                )
                line = line.strip()
                if not line or line.startswith(("#", ";")):
                    continue
            if match := _GIT_CONFIG_ENTRY.fullmatch(line):
                value = match.group("value")
                entries.append((
                    section,
                    match.group("key").casefold(),
                    # A valueless bare key is true; an explicit empty value
                    # after "=" is unset, which git parses as false.
                    ("true" if value is None else value).strip().strip('"'),
                ))
            else:
                # Continuations and unknown syntax can conceal executable keys.
                return None
    return tuple(entries)


def _git_config_entries(cwd: Path) -> tuple[tuple[str, str, str], ...] | None:
    """Parse repository-owned config entries without invoking Git.

    None means a repository governs ``cwd`` but its config could not be read;
    callers fail closed on that result.
    """
    config_paths = _git_config_paths(cwd)
    if config_paths is None:
        return None
    return _parse_git_config_entries(config_paths)


# One [section] / [section "sub"] / [section.sub] header with the
# subsection captured: remote names live in the subsection, which the entry
# parser's top-level section fold drops.
_GIT_CONFIG_SECTION_HEADER = re.compile(
    r"\[\s*(?P<section>[A-Za-z][A-Za-z0-9-]*)"
    r"(?:\s+\"(?P<quoted>[^\"]*)\"|\.(?P<dotted>[^\]\s]+))?"
    r"\s*\]"
)


def _git_dir_config_paths(git_dir: Path) -> tuple[Path, ...] | None:
    """Locate the config files one git dir reads, without invoking Git.

    A linked worktree's git dir shares the common repository's config
    through ``commondir`` and adds its own ``config.worktree``; every other
    git dir reads its own ``config`` (plus ``config.worktree`` when the
    worktree-config extension is in use). Returns None when a commondir
    pointer exists but cannot be read, so callers fail closed.
    """
    try:
        common_value = (git_dir / "commondir").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return git_dir / "config", git_dir / "config.worktree"
    except (OSError, UnicodeError):
        # An unreadable commondir would fall back to the worktree gitdir's
        # own config, which does not exist in a linked worktree; fail closed
        # like the config reads.
        return None
    common_dir = Path(common_value)
    if not common_dir.is_absolute():
        common_dir = git_dir / common_dir
    return common_dir / "config", git_dir / "config.worktree"


def _git_config_remote_url_names(
    config_paths: tuple[Path, ...],
) -> frozenset[str] | None:
    """Remote names carrying a configured ``remote.<name>.url``, or None.

    Parses the same repository-owned config grammar as
    ``_git_config_entries``, keeping the section subsection the top-level
    fold drops. A remote name resolves only when its ``url`` key is
    present, because git resolves any destination that is not a configured
    remote name as a URL. None means a config could not be read or parsed,
    so callers fail closed.
    """
    names: set[str] = set()
    for config_path in config_paths:
        if not config_path.exists():
            continue
        try:
            lines = config_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return None
        remote_name: str | None = None
        for raw_line in lines:
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith("["):
                header, separator, line = line.partition("]")
                if not separator:
                    return None
                match = _GIT_CONFIG_SECTION_HEADER.fullmatch(f"{header}]")
                if match is None:
                    # Unknown header syntax can conceal a remote definition.
                    return None
                section = match.group("section").casefold()
                subsection = match.group("quoted")
                if subsection is None:
                    subsection = match.group("dotted")
                remote_name = subsection if section == "remote" else None
                line = line.strip()
                if not line or line.startswith(("#", ";")):
                    continue
            if match := _GIT_CONFIG_ENTRY.fullmatch(line):
                if remote_name is not None and match.group("key").casefold() == "url":
                    names.add(remote_name)
            else:
                # Continuations and unknown syntax can conceal a url key.
                return None
    return frozenset(names)


def git_configured_remote_names(scope: RepositoryScope) -> frozenset[str] | None:
    """Remote names one repository scope resolves from its own config.

    Pure filesystem inspection; git is never invoked. A push destination
    operand resolves to a remote only when the repository's config carries
    ``remote.<name>.url`` for it: git resolves any destination that is not
    a configured remote name as a URL — a path, for a bare name — so a
    remote with only ``remote.<name>.mirror`` or ``remote.<name>.push``
    config is NOT resolvable. None means the repository's config could not
    be read, so callers fail closed. The scope's git dir is the config
    source: global and system config are not consulted, because the grant
    scopes the repository whose own config defines its remotes.
    """
    config_paths = _git_dir_config_paths(scope.git_dir)
    if config_paths is None:
        return None
    return _git_config_remote_url_names(config_paths)


def git_push_destination_risk(
    call: ModeledGitCall, scope: RepositoryScope
) -> str | None:
    """The unconfigured-destination vector of one push call in one scope.

    A push destination operand that is a bare name resolves only through the
    repository's configured remotes: git resolves any destination that is
    not a configured remote name as a URL — a path, for a bare name — so a
    push to an unconfigured name runs the destination's receive-pack and
    hooks, outside the repository the grant covers. The first positional is
    always the repository and ``--repo`` overrides it, so both operands are
    verified.

    Returns None when the call is not a push, carries no destination
    operand (git then resolves the destination from configuration the
    agent cannot write), or every destination resolves to a configured
    remote of the scope. A scope whose git dir does not exist governs no
    repository — git itself refuses the push there — so it is not verified.
    """
    if call.subcommand != "push":
        return None
    destinations = push_destination_operands(call)
    if not destinations:
        return None
    if not scope.git_dir.is_dir():
        # No repository governs this scope: git itself refuses the push
        # from such a directory, so no destination can run there.
        return None
    remotes = git_configured_remote_names(scope)
    if remotes is None:
        repository = scope.work_tree or scope.git_dir
        return (
            f"the configuration of the repository at {repository} could not "
            "be read, so the push destination cannot be verified as a "
            "configured remote; failing closed"
        )
    for destination in destinations:
        if destination not in remotes:
            repository = scope.work_tree or scope.git_dir
            return (
                f"destination {destination!r} is not a configured remote of "
                f"the repository at {repository}: git resolves a push "
                "destination that is not a configured remote name as a URL, "
                "so a push to it runs the destination's receive-pack and "
                "hooks, outside the repository the grant covers; no grant "
                "can authorize an unconfigured destination"
            )
    return None


def _git_value_is_active(value: str) -> bool:
    return value.casefold() not in _FALSE_GIT_CONFIG_VALUES


def _git_index_hook_risk(
    cwd: Path, entries: tuple[tuple[str, str, str], ...]
) -> str | None:
    # Readers such as status/diff can refresh and write the index. Conservatively
    # inspect every modeled reader, including stash's index-writing operations.
    # Do not interpret hooksPath quoting/expansion: any override fails closed.
    if any(section == "core" and key == "hookspath" for section, key, _ in entries):
        return "repository git config may execute arbitrary code via core.hooksPath"
    config_paths = _git_config_paths(cwd)
    if config_paths is None:
        return "repository git hooks cannot be inspected; failing closed"
    for config_path in config_paths:
        hook = config_path.parent / "hooks" / "post-index-change"
        try:
            mode = hook.stat().st_mode
        except FileNotFoundError:
            continue
        except OSError:
            return "repository git hooks cannot be inspected; failing closed"
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            return (
                "repository git hooks may execute arbitrary code via post-index-change"
            )
    return None


def git_repository_config_risk(  # noqa: PLR0911, PLR0912
    tokens: list[str], *, cwd: Path
) -> str | None:
    """The vector through which an otherwise-benign Git reader runs repo code.

    Adapts upstream v2.25.7's ``git_repository_requires_approval`` to
    chartreux's deny-only resolver: the returned string is the denial reason
    rather than an approval request. Upstream's ``git_repository_identity``
    (a stable approval-scope key) has no consumer here and is not ported.
    """
    policy = analyze_shell_command_policy(tokens)
    if not policy.inspect_git_repository or len(tokens) <= _GIT_SUBCOMMAND_INDEX:
        return None
    globals_result = _git_global_options(tokens[1:])
    if globals_result is None or globals_result[0] >= len(tokens) - 1:
        return None
    subcommand = tokens[1 + globals_result[0]].casefold()
    pager_enabled = True
    for option in tokens[1 : 1 + globals_result[0]]:
        if option in {"--no-pager", "-P"}:
            pager_enabled = False
        elif option in {"--paginate", "-p"}:
            pager_enabled = True

    def vector(key: str, remediation: str | None = None) -> str:
        # The resolver is deny-only with no passthrough, so the broadest
        # vectors carry the remediation the reader cannot discover itself.
        reason = f"repository git config may execute arbitrary code via {key}"
        return reason if remediation is None else f"{reason}; {remediation}"

    entries = _git_config_entries(cwd)
    if entries is None:
        return "repository git config is unreadable; failing closed"
    for section, key, _value in entries:
        # Includes can hide any of the executable settings checked below.
        if section in {"include", "includeif"}:
            return vector(
                f"{section}.{key}",
                f"audit or remove the [{_INCLUDE_DIRECTIVE_NAMES[section]}] "
                "directive in .git/config",
            )
    for section, key, value in entries:
        if not _git_value_is_active(value):
            continue
        if pager_enabled and subcommand in _GIT_PAGING_SUBCOMMANDS:
            if section == "core" and key == "pager":
                return vector("core.pager")
            if section == "pager" and key == subcommand:
                return vector(f"pager.{subcommand}")
    if subcommand in _GIT_READER_SUBCOMMANDS:
        for section, key, value in entries:
            if not _git_value_is_active(value):
                continue
            if section == "core" and key == "fsmonitor":
                return vector(
                    "core.fsmonitor",
                    "audit or remove the core.fsmonitor setting in .git/config",
                )
            if section == "filter" and key in {"clean", "process"}:
                return vector(
                    f"filter.{key}",
                    "audit or remove the [filter] command in .git/config",
                )
    if subcommand in _GIT_READER_SUBCOMMANDS:
        for section, key, value in entries:
            if not _git_value_is_active(value):
                continue
            if section == "diff" and key in {"command", "external", "textconv"}:
                return vector(f"diff.{key}")
    if subcommand == "log":
        for section, key, value in entries:
            if not _git_value_is_active(value):
                continue
            # Remerge output can invoke repository-owned merge drivers, and a
            # signature verifier runs the configured gpg program.
            if section == "merge" and key == "driver":
                return vector("merge.driver")
            if section == "gpg" and key == "program":
                return vector("gpg.program")
    if risk := _git_index_hook_risk(cwd, entries):
        return risk
    return None


_SED_TABLE = _OptionTable(
    long_values=frozenset({"--expression", "--file", "--line-length"}),
    short_values=frozenset("efl"),
    optional_long_values=frozenset({"--in-place"}),
    optional_short_values=frozenset("i"),
    boolean_long=frozenset({
        "--debug",
        "--follow-symlinks",
        "--help",
        "--null-data",
        "--posix",
        "--quiet",
        "--regexp-extended",
        "--sandbox",
        "--separate",
        "--silent",
        "--unbuffered",
        "--version",
    }),
)


def _sed_delimited_end(script: str, index: int, delimiter: str) -> int:
    while index < len(script):
        if script[index] == "\\":
            index += 2
        elif script[index] == delimiter:
            return index + 1
        else:
            index += 1
    raise ValueError("unterminated sed expression")


def _sed_script_paths(script: str) -> tuple[str, ...]:  # noqa: PLR0912
    """Inspect literal file commands without interpreting substitutions as code."""
    paths: list[str] = []
    index = 0
    while index < len(script):
        character = script[index]
        if character.isspace() or character in ";{},!+$" or character.isdigit():
            # `+` is the GNU relative-address joiner (`,+N`), the form the
            # statically expanded locate-then-print idiom produces. A `+`
            # outside an address is a sed runtime error, never a write, and
            # this scan continues linearly, so any later file command is
            # still found. The `first~step` address stays unmodeled.
            index += 1
            continue
        if character == "/":  # Regex address.
            index = _sed_delimited_end(script, index + 1, "/")
            continue
        # Alternate address delimiter.
        if character == "\\" and index + 1 < len(script):
            delimiter = script[index + 1]
            index = _sed_delimited_end(script, index + 2, delimiter)
            continue
        index += 1
        if character in "sy":
            if index >= len(script):
                raise ValueError("missing sed delimiter")
            delimiter = script[index]
            index = _sed_delimited_end(script, index + 1, delimiter)
            index = _sed_delimited_end(script, index, delimiter)
            if character == "y":
                continue
            # Substitution flags include a file-writing 'w' selector.
            while index < len(script) and script[index] not in ";\n}#":
                if script[index] == "w":
                    character = "w"
                    index += 1
                    break
                if script[index] not in "0123456789giImMp \t\r":
                    raise ValueError(
                        "unmodeled sed substitution flag (shell execution is not permitted)"
                    )
                index += 1
            else:
                continue
        if character in "rRwW":
            # GNU sed consumes the rest of the line as the filename, even ';'.
            end = script.find("\n", index)
            end = len(script) if end == -1 else end
            target = script[index:end].lstrip(" \t")
            if "\\" in target:
                raise ValueError("escaped sed filename cannot be safely inspected")
            if target:
                paths.append(target)
            index = end
        elif character in "aic#":
            # Text and comments are not sed commands.
            end = script.find("\n", index)
            index = len(script) if end == -1 else end
        elif character in "btT:":
            while index < len(script) and script[index] not in ";\n":
                index += 1
        elif character not in "dDgGhHlnNpPqx=z":
            # Unknown syntax may hide a file command; do not guess its boundary.
            raise ValueError("unsupported sed script syntax")
    return tuple(paths)


def _sed_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _SED_TABLE)
    has_script_option = _has_long(parsed, "--expression", "--file") or _has_short(
        parsed, frozenset("ef"), _SED_TABLE
    )
    inputs = parsed.positionals if has_script_option else parsed.positionals[1:]
    scripts = parsed.values_for("--expression", "-e")
    if not has_script_option:
        scripts = parsed.positionals[:1]
    if _has_long(parsed, "--file") or _has_short(parsed, frozenset("f"), _SED_TABLE):
        return ShellCommandPolicy(
            requires_approval=True,
            denial_reason="external sed scripts cannot be analyzed; inline the script instead",
        )
    paths: list[str] = []
    requires_approval = False
    denial_reason = None
    try:
        for script in scripts:
            paths.extend(_sed_script_paths(script))
    except ValueError as exc:
        requires_approval = True
        denial_reason = str(exc)
    for suffix in parsed.values_for("--in-place", "-i"):
        if "/" in suffix:
            requires_approval = True
        paths.extend(
            suffix.replace("*", operand) if "*" in suffix else operand + suffix
            for operand in inputs
            if suffix
        )
    return ShellCommandPolicy(
        requires_approval=requires_approval,
        denial_reason=denial_reason,
        mutates_files=_has_long(parsed, "--in-place")
        or _has_short(parsed, frozenset("i"), _SED_TABLE),
        inspect_positional_paths=True,
        option_path_values=tuple(paths),
        positional_values=inputs,
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


def _package_policy(args: list[str], *, pip: bool = False) -> ShellCommandPolicy:
    table = _OptionTable(
        long_values=_PACKAGE_PATH_OPTIONS | frozenset({"--python"}),
        short_values=frozenset("tercdwo") if pip else frozenset("o"),
    )
    parsed = _scan_options(args, table)
    try:
        option_paths = tuple(
            _package_local_path(value)
            for value in parsed.values_for(
                *_PACKAGE_PATH_OPTIONS, "-t", "-e", "-r", "-c", "-d", "-w", "-o"
            )
        )
        positional_paths = tuple(
            path
            for value in parsed.positionals
            if (path := _package_requirement_path(value)) is not None
        )
    except ValueError:
        return ShellCommandPolicy(
            requires_approval=True, denial_reason="unmodeled local package URL"
        )
    return ShellCommandPolicy(
        inspect_positional_paths=True,
        option_path_values=option_paths,
        positional_values=positional_paths,
    )


# Compilers and linkers whose -o/--output operand is the written program: the
# output file is a write-classified path operand, so it is workspace-scoped and
# checked against protected Git metadata like every other write operand. The
# assembler (as) writes its object through -o as well; it spells no long form
# (--output would abbreviate --output-target), so the modeled long form is
# inert for it.
_COMPILER_OUTPUT_COMMANDS = frozenset({
    "gcc",
    "cc",
    "clang",
    "g++",
    "c++",
    "clang++",
    "rustc",
    "ld",
    "as",
})
_COMPILER_OUTPUT_TABLE = _OptionTable(
    long_values=frozenset({"--output"}), short_values=frozenset("o")
)


def _compiler_output_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _COMPILER_OUTPUT_TABLE)
    return ShellCommandPolicy(option_path_values=parsed.values_for("--output", "-o"))


# Write-operand options of the commands whose positional operands are already
# write-classified (the mutating path commands and the file-content commands).
# Every modeled option names a file or directory the command writes, so each
# extracted value is a write-classified path operand — in the separate form
# (-C .git), the attached short form (-C.git), and the attached long form
# (--output=.git/config) alike. Options that name read targets (tar -T,
# patch -i, rsync --exclude-from) are deliberately absent: only write
# operands are extracted here.
_WRITE_OPERAND_TABLES: dict[str, _OptionTable] = {
    # cp/mv/install/ln -t/--target-directory names the destination directory.
    "cp": _OptionTable(
        long_values=frozenset({"--target-directory"}), short_values=frozenset({"t"})
    ),
    "mv": _OptionTable(
        long_values=frozenset({"--target-directory"}), short_values=frozenset({"t"})
    ),
    "install": _OptionTable(
        long_values=frozenset({"--target-directory"}), short_values=frozenset({"t"})
    ),
    "ln": _OptionTable(
        long_values=frozenset({"--target-directory"}), short_values=frozenset({"t"})
    ),
    # tee --output-error[=MODE] carries the write-error mode; GNU's modes
    # (warn, exit, ...) are not paths, so extracting the attached value is
    # conservative over-blocking, never a missed write operand.
    "tee": _OptionTable(optional_long_values=frozenset({"--output-error"})),
    # patch writes the patched result (-o/--output), the rejects
    # (-r/--reject-file), and runs under a relocated directory
    # (-d/--directory); backup names are prefixed with -B/--prefix and
    # -Y/--basename-prefix.
    "patch": _OptionTable(
        long_values=frozenset({
            "--basename-prefix",
            "--directory",
            "--output",
            "--prefix",
            "--reject-file",
        }),
        short_values=frozenset({"B", "Y", "d", "o", "r"}),
    ),
    # tar -f/--file names the archive (written when creating, read when
    # extracting — the operand is write-classified either way, matching the
    # positional scan), -C/--directory the extraction destination, and
    # --add-file a member added to the archive.
    "tar": _OptionTable(
        long_values=frozenset({"--add-file", "--directory", "--file"}),
        short_values=frozenset({"C", "f"}),
    ),
    # unzip -d names the extraction destination.
    "unzip": _OptionTable(short_values=frozenset({"d"})),
    # iconv -o/--output names the written conversion result.
    "iconv": _OptionTable(
        long_values=frozenset({"--output"}), short_values=frozenset({"o"})
    ),
    # gpg's -o/--output names the written output file (the decrypted or
    # signed result): every spelling is real, including the attached short
    # form (-oFILE) and --output=FILE.
    "gpg": _OptionTable(
        long_values=frozenset({"--output"}), short_values=frozenset({"o"})
    ),
    # xmllint's -o/--output names the written XML result. The attached
    # spellings (-oFILE, --output=FILE) are rejected by xmllint itself, so
    # extracting them is a conservative superset, never a missed operand.
    "xmllint": _OptionTable(
        long_values=frozenset({"--output"}), short_values=frozenset({"o"})
    ),
    # rsync's directory/file options name where transferred, partial,
    # backup, and compared files and the log are written.
    "rsync": _OptionTable(
        long_values=frozenset({
            "--backup-dir",
            "--compare-dest",
            "--copy-dest",
            "--link-dest",
            "--log-file",
            "--partial-dir",
            "--temp-dir",
        }),
        short_values=frozenset({"T"}),
    ),
    # bsdtar is tar-compatible: -f/--file names the archive (written when
    # creating, read when extracting — write-classified either way, matching
    # tar) and -C/--directory the extraction destination.
    "bsdtar": _OptionTable(
        long_values=frozenset({"--directory", "--file"}),
        short_values=frozenset({"C", "f"}),
    ),
    # pax -f names the archive, written in write mode (-w) and read in read
    # mode (-r); write-classified either way, matching tar.
    "pax": _OptionTable(short_values=frozenset({"f"})),
    # ar's positional archive operand is created or replaced by the mutating
    # operations (inspected positionally, like tar's), and --output names
    # the extraction destination directory.
    "ar": _OptionTable(long_values=frozenset({"--output"})),
}


def _write_operand_policy(
    table: _OptionTable,
) -> Callable[[list[str]], ShellCommandPolicy]:
    """One command's write-operand extraction over its modeled option table.

    The table carries only write-operand options, so every value the scan
    consumes is a write-classified path operand. Unknown options never
    consume a following token, so a misread option consumption can only
    leave an extra candidate, never hide a write operand.
    """

    def policy(args: list[str]) -> ShellCommandPolicy:
        parsed = _scan_options(args, table)
        return ShellCommandPolicy(
            option_path_values=tuple(value for _, value in parsed.values)
        )

    return policy


def _openssl_output_policy(args: list[str]) -> ShellCommandPolicy:
    """openssl's -out/--out operand names the written output file.

    OpenSSL spells its output option with one or two dashes and accepts the
    value separately or attached with ``=`` (``-out FILE``, ``-out=FILE``,
    ``--out FILE``, ``--out=FILE``), which the short-cluster scanner cannot
    model; extract every spelling directly. Only the exact ``-out``/``--out``
    prefixes match, so longer options (``-outform``) never do.
    """
    values: list[str] = []
    for index, token in enumerate(args):
        if token in {"-out", "--out"} and index + 1 < len(args):
            values.append(args[index + 1])
        elif token.startswith(("-out=", "--out=")):
            values.append(token.partition("=")[2])
    return ShellCommandPolicy(option_path_values=tuple(values))


# split's value-taking options keep their operands (a SIZE, a SEPARATOR) out
# of the positional scan: only the trailing PREFIX operand names the written
# output files (PREFIXaa, PREFIXab, ...).
_SPLIT_TABLE = _OptionTable(
    long_values=frozenset({
        "--additional-suffix",
        "--bytes",
        "--filter",
        "--line-bytes",
        "--lines",
        "--number",
        "--separator",
        "--suffix-length",
    }),
    short_values=frozenset({"C", "a", "b", "l", "n", "t"}),
    optional_long_values=frozenset({"--hex-suffixes", "--numeric-suffixes"}),
)


def _split_output_policy(args: list[str]) -> ShellCommandPolicy:
    """split's PREFIX operand names the written output files.

    ``split [OPTION]... [INPUT [PREFIX]]`` writes PREFIXaa, PREFIXab, ...;
    with a single positional the input is read and the prefix defaults to
    ``x`` in the working directory, so only a second positional is a write
    target. The input operand is a read and stays governed by the workspace
    authority alone, exactly like every other reader operand.
    """
    parsed = _scan_options(args, _SPLIT_TABLE)
    positionals = parsed.positionals
    prefix = (positionals[-1],) if len(positionals) > 1 else ()
    return ShellCommandPolicy(inspect_positional_paths=True, positional_values=prefix)


# csplit's --prefix/-f operand names the written output files (PREFIX00,
# PREFIX01, ...); the positional operands are the input file and match
# patterns, not write targets.
_CSPLIT_TABLE = _OptionTable(
    long_values=frozenset({"--prefix"}), short_values=frozenset({"f"})
)


def _csplit_output_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _CSPLIT_TABLE)
    return ShellCommandPolicy(option_path_values=parsed.values_for("--prefix", "-f"))


# cpio's write operands: the -D/--directory directory the copy-in and
# pass-through modes change into before writing (the extraction/copy
# destination root), the -F/--file and -O archive files of copy-out mode,
# and the pass-through destination positional (modeled by _cpio_policy).
# Member names inside an archive stay uninspected (the documented member
# residual, same as tar). Options that name read targets (-I's archive, -E's
# pattern file) and non-path values (-H's format, -M's message, -W's flag,
# -C's io size, --block-size) are consumed but never extracted; cpio's
# optional-value -R/--owner is deliberately unmodeled, so its value stays a
# positional candidate — an unmodeled option can only over-block, never
# hide a write operand.
_CPIO_TABLE = _OptionTable(
    long_values=frozenset({
        "--block-size",
        "--directory",
        "--file",
        "--format",
        "--io-size",
        "--message",
        "--pattern-file",
        "--warning",
    }),
    short_values=frozenset({"C", "D", "E", "F", "H", "I", "M", "O", "W"}),
    boolean_long=frozenset({
        "--absolute-filenames",
        "--append",
        "--dereference",
        "--device-independent",
        "--dot",
        "--extract",
        "--force-local",
        "--ignore-devno",
        "--ignore-dirnlink",
        "--link",
        "--list",
        "--make-directories",
        "--no-absolute-filenames",
        "--no-preserve-owner",
        "--null",
        "--numeric-uid-gid",
        "--only-verify-crc",
        "--pass-through",
        "--preserve-modification-time",
        "--quiet",
        "--renumber-inodes",
        "--reproducible",
        "--reset-access-time",
        "--swap-bytes",
        "--swap-halfwords",
        "--to-stdout",
        "--unconditional",
        "--verbose",
    }),
)


def _cpio_policy(args: list[str]) -> ShellCommandPolicy:
    """cpio's write operands: pass-through destinations and archive files.

    Copy-pass mode (-p/--pass-through, -l/--link) copies the files named on
    standard input into its positional destination directory, so every
    positional is a write operand. Copy-in mode (-i) extracts archive members
    into the -D/--directory directory (or the cwd without it — the member
    residual), and copy-out mode (-o) writes the archive to -F/--file or -O.
    """
    parsed = _scan_options(args, _CPIO_TABLE)
    write_values = list(parsed.values_for("--directory", "-D", "--file", "-F", "-O"))
    passthrough = _has_long(parsed, "--pass-through", "--link") or _has_short(
        parsed, frozenset({"p", "l"}), _CPIO_TABLE
    )
    if passthrough:
        # Copy-pass mode's positional operand is the destination directory.
        return ShellCommandPolicy(
            inspect_positional_paths=True,
            option_path_values=tuple(write_values),
            positional_values=parsed.positionals,
        )
    return ShellCommandPolicy(
        option_path_values=tuple(write_values), positional_values=()
    )


# strip writes its result to -o when given and mutates every positional
# operand in place otherwise, so both are write operands: strip preserves
# the executable bit, so `strip -o .git/hooks/pre-commit evil` plants a
# hook exactly like the modeled `ld -o`. strip spells no long --output
# (--output abbreviates --output-target, which the table models so the
# abbreviation consumes its operand like strip itself), and its other
# value-taking options name formats, sections, and symbols — never files
# the command writes — so they are consumed but never extracted.
_STRIP_TABLE = _OptionTable(
    long_values=frozenset({
        "--input-target",
        "--keep-section",
        "--keep-symbol",
        "--output-target",
        "--remove-relocations",
        "--remove-section",
        "--strip-symbol",
        "--target",
    }),
    short_values=frozenset({"F", "I", "K", "N", "O", "R", "o"}),
)


def _strip_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _STRIP_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True,
        option_path_values=parsed.values_for("-o"),
        positional_values=parsed.positionals,
    )


# scp's grammar is `scp [options] source... destination`, where an operand
# carrying a colon before any slash is a REMOTE spec (outside this guard:
# it writes a remote host, never local metadata). A local-to-local copy is
# a mutating path command like ln: BOTH operands are write-classified and
# inspected, so a destination resolving into metadata is denied on
# containment and a source resolving into metadata is denied the same way
# (the ln rule). The value-taking options are modeled so their operands
# (ports, identity files, ssh options) never surface as positional
# candidates.
_SCP_TABLE = _OptionTable(
    short_values=frozenset({"c", "D", "F", "i", "J", "l", "o", "P", "S"})
)


def _scp_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _SCP_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True, positional_values=parsed.positionals
    )


# zip's first positional operand is the archive the call creates or
# replaces; the remaining positionals name the files it stores (reads,
# but classified like cp's source operand: a mutating path command
# inspects every operand). The value-taking options (the -b scratch
# directory, -i/-x patterns, -n suffixes, -P password) are modeled so
# their operands never surface as positional candidates.
_ZIP_TABLE = _OptionTable(short_values=frozenset({"P", "b", "i", "n", "x"}))


def _zip_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _ZIP_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True, positional_values=parsed.positionals
    )


# 7z's grammar is `7z <command> [switches] <archive> [files...]`: every
# positional operand is write-classified (the archive is created or
# replaced by the mutating commands and read by the listing ones —
# write-classified either way, matching tar's -f; the command letter and
# the stored-file operands are classified like cp's source operand).
# The -o switch names the extraction destination and is ATTACHED-ONLY in
# 7z (`-o.git` extracts into .git; `-o .git` is rejected by 7z itself, and
# `-o=.git` creates a directory literally named "=.git"), so it is an
# optional short value: the attached spelling extracts, and a separate
# value stays a positional candidate the scan still inspects.
_SEVENZIP_TABLE = _OptionTable(optional_short_values=frozenset({"o"}))


def _sevenzip_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _SEVENZIP_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True,
        option_path_values=parsed.values_for("-o"),
        positional_values=parsed.positionals,
    )


# unar extracts into -o's directory (default: the current directory) and
# spells the option in four forms: `-o DIR`, `-o=DIR`, and the single-dash
# long `-output-directory DIR` / `-output-directory=DIR`. The attached
# short form (`-oDIR`) is rejected by unar itself; the scanner's cluster
# rules cannot model single-dash long options, so the forms are extracted
# directly. Only these options name files unar writes: the positional
# operands name the archive and the members to extract (reads).
def _unar_output_policy(args: list[str]) -> ShellCommandPolicy:
    values: list[str] = []
    for index, token in enumerate(args):
        if token in {"-o", "-output-directory"} and index + 1 < len(args):
            values.append(args[index + 1])
        elif token.startswith(("-o=", "-output-directory=")):
            values.append(token.partition("=")[2])
    return ShellCommandPolicy(option_path_values=tuple(values))


# zstd (and its twins zstdmt/unzstd/zstdcat, which are the same binary)
# mutates every positional operand in place — bzip2/xz/lzma style — and
# additionally writes to -o's file (separate form only) and to the
# --output-dir-flat/--output-dir-mirror directories, in both the `=` and
# the separate spelling, so those options are extracted as write
# operands. The remaining value-taking options (levels, sizes, formats)
# name no written file and only ever over-block as leaked candidates.
_ZSTD_TABLE = _OptionTable(
    long_values=frozenset({"--output-dir-flat", "--output-dir-mirror"}),
    short_values=frozenset({"o"}),
)


def _zstd_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _ZSTD_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True,
        option_path_values=parsed.values_for(
            "-o", "--output-dir-flat", "--output-dir-mirror"
        ),
        positional_values=parsed.positionals,
    )


# mktemp creates a file at every template positional and in the directory
# -p/--tmpdir names (the long form takes its directory only attached:
# --tmpdir=DIR). The resolver's file-content surface inspects the
# template positionals; this policy adds the -p/--tmpdir directories,
# whose separate spelling also surfaces as a positional candidate and
# whose attached spelling only the scan can reach. --suffix names no
# directory (a slash in it fails), so it is consumed but not extracted.
_MKTEMP_TABLE = _OptionTable(
    long_values=frozenset({"--suffix"}),
    short_values=frozenset({"p"}),
    optional_long_values=frozenset({"--tmpdir"}),
)


def _mktemp_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _MKTEMP_TABLE)
    return ShellCommandPolicy(option_path_values=parsed.values_for("-p", "--tmpdir"))


_COMMAND_POLICIES = {
    "chmod": _chmod_policy,
    "chown": _chown_policy,
    "date": lambda args: _standard_policy("date", args),
    "diff": lambda args: _standard_policy("diff", args),
    "du": lambda args: _standard_policy("du", args),
    "file": lambda args: _standard_policy("file", args),
    "find": _find_policy,
    "git": _git_policy,
    "gsed": _sed_policy,
    "grep": lambda args: _standard_policy("grep", args),
    "less": _less_policy,
    "md5sum": _checksum_policy,
    "more": _less_policy,
    "rm": _rm_policy,
    "sed": _sed_policy,
    "sha1sum": _checksum_policy,
    "sha256sum": _checksum_policy,
    "shasum": _checksum_policy,
    "sort": lambda args: _standard_policy("sort", args),
    "tree": lambda args: _standard_policy("tree", args),
    "uniq": _uniq_policy,
    "wc": lambda args: _standard_policy("wc", args),
    "xargs": lambda args: ShellCommandPolicy(requires_approval=True),
}
_COMMAND_POLICIES.update({
    name: _compiler_output_policy for name in _COMPILER_OUTPUT_COMMANDS
})
_COMMAND_POLICIES.update({
    name: _write_operand_policy(table) for name, table in _WRITE_OPERAND_TABLES.items()
})
_COMMAND_POLICIES["openssl"] = _openssl_output_policy
_COMMAND_POLICIES["split"] = _split_output_policy
_COMMAND_POLICIES["csplit"] = _csplit_output_policy
_COMMAND_POLICIES["cpio"] = _cpio_policy
_COMMAND_POLICIES["strip"] = _strip_policy
_COMMAND_POLICIES["scp"] = _scp_policy
_COMMAND_POLICIES["zip"] = _zip_policy
_COMMAND_POLICIES.update({name: _sevenzip_policy for name in ("7z", "7za", "7zr")})
_COMMAND_POLICIES["unar"] = _unar_output_policy
_COMMAND_POLICIES.update({
    name: _zstd_policy for name in ("zstd", "zstdcat", "zstdmt", "unzstd")
})
_COMMAND_POLICIES["mktemp"] = _mktemp_policy


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


def analyze_shell_command_policy(tokens: list[str]) -> ShellCommandPolicy:
    boundary = executor_boundary(tokens)
    if boundary and boundary.module_command:
        # Expanded module argv is inspected as its own command, with its own
        # executor boundary. The interpreter does not own that child's source.
        return ShellCommandPolicy()
    if (
        boundary
        and boundary.consumed_ranges
        and (grammar := EXECUTOR_REGISTRY.get(_command_name(tokens[0])))
        and (grammar.keyword or grammar.package_executable)
    ):
        return _executor_package_policy(tokens, boundary)
    tokens = _package_module_tokens(executor_owned_tokens(tokens))
    if not tokens:
        return ShellCommandPolicy()
    name = _command_name(tokens[0])
    if name in _PACKAGE_PATH_COMMANDS:
        return _package_policy(
            tokens[1:],
            pip=name in {"pip", "pip3"} or (name == "uv" and "pip" in tokens[1:]),
        )
    policy = _COMMAND_POLICIES.get(name)
    return policy(tokens[1:]) if policy else ShellCommandPolicy()


def _command_operand_paths(command: str, tokens: list[str]) -> list[str]:
    """Native ``keyword=value`` operands of dd that name files.

    dd's if=/of= operands name its input and output files. Other commands'
    write operands come from their option tables (see _WRITE_OPERAND_TABLES
    and the git policies), which extract separate, attached-short, and
    attached-long spellings alike.
    """
    if command == "dd":
        return [token[3:] for token in tokens[1:] if token.startswith(("if=", "of="))]
    return []


def path_candidates(
    tokens: list[str],
    *,
    inspect_positional_paths: bool,
    inspect_executable: bool = False,
) -> tuple[str, ...]:
    if not tokens:
        return ()
    executable_paths = (
        [tokens[0]] if "/" in tokens[0] or tokens[0].startswith("~") else []
    )
    boundary = executor_boundary(tokens)
    if boundary is not None and (
        boundary.inner_command_range is not None
        or boundary.module_command
        or boundary.inline_payload_ranges
        or boundary.native_targets
    ):
        owned = executor_owned_tokens(tokens)
        policy = analyze_shell_command_policy(tokens)
        # Compiled targets and interpreter scripts are files/packages, never
        # commands. Only the executor's own operands are inspected at this layer.
        # Registered outer wrappers retain their legacy executable treatment;
        # an executable selected inside another executor is always inspected.
        candidates = [
            *(executable_paths if inspect_executable else []),
            *policy.option_path_values,
            *policy.write_path_values,
        ]
        candidates.extend(policy.positional_values or ())
        candidates.extend(boundary.cwd_overrides)
        if boundary.native_targets and not is_package_command(owned):
            candidates.extend(boundary.native_targets)
        return tuple(dict.fromkeys(candidates))
    policy = analyze_shell_command_policy(tokens)
    candidates = [
        *executable_paths,
        *policy.option_path_values,
        *policy.write_path_values,
    ]
    if not (inspect_positional_paths or policy.inspect_positional_paths):
        return tuple(candidates)
    command = _command_name(tokens[0])
    candidates.extend(_command_operand_paths(command, tokens))
    if policy.positional_values is not None:
        candidates.extend(policy.positional_values)
        return tuple(candidates)
    options_ended = False
    for token in tokens[1:]:
        if token == "--":
            options_ended = True
            continue
        if not options_ended and token.startswith("-"):
            continue
        if command == "chmod" and token.startswith("+"):
            continue
        candidates.append(token)
    return tuple(candidates)
