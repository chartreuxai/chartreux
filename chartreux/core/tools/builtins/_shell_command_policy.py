from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ShellCommandPolicy:
    requires_approval: bool = False
    inspect_positional_paths: bool = False
    option_path_values: tuple[str, ...] = ()
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
    return tokens[1 : len(pattern_tokens)] == pattern_tokens[1:]


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
        frozenset("FILSW"),
    ),
    "du": _OptionTable(
        frozenset({
            "--block-size",
            "--exclude",
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
        frozenset({"--charset", "--filelimit", "--timefmt", "--sort", "--output"}),
        frozenset("HILPTo"),
    ),
    "less": _OptionTable(
        frozenset({"--color", "--lesskey-file"}),
        frozenset("Dkp"),
        frozenset({
            "--buffers",
            "--header",
            "--jump-target",
            "--LOG-FILE",
            "--log-file",
            "--max-back-scroll",
            "--pattern",
            "--prompt",
            "--quotes",
            "--shift",
            "--tabs",
            "--tag",
            "--tag-file",
            "--window",
        }),
        frozenset("bhjOPotTxyz"),
        boolean_long=frozenset({"--status-column", "--use-backslash", "--wordwrap"}),
        optional_long_values_consume_next=True,
    ),
}


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
        path_options = ("--file", "-f")
    elif command == "file":
        path_options = ("--files-from", "-f", "--magic-file", "-m")
    elif command in {"du", "wc"}:
        path_options = ("--files0-from",)
    elif command == "date":
        approval_long = ("--set",)
        approval_short = frozenset("s")
        path_options = ("--file", "-f", "--reference", "-r")
    elif command == "diff":
        path_options = ("--from-file", "--to-file")
    elif command == "less":
        approval_long = ("--log-file", "--LOG-FILE")
        approval_short = frozenset("oO")
    elif command == "tree":
        approval_long = ("--output",)
        approval_short = frozenset("o")
        inspect = True
    path_values = parsed.values_for(*path_options)
    if command == "file":
        path_values = tuple(
            path
            for option, value in parsed.values
            if option in path_options
            for path in value.split(":")
        )
    return ShellCommandPolicy(
        requires_approval=_has_long(parsed, *approval_long)
        or _has_short(parsed, approval_short, table),
        inspect_positional_paths=inspect,
        option_path_values=path_values,
        positional_values=parsed.positionals,
    )


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
        "--diff-filter",
        "--diff-algorithm",
        "--ignore-matching-lines",
        "--inter-hunk-context",
        "--line-prefix",
        "--max-count",
        "--output",
        "--output-indicator-context",
        "--output-indicator-new",
        "--output-indicator-old",
        "--rotate-to",
        "--skip-to",
        "--src-prefix",
        "--dst-prefix",
        "--stat-count",
        "--stat-graph-width",
        "--stat-name-width",
        "--word-diff-regex",
    }),
    frozenset("GLS"),
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
    boolean_long=frozenset({"--ext-diff", "--textconv"}),
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
            return ShellCommandPolicy(requires_approval=True)
    return ShellCommandPolicy()


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
        dry_run = any(
            token == "--dry-run"
            or (
                token.startswith("-")
                and not token.startswith("--")
                and "n" in token[1:]
            )
            for token in effective_options
        )
        force = any(
            token == "--force"
            or (
                token.startswith("-")
                and not token.startswith("--")
                and "f" in token[1:]
            )
            for token in effective_options
        )
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
    numeric_permissive = (
        bool(re.fullmatch(r"[0-7]{3,4}", mode)) and int(mode[-1]) & 2 != 0
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


def _git_policy(args: list[str]) -> ShellCommandPolicy:
    if _git_destructive_policy(args):
        return ShellCommandPolicy(requires_approval=True)
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
            return ShellCommandPolicy(requires_approval=True)
        if name in {"--config-env"} or token.startswith("-c"):
            return ShellCommandPolicy(requires_approval=True)
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
    if index >= len(args) or args[index] not in {"diff", "log", "show"}:
        return ShellCommandPolicy(option_path_values=tuple(paths))
    subcommand = args[index]
    parsed = _scan_options(args[index + 1 :], _GIT_SUBCOMMAND_TABLE)
    return ShellCommandPolicy(
        requires_approval=_has_long(parsed, "--output")
        or any(token in {"--ext-diff", "--textconv"} for token in parsed.options),
        inspect_positional_paths=subcommand == "diff"
        and "--no-index" in parsed.options,
        option_path_values=tuple(paths),
        positional_values=parsed.positionals,
    )


_COMMAND_POLICIES = {
    "chmod": _chmod_policy,
    "chown": _chown_policy,
    "date": lambda args: _standard_policy("date", args),
    "diff": lambda args: _standard_policy("diff", args),
    "du": lambda args: _standard_policy("du", args),
    "file": lambda args: _standard_policy("file", args),
    "find": _find_policy,
    "git": _git_policy,
    "grep": lambda args: _standard_policy("grep", args),
    "less": lambda args: _standard_policy("less", args),
    "rm": _rm_policy,
    "sort": lambda args: _standard_policy("sort", args),
    "tree": lambda args: _standard_policy("tree", args),
    "wc": lambda args: _standard_policy("wc", args),
    "xargs": lambda args: ShellCommandPolicy(requires_approval=True),
}


def analyze_shell_command_policy(tokens: list[str]) -> ShellCommandPolicy:
    if not tokens:
        return ShellCommandPolicy()
    policy = _COMMAND_POLICIES.get(tokens[0].rsplit("/", 1)[-1])
    return policy(tokens[1:]) if policy else ShellCommandPolicy()


def path_candidates(
    tokens: list[str], *, inspect_positional_paths: bool
) -> tuple[str, ...]:
    if not tokens:
        return ()
    policy = analyze_shell_command_policy(tokens)
    candidates = list(policy.option_path_values)
    if not (inspect_positional_paths or policy.inspect_positional_paths):
        return tuple(candidates)
    command = tokens[0].rsplit("/", 1)[-1]
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
