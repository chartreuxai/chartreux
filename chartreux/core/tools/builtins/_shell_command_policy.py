from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

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
    inspect_positional_paths: bool = False
    # The command is a Git reader whose repository-owned config may execute
    # code; the permission resolver inspects that config before permitting it.
    inspect_git_repository: bool = False
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
})

_GIT_PAGING_SUBCOMMANDS = _GIT_READER_SUBCOMMANDS


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


def _git_policy(args: list[str]) -> ShellCommandPolicy:
    if _git_destructive_policy(args):
        return ShellCommandPolicy(requires_approval=True)
    globals_result = _git_global_options(args)
    if globals_result is None:
        return ShellCommandPolicy(requires_approval=True)
    index, paths = globals_result
    if index >= len(args) or args[index] not in _GIT_READER_SUBCOMMANDS:
        return ShellCommandPolicy(option_path_values=tuple(paths))
    subcommand = args[index]
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
    option_paths.extend(parsed.values_for("--pathspec-from-file"))
    if subcommand in {"diff", "log"}:
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


_GIT_CONFIG_ENTRY = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9-]*)\s*(?:=\s*(?P<value>.*))?$"
)
_FALSE_GIT_CONFIG_VALUES = frozenset({"", "0", "false", "no", "off"})
# The section name is casefolded on parse; the remediation text restores the
# spelling git documents for the directive.
_INCLUDE_DIRECTIVE_NAMES = {"include": "include", "includeif": "includeIf"}
_GIT_SUBCOMMAND_INDEX = 1


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


def _git_config_entries(cwd: Path) -> tuple[tuple[str, str, str], ...] | None:
    """Parse repository-owned config entries without invoking Git.

    None means a repository governs ``cwd`` but its config could not be read;
    callers fail closed on that result.
    """
    config_paths = _git_config_paths(cwd)
    if config_paths is None:
        return None
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


def _git_value_is_active(value: str) -> bool:
    return value.casefold() not in _FALSE_GIT_CONFIG_VALUES


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
    return None


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
    "less": _less_policy,
    "md5sum": _checksum_policy,
    "more": _less_policy,
    "rm": _rm_policy,
    "sha1sum": _checksum_policy,
    "sha256sum": _checksum_policy,
    "shasum": _checksum_policy,
    "sort": lambda args: _standard_policy("sort", args),
    "tree": lambda args: _standard_policy("tree", args),
    "uniq": _uniq_policy,
    "wc": lambda args: _standard_policy("wc", args),
    "xargs": lambda args: ShellCommandPolicy(requires_approval=True),
}


# Only modeled options may advance the scan: an unknown option might consume
# the next word, hiding an inline-code selector behind it.
_INLINE_SWITCHES = {
    "python": ("c", frozenset("BEsIiuqvVxR"), frozenset("WX"), {}),
    "python3": ("c", frozenset("BEsIiuqvVxR"), frozenset("WX"), {}),
    "pypy": ("c", frozenset("BEsIiuqvVxR"), frozenset("WX"), {}),
    "pypy3": ("c", frozenset("BEsIiuqvVxR"), frozenset("WX"), {}),
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
    if not tokens or (name := _command_name(tokens[0])) not in _INLINE_SWITCHES:
        return None
    switch, flags, values, long_options = _INLINE_SWITCHES[name]
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


def analyze_shell_command_policy(tokens: list[str]) -> ShellCommandPolicy:
    if not tokens:
        return ShellCommandPolicy()
    policy = _COMMAND_POLICIES.get(_command_name(tokens[0]))
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
    command = _command_name(tokens[0])
    if command == "dd":
        candidates.extend(token[3:] for token in tokens[1:] if token.startswith("if="))
    if command == "tar":
        for index, token in enumerate(tokens[1:], 1):
            if token.startswith("--add-file="):
                candidates.append(token.partition("=")[2])
            elif token == "--add-file" and index + 1 < len(tokens):
                candidates.append(tokens[index + 1])
    if policy.positional_values is not None:
        candidates.extend(policy.positional_values)
        return tuple(candidates)
    options_ended = False
    sed_script = command == "sed" and not any(
        token in {"-e", "--expression", "-f", "--file"}
        or token.startswith(("--expression=", "--file=", "-e", "-f"))
        for token in tokens[1:]
    )
    for token in tokens[1:]:
        if token == "--":
            options_ended = True
            continue
        if not options_ended and token.startswith("-"):
            continue
        if command == "chmod" and token.startswith("+"):
            continue
        if sed_script:
            sed_script = False
            continue
        candidates.append(token)
    return tuple(candidates)
