"""The argv model, generic scanner, and normalization helpers for shell policy.

The option-table model (``_OptionTable``, ``_ParsedOptions``) and the generic
scanner (``_scan_options`` with its ``_has_*`` queries) are the hard W3b
prerequisites: the git support and classifier modules import them from here.
The argv normalization helpers (``_command_name``,
``_options_before_double_dash``, ``matches_command_prefix``) are pinned by
the golden corpus's ``argv_helpers`` family. ``_shell_command_policy`` (the
facade) re-exports every name defined here, so import paths and object
identity are unchanged by the split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Length of a single-character short option ("-i") and the minimum length of
# an unambiguous combined short-option cluster ("-ic").
_SHORT_OPTION_LENGTH = 2
_COMBINED_SHORT_OPTION_MIN_LENGTH = 3


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


def _options_before_double_dash(args: list[str]) -> list[str]:
    return args[: args.index("--")] if "--" in args else args


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
