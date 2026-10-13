"""Per-command argument-position table for static shell expansion.

`_shell_static_expansion` resolves a variable/arithmetic expansion ONLY when
the argument position that consumes the resolved value is classified
read-only by this table. A position is read-only when an expanded value can
only be consumed as data by a command that cannot write files or execute
anything — never as the script or address of a writing construct.

Position keys:

- ``option-arg:<option>`` — the value consumed by that option, spelled as the
  option appears in the command (e.g. ``option-arg:-n``).
- ``positional:<n>`` — the n-th non-option argument, 1-based.
- ``positional:<n>+`` — the n-th and every later non-option argument.

The classification is per command, not per option spelling: a command is
read-only in a position only if NO writing construct exists anywhere in that
command's grammar. Commands with writing constructs (sed: ``w``/``W`` script
commands, the ``s///w`` flag, ``-i`` in-place editing) get no read-only
positions at all — their script positions are miniature languages and their
remaining positions are conservatively unclassified. Unknown commands and
unknown positions classify as ``None`` / not read-only (fail closed).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

__all__ = [
    "ARGUMENT_POSITION_RULES",
    "ArgumentPositionRule",
    "classify_argument_position",
    "is_read_only_argument_position",
    "read_only_positions",
]

_POSITIONAL_OPEN_KEY = re.compile(r"positional:(\d+)\+")
_POSITIONAL_INDEX_KEY = re.compile(r"positional:(\d+)")


@dataclass(frozen=True)
class ArgumentPositionRule:
    """One classified argument position of a whitelisted command."""

    command: str
    position: str
    read_only: bool
    rationale: str


def _read_only(command: str, position: str, rationale: str) -> ArgumentPositionRule:
    return ArgumentPositionRule(command, position, True, rationale)


def _not_read_only(command: str, position: str, rationale: str) -> ArgumentPositionRule:
    return ArgumentPositionRule(command, position, False, rationale)


_RULES: tuple[ArgumentPositionRule, ...] = (
    # --- cat: a pure reader -------------------------------------------------
    _read_only(
        "cat",
        "positional:1+",
        "cat has no output-path option and no script grammar; it opens every"
        " operand for reading and writes only to its own stdout. An expanded"
        " value can only select what is read, never cause a write.",
    ),
    # --- wc: a pure reader --------------------------------------------------
    _read_only(
        "wc",
        "positional:1+",
        "wc only counts the bytes, words, and lines of its operands; it has no"
        " output-path option and no script grammar, so every operand is"
        " consumed as a read path.",
    ),
    _read_only(
        "wc",
        "option-arg:--files0-from",
        "the value names a file wc opens for reading (a NUL-separated list of"
        " input paths). wc has no writing construct, so the value is consumed"
        " as read-only data even though it steers which files are read.",
    ),
    # --- head: a pure reader ------------------------------------------------
    _read_only(
        "head",
        "option-arg:-n",
        "the value is parsed solely as an unsigned line count. head has no"
        " output-path option, no script grammar, and no in-place mode; a"
        " malformed count is a runtime error, and any value is pure data.",
    ),
    _read_only(
        "head",
        "option-arg:--lines",
        "long spelling of -n; the value is parsed solely as an unsigned line"
        " count by a command with no writing construct.",
    ),
    _read_only(
        "head",
        "option-arg:-c",
        "the value is parsed solely as an unsigned byte count; head has no"
        " output-path option and no in-place mode, so any value is pure"
        " data.",
    ),
    _read_only(
        "head",
        "option-arg:--bytes",
        "long spelling of -c; the value is parsed solely as an unsigned byte"
        " count by a command with no writing construct.",
    ),
    _read_only(
        "head",
        "positional:1+",
        "operands are opened for reading only; head cannot write files and"
        " cannot execute code, so an expanded operand only selects input.",
    ),
    # --- tail: a pure reader ------------------------------------------------
    _read_only(
        "tail",
        "option-arg:-n",
        "the value is parsed solely as an unsigned line count. tail has no"
        " output-path option, no script grammar, and no in-place mode; a"
        " malformed count is a runtime error, and any value is pure data.",
    ),
    _read_only(
        "tail",
        "option-arg:--lines",
        "long spelling of -n; the value is parsed solely as an unsigned line"
        " count by a command with no writing construct.",
    ),
    _read_only(
        "tail",
        "option-arg:-c",
        "the value is parsed solely as an unsigned byte count; tail has no"
        " output-path option and no in-place mode, so any value is pure"
        " data.",
    ),
    _read_only(
        "tail",
        "option-arg:--bytes",
        "long spelling of -c; the value is parsed solely as an unsigned byte"
        " count by a command with no writing construct.",
    ),
    _read_only(
        "tail",
        "positional:1+",
        "operands are opened for reading only; tail cannot write files and"
        " cannot execute code, so an expanded operand only selects input.",
    ),
    # --- grep: a pure reader ------------------------------------------------
    _read_only(
        "grep",
        "positional:1",
        "the value is compiled as a regular expression and used only to"
        " filter input. grep has no output-path option and no command or"
        " script grammar; regex metacharacters can change what is matched"
        " but can never redirect control into a writing construct.",
    ),
    _read_only(
        "grep",
        "positional:2+",
        "operands are opened for reading only; grep cannot write files and"
        " cannot execute code, so an expanded operand only selects input.",
    ),
    _read_only(
        "grep",
        "option-arg:-m",
        "the value is parsed solely as a maximum match count; grep has no"
        " writing construct, so the value is pure data.",
    ),
    _read_only(
        "grep",
        "option-arg:--max-count",
        "long spelling of -m; the value is parsed solely as a maximum match"
        " count by a command with no writing construct.",
    ),
    _read_only(
        "grep",
        "option-arg:-A",
        "the value is parsed solely as an after-context line count; grep has"
        " no writing construct, so the value is pure data.",
    ),
    _read_only(
        "grep",
        "option-arg:--after-context",
        "long spelling of -A; the value is parsed solely as an after-context"
        " line count by a command with no writing construct.",
    ),
    _read_only(
        "grep",
        "option-arg:-B",
        "the value is parsed solely as a before-context line count; grep has"
        " no writing construct, so the value is pure data.",
    ),
    _read_only(
        "grep",
        "option-arg:--before-context",
        "long spelling of -B; the value is parsed solely as a before-context"
        " line count by a command with no writing construct.",
    ),
    _read_only(
        "grep",
        "option-arg:-C",
        "the value is parsed solely as a context line count; grep has no"
        " writing construct, so the value is pure data.",
    ),
    _read_only(
        "grep",
        "option-arg:--context",
        "long spelling of -C; the value is parsed solely as a context line"
        " count by a command with no writing construct.",
    ),
    # --- cut: a pure reader -------------------------------------------------
    _read_only(
        "cut",
        "option-arg:-d",
        "the value is the output-field delimiter, consumed as a pure data"
        " filter. cut has no output-path option and no script grammar; it"
        " only reads its operands and writes to its own stdout.",
    ),
    _read_only(
        "cut",
        "option-arg:--delimiter",
        "long spelling of -d; the value is the output-field delimiter,"
        " consumed as pure data by a command with no writing construct.",
    ),
    _read_only(
        "cut",
        "option-arg:-f",
        "the value is a field-list filter, consumed as pure data. cut has no"
        " output-path option and no script grammar, so any value only"
        " changes which parts of the input are echoed.",
    ),
    _read_only(
        "cut",
        "option-arg:--fields",
        "long spelling of -f; the value is a field-list filter, consumed as"
        " pure data by a command with no writing construct.",
    ),
    _read_only(
        "cut",
        "positional:1+",
        "operands are opened for reading only; cut cannot write files and"
        " cannot execute code, so an expanded operand only selects input.",
    ),
    # --- sed: NO read-only positions ----------------------------------------
    _not_read_only(
        "sed",
        "positional:1",
        "the sed script is a miniature language whose w/W commands and"
        " s///w flag write files; an expanded value here is a script, not"
        " data. The only provably read-only sed script is the numeric-address"
        " print script statically expanded by _shell_static_expansion's sed"
        " idiom recognizer.",
    ),
    _not_read_only(
        "sed",
        "option-arg:-e",
        "the value is a sed script (same miniature write-capable language as"
        " the default script operand), so an expanded value is a script, not"
        " data.",
    ),
    _not_read_only(
        "sed",
        "option-arg:--expression",
        "long spelling of -e; the value is a write-capable sed script, not data.",
    ),
    _not_read_only(
        "sed",
        "option-arg:-f",
        "the value names a file whose CONTENT becomes the sed script, so a"
        " path-looking value is not consumed as data.",
    ),
    _not_read_only(
        "sed",
        "option-arg:--file",
        "long spelling of -f; the value names a file whose content becomes a"
        " write-capable sed script, not data.",
    ),
    _not_read_only(
        "sed",
        "positional:2+",
        "conservative: sed supports in-place editing (-i) and write commands,"
        " so no argument position of sed is classified read-only absent a"
        " whole-command read-only proof; the module-1 expansion path is the"
        " sanctioned route for sed scripts.",
    ),
)

ARGUMENT_POSITION_RULES: Mapping[str, Mapping[str, ArgumentPositionRule]] = {
    command: {rule.position: rule for rule in _RULES if rule.command == command}
    for command in dict.fromkeys(rule.command for rule in _RULES)
}
"""Command -> position key -> classification, with a per-entry rationale."""


def classify_argument_position(
    command: str, position: str
) -> ArgumentPositionRule | None:
    """Classify one argument position of a whitelisted command.

    ``position`` uses the key grammar documented in the module docstring.
    An open-ended ``positional:<n>+`` rule also covers every later
    positional index. Returns None for any command or position the table
    does not model (fail closed).
    """
    table = ARGUMENT_POSITION_RULES.get(command)
    if table is None:
        return None
    rule = table.get(position)
    if rule is not None:
        return rule
    index_match = _POSITIONAL_INDEX_KEY.fullmatch(position)
    if index_match is None:
        return None
    index = int(index_match.group(1))
    for key, open_rule in table.items():
        open_match = _POSITIONAL_OPEN_KEY.fullmatch(key)
        if open_match is not None and int(open_match.group(1)) <= index:
            return open_rule
    return None


def is_read_only_argument_position(command: str, position: str) -> bool:
    """True only when the position is modeled and classified read-only."""
    rule = classify_argument_position(command, position)
    return rule is not None and rule.read_only


def read_only_positions(command: str) -> frozenset[str]:
    """The position keys of ``command`` that are classified read-only."""
    table = ARGUMENT_POSITION_RULES.get(command)
    if table is None:
        return frozenset()
    return frozenset(key for key, rule in table.items() if rule.read_only)
