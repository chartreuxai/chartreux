"""Static expansion of whitelisted read-only shell idioms (tier 2, part 1).

This module recognizes the SHAPE of specific shell idioms and treats their
parts statically. It NEVER executes anything: no process-execution entry
point is imported or called here, and the companion test file proves it by
patching ``subprocess``/``os`` execution entry points to raise while every
recognition path runs. Reading the input files named by a recognized
construct IS allowed — reading is the read-only primitive the simulation
is built from.

Constructs:

(a) The sed locate-then-print idiom —
    ``sed -n "$(grep -n 'literal' F | head -1 | cut -d: -f1),+45p" F``.
    The inner pipeline is simulated statically over its whitelisted
    read-only stages, and the recognizer returns the FULLY EXPANDED outer
    sed script (address substituted numerically) for the caller to
    authorize under its normal policy.
(b) Zsh equals expansion (``=literal-command``) — the command name is
    resolved against PATH by stat/lookup only; unresolvable is not
    recognizable.
(c) Variable/arithmetic expansion evaluation in provably read-only
    argument positions, per the per-command table in
    ``_shell_argument_positions`` (module 2).

Everything not matching a modeled shape — non-literal inner content,
further substitution layers, redirects inside the construct, unknown
variables — is NOT recognizable and yields None (fail closed); the caller
denies or falls back to its normal policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import stat

from chartreux.core.tools.builtins._shell_argument_positions import (
    classify_argument_position,
)
from chartreux.core.tools.secret_redaction import redact

__all__ = [
    "SedAddressExpansion",
    "evaluate_expansion_for_position",
    "evaluate_static_expansion",
    "recognize_sed_locate_print",
    "resolve_equals_command",
    "sed_locate_print_source",
]

# --- shared strict-shape grammars --------------------------------------------

_HEADER_RE = re.compile(r'^sed\s+-n\s+"')
# What may follow the numeric address inside the outer sed script: a plain
# `p`, an absolute range `,Np`, or a GNU relative range `,+Np`. Nothing else —
# any other sed command (in particular any writing construct) is unmodeled.
_SUFFIX_RE = re.compile(r"^(?:,\+?\d+)?p$")
# Literal operand paths: conservative charset, and the caller-side checks
# reject option-like, glob, tilde, and equals-expansion word starts.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./+:@-]+$")
_SAFE_COMMAND_NAME_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
_DIGITS_RE = re.compile(r"\d+")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INT_VALUE_RE = re.compile(r"[+-]?\d+")
# grep -n patterns we simulate by Python substring matching must contain no
# POSIX BRE metacharacter, so substring matching is exactly grep's literal # typos:disable-line
# matching. Quoting and expansion-favored characters are excluded as well.
_PATTERN_FORBIDDEN = frozenset(".*[]^$\\'\"`")
# Variable values inlined by (c) must stay inert when the caller re-emits the
# command: no quoting, substitution, redirection, separator, or grouping
# characters (zsh never re-expands values — probed — but a re-emitted command
# string would be re-parsed, so this is defense in depth). Whitespace is
# forbidden as well: the one-argv-element guarantee is only sound for
# splitting-free values, and under a non-zsh executor a value like
# "AGENTS.md /etc/passwd" would split into unchecked operands (latent today
# because the executor is zsh with SH_WORD_SPLIT off, but denied anyway).
_VALUE_FORBIDDEN = frozenset("'\"$`\\<>&|;()\n\t \r\v\f")
# A `:` or `[` directly after a parameter or arithmetic expansion is never
# resolvable. zsh re-interprets both after a BARE `$NAME` (probed: `$X:h`
# strips the tail, `$X:s,a,b,` substitutes, `$X[1]` subscripts), which turns a
# workspace-looking operand into an out-of-workspace path; the braced `${X}`
# and arithmetic `$((X))` forms treat the suffix as literal text (probed),
# but the suffix is rejected after every expansion form as a cheap denial.
_EXPANSION_SUFFIX_FORBIDDEN = frozenset((":", "["))
# Strict-shape sizes: a recognizable script is exactly one substitution
# followed by one literal suffix; a grep stage has exactly four tokens; the
# inner pipeline is grep plus one to three selector/extractor stages.
_SCRIPT_SEGMENT_COUNT = 2
_GREP_STAGE_TOKEN_COUNT = 4
_HEAD_TAIL_TOKEN_COUNTS = frozenset({2, 3})
_COMPACT_HEAD_TAIL_TOKEN_COUNT = 2
_MIN_PIPELINE_STAGES = 2
_MAX_PIPELINE_STAGES = 4
_MIN_EQUALS_WORD_LENGTH = 2
# The pipeline simulation reads the grep source synchronously during
# analysis, so the read is bounded: a regular file of at most this size is
# simulated; anything larger is denied before a single byte is read.
_SIMULATION_READ_CAP_BYTES = 8 * 1024 * 1024  # 8 MiB
_DRIFT_PREVIEW_LENGTH = 160

logger = logging.getLogger(__name__)


# --- construct (a): the sed locate-then-print idiom --------------------------


@dataclass(frozen=True)
class SedAddressExpansion:
    """The statically expanded result of the sed locate-then-print idiom."""

    expanded_script: str
    """The full outer sed script with the address substituted numerically."""

    address: str
    """The substituted address: digits, or newline-joined digits for a
    multi-match expansion (a runtime sed error, never a write), or '' when
    the grep found no match."""

    suffix: str
    """The literal suffix from the original script (``p``, ``,Np``, ``,+Np``)."""

    addresses: tuple[int, ...]
    """The numeric line addresses the inner pipeline selected."""

    grep_pattern: str
    source_file: str
    sed_file: str


@dataclass(frozen=True)
class _GrepStage:
    pattern: str
    file: str


@dataclass(frozen=True)
class _HeadTailStage:
    count: int
    from_start: bool  # head selects the first N; tail the last N.


@dataclass(frozen=True)
class _CutStage:
    """The numeric-extraction stage: field 1, ':' delimiter, on grep -n output."""


@dataclass(frozen=True)
class _SedLocateShape:
    """The parsed strict shape of the sed locate-then-print idiom (no I/O)."""

    suffix: str
    sed_file: str
    grep_stage: _GrepStage
    selectors: tuple[_HeadTailStage | _CutStage, ...]


def _parse_sed_locate_shape(command: str) -> _SedLocateShape | None:  # noqa: PLR0911
    """Parse the idiom's strict shape; read nothing (the caller pre-checks paths)."""
    text = command.strip()
    if "`" in text:
        return None
    header = _HEADER_RE.match(text)
    if header is None:
        return None
    scanned = _scan_script(text, header.end())
    if scanned is None:
        return None
    segments, rest = scanned
    if (
        len(segments) != _SCRIPT_SEGMENT_COUNT
        or segments[0][0] != "sub"
        or segments[1][0] != "lit"
    ):
        return None
    substitution, suffix = segments[0][1], segments[1][1]
    if not _SUFFIX_RE.fullmatch(suffix):
        return None
    if "$(" in substitution or "<" in substitution or ">" in substitution:
        return None
    sed_file = rest.strip()
    if not sed_file or not _SAFE_PATH_RE.fullmatch(sed_file) or sed_file[0] == "-":
        return None
    pipeline = _parse_pipeline(substitution)
    if pipeline is None:
        return None
    grep_stage, selectors = pipeline
    return _SedLocateShape(
        suffix=suffix,
        sed_file=sed_file,
        grep_stage=grep_stage,
        selectors=tuple(selectors),
    )


def sed_locate_print_source(command: str) -> str | None:
    """The grep source path of a sed locate-then-print-shaped command, or None.

    Pure parsing — no file access — so the wiring can pre-check the source
    against the authorized roots BEFORE the recognizer reads it. Returns None
    for every shape the recognizer rejects anyway (fail closed).
    """
    shape = _parse_sed_locate_shape(command)
    if shape is None:
        return None
    return shape.grep_stage.file


def recognize_sed_locate_print(command: str) -> SedAddressExpansion | None:
    """Statically expand the sed locate-then-print idiom.

    ``command`` is the full text of ONE sed invocation exactly as it appears
    in the proposed command line — no surrounding pipeline, no separators,
    no prefix assignments. The recognizable shape is::

        sed -n "$(PIPELINE)SUFFIX" FILE

    where PIPELINE is ``grep -n 'LITERAL' FILE`` followed by up to three
    whitelisted selector/extractor stages — at most one ``head -N`` (or
    ``head -n N``), at most one ``tail -N``, and EXACTLY ONE ``cut -d: -f1``
    (compact or split spelling) in any order. The ``cut -d: -f1`` extractor
    is an ABSOLUTE requirement: it is what guarantees the expanded sed
    address is a literal numeric range, because grep -n output lines are
    always ``LINENO:content`` and field 1 is always the digits of LINENO.
    A head/tail selector WITHOUT the extractor is NOT recognizable: raw file
    content would reach the sed script (probed: a line ``w /tmp/x`` became
    a sed write command).

    SUFFIX must match ``p`` | ``,Np`` | ``,+Np``. The grep file is READ
    (read-only) to simulate the pipeline; nothing is ever executed. The
    read is gated and bounded (see ``_simulate_pipeline``): a non-regular,
    oversized, or binary source is not recognizable.

    Returns the fully expanded script, or None when the shape is not
    recognizable (fail closed).
    """
    shape = _parse_sed_locate_shape(command)
    if shape is None:
        return None
    addresses = _simulate_pipeline(shape.grep_stage, shape.selectors)
    if addresses is None:
        return None
    address = "\n".join(str(line) for line in addresses)
    return SedAddressExpansion(
        expanded_script=address + shape.suffix,
        address=address,
        suffix=shape.suffix,
        addresses=addresses,
        grep_pattern=shape.grep_stage.pattern,
        source_file=shape.grep_stage.file,
        sed_file=shape.sed_file,
    )


def _scan_script(  # noqa: PLR0912, PLR0915
    text: str, start: int
) -> tuple[list[tuple[str, str]], str] | None:
    """Scan the double-quoted sed script from ``start``.

    Returns (segments, remainder-after-closing-quote) where segments
    alternate ``("sub", content)`` for each ``$(...)`` and ``("lit", text)``
    for the literal runs between/around them, or None when the script
    string is unterminated. Inside the substitution, single- and
    double-quoted spans are tracked and parens are depth-counted, so a
    ``)`` or ``"`` inside the grep pattern is data, and quotes inside the
    substitution (which zsh re-parses there) do not close the outer script.
    """
    segments: list[tuple[str, str]] = []
    literal: list[str] = []
    depth = 0
    sub_start = 0
    in_single = False
    in_double = False
    i = start
    while i < len(text):
        char = text[i]
        if depth == 0:
            if char == '"':
                if literal:
                    segments.append(("lit", "".join(literal)))
                return segments, text[i + 1 :]
            if char == "$" and text.startswith("((", i + 1):
                # An arithmetic $((...)) inside the script: zsh performs
                # command substitution inside arithmetic, so this whole shape
                # is unmodeled. Model it as nested depth so the scan cannot
                # close early, then reject downstream (further substitution).
                if literal:
                    segments.append(("lit", "".join(literal)))
                    literal = []
                depth = 2
                sub_start = i + 3
                i += 3
                continue
            if char == "$" and text.startswith("(", i + 1):
                if literal:
                    segments.append(("lit", "".join(literal)))
                    literal = []
                depth = 1
                sub_start = i + 2
                in_single = in_double = False
                i += 2
                continue
            literal.append(char)
            i += 1
            continue
        if in_single:
            if char == "'":
                in_single = False
        elif in_double:
            if char == '"':
                in_double = False
        elif char == "'":
            in_single = True
        elif char == '"':
            in_double = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                segments.append(("sub", text[sub_start:i]))
                i += 1
                continue
        i += 1
    return None


def _tokenize_stage(stage: str) -> list[tuple[str, bool]] | None:
    """Split one pipeline stage into (text, was-single-quoted) tokens.

    Returns None for unbalanced quoting, quoting glued into a literal token
    (``ab'cd'`` — shell would concatenate; unmodeled, fail closed), or any
    double quote or backslash (expansion/escape territory, unmodeled).
    """
    tokens: list[tuple[str, bool]] = []
    i = 0
    while i < len(stage):
        char = stage[i]
        if char.isspace():
            i += 1
            continue
        if char == "'":
            end = stage.find("'", i + 1)
            if end == -1:
                return None
            tokens.append((stage[i + 1 : end], True))
            i = end + 1
            continue
        j = i
        while j < len(stage) and not stage[j].isspace():
            if stage[j] in "'\"\\|":
                return None
            j += 1
        tokens.append((stage[i:j], False))
        i = j
    return tokens


def _parse_grep_stage(stage: str) -> _GrepStage | None:  # noqa: PLR0911
    tokens = _tokenize_stage(stage)
    if tokens is None or len(tokens) != _GREP_STAGE_TOKEN_COUNT:
        return None
    (name, name_q), (flag, flag_q), (pattern, pattern_q), (path, path_q) = tokens
    if name_q or flag_q or path_q:
        return None
    if name != "grep" or flag != "-n":
        return None
    if not pattern_q or not pattern:
        return None
    if any(char in _PATTERN_FORBIDDEN or not char.isprintable() for char in pattern):
        return None
    if not _SAFE_PATH_RE.fullmatch(path) or path[0] == "-":
        return None
    return _GrepStage(pattern, path)


def _parse_head_tail_stage(stage: str) -> _HeadTailStage | None:
    tokens = _tokenize_stage(stage)
    if tokens is None or len(tokens) not in _HEAD_TAIL_TOKEN_COUNTS:
        return None
    (name, name_q), (first, first_q) = tokens[0], tokens[1]
    if name_q or first_q or name not in {"head", "tail"}:
        return None
    count: str
    if len(tokens) == _COMPACT_HEAD_TAIL_TOKEN_COUNT:
        if not first.startswith("-") or not _DIGITS_RE.fullmatch(first[1:]):
            return None
        count = first[1:]
    else:
        (option, option_q), (value, value_q) = tokens[1], tokens[2]
        if option_q or value_q or option != "-n" or not _DIGITS_RE.fullmatch(value):
            return None
        count = value
    return _HeadTailStage(int(count), from_start=name == "head")


def _parse_cut_stage(stage: str) -> _CutStage | None:
    tokens = _tokenize_stage(stage)
    if tokens is None or not tokens or tokens[0] != ("cut", False):
        return None
    forms = [
        [("-d:", False), ("-f1", False)],
        [("-d", False), (":", False), ("-f1", False)],
        [("-d:", False), ("-f", False), ("1", False)],
        [("-d", False), (":", False), ("-f", False), ("1", False)],
    ]
    if tokens[1:] not in forms:
        return None
    return _CutStage()


def _parse_pipeline(
    substitution: str,
) -> tuple[_GrepStage, list[_HeadTailStage | _CutStage]] | None:
    """Parse the inner pipeline: grep first, then whitelisted stages."""
    stages: list[str] = []
    current: list[str] = []
    in_single = False
    for char in substitution:
        if in_single:
            if char == "'":
                in_single = False
            current.append(char)
            continue
        if char == "'":
            in_single = True
            current.append(char)
            continue
        if char == "|":
            stages.append("".join(current))
            current = []
            continue
        current.append(char)
    if in_single:
        return None
    stages.append("".join(current))
    if (
        len(stages) < _MIN_PIPELINE_STAGES
        or len(stages) > _MAX_PIPELINE_STAGES
        or any(not s.strip() for s in stages)
    ):
        return None
    grep_stage = _parse_grep_stage(stages[0])
    if grep_stage is None:
        return None
    heads = tails = cuts = 0
    parsed: list[_HeadTailStage | _CutStage] = []
    for stage_text in stages[1:]:
        head_tail = _parse_head_tail_stage(stage_text)
        if head_tail is not None:
            if head_tail.from_start:
                heads += 1
            else:
                tails += 1
            parsed.append(head_tail)
            continue
        cut_stage = _parse_cut_stage(stage_text)
        if cut_stage is None:
            return None
        cuts += 1
        parsed.append(cut_stage)
    # The numeric-extraction stage is an ABSOLUTE requirement: exactly one
    # `cut -d: -f1` guarantees a digits-only address; head/tail without it
    # would feed raw file content into the sed script (probed write vector).
    if cuts != 1 or heads > 1 or tails > 1:
        return None
    return grep_stage, parsed


def _log_source_refusal(reason: str, source: str) -> None:
    """Debug-level, redacted, bounded log of a refused simulation read.

    Drift signal for tier-3 candidates only: logging never influences the
    permission outcome.
    """
    try:
        logger.debug(
            "sed locate-then-print source not simulated (%s): %s",
            reason,
            redact(source)[:_DRIFT_PREVIEW_LENGTH],
        )
    except Exception:
        pass


def _simulate_pipeline(  # noqa: PLR0911
    grep_stage: _GrepStage, stages: Sequence[_HeadTailStage | _CutStage]
) -> tuple[int, ...] | None:
    """Statically simulate the whitelisted pipeline; return line numbers.

    The read is gated before it happens: the source must stat as a regular
    file of at most ``_SIMULATION_READ_CAP_BYTES``. A FIFO, device, or socket
    source would block the synchronous analysis (probed: ``/dev/stdin`` on a
    FIFO hangs it), and an oversized source would be fully read before the
    resolver denies the command — both fail closed here instead. The read
    itself is bounded as a hard cap. A source containing a NUL byte is
    refused as well: GNU grep prints ``binary file matches`` to stderr and
    nothing on stdout for a binary file, so the runtime pipeline never
    produces the address this simulation would authorize.
    """
    source = Path(grep_stage.file)
    try:
        info = os.stat(source)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        _log_source_refusal("not a regular file", grep_stage.file)
        return None
    if info.st_size > _SIMULATION_READ_CAP_BYTES:
        _log_source_refusal("source exceeds the simulation read cap", grep_stage.file)
        return None
    try:
        with source.open("rb") as handle:
            raw = handle.read(_SIMULATION_READ_CAP_BYTES + 1)
    except OSError:
        return None
    if len(raw) > _SIMULATION_READ_CAP_BYTES:
        _log_source_refusal("source exceeds the simulation read cap", grep_stage.file)
        return None
    content = raw.decode("utf-8", errors="replace")
    if "\0" in content:
        _log_source_refusal("source contains a NUL byte (binary)", grep_stage.file)
        return None
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    entries = [
        f"{index}:{line}"
        for index, line in enumerate(lines, 1)
        if grep_stage.pattern in line
    ]
    for stage in stages:
        if isinstance(stage, _HeadTailStage):
            if stage.from_start:
                entries = entries[: stage.count]
            else:
                entries = entries[-stage.count :] if stage.count else []
        else:
            entries = [entry.split(":", 1)[0] for entry in entries]
    addresses: list[int] = []
    for entry in entries:
        if not _DIGITS_RE.fullmatch(entry):
            # Defense in depth: the extractor guarantees grep -n line numbers;
            # anything else reaching here means the shape analysis is wrong.
            return None
        addresses.append(int(entry))
    return tuple(addresses)


# --- construct (b): zsh equals expansion --------------------------------------


def resolve_equals_command(word: str, path: Sequence[str] | None = None) -> str | None:
    """Resolve a zsh equals-expansion word (``=literal-command``) by lookup only.

    ``word`` must be one UNQUOTED shell word beginning with ``=`` — quoting
    suppresses equals expansion (probed: ``'=ls'`` stays literal), so the
    caller must not pass quoted words. Exactly ONE leading ``=`` is stripped
    (probed: ``==ls`` looks up ``=ls`` and fails); words containing ``/``
    never resolve. The name is resolved against PATH directories by stat and
    access checks only — nothing is ever executed. ``path`` overrides the
    process PATH for callers that model a different environment.

    Returns the resolved absolute path, or None when the word does not
    trigger a resolvable equals expansion (fail closed).
    """
    if not word.startswith("=") or len(word) < _MIN_EQUALS_WORD_LENGTH:
        return None
    name = word[1:]
    if "/" in name or name.startswith("-") or not _SAFE_COMMAND_NAME_RE.fullmatch(name):
        return None
    directories = (
        path if path is not None else os.environ.get("PATH", "").split(os.pathsep)
    )
    for directory in directories:
        if not directory:
            continue
        candidate = f"{directory}/{name}"
        try:
            info = os.stat(candidate)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode) and os.access(candidate, os.X_OK):
            return candidate
    return None


# --- construct (c): variable/arithmetic expansion evaluation ------------------


def _suffix_reinterprets(expression: str, index: int) -> bool:
    """Whether a `:`/`[` at ``index`` would be re-interpreted after an expansion."""
    return index < len(expression) and expression[index] in _EXPANSION_SUFFIX_FORBIDDEN


def evaluate_static_expansion(  # noqa: PLR0911
    expression: str, variables: Mapping[str, str]
) -> str | None:
    """Statically resolve one word's expansion text to a literal, or None.

    ``expression`` is the raw text of a single UNQUOTED shell word (already
    de-quoted by the caller); quotes or backslashes make it unresolvable.
    Supported: plain literal text, ``$NAME``/``${NAME}`` (plain form only —
    any other ``${...}`` operator form is unresolvable), ``$((arith))`` with
    integer literals, known variable names, ``+ - * / %`` (C truncating
    semantics, as probed for zsh), parentheses, and whitespace — and
    concatenation of those. Unknown variables, command substitution
    anywhere (``$(...)`` or backtick), and nested substitution are NOT
    resolvable. A ``:`` or ``[`` directly after any expansion form is NOT
    resolvable either: after a bare ``$NAME`` zsh re-interprets both as a
    history modifier or subscript (probed), so the resolved word would
    diverge from what the shell runs. zsh never re-expands variable values
    (probed), but values must still pass a conservative literal policy (see
    ``_VALUE_FORBIDDEN``, and no leading ``=``/``~``) so a caller re-emitting
    the resolved word cannot re-enter expansion; values used inside
    arithmetic must be integer literals. The caller must place the resolved
    value as exactly one argv element. Returns None for anything
    unresolvable (fail closed).
    """
    if any(char in expression for char in "'\"\\`"):
        return None
    parts: list[str] = []
    i = 0
    while i < len(expression):
        char = expression[i]
        if char != "$":
            parts.append(char)
            i += 1
            continue
        if expression.startswith("((", i + 1):
            end = _find_arith_end(expression, i + 3)
            if end is None:
                return None
            value = _evaluate_arithmetic(expression[i + 3 : end], variables)
            if value is None:
                return None
            parts.append(str(value))
            i = end + 2
            if _suffix_reinterprets(expression, i):
                return None  # Cheap denial (literal in zsh, unmodeled here).
            continue
        if expression.startswith("{", i + 1):
            name_match = _IDENTIFIER_RE.match(expression, i + 2)
            if (
                name_match is None
                or expression[name_match.end() : name_match.end() + 1] != "}"
            ):
                return None
            resolved = _literal_value(variables.get(name_match.group(0)))
            if resolved is None:
                return None
            parts.append(resolved)
            i = name_match.end() + 1
            if _suffix_reinterprets(expression, i):
                return None  # Cheap denial (literal in zsh, unmodeled here).
            continue
        name_match = _IDENTIFIER_RE.match(expression, i + 1)
        if name_match is None:
            return None
        resolved = _literal_value(variables.get(name_match.group(0)))
        if resolved is None:
            return None
        parts.append(resolved)
        i = name_match.end()
        if _suffix_reinterprets(expression, i):
            # A bare `$NAME` followed by `:`/`[` is a zsh history
            # modifier/subscript, not a literal suffix (probed): the
            # resolved word would diverge from what the shell runs.
            return None
    return "".join(parts)


def _find_arith_end(expression: str, start: int) -> int | None:
    """Find the index of the ``))`` closing a ``$((`` opened before ``start``."""
    depth = 2
    i = start
    while i < len(expression):
        if expression[i] == "(":
            depth += 1
        elif expression[i] == ")":
            depth -= 1
            if depth == 0:
                return i - 1 if expression[i - 1] == ")" and i - 1 >= start else None
        i += 1
    return None


def _literal_value(value: str | None) -> str | None:
    """Apply the conservative literal policy to a variable value."""
    if value is None or not value:
        return None if value is None else value
    if any(char in _VALUE_FORBIDDEN for char in value):
        return None
    if value[0] in "=~":
        return None
    return value


def _evaluate_arithmetic(expression: str, variables: Mapping[str, str]) -> int | None:
    tokens = _tokenize_arithmetic(expression)
    if tokens is None or not tokens:
        return None
    result = _arith_additive(tokens, 0, variables)
    if result is None or result[1] != len(tokens):
        return None
    return result[0]


def _tokenize_arithmetic(expression: str) -> list[str] | None:
    tokens: list[str] = []
    i = 0
    while i < len(expression):
        char = expression[i]
        if char.isspace():
            i += 1
            continue
        if char.isdigit():
            end = i
            while end < len(expression) and expression[end].isdigit():
                end += 1
            tokens.append(expression[i:end])
            i = end
            continue
        if char.isalpha() or char == "_":
            end = i
            while end < len(expression) and (
                expression[end].isalnum() or expression[end] == "_"
            ):
                end += 1
            tokens.append(expression[i:end])
            i = end
            continue
        if char in "+-*/%()":
            tokens.append(char)
            i += 1
            continue
        return None
    return tokens


def _arith_additive(
    tokens: list[str], index: int, variables: Mapping[str, str]
) -> tuple[int, int] | None:
    result = _arith_multiplicative(tokens, index, variables)
    if result is None:
        return None
    value, index = result
    while index < len(tokens) and tokens[index] in {"+", "-"}:
        operator = tokens[index]
        rhs = _arith_multiplicative(tokens, index + 1, variables)
        if rhs is None:
            return None
        value = value + rhs[0] if operator == "+" else value - rhs[0]
        index = rhs[1]
    return value, index


def _arith_multiplicative(
    tokens: list[str], index: int, variables: Mapping[str, str]
) -> tuple[int, int] | None:
    result = _arith_unary(tokens, index, variables)
    if result is None:
        return None
    value, index = result
    while index < len(tokens) and tokens[index] in {"*", "/", "%"}:
        operator = tokens[index]
        rhs = _arith_unary(tokens, index + 1, variables)
        if rhs is None:
            return None
        if operator == "*":
            value *= rhs[0]
        elif operator == "/":
            quotient = _truncating_division(value, rhs[0])
            if quotient is None:
                return None
            value = quotient
        else:
            remainder = _truncating_remainder(value, rhs[0])
            if remainder is None:
                return None
            value = remainder
        index = rhs[1]
    return value, index


def _arith_unary(
    tokens: list[str], index: int, variables: Mapping[str, str]
) -> tuple[int, int] | None:
    if index >= len(tokens):
        return None
    if tokens[index] == "-":
        result = _arith_unary(tokens, index + 1, variables)
        if result is None:
            return None
        return -result[0], result[1]
    return _arith_primary(tokens, index, variables)


def _arith_primary(  # noqa: PLR0911
    tokens: list[str], index: int, variables: Mapping[str, str]
) -> tuple[int, int] | None:
    if index >= len(tokens):
        return None
    token = tokens[index]
    if _DIGITS_RE.fullmatch(token):
        return int(token), index + 1
    if token == "(":
        result = _arith_additive(tokens, index + 1, variables)
        if result is None or result[1] >= len(tokens) or tokens[result[1]] != ")":
            return None
        return result[0], result[1] + 1
    if _IDENTIFIER_RE.fullmatch(token):
        # zsh never re-expands variable values (probed: embedded $(...) or
        # nested $((...)) in a value is a syntax error, no execution), so a
        # value must already BE an integer literal to participate.
        raw = variables.get(token)
        if raw is None or not _INT_VALUE_RE.fullmatch(raw):
            return None
        return int(raw), index + 1
    return None


def _truncating_division(dividend: int, divisor: int) -> int | None:
    if divisor == 0:
        return None
    quotient = dividend // divisor
    if quotient < 0 and quotient * divisor != dividend:
        quotient += 1
    return quotient


def _truncating_remainder(dividend: int, divisor: int) -> int | None:
    if divisor == 0:
        return None
    quotient = _truncating_division(dividend, divisor)
    assert quotient is not None
    return dividend - quotient * divisor


def evaluate_expansion_for_position(
    command: str, position: str, expression: str, variables: Mapping[str, str]
) -> str | None:
    """Resolve ``expression`` only if ``position`` of ``command`` is read-only.

    ``position`` uses the argument-position key grammar of
    ``_shell_argument_positions`` (module 2). Write positions — in
    particular every sed script position — reject everything, including
    plain literals; unknown commands and unknown positions reject as
    well. Returns the resolved literal, or None (fail closed). An EMPTY
    resolution is refused too: zsh drops an empty unquoted word entirely, so
    authorizing it would describe a different argv than the runtime runs.
    """
    rule = classify_argument_position(command, position)
    if rule is None or not rule.read_only:
        return None
    resolved = evaluate_static_expansion(expression, variables)
    if not resolved:
        # None (unresolvable) or '' (an empty word zsh drops entirely).
        return None
    return resolved
