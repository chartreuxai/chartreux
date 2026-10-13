"""Symbol outline contract for Python files.

An outline is bounded, metadata-only structure for one file: the symbols a
worker needs to orient itself — classes (with methods), functions (with
nested functions), and cheap module-level bindings (callables and aliases) —
each with its *current* line range, computed from the Python AST (stdlib
:mod:`ast` only; no grammar dependency). Outlines carry no file bodies.

Degradation is a first-class outcome, never an exception. Non-Python files,
files that do not parse, and files beyond the source cap degrade to ``None``
(the file stays revision-tracked, just not outlined). The per-outline caps
(symbol count, rendered bytes, nesting depth) degrade to a *truncated*
outline — the first symbols up to the cap, flagged with ``truncated=True`` —
so a bounded outline is always produced for parseable Python.

An outline describes exactly one file state: line ranges are only meaningful
for the content they were computed from. The working-set state
(:mod:`chartreux.core.file_state`) pairs outlines with content revisions
(:mod:`chartreux.core.file_revision`) and drops an outline when the
revision moves.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field

__all__ = [
    "MAX_OUTLINE_DEPTH",
    "MAX_OUTLINE_RENDER_BYTES",
    "MAX_OUTLINE_SOURCE_BYTES",
    "MAX_OUTLINE_SYMBOLS",
    "SYMBOL_KIND_ASSIGNMENT",
    "SYMBOL_KIND_CLASS",
    "SYMBOL_KIND_FUNCTION",
    "FileOutline",
    "OutlineSymbol",
    "outline_file",
    "outline_source",
]

#: Upper bound on symbols in one outline; beyond it the outline is truncated.
MAX_OUTLINE_SYMBOLS = 512

#: Upper bound on the rendered outline text, in bytes; beyond it trailing
#: symbols are dropped (the truncation note is reserved within the cap).
MAX_OUTLINE_RENDER_BYTES = 16 * 1024

#: Upper bound on source size, in bytes; larger sources degrade to ``None``.
MAX_OUTLINE_SOURCE_BYTES = 512 * 1024

#: Upper bound on symbol nesting depth; deeper scopes are cut (truncated).
MAX_OUTLINE_DEPTH = 64

SYMBOL_KIND_CLASS = "class"
SYMBOL_KIND_FUNCTION = "function"
SYMBOL_KIND_ASSIGNMENT = "assignment"

_PYTHON_SUFFIX = ".py"
_TRUNCATION_NOTE = (
    f"... outline truncated (caps: {MAX_OUTLINE_SYMBOLS} symbols, "
    f"{MAX_OUTLINE_RENDER_BYTES} bytes)"
)
_ASSIGNMENT_VALUE_TYPES = (ast.Lambda, ast.Name, ast.Attribute, ast.Call)


class OutlineSymbol(BaseModel):
    """One outlined symbol with its current line range.

    ``name`` is the qualified name (``Class.method``, ``outer.inner``);
    ``line``/``end_line`` are the 1-based inclusive line numbers of the
    symbol's whole definition (``ast.end_lineno``), so they describe the file
    as it currently stands.
    """

    name: str
    kind: str
    line: int
    end_line: int


class FileOutline(BaseModel):
    """Bounded symbol outline of one file state (metadata only, no bodies)."""

    symbols: list[OutlineSymbol] = Field(default_factory=list)
    truncated: bool = False
    line_count: int = 0

    def render(self) -> str:
        """Render the outline as the text block served to the agent."""
        lines = [_render_symbol(symbol) for symbol in self.symbols]
        if self.truncated:
            lines.append(_TRUNCATION_NOTE)
        return "\n".join(lines)


def outline_source(
    source: str | bytes, *, path: str | None = None
) -> FileOutline | None:
    """Outline Python ``source``; ``None`` when it cannot be outlined.

    ``path`` (when given) gates on the ``.py`` suffix: non-Python files
    degrade to ``None``. Sources beyond :data:`MAX_OUTLINE_SOURCE_BYTES` and
    sources that do not parse degrade to ``None`` as well; outlines beyond
    the symbol/render caps degrade to a truncated :class:`FileOutline`.
    """
    if path is not None and Path(path).suffix != _PYTHON_SUFFIX:
        return None
    if isinstance(source, str):
        data = source.encode("utf-8", errors="replace")
    else:
        data = source
    if len(data) > MAX_OUTLINE_SOURCE_BYTES:
        return None
    try:
        tree = ast.parse(data)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    symbols: list[OutlineSymbol] = []
    truncated = _collect_scope(_module_symbol_statements(tree.body), "", 0, symbols)
    outline = FileOutline(
        symbols=symbols, truncated=truncated, line_count=len(data.splitlines())
    )
    _enforce_caps(outline)
    return outline


def outline_file(path: str | Path) -> FileOutline | None:
    """Outline the Python file at ``path``; ``None`` when it cannot be outlined.

    Non-``.py`` files, files beyond the source cap, unreadable files, and
    files that do not parse all degrade to ``None``.
    """
    file_path = Path(path)
    if file_path.suffix != _PYTHON_SUFFIX:
        return None
    try:
        if file_path.stat().st_size > MAX_OUTLINE_SOURCE_BYTES:
            return None
        data = file_path.read_bytes()
    except OSError:
        return None
    return outline_source(data, path=str(file_path))


def _render_symbol(symbol: OutlineSymbol) -> str:
    return f"{symbol.kind} {symbol.name} (lines {symbol.line}-{symbol.end_line})"


def _collect_scope(
    body: list[ast.stmt], prefix: str, depth: int, out: list[OutlineSymbol]
) -> bool:
    """Append the symbols of one scope; return True when the depth cap cut."""
    truncated = False
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            truncated |= _collect_definition(
                stmt, prefix, SYMBOL_KIND_CLASS, depth, out
            )
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            truncated |= _collect_definition(
                stmt, prefix, SYMBOL_KIND_FUNCTION, depth, out
            )
        elif depth == 0:
            symbol = _assignment_symbol(stmt)
            if symbol is not None:
                out.append(symbol)
    return truncated


def _collect_definition(
    stmt: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    prefix: str,
    kind: str,
    depth: int,
    out: list[OutlineSymbol],
) -> bool:
    """Append one class/function symbol and its scope; True when depth-cut."""
    qualified = f"{prefix}{stmt.name}"
    out.append(
        OutlineSymbol(
            name=qualified,
            kind=kind,
            line=stmt.lineno,
            end_line=stmt.end_lineno or stmt.lineno,
        )
    )
    if depth >= MAX_OUTLINE_DEPTH:
        return True
    return _collect_scope(stmt.body, f"{qualified}.", depth + 1, out)


def _module_symbol_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Return the module-level symbol statements, in source order.

    Module-level compound statements (``if``/``try``/``with``/``for``/
    ``while``/``match``) are flattened so conditionally defined symbols are
    found. Class and function bodies are *not* descended into here — they
    are separate scopes with qualified names.
    """
    found: list[ast.stmt] = []
    stack = list(reversed(body))
    while stack:
        stmt = stack.pop()
        if isinstance(
            stmt,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            found.append(stmt)
            continue
        # The stack is LIFO, so sub-bodies are pushed in reverse source order
        # (``try``: body, handlers, orelse, finalbody) to pop in source order.
        sub_bodies: list[list[ast.stmt]] = []
        nested_body = getattr(stmt, "body", None)
        if isinstance(nested_body, list):
            sub_bodies.append(nested_body)
        handlers = getattr(stmt, "handlers", None)
        if handlers is not None:
            sub_bodies.extend(handler.body for handler in handlers)
        for attribute in ("orelse", "finalbody"):
            nested = getattr(stmt, attribute, None)
            if isinstance(nested, list):
                sub_bodies.append(nested)
        cases = getattr(stmt, "cases", None)
        if cases is not None:
            sub_bodies.extend(case.body for case in cases)
        for sub_body in reversed(sub_bodies):
            stack.extend(reversed(sub_body))
    return found


def _assignment_symbol(stmt: ast.stmt) -> OutlineSymbol | None:
    """Return the module-level assignment symbol for ``stmt``, or ``None``.

    Only cheap, outline-relevant bindings are outlined: the value is a
    lambda, a name or attribute (alias), or a call (factory). Plain
    constants are excluded as noise, and assignments inside class/function
    scopes are not outlined.
    """
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return None
        target = stmt.targets[0].id
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        if stmt.value is None or not isinstance(stmt.target, ast.Name):
            return None
        target = stmt.target.id
        value = stmt.value
    else:
        return None
    if not isinstance(value, _ASSIGNMENT_VALUE_TYPES):
        return None
    return OutlineSymbol(
        name=target,
        kind=SYMBOL_KIND_ASSIGNMENT,
        line=stmt.lineno,
        end_line=stmt.end_lineno or stmt.lineno,
    )


def _enforce_caps(outline: FileOutline) -> None:
    """Apply the symbol-count and rendered-byte caps to ``outline`` in place.

    Both caps degrade to a truncated outline: the first symbols up to the
    cap are kept and ``truncated`` is set. The truncation note is reserved
    within the byte cap so the rendered outline never exceeds it.
    """
    if len(outline.symbols) > MAX_OUTLINE_SYMBOLS:
        del outline.symbols[MAX_OUTLINE_SYMBOLS:]
        outline.truncated = True
    for _ in range(2):
        budget = MAX_OUTLINE_RENDER_BYTES
        if outline.truncated:
            budget -= len(_TRUNCATION_NOTE.encode("utf-8")) + 1
        kept = 0
        total = 0
        for symbol in outline.symbols:
            cost = len(_render_symbol(symbol).encode("utf-8")) + 1
            if total + cost > budget:
                break
            total += cost
            kept += 1
        if kept == len(outline.symbols):
            return
        del outline.symbols[kept:]
        outline.truncated = True
