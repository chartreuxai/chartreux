"""Regression tests for the find pattern-operand expansion injection (S3).

A pre-release deep review demonstrated working predicate injection through
the find pattern-operand carve-out: in a directory containing files named
``-a`` and ``-exec``, ``find . -name -* printf HARMLESS_MARKER ';'`` resolved
ALWAYS, because the carve-out skipped the ``-*`` pattern operand from
path-glob inspection — but the shell pathname-expands the unquoted ``-*``
into ``-a -exec`` before find runs, so find received
``-name -a -exec printf HARMLESS_MARKER ';'`` and EXECUTED the printf. The
execution-predicate guard never fired: it only sees the unexpanded form.

The fix is fail-closed exactly like the redirect-glob denial: the inspector
cannot expand pattern operands, so it must not assume they stay inert. An
UNQUOTED, glob-bearing token in a find value-predicate position is denied
with the generic path-glob diagnostic (no work-budget attribution — the
expansion inspection never ran). Two seams cooperate:

1. The policy (``_find_policy``) surfaces every unquoted glob-bearing
   value-predicate operand as a plain path candidate, so the resolver sees
   the injection shape even though the naive positional walk never yields a
   dash-prefixed operand.
2. The resolver's pattern-operand carve-out keeps quoted and literal
   operands (the shell never pathname-expands them) but no longer skips an
   unquoted glob, which falls through to the path-glob denial.

Quoted pattern operands remain allowed (no expansion), literal pattern
operands remain allowed, and the execution-predicate guard is unchanged.
The file carries NO skip conditions by rule: if a shell binary is absent,
the affected tests FAIL, never skip.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_argv import (
    PatternGlobCandidate,
    split_shell_tokens,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.permissions import PermissionContext

# The marker the injected -exec runs in the real-shell probe.
MARKER = "HARMLESS_MARKER"

# The rendered path-glob denial marker: "Shell path glob candidate '<token>'
# in command segment '<part>' cannot be safely inspected".
PATH_GLOB_MARKER = "path glob candidate"
# The rendered execution-predicate guard: "Command denied: find execution
# predicates are not permitted".
EXEC_PREDICATE_MARKER = "find execution predicates are not permitted"

# Substrings only an expansion-provenance detail would add; the find
# pattern denial is a generic inspection failure and must contain neither.
_NO_PROVENANCE = ("budget", "glob syntax")


def _resolve(cwd: Path, command: str) -> PermissionContext:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    return result


def _run_shell(command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command], cwd=cwd, capture_output=True, text=True, timeout=60
    )


def _write_decoys(root: Path) -> None:
    # Files whose names are find predicates: the shell's expansion of an
    # unquoted `-*` yields `-a -exec`, which re-parses as an -exec predicate.
    (root / "-a").write_text("", encoding="utf-8")
    (root / "-exec").write_text("", encoding="utf-8")
    (root / "victim.txt").write_text("x\n", encoding="utf-8")


def test_unquoted_glob_pattern_operand_denies_fail_closed(tmp_path: Path) -> None:
    """The exact injection shape fails closed with the generic path-glob denial.

    The real-shell probe first proves the no-fix behavior the denial closes:
    bash expands the unquoted `-*` into `-a -exec`, so find runs the
    injected `-exec printf` and the marker reaches stdout. The resolver must
    deny the same command, and the denial is the established generic
    path-glob diagnostic — the inspector cannot expand pattern operands,
    so it does not run the expansion inspection and attributes no budget.
    """
    command = f"find . -name -* printf {MARKER} ';'"
    danger = tmp_path / "danger"
    danger.mkdir()
    _write_decoys(danger)
    proc = _run_shell(command, danger)
    assert proc.returncode == 0, (command, proc.stderr)
    assert MARKER in proc.stdout, (command, proc.stdout)

    work = tmp_path / "work"
    work.mkdir()
    _write_decoys(work)
    context = _resolve(work, command)
    assert context.permission is ToolPermission.NEVER, (command, context)
    reason = context.reason or ""
    assert PATH_GLOB_MARKER in reason, reason
    assert "cannot be safely inspected" in reason, reason

    # Policy layer: the unquoted glob-bearing pattern operand joins the
    # inspected path candidates as a PLAIN word (never a
    # PatternGlobCandidate), so the resolver's path-glob rule denies it —
    # the same fail-closed mechanism the grep option-word rule uses. The
    # naive positional walk never yields the dash-prefixed operand, so this
    # surfacing is the only way the resolver sees the injection shape.
    policy = analyze_shell_command_policy(split_shell_tokens(command))
    assert policy.option_path_values == ("-*",)
    assert not isinstance(policy.option_path_values[0], PatternGlobCandidate)


def test_quoted_pattern_operand_remains_allowed(tmp_path: Path) -> None:
    """A quoted pattern operand is pure data and stays ALLOWED.

    The shell never pathname-expands a quote-protected metacharacter, so the
    carve-out still skips the operand and the command runs: the real-shell
    probe lists the matching file, and the resolver allows the same command.
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "x.py").write_text("x\n", encoding="utf-8")
    (work / "y.py").write_text("y\n", encoding="utf-8")
    for command in (
        "find . -name '*.py'",
        "find . -iname '*.Py'",
        "find . -path '*/tests/*'",
        "find . -name '[0-9].py'",
        'find . -name "*.py"',
    ):
        proc = _run_shell(command, work)
        assert proc.returncode == 0, (command, proc.stderr)
        context = _resolve(work, command)
        assert context.permission is ToolPermission.ALWAYS, (command, context)
        # Policy layer: a quoted pattern operand is not surfaced as a path
        # candidate at all.
        policy = analyze_shell_command_policy(split_shell_tokens(command))
        assert policy.option_path_values == (), command


def test_literal_dash_pattern_operand_remains_allowed(tmp_path: Path) -> None:
    """A literal (non-glob) dash-prefixed pattern operand stays ALLOWED.

    ``find . -name -a`` passes the literal word ``-a`` through: no glob
    metacharacter, no expansion, no injection — the control the finding
    observed as correct. The real-shell probe shows find matching the decoy
    file named ``-a``, and the resolver allows the same command.
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "-a").write_text("", encoding="utf-8")
    command = "find . -name -a"
    proc = _run_shell(command, work)
    assert proc.returncode == 0, (command, proc.stderr)
    assert proc.stdout.splitlines() == ["./-a"], proc.stdout
    context = _resolve(work, command)
    assert context.permission is ToolPermission.ALWAYS, (command, context)
    policy = analyze_shell_command_policy(split_shell_tokens(command))
    assert policy.option_path_values == ()


def test_explicit_exec_predicate_still_denied(tmp_path: Path) -> None:
    """The execution-predicate guard is unchanged on the unexpanded form.

    ``find . -name x -exec printf X ';'`` carries a literal ``-exec`` token,
    so the guard denies it with the execution-predicate diagnostic — the
    control that already worked before the fix.
    """
    command = "find . -name x -exec printf X ';'"
    context = _resolve(tmp_path, command)
    assert context.permission is ToolPermission.NEVER, (command, context)
    assert EXEC_PREDICATE_MARKER in (context.reason or ""), context.reason


def test_glob_pattern_operand_with_visible_delete_token_denied(tmp_path: Path) -> None:
    """A literal side-effect token stays visible to the guard (regression).

    ``find . -name -* -delete`` carries the literal ``-delete`` unexpanded,
    so the execution-predicate guard still fires on it: the fail-closed
    pattern-operand denial must not mask or replace the literal-token guard.
    """
    command = "find . -name -* -delete"
    context = _resolve(tmp_path, command)
    assert context.permission is ToolPermission.NEVER, (command, context)
    assert EXEC_PREDICATE_MARKER in (context.reason or ""), context.reason


def test_inspired_variants_denied(tmp_path: Path) -> None:
    """Inspired variants of the injection shape all fail closed.

    The denial covers every value-predicate spelling (the carve-out's four
    plus the other value predicates), the non-dash unquoted glob, and shapes
    without the trailing ``';'`` terminator — the denial is decided at the
    pattern operand itself and does not depend on what follows it.
    """
    _write_decoys(tmp_path)
    variants = [
        "find . -iname -* printf X ';'",
        "find . -path -* printf X ';'",
        "find . -ipath -* printf X ';'",
        "find . -lname -* printf X ';'",
        "find . -name *",
        "find . -name -* printf X",
        "find . -name -* -print",
    ]
    for command in variants:
        context = _resolve(tmp_path, command)
        assert context.permission is ToolPermission.NEVER, (command, context)
        assert PATH_GLOB_MARKER in (context.reason or ""), (command, context.reason)


def test_find_pattern_denial_is_not_budget_attributed(tmp_path: Path) -> None:
    """The find pattern denial is a generic inspection failure, not a budget.

    S2 scoping: the inspector cannot expand pattern operands, so no
    expansion inspection ran and no work budget was exhausted. The denial
    renders the generic path-glob message — it must not carry the
    directory-entry budget, the match-state budget, or the unsupported
    glob-syntax provenance that only an expansion-inspection failure adds.
    """
    command = f"find . -name -* printf {MARKER} ';'"
    _write_decoys(tmp_path)
    context = _resolve(tmp_path, command)
    assert context.permission is ToolPermission.NEVER, (command, context)
    reason = context.reason or ""
    assert PATH_GLOB_MARKER in reason, reason
    assert "cannot be safely inspected" in reason, reason
    assert not any(marker in reason for marker in _NO_PROVENANCE), reason
    assert "directory-entry budget" not in reason, reason
    assert "match-state budget" not in reason, reason
