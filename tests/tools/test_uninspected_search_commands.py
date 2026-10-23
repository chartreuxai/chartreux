"""Regression tests for the egrep/fgrep/rg operand-inspection fix (D2).

egrep, fgrep and ripgrep appeared in neither the read-only command set nor
the per-command policy tables, so their file operands were entirely
uninspected: ``egrep -n foo /etc/passwd`` resolved to ALLOWED and actually
read the file, while the ``grep`` control was denied. This file pins the
closed gap across the three seams:

1. Pattern-position handling: egrep/fgrep get grep's semantics (the first
   positional is a PATTERN; ``-e PAT`` supplies the pattern and ``-f FILE``
   makes the patternfile itself a read operand); rg supports ONLY the plain
   ``rg PATTERN PATH...`` form and every option — plain or execution-bearing
   (``--pre`` runs a preprocessor) — fails closed.
2. Command registration: all three are read commands with per-command
   policies, so their file operands are path candidates.
3. Operand access: their operands are READ-classified like grep's, never
   MUTATE.

Out-of-authorized-root operands are denied; in-root operands stay allowed.
The file carries NO skip conditions by rule: if a shell binary is absent,
the affected tests FAIL, never skip.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
    path_candidates,
)
from chartreux.core.tools.builtins._shell_permission_resolver import _operand_access
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.utils import PathAccess

# A readable file outside every authorized workspace root, used by the
# resolver-denial cases (mirrors the G24 grep pattern-file control).
OUT_OF_ROOT = "/etc/passwd"
# The out-of-root file's content contains this line's token, so a real-shell
# probe matches and its zero exit code proves the file was actually read.
OUT_OF_ROOT_MATCH = "root"

# The rg fail-closed diagnostic: rendered as
# "Command denied: rg option '<token>' is not supported: only the plain
# 'rg PATTERN PATH...' form is permitted".
RG_DENIAL_MARKER = "is not supported"


def _resolve(cwd: Path, command: str) -> PermissionContext:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    return result


def _run_shell(command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command], cwd=cwd, capture_output=True, text=True, timeout=60
    )


def _write_fixtures(root: Path) -> None:
    # "a*b" lets the fixed-string fgrep pattern 'a*' match and "[0-9]" lets
    # the fixed-string fgrep pattern '[0-9]' match, while both patterns still
    # carry glob metacharacters for the pattern-position pins.
    (root / "target.py").write_text("x = 1\nroot\na*b\n[0-9]\n", encoding="utf-8")
    (root / "patterns.txt").write_text("root\n", encoding="utf-8")


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(
            f"egrep -n {OUT_OF_ROOT_MATCH} {OUT_OF_ROOT}", id="egrep-out_of_root"
        ),
        pytest.param(
            f"fgrep {OUT_OF_ROOT_MATCH} {OUT_OF_ROOT}", id="fgrep-out_of_root"
        ),
        pytest.param(f"rg {OUT_OF_ROOT_MATCH} {OUT_OF_ROOT}", id="rg-out_of_root"),
    ],
)
def test_out_of_root_operand_denied(tmp_path: Path, command: str) -> None:
    """Out-of-root file operands are denied for all three search commands.

    The real-shell probe first proves the command actually reads the
    out-of-root file (zero exit code with a matching line) — the live bug
    the fix closes — then the resolver is asserted to deny the same command.
    """
    proc = _run_shell(command, tmp_path)
    assert proc.returncode == 0, (command, proc.stderr)
    result = _resolve(tmp_path, command)
    assert result.permission is ToolPermission.NEVER, (command, result)


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("egrep '[0-9]' target.py", id="egrep-in_root"),
        pytest.param("fgrep 'a*' target.py", id="fgrep-in_root"),
        pytest.param("rg 'a*' target.py", id="rg-in_root"),
    ],
)
def test_in_root_operand_allowed(tmp_path: Path, command: str) -> None:
    """In-root file operands stay ALLOWED for all three search commands.

    Each pattern carries a glob metacharacter to pin the pattern-position
    exclusion: the pattern is regex data, never a path candidate (mirrors
    the G18/G19 pins in tests/recovery/test_grep_glob_candidates.py).
    """
    _write_fixtures(tmp_path)
    proc = _run_shell(command, tmp_path)
    assert proc.returncode == 0, (command, proc.stderr)
    result = _resolve(tmp_path, command)
    assert result.permission is ToolPermission.ALWAYS, (command, result)


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("rg --files", id="rg_unmodeled-files"),
        pytest.param("rg -e PAT target.py", id="rg_unmodeled-e_option"),
        pytest.param("rg -f patterns.txt target.py", id="rg_unmodeled-f_option"),
        pytest.param("rg --unknown-flag target.py", id="rg_unmodeled-generic_option"),
        pytest.param("rg --pre cmd target.py", id="rg_unmodeled-execution_bearing"),
        pytest.param("rg PAT target.py --pre cmd", id="rg_unmodeled-trailing_option"),
        pytest.param("rg PAT -- target.py", id="rg_unmodeled-separator"),
    ],
)
def test_rg_unmodeled_option_denied(tmp_path: Path, command: str) -> None:
    """Every rg option fails closed, plain or execution-bearing.

    rg supports only the plain ``rg PATTERN PATH...`` form: patternless
    modes (``--files``), rg's own ``-e``/``-f`` pattern options, unrecognized
    options, and option values that run programs (``--pre``) are all denied
    with the rg diagnostic naming the offending option.
    """
    result = _resolve(tmp_path, command)
    assert result.permission is ToolPermission.NEVER, (command, result)
    assert result.reason is not None
    assert RG_DENIAL_MARKER in result.reason, result.reason


@pytest.mark.parametrize(
    "probe", [pytest.param("operand-access", id="read_classification")]
)
def test_read_classification(probe: str) -> None:
    """The new commands' operands are READ-classified like grep's.

    ``_operand_access`` defaults unlisted commands' operands to MUTATE, so
    the seam is the registration in ``_INSTRUCTION_READ_COMMANDS``: without
    it, in-root reads would be wrongly denied (including protected-git-
    metadata handling). Also pins the policy shape: the pattern position is
    excluded from the path candidates and a ``-f`` patternfile surfaces as
    an inspected path value.
    """
    for command in ("grep", "egrep", "fgrep", "rg"):
        assert _operand_access(command) is PathAccess.READ, command
    assert path_candidates(
        ["rg", "PAT", "target.py"], inspect_positional_paths=True
    ) == ("target.py",)
    assert path_candidates(
        ["egrep", "PAT", "a.py", "b.py"], inspect_positional_paths=True
    ) == ("a.py", "b.py")
    policy = analyze_shell_command_policy(["egrep", "-f", "patterns.txt", "target.py"])
    assert policy.option_path_values == ("patterns.txt",)
    assert policy.positional_values == ("target.py",)


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("egrep -e '[0-9]' target.py", id="pattern_option-egrep"),
        pytest.param("fgrep -e '[0-9]' target.py", id="pattern_option-fgrep"),
    ],
)
def test_pattern_option_excludes_pattern_operand(tmp_path: Path, command: str) -> None:
    """The ``-e`` value is regex data, not a path; the file operand is inspected.

    With ``-e`` supplying the pattern, every positional is a file operand
    and the pattern value itself never becomes a path candidate — a bracket
    class in it would otherwise trigger the path-glob denial.
    """
    _write_fixtures(tmp_path)
    result = _resolve(tmp_path, command)
    assert result.permission is ToolPermission.ALWAYS, (command, result)
    proc = _run_shell(command, tmp_path)
    assert proc.returncode == 0, (command, proc.stderr)


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(
            "egrep -f patterns.txt target.py", id="pattern_file-egrep-in_root"
        ),
        pytest.param(
            f"egrep -f {OUT_OF_ROOT} target.py", id="pattern_file-egrep-out_of_root"
        ),
        pytest.param(
            "fgrep -f patterns.txt target.py", id="pattern_file-fgrep-in_root"
        ),
        pytest.param(
            f"fgrep -f {OUT_OF_ROOT} target.py", id="pattern_file-fgrep-out_of_root"
        ),
    ],
)
def test_pattern_file_operand_inspected(tmp_path: Path, command: str) -> None:
    """The ``-f`` patternfile is itself a read operand, for both commands.

    An in-root patternfile is ALLOWED; an out-of-root patternfile is DENIED
    via path authority — an implementation that inspects positionals but
    skips ``-f`` values cannot pass.
    """
    _write_fixtures(tmp_path)
    result = _resolve(tmp_path, command)
    expected = ToolPermission.NEVER if OUT_OF_ROOT in command else ToolPermission.ALWAYS
    assert result.permission is expected, (command, result)


def test_pattern_file_read_is_shell_observable(tmp_path: Path) -> None:
    """Real-shell probe: ``fgrep -f`` reads the patternfile it is denied for.

    The target file carries a line copied from the out-of-root patternfile,
    so the zero exit code proves the patternfile was actually read — the
    live behavior the resolver's inspection closes.
    """
    _write_fixtures(tmp_path)
    first_line = Path(OUT_OF_ROOT).read_text(encoding="utf-8").splitlines()[0]
    (tmp_path / "victim.txt").write_text(first_line + "\n", encoding="utf-8")
    proc = _run_shell(f"fgrep -f {OUT_OF_ROOT} victim.txt", tmp_path)
    assert proc.returncode == 0, proc.stderr
    result = _resolve(tmp_path, f"fgrep -f {OUT_OF_ROOT} victim.txt")
    assert result.permission is ToolPermission.NEVER, result
