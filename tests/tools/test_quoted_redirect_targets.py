"""C2b: quoted redirect targets are literal WRITE operands.

A redirect target whose every glob metacharacter lies inside a quoted segment
(``echo x > f"*"``, ``echo x > "f[0]"``) is literal data — the shell writes
exactly that name — so it resolves as a literal WRITE operand (C2b restores
this through the C2a quote-provenance seam; the D1 exit criterion). Any
UNQUOTED glob metacharacter (``*.py``, ``f*"y"``, ``link/*.conf``) stays
denied by the D1 redirect-glob rule, and a quoted redirect into protected Git
metadata is still denied by the metadata guard: provenance narrows only the
glob denial's applicability, nothing else.

Each allowed node also runs the real shell to pin that the literal-named file
is what gets created; each denied node pins that the resolver's refusal gates
the write (the guarded files stay untouched).
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig

DENIAL_MARKER = "contains shell glob characters"
METADATA_MARKER = "protected Git metadata"
WORKSPACE_MARKER = "outside the authorized workspace"


def _permission(cwd: Path, command: str) -> ToolPermission:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    return result.permission


def _denial_reason(cwd: Path, command: str) -> str:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    assert result.permission is ToolPermission.NEVER, (command, result)
    return result.reason or ""


def _run_shell(shell: str, command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shell, "-c", command], cwd=cwd, capture_output=True, text=True, timeout=60
    )


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-quoted_star_literal_allowed"),
        pytest.param("zsh", id="zsh-quoted_star_literal_allowed"),
    ],
)
def test_quoted_star_literal_allowed(shell: str, tmp_path: Path) -> None:
    command = 'echo x > f"*"'
    assert _permission(tmp_path, command) is ToolPermission.ALWAYS, command
    # Real-shell probe: the literal file named `f*` is what gets created.
    proc = _run_shell(shell, command, tmp_path)
    assert proc.returncode == 0, (shell, proc.stderr)
    assert (tmp_path / "f*").read_text() == "x\n", shell
    assert sorted(path.name for path in tmp_path.iterdir()) == ["f*"], shell


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-quoted_bracket_literal_allowed"),
        pytest.param("zsh", id="zsh-quoted_bracket_literal_allowed"),
    ],
)
def test_quoted_bracket_literal_allowed(shell: str, tmp_path: Path) -> None:
    command = 'echo x > "f[0]"'
    assert _permission(tmp_path, command) is ToolPermission.ALWAYS, command
    proc = _run_shell(shell, command, tmp_path)
    assert proc.returncode == 0, (shell, proc.stderr)
    assert (tmp_path / "f[0]").read_text() == "x\n", shell
    assert sorted(path.name for path in tmp_path.iterdir()) == ["f[0]"], shell


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-escaped_metachar_literal_allowed"),
        pytest.param("zsh", id="zsh-escaped_metachar_literal_allowed"),
    ],
)
def test_escaped_metachar_literal_allowed(shell: str, tmp_path: Path) -> None:
    # An escaped metacharacter counts as quoted by the shared classifier, so
    # the target is a literal write; the shell writes exactly `f*`.
    command = "echo x > f\\*"
    assert _permission(tmp_path, command) is ToolPermission.ALWAYS, command
    # Real-shell probe: the literal file named `f*` is what gets created.
    proc = _run_shell(shell, command, tmp_path)
    assert proc.returncode == 0, (shell, proc.stderr)
    assert (tmp_path / "f*").read_text() == "x\n", shell
    assert sorted(path.name for path in tmp_path.iterdir()) == ["f*"], shell


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-bare_glob_still_denied"),
        pytest.param("zsh", id="zsh-bare_glob_still_denied"),
    ],
)
def test_bare_glob_still_denied(shell: str, tmp_path: Path) -> None:
    command = "echo x > *.py"
    reason = _denial_reason(tmp_path, command)
    assert DENIAL_MARKER in reason, (command, reason)
    # The refusal gates the write: the resolver denies before execution, so
    # nothing is created (the shells' own zero-match behavior — bash creates
    # the literal name — is pinned in the D1 regression file).


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-partially_quoted_denied"),
        pytest.param("zsh", id="zsh-partially_quoted_denied"),
    ],
)
def test_partially_quoted_denied(shell: str, tmp_path: Path) -> None:
    # The bare `*` lies outside the quotes: the shell still pathname-expands.
    command = 'echo x > f*"y"'
    reason = _denial_reason(tmp_path, command)
    assert DENIAL_MARKER in reason, (command, reason)
    # The refusal gates the write: nothing is created under the permission
    # model; the literal-creation fallback is the shells' own behavior.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-symlink_dir_still_denied"),
        pytest.param("zsh", id="zsh-symlink_dir_still_denied"),
    ],
)
def test_symlink_dir_still_denied(shell: str, tmp_path: Path) -> None:
    command = "echo x > link/*.conf"

    def setup(root: Path) -> None:
        (root / "real").mkdir()
        (root / "real" / "a.conf").write_text("orig-conf")
        (root / "link").symlink_to(root / "real", target_is_directory=True)

    setup(tmp_path)
    reason = _denial_reason(tmp_path, command)
    assert DENIAL_MARKER in reason, (command, reason)
    assert (tmp_path / "real" / "a.conf").read_text() == "orig-conf"


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-git_metadata_still_denied"),
        pytest.param("zsh", id="zsh-git_metadata_still_denied"),
    ],
)
def test_git_metadata_still_denied(shell: str, tmp_path: Path) -> None:
    # A fully quoted target carries no glob metacharacters at all: the D1
    # rule does not fire, and the metadata guard denies the write instead —
    # proving quoted provenance did not open the guard.
    command = "echo x > '.git/hooks/pre-commit'"
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=60)
    (tmp_path / ".git" / "hooks" / "pre-commit").write_text("orig-hook")
    reason = _denial_reason(tmp_path, command)
    assert METADATA_MARKER in reason, (command, reason)
    assert DENIAL_MARKER not in reason, (command, reason)
    # The denial gates the write: the hook stays untouched.
    assert (tmp_path / ".git" / "hooks" / "pre-commit").read_text() == "orig-hook"


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-workspace_escape_denied"),
        pytest.param("zsh", id="zsh-workspace_escape_denied"),
    ],
)
def test_workspace_escape_denied(shell: str, tmp_path: Path) -> None:
    # Fully quoted glob, but the target lands outside the authorized root:
    # workspace authority still denies the write (the ALLOWED half of the
    # contract covers in-root targets only).
    command = 'echo x > ../"*"'
    reason = _denial_reason(tmp_path, command)
    assert WORKSPACE_MARKER in reason, (command, reason)
    assert DENIAL_MARKER not in reason, (command, reason)
    # The refusal gates the write: nothing is created.
    assert list(tmp_path.iterdir()) == []
