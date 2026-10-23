"""Actual-shell regression tests for the redirect-glob expansion fix (D1).

Shells pathname-expand redirect destinations (zsh writes every match; bash
overwrites a single match or creates the literal name), so a glob-bearing
redirect target is denied fail-closed at the synthetic-redirect seam
(``append_redirect_operand``), before every later redirect check.

Each node runs the command through the real shell binary to pin the no-fix
behavior the denial closes, then asserts the resolver denies the same
command so the write is prevented, and that the protected files are
unchanged. The file carries NO skip conditions by rule: if a shell binary
is absent, the affected nodes fail.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig

# The implemented denial marker: rendered as
# "Command denied: redirect target '<target>' contains shell glob characters; ..."
DENIAL_MARKER = "contains shell glob characters"


def _resolver_denial(cwd: Path, command: str) -> str:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    assert result.permission is ToolPermission.NEVER, (command, result)
    reason = result.reason or ""
    assert DENIAL_MARKER in reason, reason
    return reason


def _run_shell(shell: str, command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shell, "-c", command], cwd=cwd, capture_output=True, text=True, timeout=60
    )


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-single_match"),
        pytest.param("zsh", id="zsh-single_match"),
    ],
)
def test_redirect_glob_single_match_denies_overwrite(
    shell: str, tmp_path: Path
) -> None:
    command = "echo pwned > *.py"
    work = tmp_path / "work"
    work.mkdir()
    (work / "target.py").write_text("orig-target")
    # The denial gates the write: without executing anything, the resolver
    # refuses and the target file is untouched.
    _resolver_denial(work, command)
    assert (work / "target.py").read_text() == "orig-target"
    # The actual shells do expand the destination and overwrite the single
    # match — the no-fix behavior the denial closes.
    danger = tmp_path / "danger"
    danger.mkdir()
    (danger / "target.py").write_text("orig-target")
    proc = _run_shell(shell, command, danger)
    assert proc.returncode == 0, (shell, proc.stderr)
    assert (danger / "target.py").read_text() == "pwned\n"


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-ambiguous_match"),
        pytest.param("zsh", id="zsh-ambiguous_match"),
    ],
)
def test_redirect_glob_ambiguous_match_denies_multi_write(
    shell: str, tmp_path: Path
) -> None:
    command = "echo pwned > *.py"

    def setup(root: Path) -> None:
        (root / "a.py").write_text("orig-a")
        (root / "b.py").write_text("orig-b")

    # The denial gates the write: both matches stay untouched.
    work = tmp_path / "work"
    work.mkdir()
    setup(work)
    _resolver_denial(work, command)
    assert (work / "a.py").read_text() == "orig-a"
    assert (work / "b.py").read_text() == "orig-b"
    # Actual-shell no-fix behavior: bash refuses with "ambiguous redirect"
    # and writes nothing; zsh WRITES TO EVERY MATCH — the danger the denial
    # closes.
    danger = tmp_path / "danger"
    danger.mkdir()
    setup(danger)
    proc = _run_shell(shell, command, danger)
    if shell == "bash":
        assert proc.returncode != 0, (shell, proc.stderr)
        assert "ambiguous redirect" in proc.stderr, proc.stderr
        assert (danger / "a.py").read_text() == "orig-a"
        assert (danger / "b.py").read_text() == "orig-b"
    else:
        assert proc.returncode == 0, (shell, proc.stderr)
        assert (danger / "a.py").read_text() == "pwned\n"
        assert (danger / "b.py").read_text() == "pwned\n"


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-zero_match"),
        pytest.param("zsh", id="zsh-zero_match"),
    ],
)
def test_redirect_glob_zero_match_denies_literal_creation(
    shell: str, tmp_path: Path
) -> None:
    command = "echo pwned > *.py"
    # The denial gates the write: no file is created at all.
    work = tmp_path / "work"
    work.mkdir()
    _resolver_denial(work, command)
    assert list(work.iterdir()) == []
    # Actual-shell no-fix behavior: bash creates a file literally named
    # "*.py"; zsh refuses with "no matches found".
    danger = tmp_path / "danger"
    danger.mkdir()
    proc = _run_shell(shell, command, danger)
    if shell == "bash":
        assert proc.returncode == 0, (shell, proc.stderr)
        assert (danger / "*.py").read_text() == "pwned\n"
    else:
        assert proc.returncode != 0, (shell, proc.stderr)
        assert "no matches found" in proc.stderr, proc.stderr
        assert list(danger.iterdir()) == []


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-symlink_dir"),
        pytest.param("zsh", id="zsh-symlink_dir"),
    ],
)
def test_redirect_glob_symlink_dir_denies_write_through_link(
    shell: str, tmp_path: Path
) -> None:
    command = "echo pwned > link/*.conf"

    def setup(root: Path) -> None:
        (root / "real").mkdir()
        (root / "real" / "a.conf").write_text("orig-conf")
        (root / "link").symlink_to(root / "real", target_is_directory=True)

    # The denial gates the write: the real file behind the symlink stays
    # untouched.
    work = tmp_path / "work"
    work.mkdir()
    setup(work)
    _resolver_denial(work, command)
    assert (work / "real" / "a.conf").read_text() == "orig-conf"
    # Actual-shell no-fix behavior: both shells expand the glob through the
    # symlinked directory component and overwrite the real file.
    danger = tmp_path / "danger"
    danger.mkdir()
    setup(danger)
    proc = _run_shell(shell, command, danger)
    assert proc.returncode == 0, (shell, proc.stderr)
    assert (danger / "real" / "a.conf").read_text() == "pwned\n"


@pytest.mark.parametrize(
    "shell",
    [
        pytest.param("bash", id="bash-git_metadata"),
        pytest.param("zsh", id="zsh-git_metadata"),
    ],
)
def test_redirect_glob_git_metadata_denied_at_redirect_seam(
    shell: str, tmp_path: Path
) -> None:
    command = "echo pwned > .git/hooks/*.py"

    def setup(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
        # Exactly one *.py hook: with a single match both shells expand the
        # glob and overwrite the hook inside protected Git metadata.
        (root / ".git" / "hooks" / "pre-commit.py").write_text("orig-hook")

    # The redirect-glob denial fires at the synthetic-redirect seam, BEFORE
    # the git-metadata guard: the reason identifies the glob-bearing-redirect
    # rule and never the metadata guard, pinning the rule ordering.
    work = tmp_path / "work"
    work.mkdir()
    setup(work)
    reason = _resolver_denial(work, command)
    assert "protected Git metadata" not in reason, reason
    assert (work / ".git" / "hooks" / "pre-commit.py").read_text() == "orig-hook"
    # Actual-shell no-fix behavior: both shells expand the glob and overwrite
    # the hook file inside protected Git metadata.
    danger = tmp_path / "danger"
    danger.mkdir()
    setup(danger)
    proc = _run_shell(shell, command, danger)
    assert proc.returncode == 0, (shell, proc.stderr)
    assert (danger / ".git" / "hooks" / "pre-commit.py").read_text() == "pwned\n"
