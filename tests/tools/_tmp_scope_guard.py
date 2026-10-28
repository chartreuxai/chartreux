"""Environmental guard support: stray git repositories above the tmp tree.

The shell-permission tests resolve repository scope by walking from their
tmp directories to the filesystem root (``resolve_repository_scope`` follows
``current.parents`` unconditionally), so a stray repository left at an
ancestor — a ``/tmp/.git`` from a crashed tool run, for example — silently
re-labels every scope-derived assertion in those tests, and the failure
appears only under specific xdist schedules. These helpers detect such
pollution so the autouse guard in ``tests/tools/conftest.py`` can fail
loudly with a remedy instead.

The guard never deletes or modifies anything: it only detects.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest


def guard_roots(basetemp: Path) -> list[Path]:
    """The directories a tmp-dir scope walk can reach.

    The base temp directory itself (each test's ``tmp_path`` lives inside
    it) and every ancestor up to the filesystem root, innermost first.
    """
    resolved = basetemp.resolve()
    return [resolved, *resolved.parents]


def stray_git_markers(roots: Iterable[Path]) -> list[Path]:
    """Stray repository markers among ``roots``, in the order given.

    Mirrors the two repository shapes ``resolve_repository_scope`` accepts:
    a ``.git`` entry (directory or gitdir-pointer file) and a bare
    repository's marker set (a ``HEAD`` file plus ``objects/`` and
    ``refs/`` directories). The roots are the ancestors of the tmp tree —
    never the test directories themselves, where repositories are created
    on purpose.
    """
    found: list[Path] = []
    for root in roots:
        marker = root / ".git"
        if marker.exists() or marker.is_symlink():
            found.append(marker)
            continue
        if (
            (root / "HEAD").is_file()
            and (root / "objects").is_dir()
            and (root / "refs").is_dir()
        ):
            found.append(root / "HEAD")
    return found


def assert_no_stray_repository(roots: Iterable[Path]) -> None:
    """The guard's whole failure path, over injected roots.

    Fails with a clear, actionable message when any root holds a stray
    repository marker; the message names every stray marker and the remedy.
    Taking the roots as a parameter keeps the failure path testable
    against synthetic trees, without touching the real /tmp.
    """
    stray = stray_git_markers(roots)
    if not stray:
        return
    paths = ", ".join(str(marker) for marker in stray)
    pytest.fail(
        f"a stray git repository pollutes the tmp tree's ancestors: {paths}. "
        "The shell-permission tests resolve repository scope by walking "
        "from their tmp directories to the filesystem root, so this "
        "repository silently misclassifies their assertions. Remove the "
        "stray repository (it is not owned by this suite) or run the suite "
        "against a clean temporary directory: TMPDIR=<clean dir> or "
        "pytest --basetemp=<clean dir>.",
        pytrace=False,
    )
