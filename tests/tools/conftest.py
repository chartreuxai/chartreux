"""Shared fixtures for the tool tests.

Includes the environmental guard against stray git repositories above the
pytest tmp tree: the shell-permission tests resolve repository scope by
walking from their tmp directories to the filesystem root, so a stray
repository at an ancestor — a ``/tmp/.git`` left by a crashed tool run, for
example — silently misclassifies their assertions, and the failures appear
only under specific xdist schedules. The guard detects such pollution
before each test runs and fails with a clear remedy. It never deletes or
modifies anything.
"""

from __future__ import annotations

import pytest

from tests.tools._tmp_scope_guard import assert_no_stray_repository, guard_roots


@pytest.fixture(autouse=True)
def _guard_against_stray_tmp_repositories(tmp_path_factory) -> None:
    assert_no_stray_repository(guard_roots(tmp_path_factory.getbasetemp()))


@pytest.fixture
def zsh_executor(monkeypatch: pytest.MonkeyPatch):
    """Pin the zsh executor family for one test.

    The shipped default is bash, which never expands a leading ``=`` marker,
    so a test asserting the zsh equals-expansion semantics pins the family
    here instead of depending on the host's ambient ``$SHELL``.
    """
    from chartreux.core.utils import shell as shell_module

    monkeypatch.setattr(shell_module, "_locate_bash", lambda: None)
    shell_module.apply_executor_selection("/usr/bin/zsh")
    try:
        yield
    finally:
        shell_module.reset_executor_selection()
