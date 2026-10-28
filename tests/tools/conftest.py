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
