"""Shared test-support surface for the shell-command policy suite (W10, stage 19g).

The definitions here are the shared module-level surface of
``tests/tools/test_shell_command_policy.py``, relocated AST-verbatim in the
19g phase-(i) helper-relocation commit: the ``bash_tool`` fixture, the one
cross-concern entry of that file's module-level surface (52 tests request it,
spanning all four concerns — the executor, git-classification,
repository-inspection and classification clusters — plus the
``shell_resolver`` fixture body). ``EXECUTOR_REGISTRY`` is re-exported from
``chartreux.core.tools.builtins._shell_command_policy`` (the production
facade) as an identity import so the executor concern destination imports
the registry from this seam (plan row 19g); the support-module name is the
same object as the production name. The remaining module-level surface is
single-concern and stays in ``test_shell_command_policy.py`` until the
phase-(ii) concern moves: the denylist fixtures ``shell_resolver``,
``git_denylist_tool`` and ``git_denylist_resolver``, the v0.2 redirect
fixtures ``workspace`` and ``scratchpad_dir`` with the ``_v02_tool``,
``_v02_verdict`` and ``_v02_reason`` helpers, the git-dir helper
``_git_dir_with_bare_markers`` and the repository-config helper
``_repository_with_config``. No test bodies or marks live here; the test
population stays in ``test_shell_command_policy.py`` until the phase-(ii)
concern moves. Importer: ``test_shell_command_policy.py`` itself (the seam
enumeration found zero external importers of that file).
"""

from __future__ import annotations

import pytest

from chartreux.core.tools.base import BaseToolState
from chartreux.core.tools.builtins._shell_command_policy import (
    EXECUTOR_REGISTRY,  # noqa: F401 - identity re-export via the support seam
)
from chartreux.core.tools.builtins.bash import Bash, BashToolConfig


@pytest.fixture
def bash_tool(tmp_path, monkeypatch) -> Bash:
    # Explicit entries only: the default denylists are extended elsewhere and
    # must not be pinned by these tests.
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    return Bash(config_getter=lambda: config, state=BaseToolState())
