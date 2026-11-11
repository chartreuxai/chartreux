from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from chartreux.core.agent_loop._dangerous_command_grants import (
    DangerousCommandGrantStore,
)
from chartreux.core.agent_loop._execution_admission import (
    ExecutionAdmission,
    revalidate_execution_admission,
)
from chartreux.core.dangerous_commands import RepositoryScope
from chartreux.core.tools.base import BaseTool
from chartreux.core.tools.manager import ToolManager


def test_revision_blocks_stale_once_admission(tmp_path: Path) -> None:
    scope = RepositoryScope(git_dir=tmp_path, work_tree=tmp_path)
    store = DangerousCommandGrantStore()
    admission = ExecutionAdmission(
        policy_revision=store.current_policy_revision,
        once_requirements=(("git.push", scope),),
    )
    tool = Mock()
    tool.get_name.return_value = "bash"
    tool.resolve_permission.return_value = None
    manager = Mock()
    manager.get_tool_config.return_value = None
    assert (
        revalidate_execution_admission(
            cast(BaseTool, tool), None, cast(ToolManager, manager), store, admission
        )
        is None
    )
    assert not store.revoke_session((("git.push", scope),))
    denial = revalidate_execution_admission(
        cast(BaseTool, tool), None, cast(ToolManager, manager), store, admission
    )
    assert denial is not None and "policy revision is stale" in denial
    assert tool.resolve_permission.call_count == 1
    assert store.dangerous_command_authorities() == ()


@pytest.mark.parametrize("missing_store", [True, False])
def test_revision_rejects_missing_or_outdated_authority(missing_store: bool) -> None:
    store = DangerousCommandGrantStore()
    admission = ExecutionAdmission(policy_revision=store.current_policy_revision)
    store.bump_policy_revision()
    tool = Mock()
    tool.get_name.return_value = "bash"
    manager = Mock()
    denial = revalidate_execution_admission(
        cast(BaseTool, tool),
        None,
        cast(ToolManager, manager),
        None if missing_store else store,
        admission,
    )
    assert denial is not None and "policy revision is stale" in denial
    tool.resolve_permission.assert_not_called()
