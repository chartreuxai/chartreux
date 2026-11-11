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
from tests.app_server.test_tree_policy import tree, update

pytestmark = pytest.mark.asyncio


def assert_stale(
    store: DangerousCommandGrantStore, admission: ExecutionAdmission
) -> None:
    tool = Mock()
    tool.get_name.return_value = "bash"
    manager = Mock()
    denial = revalidate_execution_admission(
        cast(BaseTool, tool), None, cast(ToolManager, manager), store, admission
    )
    assert denial is not None and "policy revision is stale" in denial
    tool.resolve_permission.assert_not_called()


async def test_revocation_invalidates_consumed_once_admission(tmp_path: Path) -> None:
    async with tree(tmp_path) as (registry, runtimes):
        loop = runtimes[0].agent_loop
        store = loop._dangerous_command_grants
        scope = RepositoryScope(git_dir=tmp_path, work_tree=tmp_path)
        admission = ExecutionAdmission(
            policy_revision=store.current_policy_revision,
            once_requirements=(("git.push", scope),),
        )
        result = await registry.revoke_dangerous_command_grant(
            session_id=loop.session_id,
            operation_id="git.push",
            expected_live_generation=store.authority_generation,
            expected_disk_revision=None,
        )
        assert result.live.status == "removed"
        assert_stale(store, admission)


async def test_policy_replacement_invalidates_tree_admissions(tmp_path: Path) -> None:
    async with tree(tmp_path) as (registry, runtimes):
        stores = [runtime.agent_loop._dangerous_command_grants for runtime in runtimes]
        admissions = [
            ExecutionAdmission(policy_revision=s.current_policy_revision)
            for s in stores
        ]
        await update(registry, runtimes[0], ["blocked"])
        for store, admission in zip(stores, admissions, strict=True):
            assert store.current_policy_revision > admission.policy_revision
            assert_stale(store, admission)
