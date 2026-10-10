from __future__ import annotations

from dataclasses import replace

import pytest

from chartreux.app_server._runtime import _AgentLoopBlueprint
from chartreux.core.dispatch.presets import ORCHESTRATED_PRESET, STANDALONE_PRESET
from chartreux.core.dispatch.renderer import task_description_for_config
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


@pytest.mark.asyncio
async def test_app_server_sessions_render_distinct_bound_policies_in_process():
    config = build_test_vibe_config(
        system_prompt_id="cli", include_prompt_detail=True, include_model_info=False
    ).attach_catalog_snapshot(
        CatalogSnapshot(SHIPPED_CATALOG, "one", dispatch=ORCHESTRATED_PRESET)
    )
    source = build_test_agent_loop(config=config)
    standalone = config.model_copy(deep=True).attach_catalog_snapshot(
        replace(config.catalog_snapshot, dispatch=STANDALONE_PRESET)
    )
    first = _AgentLoopBlueprint(
        config_orchestrator=source.config_orchestrator.copy(),
        policy=source.runtime_policy,
        cwd=source.cwd,
        harness_files=source.harness_files,
    ).build()
    second = _AgentLoopBlueprint(
        config_orchestrator=FakeConfigOrchestrator(standalone),
        policy=source.runtime_policy,
        cwd=source.cwd,
        harness_files=source.harness_files,
    ).build()
    try:
        await first.refresh_system_prompt()
        await second.refresh_system_prompt()
        assert "never edit repo files yourself" in (first.messages[0].content or "")
        assert "You may implement directly" in (second.messages[0].content or "")
        first_task = task_description_for_config(first.config)
        second_task = task_description_for_config(second.config)
        assert first_task != second_task
        await first.refresh_system_prompt()
        assert task_description_for_config(first.config) == first_task
        assert task_description_for_config(second.config) == second_task
        assert (
            first.bound_dispatch_policy.policy.mode
            != second.bound_dispatch_policy.policy.mode
        )
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_launch_purposes_distinguish_shared_worker_profile():
    from chartreux.app_server._projection import project_launch_slot_purposes
    from chartreux.core.launch_types import LaunchConfig

    loop = build_test_agent_loop()
    try:
        loop.launch_overrides = LaunchConfig(model="@scout")
        assert "verification" in project_launch_slot_purposes(loop, "worker")
        assert "implementation" not in project_launch_slot_purposes(loop, "worker")
        loop.launch_overrides = LaunchConfig(model="@worker")
        # The worker profile's implementor seat is unambiguous, but the
        # reviewer profile's reviewer and execution-reviewer slots share
        # @worker with different purposes, so role identity alone cannot
        # attribute review seats.
        assert project_launch_slot_purposes(loop, "worker") == ["implementation"]
        assert project_launch_slot_purposes(loop, "reviewer") == []
    finally:
        await loop.aclose()
