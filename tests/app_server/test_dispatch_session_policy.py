from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.app_server._runtime import _AgentLoopBlueprint
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.dispatch.renderer import task_description_for_config
from chartreux.core.dispatch.session import (
    bound_identity_reference,
    resume_policy_bundle,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import POLICY_STORAGE_VERSION
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


@pytest.mark.asyncio
async def test_app_server_sessions_render_distinct_bound_policies_in_process():
    # The composed system prompt is keyed on the [agent] selection: the
    # first session composes the orchestrator + ceremony + default identity,
    # the second composes the default coder + minimal selection.
    config = build_test_vibe_config(
        include_prompt_detail=True,
        include_model_info=False,
        agent={
            "identity": "orchestrator",
            "workflow": "ceremony",
            "orchestration_strategy": "builtin:default",
        },
    ).attach_catalog_snapshot(CatalogSnapshot(SHIPPED_CATALOG, "one"))
    source = build_test_agent_loop(config=config)
    standalone = build_test_vibe_config(
        include_prompt_detail=True, include_model_info=False
    ).attach_catalog_snapshot(config.catalog_snapshot)
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
        # The orchestrator composition carries the ceremony implement region.
        assert "never edit repo files yourself" in (first.messages[0].content or "")
        # The coder composition runs the checks itself.
        assert "You are the coder" in (second.messages[0].content or "")
        assert "run the relevant checks yourself" in (second.messages[0].content or "")
        assert first.messages[0].content != second.messages[0].content
        first_task = task_description_for_config(first.config)
        second_task = task_description_for_config(second.config)
        # The task description renders from the configured route table and
        # the session model — never from the identity selection — so both
        # sessions serve the identical composition render.
        assert first_task == second_task
        assert "### Launch targets" in first_task
        assert "worker disclosure requirement" in first_task
        await first.refresh_system_prompt()
        assert task_description_for_config(first.config) == first_task
        assert task_description_for_config(second.config) == second_task
        # Each session binds its own frozen policy bundle, keyed on the
        # composed selection.
        assert first._bound_policy is not None
        assert second._bound_policy is not None
        assert bound_identity_reference(first._bound_policy) == "orchestrator"
        assert bound_identity_reference(second._bound_policy) == "coder"
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_persisted_policy_record_is_the_bound_bundle(tmp_path: Path):
    # Package 10: the persisted policy record of a conversation IS its bound
    # bundle — the storage boundary activates it back to the same frozen
    # policy hash, per conversation.
    configs = [
        build_test_vibe_config(
            agent={
                "identity": "orchestrator",
                "workflow": "ceremony",
                "orchestration_strategy": "builtin:default",
            },
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
        ),
        build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        ),
    ]
    loops = [build_test_agent_loop(config=config) for config in configs]
    hashes = []
    try:
        for loop in loops:
            bound = loop._bound_policy
            assert bound is not None
            hashes.append(bound.frozen_policy_hash)
            await loop.persist_empty_session()
            session_dir = loop.session_logger.session_dir
            assert session_dir is not None
            metadata = SessionLoader.load_metadata(session_dir)
            assert metadata.policy_storage_version == POLICY_STORAGE_VERSION == 2
            assert metadata.policy_bundle is not None
            resumed = resume_policy_bundle(metadata)
            assert resumed.frozen_policy_hash == bound.frozen_policy_hash
            assert (
                resumed.manifest.identity.selected == bound.manifest.identity.selected
            )
    finally:
        for loop in loops:
            await loop.aclose()
    # Distinct selections persist distinct frozen policy records.
    assert hashes[0] != hashes[1]


@pytest.mark.asyncio
async def test_launch_purposes_come_from_the_declared_purpose():
    from chartreux.app_server._projection import project_launch_purposes
    from chartreux.core.launch_types import LaunchConfig

    loop = build_test_agent_loop()
    try:
        # A direct launch carries no purpose.
        assert project_launch_purposes(loop) == []
        loop.launch_purpose = "verification"
        assert project_launch_purposes(loop) == ["verification"]
        loop.launch_purpose = "implementation"
        # The declared purpose is the only label: the launch's model or
        # identity no longer attributes a purpose.
        assert project_launch_purposes(loop) == ["implementation"]
        loop.launch_overrides = LaunchConfig(model="@scout")
        assert project_launch_purposes(loop) == ["implementation"]
        loop.launch_purpose = None
        assert project_launch_purposes(loop) == []
    finally:
        await loop.aclose()
