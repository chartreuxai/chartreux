from __future__ import annotations

from dataclasses import replace

import pytest

from chartreux.core.dispatch.presets import ORCHESTRATED_PRESET, STANDALONE_PRESET
from chartreux.core.dispatch.renderer import task_description_for_config
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from tests.conftest import build_test_agent_loop, build_test_vibe_config


@pytest.mark.asyncio
async def test_refresh_system_prompt_preserves_scratchpad_section() -> None:
    # Regression: refresh_system_prompt must pass scratchpad_dir, otherwise
    # it silently drops the scratchpad instructions from the system prompt.
    # This refresh must retain those instructions so the LLM stays aware of
    # the scratchpad throughout the session.
    config = build_test_vibe_config(
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
    )
    agent = build_test_agent_loop(config=config)

    initial_prompt = agent.messages[0].content or ""
    assert "Scratchpad Directory" in initial_prompt
    assert agent.scratchpad_dir is not None
    assert str(agent.scratchpad_dir) in initial_prompt

    await agent.refresh_system_prompt()

    refreshed_prompt = agent.messages[0].content or ""
    assert "Scratchpad Directory" in refreshed_prompt
    assert str(agent.scratchpad_dir) in refreshed_prompt


@pytest.mark.asyncio
async def test_internal_refresh_renders_frozen_system_prompt_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = build_test_agent_loop(
        config=build_test_vibe_config(system_prompt_id="tests"),
        frozen_system_prompt_id="tests",
    )
    rendered_ids: list[str] = []

    def render(config, *_args, **_kwargs) -> str:
        rendered_ids.append(config.system_prompt_id)
        return f"prompt:{config.system_prompt_id}"

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.get_universal_system_prompt", render
    )
    live_config = agent.config.model_copy(update={"system_prompt_id": "cli"})

    refreshed_prompt = agent._render_system_prompt(
        agent.skill_manager, config=live_config
    )

    assert rendered_ids == ["tests"]
    assert refreshed_prompt == "prompt:tests"


@pytest.mark.asyncio
async def test_saved_policy_edit_applies_next_session_not_refresh() -> None:
    config = build_test_vibe_config(
        system_prompt_id="cli", include_prompt_detail=True, include_model_info=False
    ).attach_catalog_snapshot(
        CatalogSnapshot(SHIPPED_CATALOG, "initial", dispatch=ORCHESTRATED_PRESET)
    )
    agent = build_test_agent_loop(config=config)
    initial = agent.messages[0].content
    initial_task = task_description_for_config(agent.config)
    original = agent.bound_dispatch_policy
    snapshot = agent.config.catalog_snapshot
    changed = replace(snapshot, dispatch=STANDALONE_PRESET)
    agent.config.attach_catalog_snapshot(changed)
    await agent.refresh_system_prompt()
    assert agent.messages[0].content == initial
    assert agent.bound_dispatch_policy == original
    assert task_description_for_config(agent.config) == initial_task
    fresh_config = config.model_copy(deep=True).attach_catalog_snapshot(changed)
    staged_config = fresh_config.model_copy(deep=True).attach_dispatch_policy(None)
    prepared = agent._prepare_reload_consumers(staged_config, False, reuse_backend=True)
    assert prepared.tool_manager._tool_descriptions["task"] == initial_task
    assert prepared.system_prompt == initial
    retained = agent._prepare_launch_consumers(
        fresh_config.model_copy(deep=True).attach_dispatch_policy(None),
        replace_tools=True,
        reuse_backend=True,
    )
    assert retained.tool_manager._tool_descriptions["task"] == initial_task
    fresh = build_test_agent_loop(config=fresh_config)
    assert fresh.bound_dispatch_policy.policy == STANDALONE_PRESET
    assert "You may implement directly" in (fresh.messages[0].content or "")
    assert fresh.bound_dispatch_policy != original
