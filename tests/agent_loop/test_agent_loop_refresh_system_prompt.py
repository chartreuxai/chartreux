from __future__ import annotations

import pytest

from chartreux.core.dispatch.renderer import task_description_for_config
from chartreux.core.dispatch.session import bound_identity_reference
from tests.conftest import build_test_agent_loop, build_test_vibe_config


def _description_lines(description: str) -> list[str]:
    """The description's lines, order-insensitive.

    Every render path reads the conversation's LIVE route table, so a
    description rendered after a reload carries the edited table; the
    sort keeps the comparison robust against any row-order difference
    between render paths.
    """
    return sorted(description.splitlines())


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
async def test_internal_refresh_serves_the_frozen_policy_bundle() -> None:
    # Package 10: the refresh renders the bundle captured at startup. A live
    # config copy carrying a different identity selection must not change
    # what the refresh renders — the bound bundle is the only frozen-policy
    # source (design §10: no live reread of frozen policy during prompt
    # refresh).
    agent = build_test_agent_loop(config=build_test_vibe_config())
    frozen_prompt = agent.messages[0].content or ""
    bound = agent._bound_policy
    assert bound is not None

    live_config = agent.config.model_copy(
        update={"agent": agent.config.agent.model_copy(update={"identity": "reviewer"})}
    )

    refreshed_prompt = agent._render_system_prompt(
        agent.skill_manager, config=live_config
    )

    assert refreshed_prompt == frozen_prompt
    assert agent._bound_policy is bound


@pytest.mark.asyncio
async def test_saved_identity_edit_applies_next_session_not_refresh() -> None:
    # Package 10/16b: the bound policy bundle is frozen at startup, so a
    # saved identity selection takes effect in the next session, never in a
    # refresh of the running one. The task tool description renders from the
    # route facts and the session model, so it is selection-independent and
    # stable across the edit.
    config = build_test_vibe_config(
        include_prompt_detail=True, include_model_info=False
    )
    agent = build_test_agent_loop(config=config)
    initial = agent.messages[0].content
    initial_task = task_description_for_config(agent.config)
    bound = agent._bound_policy
    assert bound is not None

    edited = config.model_copy(
        update={"agent": config.agent.model_copy(update={"identity": "reviewer"})}
    )

    # The refresh and any prepared consumer serve the frozen bundle's prompt;
    # the edited live config does not leak into the running conversation.
    await agent.refresh_system_prompt()
    assert agent.messages[0].content == initial
    assert agent._render_system_prompt(agent.skill_manager, config=edited) == initial
    assert agent._bound_policy is bound
    assert task_description_for_config(edited) == initial_task

    prepared = agent._prepare_reload_consumers(edited, False, reuse_backend=True)
    # The prepared consumers serve the conversation's LIVE route table:
    # content-identical to the initial description (the identity edit does
    # not touch the routes).
    assert _description_lines(prepared.tool_manager._tool_descriptions["task"]) == (
        _description_lines(initial_task)
    )
    assert prepared.system_prompt == initial
    retained = agent._prepare_launch_consumers(
        edited, replace_tools=True, reuse_backend=True
    )
    assert _description_lines(retained.tool_manager._tool_descriptions["task"]) == (
        _description_lines(initial_task)
    )

    # The next session binds and composes the saved selection.
    fresh = build_test_agent_loop(config=edited)
    assert fresh._bound_policy is not None
    assert bound_identity_reference(fresh._bound_policy) == "reviewer"
    assert "You are the reviewer" in (fresh.messages[0].content or "")
    assert fresh._bound_policy is not bound
