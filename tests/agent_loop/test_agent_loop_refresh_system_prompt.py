from __future__ import annotations

import pytest

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
