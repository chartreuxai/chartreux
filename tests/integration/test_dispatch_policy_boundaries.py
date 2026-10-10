"""End-to-end boundaries for loaded and session-bound dispatch policy."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.core.agents.launch import resolve_launch
from chartreux.core.agents.models import WORKER
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.dispatch.renderer import (
    render_cli_prompt_for_config,
    task_description_for_config,
)
from chartreux.core.dispatch.schema import DispatchMode
from chartreux.core.dispatch.session import bind_policy
from chartreux.core.model_catalog.loader import load_catalog
from chartreux.core.prompts import SystemPrompt
from chartreux.core.subagents import LaunchConfig
from chartreux.core.system_prompt import _interpolate_prompt
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


def _config(path: Path) -> ChartreuxConfigSchema:
    return build_test_vibe_config().attach_catalog_snapshot(load_catalog(path))


def test_dispatch_overlay_to_frozen_rendered_and_concrete_launch(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text('[dispatch]\nmode="standalone"\n[roles.scout]\nthinking="medium"\n')
    config = _config(path)
    bound = bind_policy(config)
    config.attach_dispatch_policy(bound)
    cli = render_cli_prompt_for_config(
        config, _interpolate_prompt(SystemPrompt.CLI.read())
    )
    task = task_description_for_config(config)
    assert bound.policy.mode == DispatchMode.STANDALONE
    assert "You may implement directly" in cli
    assert "never edit repo files yourself" not in cli
    assert "configured slots" in task
    slot = bound.policy.slots["mechanical"]
    identity = bound.bindings["mechanical"]
    assert bound.identity_for(slot.role) == identity
    assert identity.thinking == "medium"

    parent = FakeConfigOrchestrator(config)
    path.write_text(
        '[dispatch]\nmode="orchestrated"\n'
        '[models.later]\nthinking="high"\n'
        'deployments=[{provider="mistral", name="later-wire"}]\n'
        '[roles.scout]\nmodel="later"\nthinking="high"\n'
    )
    config.attach_catalog_snapshot(load_catalog(path))
    candidate = resolve_launch(
        profile_name=slot.profile,
        config=LaunchConfig(model=slot.role),
        parent_orchestrator=parent,
        tool_inventory={},
        profile_lookup=lambda _: WORKER,
    )
    assert candidate.committed_model.model_dump(exclude={"catalog_revision"}) == (
        identity.model_dump(exclude={"catalog_revision"})
    )
    assert (
        candidate.committed_model.catalog_revision == config.catalog_snapshot.revision
    )
    assert bound.bindings["mechanical"] == identity
    assert candidate.effective_model.alias == identity.base_model
    assert candidate.effective_model.name == identity.wire_name
    assert candidate.effective_model.provider == identity.provider
    assert candidate.effective_thinking == "medium"
    assert candidate.orchestrator.bound_dispatch_policy == bound
    assert candidate.orchestrator.config.bound_dispatch_policy == bound
    override = resolve_launch(
        profile_name=slot.profile,
        config=LaunchConfig(model=slot.role, thinking="high"),
        parent_orchestrator=parent,
        tool_inventory={},
        profile_lookup=lambda _: WORKER,
    )
    assert override.committed_model.base_model == identity.base_model
    assert override.effective_thinking == "high"


def test_dispatch_change_is_next_session_only(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text('[dispatch]\nmode="orchestrated"\n')
    config = _config(path)
    running = bind_policy(config)
    config.attach_dispatch_policy(running)
    original_cli = render_cli_prompt_for_config(
        config, _interpolate_prompt(SystemPrompt.CLI.read())
    )
    original_task = task_description_for_config(config)
    path.write_text('[dispatch]\nmode="standalone"\n')
    config.attach_catalog_snapshot(load_catalog(path))
    assert config.bound_dispatch_policy is running
    assert (
        render_cli_prompt_for_config(
            config, _interpolate_prompt(SystemPrompt.CLI.read())
        )
        == original_cli
    )
    assert task_description_for_config(config) == original_task
    assert running.policy.mode == DispatchMode.ORCHESTRATED
    fresh_config = _config(path)
    fresh_config.attach_dispatch_policy(bind_policy(fresh_config))
    assert fresh_config.bound_dispatch_policy.policy.mode == DispatchMode.STANDALONE
    assert "You may implement directly" in render_cli_prompt_for_config(
        fresh_config, _interpolate_prompt(SystemPrompt.CLI.read())
    )


def test_invalid_dispatch_fallback_renders_and_reports(tmp_path: Path, caplog):
    path = tmp_path / "models.toml"
    path.write_text('[dispatch]\nmode="invalid"\n')
    with caplog.at_level(logging.WARNING, logger="vibe"):
        config = _config(path)
        bound = bind_policy(config)
    config.attach_dispatch_policy(bound)
    cli = render_cli_prompt_for_config(
        config, _interpolate_prompt(SystemPrompt.CLI.read())
    )
    assert config.catalog_snapshot.dispatch.mode == DispatchMode.STANDALONE
    assert bound.policy.mode == DispatchMode.STANDALONE
    assert bound.render_policy.mode == DispatchMode.STANDALONE
    assert bound.diagnostics == config.catalog_snapshot.dispatch_diagnostics
    assert "S10" in bound.diagnostics[0]
    assert "You may implement directly" in cli
    assert "never edit repo files yourself" not in cli
    assert "standalone" in caplog.text
    assert str(path) in caplog.text
    assert path.read_text() == '[dispatch]\nmode="invalid"\n'


def test_legacy_roles_only_overlay_remains_usable(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text('[roles.orchestrator]\nthinking="medium"\n')
    config = _config(path)
    bound = bind_policy(config)
    config.attach_dispatch_policy(bound)
    assert bound.policy.mode == DispatchMode.STANDALONE
    assert config.catalog_snapshot.catalog.roles["orchestrator"].thinking == "medium"
    assert config.get_active_model().thinking == "medium"
    assert "You may implement directly" in render_cli_prompt_for_config(
        config, _interpolate_prompt(SystemPrompt.CLI.read())
    )
    assert task_description_for_config(config)


@pytest.mark.asyncio
async def test_bound_launch_agent_bar_identity_hides_dispatch_internals(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text('[dispatch]\nmode="standalone"\n')
    config = _config(path)
    bound = bind_policy(config)
    config.attach_dispatch_policy(bound)
    slot = bound.policy.slots["implementor"]
    candidate = resolve_launch(
        profile_name=slot.profile,
        config=LaunchConfig(model=slot.role),
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={},
        profile_lookup=lambda _: WORKER,
    )
    agent = AgentSummaryModel(
        agent_id="agent-1",
        profile=candidate.profile.name,
        availability="idle",
        base_model=candidate.committed_model.base_model,
        effective_model=candidate.effective_model.name,
        active_provider=candidate.effective_model.provider,
        effective_thinking=candidate.effective_thinking,
    )

    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield AgentBar()

    async with Host().run_test(size=(140, 24)) as pilot:
        bar = pilot.app.query_one(AgentBar)
        bar.update_agents((agent,))
        bar.open_browser()
        details = bar._agent_details(agent)
        metadata = bar.full_metadata(agent)
        assert f"Profile: {slot.profile}" in metadata
        assert f"{agent.active_provider}/{agent.effective_model}" in details
        for internal in (
            slot.role,
            "implementor",
            bound.policy.identity,
            "dispatch",
            "slot",
        ):
            assert internal not in details
            assert internal not in metadata
