from __future__ import annotations

import pytest

from chartreux.core.agents.registry import (
    apply_launch_overrides,
    apply_profile_overrides,
)
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.agent_profile import AgentProfileLayer
from chartreux.core.config.layers.launch_overrides import LaunchOverridesLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


def test_launch_layer_follows_profile_and_runtime_overrides() -> None:
    orchestrator = FakeConfigOrchestrator(
        build_test_vibe_config(enabled_tools=["parent"])
    )
    apply_profile_overrides(orchestrator, {"enabled_tools": ["profile"]})
    orchestrator.insert_layer(OverridesLayer(data={"enabled_tools": ["runtime"]}), 1)

    apply_launch_overrides(orchestrator, {"enabled_tools": ["launch"]})

    assert [type(layer) for layer in orchestrator.layers] == [
        AgentProfileLayer,
        OverridesLayer,
        LaunchOverridesLayer,
    ]
    assert orchestrator.config.enabled_tools == ["launch"]


def test_launch_layer_is_copied_without_sharing_state_or_duplicates() -> None:
    parent = FakeConfigOrchestrator(build_test_vibe_config(enabled_tools=["parent"]))
    parent.insert_layer(OverridesLayer(data={"enabled_tools": ["runtime"]}), 0)
    apply_launch_overrides(parent, {"enabled_tools": ["launch"]})
    first_child = parent._copy_for_child()
    sibling = parent._copy_for_child()

    apply_launch_overrides(first_child, {"enabled_tools": ["first"]})

    assert first_child.config.enabled_tools == ["first"]
    assert parent.config.enabled_tools == ["launch"]
    assert sibling.config.enabled_tools == ["launch"]
    assert (
        sum(isinstance(layer, LaunchOverridesLayer) for layer in first_child.layers)
        == 1
    )
    assert sum(isinstance(layer, LaunchOverridesLayer) for layer in sibling.layers) == 1


@pytest.mark.asyncio
async def test_launch_layer_does_not_add_source_restrictions() -> None:
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layer(OverridesLayer(data={"tools": {"bash": {"permission": "never"}}}))
    builder.add_layer(
        LaunchOverridesLayer(data={"tools": {"bash": {"permission": "always"}}})
    )

    candidate = await builder.build_candidate()

    assert candidate.config.tools["bash"]["permission"] == "always"
    assert [restriction.layer_name for restriction in candidate.restrictions] == [
        OverridesLayer.NAME
    ]
