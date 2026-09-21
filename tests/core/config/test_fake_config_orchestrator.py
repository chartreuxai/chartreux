from __future__ import annotations

import pytest

from chartreux.core.config.layers.overrides import OverridesLayer
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


@pytest.mark.asyncio
@pytest.mark.parametrize("child", [False, True])
async def test_source_less_fake_copy_preserves_writer_boundary(child: bool) -> None:
    parent_enabled_tools, copied_enabled_tools = ["read_file"], ["grep"]
    parent = FakeConfigOrchestrator(build_test_vibe_config())
    parent.insert_layer(OverridesLayer(data={"enabled_tools": parent_enabled_tools}), 0)
    parent.rebuild()
    owner = parent.policy_owner

    copied = parent._copy_for_child() if child else parent.copy()

    assert (copied.policy_owner != owner) is child
    assert copied.copy().policy_owner == copied.policy_owner
    assert copied.restrictions == parent.restrictions == ()
    assert copied.config == parent.config
    assert copied.config is not parent.config
    assert copied.layers[0] is not parent.layers[0]
    assert copied._bus is not parent._bus

    copied_owner = copied.policy_owner
    assert (
        await copied.set_field(
            "/session_logging/enabled", False, target_layer=OverridesLayer.NAME
        )
        == []
    )
    copied.remove_layer(0)
    copied.insert_layer(OverridesLayer(data={"enabled_tools": copied_enabled_tools}), 0)
    copied.rebuild()

    assert copied.policy_owner == copied_owner
    assert parent.policy_owner == owner
    assert copied.restrictions == parent.restrictions == ()
    assert copied.config.enabled_tools == copied_enabled_tools
    assert parent.config.enabled_tools == parent_enabled_tools
    assert copied._copy_for_child().policy_owner not in {owner, copied_owner}
