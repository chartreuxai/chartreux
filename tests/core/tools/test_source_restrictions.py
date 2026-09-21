"""Opt-in manager enforcement, not production root or factory-child wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.layers.agent_profile import AgentProfileLayer
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config


@pytest.mark.parametrize("permission", ["always", "never"])
@pytest.mark.asyncio
async def test_source_restrictions_beat_merge_grants_and_scratch(
    permission: str, tmp_path: Path
) -> None:
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        DefaultConfigLayer(schema=ChartreuxConfigSchema),
        OverridesLayer(
            name="user",
            data={
                "tools": {
                    "write_file": {
                        "permission": permission,
                        "denylist": ["*blocked.txt"],
                    }
                }
            },
        ),
        AgentProfileLayer(
            data={
                "tools": {
                    "write_file": {
                        "permission": "always",
                        "denylist": [],
                        "allowlist": ["*"],
                    }
                }
            }
        ),
    ])
    candidate = await builder.build_candidate()
    manager = ToolManager(
        lambda: candidate.config,
        restriction_getter=lambda: candidate.restrictions,
        cwd=tmp_path,
        scratchpad_dir=tmp_path,
    )
    tool = manager.get("write_file")
    args = tool.validate_arguments({
        "file_path": str(tmp_path / "blocked.txt"),
        "content": "no write",
    })
    context = tool.resolve_permission(args)
    assert context is not None and context.permission == ToolPermission.NEVER
    assert tool.config.permission == (
        ToolPermission.NEVER if permission == "never" else ToolPermission.ALWAYS
    )
    assert candidate.config.tools["write_file"]["denylist"] == []
    tool.config.denylist.clear()
    assert tool.config.denylist == ["*blocked.txt"]
    assert not (tmp_path / "blocked.txt").exists()


@pytest.mark.asyncio
async def test_fixed_inherited_sources_survive_local_mode_replacement() -> None:
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        DefaultConfigLayer(schema=ChartreuxConfigSchema),
        OverridesLayer(
            name="user", data={"tools": {"write_file": {"permission": "never"}}}
        ),
        AgentProfileLayer(data={"tools": {"edit": {"permission": "never"}}}),
    ])
    parent = await builder.build_candidate()
    inherited = tuple(
        source for source in parent.restrictions if source.kind == "source"
    )
    local = parent.restrictions
    config = build_test_vibe_config(
        tools={"write_file": {"permission": "always"}, "edit": {"permission": "always"}}
    )
    manager = ToolManager(
        lambda: config,
        restriction_getter=lambda: local,
        inherited_restrictions=inherited,
    )
    cached = manager.get("write_file")
    # Ordinary profile NEVER is a mutable child default, unlike the inherited
    # parent source denial above.
    assert manager.get_tool_config("edit").permission == ToolPermission.ALWAYS
    local = ()
    assert manager.get("write_file") is cached
    assert cached.config.permission == ToolPermission.NEVER
    assert manager.get_tool_config("edit").permission == ToolPermission.ALWAYS


@pytest.mark.asyncio
async def test_local_source_replacement_does_not_leave_a_sticky_union() -> None:
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        DefaultConfigLayer(schema=ChartreuxConfigSchema),
        OverridesLayer(
            name="user", data={"tools": {"write_file": {"permission": "never"}}}
        ),
    ])
    candidate = await builder.build_candidate()
    restrictions = candidate.restrictions
    config = build_test_vibe_config(tools={"write_file": {"permission": "always"}})
    manager = ToolManager(lambda: config, restriction_getter=lambda: restrictions)
    cached = manager.get("write_file")
    assert cached.config.permission == ToolPermission.NEVER
    restrictions = ()
    assert cached.config.permission == ToolPermission.ALWAYS
    # This deliberately supplies accepted values in a test, not an exposed policy-edit API.
    assert candidate.restrictions[-1].tools[0].denied
