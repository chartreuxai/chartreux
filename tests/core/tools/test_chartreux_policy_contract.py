"""Policy-resolution prerequisites; not full filesystem or execution-policy coverage."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config


@pytest.mark.parametrize("configured", list(ToolPermission))
@pytest.mark.parametrize("tool_name", ["write_file", "edit", "unregistered_tool"])
def test_current_tool_policy_is_read_from_live_config(
    configured: ToolPermission, tool_name: str
) -> None:
    config = build_test_vibe_config(tools={tool_name: {"permission": configured.value}})
    manager = ToolManager(lambda: config)

    assert manager.get_tool_config(tool_name).permission == configured
    assert config.tools[tool_name]["permission"] == configured.value


@pytest.mark.parametrize("tool_name", ["write_file", "edit"])
def test_cached_tools_follow_explicit_policy_edits(tool_name: str) -> None:
    config = build_test_vibe_config(tools={tool_name: {"permission": "always"}})
    managers = [ToolManager(lambda: config) for _ in range(2)]
    tools = [manager.get(tool_name) for manager in managers]
    assert all(tool.config.permission == ToolPermission.ALWAYS for tool in tools)

    # Publish new snapshots like the owning configuration service; cached tools
    # read the accepted snapshot without mutable session policy state.
    for permission in (
        ToolPermission.NEVER,
        ToolPermission.ALWAYS,
        ToolPermission.NEVER,
    ):
        config = config.model_copy(
            update={"tools": {tool_name: {"permission": permission.value}}}
        )
        for manager, tool in zip(managers, tools, strict=True):
            assert manager.get(tool_name) is tool
            assert tool.config.permission == permission


def test_invalid_tool_permission_is_not_hidden_by_runtime_defaults() -> None:
    config = build_test_vibe_config(tools={"write_file": {"permission": "invalid"}})
    manager = ToolManager(lambda: config)

    with pytest.raises(ValidationError, match="permission"):
        manager.get_tool_config("write_file")
