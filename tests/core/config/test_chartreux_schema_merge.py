from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import (
    ChartreuxConfigSchema,
    create_default_config,
)
from chartreux.core.config.layers.user import UserConfigLayer


def test_removed_bypass_tool_permissions_is_rejected() -> None:
    with pytest.raises(ValidationError, match="bypass_tool_permissions.*removed"):
        ChartreuxConfigSchema.model_validate({"bypass_tool_permissions": True})


def test_bash_allowlist_introspection_is_rejected_and_omitted() -> None:
    config = ChartreuxConfigSchema()

    with pytest.raises(ValueError, match="tools.bash.*allowlist.*removed"):
        config.build_tool_allowlist_update("bash", ["echo"])

    assert "allowlist" not in create_default_config()["tools"]["bash"]


@pytest.mark.asyncio
async def test_sparse_nested_config_documents_merge_keys_and_replace_lists(
    tmp_path: Path,
) -> None:
    lower = tmp_path / "lower.toml"
    lower.write_text(
        """
enabled_agents = ["lower-a", "lower-b"]

[project_context]
default_commit_count = 9
timeout_seconds = 4.5

[session_logging]
session_prefix = "lower"
generate_titles = true

[subagents]
idle_ttl_seconds = 90
"""
    )
    higher = tmp_path / "higher.toml"
    higher.write_text(
        """
enabled_agents = ["higher"]

[project_context]
timeout_seconds = 0

[session_logging]
enabled = false

[subagents]
max_idle_agents = 0
"""
    )

    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        UserConfigLayer(path=lower, name="lower"),
        UserConfigLayer(path=higher, name="higher"),
    ])

    config = await builder.build()

    assert config.project_context.default_commit_count == 9
    assert config.project_context.timeout_seconds == 0
    assert config.session_logging.session_prefix == "lower"
    assert config.session_logging.generate_titles is True
    assert config.session_logging.enabled is False
    assert config.subagents.idle_ttl_seconds == 90
    assert config.subagents.max_idle_agents == 0
    assert config.enabled_agents == ["higher"]
