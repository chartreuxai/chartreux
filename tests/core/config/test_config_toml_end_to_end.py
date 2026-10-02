from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, ValidationError
import pytest

from chartreux.core.config.schema import (
    ConfigFragment,
    ConfigSchema,
    WithReplaceMerge,
    WithUnionMerge,
)


class ModelsFragment(ConfigFragment):
    active_model: Annotated[str, WithReplaceMerge()] = "default-model"
    models: Annotated[list[dict[str, str]], WithUnionMerge(merge_key="alias")] = Field(
        default_factory=list
    )
    providers: Annotated[list[dict[str, str]], WithUnionMerge(merge_key="name")] = (
        Field(default_factory=list)
    )


class MinimalSchema(ConfigSchema):
    models: ModelsFragment = Field(default_factory=ModelsFragment)


@pytest.mark.asyncio
async def test_toml_to_typed_config_end_to_end(tmp_working_directory: Path) -> None:
    toml_path = tmp_working_directory / "config.toml"
    toml_path.write_text(
        """\
[models]
active_model = "mistral-large"

[[models.models]]
alias = "mistral-large"
provider = "mistral"

[[models.providers]]
name = "mistral"
api_base = "https://api.mistral.ai/v1"
"""
    )

    from chartreux.core.config.layers.user import UserConfigLayer
    from chartreux.core.config.orchestrator import ConfigOrchestrator

    layer = UserConfigLayer(path=toml_path)
    orchestrator = await ConfigOrchestrator.create(
        schema=MinimalSchema, layers=[layer], default_layer_resolver=lambda: layer
    )

    assert orchestrator.config.models.active_model == "mistral-large"
    assert orchestrator.config.models.models == [
        {"alias": "mistral-large", "provider": "mistral"}
    ]
    # Immutability (frozen schema)
    with pytest.raises(ValidationError, match="frozen"):
        orchestrator.config.models = ModelsFragment()  # type: ignore[misc]

    # Reload picks up changes
    toml_path.write_text(
        """\
[models]
active_model = "codestral"
"""
    )
    await orchestrator.reload()
    assert orchestrator.config.models.active_model == "codestral"


@pytest.mark.asyncio
async def test_toml_subagent_cap_sparse_merge_and_reload(tmp_path: Path) -> None:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
    from chartreux.core.config.layers.overrides import OverridesLayer
    from chartreux.core.config.layers.user import UserConfigLayer
    from chartreux.core.config.orchestrator import ConfigOrchestrator

    path = tmp_path / "config.toml"
    path.write_text("[subagents]\nmax_running_subagents = 2\n")
    user = UserConfigLayer(path=path)
    retention = OverridesLayer(
        data={"subagents": {"idle_ttl_seconds": 123, "max_idle_agents": 7}}
    )
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[retention, user],
        default_layer_resolver=lambda: user,
    )
    assert orchestrator.config.subagents.max_running_subagents == 2
    assert orchestrator.config.subagents.idle_ttl_seconds == 123
    assert orchestrator.config.subagents.max_idle_agents == 7
    path.write_text("[subagents]\nmax_running_subagents = 3\n")
    await orchestrator.reload()
    assert orchestrator.config.subagents.max_running_subagents == 3
    assert orchestrator.config.subagents.idle_ttl_seconds == 123
    assert orchestrator.config.subagents.max_idle_agents == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["0", "-1", "true", "false", "1.0", '"1"'])
async def test_toml_subagent_cap_rejects_invalid_values(
    tmp_path: Path, value: str
) -> None:
    from chartreux.core.config.builder import ConfigBuilder
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
    from chartreux.core.config.layers.user import UserConfigLayer

    path = tmp_path / "config.toml"
    path.write_text(f"[subagents]\nmax_running_subagents = {value}\n")
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layer(UserConfigLayer(path=path))
    with pytest.raises(ValidationError, match="max_running_subagents"):
        await builder.build()
