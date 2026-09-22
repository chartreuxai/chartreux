from __future__ import annotations

from pydantic import ValidationError
import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.default_orchestrator import build_default_orchestrator
from chartreux.core.config.layers.overrides import OverridesLayer


@pytest.mark.asyncio
async def test_shipped_catalog_is_attached_to_merged_config() -> None:
    config = (await build_default_orchestrator()).config
    assert config.catalog_snapshot is not None
    assert config.get_active_model().alias == "glm-5-3"


@pytest.mark.asyncio
async def test_legacy_model_definition_is_rejected_before_merge() -> None:
    from chartreux.core.config.builder import ConfigBuilder

    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layer(OverridesLayer(data={"models": {"glm-5-2": {"thinking": "low"}}}))
    with pytest.raises(ValidationError, match=r"chartreux models migrate"):
        await builder.build()
