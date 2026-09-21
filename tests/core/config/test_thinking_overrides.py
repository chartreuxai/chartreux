from __future__ import annotations

import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.model_catalog.resolver import ModelResolutionError
from tests.conftest import build_test_vibe_config


@pytest.mark.asyncio
async def test_thinking_overrides_merge_and_materialize_shipped_model() -> None:
    config = build_test_vibe_config(thinking_overrides={"glm-5-2": "high"})
    assert config.thinking_overrides == {"glm-5-2": "high"}
    assert config.get_active_model().thinking == "high"


def test_schema_validation_is_catalog_independent() -> None:
    assert ChartreuxConfigSchema(
        thinking_overrides={"private": "high"}
    ).thinking_overrides == {"private": "high"}


@pytest.mark.asyncio
async def test_unknown_override_patch_preserves_accepted_config() -> None:
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), session],
        default_layer_resolver=lambda: session,
    )
    before = orchestrator.config
    with pytest.raises(ModelResolutionError):
        await orchestrator.set_field("/thinking_overrides/private", "high")
    assert orchestrator.config is before
