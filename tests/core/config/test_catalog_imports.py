from __future__ import annotations

import pytest

from chartreux.core.config.default_orchestrator import build_default_orchestrator


@pytest.mark.asyncio
async def test_default_builder_attaches_shipped_catalog() -> None:
    orchestrator = await build_default_orchestrator()
    assert orchestrator.config.catalog_snapshot is not None
    assert orchestrator.config.get_active_model().alias == "glm-5-3"


def test_legacy_schema_compatibility_exports_are_removed() -> None:
    import chartreux.core.config as config

    for name in ("DEFAULT_MODELS", "DEFAULT_PROVIDERS", "get_persisted_config"):
        assert name not in config.__all__
        assert not hasattr(config, name)


def test_public_config_import_exposes_builder() -> None:
    from chartreux.core.config import build_default_orchestrator as public_builder

    assert public_builder is build_default_orchestrator
