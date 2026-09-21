from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.utils import http

pytestmark = pytest.mark.asyncio


async def make_loop(path: Path, enabled: bool) -> AgentLoop:
    path.write_text(f"enable_system_trust_store = {str(enabled).lower()}\n")
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[
            DefaultConfigLayer(schema=ChartreuxConfigSchema),
            UserConfigLayer(path=path),
            session,
        ],
        default_layer_resolver=lambda: session,
    )
    return AgentLoop(config_orchestrator=orchestrator, cwd=path.parent)


@pytest.mark.parametrize("enabled", [False, True])
async def test_catalog_backed_loop_applies_tls_policy(
    tmp_path: Path, enabled: bool
) -> None:
    loop = await make_loop(tmp_path / "config.toml", enabled)
    try:
        backend = loop.backend
        await backend.__aenter__()
        assert loop.config.enable_system_trust_store is enabled
    finally:
        await loop.aclose()


async def test_schema_validation_does_not_initialize_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http.build_ssl_context.cache_clear()
    monkeypatch.setattr(
        http.ssl,
        "create_default_context",
        lambda: pytest.fail("unexpected TLS initialization"),
    )
    ChartreuxConfigSchema(enable_system_trust_store=True)
    assert http.build_ssl_context.cache_info().misses == 0
