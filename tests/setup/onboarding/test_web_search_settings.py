"""Onboarding search settings use config sources without starting an agent loop."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.setup.onboarding.web_search_settings import OnboardingWebSearchSettings


@pytest.mark.asyncio
async def test_search_settings_save_is_revision_checked_and_rejects_shadowed_invalid_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text("")
    user = UserConfigLayer(path=path, name="user-test")
    session = OverridesLayer(data={"tools": {"web_search": {"timeout": 42}}})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: session,
    )
    service = OnboardingWebSearchSettings(orchestrator)
    before = await service.read()
    assert before.user_revision
    assert before.web_search is not None

    invalid = await service.save({"tools.web_search.timeout": 0}, before.user_revision)
    assert invalid.persistence == "not_saved"
    assert invalid.error == "validation"
    assert path.read_text() == ""

    saved = await service.save(
        {"tools.web_search.provider": "duckduckgo"}, before.user_revision
    )
    assert saved.persistence == "saved"
    assert saved.application == "applied"
    assert saved.snapshot is not None
    assert saved.snapshot.web_search is not None
    assert saved.snapshot.web_search.readiness == "ready"
    assert (
        tomllib.loads(path.read_text())["tools"]["web_search"]["provider"]
        == "duckduckgo"
    )

    stale = await service.save(
        {"tools.web_search.max_results": 6}, before.user_revision
    )
    assert stale.persistence == "not_saved"
    assert stale.error == "conflict"
