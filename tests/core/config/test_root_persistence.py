"""Roots-specific persistence coverage.

The shared persistence pairs (save/read conflicts, cancellation, durability,
canonical accumulation, generic field rejection) migrated to
``_grant_config_harness.py`` in stage 12a; this module keeps the one
roots-specific concern: a non-persistable (relative) root argument.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator


async def make_orchestrator(path: Path) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    user = UserConfigLayer(path=path)
    session = OverridesLayer(data={})
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: session,
    )


def live_state(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> tuple[object, object, object, object]:
    layer = orchestrator.get_layer("user-toml")
    return (
        orchestrator.config,
        orchestrator.restrictions,
        orchestrator.accepted_token,
        (layer.cached_data, layer.fingerprint),
    )


@pytest.mark.asyncio
async def test_invalid_root_argument_is_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)

    result = await orchestrator.save_project_root_grant(
        project=tmp_path, root=Path("relative-root"), expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert path.read_text() == 'theme = "dark"\n'
    assert live_state(orchestrator) == before
