"""Candidate-only source provenance; no accepted runtime mutation API is exercised."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.layers.agent_profile import AgentProfileLayer
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.types import LayerConfigSnapshot


def make_builder(
    *layers: ConfigLayer[RawConfig],
) -> ConfigBuilder[ChartreuxConfigSchema]:
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        DefaultConfigLayer(schema=ChartreuxConfigSchema, name="base"),
        *layers,
    ])
    return builder


@pytest.mark.asyncio
async def test_candidate_preserves_shadowed_restrictions_without_changing_merge() -> (
    None
):
    builder = make_builder(
        OverridesLayer(
            name="user",
            data={
                "tools": {
                    "write_file": {"permission": "never", "denylist": ["*private*"]}
                }
            },
        ),
        AgentProfileLayer(
            data={"tools": {"write_file": {"permission": "always", "denylist": []}}}
        ),
    )
    candidate = await builder.build_candidate()
    assert candidate.config == await builder.build()
    assert candidate.config.tools["write_file"] == {
        "permission": "always",
        "denylist": [],
    }
    user = next(
        source for source in candidate.restrictions if source.layer_name == "user"
    )
    assert user.tools[0].denied is True
    assert user.tools[0].denylist == ("*private*",)
    assert user.kind == "source"
    mode = next(
        source
        for source in candidate.restrictions
        if source.layer_name == "agent-profile"
    )
    assert mode.kind == "mode"
    assert mode.tools == ()
    frozen_field = "denied"
    with pytest.raises(FrozenInstanceError):
        setattr(user.tools[0], frozen_field, False)
    candidate.config.tools.clear()
    assert user.tools[0].denied is True


@pytest.mark.parametrize(
    "value", [{"permission": "invalid"}, {"denylist": [23]}, {"denylist": None}]
)
@pytest.mark.asyncio
async def test_shadowed_invalid_restriction_has_source_and_field(
    value: dict[str, Any],
) -> None:
    builder = make_builder(
        OverridesLayer(name="bad-source", data={"tools": {"write_file": value}}),
        OverridesLayer(
            name="higher",
            data={"tools": {"write_file": {"permission": "always", "denylist": []}}},
        ),
    )
    with pytest.raises(ValueError, match=r"bad-source.*tools.write_file"):
        await builder.build_candidate()


@pytest.mark.asyncio
async def test_preview_does_not_publish_or_poison_accepted_layer_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[tools.write_file]\npermission = "never"\n')
    layer = UserConfigLayer(path=path)
    builder = make_builder(layer)
    candidate = await builder.build_candidate()
    config = candidate.config
    orchestrator = ConfigOrchestrator(builder, candidate, lambda: layer)
    before = layer.cached_data.model_dump() if layer.cached_data else None
    fingerprint = layer.fingerprint
    path.write_text('[tools.write_file]\npermission = "always"\n')
    candidate = await orchestrator.preview_candidate(force_load=True)
    assert candidate.config.tools["write_file"]["permission"] == "always"
    assert orchestrator.config is config
    assert layer.cached_data is not None and layer.cached_data.model_dump() == before
    assert layer.fingerprint == fingerprint
    source = next(
        source for source in candidate.restrictions if source.layer_name == layer.name
    )
    assert source.locator == str(path)
    assert source.store_fingerprint is not None
    assert source.content_fingerprint
    assert source.tools == ()
    assert path.read_text() == '[tools.write_file]\npermission = "always"\n'


@pytest.mark.asyncio
async def test_candidate_rebuild_replaces_source_instead_of_accumulating_history() -> (
    None
):
    builder = make_builder(
        OverridesLayer(name="user", data={"tools": {"edit": {"permission": "never"}}})
    )
    first = await builder.build_candidate()
    second = await builder.build_candidate(
        layer_overrides={
            "user": RawConfig.model_validate({
                "tools": {"edit": {"permission": "always"}}
            })
        }
    )
    old = next(s for s in first.restrictions if s.layer_name == "user")
    new = next(s for s in second.restrictions if s.layer_name == "user")
    assert old.locator == new.locator
    assert old.content_fingerprint != new.content_fingerprint
    assert new.store_fingerprint is None
    assert old.tools[0].denied
    assert new.tools == ()
    assert (await builder.build_candidate()).restrictions == first.restrictions


class FakeUntrustedLayer(ConfigLayer[RawConfig]):
    async def _build_config_snapshot(self) -> LayerConfigSnapshot:
        raise AssertionError("untrusted source must not load")

    async def _save_to_store(self, _next_config: RawConfig) -> str:
        raise AssertionError("candidate must not persist")


@pytest.mark.asyncio
async def test_candidate_override_cannot_bypass_source_trust() -> None:
    builder = make_builder(FakeUntrustedLayer(name="untrusted"))
    candidate = await builder.build_candidate(
        layer_overrides={
            "untrusted": RawConfig.model_validate({
                "tools": {"write_file": {"permission": "never"}}
            })
        }
    )
    assert all(source.layer_name != "untrusted" for source in candidate.restrictions)


@pytest.mark.asyncio
async def test_failed_candidate_keeps_accepted_config_and_can_recover() -> None:
    layer = OverridesLayer(
        name="user", data={"tools": {"write_file": {"permission": "never"}}}
    )
    builder = make_builder(layer)
    candidate = await builder.build_candidate()
    config = candidate.config
    orchestrator = ConfigOrchestrator(builder, candidate, lambda: layer)
    with pytest.raises(ValueError):
        await orchestrator.preview_candidate(
            layer_overrides={
                "user": RawConfig.model_validate({
                    "tools": {"write_file": {"permission": "invalid"}}
                })
            }
        )
    assert orchestrator.config is config
    assert (await orchestrator.preview_candidate()).config == config


@pytest.mark.asyncio
async def test_candidate_rejects_ambiguous_or_unknown_source_names() -> None:
    builder = make_builder(OverridesLayer(name="user", data={}))
    with pytest.raises(ValueError, match="Unknown candidate layers: typo"):
        await builder.build_candidate(layer_overrides={"typo": RawConfig()})
    builder.add_layer(OverridesLayer(name="user", data={}))
    with pytest.raises(ValueError, match="unique layer names"):
        await builder.build_candidate()
