from __future__ import annotations

from collections.abc import Callable

import pytest

from chartreux.app_server._config_introspect import (
    DEFAULT_ORIGIN,
    HIDDEN_SETTINGS,
    POPULAR_SETTINGS,
    build_field_wires,
    classify_annotation,
    collect_layer_values,
)
from chartreux.app_server.protocol import ConfigFieldKind, ConfigLayerValueWire
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import ModelConfig


def test_popular_settings_are_valid_fields() -> None:
    unknown = POPULAR_SETTINGS - set(ChartreuxConfigSchema.model_fields)
    assert not unknown, f"POPULAR_SETTINGS names not in schema: {sorted(unknown)}"


def test_hidden_settings_are_valid_fields() -> None:
    unknown = HIDDEN_SETTINGS - set(ChartreuxConfigSchema.model_fields)
    assert not unknown, f"HIDDEN_SETTINGS names not in schema: {sorted(unknown)}"


@pytest.mark.parametrize(
    ("annotation", "kind"),
    [
        (bool, ConfigFieldKind.BOOL),
        (int, ConfigFieldKind.INT),
        (float, ConfigFieldKind.FLOAT),
        (str, ConfigFieldKind.STR),
        (str | None, ConfigFieldKind.STR),
        (list[str], ConfigFieldKind.LIST),
        (list[int], ConfigFieldKind.LIST),
        (dict[str, int], ConfigFieldKind.COMPLEX),
        (list[ModelConfig], ConfigFieldKind.COMPLEX),
        (ModelConfig | None, ConfigFieldKind.COMPLEX),
    ],
)
def test_classify_annotation(annotation: object, kind: ConfigFieldKind) -> None:
    assert classify_annotation(annotation)[0] is kind


def test_build_field_wires_covers_schema_and_defaults(
    make_config: Callable[..., ChartreuxConfigSchema],
) -> None:
    config = make_config()
    by_name = {wire.name: wire for wire in build_field_wires(config, {})}

    assert by_name.keys() == set(type(config).model_fields) - HIDDEN_SETTINGS
    assert not HIDDEN_SETTINGS & by_name.keys()
    assert by_name["autocopy_to_clipboard"].kind is ConfigFieldKind.BOOL
    assert "models" not in by_name
    assert by_name["thinking_overrides"].kind is ConfigFieldKind.COMPLEX
    assert by_name["theme"].path == "/theme"
    assert all(wire.origin == DEFAULT_ORIGIN for wire in by_name.values())
    assert all(wire.writable_targets == [] for wire in by_name.values())


def test_field_targets_are_copied_without_inference() -> None:
    targets = {
        "thinking_overrides": ["renamed-user"],
        "auto_compact_threshold": ["session"],
    }
    by_name = {
        wire.name: wire
        for wire in build_field_wires(
            ChartreuxConfigSchema(), {}, path_prefix="/config", writable_targets=targets
        )
    }
    assert by_name["thinking_overrides"].writable_targets == ["renamed-user"]
    assert by_name["auto_compact_threshold"].writable_targets == ["session"]
    assert by_name["auto_compact_threshold"].path == "/config/auto_compact_threshold"
    assert by_name["theme"].writable_targets == []
    targets["thinking_overrides"].append("not-offered")
    assert by_name["thinking_overrides"].writable_targets == ["renamed-user"]


def test_build_field_wires_resolves_layers(
    make_config: Callable[..., ChartreuxConfigSchema],
) -> None:
    config = make_config()
    wires = build_field_wires(
        config, {"theme": [ConfigLayerValueWire(layer="user-toml", value="dark")]}
    )
    by_name = {wire.name: wire for wire in wires}

    # Model/theme choices are resolved client-side, so the server keeps them STR.
    assert by_name["active_model"].kind is ConfigFieldKind.STR
    assert by_name["theme"].origin == "user-toml"
    assert by_name["theme"].layer_values[-1].layer == DEFAULT_ORIGIN


def test_build_field_wires_keeps_ordinary_compaction_threshold(
    make_config: Callable[..., ChartreuxConfigSchema],
) -> None:
    config = make_config(
        active_model="custom",
        auto_compact_threshold=50_000,
        models=[
            ModelConfig(
                name="custom",
                provider="mistral",
                alias="custom",
                auto_compact_threshold=200_000,
            )
        ],
    )
    wires = build_field_wires(
        config,
        {
            "auto_compact_threshold": [
                ConfigLayerValueWire(layer="overrides", value=50_000)
            ],
            "/models/custom/auto_compact_threshold": [
                ConfigLayerValueWire(layer="user-toml", value=200_000),
                ConfigLayerValueWire(layer=DEFAULT_ORIGIN, value=200_000),
            ],
        },
    )

    threshold = next(wire for wire in wires if wire.name == "auto_compact_threshold")
    assert threshold.value == 50_000
    assert threshold.path == "/auto_compact_threshold"
    assert threshold.origin == "overrides"


@pytest.mark.asyncio
async def test_collect_layer_values_groups_fields_by_priority(tmp_path) -> None:
    missing = UserConfigLayer(path=tmp_path / "missing.toml", name="missing")
    low = OverridesLayer(data={"theme": "a"}, name="user-toml")
    high = OverridesLayer(data={"theme": "b", "api_timeout": 1.0}, name="overrides")

    values = await collect_layer_values([missing, low, high])

    assert [(entry.layer, entry.value) for entry in values["theme"]] == [
        ("overrides", "b"),
        ("user-toml", "a"),
    ]
    assert [(entry.layer, entry.value) for entry in values["api_timeout"]] == [
        ("overrides", 1.0)
    ]
