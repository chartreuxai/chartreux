from __future__ import annotations

from collections.abc import Mapping, Sequence
import enum
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic_core import to_jsonable_python

from chartreux.app_server.protocol import (
    ConfigFieldKind,
    ConfigFieldWire,
    ConfigLayerValueWire,
)
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import ConfigLayer, ConfigLayerError, RawConfig
from chartreux.core.config.patch import escape_json_pointer_token

DEFAULT_ORIGIN = "default"
AUTO_COMPACT_THRESHOLD = "auto_compact_threshold"

# Internal fields populated at runtime (not by the user) that should never be
# rendered in the settings UI.
HIDDEN_SETTINGS: frozenset[str] = frozenset({"tools"})

POPULAR_SETTINGS: frozenset[str] = frozenset({
    "active_model",
    "theme",
    "mcp_servers",
    "auto_compact_threshold",
    "autocopy_to_clipboard",
    "ask_confirmation_on_exit",
    "enable_notifications",
})

_SCALAR_KINDS: tuple[tuple[type, ConfigFieldKind], ...] = (
    (bool, ConfigFieldKind.BOOL),
    (int, ConfigFieldKind.INT),
    (float, ConfigFieldKind.FLOAT),
    (str, ConfigFieldKind.STR),
)


def classify_annotation(annotation: Any) -> tuple[ConfigFieldKind, tuple[str, ...]]:
    """Map a field annotation to its editor kind and any enum choices."""
    if get_origin(annotation) in {Union, UnionType}:
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
    if get_origin(annotation) is list:
        args = get_args(annotation)
        item = args[0] if args else None
        if isinstance(item, type) and issubclass(
            item, (str, int, float, bool, Path, enum.Enum)
        ):
            return ConfigFieldKind.LIST, ()
        return ConfigFieldKind.COMPLEX, ()
    if not isinstance(annotation, type):
        return ConfigFieldKind.COMPLEX, ()
    if issubclass(annotation, enum.Enum):
        return ConfigFieldKind.ENUM, tuple(str(member.value) for member in annotation)
    return next(
        (
            (kind, ())
            for scalar_type, kind in _SCALAR_KINDS
            if issubclass(annotation, scalar_type)
        ),
        (ConfigFieldKind.COMPLEX, ()),
    )


async def collect_layer_values(
    layers: Sequence[ConfigLayer[RawConfig]],
) -> dict[str, list[ConfigLayerValueWire]]:
    """Collect each field's per-layer values, highest priority first."""
    values: dict[str, list[ConfigLayerValueWire]] = {}
    for layer in reversed(layers):
        try:
            data = (await layer.load()).model_dump(mode="json")
        except ConfigLayerError:
            continue
        for name, value in data.items():
            values.setdefault(name, []).append(
                ConfigLayerValueWire(layer=layer.name, value=value)
            )
        models = data.get("models")
        if not isinstance(models, Mapping):
            continue
        for alias, model in models.items():
            if not isinstance(alias, str) or not isinstance(model, Mapping):
                continue
            if AUTO_COMPACT_THRESHOLD not in model:
                continue
            path = _model_field_path(alias, AUTO_COMPACT_THRESHOLD)
            values.setdefault(path, []).append(
                ConfigLayerValueWire(
                    layer=layer.name, value=model[AUTO_COMPACT_THRESHOLD]
                )
            )
    return values


def build_field_wires(
    config: ChartreuxConfigSchema,
    layer_values: Mapping[str, list[ConfigLayerValueWire]],
    *,
    path_prefix: str = "",
    popular: frozenset[str] = frozenset(),
    writable_targets: Mapping[str, Sequence[str]] | None = None,
) -> list[ConfigFieldWire]:
    """Build the wire description of every config field for the settings UI."""
    json_values = config.model_dump(mode="json")
    wires: list[ConfigFieldWire] = []
    for name, info in type(config).model_fields.items():
        if name in HIDDEN_SETTINGS:
            continue
        kind, choices = classify_annotation(info.annotation)
        value = json_values.get(name)
        path = f"{path_prefix}/{escape_json_pointer_token(name)}"
        values = list(layer_values.get(name, []))
        description = (info.description or "").strip()
        if not info.is_required() and not any(
            entry.layer == DEFAULT_ORIGIN for entry in values
        ):
            values.append(
                ConfigLayerValueWire(
                    layer=DEFAULT_ORIGIN,
                    value=to_jsonable_python(
                        info.get_default(call_default_factory=True)
                    ),
                )
            )
        wires.append(
            ConfigFieldWire(
                name=name,
                kind=kind,
                description=description,
                value=value,
                path=path,
                popular=name in popular,
                enum_choices=list(choices),
                layer_values=values,
                writable_targets=list((writable_targets or {}).get(name, ())),
            )
        )
    return wires


def _model_field_path(alias: str, field: str, *, prefix: str = "") -> str:
    return (
        f"{prefix}/models/{escape_json_pointer_token(alias)}/"
        f"{escape_json_pointer_token(field)}"
    )
