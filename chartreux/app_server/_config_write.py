from __future__ import annotations

from pydantic import JsonValue

from chartreux.app_server._config_introspect import build_field_wires
from chartreux.app_server.protocol import (
    ConfigFieldWire,
    ConfigLayerValueWire,
    ConfigWriteOpWire,
)
from chartreux.core.config._catalog import CATALOG_DEFINITION_FIELDS
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.patch import AddOperationPatch, PatchOp, RemoveOperationPatch


def config_write_targets(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> list[str]:
    """Session edits first, followed by explicit installed persistence targets."""
    targets = [OverridesLayer.NAME]
    for layer in orchestrator.layers:
        if layer.name in targets:
            continue
        if isinstance(layer, ProjectConfigLayer):
            if layer.is_file_discovered and layer.is_trusted is not False:
                targets.append(layer.name)
        elif isinstance(layer, UserConfigLayer):
            targets.append(layer.name)
    return targets


def config_field_write_targets(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], field_name: str
) -> list[str]:
    """Offer catalog writes only to installed user sources, never by name."""
    if field_name in CATALOG_DEFINITION_FIELDS:
        return [
            layer.name
            for layer in orchestrator.layers
            if isinstance(layer, UserConfigLayer)
        ]
    return config_write_targets(orchestrator)


def config_write_projection(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    ops: list[ConfigWriteOpWire],
) -> tuple[list[ConfigFieldWire], dict[str, JsonValue]]:
    """Bounded changed-field outcomes; never serialize arbitrary nested secrets."""

    def redact(name: str, value: JsonValue) -> JsonValue:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if name in {"theme", "active_model"} and isinstance(value, str):
            return value
        return "[redacted]"

    requested = {
        op.path.split("/", 2)[1] for op in ops if op.path.startswith("/")
    } & ChartreuxConfigSchema.model_fields.keys()
    layer_values: dict[str, list[ConfigLayerValueWire]] = {}
    for layer in reversed(orchestrator.layers):
        if layer.cached_data is None:
            continue
        for name, value in layer.cached_data.model_dump(mode="json").items():
            if name in requested:
                layer_values.setdefault(name, []).append(
                    ConfigLayerValueWire(layer=layer.name, value=redact(name, value))
                )
    fields = [
        field
        for field in build_field_wires(orchestrator.config, layer_values)
        if field.name in requested
    ]
    for field in fields:
        field.value = redact(field.name, field.value)
        for layer in field.layer_values:
            layer.value = redact(field.name, layer.value)
    saved: dict[str, JsonValue] = {}
    for op in ops:
        name = op.path.split("/", 2)[1] if op.path.startswith("/") else ""
        if name in requested:
            saved[name] = (
                (redact(name, op.value) if op.path == f"/{name}" else "[redacted]")
                if op.op == "set"
                else None
            )
    return fields, saved


def config_write_ops_to_patches(ops: list[ConfigWriteOpWire]) -> list[PatchOp]:
    """Translate in order without redirecting paths or synthesizing definitions."""
    return [
        RemoveOperationPatch(path=op.path, target_layer_name=op.target_layer)
        if op.op == "remove"
        else AddOperationPatch(
            path=op.path, value=op.value, target_layer_name=op.target_layer
        )
        for op in ops
    ]
