from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.patch import AddOperationPatch, RemoveOperationPatch

ACTIVE_MODEL_PATH = "/active_model"
THINKING_OVERRIDES_PATH = "/thinking_overrides"


def stored_session_active_model(config: Mapping[str, JsonValue] | None) -> str | None:
    """Return a concrete model alias from a session config snapshot."""
    if config is None:
        return None
    active_model = config.get("active_model")
    return active_model if isinstance(active_model, str) and active_model else None


def override_active_model(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> str | None:
    layer = next(
        (layer for layer in orchestrator.layers if layer.name == OverridesLayer.NAME),
        None,
    )
    if layer is None:
        return None
    data = layer.cached_data
    if data is None:
        return None
    active_model = getattr(data, "active_model", None)
    return active_model if isinstance(active_model, str) and active_model else None


def active_model_is_pinned(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> bool:
    return override_active_model(orchestrator) is not None


async def set_session_active_model_override(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    active_model: str,
    *,
    reason: str,
) -> list[BaseException]:
    if override_active_model(orchestrator) == active_model:
        return []
    return await orchestrator.apply_patch(
        [
            AddOperationPatch(
                path=ACTIVE_MODEL_PATH,
                value=active_model,
                target_layer_name=OverridesLayer.NAME,
            )
        ],
        reason=reason,
    )


async def clear_session_active_model_override(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], *, reason: str
) -> list[BaseException]:
    if override_active_model(orchestrator) is None:
        return []
    return await orchestrator.apply_patch(
        [
            RemoveOperationPatch(
                path=ACTIVE_MODEL_PATH, target_layer_name=OverridesLayer.NAME
            )
        ],
        reason=reason,
    )


def config_active_model(metadata: Mapping[str, Any]) -> str | None:
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict):
        return None
    return stored_session_active_model(raw_config)


def config_thinking_overrides(metadata: Mapping[str, Any]) -> dict[str, str]:
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict):
        return {}
    value = raw_config.get("thinking_overrides")
    if not isinstance(value, dict):
        return {}
    return {
        alias: level
        for alias, level in value.items()
        if isinstance(alias, str) and isinstance(level, str)
    }


async def restore_session_thinking_overrides(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    overrides: Mapping[str, str],
    *,
    reason: str,
) -> list[BaseException]:
    layer = next(
        (layer for layer in orchestrator.layers if layer.name == OverridesLayer.NAME),
        None,
    )
    current = getattr(layer.cached_data, "thinking_overrides", None) if layer else None
    if current == overrides or (not current and not overrides):
        return []
    patch = (
        AddOperationPatch(
            path=THINKING_OVERRIDES_PATH,
            value=dict(overrides),
            target_layer_name=OverridesLayer.NAME,
        )
        if overrides
        else RemoveOperationPatch(
            path=THINKING_OVERRIDES_PATH, target_layer_name=OverridesLayer.NAME
        )
    )
    return await orchestrator.apply_patch([patch], reason=reason)
