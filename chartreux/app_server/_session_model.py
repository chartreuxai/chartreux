from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.patch import AddOperationPatch, RemoveOperationPatch

ACTIVE_MODEL_PATH = "/active_model"
THINKING_OVERRIDES_PATH = "/thinking_overrides"
AGENT_IDENTITY_PATH = "/agent/identity"


def _escape_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _overrides_layer(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> ConfigLayer[RawConfig] | None:
    return next(
        (layer for layer in orchestrator.layers if layer.name == OverridesLayer.NAME),
        None,
    )


def _canonical_model_key(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], alias: str
) -> str:
    """Map an alias to the canonical base-model key thinking overrides store.

    The orchestrator canonicalizes ``/thinking_overrides/<alias>`` writes the
    same way, so the read-back must use the same key; without a catalog
    snapshot the raw alias is the key.
    """
    snapshot = getattr(orchestrator.config, "catalog_snapshot", None)
    if snapshot is None:
        return alias
    from chartreux.core.model_catalog.resolver import ModelResolver

    try:
        return ModelResolver(snapshot).canonicalize(alias)
    except Exception:
        return alias


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


def override_thinking_level(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], alias: str
) -> str | None:
    """Return the session-scoped thinking override stored for ``alias``."""
    layer = _overrides_layer(orchestrator)
    if layer is None:
        return None
    data = layer.cached_data
    if data is None:
        return None
    overrides = getattr(data, "thinking_overrides", None)
    if not isinstance(overrides, dict):
        return None
    level = overrides.get(_canonical_model_key(orchestrator, alias))
    if level is None:
        level = overrides.get(alias)
    return level if isinstance(level, str) and level else None


async def set_session_thinking_override(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    alias: str,
    level: str,
    *,
    reason: str,
) -> list[BaseException]:
    """Per-key thinking override at ``/thinking_overrides/<alias>``.

    Unlike ``restore_session_thinking_overrides`` this never replaces the
    whole map, so other aliases' session overrides survive.
    """
    if override_thinking_level(orchestrator, alias) == level:
        return []
    return await orchestrator.apply_patch(
        [
            AddOperationPatch(
                path=f"{THINKING_OVERRIDES_PATH}/{_escape_pointer_token(alias)}",
                value=level,
                target_layer_name=OverridesLayer.NAME,
            )
        ],
        reason=reason,
    )


async def remove_session_thinking_override(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], alias: str, *, reason: str
) -> list[BaseException]:
    if override_thinking_level(orchestrator, alias) is None:
        return []
    return await orchestrator.apply_patch(
        [
            RemoveOperationPatch(
                path=f"{THINKING_OVERRIDES_PATH}/{_escape_pointer_token(alias)}",
                target_layer_name=OverridesLayer.NAME,
            )
        ],
        reason=reason,
    )


def override_agent_identity(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> str | None:
    layer = _overrides_layer(orchestrator)
    if layer is None:
        return None
    data = layer.cached_data
    if data is None:
        return None
    agent = getattr(data, "agent", None)
    if isinstance(agent, dict):
        identity = agent.get("identity")
    else:
        identity = getattr(agent, "identity", None) if agent is not None else None
    return identity if isinstance(identity, str) and identity else None


async def set_session_agent_identity_override(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    identity: str,
    *,
    reason: str,
) -> list[BaseException]:
    if override_agent_identity(orchestrator) == identity:
        return []
    return await orchestrator.apply_patch(
        [
            AddOperationPatch(
                path=AGENT_IDENTITY_PATH,
                value=identity,
                target_layer_name=OverridesLayer.NAME,
            )
        ],
        reason=reason,
    )


async def clear_session_agent_identity_override(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], *, reason: str
) -> list[BaseException]:
    if override_agent_identity(orchestrator) is None:
        return []
    return await orchestrator.apply_patch(
        [
            RemoveOperationPatch(
                path=AGENT_IDENTITY_PATH, target_layer_name=OverridesLayer.NAME
            )
        ],
        reason=reason,
    )


def config_active_model(metadata: Mapping[str, Any]) -> str | None:
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict):
        return None
    return stored_session_active_model(raw_config)


def config_thinking_overrides(metadata: Mapping[str, Any]) -> dict[str, str] | None:
    """Distinguish a historical omission from an explicit session map.

    As with historical active-model parsing, malformed containers and entries
    are filtered, not coerced. String keys and values still pass through the
    orchestrator's canonical-model and thinking-level validation on restoration.
    """
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict) or "thinking_overrides" not in raw_config:
        return None
    value = raw_config["thinking_overrides"]
    if not isinstance(value, dict):
        return {}
    return {
        alias: level
        for alias, level in value.items()
        if isinstance(alias, str) and isinstance(level, str)
    }


async def restore_session_thinking_overrides(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    overrides: Mapping[str, str] | None,
    *,
    reason: str,
) -> list[BaseException]:
    layer = next(
        (layer for layer in orchestrator.layers if layer.name == OverridesLayer.NAME),
        None,
    )
    current = getattr(layer.cached_data, "thinking_overrides", None) if layer else None
    # Omission removes the previous session's map, exposing the launch baseline
    # in lower layers. An explicit empty map is retained as session state; shallow
    # config merging still preserves lower-layer choices in the effective map.
    if current == overrides:
        return []
    patch = (
        AddOperationPatch(
            path=THINKING_OVERRIDES_PATH,
            value=dict(overrides),
            target_layer_name=OverridesLayer.NAME,
        )
        if overrides is not None
        else RemoveOperationPatch(
            path=THINKING_OVERRIDES_PATH, target_layer_name=OverridesLayer.NAME
        )
    )
    return await orchestrator.apply_patch([patch], reason=reason)
