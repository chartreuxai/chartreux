from __future__ import annotations

from pathlib import Path
from typing import Any

from chartreux.core.config._catalog import validate_catalog_scope
from chartreux.core.config._source_validation import validate_source
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.config.layer import (
    ConfigLayer,
    EmptyLayerError,
    RawConfig,
    UntrustedLayerError,
)
from chartreux.core.config.layers.agent_profile import AgentProfileLayer
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.model_catalog.loader import load_catalog


async def build_default_orchestrator(
    data: dict[str, Any] | None = None,
    *,
    harness_files: HarnessFilesManager | None = None,
) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    """Read and validate the standard layer stack without preparing runtime state.

    Loading never migrates. Explicit migration machinery remains separate from
    bootstrap and reload.

    Priority order (lowest to highest): schema defaults, user TOML, project
    TOML, CHARTREUX_* env vars, agent profile overrides, and runtime overrides.
    The agent-profile slot ships empty and is filled in place by AgentManager.
    Enabled user and project TOML layers are installed together, so a trusted
    project config inherits unspecified values from the user config. Ordinary
    edits target session overrides; durable changes require explicit save().
    """
    manager = harness_files or get_harness_files_manager()
    user_layer = (
        UserConfigLayer(path=manager.user_config_file)
        if "user" in manager.sources
        else None
    )
    project_layer = (
        ProjectConfigLayer(
            path=manager.cwd or Path.cwd(), trust_store=manager.trust_store
        )
        if manager.project_source_enabled
        else None
    )
    override_layer = OverridesLayer(data=data or {})

    def default_layer_resolver() -> ConfigLayer[RawConfig]:
        # Ordinary runtime edits are ephemeral; save() owns explicit persistence.
        return override_layer

    layers: list[ConfigLayer[RawConfig]] = [
        DefaultConfigLayer(schema=ChartreuxConfigSchema),
        *([user_layer] if user_layer is not None else []),
        *([project_layer] if project_layer is not None else []),
        EnvironmentLayer(schema=ChartreuxConfigSchema),
        # Empty slot filled in place by AgentManager when a profile is selected
        AgentProfileLayer(),
        override_layer,
    ]

    # Validate every sparse enabled source before building the effective snapshot.
    for layer in layers:
        try:
            raw = (await layer.load()).model_dump()
        except (EmptyLayerError, UntrustedLayerError):
            continue
        source = f"{layer.name} ({layer.source_locator})"
        validate_catalog_scope(ChartreuxConfigSchema, raw, layer=layer, source=source)
        validate_source(ChartreuxConfigSchema, raw, source=source)
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=layers,
        default_layer_resolver=default_layer_resolver,
        catalog_snapshot=load_catalog(),
        catalog_loader=load_catalog,
    )


async def build_user_config_orchestrator() -> ConfigOrchestrator[ChartreuxConfigSchema]:
    """Build a user-config-only orchestrator for configuration management."""
    manager = get_harness_files_manager()
    user_layer = UserConfigLayer(path=manager.user_config_file)
    validate_source(
        ChartreuxConfigSchema,
        (await user_layer.load()).model_dump(),
        source=f"{user_layer.name} ({user_layer.source_locator})",
    )
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[user_layer],
        default_layer_resolver=lambda: user_layer,
        catalog_snapshot=load_catalog(),
        catalog_loader=load_catalog,
    )
