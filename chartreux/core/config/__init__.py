from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chartreux.core.config._defaults import (
        AUTO_THEME,
        DEFAULT_API_RETRY_MAX_ELAPSED_TIME,
        DEFAULT_API_TIMEOUT,
        DEFAULT_AUTO_COMPACT_THRESHOLD,
        DEFAULT_MISTRAL_API_ENV_KEY,
        DEFAULT_MISTRAL_SERVER_URL,
        DEFAULT_THEME,
        FALLBACK_THEME,
    )
    from chartreux.core.config.chartreux_schema import (
        DEFAULT_ACTIVE_MODEL_CONFIG,
        ChartreuxConfigSchema,
        create_default_config,
        load_dotenv_values,
        resolve_api_key,
    )
    from chartreux.core.config.default_orchestrator import (
        build_default_orchestrator,
        build_user_config_orchestrator,
    )
    from chartreux.core.config.layer import (
        ConfigLayer,
        ConfigLayerError,
        ConfigPatchApplicationError,
        EmptyLayerError,
        LayerImplementationError,
        LayerNotLoadedError,
        RawConfig,
        TrustResolutionError,
        UntrustedLayerError,
    )
    from chartreux.core.config.layers.agent_profile import AgentProfileLayer
    from chartreux.core.config.layers.default import DefaultConfigLayer
    from chartreux.core.config.models import (
        THINKING_LEVELS,
        MCPHttp,
        MCPOAuth,
        MCPServer,
        MCPStaticAuth,
        MCPStdio,
        MissingAPIKeyError,
        ModelConfig,
        ProjectContextConfig,
        ProviderConfig,
        SessionLoggingConfig,
        ThinkingLevel,
    )
    from chartreux.core.config.patch import (
        AddOperationPatch,
        ConfigPatch,
        PatchOp,
        RemoveOperationPatch,
        ReplaceOperationPatch,
    )
    from chartreux.core.config.schema import (
        ConfigDefinitionError,
        ConfigFragment,
        ConfigSchema,
        DuplicateMergeMetadataError,
        MergeFieldMetadata,
        WithConcatMerge,
        WithConflictMerge,
        WithDeepMerge,
        WithReplaceMerge,
        WithShallowMerge,
        WithUnionMerge,
    )
    from chartreux.core.config.types import (
        MISSING_BACKING_STORE_DATA_FINGERPRINT,
        ConfigChangeCallback,
        ConfigChangeEvent,
        LayerConfigSnapshot,
    )
    from chartreux.core.prompts import MissingPromptFileError

    type ChartreuxConfigSchemaType = type[ChartreuxConfigSchema]

__all__ = [
    "AUTO_THEME",
    "DEFAULT_ACTIVE_MODEL_CONFIG",
    "DEFAULT_API_RETRY_MAX_ELAPSED_TIME",
    "DEFAULT_API_TIMEOUT",
    "DEFAULT_AUTO_COMPACT_THRESHOLD",
    "DEFAULT_MISTRAL_API_ENV_KEY",
    "DEFAULT_MISTRAL_SERVER_URL",
    "DEFAULT_THEME",
    "FALLBACK_THEME",
    "MISSING_BACKING_STORE_DATA_FINGERPRINT",
    "THINKING_LEVELS",
    "AddOperationPatch",
    "AgentProfileLayer",
    "ChartreuxConfigSchema",
    "ChartreuxConfigSchemaType",
    "ConfigChangeCallback",
    "ConfigChangeEvent",
    "ConfigDefinitionError",
    "ConfigFragment",
    "ConfigLayer",
    "ConfigLayerError",
    "ConfigPatch",
    "ConfigPatchApplicationError",
    "ConfigSchema",
    "DefaultConfigLayer",
    "DuplicateMergeMetadataError",
    "EmptyLayerError",
    "LayerConfigSnapshot",
    "LayerImplementationError",
    "LayerNotLoadedError",
    "MCPHttp",
    "MCPOAuth",
    "MCPServer",
    "MCPStaticAuth",
    "MCPStdio",
    "MergeFieldMetadata",
    "MissingAPIKeyError",
    "MissingPromptFileError",
    "ModelConfig",
    "PatchOp",
    "ProjectContextConfig",
    "ProviderConfig",
    "RawConfig",
    "RemoveOperationPatch",
    "ReplaceOperationPatch",
    "SessionLoggingConfig",
    "ThinkingLevel",
    "TrustResolutionError",
    "UntrustedLayerError",
    "WithConcatMerge",
    "WithConflictMerge",
    "WithDeepMerge",
    "WithReplaceMerge",
    "WithShallowMerge",
    "WithUnionMerge",
    "build_default_orchestrator",
    "build_user_config_orchestrator",
    "create_default_config",
    "load_dotenv_values",
    "resolve_api_key",
]

_MAPPING: dict[str, tuple[str, str]] = {
    "AUTO_THEME": ("chartreux.core.config._defaults", "AUTO_THEME"),
    "DEFAULT_API_RETRY_MAX_ELAPSED_TIME": (
        "chartreux.core.config._defaults",
        "DEFAULT_API_RETRY_MAX_ELAPSED_TIME",
    ),
    "DEFAULT_API_TIMEOUT": ("chartreux.core.config._defaults", "DEFAULT_API_TIMEOUT"),
    "DEFAULT_AUTO_COMPACT_THRESHOLD": (
        "chartreux.core.config._defaults",
        "DEFAULT_AUTO_COMPACT_THRESHOLD",
    ),
    "DEFAULT_MISTRAL_API_ENV_KEY": (
        "chartreux.core.config._defaults",
        "DEFAULT_MISTRAL_API_ENV_KEY",
    ),
    "DEFAULT_MISTRAL_SERVER_URL": (
        "chartreux.core.config._defaults",
        "DEFAULT_MISTRAL_SERVER_URL",
    ),
    "DEFAULT_THEME": ("chartreux.core.config._defaults", "DEFAULT_THEME"),
    "FALLBACK_THEME": ("chartreux.core.config._defaults", "FALLBACK_THEME"),
    "build_default_orchestrator": (
        "chartreux.core.config.default_orchestrator",
        "build_default_orchestrator",
    ),
    "build_user_config_orchestrator": (
        "chartreux.core.config.default_orchestrator",
        "build_user_config_orchestrator",
    ),
    "ConfigLayer": ("chartreux.core.config.layer", "ConfigLayer"),
    "ConfigLayerError": ("chartreux.core.config.layer", "ConfigLayerError"),
    "ConfigPatchApplicationError": (
        "chartreux.core.config.layer",
        "ConfigPatchApplicationError",
    ),
    "EmptyLayerError": ("chartreux.core.config.layer", "EmptyLayerError"),
    "LayerImplementationError": (
        "chartreux.core.config.layer",
        "LayerImplementationError",
    ),
    "LayerNotLoadedError": ("chartreux.core.config.layer", "LayerNotLoadedError"),
    "RawConfig": ("chartreux.core.config.layer", "RawConfig"),
    "TrustResolutionError": ("chartreux.core.config.layer", "TrustResolutionError"),
    "UntrustedLayerError": ("chartreux.core.config.layer", "UntrustedLayerError"),
    "AgentProfileLayer": (
        "chartreux.core.config.layers.agent_profile",
        "AgentProfileLayer",
    ),
    "DefaultConfigLayer": (
        "chartreux.core.config.layers.default",
        "DefaultConfigLayer",
    ),
    "THINKING_LEVELS": ("chartreux.core.config.models", "THINKING_LEVELS"),
    "MCPHttp": ("chartreux.core.config.models", "MCPHttp"),
    "MCPOAuth": ("chartreux.core.config.models", "MCPOAuth"),
    "MCPServer": ("chartreux.core.config.models", "MCPServer"),
    "MCPStaticAuth": ("chartreux.core.config.models", "MCPStaticAuth"),
    "MCPStdio": ("chartreux.core.config.models", "MCPStdio"),
    "MissingAPIKeyError": ("chartreux.core.config.models", "MissingAPIKeyError"),
    "ModelConfig": ("chartreux.core.config.models", "ModelConfig"),
    "ProjectContextConfig": ("chartreux.core.config.models", "ProjectContextConfig"),
    "ProviderConfig": ("chartreux.core.config.models", "ProviderConfig"),
    "SessionLoggingConfig": ("chartreux.core.config.models", "SessionLoggingConfig"),
    "ThinkingLevel": ("chartreux.core.config.models", "ThinkingLevel"),
    "AddOperationPatch": ("chartreux.core.config.patch", "AddOperationPatch"),
    "ConfigPatch": ("chartreux.core.config.patch", "ConfigPatch"),
    "PatchOp": ("chartreux.core.config.patch", "PatchOp"),
    "RemoveOperationPatch": ("chartreux.core.config.patch", "RemoveOperationPatch"),
    "ReplaceOperationPatch": ("chartreux.core.config.patch", "ReplaceOperationPatch"),
    "ConfigDefinitionError": ("chartreux.core.config.schema", "ConfigDefinitionError"),
    "ConfigFragment": ("chartreux.core.config.schema", "ConfigFragment"),
    "ConfigSchema": ("chartreux.core.config.schema", "ConfigSchema"),
    "DuplicateMergeMetadataError": (
        "chartreux.core.config.schema",
        "DuplicateMergeMetadataError",
    ),
    "MergeFieldMetadata": ("chartreux.core.config.schema", "MergeFieldMetadata"),
    "WithConcatMerge": ("chartreux.core.config.schema", "WithConcatMerge"),
    "WithConflictMerge": ("chartreux.core.config.schema", "WithConflictMerge"),
    "WithDeepMerge": ("chartreux.core.config.schema", "WithDeepMerge"),
    "WithReplaceMerge": ("chartreux.core.config.schema", "WithReplaceMerge"),
    "WithShallowMerge": ("chartreux.core.config.schema", "WithShallowMerge"),
    "WithUnionMerge": ("chartreux.core.config.schema", "WithUnionMerge"),
    "MISSING_BACKING_STORE_DATA_FINGERPRINT": (
        "chartreux.core.config.types",
        "MISSING_BACKING_STORE_DATA_FINGERPRINT",
    ),
    "ConfigChangeCallback": ("chartreux.core.config.types", "ConfigChangeCallback"),
    "ConfigChangeEvent": ("chartreux.core.config.types", "ConfigChangeEvent"),
    "LayerConfigSnapshot": ("chartreux.core.config.types", "LayerConfigSnapshot"),
    "DEFAULT_ACTIVE_MODEL_CONFIG": (
        "chartreux.core.config.chartreux_schema",
        "DEFAULT_ACTIVE_MODEL_CONFIG",
    ),
    "DEFAULT_TRANSCRIBE_MODELS": (
        "chartreux.core.config.chartreux_schema",
        "DEFAULT_TRANSCRIBE_MODELS",
    ),
    "DEFAULT_TRANSCRIBE_PROVIDERS": (
        "chartreux.core.config.chartreux_schema",
        "DEFAULT_TRANSCRIBE_PROVIDERS",
    ),
    "DEFAULT_TTS_MODELS": (
        "chartreux.core.config.chartreux_schema",
        "DEFAULT_TTS_MODELS",
    ),
    "DEFAULT_TTS_PROVIDERS": (
        "chartreux.core.config.chartreux_schema",
        "DEFAULT_TTS_PROVIDERS",
    ),
    "ChartreuxConfigSchema": (
        "chartreux.core.config.chartreux_schema",
        "ChartreuxConfigSchema",
    ),
    "create_default_config": (
        "chartreux.core.config.chartreux_schema",
        "create_default_config",
    ),
    "load_dotenv_values": (
        "chartreux.core.config.chartreux_schema",
        "load_dotenv_values",
    ),
    "resolve_api_key": ("chartreux.core.config.chartreux_schema", "resolve_api_key"),
    "MissingPromptFileError": ("chartreux.core.prompts", "MissingPromptFileError"),
}


def __getattr__(name: str) -> object:
    if name == "ChartreuxConfigSchemaType":
        from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

        return type[ChartreuxConfigSchema]
    if name in _MAPPING:
        import importlib

        module_name, attr_name = _MAPPING[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
