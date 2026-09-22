from __future__ import annotations

from collections.abc import Callable, MutableMapping
import os
from pathlib import Path
from typing import Annotated, Any

from dotenv import dotenv_values
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    PrivateAttr,
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from chartreux.core.config._defaults import (
    DEFAULT_API_CONNECT_TIMEOUT,
    DEFAULT_API_POOL_TIMEOUT,
    DEFAULT_API_RETRY_MAX_ELAPSED_TIME,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_WRITE_TIMEOUT,
    DEFAULT_AUTO_COMPACT_THRESHOLD,
    DEFAULT_THEME,
)

# DEFAULT_LOG_LEVEL is not imported here to avoid a circular dependency
# (vibe_schema.py imports from chartreux.observability.logging). The constant
# lives in chartreux.config_values.
from chartreux.core.config.models import (
    MCPServer,
    MissingAPIKeyError,
    ModelConfig,
    ProjectContextConfig,
    ProviderConfig,
    SessionLoggingConfig,
    SubagentsConfig,
    ThinkingLevel,
    normalize_authorized_roots,
)
from chartreux.core.config.schema import (
    ConfigSchema,
    WithConcatMerge,
    WithDeepMerge,
    WithReplaceMerge,
    WithShallowMerge,
)
from chartreux.core.llm_models import Backend
from chartreux.core.paths import GLOBAL_ENV_FILE
from chartreux.core.prompts import (
    SystemPrompt,
    UtilityPrompt,
    load_prompt,
    load_system_prompt,
)
from chartreux.utils.api_keys import resolve_api_key


def load_dotenv_values(
    env_path: Path = GLOBAL_ENV_FILE.path,
    environ: MutableMapping[str, str] = os.environ,
) -> None:
    # We allow FIFO path to support some environment management solutions (e.g. https://developer.1password.com/docs/environments/local-env-file/)
    if not env_path.is_file() and not env_path.is_fifo():
        return

    env_vars = dotenv_values(env_path)
    for key, value in env_vars.items():
        if not value:
            continue
        if environ.get(key):
            # An explicit non-empty process/shell value wins over the .env file.
            continue
        environ[key] = value


DEFAULT_ACTIVE_MODEL_CONFIG = ModelConfig(
    name="zai-glm-5-3", provider="mistral/default", alias="glm-5-3", thinking="medium"
)

# The catalog is deliberately not a field of ChartreuxConfigSchema.  It is loaded
# separately from shipped definitions plus models.toml and attached privately.
UNPINNED_ACTIVE_MODEL = ""


def _unique_by(key: str) -> Callable[[list[Any]], list[Any]]:
    def check(items: list[Any]) -> list[Any]:
        seen: set[str] = set()
        for item in items:
            value = getattr(item, key)
            if value in seen:
                raise ValueError(f"Duplicate {key} {value!r}; must be unique")
            seen.add(value)
        return items

    return check


def _non_empty(items: list[Any]) -> list[Any]:
    if not items:
        raise ValueError(
            "No models are configured. Define at least one model under [[models]]."
        )
    return items


def _expand_paths(v: Any) -> list[Path]:
    if not isinstance(v, list):
        raise ValueError("Paths must be a list")
    return [Path(p).expanduser().resolve() for p in v]


def _validate_thinking_overrides(
    value: Any, handler: ValidatorFunctionWrapHandler
) -> dict[str, ThinkingLevel]:
    try:
        result = handler(value)
        if all(result):
            return result
    except ValidationError:
        pass
    # Outside the handler: neither aliases nor values survive in error context.
    raise ValidationError.from_exception_data(
        "ThinkingOverrides",
        [
            {
                "type": PydanticCustomError(
                    "invalid_thinking_override", "Invalid thinking_overrides field"
                ),
                "loc": (),
                "input": None,
            }
        ],
        hide_input=True,
    ) from None


def _unknown_thinking_override_error(*, source: str | None = None) -> ValidationError:
    source_name = source if source is not None else "configuration"
    return ValidationError.from_exception_data(
        "ChartreuxConfigSchema",
        [
            {
                "type": PydanticCustomError(
                    "unknown_thinking_override",
                    "Invalid field 'thinking_overrides' in source "
                    "'{source}': contains unknown model aliases",
                    {"source": source_name},
                ),
                "loc": ("thinking_overrides",),
                "input": None,
            }
        ],
        hide_input=True,
    )


def _normalize_tool_configs(v: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(v, dict):
        raise ValueError("Tools must be a table of tool configuration tables")
    normalized = {
        name: cfg.model_dump() if isinstance(cfg, BaseModel) else cfg
        for name, cfg in v.items()
    }
    if any(not isinstance(cfg, dict) for cfg in normalized.values()):
        raise ValueError("Tools must be a table of tool configuration tables")
    return normalized


_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _normalize_log_level(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().upper()
    if normalized not in _LOG_LEVELS:
        raise ValueError(
            f"Invalid log level {value!r}; expected one of {sorted(_LOG_LEVELS)}"
        )
    return normalized


class ChartreuxConfigSchema(ConfigSchema):
    @model_validator(mode="before")
    @classmethod
    def _reject_removed_bypass_tool_permissions(cls, value: Any) -> Any:
        if isinstance(value, dict) and "bypass_tool_permissions" in value:
            raise ValueError(
                "'bypass_tool_permissions' was removed in v0.1; remove this key. "
                "It has no effect."
            )
        return value

    _validation_warnings: list[str] = PrivateAttr(default_factory=list)
    _catalog_snapshot: Any = PrivateAttr(default=None)
    _committed_model: Any = PrivateAttr(default=None)

    @property
    def catalog_snapshot(self) -> Any:
        """The accepted catalog snapshot; private so it cannot be serialized."""
        return self._catalog_snapshot

    def attach_catalog_snapshot(self, snapshot: Any) -> ChartreuxConfigSchema:
        """Attach builder-owned catalog authority after every model construction."""
        object.__setattr__(self, "_catalog_snapshot", snapshot)
        return self

    def attach_committed_model(self, identity: Any) -> ChartreuxConfigSchema:
        """Bind active-model materialization to a conversation's concrete identity."""
        object.__setattr__(self, "_committed_model", identity)
        return self

    @property
    def validation_warnings(self) -> tuple[str, ...]:
        return tuple(self._validation_warnings)

    # Models
    active_model: Annotated[str, WithReplaceMerge()] = UNPINNED_ACTIVE_MODEL
    thinking_overrides: Annotated[
        dict[str, ThinkingLevel],
        WithShallowMerge(),
        WrapValidator(_validate_thinking_overrides),
    ] = Field(
        default_factory=dict,
        description="Per-model thinking levels for the active session.",
    )
    allowed_models: Annotated[list[str], WithReplaceMerge()] = Field(
        default_factory=list,
        description=(
            "An explicit list of model aliases/patterns to allow. If set, only these"
            " models are selectable. An empty list allows all configured models."
            " Supports glob patterns (e.g., 'mistral-*') and regex with 're:' prefix."
        ),
    )
    compaction_model: Annotated[str, WithReplaceMerge()] = Field(
        default="",
        description="Catalog model alias for compaction; empty uses the active model. Must share its provider.",
    )
    auto_compact_threshold: Annotated[int, WithReplaceMerge()] = Field(
        default=DEFAULT_AUTO_COMPACT_THRESHOLD,
        description=(
            "Fallback token count before automatic compaction for models that "
            "do not define their own threshold."
        ),
    )
    # Projection only: runtime authority comes from accepted source contributions,
    # never from this mergeable (and externally mutable) configuration container.
    authorized_roots_by_project: Annotated[
        dict[str, list[str]],
        WithReplaceMerge(),
        AfterValidator(normalize_authorized_roots),
    ] = Field(
        default_factory=dict,
        description="User-source-only additional filesystem roots by canonical project.",
    )

    # Tools
    tools: Annotated[
        dict[str, dict[str, Any]],
        WithDeepMerge(),
        BeforeValidator(_normalize_tool_configs),
    ] = Field(default_factory=dict)
    tool_paths: Annotated[
        list[Path], WithReplaceMerge(), BeforeValidator(_expand_paths)
    ] = Field(
        default_factory=list,
        description=(
            "Additional directories or files to explore for custom tools. "
            "Paths may be absolute or relative to the current working directory. "
            "Directories are shallow-searched for tool definition files, "
            "while files are loaded directly if valid."
        ),
    )
    enabled_tools: Annotated[list[str], WithReplaceMerge()] = Field(
        default_factory=list,
        description=(
            "An explicit list of tool names/patterns to enable. If set, only these"
            " tools will be active. Supports glob patterns (e.g., 'serena_*') and"
            " regex with 're:' prefix (e.g., 're:^serena_.*')."
        ),
    )
    disabled_tools: Annotated[list[str], WithConcatMerge()] = Field(
        default_factory=list,
        description=(
            "A list of tool names/patterns to disable after 'enabled_tools' filtering. "
            "Supports glob patterns and regex with 're:' prefix."
        ),
    )
    mcp_servers: Annotated[
        list[MCPServer], WithReplaceMerge(), AfterValidator(_unique_by("name"))
    ] = Field(
        default_factory=list, description="Preferred MCP server configuration entries."
    )

    # Agents
    agent_paths: Annotated[list[Path], WithReplaceMerge()] = Field(
        default_factory=list,
        description=(
            "Additional directories to search for custom agent profiles. "
            "Each path may be absolute or relative to the current working directory."
        ),
    )
    enabled_agents: Annotated[list[str], WithReplaceMerge()] = Field(
        default_factory=list,
        description=(
            "An explicit list of agent names/patterns to enable. If set, only these"
            " agents will be available. Supports glob patterns (e.g., 'custom-*')"
            " and regex with 're:' prefix."
        ),
    )
    disabled_agents: Annotated[list[str], WithReplaceMerge()] = Field(
        default_factory=list,
        description=(
            "A list of agent names/patterns to disable. Ignored if 'enabled_agents'"
            " is set. Supports glob patterns and regex with 're:' prefix."
        ),
    )
    # Skills
    skill_paths: Annotated[
        list[Path], WithReplaceMerge(), BeforeValidator(_expand_paths)
    ] = Field(
        default_factory=list,
        description=(
            "Additional directories to search for skills. "
            "Each path may be absolute or relative to the current working directory."
        ),
    )
    enabled_skills: Annotated[list[str], WithReplaceMerge()] = Field(
        default_factory=list,
        description=(
            "An explicit list of skill names/patterns to enable. If set, only these"
            " skills will be active. Supports glob patterns (e.g., 'search-*') and"
            " regex with 're:' prefix."
        ),
    )
    disabled_skills: Annotated[list[str], WithReplaceMerge()] = Field(
        default_factory=list,
        description=(
            "A list of skill names/patterns to disable. Ignored if 'enabled_skills'"
            " is set. Supports glob patterns and regex with 're:' prefix."
        ),
    )
    # Top-level scalars
    theme: Annotated[str, WithReplaceMerge()] = DEFAULT_THEME
    disable_welcome_banner_animation: Annotated[bool, WithReplaceMerge()] = False
    show_greeting: Annotated[bool, WithReplaceMerge()] = Field(
        default=True,
        description="Show greeting at startup (Mistral providers only, once per 24h).",
    )
    autocopy_to_clipboard: Annotated[bool, WithReplaceMerge()] = True
    file_watcher_for_autocomplete: Annotated[bool, WithReplaceMerge()] = False
    ask_confirmation_on_exit: Annotated[bool, WithReplaceMerge()] = True
    displayed_workdir: Annotated[str, WithReplaceMerge()] = ""
    context_warnings: Annotated[bool, WithReplaceMerge()] = False
    show_thinking_nodes: Annotated[bool, WithReplaceMerge()] = False
    raise_on_compaction_failure: Annotated[bool, WithReplaceMerge()] = False
    system_prompt_id: Annotated[str, WithReplaceMerge()] = SystemPrompt.CLI
    compaction_prompt_id: Annotated[str, WithReplaceMerge()] = UtilityPrompt.COMPACT
    include_commit_signature: Annotated[bool, WithReplaceMerge()] = True
    include_model_info: Annotated[bool, WithReplaceMerge()] = True
    include_project_context: Annotated[bool, WithReplaceMerge()] = True
    include_prompt_detail: Annotated[bool, WithReplaceMerge()] = True
    enable_notifications: Annotated[bool, WithReplaceMerge()] = True
    enable_system_trust_store: Annotated[bool, WithReplaceMerge()] = False
    api_timeout: Annotated[float, WithReplaceMerge()] = DEFAULT_API_TIMEOUT
    api_retry_max_elapsed_time: Annotated[float, WithReplaceMerge()] = (
        DEFAULT_API_RETRY_MAX_ELAPSED_TIME
    )
    api_connect_timeout: Annotated[float, WithReplaceMerge()] = (
        DEFAULT_API_CONNECT_TIMEOUT
    )
    api_write_timeout: Annotated[float, WithReplaceMerge()] = DEFAULT_API_WRITE_TIMEOUT
    api_pool_timeout: Annotated[float, WithReplaceMerge()] = DEFAULT_API_POOL_TIMEOUT
    log_level: Annotated[
        str | None, WithReplaceMerge(), BeforeValidator(_normalize_log_level)
    ] = None

    # Nested configs (DEEP_MERGE — sparse layers preserve unspecified keys)
    project_context: Annotated[ProjectContextConfig, WithDeepMerge()] = Field(
        default_factory=ProjectContextConfig
    )
    subagents: Annotated[SubagentsConfig, WithDeepMerge()] = Field(
        default_factory=SubagentsConfig
    )
    session_logging: Annotated[SessionLoggingConfig, WithDeepMerge()] = Field(
        default_factory=SessionLoggingConfig
    )

    def resolve_default_model_alias(self) -> str:
        from chartreux.core.model_catalog.resolver import resolver_for

        return (
            resolver_for(self)
            .resolve("@orchestrator", allowed_models=self.allowed_models)
            .base_model
        )

    def available_models(self) -> dict[str, ModelConfig]:
        from chartreux.core.model_catalog.resolver import (
            ModelResolutionError,
            resolver_for,
        )

        result: dict[str, ModelConfig] = {}
        resolver = resolver_for(self)
        for base in self.catalog_snapshot.catalog.models:
            try:
                resolved = resolver.resolve(base, allowed_models=self.allowed_models)
            except ModelResolutionError:
                continue
            result[base] = resolved.materialize(
                auto_compact_threshold=self.auto_compact_threshold,
                thinking=self.thinking_overrides.get(base),
            )
        return result

    def get_active_model(self) -> ModelConfig:
        from chartreux.core.model_catalog.resolver import resolver_for

        resolver = resolver_for(self)
        resolved = (
            resolver.resolve_committed(
                self._committed_model, allowed_models=self.allowed_models
            )
            if self._committed_model is not None
            else resolver.resolve(
                self.active_model or "@orchestrator", allowed_models=self.allowed_models
            )
        )
        return resolved.materialize(
            auto_compact_threshold=self.auto_compact_threshold,
            thinking=self.thinking_overrides.get(resolved.base_model),
        )

    def get_provider_for_model(self, model: ModelConfig) -> ProviderConfig:
        definition = self.catalog_snapshot.catalog.providers.get(model.provider)
        if definition is None or definition.disabled:
            raise ValueError(
                f"Provider '{model.provider}' for model '{model.name}' not found in catalog."
            )
        return ProviderConfig(
            name=model.provider,
            api_base=definition.api_base,
            api_key_env_var=definition.api_key_env_var,
            api_style=definition.api_style,
            backend=definition.backend,
            reasoning_field_name=definition.reasoning_field_name,
            emits_finish_reason=definition.emits_finish_reason,
            supports_tool_result_images=definition.supports_tool_result_images,
            extra_headers=definition.extra_headers,
        )

    def get_compaction_model(self) -> ModelConfig:
        if not self.compaction_model:
            return self.get_active_model()
        from chartreux.core.model_catalog.resolver import resolver_for

        resolved = resolver_for(self).resolve(self.compaction_model)
        return resolved.materialize(
            auto_compact_threshold=self.auto_compact_threshold,
            thinking=self.thinking_overrides.get(resolved.base_model),
        )

    def get_active_provider(self) -> ProviderConfig:
        return self.get_provider_for_model(self.get_active_model())

    def require_active_provider_api_key(self) -> None:
        provider = self.get_active_provider()
        api_key_env = provider.api_key_env_var
        if api_key_env and not resolve_api_key(api_key_env):
            raise MissingAPIKeyError(api_key_env, provider.name)

    def get_mistral_provider(self) -> ProviderConfig | None:
        try:
            active_provider = self.get_active_provider()
            if active_provider.backend == Backend.MISTRAL:
                return active_provider
        except ValueError:
            pass
        for provider_id, definition in self.catalog_snapshot.catalog.providers.items():
            if definition.backend == Backend.MISTRAL and not definition.disabled:
                return self.get_provider_for_model(
                    ModelConfig(name="", provider=provider_id, alias="")
                )
        return None

    def is_active_model_mistral(self) -> bool:
        try:
            return self.get_active_provider().backend == Backend.MISTRAL
        except ValueError:
            return False

    def build_tool_allowlist_update(
        self,
        tool_name: str,
        patterns: list[str],
        *,
        current_allowlist: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Extend a tool's allowlist in memory and return the persist payload.

        Returns ``None`` when every pattern is already allowlisted. Callers
        persist the returned payload; the in-memory config is kept current so
        repeated calls merge from fresh state.
        """
        if tool_name == "bash":
            raise ValueError(
                "[tools.bash].allowlist was removed in v0.1; remove this key. "
                "The shell resolver's hard guards are the policy."
            )
        allowlist: list[str] = list(
            current_allowlist
            if current_allowlist is not None
            else self.tools.get(tool_name, {}).get("allowlist", [])
        )
        new_patterns = [p for p in patterns if p not in allowlist]
        if not new_patterns:
            return None
        merged = sorted(allowlist + new_patterns)
        self.tools.setdefault(tool_name, {})["allowlist"] = merged
        return {"tools": {tool_name: {"allowlist": merged}}}

    @property
    def system_prompt(self) -> str:
        return load_system_prompt(self.system_prompt_id)

    @property
    def compaction_prompt(self) -> str:
        return load_prompt(
            self.compaction_prompt_id,
            setting_name="compaction_prompt_id",
            builtins={"compact": UtilityPrompt.COMPACT.path},
        )


def create_default_config() -> dict[str, Any]:
    from chartreux.core.tools.manager import ToolManager

    config_dict = ChartreuxConfigSchema.model_construct().model_dump(
        mode="json", exclude_none=True
    )
    if tool_defaults := ToolManager.discover_tool_defaults():
        tool_defaults.get("bash", {}).pop("allowlist", None)
        config_dict["tools"] = tool_defaults
    return config_dict
