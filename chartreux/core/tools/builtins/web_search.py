from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, cast, final

from pydantic import BaseModel, Field, ValidationError, field_validator

from chartreux.core.events import ToolStreamEvent
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.search import (
    MAX_RESULTS,
    SearchProvider,
    SearchProviderError,
    SearchSource,
    validate_query,
)
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.utils.api_keys import resolve_api_key
from chartreux.utils.http import get_server_url_from_api_base
from chartreux.utils.tool_presentation import ToolEffectKind

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema

_MAX_QUERY_PREVIEW_LENGTH = 80
_MAX_DIAGNOSTIC_LENGTH = 500
SearchProviderName = Literal["auto", "mistral", "exa", "brave", "duckduckgo"]


class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query")

    @field_validator("query")
    @classmethod
    def validate_search_query(cls, value: str) -> str:
        try:
            return validate_query(value)
        except SearchProviderError as error:
            raise ValueError(error.safe_message) from error


class WebSearchResult(BaseModel):
    query: str
    provider: str
    answer: str | None
    sources: list[SearchSource]
    was_truncated: bool


class WebSearchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    provider: SearchProviderName = "auto"
    api_key_env_var: str | None = None
    base_url: str | None = None
    timeout: int = Field(default=120, gt=0)
    max_results: int = Field(default=5, ge=1, le=MAX_RESULTS)
    model: str = "mistral-vibe-cli-with-tools"


@dataclass(frozen=True, slots=True)
class ResolvedSearchProvider:
    """Network-free provider settings after credential and endpoint resolution."""

    provider: Literal["mistral", "exa", "brave", "duckduckgo"]
    api_key: str | None
    base_url: str | None
    timeout: int
    max_results: int
    model: str


@dataclass(frozen=True, slots=True)
class SearchProviderDiagnostic:
    """A safe, actionable reason web search cannot currently be used."""

    message: str
    provider: str
    env_var: str | None = None
    config_key: str | None = None


def effective_web_search_config(
    config: ChartreuxConfigSchema,
) -> WebSearchConfig | SearchProviderDiagnostic:
    """Merge the global tool override exactly as ToolManager does."""
    overrides = config.tools.get("web_search") or {}
    try:
        return WebSearchConfig.model_validate({
            **WebSearchConfig().model_dump(),
            **overrides,
        })
    except (TypeError, ValidationError) as error:
        provider = overrides.get("provider") if isinstance(overrides, dict) else None
        invalid_field = None
        if isinstance(error, ValidationError):
            for issue in error.errors():
                location = issue.get("loc", ())
                if location and isinstance(location[0], str):
                    invalid_field = location[0]
                    break
        if invalid_field == "provider":
            return _diagnostic(
                f"Invalid web search provider value {provider!r}; expected one of "
                "'auto', 'mistral', 'exa', 'brave', or 'duckduckgo'.",
                str(provider),
                config_key="tools.web_search.provider",
            )
        if invalid_field is not None:
            return _diagnostic(
                f"Invalid web search {invalid_field} value: {error}",
                str(provider or "unknown"),
                config_key=f"tools.web_search.{invalid_field}",
            )
        return _diagnostic(
            f"Invalid web search configuration: {error}",
            str(provider or "unknown"),
            config_key="tools.web_search",
        )


def resolve_web_search_provider(
    config: WebSearchConfig | SearchProviderDiagnostic,
    chartreux_config: ChartreuxConfigSchema | None = None,
) -> ResolvedSearchProvider | SearchProviderDiagnostic:
    """Resolve web-search settings without importing SDKs or making requests.

    ``chartreux_config`` supplies a configured Mistral provider for ``auto`` and
    is intentionally optional for direct tool construction in tests and plugins.
    """
    if isinstance(config, SearchProviderDiagnostic):
        return config
    provider = config.provider
    if provider in {"auto", "mistral"}:
        return _resolve_mistral(config, chartreux_config)
    if provider == "duckduckgo":
        return _resolved(config, "duckduckgo", None, config.base_url)
    if provider in {"exa", "brave"}:
        return _resolve_keyed_provider(config, cast(Literal["exa", "brave"], provider))
    return _diagnostic(
        f"Invalid web search provider value {provider!r}; expected one of "
        "'auto', 'mistral', 'exa', 'brave', or 'duckduckgo'.",
        str(provider),
        config_key="tools.web_search.provider",
    )


def _resolve_keyed_provider(
    config: WebSearchConfig, provider: Literal["exa", "brave"]
) -> ResolvedSearchProvider | SearchProviderDiagnostic:
    env_var = config.api_key_env_var or (
        "EXA_API_KEY" if provider == "exa" else "BRAVE_SEARCH_API_KEY"
    )
    api_key = resolve_api_key(env_var)
    if not api_key:
        return _missing_key(provider, env_var)
    return _resolved(config, provider, api_key, config.base_url)


def _resolve_mistral(
    config: WebSearchConfig, chartreux_config: ChartreuxConfigSchema | None
) -> ResolvedSearchProvider | SearchProviderDiagnostic:
    try:
        mistral_provider = (
            chartreux_config.get_mistral_provider()
            if chartreux_config is not None
            else None
        )
    except (AttributeError, ValueError):
        mistral_provider = None
    env_var = config.api_key_env_var or (
        mistral_provider.api_key_env_var
        if mistral_provider is not None
        else "MISTRAL_API_KEY"
    )
    api_key = resolve_api_key(env_var)
    if not api_key:
        return _missing_key("mistral", env_var)

    api_base = config.base_url or (
        mistral_provider.api_base
        if mistral_provider is not None
        else "https://api.mistral.ai/v1"
    )
    if api_base:
        server_url = get_server_url_from_api_base(api_base)
        if not server_url:
            return _diagnostic(
                "Invalid Mistral web-search base_url; expected <server_url>/v<api_version>",
                "mistral",
                config_key="tools.web_search.base_url",
            )
    else:
        server_url = None
    return _resolved(config, "mistral", api_key, server_url)


def _resolved(
    config: WebSearchConfig,
    provider: Literal["mistral", "exa", "brave", "duckduckgo"],
    api_key: str | None,
    base_url: str | None,
) -> ResolvedSearchProvider:
    return ResolvedSearchProvider(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        timeout=config.timeout,
        max_results=config.max_results,
        model=config.model,
    )


def _missing_key(provider: str, env_var: str) -> SearchProviderDiagnostic:
    return _diagnostic(
        f"Web search provider '{provider}' requires {env_var}; set that environment variable "
        "or configure tools.web_search.api_key_env_var.",
        provider,
        env_var=env_var,
        config_key="tools.web_search.api_key_env_var",
    )


def _diagnostic(
    message: str,
    provider: str,
    *,
    env_var: str | None = None,
    config_key: str | None = None,
) -> SearchProviderDiagnostic:
    return SearchProviderDiagnostic(
        _truncate_diagnostic(message),
        _truncate_diagnostic(provider),
        _truncate_diagnostic(env_var) if env_var is not None else None,
        _truncate_diagnostic(config_key) if config_key is not None else None,
    )


def _truncate_diagnostic(value: str) -> str:
    if len(value) <= _MAX_DIAGNOSTIC_LENGTH:
        return value
    return f"{value[: _MAX_DIAGNOSTIC_LENGTH - 3]}..."


class WebSearch(
    BaseTool[WebSearchArgs, WebSearchResult, WebSearchConfig, BaseToolState],
    ToolUIData[WebSearchArgs, WebSearchResult],
):
    effect_kind = ToolEffectKind.WEB_SEARCH

    @classmethod
    def validate_tool_config(cls, values: dict[str, Any]) -> WebSearchConfig:
        """Return a safe config when the global override is invalid.

        ``availability_diagnostic`` derives the corresponding user-facing
        diagnostic from the unchanged global configuration.
        """
        try:
            return WebSearchConfig.model_validate(values)
        except (TypeError, ValidationError):
            return WebSearchConfig()

    def _set_runtime_config_getter(
        self, getter: Callable[[], ChartreuxConfigSchema]
    ) -> None:
        self._runtime_config_getter = getter

    @classmethod
    def is_available(cls, config: ChartreuxConfigSchema | None = None) -> bool:
        if config is None:
            return isinstance(
                resolve_web_search_provider(WebSearchConfig()), ResolvedSearchProvider
            )
        return isinstance(
            resolve_web_search_provider(effective_web_search_config(config), config),
            ResolvedSearchProvider,
        )

    @classmethod
    def availability_diagnostic(
        cls, config: ChartreuxConfigSchema | None = None
    ) -> SearchProviderDiagnostic | None:
        effective = (
            effective_web_search_config(config)
            if config is not None
            else WebSearchConfig()
        )
        resolved = resolve_web_search_provider(effective, config)
        return resolved if isinstance(resolved, SearchProviderDiagnostic) else None

    @final
    async def run(
        self, args: WebSearchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WebSearchResult, None]:
        try:
            query = validate_query(args.query)
        except SearchProviderError as error:
            raise ToolError(error.safe_message) from error

        global_config = (
            ctx.agent_manager.config
            if ctx and ctx.agent_manager
            else getattr(self, "_runtime_config_getter", lambda: None)()
        )
        try:
            resolved = resolve_web_search_provider(self.config, global_config)
        except ValidationError as error:
            raise ToolError(f"Invalid web search configuration: {error}") from error
        if isinstance(resolved, SearchProviderDiagnostic):
            raise ToolError(resolved.message)
        if ctx is not None:
            yield ToolStreamEvent(
                tool_name=self.get_name(),
                tool_call_id=ctx.tool_call_id,
                message=f"Searching with {resolved.provider}",
            )
        try:
            provider = self._create_provider(resolved)
            response = await asyncio.wait_for(
                provider.search(query, max_results=resolved.max_results),
                timeout=resolved.timeout,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise ToolError("Search request timed out") from None
        except SearchProviderError as error:
            raise ToolError(error.safe_message) from error
        yield WebSearchResult(**response.model_dump())

    @staticmethod
    def _create_provider(settings: ResolvedSearchProvider) -> SearchProvider:
        if settings.provider == "mistral":
            module = import_module("chartreux.core.tools.search.mistral")
            return module.MistralSearchProvider(
                settings.api_key,
                model=settings.model,
                server_url=settings.base_url,
                timeout=settings.timeout,
            )
        if settings.provider == "exa":
            module = import_module("chartreux.core.tools.search.exa")
            if settings.base_url is None:
                return module.ExaSearchProvider(
                    settings.api_key, timeout=settings.timeout
                )
            return module.ExaSearchProvider(
                settings.api_key, base_url=settings.base_url, timeout=settings.timeout
            )
        if settings.provider == "brave":
            module = import_module("chartreux.core.tools.search.brave")
            if settings.base_url is None:
                return module.BraveSearchProvider(
                    settings.api_key, timeout=settings.timeout
                )
            return module.BraveSearchProvider(
                settings.api_key, base_url=settings.base_url, timeout=settings.timeout
            )
        module = import_module("chartreux.core.tools.search.duckduckgo")
        return module.DuckDuckGoSearchProvider(timeout=settings.timeout)

    @classmethod
    def format_call_display(cls, args: WebSearchArgs) -> ToolCallDisplay:
        query = args.query.strip()
        preview = (
            query
            if len(query) <= _MAX_QUERY_PREVIEW_LENGTH
            else f"{query[: _MAX_QUERY_PREVIEW_LENGTH - 3]}..."
        )
        return ToolCallDisplay(
            summary=f"Searching web: {preview}",
            verb="Searching",
            message=preview,
            settled_verb="Searched",
            settled_message=preview,
        )

    @classmethod
    def format_result_display(cls, result: WebSearchResult) -> ToolResultDisplay:
        count = len(result.sources)
        noun = "source" if count == 1 else "sources"
        return ToolResultDisplay(
            success=True,
            verb="Searched",
            message=f"{count} {noun} via {result.provider}",
            suffix="(truncated)" if result.was_truncated else "",
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching the web"
