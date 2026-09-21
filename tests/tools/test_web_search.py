from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from chartreux.core.config import ModelConfig
from chartreux.core.tools.base import BaseToolState, ToolError
from chartreux.core.tools.builtins.web_search import (
    ResolvedSearchProvider,
    SearchProviderDiagnostic,
    WebSearch,
    WebSearchArgs,
    WebSearchConfig,
    effective_web_search_config,
    resolve_web_search_provider,
)
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.search import SearchResponse
from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result


@pytest.mark.parametrize(
    ("provider", "env_var", "key", "available"),
    [
        ("mistral", "MISTRAL_API_KEY", "key", True),
        ("mistral", "MISTRAL_API_KEY", None, False),
        ("exa", "EXA_API_KEY", "key", True),
        ("exa", "EXA_API_KEY", "", False),
        ("brave", "BRAVE_SEARCH_API_KEY", "key", True),
        ("brave", "BRAVE_SEARCH_API_KEY", None, False),
        ("duckduckgo", None, None, True),
    ],
)
def test_provider_resolution_matrix(monkeypatch, provider, env_var, key, available):
    for name in ("MISTRAL_API_KEY", "EXA_API_KEY", "BRAVE_SEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    if env_var is not None and key is not None:
        monkeypatch.setenv(env_var, key)

    resolution = resolve_web_search_provider(WebSearchConfig(provider=provider))

    assert isinstance(resolution, ResolvedSearchProvider) is available
    if available:
        assert resolution.provider == provider
    else:
        assert isinstance(resolution, SearchProviderDiagnostic)
        assert resolution.env_var == env_var


def test_explicit_provider_wins_over_mistral_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral")
    monkeypatch.setenv("EXA_API_KEY", "exa")

    resolution = resolve_web_search_provider(WebSearchConfig(provider="exa"))

    assert isinstance(resolution, ResolvedSearchProvider)
    assert resolution.provider == "exa"
    assert resolution.api_key == "exa"


def test_explicit_mistral_key_override_applies_with_non_mistral_active_model(
    monkeypatch,
):
    monkeypatch.setenv("CUSTOM_MISTRAL_KEY", "key")
    config = build_test_vibe_config(
        active_model="other",
        providers=[
            {
                "name": "other",
                "api_base": "https://example.test/v1",
                "api_key_env_var": "OTHER_KEY",
                "backend": "generic",
            },
            {
                "name": "mistral-search",
                "api_base": "https://search.example/v1",
                "api_key_env_var": "CUSTOM_MISTRAL_KEY",
                "backend": "mistral",
            },
        ],
        models=[ModelConfig(name="other", provider="other", alias="other")],
    )

    resolution = resolve_web_search_provider(
        WebSearchConfig(
            provider="mistral",
            api_key_env_var="CUSTOM_MISTRAL_KEY",
            base_url="https://search.example/v1",
        ),
        config,
    )

    assert isinstance(resolution, ResolvedSearchProvider)
    assert resolution.provider == "mistral"
    assert resolution.api_key == "key"
    assert resolution.base_url == "https://search.example"


def test_invalid_mistral_base_url_never_falls_back_to_sdk_default(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "key")

    resolution = resolve_web_search_provider(
        WebSearchConfig(base_url="https://search.example")
    )

    assert isinstance(resolution, SearchProviderDiagnostic)
    assert resolution.config_key == "tools.web_search.base_url"


def test_unconfigured_search_is_hidden_from_manager(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    manager = ToolManager(lambda: build_test_vibe_config(enabled_tools=["web_search"]))

    assert "web_search" not in manager.available_tools


def test_auto_resolution_ignores_unrelated_provider_credentials(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated")
    config = build_test_vibe_config(enabled_tools=["web_search"])

    assert not WebSearch.is_available(config)
    monkeypatch.setenv("MISTRAL_API_KEY", "search-key")
    assert WebSearch.is_available(config)


def test_config_override_is_applied_at_discovery(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    config = build_test_vibe_config(
        tools={"web_search": {"provider": "duckduckgo", "max_results": 7}}
    )

    effective_config = effective_web_search_config(config)
    assert isinstance(effective_config, WebSearchConfig)
    assert effective_config.max_results == 7
    assert WebSearch.is_available(config)


def test_invalid_provider_configuration_is_unavailable_with_diagnostic():
    config = build_test_vibe_config(tools={"web_search": {"provider": "typo"}})

    resolution = resolve_web_search_provider(
        effective_web_search_config(config), config
    )

    assert not WebSearch.is_available(config)
    assert isinstance(resolution, SearchProviderDiagnostic)
    assert resolution.config_key == "tools.web_search.provider"
    assert "Invalid web search provider value 'typo'" in resolution.message
    assert WebSearch.availability_diagnostic(config) == resolution


def test_invalid_nonprovider_configuration_names_its_field():
    config = build_test_vibe_config(
        tools={"web_search": {"provider": "brave", "timeout": 0}}
    )

    resolution = effective_web_search_config(config)

    assert isinstance(resolution, SearchProviderDiagnostic)
    assert resolution.config_key == "tools.web_search.timeout"
    assert "timeout" in resolution.message
    assert "provider value" not in resolution.message


def test_invalid_provider_resolution_returns_diagnostic():
    resolution = resolve_web_search_provider(
        WebSearchConfig.model_construct(provider="typo")
    )

    assert isinstance(resolution, SearchProviderDiagnostic)
    assert "Invalid web search provider value 'typo'" in resolution.message


@pytest.mark.parametrize(
    ("enabled_tools", "disabled_tools"), [([], ["web_search"]), (["bash"], [])]
)
def test_invalid_web_search_configuration_does_not_block_other_tools(
    enabled_tools, disabled_tools
):
    config = build_test_vibe_config(
        enabled_tools=enabled_tools,
        disabled_tools=disabled_tools,
        tools={"web_search": {"provider": "typo"}},
    )

    assert "bash" in ToolManager(lambda: config).available_tools


@dataclass
class _MockProvider:
    calls: int = 0

    async def search(self, query: str, *, max_results: int) -> SearchResponse:
        self.calls += 1
        return SearchResponse(
            query=query, provider="mock", answer=None, sources=[], was_truncated=False
        )


@pytest.mark.asyncio
async def test_runtime_rechecks_credentials_without_request(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "key")
    provider = _MockProvider()
    tool = WebSearch(
        config_getter=lambda: WebSearchConfig(provider="exa"), state=BaseToolState()
    )
    monkeypatch.setattr(tool, "_create_provider", lambda _: provider)
    monkeypatch.delenv("EXA_API_KEY")

    with pytest.raises(ToolError, match="EXA_API_KEY"):
        await collect_result(tool.run(WebSearchArgs(query="news")))
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_runtime_invalid_provider_configuration_raises_tool_error():
    tool = WebSearch(
        config_getter=lambda: WebSearchConfig.model_construct(provider="typo"),
        state=BaseToolState(),
    )

    with pytest.raises(ToolError, match="Invalid web search provider value 'typo'"):
        await collect_result(tool.run(WebSearchArgs(query="news")))


@pytest.mark.asyncio
async def test_run_enforces_provider_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled = asyncio.Event()

    class SlowProvider:
        async def search(self, query: str, *, max_results: int) -> SearchResponse:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("slow provider completed unexpectedly")

    tool = WebSearch(
        config_getter=lambda: WebSearchConfig(provider="duckduckgo", timeout=1),
        state=BaseToolState(),
    )
    monkeypatch.setattr(tool, "_create_provider", lambda _: SlowProvider())

    with pytest.raises(ToolError, match="timed out"):
        await collect_result(tool.run(WebSearchArgs(query="news")))
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_run_returns_provider_response(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "key")
    provider = _MockProvider()
    tool = WebSearch(
        config_getter=lambda: WebSearchConfig(provider="exa", max_results=3),
        state=BaseToolState(),
    )
    monkeypatch.setattr(tool, "_create_provider", lambda _: provider)

    result = await collect_result(tool.run(WebSearchArgs(query="news")))

    assert result.query == "news"
    assert result.provider == "mock"
    assert provider.calls == 1


def test_config_round_trip_and_prompt():
    config = WebSearchConfig(
        provider="brave",
        api_key_env_var="CUSTOM",
        base_url="https://search.example",
        timeout=17,
        max_results=4,
        model="model",
    )

    assert WebSearchConfig.model_validate(config.model_dump()) == config
    prompt = WebSearch.get_tool_prompt()
    assert prompt is not None
    assert "untrusted" in prompt


def test_invalid_web_search_config_falls_back_during_policy_validation():
    config = build_test_vibe_config(
        disabled_tools=["web_search"], tools={"web_search": {"provider": "typo"}}
    )
    manager = ToolManager(lambda: config)

    assert manager.get_tool_config("web_search") == WebSearchConfig()
    assert manager.get_tool_config("bash").permission.name == "ASK"
    diagnostic = WebSearch.availability_diagnostic(config)
    assert diagnostic is not None
    assert diagnostic.config_key == "tools.web_search.provider"
