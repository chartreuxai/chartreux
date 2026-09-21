from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest

from chartreux.core.model_catalog.discovery import discover_models
from chartreux.core.model_catalog.presets import PRESETS
from chartreux.ui.providers.contracts import (
    DiscoveryError,
    DiscoveryResult,
    ProviderDraft,
    TLSConfig,
)


def _provider(style: str = "openai", **kwargs: object) -> ProviderDraft:
    defaults: dict[str, object] = {
        "preset": None,
        "provider_id": "test/default",
        "name": "test",
        "api_base": "https://gateway.example/proxy/v1",
        "api_style": style,
        "api_key_env_var": "TEST_API_KEY",
        "key": None,
    }
    defaults.update(kwargs)
    return ProviderDraft(**defaults)  # type: ignore[arg-type]


def _client(
    handler: Callable[[httpx.Request], Awaitable[httpx.Response]],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["openai", "openai-responses"])
async def test_openai_styles_preserve_gateway_prefix_and_auth(style: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://gateway.example/proxy/v1/models"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json={"data": [{"id": "wire/model"}]})

    async with _client(handler) as client:
        result = await discover_models(_provider(style), "secret", TLSConfig(), client)

    assert isinstance(result, DiscoveryResult)
    assert result.models[0].wire_id == "wire/model"
    assert result.diagnostics == ("Models discovered",)


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["openai", "openai-responses"])
async def test_openai_styles_normalize_trailing_base_slash(style: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://gateway.example/proxy/v1/models"
        return httpx.Response(200, json={"data": [{"id": "wire/model"}]})

    async with _client(handler) as client:
        result = await discover_models(
            _provider(style, api_base="https://gateway.example/proxy/v1/"),
            "secret",
            TLSConfig(),
            client,
        )

    assert isinstance(result, DiscoveryResult)


@pytest.mark.asyncio
async def test_anthropic_uses_its_listing_url_auth_and_pagination() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["x-api-key"] == "secret"
        assert request.headers["anthropic-version"] == "2023-06-01"
        if request.url.params.get("after_id") is None:
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "first", "display_name": "First"}],
                    "has_more": True,
                    "last_id": "first",
                },
            )
        return httpx.Response(200, json={"data": [{"id": "second"}], "has_more": False})

    async with _client(handler) as client:
        result = await discover_models(
            _provider("anthropic", api_base="https://api.example/gateway"),
            "secret",
            TLSConfig(),
            client,
        )

    assert [str(request.url) for request in requests] == [
        "https://api.example/gateway/v1/models",
        "https://api.example/gateway/v1/models?after_id=first",
    ]
    assert isinstance(result, DiscoveryResult)
    assert result.models == (
        result.models[0].__class__("first", "First"),
        result.models[1].__class__("second"),
    )


@pytest.mark.asyncio
async def test_anthropic_cursor_loop_is_malformed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"id": "same"}], "has_more": True, "last_id": "same"}
        )

    async with _client(handler) as client:
        result = await discover_models(
            _provider("anthropic"), "secret", TLSConfig(), client
        )

    assert isinstance(result, DiscoveryError)
    assert result.code == "malformed"


@pytest.mark.asyncio
async def test_links_next_pagination_and_duplicate_wire_ids() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("page") is None:
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "one"}, {"id": "two"}],
                    "links": {"next": "?page=2"},
                },
            )
        return httpx.Response(200, json={"data": [{"id": "two"}, {"id": "three"}]})

    async with _client(handler) as client:
        result = await discover_models(_provider(), "secret", TLSConfig(), client)

    assert isinstance(result, DiscoveryResult)
    assert [item.wire_id for item in result.models] == ["one", "two", "three"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "code"),
    [({}, "malformed"), ({"data": []}, "empty"), ({"data": [{}]}, "malformed")],
)
async def test_malformed_and_empty_responses(payload: object, code: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with _client(handler) as client:
        result = await discover_models(_provider(), "secret", TLSConfig(), client)

    assert isinstance(result, DiscoveryError)
    assert result.code == code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "auth_rejected"),
        (403, "auth_rejected"),
        (404, "unsupported_listing"),
        (429, "rate_limited"),
    ],
)
async def test_listing_http_errors_are_typed(status: int, code: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b"sensitive response")

    async with _client(handler) as client:
        result = await discover_models(_provider(), "secret", TLSConfig(), client)

    assert isinstance(result, DiscoveryError)
    assert result.code == code
    assert "sensitive" not in result.message


@pytest.mark.asyncio
async def test_timeout_is_connection_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with _client(handler) as client:
        result = await discover_models(_provider(), "secret", TLSConfig(), client)

    assert isinstance(result, DiscoveryError)
    assert result.code == "connection"


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        return httpx.Response(200)

    async with _client(handler) as client:
        task = asyncio.create_task(
            discover_models(_provider(), "secret", TLSConfig(), client)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_generated_auth_overrides_case_insensitive_extra_headers() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        assert request.headers["x-provider"] == "value"
        return httpx.Response(200, json={"data": [{"id": "model"}]})

    async with _client(handler) as client:
        result = await discover_models(
            _provider(extra_headers={"authorization": "wrong", "X-Provider": "value"}),
            "secret",
            TLSConfig(),
            client,
        )

    assert isinstance(result, DiscoveryResult)


def test_presets_are_exact_editable_defaults_without_removed_providers() -> None:
    actual = {
        preset.name: (
            preset.api_base,
            preset.api_style,
            preset.api_key_env_var,
            preset.backend,
            preset.reasoning_field_name,
        )
        for preset in PRESETS
    }

    assert actual == {
        "Mistral": (
            "https://api.mistral.ai/v1",
            "openai",
            "MISTRAL_API_KEY",
            "mistral",
            "reasoning_content",
        ),
        "Ollama Cloud": (
            "https://ollama.com/v1",
            "openai",
            "OLLAMA_API_KEY",
            "generic",
            "reasoning_content",
        ),
        "OpenCode Go": (
            "https://opencode.ai/zen/go/v1",
            "openai",
            "OPENCODE_API_KEY",
            "generic",
            "reasoning_content",
        ),
        "Generic OpenAI-style": (None, "openai", None, "generic", "reasoning_content"),
        "Generic Anthropic-style": (
            None,
            "anthropic",
            None,
            "generic",
            "reasoning_content",
        ),
        "Fully custom": (None, None, None, "generic", "reasoning_content"),
    }
    assert not {"OpenRouter", "OpenCode Zen"} & actual.keys()
