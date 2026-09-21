from __future__ import annotations

import json

import httpx
import pytest
import respx

from chartreux.core.tools.search.exa import ExaSearchProvider
from chartreux.core.tools.search.models import (
    MAX_SNIPPET_LENGTH,
    MAX_UPSTREAM_JSON_BYTES,
)
from chartreux.core.tools.search.provider import SearchProviderError


@pytest.fixture
def provider() -> ExaSearchProvider:
    return ExaSearchProvider(
        "exa-secret", base_url="https://gateway.example/api", timeout=3.5
    )


@pytest.mark.asyncio
@respx.mock
async def test_search_posts_expected_request_and_normalizes_results(
    provider: ExaSearchProvider,
) -> None:
    route = respx.post("https://gateway.example/api/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"title": "One", "url": "https://one.example", "text": "first"},
                    {"url": "https://two.example"},
                    {"title": "duplicate", "url": "https://one.example", "text": "two"},
                    {"title": "invalid", "url": "ftp://three.example", "text": "three"},
                ]
            },
        )
    )

    result = await provider.search("find this", max_results=7)

    assert result.query == "find this"
    assert result.provider == "exa"
    assert result.answer is None
    assert [source.model_dump() for source in result.sources] == [
        {"title": "One", "url": "https://one.example", "snippet": "first"},
        {"title": "", "url": "https://two.example", "snippet": None},
    ]
    assert result.was_truncated is False
    request = route.calls[0].request
    assert request.method == "POST"
    assert request.url == httpx.URL("https://gateway.example/api/search")
    assert request.headers["x-api-key"] == "exa-secret"
    assert json.loads(request.content) == {
        "query": "find this",
        "numResults": 7,
        "contents": {"text": {"maxCharacters": MAX_SNIPPET_LENGTH}},
    }
    assert request.extensions["timeout"]["read"] == 3.5


@pytest.mark.asyncio
@respx.mock
async def test_empty_results_succeed(provider: ExaSearchProvider) -> None:
    respx.post("https://gateway.example/api/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    result = await provider.search("none", max_results=1)

    assert result.sources == []
    assert result.was_truncated is False


@pytest.mark.asyncio
@respx.mock
async def test_malformed_envelope_fails(provider: ExaSearchProvider) -> None:
    respx.post("https://gateway.example/api/search").mock(
        return_value=httpx.Response(200, json={"results": {}})
    )

    with pytest.raises(SearchProviderError, match="malformed search response"):
        await provider.search("bad", max_results=1)


@pytest.mark.asyncio
@respx.mock
async def test_malformed_result_entry_fails(provider: ExaSearchProvider) -> None:
    respx.post("https://gateway.example/api/search").mock(
        return_value=httpx.Response(200, json={"results": [{}, "invalid"]})
    )

    with pytest.raises(SearchProviderError, match="malformed search response"):
        await provider.search("bad", max_results=1)


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_http_failures_are_sanitized(
    provider: ExaSearchProvider, status: int
) -> None:
    secret = "exa-secret response body"
    respx.post("https://gateway.example/api/search").mock(
        return_value=httpx.Response(status, text=secret)
    )

    with pytest.raises(SearchProviderError) as caught:
        await provider.search("failure", max_results=1)

    assert str(caught.value) == f"Search provider returned HTTP status {status}"
    assert secret not in str(caught.value)


@pytest.mark.asyncio
@respx.mock
async def test_network_failure_is_sanitized(provider: ExaSearchProvider) -> None:
    respx.post("https://gateway.example/api/search").mock(
        side_effect=httpx.ConnectError("exa-secret")
    )

    with pytest.raises(SearchProviderError, match="Search request failed"):
        await provider.search("failure", max_results=1)


@pytest.mark.asyncio
@respx.mock
async def test_oversized_body_fails_at_transport_seam(
    provider: ExaSearchProvider,
) -> None:
    respx.post("https://gateway.example/api/search").mock(
        return_value=httpx.Response(200, content=b"{" + b"x" * MAX_UPSTREAM_JSON_BYTES)
    )

    with pytest.raises(SearchProviderError, match="size limit"):
        await provider.search("large", max_results=1)


@pytest.mark.asyncio
@respx.mock
async def test_cross_origin_redirect_does_not_forward_credentials(
    provider: ExaSearchProvider,
) -> None:
    respx.post("https://gateway.example/api/search").mock(
        return_value=httpx.Response(
            302, headers={"Location": "https://other.example/search"}
        )
    )
    target = respx.post("https://other.example/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    with pytest.raises(SearchProviderError, match="not permitted"):
        await provider.search("redirect", max_results=1)
    assert target.call_count == 0
