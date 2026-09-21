from __future__ import annotations

import httpx
import pytest
import respx

from chartreux.core.tools.search.brave import BraveSearchProvider
from chartreux.core.tools.search.models import MAX_UPSTREAM_JSON_BYTES
from chartreux.core.tools.search.provider import SearchProviderError


@pytest.fixture
def provider() -> BraveSearchProvider:
    return BraveSearchProvider(
        "brave-secret", base_url="https://gateway.example/api", timeout=4.5
    )


@pytest.mark.asyncio
@respx.mock
async def test_search_gets_expected_request_and_normalizes_results(
    provider: BraveSearchProvider,
) -> None:
    route = respx.get("https://gateway.example/api/res/v1/web/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "One",
                            "url": "https://one.example",
                            "description": "first",
                        },
                        {"url": "https://two.example"},
                    ]
                }
            },
        )
    )

    result = await provider.search("find this", max_results=7)

    assert result.answer is None
    assert [source.model_dump() for source in result.sources] == [
        {"title": "One", "url": "https://one.example", "snippet": "first"},
        {"title": "", "url": "https://two.example", "snippet": None},
    ]
    request = route.calls[0].request
    assert request.method == "GET"
    assert request.url == httpx.URL(
        "https://gateway.example/api/res/v1/web/search?q=find+this&count=7"
    )
    assert request.headers["X-Subscription-Token"] == "brave-secret"
    assert dict(request.url.params) == {"q": "find this", "count": "7"}
    assert request.extensions["timeout"]["read"] == 4.5


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("payload", [{}, {"web": None}, {"web": {"results": []}}])
async def test_empty_results_succeed(
    provider: BraveSearchProvider, payload: dict[str, object]
) -> None:
    respx.get("https://gateway.example/api/res/v1/web/search").mock(
        return_value=httpx.Response(200, json=payload)
    )

    result = await provider.search("none", max_results=1)

    assert result.sources == []
    assert result.was_truncated is False


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("query", ["x" * 601, " ".join("word" for _ in range(76))])
async def test_search_rejects_queries_beyond_brave_limits(
    provider: BraveSearchProvider, query: str
) -> None:
    with pytest.raises(SearchProviderError, match="shorten your query"):
        await provider.search(query, max_results=1)


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "query", [" ".join("x" for _ in range(75)), ("xxxxxxx " * 74) + "xxxxxxxx"]
)
async def test_search_accepts_queries_at_brave_limits(
    provider: BraveSearchProvider, query: str
) -> None:
    route = respx.get("https://gateway.example/api/res/v1/web/search").mock(
        return_value=httpx.Response(200, json={"web": {"results": []}})
    )

    await provider.search(query, max_results=1)

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_malformed_envelope_fails(provider: BraveSearchProvider) -> None:
    respx.get("https://gateway.example/api/res/v1/web/search").mock(
        return_value=httpx.Response(200, json={"web": {}})
    )

    with pytest.raises(SearchProviderError, match="malformed search response"):
        await provider.search("bad", max_results=1)


@pytest.mark.asyncio
@respx.mock
async def test_malformed_result_entry_fails(provider: BraveSearchProvider) -> None:
    respx.get("https://gateway.example/api/res/v1/web/search").mock(
        return_value=httpx.Response(200, json={"web": {"results": [{}, "invalid"]}})
    )

    with pytest.raises(SearchProviderError, match="malformed search response"):
        await provider.search("bad", max_results=1)


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_http_failures_are_sanitized(
    provider: BraveSearchProvider, status: int
) -> None:
    secret = "brave-secret response body"
    respx.get("https://gateway.example/api/res/v1/web/search").mock(
        return_value=httpx.Response(status, text=secret)
    )

    with pytest.raises(SearchProviderError) as caught:
        await provider.search("failure", max_results=1)

    assert str(caught.value) == f"Search provider returned HTTP status {status}"
    assert secret not in str(caught.value)


@pytest.mark.asyncio
@respx.mock
async def test_network_failure_is_sanitized(provider: BraveSearchProvider) -> None:
    respx.get("https://gateway.example/api/res/v1/web/search").mock(
        side_effect=httpx.ConnectError("brave-secret")
    )

    with pytest.raises(SearchProviderError, match="Search request failed"):
        await provider.search("failure", max_results=1)


@pytest.mark.asyncio
@respx.mock
async def test_oversized_body_fails_at_transport_seam(
    provider: BraveSearchProvider,
) -> None:
    respx.get("https://gateway.example/api/res/v1/web/search").mock(
        return_value=httpx.Response(200, content=b"{" + b"x" * MAX_UPSTREAM_JSON_BYTES)
    )

    with pytest.raises(SearchProviderError, match="size limit"):
        await provider.search("large", max_results=1)


@pytest.mark.asyncio
@respx.mock
async def test_cross_origin_redirect_does_not_forward_credentials(
    provider: BraveSearchProvider,
) -> None:
    respx.get("https://gateway.example/api/res/v1/web/search").mock(
        return_value=httpx.Response(
            302, headers={"Location": "https://other.example/search"}
        )
    )
    target = respx.get("https://other.example/search").mock(
        return_value=httpx.Response(200, json={"web": {"results": []}})
    )

    with pytest.raises(SearchProviderError, match="not permitted"):
        await provider.search("redirect", max_results=1)
    assert target.call_count == 0
