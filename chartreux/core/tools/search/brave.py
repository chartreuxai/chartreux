from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlencode

from chartreux.core.tools.search.models import MAX_RESULTS, SearchResponse, SearchSource
from chartreux.core.tools.search.provider import (
    SearchProviderError,
    fetch_json,
    normalize_search_response,
    validate_query,
)

_DEFAULT_BASE_URL = "https://api.search.brave.com"
_DEFAULT_TIMEOUT = 10.0
_MAX_QUERY_LENGTH = 600
_MAX_QUERY_WORDS = 75


class BraveSearchProvider:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int) -> SearchResponse:
        query = _validate_brave_query(validate_query(query))
        result_count = max(0, min(max_results, MAX_RESULTS))
        url = f"{self._base_url}/res/v1/web/search?{urlencode({'q': query, 'count': result_count})}"
        payload = await fetch_json(
            url, headers={"X-Subscription-Token": self._api_key}, timeout=self._timeout
        )
        results = _get_results(payload)
        response = SearchResponse(
            query=query,
            provider="brave",
            answer=None,
            sources=[_to_source(result) for result in results],
            was_truncated=False,
        )
        return normalize_search_response(response, max_results=result_count)


def _validate_brave_query(query: str) -> str:
    if len(query) > _MAX_QUERY_LENGTH or len(query.split()) > _MAX_QUERY_WORDS:
        raise SearchProviderError(
            "Brave search queries are limited to 600 characters and 75 words; "
            "shorten your query"
        )
    return query


def _get_results(payload: object) -> list[Mapping[object, object]]:
    if not isinstance(payload, Mapping):
        raise SearchProviderError("Brave returned a malformed search response")
    web = payload.get("web")
    if web is None:
        return []
    if (
        not isinstance(web, Mapping)
        or not isinstance(results := web.get("results"), list)
        or not all(isinstance(result, Mapping) for result in results)
    ):
        raise SearchProviderError("Brave returned a malformed search response")
    return results


def _to_source(result: Mapping[object, object]) -> SearchSource:
    return SearchSource(
        title=_text(result.get("title")) or "",
        url=_text(result.get("url")) or "",
        snippet=_text(result.get("description")),
    )


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None
