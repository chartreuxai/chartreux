from __future__ import annotations

from collections.abc import Mapping

from chartreux.core.tools.search.models import (
    MAX_RESULTS,
    MAX_SNIPPET_LENGTH,
    SearchResponse,
    SearchSource,
)
from chartreux.core.tools.search.provider import (
    SearchProviderError,
    fetch_json,
    normalize_search_response,
    validate_query,
)

_DEFAULT_BASE_URL = "https://api.exa.ai"
_DEFAULT_TIMEOUT = 10.0


class ExaSearchProvider:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/search"
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int) -> SearchResponse:
        query = validate_query(query)
        result_count = max(0, min(max_results, MAX_RESULTS))
        payload = await fetch_json(
            self._url,
            method="POST",
            headers={"x-api-key": self._api_key},
            json_body={
                "query": query,
                "numResults": result_count,
                "contents": {"text": {"maxCharacters": MAX_SNIPPET_LENGTH}},
            },
            timeout=self._timeout,
        )
        results = _get_results(payload)
        response = SearchResponse(
            query=query,
            provider="exa",
            answer=None,
            sources=[_to_source(result) for result in results],
            was_truncated=False,
        )
        return normalize_search_response(response, max_results=result_count)


def _get_results(payload: object) -> list[Mapping[object, object]]:
    if (
        not isinstance(payload, Mapping)
        or not isinstance(results := payload.get("results"), list)
        or not all(isinstance(result, Mapping) for result in results)
    ):
        raise SearchProviderError("Exa returned a malformed search response")
    return results


def _to_source(result: Mapping[object, object]) -> SearchSource:
    return SearchSource(
        title=_text(result.get("title")) or "",
        url=_text(result.get("url")) or "",
        snippet=_text(result.get("text")),
    )


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None
