from __future__ import annotations

import asyncio
from collections.abc import Mapping

from chartreux.core.tools.search.models import MAX_RESULTS, SearchResponse, SearchSource
from chartreux.core.tools.search.provider import (
    SearchProviderError,
    normalize_search_response,
    validate_query,
)

_DEFAULT_TIMEOUT = 10.0


class DuckDuckGoSearchProvider:
    def __init__(self, *, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int) -> SearchResponse:
        query = validate_query(query)
        result_count = max(0, min(max_results, MAX_RESULTS))
        try:
            # DDGS does not expose its response stream, so its wire payload cannot be
            # capped. The deadline bounds how long this caller waits; its worker thread
            # may complete in the background after cancellation.
            results = await asyncio.wait_for(
                asyncio.to_thread(_search_ddgs, query, result_count, self._timeout),
                timeout=self._timeout,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise SearchProviderError("DuckDuckGo search timed out") from None
        except _DuckDuckGoRateLimitError as error:
            raise SearchProviderError("DuckDuckGo search was rate limited") from error
        except _DuckDuckGoTimeoutError as error:
            raise SearchProviderError("DuckDuckGo search timed out") from error
        except _DuckDuckGoRequestError as error:
            raise SearchProviderError("DuckDuckGo search request failed") from error
        except Exception as error:
            raise SearchProviderError("DuckDuckGo search request failed") from error

        response = SearchResponse(
            query=query,
            provider="duckduckgo",
            answer=None,
            sources=[_to_source(result) for result in _get_results(results)],
            was_truncated=False,
        )
        return normalize_search_response(response, max_results=result_count)


def _search_ddgs(query: str, max_results: int, timeout: float) -> object:
    """Run DDGS synchronously; imports stay lazy so tool discovery needs no ddgs install."""
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

    try:
        return DDGS(timeout=timeout).text(
            query, max_results=max_results, backend="duckduckgo"
        )
    except RatelimitException as error:
        raise _DuckDuckGoRateLimitError from error
    except TimeoutException as error:
        raise _DuckDuckGoTimeoutError from error
    except DDGSException as error:
        if str(error) == "No results found.":
            return []
        raise _DuckDuckGoRequestError from error


def _get_results(results: object) -> list[Mapping[object, object]]:
    if not isinstance(results, list) or not all(
        isinstance(result, Mapping) for result in results
    ):
        raise SearchProviderError("DuckDuckGo returned a malformed search response")
    return results


def _to_source(result: Mapping[object, object]) -> SearchSource:
    return SearchSource(
        title=_text(result.get("title")) or "",
        url=_text(result.get("href")) or "",
        snippet=_text(result.get("body")),
    )


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


class _DuckDuckGoRateLimitError(Exception):
    pass


class _DuckDuckGoTimeoutError(Exception):
    pass


class _DuckDuckGoRequestError(Exception):
    pass
