from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
from typing import Protocol
import unicodedata
from urllib.parse import urljoin, urlparse

import httpx

from chartreux.core.tools.search.models import (
    MAX_QUERY_LENGTH,
    MAX_RESULTS,
    MAX_SERIALIZED_RESULT_BYTES,
    MAX_SNIPPET_LENGTH,
    MAX_UPSTREAM_JSON_BYTES,
    SearchResponse,
    SearchSource,
)
from chartreux.utils.http import ChartreuxAsyncHTTPClient, build_ssl_context


class SearchProvider(Protocol):
    async def search(self, query: str, *, max_results: int) -> SearchResponse: ...


class SearchProviderError(Exception):
    """An error whose message is safe to expose to callers and models."""

    def __init__(self, safe_message: str) -> None:
        super().__init__(safe_message)
        self.safe_message = safe_message


def validate_query(query: str) -> str:
    if not query.strip():
        raise SearchProviderError("Search query cannot be empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise SearchProviderError(
            f"Search query cannot exceed {MAX_QUERY_LENGTH} characters"
        )
    return query


def normalize_search_response(
    response: SearchResponse, *, max_results: int
) -> SearchResponse:
    """Apply output bounds without changing URL query-string semantics."""
    limit = max(0, min(max_results, MAX_RESULTS))
    was_truncated = response.was_truncated
    answer, truncated = _truncate_text(
        _sanitize_text(response.answer), MAX_SNIPPET_LENGTH
    )
    was_truncated |= truncated

    seen_urls: set[str] = set()
    sources: list[SearchSource] = []
    for source in response.sources:
        if not _is_http_url(source.url) or source.url in seen_urls:
            continue
        seen_urls.add(source.url)
        if len(sources) >= limit:
            was_truncated = True
            continue
        title, title_truncated = _truncate_text(
            _sanitize_text(source.title), MAX_SNIPPET_LENGTH
        )
        snippet, snippet_truncated = _truncate_text(
            _sanitize_text(source.snippet), MAX_SNIPPET_LENGTH
        )
        was_truncated |= title_truncated or snippet_truncated
        sources.append(SearchSource(title=title or "", url=source.url, snippet=snippet))

    normalized = SearchResponse(
        query=response.query,
        provider=response.provider,
        answer=answer,
        sources=sources,
        was_truncated=was_truncated,
    )
    return _bound_serialized_result(normalized)


def _sanitize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return "".join(
        character
        for character in value
        if character in "\t\n\r" or unicodedata.category(character) != "Cc"
    )


def _truncate_text(value: str | None, limit: int) -> tuple[str | None, bool]:
    if value is None or len(value) <= limit:
        return value, False
    return value[:limit], True


def _is_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    except ValueError:
        return False


def _bound_serialized_result(response: SearchResponse) -> SearchResponse:
    """Ensure a response is safely serializable within the model result budget."""
    if _serialized_size(response) <= MAX_SERIALIZED_RESULT_BYTES:
        return response

    sources = list(response.sources)
    while sources and _serialized_size(
        response.model_copy(update={"sources": sources})
    ) > (MAX_SERIALIZED_RESULT_BYTES):
        sources.pop()

    response = response.model_copy(update={"sources": sources, "was_truncated": True})
    if _serialized_size(response) <= MAX_SERIALIZED_RESULT_BYTES:
        return response

    # A provider name or answer may still be excessive; trim text by UTF-8 bytes.
    answer = _truncate_to_serialized_budget(response, response.answer, "answer")
    response = response.model_copy(update={"answer": answer, "was_truncated": True})
    if _serialized_size(response) <= MAX_SERIALIZED_RESULT_BYTES:
        return response

    query = _truncate_to_serialized_budget(response, response.query, "query")
    response = response.model_copy(update={"query": query, "was_truncated": True})
    if _serialized_size(response) <= MAX_SERIALIZED_RESULT_BYTES:
        return response

    provider = _truncate_to_serialized_budget(response, response.provider, "provider")
    return response.model_copy(
        update={"provider": provider or "", "was_truncated": True}
    )


def _truncate_to_serialized_budget(
    response: SearchResponse, value: str | None, field: str
) -> str | None:
    if value is None:
        return None

    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = response.model_copy(update={field: value[:middle]})
        if _serialized_size(candidate) <= MAX_SERIALIZED_RESULT_BYTES:
            low = middle
        else:
            high = middle - 1
    return value[:low]


def _serialized_size(response: SearchResponse) -> int:
    return len(response.model_dump_json().encode("utf-8"))


async def fetch_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float,
    method: str = "GET",
    json_body: object | None = None,
) -> object:
    """Fetch a bounded JSON body, rejecting unsafe credential redirects."""
    if timeout <= 0:
        raise SearchProviderError("Search timeout must be positive")
    request_headers = dict(headers or {})
    try:
        async with ChartreuxAsyncHTTPClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            verify=build_ssl_context(),
        ) as client:
            body = await asyncio.wait_for(
                _request_with_checked_redirects(
                    client, method, url, request_headers, json_body
                ),
                timeout=timeout,
            )
    except asyncio.CancelledError:
        raise
    except SearchProviderError:
        raise
    except (httpx.TimeoutException, TimeoutError) as error:
        raise SearchProviderError("Search request timed out") from error
    except httpx.RequestError as error:
        raise SearchProviderError("Search request failed") from error

    try:
        return json.loads(body)
    except (TypeError, ValueError) as error:
        raise SearchProviderError("Search provider returned malformed JSON") from error


async def _request_with_checked_redirects(
    client: ChartreuxAsyncHTTPClient,
    method: str,
    url: str,
    headers: dict[str, str],
    json_body: object | None,
) -> bytes:
    for redirect_count in range(client.max_redirects + 1):
        async with client.stream(
            method, url, headers=headers, json=json_body
        ) as response:
            if response.is_redirect:
                location = response.headers.get("Location")
                if location is None:
                    raise SearchProviderError(
                        "Search provider returned an invalid redirect"
                    )
                redirect_url = urljoin(url, location)
                if _has_credentials(headers) and _is_cross_origin(url, redirect_url):
                    raise SearchProviderError(
                        "Search provider redirect was not permitted"
                    )
                if redirect_count == client.max_redirects:
                    raise SearchProviderError(
                        "Search provider redirected too many times"
                    )
                url = redirect_url
                continue

            body = await _read_bounded_body(response)
            if response.is_error:
                raise SearchProviderError(
                    f"Search provider returned HTTP status {response.status_code}"
                )
            return body
    raise AssertionError("unreachable")


async def _read_bounded_body(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_UPSTREAM_JSON_BYTES:
            raise SearchProviderError(
                "Search provider response exceeded the size limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _has_credentials(headers: Mapping[str, str]) -> bool:
    credential_headers = {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-subscription-token",
        "api-key",
    }
    return any(name.lower() in credential_headers for name in headers)


def _is_cross_origin(url: str, redirect_url: str) -> bool:
    source = urlparse(url)
    target = urlparse(redirect_url)
    return (source.scheme, source.netloc.lower()) != (
        target.scheme,
        target.netloc.lower(),
    )
