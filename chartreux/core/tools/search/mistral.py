from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
import json

from chartreux.core.tools.search.models import (
    MAX_RESULTS,
    MAX_UPSTREAM_JSON_BYTES,
    SearchResponse,
    SearchSource,
)
from chartreux.core.tools.search.provider import (
    SearchProviderError,
    normalize_search_response,
    validate_query,
)

_DEFAULT_TIMEOUT = 10.0


class MistralSearchProvider:
    def __init__(
        self,
        api_key: str,
        server_url: str | None,
        *,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not server_url or not server_url.strip():
            raise SearchProviderError("Mistral search requires a configured server URL")
        if timeout <= 0:
            raise SearchProviderError("Search timeout must be positive")
        self._api_key = api_key
        self._server_url = server_url
        self._model = model
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int) -> SearchResponse:
        query = validate_query(query)
        result_count = max(0, min(max_results, MAX_RESULTS))
        try:
            # Imported only on use: discovery of search providers must not load the SDK.
            from mistralai.client import Mistral

            client = Mistral(api_key=self._api_key, server_url=self._server_url)
            async with client:
                # This is an intentionally pinned SDK seam. mistralai 2.6.0 otherwise
                # enables telemetry according to its environment configuration.
                client.sdk_configuration.__dict__["telemetry"] = False
                response = await client.beta.conversations.start_async(
                    model=self._model,
                    inputs=query,
                    tools=[{"type": "web_search"}],
                    store=False,
                    timeout_ms=int(self._timeout * 1000),
                )
            _check_payload_size(response)
            return _parse_response(response, query, result_count)
        except asyncio.CancelledError:
            raise
        except SearchProviderError:
            raise
        except TimeoutError as error:
            raise SearchProviderError("Mistral search request timed out") from error
        except Exception as error:
            raise SearchProviderError("Mistral search request failed") from error


def _check_payload_size(response: object) -> None:
    """Bound parsed SDK output; the SDK owns HTTP, so wire interception is unreliable."""
    try:
        if callable(model_dump_json := getattr(response, "model_dump_json", None)):
            payload = model_dump_json()
        else:
            payload = json.dumps(response, default=_json_default, separators=(",", ":"))
        if not isinstance(payload, str):
            raise TypeError("Serialized response was not text")
        size = len(payload.encode("utf-8"))
    except (TypeError, ValueError) as error:
        raise SearchProviderError(
            "Mistral returned a malformed search response"
        ) from error
    if size > MAX_UPSTREAM_JSON_BYTES:
        raise SearchProviderError("Mistral search response exceeded the size limit")


def _json_default(value: object) -> object:
    if callable(model_dump := getattr(value, "model_dump", None)):
        return model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parse_response(response: object, query: str, max_results: int) -> SearchResponse:
    outputs = getattr(response, "outputs", None)
    if not isinstance(outputs, Iterable) or isinstance(outputs, (str, bytes, Mapping)):
        raise SearchProviderError("Mistral returned a malformed search response")

    text_parts: list[str] = []
    sources: list[SearchSource] = []
    saw_reference = False
    for entry in outputs:
        content = getattr(entry, "content", None)
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, Iterable) and not isinstance(
            content, (str, bytes, Mapping)
        ):
            for chunk in content:
                chunk_type = getattr(chunk, "type", None)
                text = getattr(chunk, "text", None)
                url = getattr(chunk, "url", None)
                if chunk_type == "text" and isinstance(text, str):
                    text_parts.append(text)
                elif chunk_type == "tool_reference":
                    saw_reference = True
                    if isinstance(url, str):
                        title = getattr(chunk, "title", None)
                        sources.append(
                            SearchSource(
                                title=title if isinstance(title, str) else "",
                                url=url,
                                snippet=None,
                            )
                        )

    answer = "".join(text_parts).strip() or None
    if answer is None and not saw_reference:
        raise SearchProviderError("Mistral returned an empty search response")
    return normalize_search_response(
        SearchResponse(
            query=query,
            provider="mistral",
            answer=answer,
            sources=sources,
            was_truncated=False,
        ),
        max_results=max_results,
    )
