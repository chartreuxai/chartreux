"""Safe, bounded model-list discovery for provider onboarding."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx

from chartreux.ui.providers.contracts import (
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
    ProviderDraft,
    TLSConfig,
)
from chartreux.utils.http import ChartreuxAsyncHTTPClient

_CONNECT_TIMEOUT = 5.0
_READ_TIMEOUT = 15.0
_OVERALL_TIMEOUT = 30.0
_MAX_PAGES = 100
_ANTHROPIC_VERSION = "2023-06-01"


def _merge_headers(*layers: Mapping[str, str] | None) -> dict[str, str]:
    """Merge header layers case-insensitively, with later layers winning."""
    merged: dict[str, str] = {}
    names: dict[str, str] = {}
    for layer in layers:
        for name, value in (layer or {}).items():
            folded = name.casefold()
            if previous := names.get(folded):
                merged.pop(previous, None)
            merged[name] = value
            names[folded] = name
    return merged


def _listing_url(provider: ProviderDraft) -> str:
    base = provider.api_base.rstrip("/")
    if provider.api_style == "anthropic":
        return f"{base}/v1/models"
    return f"{base}/models"


def _headers(provider: ProviderDraft, credential: str | None) -> dict[str, str]:
    if provider.api_style == "anthropic":
        generated = {"anthropic-version": _ANTHROPIC_VERSION}
        if credential:
            generated["x-api-key"] = credential
    else:
        generated = {}
        if credential:
            generated["Authorization"] = f"Bearer {credential}"
    return _merge_headers(provider.extra_headers, generated)


def _error_for_status(status: int) -> DiscoveryError:
    if status in {httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN}:
        return DiscoveryError(
            "auth_rejected", "Model listing credentials were rejected."
        )
    if status == httpx.codes.NOT_FOUND:
        return DiscoveryError(
            "unsupported_listing", "This provider does not support model listing."
        )
    if status == httpx.codes.TOO_MANY_REQUESTS:
        return DiscoveryError("rate_limited", "Model listing is rate limited.")
    return DiscoveryError(
        "unsupported_listing", "Model listing request was unsuccessful."
    )


def _parse_page(
    payload: object, *, anthropic: bool
) -> tuple[list[DiscoveryItem], str | None, bool]:
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("response data is not a list")

    items: list[DiscoveryItem] = []
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("response model has no string id")
        display_name = entry.get("display_name") if anthropic else None
        if display_name is not None and not isinstance(display_name, str):
            raise ValueError("response display name is not a string")
        items.append(DiscoveryItem(entry["id"], display_name))

    if anthropic:
        has_more = payload.get("has_more", False)
        if not isinstance(has_more, bool):
            raise ValueError("response has_more is not a boolean")
        after_id = payload.get("last_id")
        if after_id is not None and not isinstance(after_id, str):
            raise ValueError("response last_id is not a string")
        if has_more and not after_id:
            raise ValueError("response pagination cursor is missing")
        return items, after_id, has_more

    links = payload.get("links")
    if links is None:
        return items, None, False
    if not isinstance(links, dict):
        raise ValueError("response links is not an object")
    next_url = links.get("next")
    if next_url is not None and not isinstance(next_url, str):
        raise ValueError("response next link is not a string")
    return items, next_url, bool(next_url)


def _next_url(current: str, next_link: str) -> str:
    """Resolve a pagination link without permitting credential exfiltration."""
    candidate = (
        httpx.URL(next_link)
        if "://" in next_link
        else httpx.URL(current).join(next_link)
    )
    current_url = httpx.URL(current)
    if (candidate.scheme, candidate.host, candidate.port) != (
        current_url.scheme,
        current_url.host,
        current_url.port,
    ):
        raise ValueError("pagination link changes origin")
    return str(candidate)


async def _fetch_pages(
    client: httpx.AsyncClient, *, url: str, headers: Mapping[str, str], anthropic: bool
) -> DiscoveryResult | DiscoveryError:
    models: list[DiscoveryItem] = []
    seen_ids: set[str] = set()
    seen_cursors: set[str] = set()
    cursor: str | None = None

    for _ in range(_MAX_PAGES):
        params = {"after_id": cursor} if anthropic and cursor is not None else None
        response = await client.get(
            url, headers=headers, params=params, follow_redirects=False
        )
        if not response.is_success:
            return _error_for_status(response.status_code)
        try:
            payload: Any = response.json()
            page, cursor_or_next, more = _parse_page(payload, anthropic=anthropic)
        except (TypeError, ValueError):
            return DiscoveryError("malformed", "Model listing response was malformed.")

        for item in page:
            if item.wire_id not in seen_ids:
                seen_ids.add(item.wire_id)
                models.append(item)

        next_page = _next_page(url, cursor_or_next, more, anthropic, seen_cursors)
        if isinstance(next_page, DiscoveryError):
            return next_page
        if next_page is None:
            return _result(models)
        if anthropic:
            cursor = next_page
        else:
            url = next_page

    return DiscoveryError("malformed", "Model listing exceeded the pagination limit.")


def _next_page(
    url: str,
    cursor_or_next: str | None,
    more: bool,
    anthropic: bool,
    seen_cursors: set[str],
) -> str | DiscoveryError | None:
    if not more:
        return None
    if cursor_or_next is None:
        return DiscoveryError("malformed", "Model listing response was malformed.")
    if anthropic:
        if cursor_or_next in seen_cursors:
            return DiscoveryError(
                "malformed", "Model listing pagination did not advance."
            )
        seen_cursors.add(cursor_or_next)
        return cursor_or_next
    try:
        return _next_url(url, cursor_or_next)
    except ValueError:
        return DiscoveryError("malformed", "Model listing pagination link was invalid.")


def _result(models: list[DiscoveryItem]) -> DiscoveryResult | DiscoveryError:
    if not models:
        return DiscoveryError("empty", "Model listing returned no models.")
    return DiscoveryResult(tuple(models), ("Models discovered",))


def _request_error(error: httpx.TransportError) -> DiscoveryError:
    if isinstance(error, httpx.TimeoutException):
        return DiscoveryError("connection", "Model listing timed out.")
    if isinstance(error.__cause__, OSError) and "SSL" in str(error.__cause__).upper():
        return DiscoveryError(
            "tls", "TLS verification failed for the model listing endpoint."
        )
    return DiscoveryError("connection", "Model listing connection failed.")


async def _discover_with_client(
    client: httpx.AsyncClient, provider: ProviderDraft, credential: str | None
) -> DiscoveryResult | DiscoveryError:
    async with asyncio.timeout(_OVERALL_TIMEOUT):
        return await _fetch_pages(
            client,
            url=_listing_url(provider),
            headers=_headers(provider, credential),
            anthropic=provider.api_style == "anthropic",
        )


async def discover_models(
    provider: ProviderDraft,
    credential: str | None,
    tls: TLSConfig,
    http_client: httpx.AsyncClient | None = None,
) -> DiscoveryResult | DiscoveryError:
    """Return raw provider model IDs, or a safe error suitable for manual fallback."""
    try:
        if http_client is not None:
            return await _discover_with_client(http_client, provider, credential)
        timeout = httpx.Timeout(
            _READ_TIMEOUT, connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT
        )
        async with ChartreuxAsyncHTTPClient(
            timeout=timeout,
            follow_redirects=False,
            enable_system_trust_store=tls.enable_system_trust_store,
        ) as client:
            return await _discover_with_client(client, provider, credential)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return DiscoveryError("connection", "Model listing timed out.")
    except httpx.TransportError as error:
        return _request_error(error)


DiscoveryService = discover_models

__all__ = ["DiscoveryService", "discover_models"]
