from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import functools
from typing import TYPE_CHECKING, final
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field, model_validator

from chartreux.core.events import ToolStreamEvent
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
)
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.utils.http import ChartreuxAsyncHTTPClient, build_ssl_context
from chartreux.utils.tool_presentation import ToolEffectKind

if TYPE_CHECKING:
    from chartreux.core.events import ToolCallEvent, ToolResultEvent


_HONEST_USER_AGENT = "chartreux-cli"
_HTTP_FORBIDDEN = 403


@functools.cache
def _make_converter_class() -> type:
    from markdownify import MarkdownConverter

    class _Converter(MarkdownConverter):
        convert_script = convert_style = convert_noscript = convert_iframe = (
            convert_object
        ) = convert_embed = lambda *_, **__: ""

    return _Converter


class WebFetchArgs(BaseModel):
    url: str = Field(description="The URL to fetch content from")
    timeout: int | None = Field(
        default=None, description="Optional timeout in seconds (max 120)"
    )


class WebFetchResult(BaseModel):
    url: str
    content: str
    content_type: str
    was_truncated: bool = False


class WebFetchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK

    default_timeout: int = Field(default=30, description="Default timeout in seconds.")
    max_timeout: int = Field(default=120, description="Maximum allowed timeout.")
    max_content_bytes: int = Field(
        default=120_000,
        description="Maximum content size in bytes returned to the model.",
    )
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        description="User agent string for requests.",
    )

    @model_validator(mode="after")
    def reject_unsupported_url_denylist(self) -> WebFetchConfig:
        if self.denylist:
            raise ValueError("WebFetch URL denylist not yet supported")
        return self


class WebFetch(
    BaseTool[WebFetchArgs, WebFetchResult, WebFetchConfig, BaseToolState],
    ToolUIData[WebFetchArgs, WebFetchResult],
):
    effect_kind = ToolEffectKind.WEB_FETCH

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalise a URL to always have an http(s) scheme.

        Handles protocol-relative URLs (//example.com) and bare URLs (example.com).
        """
        raw = url.lstrip("/") if url.startswith("//") else url
        return raw if raw.startswith(("http://", "https://")) else "https://" + raw

    def resolve_permission(self, args: WebFetchArgs) -> PermissionContext | None:
        if self.config.permission in {ToolPermission.ALWAYS, ToolPermission.NEVER}:
            return PermissionContext(permission=self.config.permission)

        parsed = urlparse(self._normalize_url(args.url))
        domain = parsed.netloc or parsed.path.split("/")[0]
        if not domain:
            return None

        return PermissionContext(
            permission=ToolPermission.ASK,
            required_permissions=[
                RequiredPermission(
                    scope=PermissionScope.URL_PATTERN,
                    invocation_pattern=domain,
                    session_pattern=domain,
                    label=f"fetching from {domain}",
                )
            ],
        )

    @final
    async def run(
        self, args: WebFetchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WebFetchResult, None]:
        self._validate_args(args)

        url = self._normalize_url(args.url)
        timeout = self._resolve_timeout(args.timeout)

        content, content_type, was_truncated = await self._fetch_url(url, timeout)

        if "text/html" in content_type:
            content = _html_to_markdown(content)

        content_bytes = content.encode("utf-8")
        if len(content_bytes) > self.config.max_content_bytes:
            was_truncated = True
            content = content_bytes[: self.config.max_content_bytes].decode(
                "utf-8", errors="ignore"
            )
        if was_truncated:
            content += "\n\n[Content truncated due to size limit]"

        yield WebFetchResult(
            url=url,
            content=content,
            content_type=content_type,
            was_truncated=was_truncated,
        )

    def _validate_args(self, args: WebFetchArgs) -> None:
        if not args.url.strip():
            raise ToolError("URL cannot be empty")

        parsed = urlparse(args.url)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            raise ToolError(
                f"Invalid URL scheme: {parsed.scheme}. Must be http or https."
            )

        if args.timeout is not None:
            if args.timeout <= 0:
                raise ToolError("Timeout must be a positive number")
            if args.timeout > self.config.max_timeout:
                raise ToolError(
                    f"Timeout cannot exceed {self.config.max_timeout} seconds"
                )

    def _resolve_timeout(self, timeout: int | None) -> int:
        if timeout is None:
            return self.config.default_timeout
        return min(timeout, self.config.max_timeout)

    async def _fetch_url(self, url: str, timeout: int) -> tuple[str, str, bool]:
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            async with (
                asyncio.timeout(timeout),
                self._do_fetch(url, timeout, headers) as response,
            ):
                if response.is_error:
                    raise ToolError(
                        f"HTTP error {response.status_code}: {response.reason_phrase}"
                    )
                content, was_truncated = await self._read_content(response)
                return (
                    content.decode(response.encoding or "utf-8", errors="replace"),
                    response.headers.get("Content-Type", "text/plain"),
                    was_truncated,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise ToolError(f"Request timed out after {timeout} seconds")
        except httpx.RequestError as e:
            raise ToolError(f"Failed to fetch URL: {e}")

    async def _read_content(self, response: httpx.Response) -> tuple[bytes, bool]:
        """Read at most the model-facing byte limit, closing oversized bodies early."""
        content = bytearray()
        async for chunk in response.aiter_bytes():
            remaining = self.config.max_content_bytes - len(content)
            if len(chunk) > remaining:
                content.extend(chunk[:remaining])
                return bytes(content), True
            content.extend(chunk)
        return bytes(content), False

    @asynccontextmanager
    async def _do_fetch(
        self, url: str, timeout: int, headers: dict[str, str]
    ) -> AsyncGenerator[httpx.Response, None]:
        allowed_netloc = urlparse(url).netloc.lower()

        async with ChartreuxAsyncHTTPClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            verify=build_ssl_context(),
        ) as client:
            response = await self._fetch_with_checked_redirects(
                client, url, headers, allowed_netloc
            )

            # In case we are hitting bot detection retry once honestly
            if (
                response.status_code == _HTTP_FORBIDDEN
                and response.headers.get("cf-mitigated") == "challenge"
            ):
                await response.aclose()
                headers["User-Agent"] = _HONEST_USER_AGENT
                response = await self._fetch_with_checked_redirects(
                    client, url, headers, allowed_netloc
                )

            try:
                yield response
            finally:
                await response.aclose()

    async def _fetch_with_checked_redirects(
        self,
        client: ChartreuxAsyncHTTPClient,
        url: str,
        headers: dict[str, str],
        allowed_netloc: str,
    ) -> httpx.Response:
        for redirect_count in range(client.max_redirects + 1):
            request = client.build_request("GET", url, headers=headers)
            response = await client.send(request, stream=True)
            if not response.is_redirect:
                return response

            location = response.headers.get("Location")
            if location is None:
                return response

            redirect_url = urljoin(url, location)
            if urlparse(redirect_url).netloc.lower() != allowed_netloc:
                await response.aclose()
                raise ToolError(f"Redirect target is not permitted: {redirect_url}")

            if redirect_count == client.max_redirects:
                await response.aclose()
                raise ToolError("Too many redirects")

            await response.aclose()
            url = redirect_url

        raise AssertionError("unreachable")

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if event.args is None:
            return ToolCallDisplay(
                summary="web_fetch",
                verb="Running",
                message="web_fetch",
                settled_verb="Ran",
                settled_message="web_fetch",
            )
        if not isinstance(event.args, WebFetchArgs):
            return ToolCallDisplay(
                summary="web_fetch",
                verb="Running",
                message="web_fetch",
                settled_verb="Ran",
                settled_message="web_fetch",
            )

        parsed = urlparse(event.args.url)
        domain = parsed.netloc or event.args.url[:50]
        message = domain

        if event.args.timeout:
            message += f" (timeout {event.args.timeout}s)"

        return ToolCallDisplay(
            summary=f"Fetching: {message}",
            verb="Fetching",
            message=message,
            settled_verb="Fetched",
            settled_message=message,
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, WebFetchResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        content_len = len(event.result.content)
        content_type = event.result.content_type.split(";")[0]
        message = f"{event.result.url} ({content_len:,} chars, {content_type})"
        suffix = "(truncated)" if event.result.was_truncated else ""

        return ToolResultDisplay(
            success=True, verb="Fetched", message=message, suffix=suffix
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Fetching URL"


def _html_to_markdown(html: str) -> str:
    converter_class = _make_converter_class()
    return converter_class(heading_style="ATX", bullets="-").convert(html)
