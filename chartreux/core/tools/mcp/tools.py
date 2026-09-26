from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
import contextlib
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from chartreux.core.events import ToolStreamEvent
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
)
from chartreux.core.tools.mcp.authorization import (
    MCPAuthorizationProvider,
    MCPAuthorizationRef,
    MCPAuthorizationRequired,
    MCPAuthorizationRequiredSink,
    MCPAuthorizationSnapshot,
)
from chartreux.core.tools.remote import MCPTool, MCPToolResult, RemoteTool, _OpenArgs
from chartreux.core.tools.secret_redaction import scrub_child_env
from chartreux.core.tools.ui import ToolResultDisplay
from chartreux.observability.logging import logger
from chartreux.utils.http import ChartreuxAsyncHTTPClient, build_ssl_context
from chartreux.utils.io import decode_console_safe
from chartreux.utils.untrusted_content import frame_untrusted_content

if TYPE_CHECKING:
    from chartreux.core.events import ToolResultEvent
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters


def _mcp_sdk_attribute(name: str) -> Any:
    return getattr(sys.modules[__name__], name)


def __getattr__(name: str) -> Any:
    if name == "ClientSession":
        from mcp import ClientSession

        return ClientSession
    if name == "OAuthFlowError":
        from mcp.client.auth import OAuthFlowError

        return OAuthFlowError
    if name == "StdioServerParameters":
        from mcp.client.stdio import StdioServerParameters

        return StdioServerParameters
    if name == "get_default_environment":
        from mcp.client.stdio import get_default_environment

        return get_default_environment
    if name == "stdio_client":
        from mcp.client.stdio import stdio_client

        return stdio_client
    if name == "streamable_http_client":
        from mcp.client.streamable_http import streamable_http_client

        return streamable_http_client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Mirrors MCP's default Streamable HTTP timeout values while avoiding an import from
# mcp.shared._httpx_utils, which is an internal module.
_MCP_DEFAULT_TIMEOUT = 30.0
_MCP_DEFAULT_SSE_READ_TIMEOUT = 300.0

# The SDK converts a read_timeout_seconds overrun into McpError with this code
# (httpx.codes.REQUEST_TIMEOUT). This holds for ANY request the session makes —
# including initialize — so a server that hangs on startup counts the same as
# one that hangs on a call (desirable: a hung init means a hung server).
_MCP_CALL_TIMEOUT_CODE = 408


def _is_call_timeout(exc: BaseException) -> bool:
    """Classify the SDK's call-timeout signal, unwrapping exception groups."""
    from mcp.shared.exceptions import McpError

    if isinstance(exc, BaseExceptionGroup):
        return any(_is_call_timeout(child) for child in exc.exceptions)
    return isinstance(exc, McpError) and exc.error.code == _MCP_CALL_TIMEOUT_CODE


_MAX_PRESENTATION_SCHEMA_DEPTH = 64


def _presentation_schema(
    schema: dict[str, Any], source: str, depth: int = 0
) -> dict[str, Any]:
    """Frame schema prose in the model projection, not the executable schema."""
    if depth >= _MAX_PRESENTATION_SCHEMA_DEPTH:
        # Keep other tools and shallow fields available; omit all untrusted
        # content in the deep subtree rather than failing the whole catalog.
        return {
            "description": "MCP schema omitted: nesting exceeds 64 levels; simplify the server tool schema."
        }
    result = dict(schema)
    for key in ("description", "title"):
        if isinstance(result.get(key), str):
            result[key] = frame_untrusted_content(result[key], source)
    for key in (
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ):
        value = result.get(key)
        if isinstance(value, dict):
            result[key] = {
                name: _presentation_schema(child, source, depth + 1)
                if isinstance(child, dict)
                else child
                for name, child in value.items()
            }
    for key in (
        "items",
        "additionalProperties",
        "unevaluatedProperties",
        "not",
        "if",
        "then",
        "else",
        "contains",
    ):
        value = result.get(key)
        if isinstance(value, dict):
            result[key] = _presentation_schema(value, source, depth + 1)
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        value = result.get(key)
        if isinstance(value, list):
            result[key] = [
                _presentation_schema(child, source, depth + 1)
                if isinstance(child, dict)
                else child
                for child in value
            ]
    return result


def _mcp_description(description: str, source: str) -> str:
    return frame_untrusted_content(description, source)


class MCPServerCooldownError(RuntimeError):
    """An MCP server is failing fast inside a hung-server cooldown window."""

    code = "mcp_server_cooldown"

    def __init__(self, server: str, retry_in: float) -> None:
        self.server = server
        self.retry_in = retry_in
        super().__init__(
            f"MCP server {server!r} is cooling down after repeated request "
            f"timeouts; failing fast, retry in {retry_in:.0f}s"
        )


class MCPServerCircuitBreaker:
    """In-memory per-server cooldown for MCP servers that hang on requests.

    Mirrors the LLM-side ``AvailabilityRegistry`` pattern: timeouts accumulated
    since the last successful response put the server into a short cooldown
    during which calls fail fast instead of burning the full tool timeout; once
    the cooldown expires, a single-flight recovery probe (the next real call)
    re-admits the server on success. Any request timeout counts, including
    ``initialize``: the SDK wraps every request that overruns its
    ``read_timeout_seconds`` in McpError code 408, and a server that hangs on
    startup is the same failure as one that hangs on a call. Non-timeout
    failures (auth errors, tool-level errors, transport deaths, cancellation)
    never trip the breaker and never reset the timeout counter — they only
    release an in-flight recovery probe — so the counter is cumulative since
    the last success, which keeps the breaker fail-closed: timeouts keep
    accumulating across unrelated failures until a completed response proves
    the server responsive again.
    """

    TIMEOUT_THRESHOLD = 3
    COOLDOWN_SEC = 30.0
    # Honest retry estimate while a recovery probe is in flight: the probe
    # resolves within one tool timeout (success re-admits immediately, a
    # timeout re-trips the cooldown), so "retry in 0s" would be a lie — an
    # immediate retry fails fast too.
    PROBE_RETRY_SEC = 5.0

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._timeouts_since_success = 0
        self._cooldown_until: float | None = None
        self._epoch = 0
        self._next_admission = 0
        self._probe_owner: tuple[int, int] | None = None

    def admission(self, server: str) -> tuple[int, int]:
        """Return an ownership token for this admitted call."""
        until = self._cooldown_until
        if until is not None:
            now = self._clock()
            if self._probe_owner is not None:
                raise MCPServerCooldownError(
                    server, max(until - now, self.PROBE_RETRY_SEC)
                )
            if now < until:
                raise MCPServerCooldownError(server, until - now)
        self._next_admission += 1
        token = (self._epoch, self._next_admission)
        if until is not None:
            self._probe_owner = token
        return token

    def record_timeout(self, token: tuple[int, int]) -> None:
        """Only a timeout admitted in the current generation can alter state."""
        if token[0] != self._epoch or (
            self._cooldown_until is not None and self._probe_owner != token
        ):
            return
        self._timeouts_since_success += 1
        if self._probe_owner == token:
            self._probe_owner = None
        if self._timeouts_since_success >= self.TIMEOUT_THRESHOLD:
            self._cooldown_until = self._clock() + self.COOLDOWN_SEC
            self._epoch += 1
            self._probe_owner = None

    def record_success(self, token: tuple[int, int]) -> None:
        """A current response clears state, never another generation's probe."""
        if token[0] != self._epoch or (
            self._cooldown_until is not None and self._probe_owner != token
        ):
            return
        self._timeouts_since_success = 0
        self._cooldown_until = None
        self._probe_owner = None
        self._epoch += 1

    def record_probe_incomplete(self, token: tuple[int, int]) -> None:
        """Release only this call's probe; retain the timeout counter."""
        if self._probe_owner == token:
            self._probe_owner = None


def _stderr_logger_thread(read_fd: int) -> None:
    with open(read_fd, "rb") as f:
        for line in iter(f.readline, b""):
            decoded = decode_console_safe(line).rstrip()
            if decoded:
                logger.debug(f"[MCP stderr] {decoded}")


@contextlib.asynccontextmanager
async def _mcp_stderr_capture() -> AsyncGenerator[TextIO, None]:
    r, w = os.pipe()
    errlog = None
    thread_started = False
    try:
        thread = threading.Thread(target=_stderr_logger_thread, args=(r,), daemon=True)
        thread.start()
        thread_started = True
        errlog = os.fdopen(w, "w")
        yield errlog
    finally:
        if errlog is not None:
            errlog.close()
        elif thread_started:
            os.close(w)
        else:
            os.close(r)
            os.close(w)


@dataclass(frozen=True)
class MCPHttpAuthorizationRuntime:
    provider: MCPAuthorizationProvider
    reference: MCPAuthorizationRef
    required_sink: MCPAuthorizationRequiredSink | None = None


class _MCPContentBlock(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    type: str | None = None
    text: str | None = None


class _MCPResultIn(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    isError: bool = False
    structuredContent: dict[str, Any] | None = None
    content: list[_MCPContentBlock] | None = None

    @field_validator("structuredContent", mode="before")
    @classmethod
    def _normalize_structured(cls, v: Any) -> dict[str, Any] | None:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        dump = getattr(v, "model_dump", None)
        if callable(dump):
            try:
                v = dump()
            except Exception:
                return None
        return v if isinstance(v, dict) else None


def _parse_call_result(server: str, tool: str, result_obj: Any) -> MCPToolResult:
    parsed = _MCPResultIn.model_validate(result_obj)
    parts = []
    for block in parsed.content or []:
        if block.type in {None, "text"} and block.text is not None:
            parts.append(block.text)
        else:
            parts.append(
                f"[Unsupported MCP content omitted: {block.type or 'unknown'}]"
            )
    if parsed.structuredContent is not None:
        # Structured content reaches the model as a `structured: {...}` line in
        # the tool-response text, so it rides inside the same untrusted frame
        # as the text blocks instead of arriving unframed.
        try:
            serialized = json.dumps(
                parsed.structuredContent, default=repr, sort_keys=True
            )
        except (TypeError, ValueError):
            serialized = repr(parsed.structuredContent)
        parts.append(f"structured: {serialized}")
    text = "\n".join(parts) if parts else None
    return MCPToolResult(
        server=server,
        tool=tool,
        ok=not parsed.isError,
        # Remote MCP servers are untrusted: their output can carry
        # prompt-injection payloads, so frame it as data for the model.
        text=(
            frame_untrusted_content(text, f"MCP server {server}")
            if text is not None
            else None
        ),
        structured=parsed.structuredContent,
    )


def create_vibe_mcp_http_client(
    headers: dict[str, str] | None, *, auth: httpx.Auth | None = None
) -> ChartreuxAsyncHTTPClient:
    return ChartreuxAsyncHTTPClient(
        follow_redirects=False,
        headers=headers,
        auth=auth,
        timeout=httpx.Timeout(_MCP_DEFAULT_TIMEOUT, read=_MCP_DEFAULT_SSE_READ_TIMEOUT),
        verify=build_ssl_context(),
    )


async def _list_all_tools(session: ClientSession) -> list[RemoteTool]:
    tools: list[RemoteTool] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(100):
        response = (
            await session.list_tools(cursor=cursor)
            if cursor is not None
            else await session.list_tools()
        )
        tools.extend(RemoteTool.model_validate(tool) for tool in response.tools)
        cursor = getattr(response, "nextCursor", None)
        if not cursor:
            return tools
        if not isinstance(cursor, str) or cursor in seen:
            raise ValueError("Invalid or repeated MCP discovery cursor")
        seen.add(cursor)
    raise ValueError("MCP discovery pagination limit exceeded")


async def list_tools_http(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | None = None,
    startup_timeout_sec: float | None = None,
) -> list[RemoteTool]:
    ClientSession = _mcp_sdk_attribute("ClientSession")
    streamable_http_client = _mcp_sdk_attribute("streamable_http_client")

    timeout = timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    async with (
        asyncio.timeout(startup_timeout_sec or _MCP_DEFAULT_TIMEOUT),
        create_vibe_mcp_http_client(headers, auth=auth) as http_client,
    ):
        async with streamable_http_client(url, http_client=http_client) as (
            read,
            write,
            _,
        ):
            async with ClientSession(
                read, write, read_timeout_seconds=timeout
            ) as session:
                await session.initialize()
                return await _list_all_tools(session)


async def call_tool_http(
    url: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | None = None,
    startup_timeout_sec: float | None = None,
    tool_timeout_sec: float | None = None,
) -> MCPToolResult:
    ClientSession = _mcp_sdk_attribute("ClientSession")
    streamable_http_client = _mcp_sdk_attribute("streamable_http_client")

    init_timeout = (
        timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    )
    call_timeout = timedelta(seconds=tool_timeout_sec) if tool_timeout_sec else None
    async with create_vibe_mcp_http_client(headers, auth=auth) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read,
            write,
            _,
        ):
            async with ClientSession(
                read, write, read_timeout_seconds=init_timeout
            ) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool_name, arguments, read_timeout_seconds=call_timeout
                )
                return _parse_call_result(url, tool_name, result)


def create_mcp_http_proxy_tool_class(
    *,
    url: str,
    remote: RemoteTool,
    alias: str | None = None,
    server_hint: str | None = None,
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | None = None,
    authorization_runtime: MCPHttpAuthorizationRuntime | None = None,
    startup_timeout_sec: float | None = None,
    tool_timeout_sec: float | None = None,
) -> type[BaseTool[_OpenArgs, MCPToolResult, BaseToolConfig, BaseToolState]]:
    from urllib.parse import urlparse

    def _alias_from_url(url: str) -> str:
        p = urlparse(url)
        host = (p.hostname or "mcp").replace(".", "_")
        port = f"_{p.port}" if p.port else ""
        return f"{host}{port}"

    computed_alias = alias or _alias_from_url(url)
    published_name = f"{computed_alias}_{remote.name}"

    class MCPHttpProxyTool(MCPTool):
        description: ClassVar[str] = (
            (f"[{computed_alias}] " if computed_alias else "")
            + _mcp_description(
                remote.description or f"MCP tool '{remote.name}' from {url}",
                f"MCP server {computed_alias}",
            )
            + (f"\nHint: {server_hint}" if server_hint else "")
        )
        _server_name: ClassVar[str] = computed_alias
        _mcp_url: ClassVar[str] = url
        _remote_name: ClassVar[str] = remote.name
        _input_schema: ClassVar[dict[str, Any]] = remote.input_schema
        _headers: ClassVar[dict[str, str]] = dict(headers or {})
        _auth: ClassVar[httpx.Auth | None] = auth
        _authorization_runtime: ClassVar[MCPHttpAuthorizationRuntime | None] = (
            authorization_runtime
        )
        _startup_timeout_sec: ClassVar[float | None] = startup_timeout_sec
        _tool_timeout_sec: ClassVar[float | None] = tool_timeout_sec
        # Session-scoped hung-server cooldown state; the class is created per
        # discovery, so this is in-memory per session and never persisted.
        _breaker: ClassVar[MCPServerCircuitBreaker] = MCPServerCircuitBreaker()

        @classmethod
        def get_name(cls) -> str:
            return published_name

        @classmethod
        def get_parameters(cls) -> dict[str, Any]:
            return _presentation_schema(
                cls._input_schema, f"MCP server {computed_alias}"
            )

        async def run(
            self, args: _OpenArgs, ctx: InvokeContext | None = None
        ) -> AsyncGenerator[ToolStreamEvent | MCPToolResult, None]:
            try:
                payload = args.model_dump(exclude_unset=True)
                if self._authorization_runtime is None:
                    yield await self._call_remote(payload)
                    return
                yield await self._call_authorized(payload)
            except Exception as exc:
                if isinstance(exc, ToolError):
                    raise
                raise ToolError(
                    "MCP call failed: "
                    + frame_untrusted_content(
                        str(exc), f"MCP server {self._server_name} error"
                    )
                ) from exc

        @classmethod
        async def _call_authorized(cls, payload: dict[str, Any]) -> MCPToolResult:
            runtime = cls._authorization_runtime
            if runtime is None:
                return await cls._call_remote(payload)
            authorization = await runtime.provider.resolve(runtime.reference)
            if isinstance(authorization, MCPAuthorizationRequired):
                await _publish_authorization_required(runtime, authorization)
                raise ToolError(
                    f"MCP server '{cls._server_name}' needs re-authentication."
                )
            try:
                return await cls._call_remote(
                    payload, headers=dict(authorization.headers)
                )
            except Exception as exc:
                if not is_authorization_rejection(exc):
                    raise
                replacement = await runtime.provider.reject(
                    runtime.reference,
                    observed_connection_revision=authorization.connection_revision,
                    reason="http_unauthorized",
                )
                if (
                    isinstance(replacement, MCPAuthorizationSnapshot)
                    and replacement.connection_revision
                    != authorization.connection_revision
                ):
                    try:
                        return await cls._call_remote(
                            payload, headers=dict(replacement.headers)
                        )
                    except Exception as retry_exc:
                        if not is_authorization_rejection(retry_exc):
                            raise
                        replacement = await runtime.provider.reject(
                            runtime.reference,
                            observed_connection_revision=(
                                replacement.connection_revision
                            ),
                            reason="http_unauthorized",
                        )
                required = authorization_required_result(
                    replacement,
                    observed_connection_revision=authorization.connection_revision,
                )
                await _publish_authorization_required(runtime, required)
                raise ToolError(
                    f"MCP server '{cls._server_name}' rejected authentication."
                ) from exc

        @classmethod
        async def _call_remote(
            cls, payload: dict[str, Any], *, headers: dict[str, str] | None = None
        ) -> MCPToolResult:
            admission = cls._breaker.admission(cls._server_name)
            try:
                result = await call_tool_http(
                    cls._mcp_url,
                    cls._remote_name,
                    payload,
                    headers=headers if headers is not None else cls._headers,
                    auth=cls._auth,
                    startup_timeout_sec=cls._startup_timeout_sec,
                    tool_timeout_sec=cls._tool_timeout_sec,
                )
            except BaseException as exc:
                if _is_call_timeout(exc):
                    cls._breaker.record_timeout(admission)
                else:
                    cls._breaker.record_probe_incomplete(admission)
                raise
            cls._breaker.record_success(admission)
            return result

        @classmethod
        def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
            if not isinstance(event.result, MCPToolResult):
                return ToolResultDisplay(
                    success=False,
                    message=event.error or event.skip_reason or "No result",
                )

            return ToolResultDisplay(
                success=event.result.ok, verb="Ran", message=event.result.tool
            )

        @classmethod
        def get_status_text(cls) -> str:
            return f"Calling MCP tool {remote.name}"

    MCPHttpProxyTool.__name__ = f"MCP_{computed_alias}__{remote.name}"
    return MCPHttpProxyTool


async def _publish_authorization_required(
    runtime: MCPHttpAuthorizationRuntime, required: MCPAuthorizationRequired
) -> None:
    if runtime.required_sink is None:
        return
    result = runtime.required_sink(runtime.reference.server_name, required)
    if inspect.isawaitable(result):
        await result


def authorization_required_result(
    result: MCPAuthorizationSnapshot | MCPAuthorizationRequired,
    *,
    observed_connection_revision: str,
) -> MCPAuthorizationRequired:
    if isinstance(result, MCPAuthorizationRequired):
        return result
    return MCPAuthorizationRequired(
        reason="rejected",
        descriptor_revision=result.descriptor_revision,
        observed_connection_revision=observed_connection_revision,
    )


def is_authorization_rejection(exc: BaseException) -> bool:
    OAuthFlowError = _mcp_sdk_attribute("OAuthFlowError")

    if isinstance(exc, BaseExceptionGroup):
        return any(is_authorization_rejection(child) for child in exc.exceptions)
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == httpx.codes.UNAUTHORIZED
    return isinstance(exc, OAuthFlowError)


def _stdio_environment(env: dict[str, str] | None) -> dict[str, str]:
    """Resolve the explicit environment for an MCP stdio child process.

    The MCP SDK substitutes a small safe default for an unset env rather than
    inheriting ``os.environ``; resolving it here pins that behavior and applies
    chartreux's credential scrubbing to the inherited portion. Explicit
    per-server ``env`` entries are merged last, so they act as passthrough for
    that server.
    """
    get_default_environment = _mcp_sdk_attribute("get_default_environment")
    return {**scrub_child_env(get_default_environment()), **(env or {})}


def build_stdio_params(
    command: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None
) -> StdioServerParameters:
    StdioServerParameters = _mcp_sdk_attribute("StdioServerParameters")

    return StdioServerParameters(
        command=command[0], args=command[1:], env=_stdio_environment(env), cwd=cwd
    )


async def enter_stdio_session(
    stack: contextlib.AsyncExitStack,
    params: StdioServerParameters,
    *,
    init_timeout: timedelta | None,
) -> ClientSession:
    """Enter the stderr-capture, stdio_client, and ClientSession contexts on *stack*.

    The caller owns ``stack`` and decides when to close it. Returns an initialized
    session. The one-shot helpers close the stack immediately; the connection pool
    keeps it open for the session lifetime.
    """
    ClientSession = _mcp_sdk_attribute("ClientSession")
    stdio_client = _mcp_sdk_attribute("stdio_client")

    errlog = await stack.enter_async_context(_mcp_stderr_capture())
    read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
    session = await stack.enter_async_context(
        ClientSession(read, write, read_timeout_seconds=init_timeout)
    )
    await session.initialize()
    return session


async def list_tools_stdio(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    startup_timeout_sec: float | None = None,
) -> list[RemoteTool]:
    params = build_stdio_params(command, env=env, cwd=cwd)
    timeout = timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    async with (
        asyncio.timeout(startup_timeout_sec or _MCP_DEFAULT_TIMEOUT),
        contextlib.AsyncExitStack() as stack,
    ):
        session = await enter_stdio_session(stack, params, init_timeout=timeout)
        return await _list_all_tools(session)


async def call_tool_stdio(
    command: list[str],
    tool_name: str,
    arguments: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    startup_timeout_sec: float | None = None,
    tool_timeout_sec: float | None = None,
) -> MCPToolResult:
    params = build_stdio_params(command, env=env, cwd=cwd)
    init_timeout = (
        timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    )
    call_timeout = timedelta(seconds=tool_timeout_sec) if tool_timeout_sec else None
    async with contextlib.AsyncExitStack() as stack:
        session = await enter_stdio_session(stack, params, init_timeout=init_timeout)
        result = await session.call_tool(
            tool_name, arguments, read_timeout_seconds=call_timeout
        )
        return _parse_call_result("stdio:" + " ".join(command), tool_name, result)


def create_mcp_stdio_proxy_tool_class(
    *,
    command: list[str],
    remote: RemoteTool,
    alias: str | None = None,
    server_hint: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    startup_timeout_sec: float | None = None,
    tool_timeout_sec: float | None = None,
) -> type[BaseTool[_OpenArgs, MCPToolResult, BaseToolConfig, BaseToolState]]:
    def _alias_from_command(cmd: list[str]) -> str:
        prog = Path(cmd[0]).name.replace(".", "_") if cmd else "mcp"
        digest = hashlib.blake2s(
            "\0".join(cmd).encode("utf-8"), digest_size=4
        ).hexdigest()
        return f"{prog}_{digest}"

    computed_alias = alias or _alias_from_command(command)
    published_name = f"{computed_alias}_{remote.name}"

    class MCPStdioProxyTool(MCPTool):
        description: ClassVar[str] = (
            (f"[{computed_alias}] " if computed_alias else "")
            + _mcp_description(
                remote.description
                or f"MCP tool '{remote.name}' from stdio command: {' '.join(command)}",
                f"MCP server {computed_alias}",
            )
            + (f"\nHint: {server_hint}" if server_hint else "")
        )
        _server_name: ClassVar[str] = computed_alias
        _stdio_command: ClassVar[list[str]] = command
        _remote_name: ClassVar[str] = remote.name
        _input_schema: ClassVar[dict[str, Any]] = remote.input_schema
        _env: ClassVar[dict[str, str] | None] = env
        _cwd: ClassVar[str | None] = cwd
        _startup_timeout_sec: ClassVar[float | None] = startup_timeout_sec
        _tool_timeout_sec: ClassVar[float | None] = tool_timeout_sec
        # Session-scoped hung-server cooldown for the no-pool one-shot fallback;
        # pooled calls carry their own breaker on the pool (keyed per server).
        _breaker: ClassVar[MCPServerCircuitBreaker] = MCPServerCircuitBreaker()

        @classmethod
        def get_name(cls) -> str:
            return published_name

        @classmethod
        def get_parameters(cls) -> dict[str, Any]:
            return _presentation_schema(
                cls._input_schema, f"MCP server {computed_alias}"
            )

        async def run(
            self, args: _OpenArgs, ctx: InvokeContext | None = None
        ) -> AsyncGenerator[ToolStreamEvent | MCPToolResult, None]:
            try:
                payload = args.model_dump(exclude_unset=True)
                pool = ctx.mcp_pool if ctx else None
                if pool is not None:
                    yield await pool.call_tool(
                        server_name=self._server_name,
                        command=self._stdio_command,
                        tool_name=self._remote_name,
                        arguments=payload,
                        env=self._env,
                        cwd=self._cwd,
                        startup_timeout_sec=self._startup_timeout_sec,
                        tool_timeout_sec=self._tool_timeout_sec,
                    )
                    return
                admission = self._breaker.admission(self._server_name)
                try:
                    result = await call_tool_stdio(
                        self._stdio_command,
                        self._remote_name,
                        payload,
                        env=self._env,
                        cwd=self._cwd,
                        startup_timeout_sec=self._startup_timeout_sec,
                        tool_timeout_sec=self._tool_timeout_sec,
                    )
                except BaseException as exc:
                    if _is_call_timeout(exc):
                        self._breaker.record_timeout(admission)
                    else:
                        self._breaker.record_probe_incomplete(admission)
                    raise
                self._breaker.record_success(admission)
                yield result
            except Exception as exc:
                raise ToolError(
                    "MCP stdio call failed: "
                    + frame_untrusted_content(
                        str(exc), f"MCP server {self._server_name} error"
                    )
                ) from exc

        @classmethod
        def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
            if not isinstance(event.result, MCPToolResult):
                return ToolResultDisplay(
                    success=False,
                    message=event.error or event.skip_reason or "No result",
                )

            return ToolResultDisplay(
                success=event.result.ok, verb="Ran", message=event.result.tool
            )

        @classmethod
        def get_status_text(cls) -> str:
            return f"Calling MCP tool {remote.name}"

    MCPStdioProxyTool.__name__ = f"MCP_STDIO_{computed_alias}__{remote.name}"
    return MCPStdioProxyTool
