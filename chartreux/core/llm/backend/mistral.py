from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Future
from contextlib import suppress
import json
import logging
import types
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

import httpx
from mistralai.client import Mistral
from mistralai.client._hooks.types import AfterErrorHook
from mistralai.client.errors import SDKError
from mistralai.client.models import (
    AssistantMessage,
    AssistantMessageContent,
    ChatCompletionRequestMessage,
    ChatCompletionStreamRequestToolChoice,
    ContentChunk,
    FileChunk,
    Function,
    FunctionCall as MistralFunctionCall,
    FunctionName,
    ImageURL,
    ImageURLChunk,
    SystemMessage,
    TextChunk,
    ThinkChunk,
    Tool,
    ToolCall as MistralToolCall,
    ToolChoice,
    ToolChoiceEnum,
    ToolMessage,
    UserMessage,
)
from mistralai.client.types.basemodel import UNSET
from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

from chartreux.core.config._defaults import (
    DEFAULT_API_CONNECT_TIMEOUT,
    DEFAULT_API_POOL_TIMEOUT,
    DEFAULT_API_RETRY_MAX_ELAPSED_TIME,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_WRITE_TIMEOUT,
)
from chartreux.core.llm.backend._image import to_data_uri as _to_data_uri
from chartreux.core.llm.backend._tool_images import has_tool_images, project_tool_images
from chartreux.core.llm.backend.base import (
    MODEL_HTTP_KEEPALIVE_EXPIRY_SECONDS,
    get_thinking_wire_value,
)
from chartreux.core.llm.exceptions import BackendError, BackendErrorBuilder
from chartreux.core.llm.thinking_levels import (
    MISTRAL_THINKING_LEVELS,
    get_thinking_levels,
)
from chartreux.core.llm_models import (
    AvailableTool,
    Content,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StopInfo,
    StrToolChoice,
    ToolCall,
)
from chartreux.core.utils import (
    RequestRetryBudget,
    RetryObserver,
    RetryReason,
    async_retry,
)
from chartreux.utils.api_keys import ApiKeyOrigin, resolve_api_key_with_origin
from chartreux.utils.http import ChartreuxAsyncHTTPClient, get_server_url_from_api_base

if TYPE_CHECKING:
    from chartreux.core.config import ModelConfig, ProviderConfig

logger = logging.getLogger("vibe")

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_ERRORS = (httpx.NetworkError, httpx.TimeoutException)
_MAX_CONNECTIONS = 20
_SDK_REASONING_EFFORT_HEADER = "x-chartreux-sdk-reasoning-effort"


def _log_delivery_failure(future: Future[None]) -> None:
    with suppress(CancelledError):
        if (error := future.exception()) is not None:
            logger.warning("Could not report retry: %s", error)


class _RetryNoticeHook(AfterErrorHook):
    def __init__(self, report: Callable[[Exception], None]) -> None:
        self._report = report

    def after_error(
        self, hook_ctx: Any, response: httpx.Response | None, error: Exception | None
    ) -> tuple[httpx.Response | None, Exception | None]:
        if error is not None:
            self._report(error)
        return response, error


def _register_retry_hook(client: Mistral, hook: AfterErrorHook) -> None:
    registry = client.sdk_configuration.__dict__["_hooks"]
    registry.register_after_error_hook(hook)


def _cached_tokens(usage: object | None) -> int:
    # Mistral reports cache hits under usage.prompt_tokens_details.cached_tokens.
    # The SDK keeps this nested block as an untyped extra, so both its shape
    # (dict vs object) and its value type are unvalidated; coerce defensively
    # and fall back to 0 on anything odd.
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    value = (
        details.get("cached_tokens")
        if isinstance(details, dict)
        else getattr(details, "cached_tokens", None)
    )
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class ParsedContent(NamedTuple):
    content: Content
    reasoning_content: Content | None


class MistralMapper:
    def prepare_message(
        self, msg: LLMMessage, *, include_reasoning_content: bool = True
    ) -> ChatCompletionRequestMessage:
        match msg.role:
            case Role.system:
                return SystemMessage(role="system", content=msg.content or "")
            case Role.user:
                if msg.images:
                    user_parts: list[ContentChunk] = []
                    if msg.content:
                        user_parts.append(TextChunk(type="text", text=msg.content))
                    user_parts.extend(
                        ImageURLChunk(
                            type="image_url", image_url=ImageURL(url=_to_data_uri(att))
                        )
                        for att in msg.images
                    )
                    return UserMessage(role="user", content=user_parts)
                return UserMessage(role="user", content=msg.content)
            case Role.assistant:
                content: AssistantMessageContent
                if include_reasoning_content and msg.reasoning_content:
                    chunks: list[ContentChunk] = [
                        ThinkChunk(
                            type="thinking",
                            thinking=[
                                TextChunk(type="text", text=msg.reasoning_content)
                            ],
                        )
                    ]
                    if msg.content:
                        chunks.append(TextChunk(type="text", text=msg.content))
                    content = chunks
                else:
                    content = msg.content or ""

                return AssistantMessage(
                    role="assistant",
                    content=content,
                    tool_calls=[
                        MistralToolCall(
                            function=MistralFunctionCall(
                                name=tc.function.name or "",
                                arguments=tc.function.arguments or "",
                            ),
                            id=tc.id,
                            type=tc.type,
                            index=tc.index,
                        )
                        for tc in msg.tool_calls or []
                    ],
                )
            case Role.tool:
                return ToolMessage(
                    role="tool",
                    content=msg.content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )

    def prepare_tool(self, tool: AvailableTool) -> Tool:
        return Tool(
            type="function",
            function=Function(
                name=tool.function.name,
                description=tool.function.description,
                parameters=tool.function.parameters,
            ),
        )

    def prepare_tool_choice(
        self, tool_choice: StrToolChoice | AvailableTool
    ) -> ChatCompletionStreamRequestToolChoice:
        if isinstance(tool_choice, str):
            return cast(ToolChoiceEnum, tool_choice)

        return ToolChoice(
            type="function", function=FunctionName(name=tool_choice.function.name)
        )

    def _extract_thinking_text(self, chunk: ThinkChunk) -> str:
        thinking_content = getattr(chunk, "thinking", None)
        if not thinking_content:
            return ""
        parts = []
        for inner in thinking_content:
            if hasattr(inner, "type") and inner.type == "text":
                parts.append(getattr(inner, "text", ""))
            elif isinstance(inner, str):
                parts.append(inner)
        return "".join(parts)

    def parse_content(self, content: AssistantMessageContent) -> ParsedContent:
        if isinstance(content, str):
            return ParsedContent(content=content, reasoning_content=None)

        concat_content = ""
        concat_reasoning = ""
        for chunk in content:
            if isinstance(chunk, FileChunk):
                continue
            if isinstance(chunk, TextChunk):
                concat_content += chunk.text
            elif isinstance(chunk, ThinkChunk):
                concat_reasoning += self._extract_thinking_text(chunk)
        return ParsedContent(
            content=concat_content,
            reasoning_content=concat_reasoning if concat_reasoning else None,
        )

    def parse_tool_calls(self, tool_calls: list[MistralToolCall]) -> list[ToolCall]:
        return [
            ToolCall(
                id=tool_call.id,
                function=FunctionCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments
                    if isinstance(tool_call.function.arguments, str)
                    else json.dumps(tool_call.function.arguments, ensure_ascii=False),
                ),
                index=tool_call.index,
            )
            for tool_call in tool_calls
        ]


def _merge_headers(
    defaults: dict[str, str], overrides: dict[str, str] | None
) -> dict[str, str]:
    """Merge headers case-insensitively, with per-call values taking precedence."""
    merged = dict(defaults)
    names = {name.casefold(): name for name in merged}
    for name, value in (overrides or {}).items():
        previous = names.get(name.casefold())
        if previous is not None:
            merged.pop(previous, None)
        merged[name] = value
        names[name.casefold()] = name
    return merged


def _sdk_headers(
    defaults: dict[str, str],
    overrides: dict[str, str] | None,
    reasoning_effort: str | None,
) -> dict[str, str]:
    headers = _merge_headers(defaults, overrides)
    if reasoning_effort == "max":
        # mistralai 2.6.0 accepts unknown enum values but drops them while
        # serializing. Preserve the value for the HTTP client hook below.
        headers[_SDK_REASONING_EFFORT_HEADER] = reasoning_effort
    return headers


class MistralBackend:
    THINKING_LEVELS: ClassVar[dict[str, str | None]] = MISTRAL_THINKING_LEVELS

    @staticmethod
    def _thinking_levels(model_name: str) -> Mapping[str, str | None]:
        levels = get_thinking_levels("mistral", None, model_name)
        assert levels is not None
        return levels

    def __init__(
        self,
        provider: ProviderConfig,
        timeout: float = DEFAULT_API_TIMEOUT,
        retry_max_elapsed_time: float = DEFAULT_API_RETRY_MAX_ELAPSED_TIME,
        retry_budget: RequestRetryBudget | None = None,
        connect_timeout: float = DEFAULT_API_CONNECT_TIMEOUT,
        write_timeout: float = DEFAULT_API_WRITE_TIMEOUT,
        pool_timeout: float = DEFAULT_API_POOL_TIMEOUT,
        enable_system_trust_store: bool = False,
        on_retry: RetryObserver | None = None,
    ) -> None:
        self._client: Mistral | None = None
        self._http_client: ChartreuxAsyncHTTPClient | None = None
        self._provider = provider
        self._enable_system_trust_store = enable_system_trust_store
        self._transport_timeouts = {
            "connect": connect_timeout,
            "write": write_timeout,
            "pool": pool_timeout,
        }
        self._on_retry = on_retry
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mapper = MistralMapper()
        resolved_api_key = resolve_api_key_with_origin(self._provider.api_key_env_var)
        self._api_key = resolved_api_key[0] if resolved_api_key else None
        self._api_key_origin: ApiKeyOrigin | None = (
            resolved_api_key[1] if resolved_api_key else None
        )

        reasoning_field = getattr(provider, "reasoning_field_name", "reasoning_content")
        if reasoning_field != "reasoning_content":
            raise ValueError(
                f"Mistral backend does not support custom reasoning_field_name "
                f"(got '{reasoning_field}'). Mistral uses ThinkChunk for reasoning."
            )

        # Mistral SDK takes server URL without api version as input
        server_url = get_server_url_from_api_base(self._provider.api_base)
        if not server_url:
            raise ValueError(
                f"Invalid API base URL: {self._provider.api_base}. "
                "Expected format: <server_url>/v<api_version>"
            )
        self._server_url = server_url
        self._timeout = timeout
        self._retry_max_elapsed_time = retry_max_elapsed_time
        self._retry_budget = retry_budget
        # Retained for diagnostics/backward-compatible inspection only. It is
        # deliberately not passed to the SDK.
        self._retry_config = self._build_retry_config()

    def _attach_api_key_origin(self, error: BackendError) -> BackendError:
        error.api_key_origin = self._api_key_origin
        error.args = (error._fmt(),)
        return error

    def _build_retry_config(self) -> RetryConfig:
        return RetryConfig(
            strategy="backoff",
            backoff=BackoffStrategy(
                initial_interval=500,
                max_interval=30000,
                exponent=1.5,
                max_elapsed_time=int(self._retry_max_elapsed_time * 1000),
            ),
            retry_connection_errors=True,
        )

    def _with_retry(self, call: Callable[..., Any]) -> Callable[..., Any]:
        return async_retry(
            tries=None,
            max_elapsed_time=(
                self._retry_max_elapsed_time if self._retry_budget is None else None
            ),
            budget=self._retry_budget,
            on_retry=self._on_retry,
        )(call)

    async def _bound_transport_timeouts(self, request: httpx.Request) -> None:
        """Cap connect, write and pool waits without shortening the read budget.

        The SDK sends a single scalar timeout per request, which httpx expands
        onto all four axes. A budget sized for a long model turn then also
        governs opening a socket and acquiring a pooled connection, so an
        unreachable host stalls for that whole budget before erroring. Each cap
        is a ceiling rather than a value, so a caller asking for a shorter
        overall timeout still wins on every axis.
        """
        timeout = dict(request.extensions.get("timeout", {}))
        for axis, limit in self._transport_timeouts.items():
            current = timeout.get(axis)
            timeout[axis] = limit if current is None else min(current, limit)
        request.extensions = {**request.extensions, "timeout": timeout}
        if reasoning_effort := request.headers.pop(_SDK_REASONING_EFFORT_HEADER, None):
            body = json.loads(request.content)
            body["reasoning_effort"] = reasoning_effort
            content = json.dumps(body, separators=(",", ":")).encode()
            # The SDK has already built the request, so update both its stream
            # and cached content before httpx sends it.
            request.stream = httpx.ByteStream(content)
            request._content = content
            request.headers["content-length"] = str(len(content))

    async def _on_response(self, response: httpx.Response) -> None:
        """Release the connection behind a retryable response, then report it.

        Streaming requests are issued with the body unread, and the SDK holds a
        failed response alive across the retry backoff without closing it, so its
        pooled connection stays checked out for the lifetime of the client.
        Reading the body here returns the connection to the pool and keeps the
        error payload available to the terminal error path.

        A truncated or badly encoded body is not retryable to the SDK, so letting
        that surface would turn a rate limit into a dead turn. Close the response
        instead and let the retry proceed on the status code alone.
        """
        if response.status_code not in _RETRYABLE_STATUS_CODES:
            return
        try:
            await response.aread()
        except Exception:
            logger.debug(
                "Could not read the body of a %s response", response.status_code
            )
            await response.aclose()
        await self._notice_retry(RetryReason.from_http_status(response.status_code))

    def _report_error(self, error: Exception) -> None:
        # On a worker thread inside basesdk's except block: raising would
        # replace the error being reported on.
        if self._loop is None or not isinstance(error, _RETRYABLE_ERRORS):
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._notice_retry(RetryReason.from_error(error)), self._loop
            )
        except RuntimeError:  # loop already closed
            return
        future.add_done_callback(_log_delivery_failure)

    async def _notice_retry(self, reason: RetryReason) -> None:
        logger.warning(
            "Retrying request category=%s detail=%s", reason.category, reason.detail
        )
        if self._on_retry is not None:
            await self._on_retry(reason)

    async def __aenter__(self) -> MistralBackend:
        self._client = self._create_mistral_client()
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        client = self._client
        http_client = self._http_client
        self._client = None
        self._http_client = None
        try:
            if client is not None:
                await client.__aexit__(
                    exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb
                )
        finally:
            if http_client is not None:
                await http_client.aclose()

    async def aclose(self) -> None:
        await self.__aexit__(None, None, None)

    def _create_mistral_client(self) -> Mistral:
        self._loop = asyncio.get_running_loop()
        self._http_client = ChartreuxAsyncHTTPClient(
            enable_system_trust_store=self._enable_system_trust_store,
            follow_redirects=True,
            event_hooks={"request": [self._bound_transport_timeouts]},
            limits=httpx.Limits(
                max_connections=_MAX_CONNECTIONS,
                keepalive_expiry=MODEL_HTTP_KEEPALIVE_EXPIRY_SECONDS,
            ),
        )
        client = Mistral(
            api_key=self._api_key,
            server_url=self._server_url,
            timeout_ms=int(self._timeout * 1000),
            # mistralai 2.6.0 accepts ``None`` as OptionalNullable and stores it
            # on SDKConfiguration; this disables the SDK's opaque retry layer.
            retry_config=None,
            async_client=self._http_client,
        )
        # Disable the Mistral SDK's built-in telemetry: Chartreux sends no product
        # telemetry. mistralai 2.6.0 consults this instance setting before its
        # telemetry environment variable; it is not a constructor keyword.
        client.sdk_configuration.__dict__["telemetry"] = False
        return client

    def _get_client(self) -> Mistral:
        if self._client is None:
            self._client = self._create_mistral_client()
        return self._client

    async def complete(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
        metadata: dict[str, str] | None = None,
    ) -> LLMChunk:
        try:
            reasoning_effort = cast(
                Any,
                get_thinking_wire_value(
                    self._thinking_levels(model.name), model.thinking
                ),
            )
            projected_messages = (
                project_tool_images(messages) if has_tool_images(messages) else messages
            )
            response = await self._with_retry(self._get_client().chat.complete_async)(
                model=model.name,
                messages=[
                    self._mapper.prepare_message(
                        msg, include_reasoning_content=model.thinking != "off"
                    )
                    for msg in projected_messages
                ],
                temperature=temperature,
                tools=[self._mapper.prepare_tool(tool) for tool in tools]
                if tools
                else None,
                max_tokens=max_tokens,
                tool_choice=self._mapper.prepare_tool_choice(tool_choice)
                if tool_choice
                else None,
                http_headers=_sdk_headers(
                    self._provider.extra_headers, extra_headers, reasoning_effort
                ),
                metadata=metadata,
                stream=False,
                reasoning_effort=reasoning_effort
                if reasoning_effort is not None
                else UNSET,
            )

            choice = response.choices[0]
            message = choice.message
            parsed = (
                self._mapper.parse_content(message.content)
                if message and message.content
                else ParsedContent(content="", reasoning_content=None)
            )
            return LLMChunk(
                message=LLMMessage(
                    role=Role.assistant,
                    content=parsed.content,
                    reasoning_content=parsed.reasoning_content,
                    tool_calls=self._mapper.parse_tool_calls(message.tool_calls)
                    if message and message.tool_calls
                    else None,
                ),
                usage=LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    cached_tokens=_cached_tokens(response.usage),
                ),
                stop=(
                    StopInfo(reason=str(choice.finish_reason))
                    if choice.finish_reason is not None
                    else None
                ),
            )

        except SDKError as e:
            backend_error = self._attach_api_key_origin(
                BackendErrorBuilder.build_http_error(
                    provider=self._provider.name,
                    endpoint=self._server_url,
                    error=e,
                    response=e.raw_response,
                    model=model.name,
                    messages=messages,
                    temperature=temperature,
                    has_tools=bool(tools),
                    tool_choice=tool_choice,
                )
            )
            if backend_error.status in {
                httpx.codes.UNAUTHORIZED,
                httpx.codes.FORBIDDEN,
            }:
                raise backend_error from None
            raise backend_error from e
        except (httpx.RequestError, httpx.StreamError) as e:
            raise self._attach_api_key_origin(
                BackendErrorBuilder.build_request_error(
                    provider=self._provider.name,
                    endpoint=self._server_url,
                    error=e,
                    model=model.name,
                    messages=messages,
                    temperature=temperature,
                    has_tools=bool(tools),
                    tool_choice=tool_choice,
                )
            ) from e

    async def complete_streaming(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
        metadata: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        try:
            reasoning_effort = cast(
                Any,
                get_thinking_wire_value(
                    self._thinking_levels(model.name), model.thinking
                ),
            )
            projected_messages = (
                project_tool_images(messages) if has_tool_images(messages) else messages
            )

            stream = await self._with_retry(self._get_client().chat.stream_async)(
                model=model.name,
                messages=[
                    self._mapper.prepare_message(
                        msg, include_reasoning_content=model.thinking != "off"
                    )
                    for msg in projected_messages
                ],
                temperature=temperature,
                tools=[self._mapper.prepare_tool(tool) for tool in tools]
                if tools
                else None,
                max_tokens=max_tokens,
                tool_choice=self._mapper.prepare_tool_choice(tool_choice)
                if tool_choice
                else None,
                http_headers=_sdk_headers(
                    self._provider.extra_headers, extra_headers, reasoning_effort
                ),
                metadata=metadata,
                reasoning_effort=reasoning_effort
                if reasoning_effort is not None
                else UNSET,
            )
            correlation_id = stream.response.headers.get("mistral-correlation-id")
            # Close the underlying httpx response on every exit path (normal
            # completion, early termination of the outer generator, or an error
            # mid-stream). Without this the connection stays checked out of the
            # pool and a long-lived session eventually hits PoolTimeout.
            async with stream:
                async for chunk in stream:
                    # Some models terminate the stream with a usage-only chunk that
                    # carries no choices.
                    choice = chunk.data.choices[0] if chunk.data.choices else None
                    delta = choice.delta if choice else None
                    parsed = (
                        self._mapper.parse_content(delta.content)
                        if delta and delta.content
                        else ParsedContent(content="", reasoning_content=None)
                    )
                    yield LLMChunk(
                        message=LLMMessage(
                            role=Role.assistant,
                            content=parsed.content,
                            reasoning_content=parsed.reasoning_content,
                            tool_calls=self._mapper.parse_tool_calls(delta.tool_calls)
                            if delta and delta.tool_calls
                            else None,
                        ),
                        usage=LLMUsage(
                            prompt_tokens=chunk.data.usage.prompt_tokens or 0
                            if chunk.data.usage
                            else 0,
                            completion_tokens=chunk.data.usage.completion_tokens or 0
                            if chunk.data.usage
                            else 0,
                            cached_tokens=_cached_tokens(chunk.data.usage),
                        ),
                        correlation_id=correlation_id,
                        stop=(
                            StopInfo(reason=str(choice.finish_reason))
                            if choice and choice.finish_reason is not None
                            else None
                        ),
                    )

        except SDKError as e:
            backend_error = self._attach_api_key_origin(
                BackendErrorBuilder.build_http_error(
                    provider=self._provider.name,
                    endpoint=self._server_url,
                    error=e,
                    response=e.raw_response,
                    model=model.name,
                    messages=messages,
                    temperature=temperature,
                    has_tools=bool(tools),
                    tool_choice=tool_choice,
                )
            )
            if backend_error.status in {
                httpx.codes.UNAUTHORIZED,
                httpx.codes.FORBIDDEN,
            }:
                raise backend_error from None
            raise backend_error from e
        except (httpx.RequestError, httpx.StreamError) as e:
            raise self._attach_api_key_origin(
                BackendErrorBuilder.build_request_error(
                    provider=self._provider.name,
                    endpoint=self._server_url,
                    error=e,
                    model=model.name,
                    messages=messages,
                    temperature=temperature,
                    has_tools=bool(tools),
                    tool_choice=tool_choice,
                )
            ) from e
