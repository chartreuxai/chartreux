from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing
import functools
import json
import types
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from chartreux.core.config._defaults import (
    DEFAULT_API_CONNECT_TIMEOUT,
    DEFAULT_API_POOL_TIMEOUT,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_WRITE_TIMEOUT,
)
from chartreux.core.llm.backend._image import to_data_uri as _to_data_uri
from chartreux.core.llm.backend._tool_images import has_tool_images, project_tool_images
from chartreux.core.llm.backend.anthropic import AnthropicAdapter
from chartreux.core.llm.backend.base import (
    MODEL_HTTP_KEEPALIVE_EXPIRY_SECONDS,
    APIAdapter,
    ParsedStreamChunk,
    PreparedRequest,
    build_chat_payload,
    finalize_chat_request,
)
from chartreux.core.llm.backend.openai_responses import (
    OpenAIResponsesAdapter,
    OpenAIResponsesStreamError,
)
from chartreux.core.llm.exceptions import BackendError, BackendErrorBuilder, ModelCall
from chartreux.core.llm.thinking_levels import (
    OPENAI_THINKING_LEVELS,
    get_thinking_levels,
)
from chartreux.core.llm_models import (
    AvailableTool,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StopInfo,
    StrToolChoice,
)
from chartreux.core.utils import (
    AdaptivePacer,
    RequestRetryBudget,
    RetryCategory,
    RetryObserver,
    RetryReason,
    async_generator_retry,
    async_retry,
)
from chartreux.core.utils.sse import iter_sse_lines
from chartreux.utils.api_keys import ApiKeyOrigin, resolve_api_key_with_origin
from chartreux.utils.http import ChartreuxAsyncHTTPClient

if TYPE_CHECKING:
    from chartreux.core.config import ModelConfig, ProviderConfig


class OpenAIAdapter(APIAdapter):
    endpoint: ClassVar[str] = "/chat/completions"
    THINKING_LEVELS: ClassVar[dict[str, str | None]] = OPENAI_THINKING_LEVELS

    def _reasoning_to_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        if field_name != "reasoning_content" and "reasoning_content" in msg_dict:
            msg_dict[field_name] = msg_dict.pop("reasoning_content")
        return msg_dict

    def _reasoning_from_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        if field_name != "reasoning_content" and field_name in msg_dict:
            msg_dict["reasoning_content"] = msg_dict.pop(field_name)
        return msg_dict

    def _user_with_images_to_parts(
        self, msg_dict: dict[str, Any], source: LLMMessage
    ) -> dict[str, Any]:
        if source.role != Role.user or not source.images:
            return msg_dict
        parts: list[dict[str, Any]] = []
        text = msg_dict.get("content")
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
        parts.extend(
            {"type": "image_url", "image_url": {"url": _to_data_uri(att)}}
            for att in source.images
        )
        msg_dict["content"] = parts
        return msg_dict

    def _convert_messages(
        self, messages: Sequence[LLMMessage], provider: ProviderConfig
    ) -> list[dict[str, Any]]:
        field_name = provider.reasoning_field_name
        projected_messages = (
            project_tool_images(messages) if has_tool_images(messages) else messages
        )
        return [
            self._user_with_images_to_parts(
                self._reasoning_to_api(
                    msg.model_dump(
                        exclude_none=True,
                        exclude={
                            "message_id": True,
                            "reasoning_message_id": True,
                            "reasoning_payloads": True,
                            "injected": True,
                            "images": True,
                            "tool_result": True,
                            "user_display_content": True,
                            "input_text": True,
                            "resources": True,
                            "manual_shell": True,
                            "context_boundary": True,
                            "deployment_identity": True,
                            "tool_calls": {"__all__": {"presentation"}},
                        },
                    ),
                    field_name,
                ),
                msg,
            )
            for msg in projected_messages
        ]

    def prepare_request(
        self,
        *,
        model_name: str,
        messages: Sequence[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        enable_streaming: bool,
        provider: ProviderConfig,
        api_key: str | None = None,
        thinking: str = "off",
    ) -> PreparedRequest:
        converted_messages = self._convert_messages(messages, provider)

        thinking_levels = get_thinking_levels(
            str(getattr(provider, "backend", "generic")),
            getattr(provider, "api_style", "openai"),
            model_name,
        )
        assert thinking_levels is not None
        payload = build_chat_payload(
            model_name=model_name,
            messages=converted_messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            thinking=thinking,
            thinking_levels=thinking_levels,
        )
        stream_options: dict[str, Any] = {"include_usage": True}
        if provider.name == "mistral":
            stream_options["stream_tool_calls"] = True

        return finalize_chat_request(
            payload=payload,
            enable_streaming=enable_streaming,
            stream_options=stream_options,
            api_key=api_key,
            endpoint=self.endpoint,
        )

    def _parse_message(
        self, data: dict[str, Any], field_name: str
    ) -> LLMMessage | None:
        if data.get("choices"):
            choice = data["choices"][0]
            if "message" in choice:
                msg_dict = self._reasoning_from_api(choice["message"], field_name)
                return LLMMessage.model_validate(msg_dict)
            if "delta" in choice:
                msg_dict = self._reasoning_from_api(choice["delta"], field_name)
                if msg_dict.get("role") is None:
                    msg_dict["role"] = Role.assistant
                return LLMMessage.model_validate(msg_dict)
            raise ValueError("Invalid response data: missing message or delta")

        if "message" in data:
            msg_dict = self._reasoning_from_api(data["message"], field_name)
            return LLMMessage.model_validate(msg_dict)
        if "delta" in data:
            msg_dict = self._reasoning_from_api(data["delta"], field_name)
            if msg_dict.get("role") is None:
                msg_dict["role"] = Role.assistant
            return LLMMessage.model_validate(msg_dict)

        return None

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk:
        message = self._parse_message(data, provider.reasoning_field_name)
        if message is None:
            message = LLMMessage(role=Role.assistant, content="")

        usage_data = data.get("usage") or {}
        prompt_details = usage_data.get("prompt_tokens_details") or {}
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            cached_tokens=prompt_details.get("cached_tokens", 0),
        )
        choices = data.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        stop = (
            StopInfo(reason=str(finish_reason)) if finish_reason is not None else None
        )

        return LLMChunk(message=message, usage=usage, stop=stop)


_ADAPTERS: dict[str, Callable[[ModelCall | None], APIAdapter]] = {
    "openai": lambda _: OpenAIAdapter(),
    "anthropic": AnthropicAdapter,
    "openai-responses": OpenAIResponsesAdapter,
}

_ADAPTER_ENDPOINTS = {
    "openai": OpenAIAdapter.endpoint,
    "anthropic": AnthropicAdapter.endpoint,
    "openai-responses": OpenAIResponsesAdapter.endpoint,
}


def validate_api_style(api_style: str) -> None:
    if api_style not in _ADAPTERS:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unsupported API style '{api_style}'; supported styles: {supported}"
        )


def _get_adapter(api_style: str, model_call: ModelCall | None = None) -> APIAdapter:
    """Build the adapter for the given API style.

    Adapters are built per request: several of them buffer state while parsing a
    streamed response, and a shared instance would let concurrent sessions observe
    each other's partial state.
    """
    return _ADAPTERS[api_style](model_call)


def _attach_auth_origin(
    error: BackendError, origin: ApiKeyOrigin | None
) -> BackendError:
    """Attach the selected credential's origin to authentication failures."""
    if error.status in {httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN}:
        error.api_key_origin = origin
        error.args = (error._fmt(),)
    return error


def _merge_headers(*layers: dict[str, str] | None) -> dict[str, str]:
    """Merge header layers case-insensitively, with later layers winning."""
    merged: dict[str, str] = {}
    names: dict[str, str] = {}
    for layer in layers:
        for name, value in (layer or {}).items():
            folded = name.casefold()
            previous = names.get(folded)
            if previous is not None:
                merged.pop(previous, None)
            merged[name] = value
            names[folded] = name
    return merged


class GenericBackend:
    def __init__(  # noqa: PLR0913
        self,
        *,
        client: ChartreuxAsyncHTTPClient | None = None,
        provider: ProviderConfig,
        timeout: float = DEFAULT_API_TIMEOUT,
        retry_max_elapsed_time: float = 300.0,
        retry_budget: RequestRetryBudget | None = None,
        connect_timeout: float = DEFAULT_API_CONNECT_TIMEOUT,
        write_timeout: float = DEFAULT_API_WRITE_TIMEOUT,
        pool_timeout: float = DEFAULT_API_POOL_TIMEOUT,
        enable_system_trust_store: bool = False,
        on_retry: RetryObserver | None = None,
        pacer: AdaptivePacer | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            client: Optional Chartreux HTTP client to use. If not provided, one will be created.
            retry_max_elapsed_time: Total wall-clock budget for retrying retryable
                failures (429, 5xx, network/timeout errors). Retries continue for as
                long as the budget allows, matching the Mistral backend's behavior.
            on_retry: Notified before each retry backoff.
            pacer: Adaptive call pacing. Defaults to a fresh `AdaptivePacer`, which
                is a no-op until the first rate limit and then spaces subsequent
                calls out, recovering once a quiet window passes. Inject a custom
                one to tune the schedule or to stub timing in tests.
        """
        validate_api_style(provider.api_style)
        self._client = client
        self._owns_client = client is None
        self._enable_system_trust_store = enable_system_trust_store
        self._provider = provider
        self._timeout = timeout
        self._http_timeout = httpx.Timeout(
            timeout, connect=connect_timeout, write=write_timeout, pool=pool_timeout
        )
        self._retry_max_elapsed_time = retry_max_elapsed_time
        self._pacer = pacer if pacer is not None else AdaptivePacer()
        self._user_on_retry = on_retry

        async def paced_on_retry(reason: RetryReason) -> None:
            if reason.category is RetryCategory.RATE_LIMITED:
                self._pacer.on_rate_limited()
            if self._user_on_retry is not None:
                await self._user_on_retry(reason)

        # Budget-bounded retry: retry for as long as `retry_max_elapsed_time`
        # allows rather than for a fixed attempt count, so a transient 429 or
        # outage is waited out the same way the Mistral SDK backend does.
        retried = async_retry(
            tries=None,
            max_elapsed_time=retry_max_elapsed_time if retry_budget is None else None,
            budget=retry_budget,
            on_retry=paced_on_retry,
        )(self._send_request)
        self._make_request = self._pace(retried)
        retried_stream = async_generator_retry(
            tries=None,
            max_elapsed_time=retry_max_elapsed_time if retry_budget is None else None,
            budget=retry_budget,
            on_retry=paced_on_retry,
        )(self._send_parsed_streaming_request)
        self._make_streaming_request = self._pace_stream(retried_stream)

    def _pace(self, fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await self._pacer.acquire()
            succeeded = False
            try:
                result = await fn(*args, **kwargs)
                succeeded = True
                return result
            finally:
                if succeeded:
                    self._pacer.on_success()
                else:
                    self._pacer.on_failure()

        return wrapper

    def _pace_stream(
        self, fn: Callable[..., AsyncGenerator[Any]]
    ) -> Callable[..., AsyncGenerator[Any]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> AsyncGenerator[Any]:
            await self._pacer.acquire()
            succeeded = False
            try:
                async with aclosing(fn(*args, **kwargs)) as stream:
                    async for item in stream:
                        yield item
                    succeeded = True
            finally:
                if succeeded:
                    self._pacer.on_success()
                else:
                    self._pacer.on_failure()

        return wrapper

    async def __aenter__(self) -> GenericBackend:
        if self._client is None:
            self._client = ChartreuxAsyncHTTPClient(
                timeout=self._http_timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=MODEL_HTTP_KEEPALIVE_EXPIRY_SECONDS,
                ),
                enable_system_trust_store=self._enable_system_trust_store,
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> ChartreuxAsyncHTTPClient:
        if self._client is None:
            self._client = ChartreuxAsyncHTTPClient(
                timeout=self._http_timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=MODEL_HTTP_KEEPALIVE_EXPIRY_SECONDS,
                ),
                enable_system_trust_store=self._enable_system_trust_store,
            )
            self._owns_client = True
        return self._client

    async def complete(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> LLMChunk:
        resolved_api_key = resolve_api_key_with_origin(self._provider.api_key_env_var)
        api_key, api_key_origin = resolved_api_key or (None, None)

        api_style = getattr(self._provider, "api_style", "openai")
        model_call = ModelCall(
            provider=self._provider.name,
            endpoint=f"{self._provider.api_base}{_ADAPTER_ENDPOINTS.get(api_style, '')}",
            model=model.name,
            messages=messages,
            temperature=temperature,
            has_tools=bool(tools),
            tool_choice=tool_choice,
            api_key_origin=api_key_origin,
        )
        adapter = _get_adapter(api_style, model_call)

        req = adapter.prepare_request(
            model_name=model.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=False,
            provider=self._provider,
            api_key=api_key,
            thinking=model.thinking,
        )

        # Provider defaults apply to every backend call. Per-call headers override
        # them case-insensitively; adapter headers are applied last so generated
        # authentication and protocol headers retain their protection.
        headers = _merge_headers(
            self._provider.extra_headers, extra_headers, req.headers
        )

        base = req.base_url or self._provider.api_base
        url = f"{base}{req.endpoint}"

        try:
            response = await self._make_request(url, req.body, headers)
            return adapter.parse_response(response, self._provider)
        except OpenAIResponsesStreamError as e:
            backend_error = BackendErrorBuilder.build_stream_error(
                provider=self._provider.name,
                endpoint=url,
                status=e.status,
                error_type=e.error_type,
                error_message=e.message,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            )
            backend_error = _attach_auth_origin(backend_error, api_key_origin)
            if backend_error.status in {
                httpx.codes.UNAUTHORIZED,
                httpx.codes.FORBIDDEN,
            }:
                raise backend_error from None
            raise backend_error from e
        except httpx.HTTPStatusError as e:
            backend_error = BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                response=e.response,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            )
            backend_error = _attach_auth_origin(backend_error, api_key_origin)
            if backend_error.status in {
                httpx.codes.UNAUTHORIZED,
                httpx.codes.FORBIDDEN,
            }:
                raise backend_error from None
            raise backend_error from e
        except httpx.RequestError as e:
            backend_error = BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            )
            raise _attach_auth_origin(backend_error, api_key_origin) from e

    async def complete_streaming(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        resolved_api_key = resolve_api_key_with_origin(self._provider.api_key_env_var)
        api_key, api_key_origin = resolved_api_key or (None, None)

        api_style = getattr(self._provider, "api_style", "openai")
        model_call = ModelCall(
            provider=self._provider.name,
            endpoint=f"{self._provider.api_base}{_ADAPTER_ENDPOINTS.get(api_style, '')}",
            model=model.name,
            messages=messages,
            temperature=temperature,
            has_tools=bool(tools),
            tool_choice=tool_choice,
            api_key_origin=api_key_origin,
        )
        adapter = _get_adapter(api_style, model_call)

        req = adapter.prepare_request(
            model_name=model.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=True,
            provider=self._provider,
            api_key=api_key,
            thinking=model.thinking,
        )

        # Provider defaults apply to every backend call. Per-call headers override
        # them case-insensitively; adapter headers are applied last so generated
        # authentication and protocol headers retain their protection.
        headers = _merge_headers(
            self._provider.extra_headers, extra_headers, req.headers
        )

        base = req.base_url or self._provider.api_base
        url = f"{base}{req.endpoint}"

        try:
            async with aclosing(
                self._make_streaming_request(url, req.body, headers, adapter)
            ) as stream:
                async for stream_chunk in stream:
                    yield stream_chunk.chunk
        except OpenAIResponsesStreamError as e:
            backend_error = BackendErrorBuilder.build_stream_error(
                provider=self._provider.name,
                endpoint=url,
                status=e.status,
                error_type=e.error_type,
                error_message=e.message,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            )
            backend_error = _attach_auth_origin(backend_error, api_key_origin)
            if backend_error.status in {
                httpx.codes.UNAUTHORIZED,
                httpx.codes.FORBIDDEN,
            }:
                raise backend_error from None
            raise backend_error from e
        except httpx.HTTPStatusError as e:
            backend_error = BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                response=e.response,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            )
            backend_error = _attach_auth_origin(backend_error, api_key_origin)
            if backend_error.status in {
                httpx.codes.UNAUTHORIZED,
                httpx.codes.FORBIDDEN,
            }:
                raise backend_error from None
            raise backend_error from e
        except httpx.RequestError as e:
            backend_error = BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            )
            raise _attach_auth_origin(backend_error, api_key_origin) from e

    async def _send_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.post(url, content=data, headers=headers)
        response.raise_for_status()

        return response.json()

    async def _send_parsed_streaming_request(
        self, url: str, data: bytes, headers: dict[str, str], adapter: APIAdapter
    ) -> AsyncGenerator[ParsedStreamChunk]:
        async with aclosing(
            self._send_streaming_request(url, data, headers)
        ) as response_stream:
            async with aclosing(
                adapter.parse_stream(response_stream, self._provider)
            ) as parsed_stream:
                async for parsed in parsed_stream:
                    yield parsed

    async def _send_streaming_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> AsyncGenerator[dict[str, Any]]:
        client = self._get_client()
        async with client.stream(
            method="POST", url=url, content=data, headers=headers
        ) as response:
            if not response.is_success:
                await response.aread()
            response.raise_for_status()
            async for line in iter_sse_lines(response):
                if line.strip() == "":
                    continue

                if line.startswith(":"):
                    continue

                DELIM_CHAR = ":"
                if f"{DELIM_CHAR} " not in line:
                    raise ValueError(
                        f"Stream chunk improperly formatted. "
                        f"Expected `key{DELIM_CHAR} value`."
                    )
                delim_index = line.find(DELIM_CHAR)
                key = line[0:delim_index]
                value = line[delim_index + 2 :]

                if key != "data":
                    # This might be the case with openrouter, so we just ignore it
                    continue
                if value.strip() == "[DONE]":
                    return
                try:
                    chunk_data = json.loads(value.strip())
                except json.JSONDecodeError:
                    raise ValueError("Stream chunk contains malformed JSON.") from None
                yield chunk_data

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None
