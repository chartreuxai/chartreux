from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass
from http import HTTPStatus
import time
from typing import Literal, Protocol

from chartreux.core.agent_loop.errors import (
    AgentLoopLLMResponseError,
    ImagesNotSupportedError,
)
from chartreux.core.compaction.context import (
    reorder_for_tool_adjacency,
    select_model_context,
)
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.errors import (
    ContextTooLongError,
    RateLimitError,
    RefusalError,
    ResponseTooLongError,
)
from chartreux.core.llm.backend.factory import create_backend
from chartreux.core.llm.exceptions import BackendError, IncompleteStreamError
from chartreux.core.llm.types import BackendLike
from chartreux.core.llm_models import (
    AvailableTool,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StrToolChoice,
)
from chartreux.core.session_types import AgentStats, CommittedModelIdentity
from chartreux.core.utils import RetryObserver
from chartreux.observability.logging import log_model_call_success, logger


@dataclass(frozen=True)
class CompletionInputs:
    model: ModelConfig
    provider_name: str
    emits_finish_reason: bool
    messages: tuple[LLMMessage, ...]
    tools: list[AvailableTool] | None
    tool_choice: StrToolChoice | AvailableTool | None
    extra_headers: dict[str, str]
    metadata: dict[str, str]
    max_tokens: int | None


@dataclass(frozen=True)
class CallResources:
    backend: BackendLike
    stats: AgentStats
    process_message: Callable[[LLMMessage], LLMMessage]


@dataclass(frozen=True)
class TranscriptAppend:
    message: LLMMessage
    kind: Literal["complete", "interrupted"]
    committed_model: CommittedModelIdentity | None = None


class TranscriptSink(Protocol):
    def __call__(self, outcome: TranscriptAppend, /) -> None:
        """Synchronously append to the authoritative transcript without raising."""


def select_backend(
    config: ChartreuxConfigSchema,
    *,
    on_retry: RetryObserver,
    factory: Callable[..., BackendLike] = create_backend,
) -> BackendLike:
    """Create a backend from the selected provider configuration."""
    provider = config.get_active_provider()
    return factory(
        provider=provider,
        on_retry=on_retry,
        timeout=config.api_timeout,
        retry_max_elapsed_time=config.api_retry_max_elapsed_time,
        connect_timeout=config.api_connect_timeout,
        write_timeout=config.api_write_timeout,
        pool_timeout=config.api_pool_timeout,
        enable_system_trust_store=config.enable_system_trust_store,
    )


def messages_for_backend(
    messages: Sequence[LLMMessage], active_model: ModelConfig
) -> Sequence[LLMMessage]:
    """Project durable history into provider-safe model context."""
    messages = reorder_for_tool_adjacency(select_model_context(messages))
    if active_model.supports_images or not any(message.images for message in messages):
        return messages
    raise ImagesNotSupportedError(active_model.alias)


def apply_usage(
    stats: AgentStats, usage: LLMUsage, *, time_seconds: float, model: ModelConfig
) -> None:
    """Apply one model call's usage and price it at this call's deployment."""
    stats.last_turn_duration = time_seconds
    stats.last_turn_prompt_tokens = usage.prompt_tokens
    stats.last_turn_completion_tokens = usage.completion_tokens
    stats.last_turn_cached_tokens = usage.cached_tokens
    stats.session_prompt_tokens += usage.prompt_tokens
    stats.session_completion_tokens += usage.completion_tokens
    stats.session_cached_tokens += usage.cached_tokens
    stats.context_tokens = usage.prompt_tokens + usage.completion_tokens
    uncached = max(0, usage.prompt_tokens - usage.cached_tokens)
    known_cost = 0.0
    has_unknown = False
    if uncached:
        if model.input_price_known:
            known_cost += uncached * model.input_price
        else:
            has_unknown = True
    if usage.cached_tokens:
        if model.cached_input_price_known:
            assert model.cached_input_price is not None
            known_cost += usage.cached_tokens * model.cached_input_price
        else:
            has_unknown = True
    if usage.completion_tokens:
        if model.output_price_known:
            known_cost += usage.completion_tokens * model.output_price
        else:
            has_unknown = True
    stats.known_cost_total += known_cost / 1_000_000
    if has_unknown:
        stats.has_unknown_cost = True
    if time_seconds > 0 and usage.completion_tokens > 0:
        stats.tokens_per_second = usage.completion_tokens / time_seconds


class LLMGateway:
    """Own provider payload projection, response normalization, and call accounting."""

    async def complete(
        self, inputs: CompletionInputs, resources: CallResources
    ) -> LLMChunk:
        """Account and normalize; no transcript append, refusal check, or success log."""
        start_time = time.perf_counter()
        try:
            backend_messages = messages_for_backend(inputs.messages, inputs.model)
            logger.debug(
                "Model call starting model=%s provider=%s messages=%d tools=%d thinking=%s",
                inputs.model.alias,
                inputs.provider_name,
                len(backend_messages),
                len(inputs.tools) if inputs.tools else 0,
                inputs.model.thinking,
            )
            result = await resources.backend.complete(
                model=inputs.model,
                messages=backend_messages,
                temperature=inputs.model.temperature,
                tools=inputs.tools,
                tool_choice=inputs.tool_choice,
                extra_headers=inputs.extra_headers,
                max_tokens=inputs.max_tokens,
                metadata=inputs.metadata,
            )
            elapsed = time.perf_counter() - start_time
            if result.usage is None:
                raise AgentLoopLLMResponseError(
                    "Usage data missing in non-streaming completion response"
                )
            apply_usage(
                resources.stats, result.usage, time_seconds=elapsed, model=inputs.model
            )
            return LLMChunk(
                message=resources.process_message(result.message),
                usage=result.usage,
                stop=result.stop,
            )
        except Exception as error:
            _log_model_call_failure(
                inputs.model.alias,
                inputs.provider_name,
                error,
                int((time.perf_counter() - start_time) * 1000),
            )
            raise _map_call_error(error, inputs) from error

    async def chat(
        self,
        inputs: CompletionInputs,
        resources: CallResources,
        *,
        transcript: TranscriptSink,
    ) -> LLMChunk:
        """Run a completion, append it, check refusal, and record success."""
        start_time = time.perf_counter()
        result = await self.complete(inputs, resources)
        transcript(TranscriptAppend(message=result.message, kind="complete"))
        if result.stop and result.stop.is_refusal:
            raise _refusal_error(inputs.provider_name, inputs.model.name, result)
        _log_success(inputs.model.alias, time.perf_counter() - start_time, result.usage)
        return result

    def chat_streaming(
        self,
        inputs: CompletionInputs,
        resources: CallResources,
        *,
        transcript: TranscriptSink,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Stream, aggregate, validate, account, append, check refusal, and log."""
        return self._stream(inputs, resources, transcript=transcript)

    async def _stream(
        self,
        inputs: CompletionInputs,
        resources: CallResources,
        *,
        transcript: TranscriptSink,
    ) -> AsyncGenerator[LLMChunk, None]:
        chunk_agg: LLMChunk | None = None
        usage: LLMUsage | None = None
        usage_accounted = False
        transcript_appended = False
        start_time = time.perf_counter()
        try:
            backend_messages = messages_for_backend(inputs.messages, inputs.model)
            logger.debug(
                "Model call starting model=%s provider=%s messages=%d tools=%d streaming=%s",
                inputs.model.alias,
                inputs.provider_name,
                len(backend_messages),
                len(inputs.tools) if inputs.tools else 0,
                True,
            )
            async for chunk in resources.backend.complete_streaming(
                model=inputs.model,
                messages=backend_messages,
                temperature=inputs.model.temperature,
                tools=inputs.tools,
                tool_choice=inputs.tool_choice,
                extra_headers=inputs.extra_headers,
                max_tokens=inputs.max_tokens,
                metadata=inputs.metadata,
            ):
                processed_chunk = LLMChunk(
                    message=resources.process_message(chunk.message),
                    usage=chunk.usage,
                    stop=chunk.stop,
                )
                chunk_agg = (
                    processed_chunk
                    if chunk_agg is None
                    else chunk_agg + processed_chunk
                )
                if chunk.usage is not None:
                    usage = chunk.usage if usage is None else usage + chunk.usage
                yield processed_chunk

            elapsed = time.perf_counter() - start_time
            if chunk_agg is None or (
                inputs.emits_finish_reason and chunk_agg.stop is None
            ):
                raise IncompleteStreamError(inputs.provider_name, inputs.model.name)
            if chunk_agg.usage is None:
                raise AgentLoopLLMResponseError(
                    "Usage data missing in final chunk of streamed completion"
                )
            assert usage is not None
            apply_usage(
                resources.stats, usage, time_seconds=elapsed, model=inputs.model
            )
            usage_accounted = True
            transcript(TranscriptAppend(message=chunk_agg.message, kind="complete"))
            transcript_appended = True
            if chunk_agg.stop and chunk_agg.stop.is_refusal:
                raise _refusal_error(inputs.provider_name, inputs.model.name, chunk_agg)
            _log_success(inputs.model.alias, elapsed, usage)
        except asyncio.CancelledError:
            if not usage_accounted:
                _apply_interrupted_usage(
                    resources.stats,
                    usage,
                    time_seconds=time.perf_counter() - start_time,
                    model=inputs.model,
                )
            if not transcript_appended:
                _append_interrupted(transcript, chunk_agg)
            raise
        except GeneratorExit:
            if not usage_accounted:
                _apply_interrupted_usage(
                    resources.stats,
                    usage,
                    time_seconds=time.perf_counter() - start_time,
                    model=inputs.model,
                )
            if not transcript_appended:
                _append_interrupted(transcript, chunk_agg)
            raise
        except Exception as error:
            _log_model_call_failure(
                inputs.model.alias,
                inputs.provider_name,
                error,
                int((time.perf_counter() - start_time) * 1000),
            )
            if not usage_accounted:
                _apply_interrupted_usage(
                    resources.stats,
                    usage,
                    time_seconds=time.perf_counter() - start_time,
                    model=inputs.model,
                )
            if not transcript_appended:
                _append_interrupted(transcript, chunk_agg)
            raise _map_call_error(error, inputs) from error


def _apply_interrupted_usage(
    stats: AgentStats,
    usage: LLMUsage | None,
    *,
    time_seconds: float,
    model: ModelConfig,
) -> None:
    """Account reported partial usage, or mark an interrupted stream unpriced."""
    if usage is None:
        stats.has_unknown_cost = True
        return
    apply_usage(stats, usage, time_seconds=time_seconds, model=model)


def _append_interrupted(transcript: TranscriptSink, chunk: LLMChunk | None) -> None:
    if chunk is None:
        return
    message = chunk.message
    if not (message.content or message.reasoning_content or message.tool_calls):
        return
    transcript(
        TranscriptAppend(
            message=LLMMessage(
                role=Role.assistant,
                content=message.content,
                reasoning_content=message.reasoning_content,
                reasoning_payloads=message.reasoning_payloads,
                reasoning_message_id=message.reasoning_message_id,
                tool_calls=message.tool_calls,
                message_id=message.message_id,
            ),
            kind="interrupted",
        )
    )


def _log_success(alias: str, elapsed: float, usage: LLMUsage | None) -> None:
    log_model_call_success(
        alias,
        int(elapsed * 1000),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        cached_tokens=usage.cached_tokens if usage else 0,
    )


def _refusal_error(provider: str, model: str, chunk: LLMChunk) -> RefusalError:
    stop = chunk.stop
    return RefusalError(
        provider,
        model,
        category=stop.category if stop else None,
        explanation=stop.explanation if stop else None,
    )


def _map_call_error(error: Exception, inputs: CompletionInputs) -> Exception:
    if isinstance(
        error, (ImagesNotSupportedError, RefusalError, IncompleteStreamError)
    ):
        return error
    if isinstance(error, BackendError) and error.is_invalid_model:
        return error
    if _is_non_retryable_error(error):
        return error
    if _should_raise_rate_limit_error(error):
        mapped: Exception = RateLimitError(inputs.provider_name, inputs.model.name)
    elif _is_context_too_long_error(error):
        mapped = ContextTooLongError(inputs.provider_name, inputs.model.name)
    elif _is_response_too_long_error(error):
        mapped = ResponseTooLongError(inputs.provider_name, inputs.model.name)
    else:
        mapped = RuntimeError(
            f"API error from {inputs.provider_name} (model: {inputs.model.name}): {error}"
        )
    return mapped


def _log_model_call_failure(
    alias: str, provider: str, error: Exception, duration_ms: int
) -> None:
    backend_error = _extract_backend_error(error)
    if backend_error is not None:
        logger.warning(
            "Model call failed model=%s duration_ms=%d\n%s",
            alias,
            duration_ms,
            str(backend_error),
        )
    else:
        logger.warning(
            "Model call failed model=%s provider=%s error=%s duration_ms=%d",
            alias,
            provider,
            type(error).__name__,
            duration_ms,
        )


def _extract_backend_error(error: BaseException) -> BackendError | None:
    if isinstance(error, BackendError):
        return error
    if isinstance(error, RuntimeError) and isinstance(error.__cause__, BackendError):
        return error.__cause__
    return None


def _should_raise_rate_limit_error(error: Exception) -> bool:
    return (
        isinstance(error, BackendError) and error.status == HTTPStatus.TOO_MANY_REQUESTS
    )


def _is_context_too_long_error(error: Exception) -> bool:
    if isinstance(error, BackendError):
        return error.is_context_too_long
    if isinstance(error, RuntimeError) and isinstance(error.__cause__, BackendError):
        return error.__cause__.is_context_too_long
    return False


def _is_response_too_long_error(error: Exception) -> bool:
    if isinstance(error, BackendError):
        return error.is_response_too_long
    if isinstance(error, RuntimeError) and isinstance(error.__cause__, BackendError):
        return error.__cause__.is_response_too_long
    return False


def _is_non_retryable_error(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if getattr(current, "non_retryable", False):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False
