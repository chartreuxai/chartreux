from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import aclosing, asynccontextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
import time
from typing import Literal, Protocol
from uuid import uuid4

from chartreux.core.agent_loop.errors import (
    AgentLoopLLMResponseError,
    EmptyLLMResponseError,
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
from chartreux.core.llm.failures import RequestRetryBudget
from chartreux.core.llm.types import BackendLike
from chartreux.core.llm_models import (
    AvailableTool,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StrToolChoice,
    posting_time,
)
from chartreux.core.session_types import AgentStats, CommittedModelIdentity
from chartreux.core.usage import (
    AccountingSink,
    CallAccountingOutcome,
    UsageAttribution,
    UsageOutcome,
    UsagePurpose,
    UsageRecord,
    UsageState,
    UsageTokens,
    capture_prices,
    price_usage,
)
from chartreux.core.utils.retry import (
    RetryCategory,
    RetryObserver,
    RetryReason,
    async_generator_retry,
    async_retry,
    bind_retry_budget,
)
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
    # Sideband calls are priced, but do not measure the live conversation.
    account_conversation: bool = True
    purpose: UsagePurpose = UsagePurpose.CONVERSATION


@dataclass(frozen=True)
class CallResources:
    backend: BackendLike
    stats: AgentStats
    process_message: Callable[[LLMMessage], LLMMessage]
    accounting_sink: AccountingSink | None = None
    usage_attribution: UsageAttribution | None = None
    retry_budget: RequestRetryBudget | None = None
    on_retry: RetryObserver | None = None
    admit_replay: Callable[[], bool] | None = None


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
    stats: AgentStats,
    usage: LLMUsage,
    *,
    time_seconds: float,
    model: ModelConfig,
    account_conversation: bool = True,
) -> None:
    """Price every call; only conversation calls replace the context measurement."""
    if account_conversation:
        stats.last_turn_duration = time_seconds
        stats.last_turn_prompt_tokens = usage.prompt_tokens
        stats.last_turn_completion_tokens = usage.completion_tokens
        stats.last_turn_cached_tokens = usage.cached_tokens
    stats.session_prompt_tokens += usage.prompt_tokens
    stats.session_completion_tokens += usage.completion_tokens
    stats.session_cached_tokens += usage.cached_tokens
    if account_conversation:
        stats.context_tokens = usage.prompt_tokens + usage.completion_tokens
    if usage.cached_tokens and model.cached_input_price_known:
        assert model.cached_input_price is not None
    pricing = price_usage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cached_input_tokens=usage.cached_tokens,
        prices=capture_prices(model),
    )
    stats.known_cost_total += pricing.known_cost_usd
    if pricing.has_unknown_cost:
        stats.has_unknown_cost = True
    if account_conversation and time_seconds > 0 and usage.completion_tokens > 0:
        stats.tokens_per_second = usage.completion_tokens / time_seconds


class CallFinalizer:
    """One invocation's frozen identity and exactly-once accounting settlement.

    Backends trip start at the transport boundary, not on backend entry. Streaming
    and utility callers can reuse this primitive with their own accumulated usage.
    """

    def __init__(self, inputs: CompletionInputs, resources: CallResources) -> None:
        self.started = False
        self.usage: LLMUsage | None = None
        self.outcome = UsageOutcome.FAILED
        self._settled = False
        self._sink = resources.accounting_sink
        self._prices = capture_prices(inputs.model)
        self._attribution = (
            resources.usage_attribution.model_copy(
                update={
                    "model": inputs.model.alias,
                    "provider": inputs.provider_name,
                    "wire_name": inputs.model.name,
                    "purpose": inputs.purpose,
                }
            )
            if resources.usage_attribution is not None
            else None
        )

    def start(self) -> None:
        self.started = True

    def capture_error_usage(self, error: BaseException) -> None:
        """Accept reported usage attached to a failure, never inferred counts."""
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            usage = getattr(current, "usage", None)
            if isinstance(usage, LLMUsage):
                self.usage = usage
                return
            current = current.__cause__

    async def _write(self, occurred_at: datetime) -> None:
        try:
            assert self._sink is not None and self._attribution is not None
            usage = self.usage
            tokens = UsageTokens(
                input_tokens=usage.prompt_tokens
                if usage is not None and usage.prompt_tokens_reported
                else None,
                output_tokens=usage.completion_tokens
                if usage is not None and usage.completion_tokens_reported
                else None,
                cached_input_tokens=usage.cached_tokens
                if usage is not None and usage.cached_tokens_reported
                else None,
            )
            outcome = CallAccountingOutcome(
                **tokens.model_dump(),
                outcome=self.outcome,
                usage_state=UsageState.PARTIAL
                if tokens.presence_state == UsageState.COMPLETE
                and usage is not None
                and not usage.is_final
                else tokens.presence_state,
            )
            pricing = price_usage(**tokens.model_dump(), prices=self._prices)
            record = UsageRecord(
                **self._attribution.model_dump(),
                **outcome.model_dump(),
                record_id=str(uuid4()),
                occurred_at=occurred_at,
                prices_usd_per_million=self._prices,
                known_cost_usd=pricing.known_cost_usd,
                has_unknown_cost=pricing.has_unknown_cost
                or outcome.usage_state != UsageState.COMPLETE,
            )
            await self._sink(record)
        except (Exception, asyncio.CancelledError):
            # Do not expose exception text or let a sink error trigger failover.
            logger.warning(
                "Usage accounting failed; recorded usage coverage is degraded"
            )

    async def finalize(self) -> None:
        if self._settled:
            return
        self._settled = True
        if not self.started or self._sink is None:
            return
        if self._attribution is None:
            logger.warning(
                "Usage attribution unavailable; recorded usage coverage is degraded"
            )
            return
        write = self._write(datetime.now(UTC))
        try:
            task = asyncio.create_task(write)
        except RuntimeError:
            write.close()
            logger.warning(
                "Usage accounting failed; recorded usage coverage is degraded"
            )
            return
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Retain and settle the one write even under repeated cancellation.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass
            task.result()
            raise


@asynccontextmanager
async def _budgeted_aclosing(
    stream: AsyncGenerator[LLMChunk, None], budget: RequestRetryBudget | None
) -> AsyncGenerator[AsyncGenerator[LLMChunk, None], None]:
    """Close with the attempt budget without binding it across outward yields."""
    closing = aclosing(stream)
    try:
        yield await closing.__aenter__()
    finally:
        with bind_retry_budget(budget) if budget is not None else nullcontext():
            await closing.__aexit__(None, None, None)


class LLMGateway:
    """Own provider payload projection, response normalization, and call accounting."""

    async def complete(
        self, inputs: CompletionInputs, resources: CallResources
    ) -> LLMChunk:
        """Reject empty attempts before replaying the identical request once."""
        call = async_retry(
            tries=2,
            budget=resources.retry_budget,
            is_retryable=lambda error: _admit_empty_replay(error, resources),
            on_retry=_empty_retry_observer(resources),
        )(self._complete_attempt)
        return await call(inputs, resources)

    async def _complete_attempt(
        self, inputs: CompletionInputs, resources: CallResources
    ) -> LLMChunk:
        from chartreux.core.llm.backend.generic import request_start_observer

        start_time = time.perf_counter()
        finalizer = CallFinalizer(inputs, resources)
        observer_token = request_start_observer.set(finalizer.start)
        original_error: BaseException | None = None
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
            with (
                bind_retry_budget(resources.retry_budget)
                if resources.retry_budget is not None
                else nullcontext()
            ):
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
            # Preserve raw usage before response processing can fail.
            finalizer.usage = result.usage
            finalizer.outcome = (
                UsageOutcome.REFUSED
                if result.stop and result.stop.is_refusal
                else UsageOutcome.COMPLETED
            )
            posted_at = posting_time()
            elapsed = time.perf_counter() - start_time
            try:
                result = LLMChunk(
                    message=resources.process_message(result.message).model_copy(
                        update={"posted_at": posted_at}
                    ),
                    usage=result.usage,
                    stop=result.stop,
                )
            except Exception:
                # Development accounted raw usage even when normalization failed.
                _apply_interrupted_usage(
                    resources.stats,
                    result.usage,
                    time_seconds=elapsed,
                    model=inputs.model,
                    account_conversation=inputs.account_conversation,
                )
                raise
            try:
                _validate_output(result, inputs)
            except (
                EmptyLLMResponseError,
                IncompleteLLMResponseError,
                ResponseTooLongError,
            ):
                _apply_interrupted_usage(
                    resources.stats,
                    result.usage,
                    time_seconds=elapsed,
                    model=inputs.model,
                    account_conversation=inputs.account_conversation,
                )
                raise
            if result.usage is None:
                raise AgentLoopLLMResponseError(
                    "Usage data missing in non-streaming completion response"
                )
            apply_usage(
                resources.stats,
                result.usage,
                time_seconds=elapsed,
                model=inputs.model,
                account_conversation=inputs.account_conversation,
            )
            return result
        except asyncio.CancelledError as error:
            original_error = error
            finalizer.outcome = UsageOutcome.INTERRUPTED
            finalizer.capture_error_usage(error)
            raise
        except Exception as error:
            original_error = error
            finalizer.outcome = UsageOutcome.FAILED
            finalizer.capture_error_usage(error)
            _log_model_call_failure(
                inputs.model.alias,
                inputs.provider_name,
                error,
                int((time.perf_counter() - start_time) * 1000),
            )
            raise _map_call_error(error, inputs) from error
        finally:
            request_start_observer.reset(observer_token)
            try:
                await finalizer.finalize()
            except asyncio.CancelledError:
                if original_error is None:
                    raise

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
        stream = async_generator_retry(
            tries=2,
            budget=resources.retry_budget,
            is_retryable=lambda error: _admit_empty_replay(error, resources),
            on_retry=_empty_retry_observer(resources),
        )(self._stream)
        return stream(inputs, resources, transcript=transcript)

    async def _stream(  # noqa: PLR0912, PLR0915 -- explicit stream exit/settlement paths
        self,
        inputs: CompletionInputs,
        resources: CallResources,
        *,
        transcript: TranscriptSink,
    ) -> AsyncGenerator[LLMChunk, None]:
        from chartreux.core.llm.backend.generic import request_start_observer

        chunk_agg: LLMChunk | None = None
        usage: LLMUsage | None = None
        usage_accounted = False
        transcript_appended = False
        published = False
        buffered: list[LLMChunk] = []
        start_time = time.perf_counter()
        finalizer = CallFinalizer(inputs, resources)
        original_error: BaseException | None = None
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
            async with _budgeted_aclosing(
                resources.backend.complete_streaming(
                    model=inputs.model,
                    messages=backend_messages,
                    temperature=inputs.model.temperature,
                    tools=inputs.tools,
                    tool_choice=inputs.tool_choice,
                    extra_headers=inputs.extra_headers,
                    max_tokens=inputs.max_tokens,
                    metadata=inputs.metadata,
                ),
                resources.retry_budget,
            ) as stream:
                while True:
                    # Scope context to backend advancement, never across a yield.
                    observer_token = request_start_observer.set(finalizer.start)
                    try:
                        with (
                            bind_retry_budget(resources.retry_budget)
                            if resources.retry_budget is not None
                            else nullcontext()
                        ):
                            chunk = await anext(stream)
                    except StopAsyncIteration:
                        break
                    finally:
                        request_start_observer.reset(observer_token)
                    # Capture raw reported usage before message processing can fail.
                    if chunk.usage is not None:
                        finalizer.usage = (
                            chunk.usage
                            if finalizer.usage is None
                            else finalizer.usage + chunk.usage
                        )
                    posted_at = (
                        chunk_agg.message.posted_at
                        if chunk_agg is not None
                        else posting_time()
                    )
                    processed_chunk = LLMChunk(
                        message=resources.process_message(chunk.message).model_copy(
                            update={"posted_at": posted_at}
                        ),
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
                    if published:
                        yield processed_chunk
                        continue
                    buffered.append(processed_chunk)
                    if not (
                        _valid_final_output(processed_chunk.message)
                        or (processed_chunk.message.reasoning_content or "").strip()
                    ):
                        continue
                    published = True
                    for fragment in buffered:
                        yield fragment
                    buffered.clear()

            elapsed = time.perf_counter() - start_time
            if chunk_agg is None:
                chunk_agg = LLMChunk(message=LLMMessage(role=Role.assistant))
            _validate_output(chunk_agg, inputs)
            if inputs.emits_finish_reason and chunk_agg.stop is None:
                raise IncompleteStreamError(inputs.provider_name, inputs.model.name)
            if chunk_agg.usage is None:
                raise AgentLoopLLMResponseError(
                    "Usage data missing in final chunk of streamed completion"
                )
            assert usage is not None
            apply_usage(
                resources.stats,
                usage,
                time_seconds=elapsed,
                model=inputs.model,
                account_conversation=inputs.account_conversation,
            )
            usage_accounted = True
            transcript(TranscriptAppend(message=chunk_agg.message, kind="complete"))
            transcript_appended = True
            finalizer.outcome = (
                UsageOutcome.REFUSED
                if chunk_agg.stop and chunk_agg.stop.is_refusal
                else UsageOutcome.COMPLETED
            )
            if chunk_agg.stop and chunk_agg.stop.is_refusal:
                raise _refusal_error(inputs.provider_name, inputs.model.name, chunk_agg)
            _log_success(inputs.model.alias, elapsed, usage)
        except asyncio.CancelledError as error:
            original_error = error
            finalizer.outcome = UsageOutcome.INTERRUPTED
            finalizer.capture_error_usage(error)
            if not usage_accounted:
                _apply_interrupted_usage(
                    resources.stats,
                    usage,
                    time_seconds=time.perf_counter() - start_time,
                    model=inputs.model,
                    account_conversation=inputs.account_conversation,
                )
            if not transcript_appended and published:
                _append_interrupted(transcript, chunk_agg)
            raise
        except GeneratorExit as error:
            original_error = error
            finalizer.outcome = UsageOutcome.INTERRUPTED
            if not usage_accounted:
                _apply_interrupted_usage(
                    resources.stats,
                    usage,
                    time_seconds=time.perf_counter() - start_time,
                    model=inputs.model,
                    account_conversation=inputs.account_conversation,
                )
            if not transcript_appended and published:
                _append_interrupted(transcript, chunk_agg)
            raise
        except Exception as error:
            original_error = error
            finalizer.outcome = (
                UsageOutcome.REFUSED
                if isinstance(error, RefusalError)
                else UsageOutcome.FAILED
            )
            finalizer.capture_error_usage(error)
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
                    account_conversation=inputs.account_conversation,
                )
            if not transcript_appended and published:
                _append_interrupted(transcript, chunk_agg)
            raise _map_call_error(error, inputs) from error
        finally:
            try:
                await finalizer.finalize()
            except asyncio.CancelledError:
                if original_error is None:
                    raise


class IncompleteLLMResponseError(AgentLoopLLMResponseError):
    """A provider explicitly marked its terminal response incomplete."""


def _valid_final_output(message: LLMMessage) -> bool:
    return bool((message.content or "").strip()) or bool(message.tool_calls)


def _validate_output(chunk: LLMChunk, inputs: CompletionInputs) -> None:
    stop = chunk.stop
    disposition = "accepted"
    error: Exception | None = None
    if stop and stop.is_refusal:
        disposition = "refused"
    elif stop and stop.reason == "incomplete":
        disposition = "incomplete"
        if not _valid_final_output(chunk.message):
            if stop.category == "max_output_tokens":
                error = ResponseTooLongError(inputs.provider_name, inputs.model.name)
            else:
                error = IncompleteLLMResponseError(
                    "The provider marked the response incomplete."
                )
    elif not _valid_final_output(chunk.message):
        disposition = "empty"
        error = EmptyLLMResponseError(inputs.provider_name, inputs.model.name)
    usage = chunk.usage
    logger.debug(
        "Model output disposition=%s content_length=%d tool_calls=%d stop_reason=%s prompt_tokens=%s completion_tokens=%s cached_tokens=%s",
        disposition,
        len((chunk.message.content or "").strip()),
        len(chunk.message.tool_calls or []),
        stop.reason if stop else None,
        usage.prompt_tokens if usage else None,
        usage.completion_tokens if usage else None,
        usage.cached_tokens if usage else None,
    )
    if error is not None:
        raise error


def _admit_empty_replay(error: Exception, resources: CallResources) -> bool:
    return isinstance(error, EmptyLLMResponseError) and (
        resources.admit_replay is None or resources.admit_replay()
    )


def _empty_retry_observer(resources: CallResources) -> RetryObserver:
    async def observe(reason: RetryReason) -> None:
        if resources.on_retry is not None:
            await resources.on_retry(
                RetryReason(
                    RetryCategory.UNKNOWN, "Empty assistant response; retrying once"
                )
            )

    return observe


def _apply_interrupted_usage(
    stats: AgentStats,
    usage: LLMUsage | None,
    *,
    time_seconds: float,
    model: ModelConfig,
    account_conversation: bool,
) -> None:
    """Account reported partial usage, or mark an interrupted stream unpriced."""
    if usage is None:
        stats.has_unknown_cost = True
        return
    apply_usage(
        stats,
        usage,
        time_seconds=time_seconds,
        model=model,
        account_conversation=account_conversation,
    )


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
                posted_at=message.posted_at,
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
        error,
        (
            EmptyLLMResponseError,
            IncompleteLLMResponseError,
            ResponseTooLongError,
            ImagesNotSupportedError,
            RefusalError,
            IncompleteStreamError,
        ),
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
