from __future__ import annotations

import asyncio

from chartreux.core.config import ChartreuxConfigSchema, ModelConfig, ProviderConfig
from chartreux.core.llm.backend.factory import create_backend
from chartreux.core.llm.backend.generic import request_start_observer
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session_types import AgentStats
from chartreux.core.usage import (
    AccountingSink,
    UsageAttribution,
    UsageOutcome,
    UsagePurpose,
)
from chartreux.utils.api_keys import resolve_api_key
from chartreux.utils.http import get_user_agent


def select_utility_model(
    config: ChartreuxConfigSchema,
) -> tuple[ModelConfig, ProviderConfig]:
    """Use the active model/provider for secondary utility completions."""
    active = config.get_active_model()
    return active, config.get_provider_for_model(active)


def is_fast_utility_model(_config: ChartreuxConfigSchema) -> bool:
    """Whether utility completions use a dedicated cheap fast model."""
    return False


async def run_utility_completion(
    *,
    config: ChartreuxConfigSchema,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    request_timeout_seconds: float,
    retry_budget_seconds: float,
    skip_if_no_key: bool = False,
    accounting_sink: AccountingSink | None = None,
    usage_attribution: UsageAttribution | None = None,
    purpose: UsagePurpose = UsagePurpose.CONVERSATION,
) -> str | None:
    """Run a single non-streaming completion for a background nicety.

    Owns model selection, backend construction, budgets, and user-agent so
    features don't re-implement the seam. Returns
    the raw message content; the caller decides how to clean and interpret it.

    ``skip_if_no_key`` returns None before any network setup when the selected
    provider declares an API-key env var that is unset. It's for callers on a
    latency-sensitive path with a ready fallback (e.g. worktree naming at
    startup); leave it False so a missing key surfaces as a real backend failure
    instead of a silent no-op. A provider with an empty ``api_key_env_var`` is
    never skipped, since it needs no key to reach.
    """
    from chartreux.core.agent_loop.llm_gateway import (
        CallFinalizer,
        CallResources,
        CompletionInputs,
    )

    if not user_content.strip():
        return None
    model, provider = select_utility_model(config)
    if (
        skip_if_no_key
        and provider.api_key_env_var
        and not resolve_api_key(provider.api_key_env_var)
    ):
        return None
    backend = create_backend(
        provider=provider,
        timeout=request_timeout_seconds,
        retry_max_elapsed_time=retry_budget_seconds,
    )
    messages = [
        LLMMessage(role=Role.system, content=system_prompt),
        LLMMessage(role=Role.user, content=user_content),
    ]
    headers = {"user-agent": get_user_agent(provider.backend)}
    finalizer = CallFinalizer(
        CompletionInputs(
            model=model,
            provider_name=provider.name,
            emits_finish_reason=provider.emits_finish_reason,
            messages=tuple(messages),
            tools=None,
            tool_choice=None,
            extra_headers=headers,
            metadata={},
            max_tokens=max_tokens,
            account_conversation=False,
            purpose=purpose,
        ),
        CallResources(
            backend=backend,
            stats=AgentStats(),
            process_message=lambda message: message,
            accounting_sink=accounting_sink,
            usage_attribution=usage_attribution,
        ),
    )
    observer_token = request_start_observer.set(finalizer.start)
    original_error: BaseException | None = None
    try:
        async with backend:
            result = await backend.complete(
                model=model,
                messages=messages,
                temperature=0.0,
                tools=None,
                tool_choice=None,
                max_tokens=max_tokens,
                extra_headers=headers,
            )
            finalizer.usage = result.usage
            finalizer.outcome = (
                UsageOutcome.REFUSED
                if result.stop and result.stop.is_refusal
                else UsageOutcome.COMPLETED
            )
        return result.message.content
    except asyncio.CancelledError as error:
        original_error = error
        finalizer.outcome = UsageOutcome.INTERRUPTED
        finalizer.capture_error_usage(error)
        raise
    except Exception as error:
        original_error = error
        finalizer.outcome = UsageOutcome.FAILED
        finalizer.capture_error_usage(error)
        raise
    finally:
        request_start_observer.reset(observer_token)
        try:
            await finalizer.finalize()
        except asyncio.CancelledError:
            if original_error is None:
                raise
