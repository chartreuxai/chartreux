from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, Protocol

from chartreux.utils import VIBE_WARNING_TAG

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema
    from chartreux.core.message_list import MessageList
    from chartreux.core.session_types import AgentStats


class MiddlewareAction(StrEnum):
    CONTINUE = auto()
    STOP = auto()
    COMPACT = auto()
    INJECT_MESSAGE = auto()


class ResetReason(StrEnum):
    STOP = auto()
    COMPACT = auto()


@dataclass
class ConversationContext:
    messages: MessageList
    stats: AgentStats
    config: ChartreuxConfigSchema


@dataclass
class MiddlewareResult:
    action: MiddlewareAction = MiddlewareAction.CONTINUE
    message: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationMiddleware(Protocol):
    async def before_turn(self, context: ConversationContext) -> MiddlewareResult: ...

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None: ...


class TurnLimitMiddleware:
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if context.stats.steps - 1 >= self.max_turns:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=f"Turn limit of {self.max_turns} reached",
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class PriceLimitMiddleware:
    def __init__(self, max_price: float) -> None:
        self.max_price = max_price

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        lower_bound = context.stats.known_cost_total
        if lower_bound > self.max_price:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=f"Price limit exceeded: ${lower_bound:.4f} > ${self.max_price:.2f}",
            )
        if context.stats.has_unknown_cost:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=(
                    "Budget unverifiable: usage with unknown pricing was recorded "
                    f"(known-cost lower bound ${lower_bound:.4f})"
                ),
                metadata={"code": "budget-unverifiable"},
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class TokenLimitMiddleware:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if context.stats.session_total_llm_tokens > self.max_tokens:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=(
                    "Token limit exceeded: "
                    f"{context.stats.session_total_llm_tokens:,} > {self.max_tokens:,}"
                ),
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class RunBudgetMiddleware:
    """Per-run budget checkpoint over anchor-relative stat deltas.

    Holds only its configured maxes; all consumption figures are read from
    ``context.stats`` (the single ledger) relative to the run baseline the
    admission path wrote (``AgentStats.anchor_run_baseline``). The lazy
    anchor below is fail-safe: an unanchored run defaults the baseline to
    the current stats, so no budget is consumed and no stop fires.
    """

    def __init__(
        self,
        run_max_price: float | None = None,
        run_max_session_tokens: int | None = None,
    ) -> None:
        self.run_max_price = run_max_price
        self.run_max_session_tokens = run_max_session_tokens

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        return self.evaluate(context.stats)

    def evaluate(self, stats: AgentStats) -> MiddlewareResult:
        if (
            stats.run_anchor_steps is None
            or stats.run_anchor_known_cost_total is None
            or stats.run_anchor_session_tokens is None
            or stats.run_anchor_has_unknown_cost is None
        ):
            stats.anchor_run_baseline()

        if (
            self.run_max_price is not None
            and stats.run_anchor_known_cost_total is not None
        ):
            run_cost = stats.known_cost_total - stats.run_anchor_known_cost_total
            if run_cost > self.run_max_price:
                return MiddlewareResult(
                    action=MiddlewareAction.STOP,
                    reason=(
                        "Run price limit exceeded: "
                        f"${run_cost:.4f} > ${self.run_max_price:.2f}"
                    ),
                )

        if (
            self.run_max_session_tokens is not None
            and stats.run_anchor_session_tokens is not None
        ):
            run_tokens = (
                stats.session_total_llm_tokens - stats.run_anchor_session_tokens
            )
            if run_tokens > self.run_max_session_tokens:
                return MiddlewareResult(
                    action=MiddlewareAction.STOP,
                    reason=(
                        "Run token limit exceeded: "
                        f"{run_tokens:,} > {self.run_max_session_tokens:,}"
                    ),
                )

        if stats.has_unknown_cost and stats.run_anchor_has_unknown_cost is False:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=(
                    "Budget unverifiable: usage with unknown pricing was recorded "
                    "during this run"
                ),
                metadata={"code": "budget-unverifiable"},
            )

        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        # No-op, like the other budget middlewares: reset fires at
        # resume/compaction boundaries where re-anchoring would corrupt the
        # run budget (the run re-anchors at admission instead).
        pass


class AutoCompactMiddleware:
    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        threshold = context.config.get_active_model().auto_compact_threshold

        if threshold > 0 and context.stats.context_tokens >= threshold:
            return MiddlewareResult(action=MiddlewareAction.COMPACT)
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class ContextWarningMiddleware:
    def __init__(self, threshold_percent: float = 0.5) -> None:
        self.threshold_percent = threshold_percent
        self.has_warned = False

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if self.has_warned:
            return MiddlewareResult()

        max_context = context.config.get_active_model().auto_compact_threshold
        if max_context <= 0:
            return MiddlewareResult()

        if context.stats.context_tokens >= max_context * self.threshold_percent:
            self.has_warned = True

            percentage_used = (context.stats.context_tokens / max_context) * 100
            warning_msg = f"<{VIBE_WARNING_TAG}>You have used {percentage_used:.0f}% of your total context ({context.stats.context_tokens:,}/{max_context:,} tokens)</{VIBE_WARNING_TAG}>"

            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE, message=warning_msg
            )

        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        self.has_warned = False


class MiddlewarePipeline:
    def __init__(self) -> None:
        self.middlewares: list[ConversationMiddleware] = []

    def add(self, middleware: ConversationMiddleware) -> MiddlewarePipeline:
        self.middlewares.append(middleware)
        return self

    def clear(self) -> None:
        self.middlewares.clear()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        for mw in self.middlewares:
            mw.reset(reset_reason)

    async def run_before_turn(self, context: ConversationContext) -> MiddlewareResult:
        messages_to_inject = []
        for mw in self.middlewares:
            result = await mw.before_turn(context)
            if result.action == MiddlewareAction.INJECT_MESSAGE and result.message:
                messages_to_inject.append(result.message)
            elif result.action in {MiddlewareAction.STOP, MiddlewareAction.COMPACT}:
                return result
        if messages_to_inject:
            combined_message = "\n\n".join(messages_to_inject)
            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE, message=combined_message
            )

        return MiddlewareResult()
