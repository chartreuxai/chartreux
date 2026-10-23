from __future__ import annotations

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.message_list import MessageList
from chartreux.core.middleware import (
    ConversationContext,
    MiddlewareAction,
    ResetReason,
    RunBudgetMiddleware,
)
from chartreux.core.session_types import AgentStats


@pytest.fixture
def ctx(vibe_config: ChartreuxConfigSchema) -> ConversationContext:
    return ConversationContext(
        messages=MessageList(), stats=AgentStats(), config=vibe_config
    )


class TestRunBudgetMiddleware:
    @pytest.mark.asyncio
    async def test_run_budget_price_delta_stop(self, ctx: ConversationContext) -> None:
        ctx.stats.known_cost_total = 0.0
        ctx.stats.anchor_run_baseline()
        ctx.stats.known_cost_total = 0.75

        result = await RunBudgetMiddleware(run_max_price=0.5).before_turn(ctx)

        assert result.action is MiddlewareAction.STOP
        assert result.reason == "Run price limit exceeded: $0.7500 > $0.50"
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_run_budget_prior_consumption_not_counted(
        self, ctx: ConversationContext
    ) -> None:
        ctx.stats.known_cost_total = 2.0
        ctx.stats.anchor_run_baseline()
        ctx.stats.known_cost_total = 2.25

        result = await RunBudgetMiddleware(run_max_price=0.5).before_turn(ctx)

        assert result.action is MiddlewareAction.CONTINUE
        assert result.reason is None

    @pytest.mark.asyncio
    async def test_run_budget_token_delta_stop(self, ctx: ConversationContext) -> None:
        ctx.stats.session_prompt_tokens = 10
        ctx.stats.session_completion_tokens = 5
        ctx.stats.anchor_run_baseline()
        ctx.stats.session_prompt_tokens = 30
        ctx.stats.session_completion_tokens = 10

        result = await RunBudgetMiddleware(run_max_session_tokens=20).before_turn(ctx)

        assert result.action is MiddlewareAction.STOP
        assert result.reason == "Run token limit exceeded: 25 > 20"

    @pytest.mark.asyncio
    async def test_run_budget_unverifiable_stop_carries_code(
        self, ctx: ConversationContext
    ) -> None:
        ctx.stats.anchor_run_baseline()
        ctx.stats.has_unknown_cost = True

        result = await RunBudgetMiddleware(run_max_price=10.0).before_turn(ctx)

        assert result.action is MiddlewareAction.STOP
        assert result.metadata == {"code": "budget-unverifiable"}

    @pytest.mark.asyncio
    async def test_run_budget_pre_run_unknown_latch_does_not_stop(
        self, ctx: ConversationContext
    ) -> None:
        ctx.stats.has_unknown_cost = True
        ctx.stats.anchor_run_baseline()

        result = await RunBudgetMiddleware(run_max_price=10.0).before_turn(ctx)

        assert result.action is MiddlewareAction.CONTINUE
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_run_budget_unanchored_self_anchors_fail_safe(
        self, ctx: ConversationContext
    ) -> None:
        ctx.stats.known_cost_total = 10.0
        ctx.stats.session_prompt_tokens = 300
        ctx.stats.session_completion_tokens = 200

        middleware = RunBudgetMiddleware(run_max_price=0.5)
        first = await middleware.before_turn(ctx)

        assert first.action is MiddlewareAction.CONTINUE
        assert ctx.stats.run_anchor_known_cost_total == 10.0
        assert ctx.stats.run_anchor_session_tokens == 500

        ctx.stats.known_cost_total = 10.75
        second = await middleware.before_turn(ctx)

        assert second.action is MiddlewareAction.STOP

    @pytest.mark.asyncio
    async def test_run_budget_price_delta_at_max_continues(
        self, ctx: ConversationContext
    ) -> None:
        ctx.stats.known_cost_total = 0.0
        ctx.stats.anchor_run_baseline()
        ctx.stats.known_cost_total = 0.5

        result = await RunBudgetMiddleware(run_max_price=0.5).before_turn(ctx)
        assert result.action is MiddlewareAction.CONTINUE

        ctx.stats.known_cost_total = 0.5001
        stopped = await RunBudgetMiddleware(run_max_price=0.5).before_turn(ctx)
        assert stopped.action is MiddlewareAction.STOP

    @pytest.mark.asyncio
    async def test_run_budget_token_delta_at_max_continues(
        self, ctx: ConversationContext
    ) -> None:
        ctx.stats.session_prompt_tokens = 0
        ctx.stats.session_completion_tokens = 0
        ctx.stats.anchor_run_baseline()
        ctx.stats.session_prompt_tokens = 12
        ctx.stats.session_completion_tokens = 8

        result = await RunBudgetMiddleware(run_max_session_tokens=20).before_turn(ctx)
        assert result.action is MiddlewareAction.CONTINUE

        ctx.stats.session_completion_tokens = 9
        stopped = await RunBudgetMiddleware(run_max_session_tokens=20).before_turn(ctx)
        assert stopped.action is MiddlewareAction.STOP

    @pytest.mark.asyncio
    async def test_run_budget_reset_is_noop(self, ctx: ConversationContext) -> None:
        ctx.stats.known_cost_total = 1.0
        ctx.stats.anchor_run_baseline()
        ctx.stats.known_cost_total = 2.0

        middleware = RunBudgetMiddleware(run_max_price=0.5)
        middleware.reset(ResetReason.COMPACT)

        assert ctx.stats.run_anchor_known_cost_total == 1.0
        result = await middleware.before_turn(ctx)
        assert result.action is MiddlewareAction.STOP
