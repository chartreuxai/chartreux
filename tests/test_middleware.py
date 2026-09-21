from __future__ import annotations

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.message_list import MessageList
from chartreux.core.middleware import (
    ConversationContext,
    MiddlewareAction,
    PriceLimitMiddleware,
    TokenLimitMiddleware,
)
from chartreux.core.session_types import AgentStats


@pytest.fixture
def ctx(vibe_config: ChartreuxConfigSchema) -> ConversationContext:
    return ConversationContext(
        messages=MessageList(), stats=AgentStats(), config=vibe_config
    )


class TestTokenLimitMiddleware:
    @pytest.mark.asyncio
    async def test_stops_when_session_total_tokens_exceeds_limit(
        self, ctx: ConversationContext
    ) -> None:
        middleware = TokenLimitMiddleware(14)
        ctx.stats.session_prompt_tokens = 10
        ctx.stats.session_completion_tokens = 5

        result = await middleware.before_turn(ctx)

        assert result.action == MiddlewareAction.STOP
        assert result.reason == "Token limit exceeded: 15 > 14"

    @pytest.mark.asyncio
    async def test_allows_when_session_total_tokens_matches_limit(
        self, ctx: ConversationContext
    ) -> None:
        middleware = TokenLimitMiddleware(15)
        ctx.stats.session_prompt_tokens = 10
        ctx.stats.session_completion_tokens = 5

        result = await middleware.before_turn(ctx)

        assert result.action == MiddlewareAction.CONTINUE
        assert result.reason is None


class TestPriceLimitMiddleware:
    @pytest.mark.asyncio
    async def test_fresh_stats_do_not_stop_budget_enforcement(
        self, ctx: ConversationContext
    ) -> None:
        result = await PriceLimitMiddleware(0.0).before_turn(ctx)

        assert ctx.stats.has_unknown_cost is False
        assert result.action is MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_unpriced_usage_stops_budget_enforcement(
        self, ctx: ConversationContext
    ) -> None:
        ctx.stats.has_unknown_cost = True

        result = await PriceLimitMiddleware(1.0).before_turn(ctx)

        assert result.action is MiddlewareAction.STOP
        assert result.metadata == {"code": "budget-unverifiable"}
