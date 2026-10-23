from __future__ import annotations

from typing import Any

import pytest

from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.events import AssistantEvent
from chartreux.core.middleware import (
    AutoCompactMiddleware,
    MiddlewareAction,
    PriceLimitMiddleware,
    RunBudgetMiddleware,
    TokenLimitMiddleware,
    TurnLimitMiddleware,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.utils import VIBE_STOP_EVENT_TAG
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _middleware_types(agent) -> list[type]:
    return [type(mw) for mw in agent.middleware_pipeline.middlewares]


def test_run_budget_middleware_added_after_session_limits() -> None:
    agent = build_test_agent_loop(
        max_turns=10,
        max_price=1.0,
        max_session_tokens=1000,
        run_max_price=0.5,
        run_max_session_tokens=500,
    )

    types = _middleware_types(agent)
    assert types.index(TurnLimitMiddleware) < types.index(PriceLimitMiddleware)
    assert types.index(PriceLimitMiddleware) < types.index(TokenLimitMiddleware)
    assert types.index(TokenLimitMiddleware) < types.index(RunBudgetMiddleware)
    assert types[types.index(RunBudgetMiddleware) + 1] is AutoCompactMiddleware

    policy = agent.runtime_policy
    assert policy.run_max_price == 0.5
    assert policy.run_max_session_tokens == 500


def test_run_budget_absent_when_maxes_none() -> None:
    agent = build_test_agent_loop(max_turns=10, max_price=1.0, max_session_tokens=1000)

    types = _middleware_types(agent)
    assert RunBudgetMiddleware not in types
    assert types[types.index(TokenLimitMiddleware) + 1] is AutoCompactMiddleware
    assert agent.runtime_policy.run_max_price is None
    assert agent.runtime_policy.run_max_session_tokens is None


def test_admit_empty_replay_consults_run_budget() -> None:
    agent = build_test_agent_loop()
    agent._run_max_price = 0.5

    agent.stats.anchor_run_baseline()
    agent.stats.known_cost_total = 1.0
    assert agent._admit_empty_replay() is False

    # Pre-run consumption sits in the anchor: only the run delta counts.
    agent.stats.known_cost_total = 2.0
    agent.stats.anchor_run_baseline()
    assert agent._admit_empty_replay() is True


# --- B1: replay exhaustion propagates through the middleware budget-STOP path.


def _budget_config(*, unpriced: bool = False) -> ChartreuxConfigSchema:
    """A one-model test config with explicit $1/$2 per-million prices."""
    config = build_test_vibe_config(
        providers=[{"name": "test", "api_base": "https://test.invalid"}],
        models=[
            ModelConfig(
                name="test",
                provider="test",
                alias="test",
                input_price=1,
                output_price=2,
            )
        ],
        active_model="test",
    )
    if unpriced:
        assert config.catalog_snapshot is not None
        raw = config.catalog_snapshot.catalog.model_dump(mode="json")
        raw["models"]["test"]["deployments"][0]["prices"]["input"] = None
        raw["models"]["test"]["deployments"][0]["prices"]["output"] = None
        config.attach_catalog_snapshot(
            CatalogSnapshot(ModelCatalog.model_validate(raw), "unpriced")
        )
    return config


def _empty_chunk(prompt: int, completion: int) -> Any:
    return mock_llm_chunk(
        content=" \n", prompt_tokens=prompt, completion_tokens=completion
    )


async def _drive(agent: Any, message: str = "hello") -> list[Any]:
    events = []
    async for event in agent.act(message):
        events.append(event)
    return events


def _stop_events(events: list[Any]) -> list[AssistantEvent]:
    return [
        event
        for event in events
        if isinstance(event, AssistantEvent) and event.stopped_by_middleware
    ]


def _stop_content(reason: str) -> str:
    return f"<{VIBE_STOP_EVENT_TAG}>{reason}</{VIBE_STOP_EVENT_TAG}>"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
async def test_session_token_exhaustion_emits_budget_stop(streaming) -> None:
    # Isolated session-token denial with the run budgets absent and then
    # present-but-affordable: the emitted reason is the absolute session-total
    # one from TokenLimitMiddleware, which precedes the run budget in the
    # pipeline.
    scenarios = [
        ("run budgets absent", {"max_session_tokens": 100}),
        (
            "run budgets affordable",
            {"max_session_tokens": 100, "run_max_session_tokens": 1_000_000},
        ),
    ]
    for label, budget_kwargs in scenarios:
        backend = FakeBackend([[_empty_chunk(80, 30)]])
        agent = build_test_agent_loop(
            config=_budget_config(),
            backend=backend,
            enable_streaming=streaming,
            **budget_kwargs,
        )
        try:
            events = await _drive(agent)
        finally:
            await agent.aclose()
        stops = _stop_events(events)
        assert len(stops) == 1, (label, events)
        assert stops[0].content == _stop_content("Token limit exceeded: 110 > 100"), (
            label
        )
        assert stops[0].middleware_metadata == {}, label
        assert len(backend.requests_messages) == 1, label
        assert agent.stats.session_total_llm_tokens == 110, label


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
async def test_budget_stop_pipeline_precedence_matches_middleware_order(
    streaming,
) -> None:
    # Fresh loops for each overlap class; the emitted reason/metadata is
    # compared with the REAL pipeline result on the post-attempt stats, never
    # a reconstructed label. Pipeline order: turn limit -> absolute price ->
    # session tokens -> run budget (run price -> run tokens -> newly unknown
    # cost) -> compaction.
    scenarios = [
        # (label, unpriced, budgets, chunk, seed, expected requests)
        (
            "session price vs session tokens vs run stops",
            False,
            {
                "max_price": 0.1,
                "max_session_tokens": 100_000,
                "run_max_price": 0.1,
                "run_max_session_tokens": 100_000,
            },
            (100_000, 50_000),
            None,
            1,
        ),
        (
            "unknown session price vs session tokens",
            True,
            {"max_price": 1.0, "max_session_tokens": 100},
            (80, 30),
            None,
            1,
        ),
        (
            "session tokens vs run price",
            False,
            {"max_session_tokens": 100, "run_max_price": 0.0001},
            (80, 30),
            None,
            1,
        ),
        (
            "run price vs run tokens",
            False,
            {"run_max_price": 0.1, "run_max_session_tokens": 100_000},
            (100_000, 50_000),
            None,
            1,
        ),
        (
            "run tokens vs unknown",
            True,
            {"run_max_session_tokens": 100},
            (80, 30),
            None,
            1,
        ),
        (
            # The turn limit can only fire at the pre-turn boundary (steps
            # are frozen mid-turn), so this overlap is exercised through the
            # ordinary top-of-loop stop with the budgets also exceeded.
            "turn limit first",
            False,
            {"max_turns": 0, "max_price": 0.1, "run_max_price": 0.1},
            None,
            {"known_cost_total": 1.0},
            0,
        ),
    ]
    for label, unpriced, budgets, chunk, seed, expected_requests in scenarios:
        backend = FakeBackend([[_empty_chunk(*chunk)]] if chunk else None)
        agent = build_test_agent_loop(
            config=_budget_config(unpriced=unpriced),
            backend=backend,
            enable_streaming=streaming,
            **budgets,
        )
        if seed is not None:
            agent.stats.known_cost_total = seed["known_cost_total"]
        try:
            events = await _drive(agent)
            expected = await agent.middleware_pipeline.run_before_turn(
                agent._get_context()
            )
        finally:
            await agent.aclose()
        assert expected.action is MiddlewareAction.STOP, label
        assert expected.reason is not None, label
        stops = _stop_events(events)
        assert len(stops) == 1, (label, events)
        assert stops[0].content == _stop_content(expected.reason), label
        assert stops[0].middleware_metadata == expected.metadata, label
        assert len(backend.requests_messages) == expected_requests, label
