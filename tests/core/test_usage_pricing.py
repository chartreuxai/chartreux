from __future__ import annotations

from itertools import product

import pytest

from chartreux.core.agent_loop.llm_gateway import apply_usage
from chartreux.core.config import ModelConfig
from chartreux.core.llm_models import LLMUsage
from chartreux.core.session_types import AgentStats
from chartreux.core.usage import UsagePrices, capture_prices, price_usage


def test_component_pricing_subtracts_cached_input() -> None:
    result = price_usage(
        input_tokens=1_000,
        output_tokens=500,
        cached_input_tokens=400,
        prices=UsagePrices(input=2, output=4, cached_input=1),
    )
    assert result.input.tokens == 600
    assert result.input.known_cost_usd == pytest.approx(0.0012)
    assert result.cached_input.known_cost_usd == pytest.approx(0.0004)
    assert result.output.known_cost_usd == pytest.approx(0.002)
    assert result.known_cost_usd == pytest.approx(0.0036)
    assert not result.has_unknown_cost


def test_cached_input_is_not_clamped_to_prompt() -> None:
    result = price_usage(
        input_tokens=100,
        output_tokens=0,
        cached_input_tokens=200,
        prices=UsagePrices(input=2, output=4, cached_input=1),
    )
    assert result.input.tokens == 0
    assert result.cached_input.tokens == 200
    assert result.known_cost_usd == 0.0002


def test_unknown_components_do_not_use_fallback_prices() -> None:
    result = price_usage(
        input_tokens=1_000,
        output_tokens=500,
        cached_input_tokens=400,
        prices=UsagePrices(input=2),
    )
    assert result.known_cost_usd == 0.0012
    assert result.input.price_known
    assert not result.input.has_unknown_cost
    assert not result.cached_input.price_known
    assert result.cached_input.has_unknown_cost
    assert result.output.has_unknown_cost
    assert result.has_unknown_cost


@pytest.mark.parametrize("price", [None, 0.0])
def test_reported_zero_is_not_missing(price: float | None) -> None:
    prices = UsagePrices(input=price, output=price, cached_input=price)
    zero = price_usage(
        input_tokens=0, output_tokens=0, cached_input_tokens=0, prices=prices
    )
    missing = price_usage(
        input_tokens=None, output_tokens=None, cached_input_tokens=None, prices=prices
    )
    assert zero.known_cost_usd == missing.known_cost_usd == 0
    assert not zero.has_unknown_cost
    assert missing.has_unknown_cost
    assert zero.input.price_known == (price is not None)
    assert missing.input.tokens is None


def test_free_deployment_with_nonzero_tokens_is_priced_zero() -> None:
    result = price_usage(
        input_tokens=100,
        output_tokens=50,
        cached_input_tokens=20,
        prices=UsagePrices(input=0, output=0, cached_input=0),
    )
    assert result.known_cost_usd == 0
    assert not result.has_unknown_cost
    assert all(
        c.price_known for c in (result.input, result.output, result.cached_input)
    )


@pytest.mark.parametrize("input_tokens,cached_tokens", [(100, None), (None, 20)])
def test_partial_input_cannot_invent_uncached_count(
    input_tokens: int | None, cached_tokens: int | None
) -> None:
    result = price_usage(
        input_tokens=input_tokens,
        output_tokens=50,
        cached_input_tokens=cached_tokens,
        prices=UsagePrices(input=2, output=4, cached_input=1),
    )
    assert result.input.tokens is None
    assert result.input.has_unknown_cost
    assert result.output.known_cost_usd == 0.0002
    assert result.known_cost_usd == (50 * 4 + (cached_tokens or 0)) / 1_000_000
    assert result.has_unknown_cost


def test_capture_prices_is_frozen_and_honors_known_flags() -> None:
    model = ModelConfig(
        name="wire",
        provider="provider",
        alias="base",
        input_price=0,
        output_price=7,
        cached_input_price=2,
        output_price_known=False,
        cached_input_price_known=False,
    )
    prices = capture_prices(model)
    assert prices == UsagePrices(input=0, output=None, cached_input=None)
    model.input_price = 10
    model.output_price_known = True
    assert prices.input == 0
    assert prices.output is None


def test_absent_cached_rate_is_unknown() -> None:
    model = ModelConfig(name="wire", provider="provider", alias="base")
    assert capture_prices(model).cached_input is None


@pytest.mark.parametrize("known", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("counts", [(0, 0, 0), (100, 50, 20), (100, 0, 200)])
def test_gateway_preserves_legacy_pricing_exactly(
    known: tuple[bool, bool, bool], counts: tuple[int, int, int]
) -> None:
    input_known, cached_known, output_known = known
    prompt, completion, cached = counts
    model = ModelConfig(
        name="wire",
        provider="provider",
        alias="base",
        input_price=0.7,
        output_price=3.1,
        cached_input_price=0.2,
        input_price_known=input_known,
        output_price_known=output_known,
        cached_input_price_known=cached_known,
    )
    expected_cost = 0.0
    expected_unknown = False
    for count, price, is_known in (
        (max(0, prompt - cached), 0.7, input_known),
        (cached, 0.2, cached_known),
        (completion, 3.1, output_known),
    ):
        if count:
            if is_known:
                expected_cost += count * price
            else:
                expected_unknown = True
    stats = AgentStats(known_cost_total=1.0)
    apply_usage(
        stats,
        LLMUsage(
            prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached
        ),
        time_seconds=1,
        model=model,
    )
    assert stats.known_cost_total == 1.0 + expected_cost / 1_000_000
    assert stats.has_unknown_cost == expected_unknown


def test_gateway_preserves_invalid_known_cached_rate_assertion() -> None:
    model = ModelConfig(name="wire", provider="provider", alias="base")
    with pytest.raises(AssertionError):
        apply_usage(
            AgentStats(),
            LLMUsage(prompt_tokens=10, completion_tokens=0, cached_tokens=1),
            time_seconds=1,
            model=model,
        )


def test_gateway_unknown_cost_flag_is_sticky() -> None:
    stats = AgentStats(has_unknown_cost=True)
    apply_usage(
        stats,
        LLMUsage(prompt_tokens=0, completion_tokens=0, cached_tokens=0),
        time_seconds=0,
        model=ModelConfig(name="wire", provider="provider", alias="base"),
    )
    assert stats.has_unknown_cost
