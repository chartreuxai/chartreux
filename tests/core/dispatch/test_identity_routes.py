from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.dispatch.route_models import (
    PackageTier,
    PurposeRoute,
    ReplacementCapability,
    RouteTarget,
)
from chartreux.core.dispatch.routes import (
    COST_TIER_DEFAULT,
    SESSION_MODEL,
    RouteResolutionError,
    cheapest_capable_model,
    resolve_route,
    shipped_routes,
    validate_canonical_diversity,
    validate_route_table_diversity,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.resolver import ModelResolver
from tests.conftest import cost_tier_catalog

# The four shipped cost-tier-aware purposes (P1-c1; P0 decision 5).
COST_TIER_PURPOSES = ("implementation", "mechanical-edit", "search", "verification")


def _cost_tier_inputs() -> tuple[dict[str, object], dict[str, float]]:
    """The selectable aliases and price totals derived from the catalog
    fixture through the production derivation.
    """
    resolver = ModelResolver(CatalogSnapshot(cost_tier_catalog(), "cost-tier-fixture"))
    aliases: dict[str, object] = {base: None for base in resolver.eligible_models()}
    return aliases, resolver.catalog_prices()


def test_shipped_cost_tier_routes_default_to_the_cheapest_capable_model() -> None:
    """The four shipped cost-tier routes no longer bind ``$session``; the
    catalog fixture names the concrete default (the capability criterion is
    delegated to this fixture per settlement decision 5), so the pin is
    environment-independent.
    """
    routes = shipped_routes()
    for purpose in COST_TIER_PURPOSES:
        target = routes[purpose].targets[0]
        assert target.model == COST_TIER_DEFAULT
        assert target.model != SESSION_MODEL
    aliases, prices = _cost_tier_inputs()
    assert "budget-model" in aliases
    for purpose in COST_TIER_PURPOSES:
        resolved = resolve_route(
            purpose, aliases, session_model="session-model", catalog_prices=prices
        )
        assert resolved.model == "budget-model"
    # The tiers are unchanged: implementation stays substantive, the
    # mechanical trio stays mechanical.
    assert routes["implementation"].targets[0].tier is PackageTier.SUBSTANTIVE
    for purpose in ("mechanical-edit", "search", "verification"):
        assert routes[purpose].targets[0].tier is PackageTier.MECHANICAL


def test_other_shipped_routes_keep_their_session_bindings() -> None:
    """The plan's decision covers exactly the four cost-tier purposes: every
    other shipped route keeps its ``$session`` binding and resolves to the
    session model as before, and review.deep keeps its three-seat fan-out and
    distinct-model requirement.
    """
    routes = shipped_routes()
    for purpose in (
        "research",
        "exploration",
        "design-analysis",
        "planning-analysis",
        "implementation-demanding-settled",
        "review.quick",
        "review.standard",
    ):
        assert routes[purpose].targets[0].model == SESSION_MODEL
    deep = routes["review.deep"]
    assert [target.model for target in deep.targets] == [
        SESSION_MODEL,
        SESSION_MODEL,
        SESSION_MODEL,
    ]
    assert [target.seat for target in deep.targets] == [
        "analytical",
        "peer",
        "execution",
    ]
    assert deep.requires_distinct_models == 3
    for purpose in ("research", "exploration"):
        assert (
            resolve_route(purpose, {"session": 1}, session_model="session").model
            == "session"
        )


def test_unpriced_models_are_never_the_cost_tier_default() -> None:
    """The capability criterion: a model without price data is never the
    cost-tier default, however cheap or however selectable it is.
    """
    aliases, prices = _cost_tier_inputs()
    assert "mystery-model" in aliases
    assert "mystery-model" not in prices
    resolved = resolve_route(
        "implementation", aliases, session_model="session-model", catalog_prices=prices
    )
    assert resolved.model == "budget-model"


def test_cost_tier_default_fails_closed_without_price_data() -> None:
    """No price data, no default: the sentinel never falls back to the
    session binding, and the remediation names the explicit-configuration
    escape hatch.
    """
    with pytest.raises(RouteResolutionError, match="price"):
        resolve_route("implementation", {"budget-model": 1}, session_model="budget")
    with pytest.raises(RouteResolutionError, match="price"):
        resolve_route(
            "implementation",
            {"budget-model": 1},
            session_model="budget",
            catalog_prices={},
        )
    with pytest.raises(RouteResolutionError, match="price"):
        cheapest_capable_model({})
    # Price data with no selectable priced base fails closed the same way.
    with pytest.raises(RouteResolutionError, match="price"):
        resolve_route(
            "implementation",
            {"budget-model": 1},
            session_model="budget",
            catalog_prices={"elsewhere": 1.0},
        )


def test_cost_tier_default_is_deterministic_on_price_ties() -> None:
    """Two priced models with the same total resolve deterministically."""
    assert (
        cheapest_capable_model({"beta-model": 1.5, "alpha-model": 1.5}) == "alpha-model"
    )


def test_catalog_size_does_not_change_default_routes() -> None:
    assert shipped_routes() == shipped_routes()
    one = {"session": object()}
    ten = {"session": object(), **{f"m{i}": object() for i in range(9)}}
    assert resolve_route("research", one, session_model="session") == resolve_route(
        "research", ten, session_model="session"
    )
    # The cost-tier default is likewise independent of the alias-set size:
    # the priced selectable bases decide, never the catalog's size.
    assert resolve_route(
        "implementation", one, session_model="session", catalog_prices={"session": 1.0}
    ) == resolve_route(
        "implementation",
        ten,
        session_model="session",
        catalog_prices={"session": 1.0, **{f"m{i}": 9.0 for i in range(9)}},
    )


def test_research_and_explicit_overrides() -> None:
    assert (
        resolve_route("research", {"session": 1}, session_model="session").identity
        == "researcher"
    )
    result = resolve_route(
        "implementation",
        {"chosen": 1},
        session_model="session",
        model="chosen",
        identity="reviewer",
    )
    assert (result.model, result.identity) == ("chosen", "reviewer")


def test_multiseat_requires_selector_and_bindings() -> None:
    catalog = {"a": 1, "b": 2, "c": 3}
    routes = {
        "x": PurposeRoute(
            targets=(
                RouteTarget(
                    model="a",
                    identity="reviewer",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="analytical",
                ),
                RouteTarget(
                    model="b",
                    identity="reviewer",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="peer",
                ),
                RouteTarget(
                    model="c",
                    identity="coder",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="execution",
                    overlay="review-execution",
                ),
            )
        )
    }
    try:
        resolve_route(
            "x", catalog, session_model="a", routes=routes, require_diversity=False
        )
    except RouteResolutionError as error:
        assert "explicit seat" in str(error)
    else:
        raise AssertionError("seat selector must be required")
    assert (
        resolve_route(
            "x",
            catalog,
            session_model="a",
            routes=routes,
            seat="analytical",
            require_diversity=False,
        ).identity
        == "reviewer"
    )
    execution = resolve_route(
        "x",
        catalog,
        session_model="a",
        routes=routes,
        seat="execution",
        require_diversity=False,
    )
    assert (execution.identity, execution.overlay) == ("coder", "review-execution")


def test_canonical_diversity_counts_models_not_providers() -> None:
    try:
        validate_canonical_diversity(
            ["a", "a2", "b"], 3, {"a": "A", "a2": "A", "b": "B"}
        )
    except RouteResolutionError:
        pass
    else:
        raise AssertionError("A/A/B must fail diversity=3")
    try:
        validate_canonical_diversity(
            ["provider1", "provider2"], 2, {"provider1": "A", "provider2": "A"}
        )
    except RouteResolutionError:
        pass
    else:
        raise AssertionError("provider fanout must count once")


def test_tier_and_replacement_capability_are_distinct() -> None:
    target = RouteTarget(
        model=SESSION_MODEL,
        identity="coder",
        tier=PackageTier.MECHANICAL,
        replacement_capability=ReplacementCapability.NONE,
    )
    assert target.tier is PackageTier.MECHANICAL
    assert target.replacement_capability is ReplacementCapability.NONE
    deep = shipped_routes()["review.deep"]
    assert [t.identity for t in deep.targets] == ["reviewer", "reviewer", "coder"]


@pytest.mark.asyncio
async def test_config_loaded_purpose_routes_override_beats_the_default(
    tmp_path: Path,
) -> None:
    """A config-loaded route entry replaces the shipped default at resolution."""
    toml_path = tmp_path / "config.toml"
    toml_path.write_text(
        """\
[[orchestration.routes.research.targets]]
model = "fast-model"
identity = "coder"
tier = "mechanical"
"""
    )
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        DefaultConfigLayer(schema=ChartreuxConfigSchema),
        UserConfigLayer(path=toml_path, name="user"),
    ])
    config = await builder.build()
    routes = config.purpose_routes.routes
    # The sparse override deep-merges onto the shipped defaults: every shipped
    # purpose survives, and the explicit entry wins for its own purpose.
    assert set(routes) == set(shipped_routes())
    overridden = resolve_route(
        "research",
        {"fast-model": 1, "session": 1},
        session_model="session",
        routes=routes,
    )
    assert (overridden.model, overridden.identity, overridden.tier) == (
        "fast-model",
        "coder",
        "mechanical",
    )
    # Untouched purposes keep the shipped default through the same table —
    # for the cost-tier-aware implementation route, the shipped
    # $cheapest-capable binding resolves through the price data (a user
    # [orchestration.routes] entry naming a concrete model wins over it, as
    # the research entry above wins over its shipped default).
    kept = resolve_route(
        "implementation",
        {"session": 1},
        session_model="session",
        routes=routes,
        catalog_prices={"session": 1.0},
    )
    assert (kept.model, kept.identity, kept.tier) == ("session", "coder", "substantive")
    assert kept.model != COST_TIER_DEFAULT
    # The shipped default binds research to the session researcher; the
    # explicit entry beats exactly that default.
    default = resolve_route("research", {"session": 1}, session_model="session")
    assert (default.model, default.identity) == ("session", "researcher")


def test_deep_review_default_reports_missing_model_diversity() -> None:
    """Below 3 distinct canonical models the deep-review default fails closed."""
    try:
        resolve_route(
            "review.deep", {"session": 1}, session_model="session", seat="analytical"
        )
    except RouteResolutionError as error:
        message = str(error)
        assert "'review.deep'" in message
        assert "requires 3 distinct canonical models" in message
        assert "resolved 1" in message
        assert "Configure distinct models explicitly" in message
    else:
        raise AssertionError("deep review must report missing diversity")
    # Partial duplication (A/A/B) fails the same requirement.
    try:
        resolve_route(
            "review.deep",
            {"session": 1, "other": 1},
            session_model="session",
            seat="analytical",
            model="other",
        )
    except RouteResolutionError as error:
        assert "resolved 2" in str(error)
    else:
        raise AssertionError("A/A/B must fail diversity=3")


def _reload_target(
    model: str, identity: str = "reviewer", seat: str | None = None
) -> RouteTarget:
    return RouteTarget(
        model=model, identity=identity, tier=PackageTier.SUBSTANTIVE, seat=seat
    )


def test_reload_diversity_validation_accepts_the_shipped_table() -> None:
    """ADR 0022 check 6: the shipped table passes reload-time validation.
    review.deep's three-model requirement is deferred (every seat binds a
    launch marker), and every other route's requirement is trivially met.
    """
    validate_route_table_diversity(
        shipped_routes(), catalog={}, session_model="session-model"
    )


def test_reload_diversity_validation_rejects_a_statically_unmet_requirement() -> None:
    """An edited table that binds all three seats to one explicit model is
    statically unmet and rejects the whole reload (fail-closed).
    """
    routes = {
        "review.deep": PurposeRoute(
            targets=(
                _reload_target("glm-5-3", seat="analytical"),
                _reload_target("glm-5-3", seat="peer"),
                _reload_target("glm-5-3", seat="execution"),
            ),
            requires_distinct_models=3,
        )
    }
    with pytest.raises(RouteResolutionError, match="'review.deep'.*resolved 1"):
        validate_route_table_diversity(
            routes, catalog={}, session_model="session-model"
        )


def test_reload_diversity_validation_accepts_distinct_explicit_models() -> None:
    routes = {
        "review.deep": PurposeRoute(
            targets=(
                _reload_target("model-a", seat="analytical"),
                _reload_target("model-b", seat="peer"),
                _reload_target("model-c", seat="execution"),
            ),
            requires_distinct_models=3,
        )
    }
    validate_route_table_diversity(routes, catalog={}, session_model="session-model")


def test_reload_diversity_validation_rejects_more_requirements_than_targets() -> None:
    routes = {
        "research": PurposeRoute(
            targets=(_reload_target("model-a", identity="researcher"),),
            requires_distinct_models=2,
        )
    }
    with pytest.raises(RouteResolutionError, match="only 1 targets"):
        validate_route_table_diversity(
            routes, catalog={}, session_model="session-model"
        )


def test_reload_diversity_validation_rejects_a_mixed_marker_and_duplicate() -> None:
    """One explicit binding makes the route statically decidable: the mixed
    $session/model-a/model-a table resolves to two canonical models and
    fails the three-model requirement at reload, not only at launch.
    """
    routes = {
        "review.deep": PurposeRoute(
            targets=(
                _reload_target(SESSION_MODEL, seat="analytical"),
                _reload_target("model-a", seat="peer"),
                _reload_target("model-a", seat="execution"),
            ),
            requires_distinct_models=3,
        )
    }
    with pytest.raises(RouteResolutionError, match="resolved 2"):
        validate_route_table_diversity(
            routes, catalog={}, session_model="session-model"
        )


def test_reload_diversity_validation_counts_canonical_models_not_wire_names() -> None:
    """The reload-time check counts canonical models exactly as launch does:
    two wire names on one canonical base resolve to a single model.
    """
    routes = {
        "review.deep": PurposeRoute(
            targets=(
                _reload_target("wire-a", seat="analytical"),
                _reload_target("wire-b", seat="peer"),
            ),
            requires_distinct_models=2,
        )
    }
    with pytest.raises(RouteResolutionError, match="resolved 1"):
        validate_route_table_diversity(
            routes,
            catalog={},
            session_model="session-model",
            canonical_models={"wire-a": "model-x", "wire-b": "model-x"},
        )
