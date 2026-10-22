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
    SESSION_MODEL,
    RouteResolutionError,
    resolve_route,
    shipped_routes,
    validate_canonical_diversity,
)


def test_catalog_size_does_not_change_default_routes() -> None:
    assert shipped_routes() == shipped_routes()
    one = {"session": object()}
    ten = {"session": object(), **{f"m{i}": object() for i in range(9)}}
    assert resolve_route(
        "implementation", one, session_model="session"
    ) == resolve_route("implementation", ten, session_model="session")


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
    # Untouched purposes keep the shipped default through the same table.
    kept = resolve_route(
        "implementation", {"session": 1}, session_model="session", routes=routes
    )
    assert (kept.model, kept.identity, kept.tier) == ("session", "coder", "substantive")
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
