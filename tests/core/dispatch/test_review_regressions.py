"""Review regressions at the successor routing, catalog, and launch boundaries.

Package 16b retired the dispatch slot/preset surface: routing is configured
through the purpose routes (``[orchestration.routes.<purpose>]``), the
conversation posture through the ``[agent]`` selections, and a legacy
``[dispatch]`` table is reported loudly at load. These tests pin the successor
surfaces at the boundaries the old policy/render/resume tests guarded.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from chartreux.core.agents.identities import builtin_identity_names
from chartreux.core.agents.launch import _resolved_model
from chartreux.core.dispatch import (
    SHIPPED_PURPOSES,
    PackageTier,
    RouteResolutionError,
    resolve_route,
    shipped_routes,
)
from chartreux.core.dispatch.routes import COST_TIER_DEFAULT
from chartreux.core.model_catalog.contracts import CatalogChanges
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot, load_catalog
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.core.subagents import InvalidLaunchModelError
from tests.conftest import (
    build_test_vibe_config,
    cost_tier_catalog,
    multi_model_catalog,
)


def _cost_tier_inputs() -> tuple[dict[str, object], dict[str, float]]:
    """The selectable aliases and price totals from the P1-c1 catalog
    fixture (the concrete default the fixture pins is ``budget-model``).
    """
    resolver = ModelResolver(CatalogSnapshot(cost_tier_catalog(), "cost-tier-fixture"))
    aliases: dict[str, object] = {base: None for base in resolver.eligible_models()}
    return aliases, resolver.catalog_prices()


def test_shipped_routes_cover_the_purpose_vocabulary() -> None:
    routes = shipped_routes()
    # Every purpose the composed artifact renders a "Purpose meanings" row for
    # has a shipped route; every route target names a registry identity.
    assert set(routes) == set(SHIPPED_PURPOSES)
    identities = set(builtin_identity_names())
    for route in routes.values():
        assert route.targets
        for target in route.targets:
            assert target.identity in identities
            assert target.tier in {PackageTier.MECHANICAL, PackageTier.SUBSTANTIVE}
            assert target.model
    mechanical = {
        purpose
        for purpose, route in routes.items()
        if any(t.tier == PackageTier.MECHANICAL for t in route.targets)
    }
    assert {"mechanical-edit", "search", "verification"} <= mechanical


def test_resolve_route_resolves_every_purpose_and_seat() -> None:
    aliases = dict(multi_model_catalog().models)
    cost_aliases, cost_prices = _cost_tier_inputs()

    # The cost-tier-aware implementation route resolves to the catalog
    # fixture's cheapest capable model, not the session model.
    resolved = resolve_route(
        "implementation",
        cost_aliases,
        session_model="beta-model",
        catalog_prices=cost_prices,
    )
    assert resolved.model == "budget-model"
    assert resolved.model != COST_TIER_DEFAULT
    assert resolved.identity == "coder"
    assert resolved.tier == PackageTier.SUBSTANTIVE.value
    assert resolved.seat is None

    # Mechanical-tier purposes resolve to the mechanical tier, on the same
    # cost-tier default.
    for purpose in ("mechanical-edit", "search", "verification"):
        resolved = resolve_route(
            purpose,
            cost_aliases,
            session_model="beta-model",
            catalog_prices=cost_prices,
        )
        assert resolved.model == "budget-model"
        assert resolved.tier == PackageTier.MECHANICAL.value

    # review.deep is multi-seat: a seat is required, and every seat resolves
    # to its own target (the execution seat carries its overlay).
    with pytest.raises(RouteResolutionError, match="requires an explicit seat"):
        resolve_route("review.deep", aliases, session_model="beta-model")
    seats = {
        "analytical": ("reviewer", None),
        "peer": ("reviewer", None),
        "execution": ("coder", "review-execution"),
    }
    for seat, (identity, overlay) in seats.items():
        resolved = resolve_route(
            "review.deep",
            aliases,
            session_model="beta-model",
            seat=seat,
            require_diversity=False,
        )
        assert resolved.seat == seat
        assert resolved.identity == identity
        assert resolved.overlay == overlay


def test_resolve_route_rejects_unknown_purpose_and_unavailable_model() -> None:
    aliases = dict(multi_model_catalog().models)

    with pytest.raises(RouteResolutionError, match="Unsupported purpose"):
        resolve_route("not-a-purpose", aliases, session_model="beta-model")
    # A session-bound purpose with a session model outside the selectable
    # aliases fails the availability check (the cost-tier default cannot
    # produce this failure: it resolves within the selectable, priced set).
    with pytest.raises(RouteResolutionError, match="unavailable"):
        resolve_route("research", aliases, session_model="removed-model")
    # review.deep requires three distinct canonical models: a one-model
    # session cannot satisfy it, even with a seat.
    with pytest.raises(RouteResolutionError, match="distinct canonical models"):
        resolve_route("review.deep", aliases, session_model="beta-model", seat="peer")


def test_write_boundary_carries_no_dispatch_policy() -> None:
    # The models catalog no longer owns executable dispatch policy: the
    # writer's change contract has no dispatch field to validate or persist,
    # so the retired table stays a load-time concern (see test_overlay.py).
    assert "dispatch" not in {
        field.name for field in dataclasses.fields(CatalogChanges)
    }


def test_legacy_dispatch_report_is_projected_to_clients(tmp_path: Path) -> None:
    from chartreux.app_server._projection import project_config_view

    path = tmp_path / "models.toml"
    path.write_text('[dispatch]\nmode = "standalone"\n')
    snapshot = load_catalog(path)
    assert len(snapshot.dispatch_diagnostics) == 1
    report = snapshot.dispatch_diagnostics[0]
    assert "retired" in report
    assert "[orchestration.routes.<purpose>]" in report

    config = build_test_vibe_config().attach_catalog_snapshot(snapshot)
    assert report in config.validation_warnings
    assert report in project_config_view(config).validation_warnings


def test_role_launch_resolves_live_against_the_catalog() -> None:
    catalog = multi_model_catalog()
    config = build_test_vibe_config().attach_catalog_snapshot(
        CatalogSnapshot(catalog, "live")
    )

    model, _identity = _resolved_model(config, "@scout")
    assert model.alias == "alpha-model"
    assert model.name == "alpha-wire"
    model, _identity = _resolved_model(config, "@main")
    assert model.alias == "beta-model"

    # A role bound to a model the catalog no longer carries fails the launch
    # — roles resolve live, never from a remembered binding.
    models = dict(catalog.models)
    models.pop("alpha-model")
    config.attach_catalog_snapshot(
        CatalogSnapshot(catalog.model_copy(update={"models": models}), "after")
    )
    with pytest.raises(InvalidLaunchModelError):
        _resolved_model(config, "@scout")
    # The surviving roles still resolve: the failure is not a blanket one.
    model, _identity = _resolved_model(config, "@main")
    assert model.alias == "beta-model"


def test_startup_broken_catalog_fails_closed_without_standalone_fallback() -> None:
    # An empty catalog resolves nothing: no role launch silently falls back to
    # a standalone/default model, and no route resolves against it.
    catalog = SHIPPED_CATALOG.model_copy(update={"models": {}})
    config = build_test_vibe_config().attach_catalog_snapshot(
        CatalogSnapshot(catalog, "broken")
    )
    with pytest.raises(InvalidLaunchModelError):
        _resolved_model(config, "@main")
    # An empty catalog resolves no route: the cost-tier default fails closed
    # on the empty priced set (no session-model fallback), loudly.
    with pytest.raises(RouteResolutionError, match="price"):
        resolve_route("implementation", {}, session_model="glm-5-3")
