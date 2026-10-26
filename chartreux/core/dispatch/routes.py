from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from chartreux.core.dispatch.route_models import PackageTier, PurposeRoute, RouteTarget

SESSION_MODEL = "$session"
# The cost-tier default: resolved at launch to the cheapest capable model in
# the attached catalog's price data (P0 decision 5); the capability criterion
# is pinned by the route tests' catalog fixture. With no priced model the
# default fails closed — it never silently falls back to the session binding.
COST_TIER_DEFAULT = "$cheapest-capable"


@dataclass(frozen=True)
class ResolvedTarget:
    model: str
    identity: str
    tier: str
    seat: str | None = None
    overlay: str | None = None
    replacement_capability: str = "none"


class RouteResolutionError(ValueError):
    """A requested purpose route cannot be resolved without guessing."""


def cheapest_capable_model(catalog_prices: Mapping[str, float]) -> str:
    """The selectable canonical model with the lowest known token price.

    ``catalog_prices`` maps canonical base names to their derived input+output
    price totals (see ``ModelResolver.catalog_prices``); only priced models
    enter it, so an empty mapping fails closed — the cost-tier default is
    never guessed and never silently falls back to the session binding.
    Deterministic: total price ascending, then name ascending.
    """
    if not catalog_prices:
        raise RouteResolutionError(
            "The cost-tier default cannot resolve: no selectable model "
            "carries known input and output prices. Add price data to the "
            "catalog or configure the route's model explicitly."
        )
    return min(catalog_prices, key=lambda name: (catalog_prices[name], name))


def _cost_tier_candidates(
    catalog: Mapping[str, object],
    canonical_models: Mapping[str, str] | None,
    catalog_prices: Mapping[str, float],
) -> dict[str, float]:
    """The priced canonical bases among the selectable catalog aliases."""
    canonical = canonical_models or {}
    bases = {canonical.get(alias, alias) for alias in catalog}
    return {base: price for base, price in catalog_prices.items() if base in bases}


def _effective_target_model(
    target_model: str,
    *,
    catalog: Mapping[str, object],
    session_model: str,
    canonical_models: Mapping[str, str] | None,
    catalog_prices: Mapping[str, float] | None,
) -> str:
    """The concrete model one route-target binding names.

    ``$session`` resolves to the session model; the cost-tier default
    resolves to the cheapest capable model among the catalog's selectable,
    priced bases; any other binding is literal.
    """
    if target_model == SESSION_MODEL:
        return session_model
    if target_model == COST_TIER_DEFAULT:
        return cheapest_capable_model(
            _cost_tier_candidates(catalog, canonical_models, catalog_prices or {})
        )
    return target_model


def shipped_routes() -> Mapping[str, PurposeRoute]:
    """Return deterministic route defaults.

    Every target binds the session model except the cost-tier-aware four
    (``implementation``, ``mechanical-edit``, ``search``, ``verification``),
    which bind ``$cheapest-capable``: the cheapest capable model in the
    attached catalog's price data, resolved at launch (P0 decision 5).
    """

    def target(
        identity: str,
        tier: PackageTier = PackageTier.SUBSTANTIVE,
        seat: str | None = None,
        overlay: str | None = None,
        model: str = SESSION_MODEL,
    ) -> RouteTarget:
        return RouteTarget(
            model=model, identity=identity, tier=tier, seat=seat, overlay=overlay
        )

    return {
        "implementation": PurposeRoute(
            targets=(target("coder", model=COST_TIER_DEFAULT),)
        ),
        "mechanical-edit": PurposeRoute(
            targets=(target("coder", PackageTier.MECHANICAL, model=COST_TIER_DEFAULT),)
        ),
        "research": PurposeRoute(targets=(target("researcher"),)),
        "search": PurposeRoute(
            targets=(
                target("researcher", PackageTier.MECHANICAL, model=COST_TIER_DEFAULT),
            )
        ),
        "exploration": PurposeRoute(targets=(target("researcher"),)),
        "verification": PurposeRoute(
            targets=(target("coder", PackageTier.MECHANICAL, model=COST_TIER_DEFAULT),)
        ),
        "design-analysis": PurposeRoute(targets=(target("advisor"),)),
        "planning-analysis": PurposeRoute(targets=(target("advisor"),)),
        "implementation-demanding-settled": PurposeRoute(targets=(target("coder"),)),
        "review.quick": PurposeRoute(targets=(target("reviewer"),)),
        "review.standard": PurposeRoute(targets=(target("reviewer"),)),
        "review.deep": PurposeRoute(
            targets=(
                target("reviewer", seat="analytical"),
                target("reviewer", seat="peer"),
                target("coder", seat="execution", overlay="review-execution"),
            ),
            requires_distinct_models=3,
        ),
    }


def resolve_route(
    purpose: str,
    catalog: Mapping[str, object],
    *,
    session_model: str,
    seat: str | None = None,
    model: str | None = None,
    identity: str | None = None,
    routes: Mapping[str, PurposeRoute] | None = None,
    canonical_models: Mapping[str, str] | None = None,
    require_diversity: bool = True,
    catalog_prices: Mapping[str, float] | None = None,
) -> ResolvedTarget:
    """Resolve one target. Catalog keys are selectable model aliases.

    ``catalog_prices`` maps canonical base names to their derived input+output
    price totals; the cost-tier default (``$cheapest-capable``) resolves
    through it and fails closed without it.
    """
    # The catalog is consulted only to reject unavailable explicit bindings;
    # its size never selects a route default.
    table = routes if routes is not None else shipped_routes()
    route = table.get(purpose)
    if route is None:
        raise RouteResolutionError(
            f"Unsupported purpose {purpose!r}; configure a route or choose a supported purpose."
        )
    if len(route.targets) > 1 and seat is None:
        raise RouteResolutionError(
            f"Purpose {purpose!r} requires an explicit seat; fan-out is not supported."
        )
    if seat is None:
        selected = next(iter(route.targets), None)
        if selected is None:
            raise RouteResolutionError(
                f"Purpose {purpose!r} has no configured targets."
            )
    else:
        matches = [item for item in route.targets if item.seat == seat]
        if len(matches) != 1:
            raise RouteResolutionError(
                f"Purpose {purpose!r} has no unique seat {seat!r}."
            )
        selected = matches[0]
    selected_model = (
        model
        if model is not None
        else _effective_target_model(
            selected.model,
            catalog=catalog,
            session_model=session_model,
            canonical_models=canonical_models,
            catalog_prices=catalog_prices,
        )
    )
    if selected_model not in catalog:
        raise RouteResolutionError(
            f"Model binding {selected_model!r} for purpose {purpose!r} is unavailable; configure it explicitly."
        )
    if require_diversity and route.requires_distinct_models > 1:
        names: Sequence[str] = [
            _effective_target_model(
                item.model,
                catalog=catalog,
                session_model=session_model,
                canonical_models=canonical_models,
                catalog_prices=catalog_prices,
            )
            for item in route.targets
        ]
        if model is not None:
            names = [
                model
                if (item.model == selected.model and item.seat == selected.seat)
                else _effective_target_model(
                    item.model,
                    catalog=catalog,
                    session_model=session_model,
                    canonical_models=canonical_models,
                    catalog_prices=catalog_prices,
                )
                for item in route.targets
            ]
        canonical = canonical_models or {}
        count = len({canonical.get(name, name) for name in names})
        if count < route.requires_distinct_models:
            raise RouteResolutionError(
                f"Purpose {purpose!r} requires {route.requires_distinct_models} distinct canonical models; resolved {count}. Configure distinct models explicitly."
            )
    return ResolvedTarget(
        selected_model,
        identity or selected.identity,
        selected.tier.value,
        selected.seat,
        selected.overlay,
        selected.replacement_capability.value,
    )


def validate_canonical_diversity(
    models: Sequence[str],
    required: int,
    canonical_models: Mapping[str, str] | None = None,
) -> None:
    canonical = canonical_models or {}
    count = len({canonical.get(model, model) for model in models})
    if count < required:
        raise RouteResolutionError(
            f"Requires {required} distinct canonical models; found {count}."
        )


def _is_launch_marker(model: str) -> bool:
    """Whether a target binding defers to launch-time resolution."""
    return model in {SESSION_MODEL, COST_TIER_DEFAULT}


def validate_route_table_diversity(
    routes: Mapping[str, PurposeRoute],
    *,
    catalog: Mapping[str, object],
    session_model: str,
    canonical_models: Mapping[str, str] | None = None,
    catalog_prices: Mapping[str, float] | None = None,
) -> None:
    """Validate a route table's distinct-model requirements at reload (ADR 0022).

    ``requires_distinct_models`` is validated HERE — at the reload that swaps
    the table in — not only at launch, so an invalid edited table rejects the
    whole reload and the prior table stays effective (fail-closed). The
    requirement is checked over canonical models exactly as launch checks it,
    with one deferral: a route whose targets ALL bind the launch-time markers
    (``$session`` / ``$cheapest-capable``) keeps its launch-time semantics —
    the shipped ``review.deep`` default is exactly that case, and its unmet
    requirement is reported in the launch-target table, never faked. The
    moment any target carries an explicit binding, the route's diversity is
    statically decidable and must hold.

    Raises :class:`RouteResolutionError` with an actionable message naming the
    purpose; the caller (the reload guard) turns it into a rejected reload.
    """
    for purpose, route in routes.items():
        if route.requires_distinct_models > len(route.targets):
            raise RouteResolutionError(
                f"Purpose {purpose!r} requires "
                f"{route.requires_distinct_models} distinct canonical models "
                f"but configures only {len(route.targets)} targets. Add "
                "targets or lower the requirement."
            )
        if route.requires_distinct_models <= 1:
            continue
        if all(_is_launch_marker(target.model) for target in route.targets):
            # Every binding defers to launch-time resolution (the session
            # model, the price data): the requirement stays a launch-time
            # rule, exactly as for the shipped review.deep default.
            continue
        names = [
            _effective_target_model(
                target.model,
                catalog=catalog,
                session_model=session_model,
                canonical_models=canonical_models,
                catalog_prices=catalog_prices,
            )
            for target in route.targets
        ]
        canonical = canonical_models or {}
        count = len({canonical.get(name, name) for name in names})
        if count < route.requires_distinct_models:
            raise RouteResolutionError(
                f"Purpose {purpose!r} requires "
                f"{route.requires_distinct_models} distinct canonical models; "
                f"resolved {count}. Configure distinct models explicitly."
            )
