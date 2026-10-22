from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from chartreux.core.dispatch.route_models import PackageTier, PurposeRoute, RouteTarget

SESSION_MODEL = "$session"


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


def shipped_routes() -> Mapping[str, PurposeRoute]:
    """Return deterministic route defaults; all models use the session binding."""

    def target(
        identity: str,
        tier: PackageTier = PackageTier.SUBSTANTIVE,
        seat: str | None = None,
        overlay: str | None = None,
    ) -> RouteTarget:
        return RouteTarget(
            model=SESSION_MODEL,
            identity=identity,
            tier=tier,
            seat=seat,
            overlay=overlay,
        )

    return {
        "implementation": PurposeRoute(targets=(target("coder"),)),
        "mechanical-edit": PurposeRoute(
            targets=(target("coder", PackageTier.MECHANICAL),)
        ),
        "research": PurposeRoute(targets=(target("researcher"),)),
        "search": PurposeRoute(targets=(target("researcher", PackageTier.MECHANICAL),)),
        "exploration": PurposeRoute(targets=(target("researcher"),)),
        "verification": PurposeRoute(
            targets=(target("coder", PackageTier.MECHANICAL),)
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
) -> ResolvedTarget:
    """Resolve one target. Catalog keys are selectable model aliases."""
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
        else (session_model if selected.model == SESSION_MODEL else selected.model)
    )
    if selected_model not in catalog:
        raise RouteResolutionError(
            f"Model binding {selected_model!r} for purpose {purpose!r} is unavailable; configure it explicitly."
        )
    if require_diversity and route.requires_distinct_models > 1:
        names: Sequence[str] = [
            item.model if item.model != SESSION_MODEL else session_model
            for item in route.targets
        ]
        if model is not None:
            names = [
                model
                if (item.model == selected.model and item.seat == selected.seat)
                else (session_model if item.model == SESSION_MODEL else item.model)
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
