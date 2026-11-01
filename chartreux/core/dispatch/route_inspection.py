"""Side-effect-free inspection of a purpose-route table (v0.7.0, O2).

One inspection routine feeds both observability surfaces: the ``/routes``
browser renders the conversation's LIVE table, and ``chartreux doctor``
renders the independently loaded configuration. Every function here is a pure
read over an explicit context — nothing launches an agent, resolves a
deployment, mutates configuration, or starts a conversation — and target
previews resolve through the SAME primitives the launch path uses, so a
preview can never disagree with what a launch would actually bind.

Two facts stay deliberately separate:

* ``reload validation`` — the structural ``requires_distinct_models`` check the
  reload guard applies. A route whose targets ALL bind launch-time markers
  (``$session`` / ``$cheapest-capable``) passes reload validation with the
  requirement deferred to launch, exactly like the shipped ``review.deep``
  default.
* ``launch diversity`` — the distinct-canonical-model count a launch would
  actually get, or ``UNKNOWN`` when any target's binding cannot be resolved.
  An unmet threshold is reported on the route, never folded into a target's
  model error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chartreux.core.dispatch.route_models import PurposeRoute, RouteTarget
from chartreux.core.dispatch.routes import (
    RouteResolutionError,
    distinct_canonical_model_count,
    is_launch_marker,
    resolve_route,
    validate_route_diversity,
)
from chartreux.route_labels import PREVIEW_LABEL as PREVIEW_LABEL, UNKNOWN as UNKNOWN

if TYPE_CHECKING:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
    from chartreux.core.model_catalog.resolver import ModelResolver

# The doctor section's heading: the table it renders was loaded independently
# of any running conversation.
DOCTOR_SECTION_LABEL = "Independently loaded configuration"


@dataclass(frozen=True)
class RouteTargetReport:
    """One configured route target, resolved as far as it can be.

    ``binding`` is the configured model expression, ``marker`` the launch-time
    marker it defers to (``None`` for an explicit binding), and ``preview`` the
    model a launch would resolve today — ``None`` with an ``error`` when the
    binding cannot be resolved. The configured fields (seat, identity, tier,
    overlay) come from the target itself, never from resolution.
    """

    seat: str | None
    identity: str
    tier: str
    overlay: str | None
    binding: str
    marker: str | None
    preview: str | None
    error: str | None

    def summary(self) -> str:
        preview = self.preview if self.preview is not None else UNKNOWN
        fields = [
            f"seat={self.seat or '-'}",
            f"binding={self.binding} -> {preview} ({PREVIEW_LABEL})",
            f"identity={self.identity}",
            f"tier={self.tier}",
            f"overlay={self.overlay or '-'}",
        ]
        if self.marker is not None:
            fields.insert(1, f"marker={self.marker}")
        if self.error is not None:
            fields.append(f"error={self.error}")
        return "target " + " | ".join(fields)


@dataclass(frozen=True)
class RouteReport:
    """One purpose route: both validation facts plus every target."""

    purpose: str
    requires_distinct_models: int
    # None means UNKNOWN: resolution was incomplete, so no count is claimed.
    resolved_distinct_models: int | None
    reload_status: str
    reload_detail: str | None
    # "pass" | "fail" | "unknown" — the launch-time verdict, kept separate
    # from individual target errors.
    launch_status: str
    launch_detail: str | None
    targets: tuple[RouteTargetReport, ...]

    @property
    def resolved_label(self) -> str:
        return (
            UNKNOWN
            if self.resolved_distinct_models is None
            else str(self.resolved_distinct_models)
        )

    def summary(self) -> str:
        fields = [
            f"purpose={self.purpose}",
            f"requires_distinct_models={self.requires_distinct_models}",
            f"resolved_distinct_models={self.resolved_label}",
            f"reload validation={self.reload_status}",
            f"launch diversity={self.launch_status}",
        ]
        if self.reload_detail is not None:
            fields.append(f"reload detail={self.reload_detail}")
        if self.launch_detail is not None:
            fields.append(f"launch detail={self.launch_detail}")
        return "route " + " | ".join(fields)


@dataclass(frozen=True)
class RouteTableReport:
    """The inspected table. ``error`` reports an inspection that cannot start."""

    routes: tuple[RouteReport, ...]
    error: str | None = None


@dataclass(frozen=True)
class RouteInspectionContext:
    """The explicit inputs one inspection reads.

    ``catalog`` holds the selectable model aliases as keys (``resolve_route``
    consults it to reject unavailable bindings); ``catalog_prices`` holds the
    per-base price totals the cost-tier default resolves through.
    """

    routes: Mapping[str, PurposeRoute]
    catalog: Mapping[str, object]
    session_model: str
    canonical_models: Mapping[str, str]
    catalog_prices: Mapping[str, float]

    @classmethod
    def from_config(cls, config: ChartreuxConfigSchema) -> RouteInspectionContext:
        """The table and catalog the configuration itself carries."""
        from chartreux.core.model_catalog.resolver import resolver_for

        return cls.from_resolver(
            resolver_for(config),
            routes=config.purpose_routes.routes,
            session_model=config.active_model or "@main",
            allowed_models=config.allowed_models,
        )

    @classmethod
    def from_resolver(
        cls,
        resolver: ModelResolver,
        *,
        routes: Mapping[str, PurposeRoute],
        session_model: str,
        allowed_models: Sequence[str] = (),
    ) -> RouteInspectionContext:
        """Derive the launch-path route inputs from a catalog resolver."""
        allowed = tuple(allowed_models)
        catalog: dict[str, object] = {
            base: None for base in resolver.eligible_models(allowed)
        }
        catalog.update({f"@{name}": None for name in resolver.eligible_roles(allowed)})
        return cls(
            routes=dict(routes),
            catalog=catalog,
            session_model=session_model,
            canonical_models=resolver.canonical_model_map(),
            catalog_prices=resolver.catalog_prices(allowed),
        )


def inspect_config_routes(config: ChartreuxConfigSchema) -> RouteTableReport:
    """Inspect the purpose routes one configuration carries.

    Never raises for an unattachable catalog: an inspection that cannot start
    reports why in ``error`` so a surface can say so instead of inventing
    per-target failures.
    """
    from chartreux.core.model_catalog.resolver import ModelResolutionError

    try:
        context = RouteInspectionContext.from_config(config)
    except ModelResolutionError as exc:
        return RouteTableReport(routes=(), error=str(exc))
    return inspect_route_table(context)


def inspect_route_table(context: RouteInspectionContext) -> RouteTableReport:
    """Inspect every route independently, in deterministic purpose order."""
    return RouteTableReport(
        routes=tuple(
            _inspect_route(purpose, route, context)
            for purpose, route in sorted(context.routes.items())
        )
    )


def _inspect_route(
    purpose: str, route: PurposeRoute, context: RouteInspectionContext
) -> RouteReport:
    targets = tuple(
        _inspect_target(purpose, target, context) for target in route.targets
    )
    reload_status, reload_detail = _reload_verdict(purpose, route, context)
    count, launch_status, launch_detail = _launch_verdict(
        purpose, route, targets, context
    )
    return RouteReport(
        purpose=purpose,
        requires_distinct_models=route.requires_distinct_models,
        resolved_distinct_models=count,
        reload_status=reload_status,
        reload_detail=reload_detail,
        launch_status=launch_status,
        launch_detail=launch_detail,
        targets=targets,
    )


def _reload_verdict(
    purpose: str, route: PurposeRoute, context: RouteInspectionContext
) -> tuple[str, str | None]:
    """The reload guard's verdict for this route, reported without raising."""
    deferred = all(is_launch_marker(target.model) for target in route.targets)
    try:
        validate_route_diversity(
            purpose,
            route,
            catalog=context.catalog,
            session_model=context.session_model,
            canonical_models=context.canonical_models,
            catalog_prices=context.catalog_prices,
        )
    except RouteResolutionError as exc:
        return "fail", str(exc)
    if deferred and route.requires_distinct_models > 1:
        return (
            "pass",
            "every target binds a launch-time marker "
            "($session / $cheapest-capable); the distinct-model requirement "
            "is validated at launch, not at reload",
        )
    return "pass", None


def _launch_verdict(
    purpose: str,
    route: PurposeRoute,
    targets: tuple[RouteTargetReport, ...],
    context: RouteInspectionContext,
) -> tuple[int | None, str, str | None]:
    """The launch-time diversity verdict, separate from target model errors."""
    resolved = [target.preview for target in targets]
    if any(preview is None for preview in resolved):
        unresolved = sum(preview is None for preview in resolved)
        return (
            None,
            "unknown",
            f"Diversity is {UNKNOWN}: {unresolved} of {len(targets)} target(s) "
            "could not be resolved, so the distinct-model count cannot be decided.",
        )
    names = [preview for preview in resolved if preview is not None]
    count = distinct_canonical_model_count(names, context.canonical_models)
    required = route.requires_distinct_models
    if required <= 1 or count >= required:
        return count, "pass", None
    return (
        count,
        "fail",
        f"Purpose {purpose!r} requires {required} distinct canonical models; "
        f"resolved {count}. Configure distinct models explicitly.",
    )


def _inspect_target(
    purpose: str, target: RouteTarget, context: RouteInspectionContext
) -> RouteTargetReport:
    preview, error = _resolve_target(purpose, target, context)
    return RouteTargetReport(
        seat=target.seat,
        identity=target.identity,
        tier=target.tier.value,
        overlay=target.overlay,
        binding=target.model,
        marker=target.model if is_launch_marker(target.model) else None,
        preview=preview,
        error=error,
    )


def _resolve_target(
    purpose: str, target: RouteTarget, context: RouteInspectionContext
) -> tuple[str | None, str | None]:
    """Resolve one target through the launch path, isolated from its siblings.

    The target is resolved as the sole target of a synthetic single-target
    route: that keeps the launch grammar (marker resolution, the catalog
    availability check, the actionable error text) and makes one target's
    failure independent of every other target's.
    """
    try:
        resolved = resolve_route(
            purpose,
            context.catalog,
            session_model=context.session_model,
            routes={purpose: PurposeRoute(targets=(target,))},
            canonical_models=context.canonical_models,
            catalog_prices=context.catalog_prices,
            require_diversity=False,
        )
    except RouteResolutionError as exc:
        return None, str(exc)
    return resolved.model, None
