"""Purpose vocabulary and purpose routes: the routing successor surface.

The dispatch slot/preset/schema surface is retired (package 16b): the
``DispatchMode``/``DispatchPolicy``/``DispatchSlot`` models, the shipped
presets, and the legacy session-policy bind are gone (the
``chartreux.core.dispatch.schema``/``presets`` modules no longer exist).
Routing is configured through ``[orchestration.routes.<purpose>]`` — the
purpose routes below — and the conversation posture through the ``[agent]``
identity/workflow/strategy selections.
"""

from __future__ import annotations

from chartreux.core.dispatch.purposes import (
    SHIPPED_PURPOSES,
    PurposeDefinition,
    PurposeId,
    PurposeVocabulary,
)
from chartreux.core.dispatch.route_models import (
    PackageTier,
    PurposeRoute,
    ReplacementCapability,
    RouteTarget,
)
from chartreux.core.dispatch.routes import (
    COST_TIER_DEFAULT,
    SESSION_MODEL,
    ResolvedTarget,
    RouteResolutionError,
    cheapest_capable_model,
    resolve_route,
    shipped_routes,
    validate_canonical_diversity,
)

__all__ = [
    "COST_TIER_DEFAULT",
    "SESSION_MODEL",
    "SHIPPED_PURPOSES",
    "PackageTier",
    "PurposeDefinition",
    "PurposeId",
    "PurposeRoute",
    "PurposeVocabulary",
    "ReplacementCapability",
    "ResolvedTarget",
    "RouteResolutionError",
    "RouteTarget",
    "cheapest_capable_model",
    "resolve_route",
    "shipped_routes",
    "validate_canonical_diversity",
]
