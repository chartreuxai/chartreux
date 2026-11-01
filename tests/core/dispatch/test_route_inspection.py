from __future__ import annotations

import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.dispatch.route_inspection import (
    DOCTOR_SECTION_LABEL,
    PREVIEW_LABEL,
    UNKNOWN,
    RouteInspectionContext,
    RouteTableReport,
    inspect_config_routes,
    inspect_route_table,
)
from chartreux.core.dispatch.route_models import PackageTier, PurposeRoute, RouteTarget
from chartreux.core.dispatch.routes import (
    COST_TIER_DEFAULT,
    SESSION_MODEL,
    RouteResolutionError,
    validate_route_table_diversity,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from tests.conftest import build_test_vibe_config, cost_tier_catalog

TWO_TARGET_ROUTE = PurposeRoute(
    targets=(
        RouteTarget(
            model="premium-model",
            identity="reviewer",
            tier=PackageTier.SUBSTANTIVE,
            seat="analytical",
        ),
        RouteTarget(
            model="budget-model",
            identity="coder",
            tier=PackageTier.MECHANICAL,
            seat="execution",
            overlay="review-execution",
        ),
    ),
    requires_distinct_models=2,
)

ALL_MARKER_ROUTE = PurposeRoute(
    targets=(
        RouteTarget(
            model=SESSION_MODEL,
            identity="reviewer",
            tier=PackageTier.SUBSTANTIVE,
            seat="analytical",
        ),
        RouteTarget(
            model=SESSION_MODEL,
            identity="reviewer",
            tier=PackageTier.SUBSTANTIVE,
            seat="peer",
        ),
        RouteTarget(
            model=SESSION_MODEL,
            identity="coder",
            tier=PackageTier.SUBSTANTIVE,
            seat="execution",
            overlay="review-execution",
        ),
    ),
    requires_distinct_models=3,
)


def make_config(
    routes: dict[str, PurposeRoute],
    *,
    active_model: str = "budget-model",
    attach_catalog: bool = True,
) -> ChartreuxConfigSchema:
    config = build_test_vibe_config(
        orchestration={"routes": routes}, active_model=active_model
    )
    if attach_catalog:
        config.attach_catalog_snapshot(CatalogSnapshot(cost_tier_catalog(), "fixture"))
    else:
        object.__setattr__(config, "_catalog_snapshot", None)
    return config


def report_for(
    routes: dict[str, PurposeRoute],
    *,
    active_model: str = "budget-model",
    attach_catalog: bool = True,
) -> RouteTableReport:
    return inspect_config_routes(
        make_config(routes, active_model=active_model, attach_catalog=attach_catalog)
    )


def route(report: RouteTableReport, purpose: str):
    return next(item for item in report.routes if item.purpose == purpose)


def test_two_target_route_reports_every_design_field() -> None:
    report = report_for({"review.custom": TWO_TARGET_ROUTE})
    (inspected,) = report.routes

    assert report.error is None
    assert inspected.purpose == "review.custom"
    assert inspected.requires_distinct_models == 2
    assert inspected.resolved_distinct_models == 2
    assert inspected.reload_status == "pass"
    assert inspected.launch_status == "pass"

    analytical, execution = inspected.targets
    assert analytical.seat == "analytical"
    assert analytical.binding == "premium-model"
    assert analytical.marker is None
    assert analytical.preview == "premium-model"
    assert analytical.identity == "reviewer"
    assert analytical.tier == PackageTier.SUBSTANTIVE.value
    assert analytical.overlay is None
    assert analytical.error is None

    assert execution.seat == "execution"
    assert execution.binding == "budget-model"
    assert execution.preview == "budget-model"
    assert execution.identity == "coder"
    assert execution.tier == PackageTier.MECHANICAL.value
    assert execution.overlay == "review-execution"
    assert execution.error is None


def test_every_binding_preview_is_labeled_a_preview() -> None:
    report = report_for({
        "review.custom": TWO_TARGET_ROUTE,
        "implementation": PurposeRoute(
            targets=(
                RouteTarget(
                    model=COST_TIER_DEFAULT,
                    identity="coder",
                    tier=PackageTier.SUBSTANTIVE,
                ),
            )
        ),
    })
    targets = [target for item in report.routes for target in item.targets]
    assert len(targets) == 3
    for target in targets:
        assert PREVIEW_LABEL in target.summary()
        assert f"binding={target.binding} ->" in target.summary()
    cost_tier = route(report, "implementation").targets[0]
    assert cost_tier.marker == COST_TIER_DEFAULT
    assert cost_tier.preview == "budget-model"


def test_mixed_targets_retain_the_valid_row_and_report_unknown_diversity() -> None:
    mixed = PurposeRoute(
        targets=(
            RouteTarget(
                model="budget-model",
                identity="reviewer",
                tier=PackageTier.SUBSTANTIVE,
                seat="analytical",
            ),
            RouteTarget(
                model="missing-model",
                identity="coder",
                tier=PackageTier.SUBSTANTIVE,
                seat="peer",
            ),
        ),
        requires_distinct_models=2,
    )
    report = report_for({"review.custom": TWO_TARGET_ROUTE, "review.mixed": mixed})

    # Every row survives: one target's failure never removes another route.
    assert [item.purpose for item in report.routes] == ["review.custom", "review.mixed"]
    assert route(report, "review.custom").launch_status == "pass"

    inspected = route(report, "review.mixed")
    valid, invalid = inspected.targets
    assert valid.preview == "budget-model" and valid.error is None
    assert invalid.preview is None
    assert "missing-model" in (invalid.error or "")
    # Incomplete resolution means the count is UNKNOWN, never a claimed value.
    assert inspected.resolved_distinct_models is None
    assert inspected.resolved_label == UNKNOWN
    assert inspected.launch_status == "unknown"
    assert UNKNOWN in (inspected.launch_detail or "")
    # The reload guard is structural and still passes: two distinct bindings.
    assert inspected.reload_status == "pass"


def test_unmet_threshold_is_reported_separately_from_a_model_error() -> None:
    # Two targets bind the same model, so the threshold is decidable and unmet;
    # the third target's binding is unavailable. The two facts never merge.
    mixed = PurposeRoute(
        targets=(
            RouteTarget(
                model="budget-model",
                identity="reviewer",
                tier=PackageTier.SUBSTANTIVE,
                seat="analytical",
            ),
            RouteTarget(
                model="budget-model",
                identity="reviewer",
                tier=PackageTier.SUBSTANTIVE,
                seat="peer",
            ),
            RouteTarget(
                model="missing-model",
                identity="coder",
                tier=PackageTier.SUBSTANTIVE,
                seat="execution",
            ),
        ),
        requires_distinct_models=3,
    )
    inspected = route(report_for({"review.mixed": mixed}), "review.mixed")
    assert inspected.resolved_distinct_models is None
    assert inspected.launch_status == "unknown"
    assert inspected.launch_detail is not None
    assert "missing-model" not in inspected.launch_detail
    failed = [target for target in inspected.targets if target.error is not None]
    assert len(failed) == 1
    assert "missing-model" in (failed[0].error or "")
    assert "distinct canonical models" not in (failed[0].error or "")


def test_all_marker_route_displays_both_facts() -> None:
    inspected = route(report_for({"review.deep": ALL_MARKER_ROUTE}), "review.deep")

    # Reload validation passes: every binding defers to launch.
    assert inspected.reload_status == "pass"
    assert "validated at launch" in (inspected.reload_detail or "")
    # Launch-time diversity fails: the same session model three times.
    assert inspected.resolved_distinct_models == 1
    assert inspected.launch_status == "fail"
    assert "requires 3 distinct canonical models; resolved 1" in (
        inspected.launch_detail or ""
    )
    summary = inspected.summary()
    assert "reload validation=pass" in summary
    assert "launch diversity=fail" in summary


def test_reload_verdict_matches_the_reload_guard() -> None:
    unsatisfiable = PurposeRoute(
        targets=(
            RouteTarget(
                model="budget-model", identity="coder", tier=PackageTier.SUBSTANTIVE
            ),
        ),
        requires_distinct_models=2,
    )
    routes = {"review.custom": TWO_TARGET_ROUTE, "review.bad": unsatisfiable}
    report = report_for(routes)
    assert route(report, "review.custom").reload_status == "pass"
    assert route(report, "review.bad").reload_status == "fail"

    with pytest.raises(RouteResolutionError, match="review.bad"):
        validate_route_table_diversity(
            routes,
            catalog={"budget-model": None, "premium-model": None},
            session_model="budget-model",
        )
    # The all-marker route is the deferral case: the guard lets it through.
    validate_route_table_diversity(
        {"review.deep": ALL_MARKER_ROUTE},
        catalog={"budget-model": None},
        session_model="budget-model",
    )


def test_inspection_does_not_mutate_its_inputs() -> None:
    routes = {"review.custom": TWO_TARGET_ROUTE}
    report = report_for(routes)
    assert routes["review.custom"] is TWO_TARGET_ROUTE
    assert report == report_for(routes)


def test_unattached_catalog_reports_an_inspection_error() -> None:
    report = report_for({"review.custom": TWO_TARGET_ROUTE}, attach_catalog=False)
    assert report.routes == ()
    assert report.error is not None
    assert "catalog" in report.error.lower()


def test_inspection_context_derives_the_launch_inputs() -> None:
    context = RouteInspectionContext.from_config(
        make_config({"review.custom": TWO_TARGET_ROUTE})
    )
    assert context.session_model == "budget-model"
    assert "premium-model" in context.catalog
    assert (
        context.catalog_prices["budget-model"] < context.catalog_prices["premium-model"]
    )
    assert context.canonical_models["budget-model"] == "budget-model"
    assert inspect_route_table(context).routes[0].purpose == "review.custom"


def test_doctor_section_label_is_the_pinned_literal() -> None:
    assert DOCTOR_SECTION_LABEL == "Independently loaded configuration"
