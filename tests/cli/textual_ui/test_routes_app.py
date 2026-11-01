from __future__ import annotations

import pytest

from chartreux.app_server.config import RouteTargetView, RouteView
from chartreux.cli.textual_ui.screens.routes import (
    RoutesList,
    RoutesScreen,
    RoutesSnapshot,
)
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.core.dispatch.route_inspection import PREVIEW_LABEL
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.cli.textual_ui.test_app_server_requests import _wait_until
from tests.conftest import build_test_chartreux_app

MARKER_DETAIL = (
    "every target binds a launch-time marker ($session / $cheapest-capable); "
    "the distinct-model requirement is validated at launch, not at reload"
)
DIVERSITY_DETAIL = (
    "Purpose 'review.deep' requires 3 distinct canonical models; resolved 1. "
    "Configure distinct models explicitly."
)


def target(
    seat: str | None = "analytical",
    *,
    identity: str = "reviewer",
    tier: str = "substantive",
    overlay: str | None = None,
    binding: str = "$session",
    marker: str | None = "$session",
    preview: str | None = "model-b",
    error: str | None = None,
) -> RouteTargetView:
    return RouteTargetView(
        seat=seat,
        identity=identity,
        tier=tier,
        overlay=overlay,
        binding=binding,
        marker=marker,
        preview=preview,
        error=error,
    )


REVIEW_DEEP = RouteView(
    purpose="review.deep",
    requires_distinct_models=3,
    resolved_distinct_models=1,
    reload_status="pass",
    reload_detail=MARKER_DETAIL,
    launch_status="fail",
    launch_detail=DIVERSITY_DETAIL,
    targets=[
        target(),
        target("peer"),
        target("execution", identity="coder", overlay="review-execution"),
    ],
)

REVIEW_MIXED = RouteView(
    purpose="review.mixed",
    requires_distinct_models=2,
    resolved_distinct_models=None,
    reload_status="pass",
    launch_status="unknown",
    launch_detail=(
        "Diversity is UNKNOWN: 1 of 2 target(s) could not be resolved, so the "
        "distinct-model count cannot be decided."
    ),
    targets=[
        target(binding="budget-model", marker=None, preview="budget-model"),
        target(
            "peer",
            identity="coder",
            binding="missing-model",
            marker=None,
            preview=None,
            error=(
                "Model binding 'missing-model' for purpose 'review.mixed' is "
                "unavailable; configure it explicitly."
            ),
        ),
    ],
)


def text(screen: RoutesScreen, selector: str) -> str:
    return str(screen.query_one(selector, NoMarkupStatic).content)


def rows(screen: RoutesScreen) -> list[str]:
    return [str(option.prompt) for option in screen.query_one(RoutesList).options]


@pytest.mark.asyncio
async def test_routes_slash_open_renders_the_live_table_and_dismisses() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        app.config.routes = [REVIEW_DEEP, REVIEW_MIXED]
        composer = app.query_one(ChatInputContainer)
        composer.value = "unfinished draft"
        assert composer.input_widget is not None
        composer.input_widget.focus()
        await app._dispatch_submitted_value("/routes")
        await _wait_until(pilot, lambda: isinstance(app.screen, RoutesScreen))
        screen = app.screen
        assert isinstance(screen, RoutesScreen)
        assert screen.read_routes == app._read_live_routes

        rendered = rows(screen)
        assert any("review.deep" in row for row in rendered)
        assert any(
            "requires 3 distinct" in row and "resolved 1" in row for row in rendered
        )
        # Both validation facts are visible for the all-marker default.
        assert any("reload pass" in row and "launch fail" in row for row in rendered)
        assert any("resolved UNKNOWN" in row for row in rendered)
        # Every preview carries the anti-misleading label.
        assert PREVIEW_LABEL in text(screen, "#routes-heading")

        details = text(screen, "#routes-detail-text")
        assert "Purpose: review.deep" in details
        assert "reload validation pass" in details
        assert "launch diversity fail" in details
        assert "requires 3 distinct canonical models; resolved 1" in details
        assert f"binding $session -> model-b ({PREVIEW_LABEL})" in details
        assert "identity reviewer" in details
        assert "tier substantive" in details
        assert "overlay review-execution" in details

        # The second row reports the per-target model error and UNKNOWN count.
        await pilot.press("down")
        await _wait_until(
            pilot, lambda: "review.mixed" in text(screen, "#routes-detail-text")
        )
        mixed = text(screen, "#routes-detail-text")
        assert "resolved UNKNOWN" in mixed
        assert "launch diversity unknown" in mixed
        assert "error: Model binding 'missing-model'" in mixed

        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._routes_worker is None)
        assert composer.value == "unfinished draft"
        assert app.focused is composer.input_widget


@pytest.mark.asyncio
async def test_routes_refresh_rereads_the_live_source() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        app.config.routes = [REVIEW_DEEP]
        await app._show_routes()
        await _wait_until(pilot, lambda: isinstance(app.screen, RoutesScreen))
        screen = app.screen
        assert isinstance(screen, RoutesScreen)
        assert len(rows(screen)) == 1

        app.config.routes = [REVIEW_DEEP, REVIEW_MIXED]
        await pilot.press("r")
        await _wait_until(pilot, lambda: len(rows(screen)) == 2)
        assert any("review.mixed" in row for row in rows(screen))

        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._routes_worker is None)


@pytest.mark.asyncio
async def test_reload_swap_is_reflected_in_the_open_browser() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        app.config.routes = [REVIEW_DEEP]
        await app._show_routes()
        await _wait_until(pilot, lambda: isinstance(app.screen, RoutesScreen))
        screen = app.screen
        assert isinstance(screen, RoutesScreen)
        assert any("review.deep" in row for row in rows(screen))

        # A reload commits a new table and republishes the config view; the
        # open browser follows the LIVE source without being reopened.
        before = app.config.model_copy(deep=True)
        app.config.routes = [REVIEW_MIXED]
        app.app_server.resources.config.publish_change(before)
        await _wait_until(
            pilot, lambda: any("review.mixed" in row for row in rows(screen))
        )
        assert not any("review.deep" in row for row in rows(screen))
        assert "review.mixed" in text(screen, "#routes-detail-text")

        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._routes_worker is None)


@pytest.mark.asyncio
async def test_routes_duplicate_open_and_read_error_keeps_rows() -> None:
    app = build_test_chartreux_app()
    calls = {"count": 0}

    def flaky() -> RoutesSnapshot:
        calls["count"] += 1
        if calls["count"] == 1:
            return RoutesSnapshot((REVIEW_DEEP,))
        raise RuntimeError("private failure")

    app._read_live_routes = flaky
    async with app.run_test(size=(120, 40)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        await app._show_routes()
        worker = app._routes_worker
        await app._show_routes()
        assert app._routes_worker is worker
        await _wait_until(pilot, lambda: isinstance(app.screen, RoutesScreen))
        screen = app.screen
        assert isinstance(screen, RoutesScreen)
        await _wait_until(pilot, lambda: len(rows(screen)) == 1)
        await app._show_routes()
        assert sum(isinstance(item, RoutesScreen) for item in app.screen_stack) == 1

        await pilot.press("r")
        await _wait_until(pilot, lambda: "Failed:" in text(screen, "#routes-status"))
        assert "private failure" not in text(screen, "#routes-status")
        assert "previous rows retained" in text(screen, "#routes-status")
        assert len(rows(screen)) == 1

        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._routes_worker is None)


@pytest.mark.asyncio
async def test_uninspectable_table_renders_the_reason_without_rows() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        app.config.routes = []
        app.config.routes_error = "No validated model catalog is attached"
        await app._show_routes()
        await _wait_until(pilot, lambda: isinstance(app.screen, RoutesScreen))
        screen = app.screen
        assert isinstance(screen, RoutesScreen)
        await _wait_until(
            pilot, lambda: "Route table unavailable" in text(screen, "#routes-status")
        )
        assert "No validated model catalog is attached" in text(
            screen, "#routes-status"
        )
        assert rows(screen) == []
        assert text(screen, "#routes-detail-text") == ""

        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._routes_worker is None)
