"""Usage browser matrix. Inspect rendered strips before --snapshot-update."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from rich.console import Console
from textual.app import App, ComposeResult
from textual.pilot import Pilot
from textual.widgets import Button, Static

from chartreux.app_server.models import (
    UsageComponentBreakdown,
    UsageComponentSummary,
    UsageCoverageWarning,
    UsageModelSummary,
    UsageWindow,
    UsageWindowSummaries,
    UsageWindowSummary,
)
from chartreux.app_server.protocol import UsageReadResponse, UsageUpdatedParams
from chartreux.cli.textual_ui.screens.usage import (
    UsageDetails,
    UsageModels,
    UsageScreen,
)
from tests.cli.textual_ui.test_usage_screen import FakeUsage, UsageApp, snapshot, text
from tests.snapshots.snap_compare import SnapCompare
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


class UsageSnapshotApp(App[None]):
    """No ledger, backend, clock, or live reconciliation in visual fixtures."""

    def __init__(self, state: str = "ready", *, ascii_chrome: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(ascii_chrome=ascii_chrome)
        self.state = state
        self.calls: list[tuple[UsageWindow, str | None]] = []
        self.browser = UsageScreen(self.read_usage, project_key="snapshot-project")

    def compose(self) -> ComposeResult:
        yield Static("Conversation", id="opener")

    async def on_load(self) -> None:
        install_snapshot_wake()

    def on_mount(self) -> None:
        self.push_screen(self.browser)

    async def read_usage(
        self, window: UsageWindow, project: str | None
    ) -> UsageReadResponse:
        self.calls.append((window, project))
        if self.state == "loading":
            await asyncio.Future()
        if self.state == "error":
            raise RuntimeError("not displayed")
        start = datetime(2026, 6, 15, tzinfo=UTC)
        empty = self.state == "empty"
        unknown = self.state == "unknown"
        incomplete = self.state in {"incomplete", "unknown"}
        summaries = {}
        for name, begin, end in (
            ("day", start, start + timedelta(days=1)),
            ("week", start, start + timedelta(days=7)),
            ("month", start.replace(day=1), start.replace(month=7, day=1)),
        ):
            summaries[name] = UsageWindowSummary(
                start_local=begin,
                end_local=end,
                start_utc=begin,
                end_utc=end,
                timezone="UTC",
                degraded=self.state == "incomplete",
                requests=0 if empty else 10,
                input_tokens=0 if empty or unknown else 120_000,
                output_tokens=0 if empty or unknown else 24_000,
                cached_input_tokens=0 if empty or unknown else 20_000,
                known_cost_usd=0 if empty or unknown else 12.34,
                has_known_cost=not unknown,
                has_unknown_cost=incomplete,
                has_unknown_tokens=incomplete,
            )
        components = []
        for count, cost in ((100_000, 8.0), (20_000, 0.34), (24_000, 4.0)):
            components.append(
                UsageComponentSummary(
                    tokens=0 if empty or unknown else count,
                    known_cost_usd=0 if empty or unknown else cost,
                    has_known_cost=not unknown,
                    has_unknown_cost=incomplete,
                    has_unknown_tokens=incomplete,
                )
            )
        models = []
        for index, cost in enumerate((5.0, 3.0, 2.0, 1.0, 1.34)):
            models.append(
                UsageModelSummary(
                    model=(
                        "very-long-model-identity-with-a-distinguishing-deployment-suffix"
                        "-and-a-provider-specific-release-identifier-that-needs-wrapping"
                        if self.state == "long" and index == 0
                        else f"model-{index + 1}"
                    ),
                    provider="snapshot-provider",
                    wire_name=f"deployment-{index + 1}",
                    requests=2,
                    input_tokens=0 if unknown else 24_000,
                    output_tokens=0 if unknown else 4_800,
                    cached_input_tokens=0 if unknown else 4_000,
                    known_cost_usd=0 if unknown else cost,
                    has_known_cost=not unknown,
                    has_unknown_cost=incomplete and index == 0,
                    has_unknown_tokens=incomplete and index == 0,
                )
            )
        return UsageReadResponse(
            window=window,
            as_of=start.replace(hour=12, minute=34),
            revision=1,
            project_key="snapshot-project",
            summaries=UsageWindowSummaries(**summaries),
            components=UsageComponentBreakdown(
                uncached_input=components[0],
                cached_input=components[1],
                output=components[2],
            ),
            models=[] if empty else models,
            warnings=[UsageCoverageWarning(code="write-failed")]
            if self.state == "incomplete"
            else [],
        )


async def assert_browser(pilot: Pilot, *, cost: str = "$12.34") -> None:
    await pilot.pause()
    screen = cast(UsageScreen, pilot.app.screen)
    content = screen.query_one("#usage-content")
    fullscreen = screen.size.width < 84 or screen.size.height < 28
    assert content.has_class("fullscreen") == fullscreen
    assert text(screen, "#usage-title") == "Usage"
    assert content.border_title == ("" if fullscreen else "Usage")
    assert not screen.query_one(UsageDetails).display
    assert text(screen, "#usage-detail-text") == ""
    rendered = "\n".join(strip.text for strip in screen._compositor.render_strips())
    assert "Usage" in rendered
    assert "Cost (USD)" in rendered
    assert "Only ledger records" not in rendered
    assert "F1 Help" in rendered
    if cost:
        assert cost in rendered
    for name, label in (
        ("day", "Day"),
        ("week", "Week"),
        ("month", "Month"),
        ("all", "All projects"),
        ("current", "Current project"),
    ):
        button = screen.query_one(f"#usage-{name}", Button)
        assert label in str(button.label)
        assert label in rendered
        assert button.region.right <= content.content_region.right
        assert button.region.bottom <= screen.query_one("#usage-heading").region.y
    if cost and screen.snapshot:
        assert cost in str(screen.query_one(UsageModels).options[-1].prompt)
    assert (
        screen.query_one("#usage-hint").region.bottom <= content.content_region.bottom
    )
    assert "Esc" in text(screen, "#usage-hint")
    if screen.snapshot and screen.snapshot.models:
        models = screen.query_one(UsageModels)
        if screen.size == (80, 24):
            assert models.content_size.height >= 5
            for row in screen.snapshot.models:
                assert row.model[:29] in rendered
        for option in models.options:
            assert (
                cell_len(str(option.prompt)) <= models.scrollable_content_region.width
            )
        strips = screen._compositor.render_strips()
        for index, option in enumerate(models.options[: models.content_size.height]):
            assert str(option.prompt).strip() in strips[models.region.y + index].text


@pytest.mark.parametrize(
    "size", [(80, 24), (100, 32), (50, 24), (83, 28), (84, 27), (84, 28)]
)
def test_usage_geometry(snap_compare: SnapCompare, size: tuple[int, int]) -> None:
    app = UsageSnapshotApp()

    async def before(pilot: Pilot) -> None:
        await assert_browser(pilot)
        app.browser.query_one(UsageModels).focus()
        await pilot.pause()
        assert app.browser.query_one(UsageModels).has_focus

    assert snap_compare(app, terminal_size=size, run_before=before)


@pytest.mark.parametrize("window", ["day", "week", "month"])
@pytest.mark.parametrize("scope", ["all", "current"])
def test_usage_selectors(
    snap_compare: SnapCompare, window: UsageWindow, scope: str
) -> None:
    app = UsageSnapshotApp()

    async def before(pilot: Pilot) -> None:
        await assert_browser(pilot)
        await pilot.click(f"#usage-{window}")
        await pilot.click(f"#usage-{scope}")
        # Capture the resting focus treatment, not Textual's timed press flash.
        await pilot.pause(0.4)
        await assert_browser(pilot)
        assert app.calls[-1] == (
            window,
            "snapshot-project" if scope == "current" else None,
        )
        assert app.focused is app.browser.query_one(f"#usage-{scope}")
        assert app.browser.snapshot and app.browser.snapshot.window == window
        assert "(*)" in str(app.browser.query_one(f"#usage-{window}", Button).label)
        assert "(*)" in str(app.browser.query_one(f"#usage-{scope}", Button).label)

    assert snap_compare(app, terminal_size=(80, 24), run_before=before)


@pytest.mark.parametrize(
    "state,cost,status",
    [
        ("loading", "Estimated cost: —", "Loading recorded usage"),
        ("error", "Estimated cost: —", "Failed: Usage read failed"),
        ("empty", "$0.00", "No recorded calls"),
        ("incomplete", "Estimated cost: —", "Incomplete coverage"),
        ("unknown", "Unknown", "Incomplete accounting"),
    ],
)
def test_usage_states(
    snap_compare: SnapCompare, state: str, cost: str, status: str
) -> None:
    app = UsageSnapshotApp(state)

    async def before(pilot: Pilot) -> None:
        expected_cost = (
            ""
            if state in {"loading", "error"}
            else "—"
            if state == "incomplete"
            else cost
        )
        await assert_browser(pilot, cost=expected_cost)
        expected_status = status if state in {"loading", "error"} else ""
        if expected_status:
            assert expected_status in text(app.browser, "#usage-status")
        else:
            assert text(app.browser, "#usage-status") == ""
        if state == "incomplete":
            assert app.browser.snapshot
            assert all(
                summary.degraded
                for summary in (
                    app.browser.snapshot.summaries.day,
                    app.browser.snapshot.summaries.week,
                    app.browser.snapshot.summaries.month,
                )
            )
            total_text = str(app.browser.query_one(UsageModels).options[-1].prompt)
            assert "$12.34" not in total_text and "$0.00" not in total_text
            assert "TOTAL" in total_text and "10" in total_text
        assert app.focused is app.browser.query_one("#usage-day")

    assert snap_compare(app, terminal_size=(80, 24), run_before=before)


@pytest.mark.parametrize("expanded", [False, True])
def test_usage_long_identity(snap_compare: SnapCompare, expanded: bool) -> None:
    app = UsageSnapshotApp("long")

    async def before(pilot: Pilot) -> None:
        await assert_browser(pilot)
        if expanded:
            await pilot.press("d")
            await pilot.pause()
            assert app.browser.query_one(UsageDetails).has_focus
            assert "distinguishing-deployment-suffix" in text(
                app.browser, "#usage-detail-text"
            )
            assert "Known cost: $5.00" in text(app.browser, "#usage-detail-text")
            rendered = "\n".join(
                strip.text for strip in app.screen._compositor.render_strips()
            )
            assert "Known cost: $5.00" in rendered
            assert "release-identifier-that-needs-wrapping" in rendered
            assert "Esc Back" in rendered
        else:
            app.browser.query_one(UsageModels).focus()
            await pilot.pause()
            assert app.browser.query_one(UsageModels).has_focus

    assert snap_compare(app, terminal_size=(80, 24), run_before=before)


def test_usage_new_usage_hint(snap_compare: SnapCompare) -> None:
    app = UsageSnapshotApp()

    async def before(pilot: Pilot) -> None:
        await assert_browser(pilot)
        screen = app.browser
        assert screen.snapshot
        frozen = screen.snapshot.model_dump()
        screen.query_one(UsageModels).focus()
        screen.usage_updated(
            UsageUpdatedParams(
                as_of=screen.snapshot.as_of,
                revision=2,
                summaries=screen.snapshot.summaries,
            )
        )
        await pilot.pause()
        assert screen.snapshot.model_dump() == frozen
        assert text(screen, "#usage-status") == ""
        assert screen.query_one(UsageModels).has_focus

    assert snap_compare(app, terminal_size=(80, 24), run_before=before)


@pytest.mark.parametrize(
    "theme,ascii_chrome",
    [
        ("textual-dark", False),
        ("textual-light", False),
        ("ansi-dark", True),
        ("ansi-light", True),
    ],
)
def test_usage_themes(
    snap_compare: SnapCompare, theme: str, ascii_chrome: bool
) -> None:
    app = UsageSnapshotApp(ascii_chrome=ascii_chrome)
    app.theme = theme

    async def before(pilot: Pilot) -> None:
        await assert_browser(pilot)
        models = app.browser.query_one(UsageModels)
        models.focus()
        await pilot.pause()
        assert models.has_focus
        assert str(models.options[0].prompt).startswith(">" if ascii_chrome else "▸")
        if ascii_chrome:
            assert "Arrows Move" in text(app.browser, "#usage-hint")
            assert "·" not in text(app.browser, "#usage-hint")

    # Match the ANSI export convention used by the Web search snapshots.
    original = Console.export_svg

    def export_with_palette(console: Console, *args: Any, **kwargs: Any) -> str:
        kwargs["theme"] = app.ansi_theme
        return original(console, *args, **kwargs)

    with patch.object(Console, "export_svg", export_with_palette):
        assert snap_compare(app, terminal_size=(100, 32), run_before=before)


@pytest.mark.asyncio
@pytest.mark.parametrize("height", [32, 40])
async def test_details_reclaims_unused_table_rows(height: int) -> None:
    service = FakeUsage()
    service.response = snapshot(count=2)
    service.response.models[0].wire_name = "long-deployment-name-" * 20
    app = UsageApp(service)
    async with app.run_test(size=(100, height)) as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        screen = app.browser
        models = screen.query_one(UsageModels)
        details = screen.query_one(UsageDetails)
        assert models.size.height == models.option_count == 3
        assert details.region.y == models.region.bottom
        assert details.virtual_size.height >= 9
        assert details.size.height == details.virtual_size.height
        assert not details.show_vertical_scrollbar
        rendered = "\n".join(strip.text for strip in screen._compositor.render_strips())
        assert "Known cost: $0.00" in rendered
        assert "Only ledger records" not in rendered
        assert "Local calendar:" not in rendered

        # An explicit refresh commits new row counts even while Details is open.
        service.response = snapshot(revision=2, count=60)
        service.response.models[0].wire_name = "long-deployment-name-" * 20
        await pilot.press("r")
        await pilot.pause()
        assert models.show_vertical_scrollbar
        assert details.size.height == 8
        assert details.show_vertical_scrollbar
        assert details.region.y == models.region.bottom
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert details.size.height == 8
        assert models.size.height >= 6
        assert details.region.y == models.region.bottom
        assert screen.query_one("#usage-hint").region.bottom <= 24

        service.response = snapshot(revision=3, count=2)
        await pilot.press("r")
        await pilot.resize_terminal(100, height)
        await pilot.pause()
        assert models.size.height == 3
        assert details.region.y == models.region.bottom
        assert not models.show_vertical_scrollbar
        assert not details.show_vertical_scrollbar
        assert details.scroll_y == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("lines", [3, 15])
async def test_bounded_details_scrollbar_tracks_actual_overflow(lines: int) -> None:
    service = FakeUsage()
    service.response = snapshot(count=60)
    app = UsageApp(service)
    with patch.object(
        app.browser, "_details_text", return_value="Detail\n" * (lines - 1) + "End"
    ):
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            await pilot.press("d")
            await pilot.pause()
            models = app.browser.query_one(UsageModels)
            details = app.browser.query_one(UsageDetails)
            assert models.show_vertical_scrollbar
            assert details.size.height == 8
            assert details.show_vertical_scrollbar == (lines > 8)
            assert details.region.y == models.region.bottom


@pytest.mark.asyncio
async def test_a12_model_total_details_and_hint_remain_visible() -> None:
    app = UsageApp(FakeUsage())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        models = app.browser.query_one(UsageModels)
        details = app.browser.query_one(UsageDetails)
        strips = app.browser._compositor.render_strips()
        rendered_table = "\n".join(
            strip.text for strip in strips[models.region.y : models.region.bottom]
        )
        for index in range(5):
            assert f"model-{index}" in rendered_table
        assert "TOTAL" in rendered_table
        assert details.region.y == models.region.bottom
        assert details.size.height >= 8
        assert app.browser.query_one("#usage-hint").region.bottom <= 24
