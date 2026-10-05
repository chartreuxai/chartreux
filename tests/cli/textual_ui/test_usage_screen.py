from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.pilot import Pilot
from textual.widgets import Button, Input

from chartreux.app_server.models import (
    UsageCoverageWarning,
    UsageModelSummary,
    UsageWindow,
    UsageWindowSummaries,
    UsageWindowSummary,
)
from chartreux.app_server.protocol import (
    Notification,
    UsageReadParams,
    UsageReadResponse,
    UsageUpdatedParams,
)
from chartreux.app_server.resources import UsageResource
from chartreux.cli.textual_ui.screens.usage import (
    TOTAL_KEY,
    UsageDetails,
    UsageHints,
    UsageModels,
    UsageScreen,
    bounded_number,
    row_key,
    table_values,
    usage_cost,
)
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


def snapshot(
    window: UsageWindow = "day", revision: int = 1, count: int = 5
) -> UsageReadResponse:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    summary = UsageWindowSummary(
        start_local=start,
        end_local=start + timedelta(days=1),
        start_utc=start,
        end_utc=start + timedelta(days=1),
        timezone="UTC",
        requests=count,
        has_known_cost=True,
        known_cost_usd=12.34 if count else 0,
    )
    return UsageReadResponse(
        as_of=start,
        revision=revision,
        window=window,
        summaries=UsageWindowSummaries(day=summary, week=summary, month=summary),
        project_key="project",
        models=[
            UsageModelSummary(
                model=f"model-{i}",
                provider="provider",
                wire_name=f"wire-{i}",
                requests=1,
                has_known_cost=True,
            )
            for i in range(count)
        ],
    )


class FakeUsage:
    def __init__(self) -> None:
        self.calls: list[tuple[UsageWindow, str | None]] = []
        self.response = snapshot()
        self.error = False
        self.pending: list[asyncio.Future[UsageReadResponse]] = []
        self.defer = False
        self.callback: Callable[[UsageUpdatedParams], None] | None = None

    async def read(self, window: UsageWindow, project: str | None) -> UsageReadResponse:
        self.calls.append((window, project))
        if self.defer:
            future: asyncio.Future[UsageReadResponse] = (
                asyncio.get_running_loop().create_future()
            )
            self.pending.append(future)
            return await future
        if self.error:
            raise RuntimeError("private error")
        return self.response.model_copy(update={"window": window}, deep=True)

    def subscribe(
        self, callback: Callable[[UsageUpdatedParams], None]
    ) -> Callable[[], None]:
        self.callback = callback

        def unsubscribe() -> None:
            self.callback = None

        return unsubscribe

    def update(self, revision: int) -> None:
        assert self.callback
        self.callback(
            UsageUpdatedParams(
                as_of=self.response.as_of,
                revision=revision,
                summaries=self.response.summaries,
            )
        )


class UsageApp(App[None]):
    def __init__(self, service: FakeUsage | UsageResource) -> None:
        super().__init__()
        self.config = SimpleNamespace(ascii_chrome=False)
        self.browser = UsageScreen(service.read, service.subscribe)

    def compose(self) -> ComposeResult:
        yield Input(id="composer")

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self.push_screen(self.browser)


def text(screen: UsageScreen, selector: str) -> str:
    return str(screen.query_one(selector, NoMarkupStatic).content)


@pytest.mark.asyncio
async def test_defaults_geometry_and_close_focus() -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.browser
        assert service.calls == [("day", None)]
        assert not screen.current_project and screen.window == "day"
        assert not screen.query_one(UsageDetails).display
        assert text(screen, "#usage-detail-text") == ""
        for removed in (
            "usage-context",
            "usage-disclosure",
            "usage-summary",
            "usage-breakdown",
            "usage-actions",
        ):
            assert not screen.query(f"#{removed}")
        models = screen.query_one(UsageModels)
        assert models.option_count == 6 and models.content_size.height >= 5
        assert screen.query_one("#usage-content").has_class("fullscreen")
        assert "Input" not in text(screen, "#usage-heading")
        await pilot.resize_terminal(100, 32)
        await pilot.pause()
        assert not screen.query_one("#usage-content").has_class("fullscreen")
        await pilot.press("escape")
        await pilot.pause()
        assert app.focused is app.query_one(Input)
        assert service.callback is None


async def click_shortcut(pilot: Pilot, screen: UsageScreen, action: str) -> None:
    hint = screen.query_one(UsageHints)
    start, end, _ = next(target for target in hint.targets if target[2] == action)
    await pilot.click(hint, offset=((start + end) // 2, 0))


@pytest.mark.asyncio
async def test_keyboard_mouse_selectors_details_and_refresh() -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("right", "enter")
        await pilot.pause()
        assert service.calls[-1] == ("week", None)
        await pilot.click("#usage-month")
        await pilot.pause()
        assert service.calls[-1] == ("month", None)
        await pilot.click("#usage-current")
        await pilot.pause()
        assert service.calls[-1] == ("month", "project")
        await pilot.click("#usage-all")
        await pilot.pause()
        assert service.calls[-1] == ("month", None)
        await pilot.press("d")
        await pilot.pause()
        assert app.browser._expanded and app.browser.query_one(UsageDetails).has_focus
        assert "wire-0" in text(app.browser, "#usage-detail-text")
        await pilot.press("d")
        await pilot.pause()
        assert not app.browser._expanded
        before = len(service.calls)
        await click_shortcut(pilot, app.browser, "refresh")
        await pilot.pause()
        assert len(service.calls) == before + 1
        await click_shortcut(pilot, app.browser, "details")
        await pilot.pause()
        assert app.browser._expanded
        await pilot.press("escape")
        await click_shortcut(pilot, app.browser, "close")
        await pilot.pause()
        assert app.focused is app.query_one(Input)


@pytest.mark.asyncio
async def test_live_updates_duplicates_and_reconnect_reset() -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.browser
        models = screen.query_one(UsageModels)
        models.highlighted = 3
        models.focus()
        await pilot.pause()
        key = screen._selected()
        service.response.models.reverse()
        service.response.revision = 2
        service.update(2)
        await pilot.pause()
        assert screen.snapshot and screen.snapshot.revision == 2
        assert screen._selected() == key and models.has_focus
        assert screen.snapshot.models[0].model == "model-4"
        assert not hasattr(screen, "_hint_revision")
        assert text(screen, "#usage-status") == ""
        calls = len(service.calls)
        service.update(2)
        await pilot.pause()
        assert len(service.calls) == calls
        service.response = snapshot(revision=1)
        service.update(1)
        await pilot.pause()
        assert screen.snapshot.revision == 1
        assert len(service.calls) == calls + 1


@pytest.mark.asyncio
async def test_notification_during_read_has_one_follow_up_without_loading() -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        service.defer = True
        service.update(2)
        await pilot.pause()
        assert len(service.pending) == 1
        assert not app.browser.query_one("#usage-status").display
        service.update(3)
        service.update(3)
        service.update(4)
        await pilot.pause()
        assert len(service.pending) == 1
        service.pending[0].set_result(snapshot(revision=2))
        await pilot.pause()
        assert len(service.pending) == 2
        service.pending[1].set_result(snapshot(revision=4))
        await pilot.pause()
        assert app.browser.snapshot and app.browser.snapshot.revision == 4
        assert len(service.calls) == 3
        assert text(app.browser, "#usage-status") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("close_key", ["d", "escape"])
async def test_in_flight_freeze_and_silent_catch_up(close_key: str) -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.browser
        original = screen.snapshot
        service.defer = True
        service.update(2)
        await pilot.pause()
        await pilot.press("d")
        service.pending[0].set_result(snapshot(revision=2))
        await pilot.pause()
        assert screen.snapshot is original and screen._dirty
        service.update(3)
        await pilot.pause()
        assert len(service.pending) == 1
        await pilot.press(close_key)
        await pilot.pause()
        assert len(service.pending) == 2
        assert text(screen, "#usage-status") == ""
        service.pending[1].set_result(snapshot(revision=3))
        await pilot.pause()
        assert screen.snapshot and screen.snapshot.revision == 3
        assert not screen._dirty and text(screen, "#usage-status") == ""


@pytest.mark.asyncio
async def test_manual_refresh_and_selector_commit_in_details_preserve_scroll() -> None:
    service = FakeUsage()
    service.response.models[0].wire_name = "long-deployment-name-" * 30
    app = UsageApp(service)
    async with app.run_test(size=(50, 24)) as pilot:
        await pilot.pause()
        screen = app.browser
        await pilot.press("d")
        pane = screen.query_one(UsageDetails)
        pane.scroll_to(y=3, animate=False, immediate=True)
        await pilot.pause()
        offset = pane.scroll_y
        assert offset > 0
        service.response.models[0].requests = 123
        service.response.revision = 2
        service.update(2)
        await pilot.pause()
        assert len(service.calls) == 1
        assert "Requests: 123" not in text(screen, "#usage-detail-text")
        await pilot.press("r")
        await pilot.pause()
        assert "Requests: 123" in text(screen, "#usage-detail-text")
        assert pane.scroll_y == offset and screen._expanded
        await pilot.click("#usage-week")
        await pilot.pause()
        assert screen.snapshot and screen.snapshot.window == "week"
        assert marked(screen) == {"usage-week", "usage-all"}
        assert pane.scroll_y == offset
        await pilot.press("escape")
        await pilot.pause()
        assert len(service.calls) == 3


@pytest.mark.asyncio
async def test_automatic_failure_visible_without_retry_loop() -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        original = app.browser.snapshot
        service.error = True
        service.update(2)
        await pilot.pause()
        assert app.browser.snapshot is original
        assert "previous snapshot retained · r Retry" in text(
            app.browser, "#usage-status"
        )
        await pilot.pause()
        assert len(service.calls) == 2


@pytest.mark.asyncio
async def test_real_resource_read_publication_and_reconnect_do_not_loop() -> None:
    service = FakeUsage()

    async def request(method: str, params: UsageReadParams) -> dict[str, object]:
        assert method == "usage/read"
        response = await service.read(params.window, params.project_key)
        return response.model_dump(mode="json")

    client = MagicMock()
    client.request = AsyncMock(side_effect=request)
    connection = MagicMock()
    connection.connect_host = AsyncMock(return_value=client)
    resource = UsageResource(connection)
    app = UsageApp(resource)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(service.calls) == 1  # Initial read publishes before returning.
        service.response = snapshot(revision=2, count=8)
        update = UsageUpdatedParams(
            as_of=service.response.as_of,
            revision=2,
            summaries=service.response.summaries,
        )
        notification = Notification(
            method=update.NOTIFICATION_METHOD, params=update.model_dump(mode="json")
        )
        await resource.consume_notification(notification)
        await pilot.pause()
        assert app.browser.snapshot and len(app.browser.snapshot.models) == 8
        assert len(service.calls) == 2
        await resource.consume_notification(notification)
        await pilot.pause()
        assert len(service.calls) == 2
        service.response = snapshot(revision=1, count=3)
        await resource.refresh_after_reconnect()
        await pilot.pause()
        assert app.browser.snapshot.revision == 1
        assert len(app.browser.snapshot.models) == 3
        assert len(service.calls) == 4  # Reconnect read plus one browser read.
        await pilot.pause()
        assert len(service.calls) == 4
        service.response = snapshot(revision=1, count=7)
        service.response.models[0].model = "restarted-host-model"
        await resource.refresh_after_reconnect()
        await pilot.pause()
        assert app.browser.snapshot.revision == 1
        assert app.browser.snapshot.summaries.day.requests == 7
        assert app.browser.snapshot.models[0].model == "restarted-host-model"
        assert app.browser.query_one(UsageModels).option_count == 8
        rendered = "\n".join(
            strip.text for strip in app.browser._compositor.render_strips()
        )
        assert "restarted-host-model" in rendered
        assert len(service.calls) == 6  # Same revision still causes a browser read.
        await pilot.pause()
        assert len(service.calls) == 6
        service.defer = True
        resource._publish(update)
        await pilot.pause()
        assert len(service.pending) == 1
        resource._publish(update.model_copy(update={"revision": 3}))
        resource._publish(update.model_copy(update={"revision": 4}))
        service.pending[0].set_result(snapshot(revision=2))
        await pilot.pause()
        assert len(service.pending) == 2
        service.pending[1].set_result(snapshot(revision=4))
        await pilot.pause()
        assert app.browser.snapshot.revision == 4
        assert len(service.calls) == 8


@pytest.mark.asyncio
async def test_manual_refresh_supersedes_frozen_automatic_read() -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        service.defer = True
        service.update(2)
        await pilot.pause()
        await pilot.press("d", "r")
        await pilot.pause()
        assert len(service.pending) == 2
        response = snapshot(revision=3)
        response.models[0].requests = 456
        service.pending[1].set_result(response)
        await pilot.pause()
        assert "Requests: 456" in text(app.browser, "#usage-detail-text")
        service.pending[0].set_result(snapshot(revision=2))
        await pilot.pause()
        assert app.browser.snapshot and app.browser.snapshot.revision == 3
        assert not app.browser._dirty
        await pilot.press("escape")
        await pilot.pause()
        assert len(service.calls) == 3


@pytest.mark.asyncio
async def test_live_many_row_anchor_selection_fallback_and_resize() -> None:
    service = FakeUsage()
    service.response = snapshot(count=60)
    app = UsageApp(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.browser
        models = screen.query_one(UsageModels)
        models.focus()
        models.highlighted = 25
        await pilot.pause()
        models.scroll_to(y=20, animate=False, immediate=True)
        await pilot.pause()
        selected = screen._selected()
        anchor = models.options[int(models.scroll_y)].id
        assert models.scroll_y == 20
        inserted = (
            snapshot(count=1)
            .models[0]
            .model_copy(update={"model": "inserted", "wire_name": "inserted"})
        )
        for revision in range(2, 6):
            rows = service.response.models
            if revision == 2:
                rows.insert(0, inserted)
            elif revision == 3:
                rows.pop(1)
            elif revision == 4:
                rows[10:30] = reversed(rows[10:30])
            service.response.revision = revision
            service.update(revision)
            await pilot.pause()
            assert screen._selected() == selected
            assert models.options[int(models.scroll_y)].id == anchor
        await pilot.resize_terminal(100, 32)
        await pilot.pause()
        assert screen._selected() == selected
        assert models.options[int(models.scroll_y)].id == anchor
        old_anchor_index = int(models.scroll_y)
        old_selected_index = models.highlighted
        service.response.models = [
            row
            for row in service.response.models
            if row_key(row) not in {anchor, selected}
        ]
        service.response.revision = 6
        service.update(6)
        await pilot.pause()
        assert models.highlighted == old_selected_index
        assert models.scroll_y == old_anchor_index
        models.highlighted = models.option_count - 1
        await pilot.pause()
        service.response.revision = 7
        service.update(7)
        await pilot.pause()
        assert screen._selected() == TOTAL_KEY
        service.response = snapshot(revision=8, count=0)
        service.update(8)
        await pilot.pause()
        assert screen._selected() == TOTAL_KEY and models.scroll_y == 0


@pytest.mark.asyncio
async def test_stale_read_and_close_during_read() -> None:
    service = FakeUsage()
    service.defer = True
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "Loading" in text(app.browser, "#usage-status")
        await pilot.click("#usage-month")
        await pilot.pause()
        service.pending[1].set_result(snapshot("month", 2))
        await pilot.pause()
        service.pending[0].set_result(snapshot("day", 1))
        await pilot.pause()
        assert app.browser.snapshot and app.browser.snapshot.window == "month"
        await pilot.press("r")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert service.pending[-1].cancelled()
        assert app.focused is app.query_one(Input)


@pytest.mark.asyncio
async def test_error_empty_incomplete_and_unknown_states() -> None:
    service = FakeUsage()
    service.error = True
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "Failed:" in text(app.browser, "#usage-status")
        assert "private error" not in text(app.browser, "#usage-status")
        service.error = False
        service.response = snapshot(count=0)
        await pilot.press("r")
        await pilot.pause()
        assert text(app.browser, "#usage-status") == ""
        assert "$0.00" in str(app.browser.query_one(UsageModels).options[-1].prompt)
        service.response = snapshot()
        service.response.warnings = [UsageCoverageWarning(code="write-failed")]
        service.response.summaries.day.has_unknown_cost = True
        service.response.models[0].has_unknown_tokens = True
        await pilot.press("r")
        await pilot.pause()
        assert text(app.browser, "#usage-status") == ""
        assert "$12.34+" in str(app.browser.query_one(UsageModels).options[-1].prompt)
        await pilot.press("d")
        await pilot.pause()
        assert "write-failed" in text(app.browser, "#usage-detail-text")
        assert "Incomplete accounting" in text(app.browser, "#usage-detail-text")
        await pilot.press("escape")
        service.response.summaries.day.has_known_cost = False
        await pilot.press("r")
        await pilot.pause()
        assert "Unknown" in str(app.browser.query_one(UsageModels).options[-1].prompt)
        service.response.summaries.day.state = "unavailable"
        await pilot.press("r")
        await pilot.pause()
        assert "Unavailable" in text(app.browser, "#usage-status")
        assert "—" in str(app.browser.query_one(UsageModels).options[-1].prompt)
        await pilot.press("d")
        assert "Only ledger records" not in text(app.browser, "#usage-detail-text")


@pytest.mark.asyncio
async def test_keyboard_scope_and_model_list_boundaries() -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("tab", "right", "enter")
        await pilot.pause()
        assert service.calls[-1] == ("day", "project")
        await pilot.press("left", "enter")
        await pilot.pause()
        assert service.calls[-1] == ("day", None)
        models = app.browser.query_one(UsageModels)
        models.focus()
        await pilot.press("down", "down", "down", "down")
        await pilot.pause()
        assert models.highlighted == 4
        await pilot.press("down")
        await pilot.pause()
        assert models.highlighted == 5 and models.has_focus
        assert models.highlighted_option and models.highlighted_option.id == TOTAL_KEY
        await pilot.press("down")
        await pilot.pause()
        assert models.has_focus and models.highlighted == 5
        models.highlighted = 0
        await pilot.press("up")
        assert models.has_focus and models.highlighted == 0
        await pilot.pause()
        assert models.has_focus
        await pilot.press("f1")
        await pilot.pause()
        assert "Current project is unavailable" in text(
            app.browser, "#usage-detail-text"
        )
        await pilot.press("escape")
        assert not app.browser._expanded


@pytest.mark.asyncio
async def test_no_attached_project_and_duplicate_model_deployments() -> None:
    service = FakeUsage()
    service.response.project_key = None
    service.response.models[1].model = service.response.models[0].model
    service.response.models[1].provider = service.response.models[0].provider
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.browser.query_one("#usage-current").disabled
        models = app.browser.query_one(UsageModels)
        assert models.options[0].id != models.options[1].id
        models.highlighted = 1
        models.focus()
        await pilot.press("d")
        await pilot.pause()
        assert "wire-1" in text(app.browser, "#usage-detail-text")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 32), (80, 24)])
async def test_summary_columns_and_cell_aware_identity(size: tuple[int, int]) -> None:
    service = FakeUsage()
    for row in service.response.models[:2]:
        row.model = "界" * 20
        row.provider = "same-provider"
    app = UsageApp(service)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.browser
        assert "$12.34" in str(screen.query_one(UsageModels).options[-1].prompt)
        heading = text(screen, "#usage-heading")
        models = screen.query_one(UsageModels)
        prompt = str(models.options[0].prompt)
        assert "…" in prompt
        assert cell_len(prompt) == cell_len(heading)
        # Each numeric column ends at the same terminal cell as its heading.
        row_ends = [
            cell_len(prompt[: match.end()]) for match in re.finditer(r"\S+", prompt)
        ][2:]
        labels = (
            ["Requests", "Input", "Output", "Cached", "Cost (USD)"]
            if size[0] == 100
            else ["Requests", "Cost (USD)"]
        )
        heading_ends = [
            cell_len(heading[: heading.index(label) + len(label)]) for label in labels
        ]
        assert row_ends[-len(labels) :] == heading_ends
        models.highlighted = 1
        await pilot.press("d")
        await pilot.pause()
        assert "wire-1" in text(screen, "#usage-detail-text")
        assert "界" * 20 in text(screen, "#usage-detail-text")


@pytest.mark.asyncio
async def test_details_availability_and_state_aware_errors() -> None:
    service = FakeUsage()
    service.defer = True
    app = UsageApp(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.browser
        models = screen.query_one(UsageModels)
        assert "Loading" in text(screen, "#usage-status")
        await pilot.press("d")
        assert screen._expanded
        assert "loading" in text(screen, "#usage-detail-text")
        assert "Only ledger records" not in text(screen, "#usage-detail-text")
        await pilot.press("shift+tab")
        assert models.has_focus and screen._expanded
        await pilot.press("escape")
        assert models.has_focus and not screen._expanded
        await pilot.press("enter")
        assert screen._expanded and screen.query_one(UsageDetails).has_focus
        await pilot.press("d")
        assert models.has_focus and not screen._expanded
        service.pending[0].set_exception(RuntimeError("private"))
        await pilot.pause()
        status = text(screen, "#usage-status")
        assert "no snapshot" in status and "retained" not in status
        assert "r Retry" in status and "private" not in status
        await pilot.press("d")
        assert status in text(screen, "#usage-detail-text")
        await pilot.press("escape", "r")
        await pilot.pause()
        service.pending[1].set_result(snapshot(count=0))
        await pilot.pause()
        assert text(screen, "#usage-status") == ""
        assert not screen.query_one("#usage-status").display
        assert models.option_count == 1
        await pilot.press("d")
        assert "TOTAL" in text(screen, "#usage-detail-text")
        await pilot.press("escape", "r")
        await pilot.pause()
        service.pending[2].set_exception(RuntimeError("private"))
        await pilot.pause()
        assert "previous snapshot retained · r Retry" in text(screen, "#usage-status")
        assert screen.snapshot and screen.snapshot.summaries.day.requests == 0
        await pilot.press("d")
        assert "previous snapshot retained" in text(screen, "#usage-detail-text")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (50, 24)])
async def test_below_table_details_lifecycle_and_visible_model_budget(
    size: tuple[int, int],
) -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.browser
        models = screen.query_one(UsageModels)
        models.focus()
        await pilot.press("enter")
        await pilot.pause()
        pane = screen.query_one(UsageDetails)
        assert pane.has_focus
        # Details uses its content height when it fits, otherwise available space.
        details_height = 8
        assert pane.size.height == details_height
        assert pane.max_scroll_y == 0
        assert models.display and models.content_size.height >= 6
        strips = screen._compositor.render_strips()
        table_text = "\n".join(
            strip.text for strip in strips[models.region.y : models.region.bottom]
        )
        # Five actual MODEL rows, excluding the independently visible TOTAL.
        for index in range(5):
            assert f"model-{index}" in table_text
        assert "TOTAL" in table_text
        assert pane.region.y >= models.region.bottom
        assert screen.query_one(UsageHints).region.bottom <= 24
        assert "d Hide details" in text(screen, "#usage-hint")
        height = models.size.height
        await pilot.press("down", "down")
        assert pane.scroll_y == 0
        await pilot.press("shift+tab", "down")
        await pilot.pause()
        assert models.has_focus and screen._expanded
        assert "wire-1" in text(screen, "#usage-detail-text")
        assert models.size.height == height and pane.size.height == details_height
        models.highlighted = 5
        await pilot.pause()
        assert "TOTAL" in text(screen, "#usage-detail-text")
        await pilot.press("d")
        assert models.has_focus and models.highlighted == 5
        assert not pane.display and text(screen, "#usage-detail-text") == ""
        assert "d Details" in text(screen, "#usage-hint")
        assert "Inspect" not in text(screen, "#usage-hint")
        assert not hasattr(screen, "action_inspect")


@pytest.mark.parametrize(
    "value,expected",
    [
        (999, "999"),
        (1000, "1.0K"),
        (12_399, "12.3K"),
        (999_999, "999.9K"),
        (1_000_000, "1.0M"),
        (4_599_999, "4.5M"),
        (999_999_999, "999.9M"),
        (1_000_000_000, "1.0B"),
        (2_199_999_999, "2.1B"),
    ],
)
def test_bounded_decimal_boundaries(value: int, expected: str) -> None:
    assert bounded_number(value, 9) == expected


@pytest.mark.parametrize(
    "value,width,expected",
    [
        (12_399, 8, "$12.3K+"),
        (12_399, 6, "$12K+"),
        (1.239, 12, "$1.23+"),
        (999.999, 12, "$999.99+"),
        (1.999e30, 12, "$1.9e30+"),
    ],
)
def test_bounded_costs_truncate_and_preserve_affixes(
    value: float, width: int, expected: str
) -> None:
    result = bounded_number(value, width, prefix="$", incomplete=True)
    assert result == expected and cell_len(result) <= width


def test_scientific_only_after_billion_cannot_fit() -> None:
    assert bounded_number(10**15, 9) == "1000000B"
    assert bounded_number(19 * 10**29, 8) == "1.9e30"
    with pytest.raises(ValueError, match="cannot retain"):
        bounded_number(10**30, 4, prefix="$", incomplete=True)


@pytest.mark.parametrize("value", [1.239, 12_399.999, 1.999e30, 0.000001239])
def test_details_cost_preserves_unscaled_wire_precision(value: float) -> None:
    from decimal import Decimal

    summary = snapshot().summaries.day
    summary.known_cost_usd = value
    summary.has_unknown_cost = True
    result = usage_cost(summary)
    assert result.startswith("$") and result.endswith("+")
    assert Decimal(result[1:-1]) == Decimal(str(value))
    assert not any(marker in result for marker in ("K", "M", "B", "e"))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 32)])
async def test_total_uses_displayed_summary_and_opens_exact_details(
    size: tuple[int, int],
) -> None:
    service = FakeUsage()
    summary = service.response.summaries.day
    summary.requests = 12_399
    summary.input_tokens = 4_599_999
    summary.output_tokens = 2_199_999_999
    summary.cached_input_tokens = 999_999
    summary.known_cost_usd = 12_399.999
    summary.has_unknown_cost = summary.has_unknown_tokens = True
    # Deliberately inconsistent model sums and requested-window summary.
    service.response.summaries.week = summary.model_copy(update={"requests": 888})
    app = UsageApp(service)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.browser
        screen.window = "week"
        screen._render_snapshot()
        models = screen.query_one(UsageModels)
        prompt = str(models.options[-1].prompt)
        assert models.options[-1].id == TOTAL_KEY
        assert "12.3K" in prompt and "$12.3K+" in prompt
        assert "888" not in prompt
        assert cell_len(prompt) == cell_len(text(screen, "#usage-heading"))
        models.highlighted = models.option_count - 1
        models.focus()
        await pilot.press("enter")
        await pilot.pause()
        details = text(screen, "#usage-detail-text")
        for expected in (
            "TOTAL",
            "Requests: 12,399",
            "Input: 4,599,999+",
            "Output: 2,199,999,999+",
            "Cached (subset of input): 999,999+",
            "$12399.999+",
            "Uncached input:",
        ):
            assert expected in details
        assert_no_snapshot_metadata(details)
        assert "Cost (USD)" in text(screen, "#usage-heading")
        assert screen._expanded and screen.query_one(UsageDetails).has_focus


def assert_no_snapshot_metadata(details: str) -> None:
    for removed in (
        "period/scope:",
        "Local calendar:",
        "Timezone:",
        "Currency:",
        "Snapshot as of:",
        "Only ledger records",
        "Recorded usage only",
        "sessions before the ledger",
    ):
        assert removed not in details


@pytest.mark.asyncio
async def test_model_details_include_requests_exact_cost_and_identity() -> None:
    service = FakeUsage()
    row = service.response.models[0]
    row.requests = 12_399
    row.input_tokens = 4_599_999
    row.known_cost_usd = 1.239
    row.has_unknown_cost = True
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")
        details = text(app.browser, "#usage-detail-text")
        assert "Requests: 12,399" in details and "Input: 4,599,999" in details
        assert "Known cost: $1.239+" in details
        assert "Provider: provider" in details and "Wire name: wire-0" in details
        assert "Incomplete accounting" in details
        assert_no_snapshot_metadata(details)
        await pilot.press("f1")
        assert_no_snapshot_metadata(text(app.browser, "#usage-detail-text"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["degraded", "unknown", "unavailable", "loading", "empty"]
)
async def test_total_preserves_accounting_states(state: str) -> None:
    service = FakeUsage()
    summary = service.response.summaries.day
    if state == "degraded":
        summary.degraded = True
    elif state == "unknown":
        summary.has_known_cost = False
        summary.has_unknown_cost = summary.has_unknown_tokens = True
    elif state in {"unavailable", "loading"}:
        summary.state = "unavailable" if state == "unavailable" else "loading"
    else:
        service.response = snapshot(count=0)
        summary = service.response.summaries.day
    app = UsageApp(service)
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        models = app.browser.query_one(UsageModels)
        assert models.options[-1].id == TOTAL_KEY
        prompt = str(models.options[-1].prompt)
        expected = table_values(summary, narrow=False)
        for cell in expected:
            assert cell in prompt
        if state == "unknown":
            assert "Unknown" in prompt and "0+" in prompt and "$0" not in prompt
        elif state in {"degraded", "unavailable", "loading"}:
            assert "—" in prompt and "$0" not in prompt
        else:
            assert "$0.00" in prompt
        models.highlighted = models.option_count - 1
        models.focus()
        await pilot.press("enter")
        assert app.browser._expanded
        assert "TOTAL" in text(app.browser, "#usage-detail-text")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 32), (80, 24), (50, 24)])
async def test_shortcut_pointer_targets_and_help_discovery(
    size: tuple[int, int],
) -> None:
    service = FakeUsage()
    service.response.project_key = None
    app = UsageApp(service)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.browser
        assert screen.query_one("#usage-current").tooltip == "No attached project"
        assert "F1 Help" in "\n".join(
            strip.text for strip in screen._compositor.render_strips()
        )
        before = len(service.calls)
        await click_shortcut(pilot, screen, "refresh")
        await pilot.pause()
        assert len(service.calls) == before + 1
        await click_shortcut(pilot, screen, "details")
        await pilot.pause()
        assert "No attached project" in text(screen, "#usage-detail-text")
        assert "F1 Help" in "\n".join(
            strip.text for strip in screen._compositor.render_strips()
        )
        await click_shortcut(pilot, screen, "close")
        await pilot.pause()
        assert not screen._expanded and screen.query_one(UsageModels).has_focus
        await pilot.press("f1")
        await pilot.pause()
        assert "No attached project" in text(screen, "#usage-detail-text")
        assert "d: Details" in text(screen, "#usage-detail-text")
        assert "Inspect" not in text(screen, "#usage-detail-text")
        await pilot.press("escape")
        await click_shortcut(pilot, screen, "close")
        await pilot.pause()
        assert app.focused is app.query_one(Input)


@pytest.mark.asyncio
async def test_unavailable_status_is_not_read_failure_and_larger_table_budget() -> None:
    service = FakeUsage()
    service.response.summaries.day.state = "unavailable"
    app = UsageApp(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.browser
        status = screen.query_one("#usage-status")
        assert status.size.height == 1
        assert "Unavailable" in text(screen, "#usage-status")
        assert "Failed" not in text(screen, "#usage-status")
        await pilot.press("d")
        assert "Snapshot state: unavailable" in text(screen, "#usage-detail-text")
        models = screen.query_one(UsageModels)
        pane = screen.query_one(UsageDetails)
        table_height = models.size.height
        details_height = pane.size.height
        assert table_height == models.option_count
        await pilot.resize_terminal(100, 32)
        await pilot.pause()
        assert models.size.height == table_height
        assert pane.size.height == details_height
        assert pane.size.height >= screen.query_one("#usage-detail-text").size.height
        assert pane.max_scroll_y == 0
        await pilot.resize_terminal(80, 20)
        await pilot.pause()
        assert models.size.height == table_height
        assert pane.size.height == details_height
        assert pane.max_scroll_y == 0
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert models.size.height == table_height
        assert pane.size.height == details_height
        assert pane.max_scroll_y == 0


def marked(screen: UsageScreen) -> set[str | None]:
    return {button.id for button in screen.query(Button) if "(*)" in str(button.label)}


@pytest.mark.asyncio
async def test_group_navigation_remembers_candidates_and_table_position() -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.browser
        await pilot.press("left")
        assert app.focused and app.focused.id == "usage-day"
        await pilot.press("right", "right", "right")
        assert app.focused and app.focused.id == "usage-month"
        await pilot.press("tab", "left")
        assert app.focused and app.focused.id == "usage-all"
        await pilot.press("right", "right")
        assert app.focused and app.focused.id == "usage-current"
        await pilot.press("tab", "down", "down")
        models = screen.query_one(UsageModels)
        assert models.has_focus and models.highlighted == 2
        await pilot.press("tab")
        assert app.focused and app.focused.id == "usage-month"
        await pilot.press("shift+tab")
        assert models.has_focus and models.highlighted == 2
        await pilot.press("shift+tab")
        assert app.focused and app.focused.id == "usage-current"
        await pilot.press("shift+tab")
        assert app.focused and app.focused.id == "usage-month"
        await pilot.press("d", "tab")
        assert app.focused and app.focused.id == "usage-month"
        await pilot.press("tab")
        assert app.focused and app.focused.id == "usage-current"
        await pilot.press("tab", "tab")
        assert screen.query_one(UsageDetails).has_focus
        await pilot.press("shift+tab")
        assert models.has_focus and models.highlighted == 2 and screen._expanded
        await pilot.press("tab", "tab", "shift+tab")
        assert screen.query_one(UsageDetails).has_focus
        assert service.calls == [("day", None)]
        assert marked(screen) == {"usage-day", "usage-all"}


@pytest.mark.asyncio
async def test_disabled_scope_skipped_and_space_activates_selectors() -> None:
    service = FakeUsage()
    service.response.project_key = None
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("tab", "right")
        assert app.focused and app.focused.id == "usage-all"
        await pilot.press("tab")
        assert app.browser.query_one(UsageModels).has_focus
        await pilot.press("shift+tab")
        assert app.focused and app.focused.id == "usage-all"
        assert service.calls == [("day", None)]
        await pilot.press("space")
        await pilot.pause()
        assert service.calls == [("day", None), ("day", None)]
        await pilot.press("shift+tab", "right", "space")
        await pilot.pause()
        assert service.calls[-1] == ("week", None)
        await pilot.press("right", "enter")
        await pilot.pause()
        assert service.calls[-1] == ("month", None)


@pytest.mark.asyncio
async def test_requested_markers_commit_only_on_accepted_reads() -> None:
    service = FakeUsage()
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.browser
        original = screen.snapshot
        models = screen.query_one(UsageModels)
        original_rows = [str(option.prompt) for option in models.options]
        service.defer = True
        await pilot.press("right", "right", "space")
        await pilot.pause()
        assert screen.window == "month" and screen.snapshot is original
        assert marked(screen) == {"usage-day", "usage-all"}
        assert "Month / All projects" in text(screen, "#usage-status")
        await pilot.press("tab", "right", "enter")
        await pilot.pause()
        assert "Month / Current project" in text(screen, "#usage-status")
        service.pending[1].set_exception(RuntimeError("private"))
        await pilot.pause()
        assert marked(screen) == {"usage-day", "usage-all"}
        assert screen.snapshot is original
        assert [str(option.prompt) for option in models.options] == original_rows
        service.pending[0].set_result(snapshot("month"))
        await pilot.pause()
        assert marked(screen) == {"usage-day", "usage-all"}  # stale read
        await pilot.press("r")
        await pilot.pause()
        response = snapshot("month")
        response.summaries.month.state = "unavailable"
        service.pending[2].set_result(response)
        await pilot.pause()
        assert marked(screen) == {"usage-month", "usage-current"}
        assert screen.snapshot and screen.snapshot.window == "month"
        assert "Unavailable" in text(screen, "#usage-status")
        assert "—" in str(models.options[-1].prompt)


@pytest.mark.asyncio
async def test_initial_markers_and_group_help() -> None:
    service = FakeUsage()
    service.defer = True
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.browser
        assert screen.snapshot is None and marked(screen) == set()
        assert "Day / All projects" in text(screen, "#usage-status")
        service.pending[0].set_exception(RuntimeError("private"))
        await pilot.pause()
        assert marked(screen) == set()
        await pilot.press("f1")
        help_text = text(screen, "#usage-detail-text")
        for expected in (
            "Tab/Shift+Tab: move between groups",
            "Left/Right: move within selectors",
            "Up/Down: move through model rows and TOTAL",
            "Enter/Space: activate a selector",
            "Navigation alone never fetches",
            "(*) marks the displayed snapshot",
            "d: Details",
            "shortcut row offers pointer actions",
            "F1: Help",
        ):
            assert expected in help_text
        assert "Inspect" not in help_text
        assert (
            "Status-line Today/Week/Month spend is recorded across all projects; "
            "the Current project filter here does not change it."
        ) in help_text


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 32), (80, 24)])
async def test_details_height_stays_fixed_while_navigating_wrapped_rows(
    size: tuple[int, int],
) -> None:
    service = FakeUsage()
    service.response = snapshot(count=2)
    service.response.models[1].model = "long-model-identity-" * 40
    service.response.models[1].wire_name = "long-deployment-" * 40
    app = UsageApp(service)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.browser
        await pilot.press("d")
        await pilot.pause()
        models = screen.query_one(UsageModels)
        pane = screen.query_one(UsageDetails)
        height = pane.size.height
        table_region = models.region
        assert pane.max_scroll_y == 0
        assert "r Refresh" in text(screen, "#usage-hint")
        await pilot.press("shift+tab", "down")
        await pilot.pause()
        assert models.highlighted == 1
        assert pane.size.height == height and models.region == table_region
        assert pane.max_scroll_y > 0 and pane.show_vertical_scrollbar
        await pilot.press("tab", "down", "down")
        await pilot.pause()
        assert pane.scroll_y > 0 and pane.size.height == height
        await pilot.press("shift+tab", "up")
        await pilot.pause()
        assert pane.size.height == height and pane.max_scroll_y == 0
        # Reopening / resizing may measure the long row afresh.
        await pilot.press("down", "d", "d")
        await pilot.pause()
        assert pane.size.height > height
        await pilot.resize_terminal(size[0], size[1] + 4)
        await pilot.pause()
        assert pane.size.height > height


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("key", ["enter", "d"])
async def test_initial_state_row_has_cursor_and_opens_only_snapshot_details(
    failed: bool, key: str
) -> None:
    service = FakeUsage()
    service.defer = not failed
    service.error = failed
    app = UsageApp(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.browser
        await pilot.press("tab", "tab")
        models = screen.query_one(UsageModels)
        assert models.has_focus and models.option_count == 1
        assert models.highlighted_option and models.highlighted_option.id == "empty"
        prompt = str(models.highlighted_option.prompt)
        assert prompt.startswith("▸ ")
        assert ("Read failed" if failed else "Loading…") in prompt
        assert screen.snapshot is None
        await pilot.press(key)
        await pilot.pause()
        details = text(screen, "#usage-detail-text")
        assert screen._expanded and screen.query_one(UsageDetails).has_focus
        assert "Model:" not in details and "Requests:" not in details
        assert ("No model rows available" if failed else "finishes loading") in details
        assert service.calls == [("day", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 32), (80, 24)])
async def test_ascii_identity_truncation_preserves_cell_alignment(
    size: tuple[int, int],
) -> None:
    service = FakeUsage()
    service.response.models[0].model = "界" * 30
    app = UsageApp(service)
    app.config = SimpleNamespace(ascii_chrome=True)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.browser
        models = screen.query_one(UsageModels)
        models.focus()
        await pilot.pause()
        prompt = str(models.options[0].prompt)
        assert "..." in prompt and "…" not in prompt
        assert prompt.startswith(">")
        assert cell_len(prompt) == cell_len(text(screen, "#usage-heading"))
