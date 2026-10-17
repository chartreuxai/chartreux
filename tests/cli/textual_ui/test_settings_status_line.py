"""Status line settings screen tests for the settings app suite."""

from __future__ import annotations

from typing import cast

import pytest
from textual.app import App
from textual.widgets import OptionList

from chartreux.cli.textual_ui.screens.settings import SettingsOptionList, SettingsScreen
from chartreux.cli.textual_ui.screens.status_line_settings import (
    StatusLineOptionList,
    StatusLineSettingsResult,
    StatusLineSettingsScreen,
)
from chartreux.ui.settings_service import SettingsSaveOutcome, SettingsService
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.cli.textual_ui._settings_app_support import FakeService, Harness


@pytest.mark.asyncio
async def test_status_line_backing_leaves_are_not_generic_list_rows() -> None:
    async with Harness(FakeService()).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        assert all(not item.path.startswith("status_line.") for item in screen.catalog)
        composite = next(item for item in screen.catalog if item.path == "status_line")
        assert composite.control == "status_line"
        assert screen._row(composite, selected=False).plain.startswith("  ")
        assert screen._row(composite, selected=True).plain.startswith("▸ ")
        assert screen.fields["status_line.segments"].effective_value == [
            "directory",
            "pid",
            "context",
        ]


class StatusLineHarness(App[None]):
    def __init__(self, service: FakeService) -> None:
        super().__init__()
        self.service = service
        self.result: StatusLineSettingsResult | None = None

    def on_mount(self) -> None:
        self.push_screen(
            StatusLineSettingsScreen(
                cast(SettingsService, self.service), self.service.snapshot
            ),
            self._receive,
        )

    def _receive(self, result: StatusLineSettingsResult | None) -> None:
        self.result = result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "theme", ["textual-dark", "textual-light", "ansi-dark", "ansi-light"]
)
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
@pytest.mark.parametrize("plain_chrome", [False, True])
async def test_status_line_focused_rows_remain_readable(
    theme: str,
    size: tuple[int, int],
    plain_chrome: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from rich.color import ColorType

    if plain_chrome:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    app = StatusLineHarness(FakeService())
    app.theme = theme
    monkeypatch.setattr(
        app, "config", SimpleNamespace(ascii_chrome=plain_chrome), raising=False
    )

    def assert_readable(options: OptionList, row: int, label: str) -> None:
        strip = options.render_line(row)
        assert label in strip.text
        assert strip.cell_length == options.scrollable_content_region.width
        padding_style = list(strip)[-1].style
        assert padding_style is not None
        styles = [segment.style for segment in strip if segment.text.strip()]
        assert styles
        for style in styles:
            assert style is not None and style.bold and not style.reverse
            assert style.color is not None and style.bgcolor is not None
            assert style.color.type != ColorType.DEFAULT
            assert style.bgcolor.type != ColorType.DEFAULT
            assert style.color != style.bgcolor
            # Text and padding share the same block cursor, with no inline colors.
            assert style.color == padding_style.color
            assert style.bgcolor == padding_style.bgcolor

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = cast(StatusLineSettingsScreen, app.screen)
        options = screen.query_one(StatusLineOptionList)
        assert options.has_focus
        assert_readable(options, 0, "Directory")
        assert options.render_line(0).text.startswith("> " if plain_chrome else "▸ ")
        options.highlighted = 3  # An off row must not retain its muted inline color.
        await pilot.pause()
        assert_readable(options, 3, "Model")
        await pilot.press("d")
        await pilot.pause()
        assert not options.has_focus
        assert not any(
            segment.style and segment.style.bold for segment in options.render_line(3)
        )
        await pilot.press("escape")
        action_rows = screen.query_one("#status-line-settings-actions", OptionList)
        action_rows.focus()
        action_rows.highlighted = action_rows.get_option_index("apply")
        await pilot.pause()
        assert_readable(action_rows, 0, "Apply changes")
        await pilot.press("down")
        await pilot.pause()
        assert_readable(action_rows, 1, "Back")
        options.focus()
        await pilot.press("space", "escape")
        await pilot.pause()
        actions = screen.query_one(
            "#status-line-settings-confirmation-actions", OptionList
        )
        assert_readable(actions, 0, "[Cancel]")
        await pilot.press("down")
        await pilot.pause()
        assert_readable(actions, 1, "[Discard edits]")


@pytest.mark.asyncio
@pytest.mark.parametrize("group", ["options", "actions"])
@pytest.mark.parametrize("edge", ["first", "last"])
async def test_status_line_group_arrows_are_bounded(group: str, edge: str) -> None:
    service = FakeService()
    async with StatusLineHarness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        options = screen.query_one(f"#status-line-settings-{group}", OptionList)
        options.focus()
        options.highlighted = 0 if edge == "first" else options.option_count - 1
        await pilot.pause()
        selected = options.highlighted_option
        before = screen.draft.model_copy(deep=True)
        order = screen.order.copy()
        keys = (
            ("up", "k", "pageup", "home")
            if edge == "first"
            else ("down", "j", "pagedown", "end")
        )
        await pilot.press(*keys)
        assert options.has_focus and options.highlighted_option is selected
        assert screen.draft == before and screen.order == order and not service.saved
        assert [
            row.id
            for row in screen.query_one(
                "#status-line-settings-actions", OptionList
            ).options
        ] == ["apply", "back"]


@pytest.mark.asyncio
async def test_status_line_tab_bookmarks_and_reorder_identity() -> None:
    service = FakeService()
    async with StatusLineHarness(service).run_test(size=(50, 20)) as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        config = screen.query_one("#status-line-settings-options", OptionList)
        actions = screen.query_one("#status-line-settings-actions", OptionList)
        config.highlighted = config.get_option_index("spend-month")
        await pilot.pause()
        scroll = config.scroll_y
        await pilot.press("tab", "down")
        assert actions.has_focus and screen._selected() == "back"
        await pilot.press("tab")
        assert config.has_focus and screen._selected() == "spend-month"
        assert config.scroll_y == scroll
        await pilot.press("alt+up")
        assert screen._selected() == "spend-month"
        await pilot.press("shift+tab")
        assert actions.has_focus and screen._selected() == "back"
        await pilot.press("shift+tab")
        assert config.has_focus and screen._selected() == "spend-month"
        await pilot.press("f1")
        assert screen.query_one("#status-line-settings-details").has_focus
        await pilot.press("shift+tab")
        assert actions.has_focus and screen._selected() == "back"
        assert not screen._help_open
        await pilot.press("d", "tab")
        assert config.has_focus and screen._selected() == "spend-month"
        assert not screen._details_open
        assert not service.saved and not screen.dirty


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["directory", "pid", "context", "separator"])
async def test_status_line_enter_space_cycle_config_without_saving(name: str) -> None:
    service = FakeService()
    async with StatusLineHarness(service).run_test() as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        config = screen.query_one("#status-line-settings-options", OptionList)
        config.highlighted = config.get_option_index(name)
        await pilot.pause()
        before = screen.draft.model_copy(deep=True)
        await pilot.press("enter")
        assert screen.draft != before and not service.saved
        await pilot.press("space")
        assert screen.draft == before and not service.saved
        await pilot.press("tab")
        actions = screen.query_one("#status-line-settings-actions", OptionList)
        for action in ("apply", "back"):
            actions.highlighted = actions.get_option_index(action)
            await pilot.press("space", "alt+up", "alt+down")
            assert pilot.app.screen is screen and not screen._confirmation
            assert screen.draft == before and not service.saved


@pytest.mark.asyncio
@pytest.mark.parametrize("opener", ["directory", "separator", "apply", "back"])
async def test_status_line_inspection_confirmation_restores_exact_opener(
    opener: str,
) -> None:
    service = FakeService()
    async with StatusLineHarness(service).run_test(size=(50, 20)) as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        await pilot.press("space")
        group = "actions" if opener in {"apply", "back"} else "options"
        options = screen.query_one(f"#status-line-settings-{group}", OptionList)
        options.highlighted = options.get_option_index(opener)
        options.focus()
        await pilot.pause()
        scroll = options.scroll_y
        await pilot.press("d", "f1", "escape")
        assert screen._details_open and not screen._help_open
        assert screen.query_one("#status-line-settings-details").has_focus
        await pilot.press("escape")
        assert options.has_focus and screen._selected() == opener
        assert options.scroll_y == scroll
        await pilot.press("escape")
        assert screen._confirmation == "discard"
        await pilot.press("tab", "shift+tab")
        assert screen.query_one("#status-line-settings-confirmation-actions").has_focus
        await pilot.press("escape")
        assert options.has_focus and screen._selected() == opener
        assert options.scroll_y == scroll and screen.dirty and not service.saved
        # Confirmation owns this press even with an inspection region underneath.
        await pilot.press("f1")
        screen._confirm("discard", "Discard?")
        await pilot.press("escape")
        assert screen._help_open and not screen._confirmation
        assert screen.query_one("#status-line-settings-details").has_focus
        await pilot.press("escape")
        assert options.has_focus and screen._selected() == opener


@pytest.mark.asyncio
@pytest.mark.parametrize("pointer", [False, True])
async def test_status_line_action_pointer_keyboard_parity(pointer: bool) -> None:
    service = FakeService()
    async with StatusLineHarness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        await pilot.press("space")
        actions = screen.query_one("#status-line-settings-actions", OptionList)
        if pointer:
            await pilot.click("#status-line-settings-actions", offset=(3, 1), times=2)
        else:
            actions.highlighted = actions.get_option_index("back")
            actions.focus()
            await pilot.press("enter")
        assert screen._confirmation == "discard" and not service.saved
        if pointer:
            await pilot.click(
                "#status-line-settings-confirmation-actions", offset=(3, 0)
            )
        else:
            await pilot.press("enter")
        assert not screen._confirmation and screen.dirty
        assert actions.has_focus and screen._selected() == "back"
        if pointer:
            await pilot.click("#status-line-settings-actions", offset=(3, 0), times=2)
        else:
            actions.highlighted = actions.get_option_index("apply")
            await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"status_line.directory_style": "path"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("pointer", [False, True])
async def test_status_line_controls_pointer_keyboard_parity(pointer: bool) -> None:
    service = FakeService()
    service.outcome = SettingsSaveOutcome(
        "not_saved", "unchanged", error="reset failed"
    )
    for field in service.snapshot.fields:
        if field.path == "status_line.separator":
            field.saved_explicit = True
            field.saved_value = "space"
    async with StatusLineHarness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        config = screen.query_one("#status-line-settings-options", OptionList)
        config.highlighted = config.get_option_index("pid")
        await pilot.pause()

        async def command(key: str, x: int) -> None:
            if pointer:
                await pilot.click("#status-line-settings-controls", offset=(x, 0))
            else:
                await pilot.press(key)

        await command("alt+down", 10)
        assert screen.order.index("pid") == 2 and screen._selected() == "pid"
        await command("alt+up", 1)
        assert screen.order.index("pid") == 1 and not screen.dirty
        await command("d", 28)
        assert screen._details_open
        await command("f1", 37)
        assert screen._help_open and screen._details_open
        before = screen.draft.model_copy(deep=True)
        await command("alt+down", 10)
        await command("ctrl+r", 21)
        await pilot.click("#status-line-settings-actions", offset=(3, 0))
        assert screen.draft == before and not service.saved and not screen._confirmation
        await command("escape", 43)
        assert not screen._help_open and screen._details_open
        await command("escape", 43)
        assert not screen._details_open and config.has_focus
        await command("ctrl+r", 21)
        assert screen._confirmation == "reset"
        if pointer:
            await pilot.click(
                "#status-line-settings-confirmation-actions", offset=(3, 1)
            )
        else:
            await pilot.press("down", "enter")
        await pilot.pause()
        assert service.saved == [{"status_line.separator": None}]
        assert config.has_focus and screen._selected() == "pid"
        assert "reset failed" in screen._feedback


@pytest.mark.asyncio
async def test_status_line_two_apply_events_before_worker_start_save_once() -> None:
    service = FakeService()
    app = StatusLineHarness(service)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = cast(StatusLineSettingsScreen, app.screen)
        screen.draft.directory_style = "path"
        options = screen.query_one("#status-line-settings-actions", OptionList)
        apply = options.get_option("apply")
        # Two Enter selections dispatched without yielding to the save worker.
        for _ in range(2):
            screen.on_option_list_option_selected(
                OptionList.OptionSelected(
                    options, apply, options.get_option_index("apply")
                )
            )
        assert screen._busy
        assert "Wait for save" in str(
            screen.query_one("#status-line-settings-hint", NoMarkupStatic).content
        )
        await pilot.pause()
        assert service.saved == [{"status_line.directory_style": "path"}]


@pytest.mark.asyncio
async def test_status_line_cycles_reorders_and_saves_only_changed_leaves() -> None:
    service = FakeService()
    app = StatusLineHarness(service)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = cast(StatusLineSettingsScreen, app.screen)
        options = screen.query_one(StatusLineOptionList)
        preview = screen.query_one("#status-line-settings-preview", NoMarkupStatic)
        await pilot.pause()
        assert "pid 4242" in str(preview.content)
        assert screen._example.home_directory is not None
        await pilot.press("enter", "down", "space", "right_square_bracket")
        assert screen.draft.directory_style == "path"
        assert screen.draft.segments == ["directory", "context"]
        assert screen._selected() == "pid"
        assert not service.saved
        options.highlighted = options.get_option_index("separator")
        await pilot.press("space", "alt+up")
        assert screen.draft.separator == "space" and screen._selected() == "separator"
        actions = screen.query_one("#status-line-settings-actions", OptionList)
        actions.highlighted = actions.get_option_index("apply")
        actions.focus()
        await pilot.press("space")
        assert not service.saved
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [
            {
                "status_line.segments": ["directory", "context"],
                "status_line.directory_style": "path",
                "status_line.separator": "space",
            }
        ]
        assert service.revisions == ["revision"]
        assert app.result and not app.result.needs_refresh


@pytest.mark.asyncio
async def test_background_jobs_settings_toggle_reorder_preview_and_save() -> None:
    service = FakeService()
    app = StatusLineHarness(service)
    async with app.run_test(size=(100, 32)) as pilot:
        screen = cast(StatusLineSettingsScreen, app.screen)
        options = screen.query_one(StatusLineOptionList)
        options.highlighted = options.get_option_index("background-jobs")
        await pilot.press("space")
        assert screen.draft.segments == [
            "directory",
            "pid",
            "context",
            "background-jobs",
        ]
        preview = screen.query_one("#status-line-settings-preview", NoMarkupStatic)
        assert "Jobs 2" in str(preview.content)
        await pilot.press("space")
        assert "background-jobs" not in screen.draft.segments
        assert "Jobs" not in str(preview.content)
        await pilot.press("space", *(["alt+up"] * 6))
        assert screen._selected() == "background-jobs"
        assert screen.draft.segments == [
            "directory",
            "pid",
            "background-jobs",
            "context",
        ]
        expected = list(screen.draft.segments)
        screen.action_apply()
        await pilot.pause()
        assert service.saved == [{"status_line.segments": expected}]
        assert service.revisions == ["revision"]


@pytest.mark.asyncio
async def test_status_line_off_only_reorder_resize_hover_and_mouse() -> None:
    service = FakeService()
    async with StatusLineHarness(service).run_test(size=(100, 32)) as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        options = screen.query_one(StatusLineOptionList)
        options.highlighted = 3  # Model is off, after enabled segments.
        await pilot.press("right_square_bracket")
        assert screen._selected() == "model" and not screen.dirty
        await pilot.resize_terminal(50, 20)
        await pilot.pause()
        await pilot.hover("#status-line-settings-options", offset=(2, 1))
        assert screen._selected() == "model" and options.has_focus
        screen.action_apply()
        await pilot.pause()
        assert not service.saved
        await pilot.click("#status-line-settings-options", offset=(2, 0))
        assert screen._selected() == "directory" and not screen.dirty
        await pilot.click("#status-line-settings-options", offset=(2, 0), times=2)
        await pilot.pause()
        assert screen.draft.directory_style == "path"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["not_saved", "conflict", "exception"])
async def test_status_line_failure_retains_draft_focus_and_conflict_blocks_retry(
    failure: str,
) -> None:
    class FailingService(FakeService):
        async def save(
            self, changed_leaves: dict[str, object], expected_revision: str | None
        ) -> SettingsSaveOutcome:
            if failure == "exception":
                raise ValueError("invalid draft")
            return await super().save(changed_leaves, expected_revision)

    service = FailingService()
    service.outcome = SettingsSaveOutcome(
        "not_saved",
        "unchanged",
        error="conflict" if failure == "conflict" else "invalid draft",
    )
    async with StatusLineHarness(service).run_test() as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        await pilot.press("enter", "tab", "enter")
        await pilot.pause()
        assert pilot.app.screen is screen and screen.dirty
        actions = screen.query_one("#status-line-settings-actions", OptionList)
        assert actions.has_focus
        assert actions.highlighted_option and actions.highlighted_option.id == "apply"
        assert "Failed:" in screen._feedback
        if failure == "conflict":
            screen.action_apply()
            await pilot.pause()
            assert len(service.saved) == 1
            assert screen._needs_refresh


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["saved", "shadowed", "durability", "application", "unknown"]
)
async def test_status_line_saved_outcome_payload(kind: str) -> None:
    service = FakeService()
    after = service.snapshot.model_copy(update={"user_revision": "after"})
    service.outcome = SettingsSaveOutcome(
        "durability_uncertain" if kind == "durability" else "saved",
        "failed" if kind == "application" else "applied",
        None if kind == "unknown" else after,
        ("status_line.directory_style",) if kind == "shadowed" else (),
        "snapshot_unknown" if kind == "unknown" else None,
    )
    app = StatusLineHarness(service)
    async with app.run_test() as pilot:
        screen = cast(StatusLineSettingsScreen, app.screen)
        await pilot.press("space")
        screen.action_apply()
        await pilot.pause()
        assert app.result
        assert app.result.snapshot is (None if kind == "unknown" else after)
        assert app.result.revision == (None if kind == "unknown" else "after")
        assert app.result.needs_refresh is (kind == "unknown")
        assert ("Warning:" in app.result.feedback) is (kind != "saved")


@pytest.mark.asyncio
async def test_status_line_escape_precedence_and_cancel_default() -> None:
    async with StatusLineHarness(FakeService()).run_test() as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        await pilot.press("enter", "d", "f1")
        screen._busy = True
        await pilot.press("escape")
        assert screen._help_open and screen._details_open
        screen._busy = False
        await pilot.press("escape")
        assert not screen._help_open and screen._details_open
        await pilot.press("escape")
        assert not screen._details_open
        await pilot.press("escape")
        actions = screen.query_one(
            "#status-line-settings-confirmation-actions", OptionList
        )
        assert screen._confirmation == "discard"
        assert actions.highlighted_option and actions.highlighted_option.id == "cancel"
        assert "[Cancel]" in str(actions.highlighted_option.prompt)
        assert actions.virtual_size.height == 2 and actions.region.height == 2
        await pilot.press("enter")
        assert not screen._confirmation and screen.dirty
        await pilot.press("escape", "escape")
        assert not screen._confirmation and screen.dirty
        await pilot.press("escape", "down", "enter")
        await pilot.pause()
        assert pilot.app.screen is not screen


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("failed", [False, True])
async def test_status_line_reset_saved_explicit_only_preserves_draft_on_failure(
    saved: bool, failed: bool
) -> None:
    service = FakeService()
    for field in service.snapshot.fields:
        if field.path == "status_line.separator":
            field.saved_explicit = saved
            field.saved_value = "space" if saved else None
    if failed:
        service.outcome = SettingsSaveOutcome(
            "not_saved", "unchanged", error="reset failed"
        )
    async with StatusLineHarness(service).run_test() as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        await pilot.press("space", "ctrl+r")
        if not saved:
            assert not screen._confirmation and "no user override" in screen._feedback
            assert not service.saved
        else:
            assert screen._confirmation == "reset"
            await pilot.press("down", "enter")
            await pilot.pause()
            assert service.saved == [{"status_line.separator": None}]
            if failed:
                assert screen.dirty and screen.query_one(StatusLineOptionList).has_focus
                assert pilot.app.screen is screen
            else:
                assert pilot.app.screen is not screen


@pytest.mark.asyncio
async def test_status_line_view_only_and_pending_indication() -> None:
    service = FakeService()
    service.snapshot.user_revision = None
    app = StatusLineHarness(service)
    app._pending_callbacks = [object()]  # type: ignore[attr-defined]
    async with app.run_test() as pilot:
        screen = cast(StatusLineSettingsScreen, app.screen)
        assert screen.query_one("#status-line-settings-pending-action").display
        before = screen.draft.model_copy(deep=True)
        await pilot.press("enter", "space", "right_square_bracket", "ctrl+r")
        screen.action_apply()
        assert screen.draft == before and not service.saved
        assert "view only" in screen._feedback


@pytest.mark.asyncio
async def test_status_line_nested_save_restores_opener_and_revision() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(100, 32)) as pilot:
        parent = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"status", "enter")
        child = cast(StatusLineSettingsScreen, pilot.app.screen)
        assert parent.is_mounted and parent in pilot.app.screen_stack
        await pilot.press("space")
        child.action_apply()
        await pilot.pause()
        assert pilot.app.screen is parent
        options = parent.query_one(SettingsOptionList)
        assert options._query == "status" and options.has_focus
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "status_line"
        )
        assert parent.snapshot.user_revision == "revision-1"
        assert parent.fields["status_line.directory_style"].effective_value == "path"
        assert "Saved:" in str(
            parent.query_one("#settings-help", NoMarkupStatic).content
        )
        await pilot.press("enter")
        child = cast(StatusLineSettingsScreen, pilot.app.screen)
        assert child.opening.directory_style == "path"
        await pilot.press("escape")
        await pilot.pause()
        item = next(
            item for item in parent.catalog if item.path == "ask_confirmation_on_exit"
        )
        await parent._write(item, True)
        assert service.revisions == ["revision", "revision-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("dismissal", ["escape", "back"])
async def test_status_line_warning_survives_unchanged_child_return(
    dismissal: str,
) -> None:
    service = FakeService()
    service.outcome = SettingsSaveOutcome(
        "saved", "failed", service.snapshot, error="application failed"
    )
    async with Harness(service).run_test(size=(100, 32)) as pilot:
        parent = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"status", "enter", "space")
        child = cast(StatusLineSettingsScreen, pilot.app.screen)
        child.action_apply()
        await pilot.pause()
        assert pilot.app.screen is parent
        warning = parent._unresolved["status_line"]
        assert "application failed" in warning
        assert warning in str(
            parent.query_one("#settings-help", NoMarkupStatic).content
        )

        await pilot.press("enter")
        child = cast(StatusLineSettingsScreen, pilot.app.screen)
        assert not child.dirty and not child._feedback
        if dismissal == "back":
            options = child.query_one("#status-line-settings-actions", OptionList)
            options.highlighted = options.get_option_index("back")
            options.focus()
            await pilot.press("enter")
        else:
            await pilot.press("escape")
        await pilot.pause()
        assert pilot.app.screen is parent
        assert len(service.saved) == 1
        assert parent._unresolved["status_line"] == warning
        assert parent._error == warning
        assert warning in str(
            parent.query_one("#settings-help", NoMarkupStatic).content
        )

        # A subsequent successful save, unlike inspection, resolves the warning.
        service.outcome = None
        await pilot.press("enter", "space")
        child = cast(StatusLineSettingsScreen, pilot.app.screen)
        child.action_apply()
        await pilot.pause()
        assert pilot.app.screen is parent
        assert "status_line" not in parent._unresolved
        assert "Saved:" in str(
            parent.query_one("#settings-help", NoMarkupStatic).content
        )


@pytest.mark.asyncio
async def test_status_line_parent_unknown_snapshot_blocks_writes_and_grouped_reset() -> (
    None
):
    service = FakeService()
    async with Harness(service).run_test() as pilot:
        parent = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"status_line")
        help_text = str(parent.query_one("#settings-help", NoMarkupStatic).content)
        assert "Saved user overrides: 0/4" in help_text and "Not Set" not in help_text
        parent.fields["status_line.separator"].saved_explicit = True
        await pilot.press("ctrl+r", "down", "enter")
        await pilot.pause()
        assert service.saved == [{"status_line.separator": None}]
        service.outcome = SettingsSaveOutcome(
            "saved", "applied", error="snapshot_unknown"
        )
        await pilot.press("enter", "space")
        child = cast(StatusLineSettingsScreen, pilot.app.screen)
        child.action_apply()
        await pilot.pause()
        assert pilot.app.screen is parent and parent._needs_refresh
        assert parent.snapshot.user_revision is None
        assert "current state unknown" in str(
            parent.query_one("#settings-help", NoMarkupStatic).content
        )
        item = next(
            item for item in parent.catalog if item.path == "ask_confirmation_on_exit"
        )
        await parent._write(item, True)
        assert len(service.saved) == 2


@pytest.mark.asyncio
async def test_status_line_real_app_escape_and_pending_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from unittest.mock import Mock

    from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp

    service = FakeService()
    app = BaseSnapshotTestApp()
    interrupt = Mock(return_value=False)
    monkeypatch.setattr(app, "_try_interrupt", interrupt)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        parent = SettingsScreen(cast(SettingsService, service), service.snapshot)
        await app.push_screen(parent)
        await pilot.press(*"status_line", "enter")
        child = cast(StatusLineSettingsScreen, app.screen)
        assert app.check_action("interrupt", ()) is False
        app._pending_local_question = asyncio.get_running_loop().create_future()
        app._indicate_pending_action()
        assert child.query_one("#status-line-settings-pending-action").display
        assert app._secondary_surface_active()
        await pilot.press("escape")
        await pilot.pause()
        interrupt.assert_not_called()
        assert app.screen is parent and parent.is_mounted
        app._pending_local_question = None
        app._indicate_pending_action()
        assert not parent.query_one("#settings-pending-action").display


@pytest.mark.asyncio
async def test_status_line_unfiltered_opener_scroll_and_read_only_round_trip() -> None:
    service = FakeService()
    service.snapshot.user_revision = None
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        parent = cast(SettingsScreen, pilot.app.screen)
        options = parent.query_one(SettingsOptionList)
        options.highlighted = next(
            i
            for i in range(options.option_count)
            if options.get_option_at_index(i).id == "status_line"
        )
        await pilot.pause()
        options.scroll_to(y=5, animate=False, force=True, immediate=True)
        scroll_y = options.scroll_y
        await pilot.press("enter")
        child = cast(StatusLineSettingsScreen, pilot.app.screen)
        assert child.snapshot.view_only
        await pilot.press("enter", "escape")
        await pilot.pause()
        assert pilot.app.screen is parent and options.has_focus
        assert options._query == "" and options.scroll_y == scroll_y
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "status_line"
        )
        assert not service.saved


@pytest.mark.asyncio
async def test_status_line_all_cycles_required_states_and_preview_focus_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    service = FakeService()
    async with StatusLineHarness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        await pilot.pause()
        update = Mock(wraps=screen._update_preview)
        monkeypatch.setattr(screen, "_update_preview", update)
        await pilot.press("j", "k")
        await pilot.hover("#status-line-settings-options", offset=(2, 2))
        assert update.call_count == 0
        options = screen.query_one(StatusLineOptionList)
        for index in range(8):
            options.highlighted = index
            name = screen._selected()
            before = screen.draft.model_copy(deep=True)
            await pilot.press("enter", "space")
            assert screen.draft == before
            assert {"directory", "context"}.issubset(screen.draft.segments)
            if name == "context":
                await pilot.press("space")
                assert screen.draft.context_style == "tokens"
        options.highlighted = 2
        screen.action_move_up()
        assert screen.draft.segments == ["directory", "context", "pid"]
        screen.action_apply()
        await pilot.pause()
        assert service.saved == [
            {
                "status_line.segments": ["directory", "context", "pid"],
                "status_line.context_style": "tokens",
            }
        ]


@pytest.mark.asyncio
async def test_status_line_details_scroll_and_narrow_footer_exit_visible() -> None:
    async with StatusLineHarness(FakeService()).run_test(size=(50, 20)) as pilot:
        screen = cast(StatusLineSettingsScreen, pilot.app.screen)
        await pilot.pause()
        hint = screen.query_one("#status-line-settings-hint", NoMarkupStatic)
        assert (
            "Esc" in str(hint.content)
            and len(str(hint.content)) <= hint.content_size.width
        )
        await pilot.press("d")
        details = screen.query_one("#status-line-settings-details")
        assert details.has_focus
        await pilot.press("pagedown")
        await pilot.pause()
        assert details.scroll_y > 0
        await pilot.press("escape")
        assert screen.query_one(StatusLineOptionList).has_focus
        assert screen._selected() == "directory"
