"""Help footer, hints, and detail view tests for the settings app suite."""

from __future__ import annotations

from typing import cast

import pytest
from textual.widgets import Input, OptionList

from chartreux.cli.textual_ui.screens.settings import (
    SettingsChecklist,
    SettingsOptionList,
    SettingsScreen,
)
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.cli.textual_ui._settings_app_support import (
    FakeService,
    Harness,
    focus_action,
    pointer_shortcut_offset,
)


@pytest.mark.asyncio
async def test_footer_help_is_reachable_and_restores_list_focus() -> None:
    async with Harness(FakeService()).run_test(size=(50, 20)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one(SettingsOptionList)
        hint = screen.query_one("#settings-hint", NoMarkupStatic)
        assert "Ctrl+D" not in str(hint.content)
        await pilot.press("f1")
        help_widget = screen.query_one("#settings-help", NoMarkupStatic)
        assert help_widget.has_focus
        assert "Ctrl+D" in str(help_widget.content)
        assert "Tab/Shift+Tab" in str(help_widget.content)
        assert "Tab/Shift+Tab" in str(hint.content)
        await pilot.pause()
        assert help_widget.max_scroll_y > 0 and help_widget.scroll_y == 0
        text = screen.query_one("#settings-help-text", NoMarkupStatic)
        opening_y = text.region.y
        await pilot.press("down")
        await pilot.pause()
        assert help_widget.scroll_y > 0
        assert text.region.y < opening_y
        await pilot.press("home")
        await pilot.pause()
        assert help_widget.scroll_y == 0
        await pilot.press("pagedown")
        await pilot.pause()
        assert help_widget.scroll_y > 0
        await pilot.press("escape")
        assert options.has_focus


@pytest.mark.asyncio
async def test_help_over_confirmation_escape_unwinds_one_level() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"enable_system_trust_store", "enter")
        confirmation = screen._confirmation
        assert confirmation is not None
        actions = screen.query_one("#settings-confirmation-actions", OptionList)
        assert actions.has_focus
        await pilot.press("f1")
        assert screen._help_open and screen.query_one("#settings-help").has_focus
        await pilot.press("escape")
        assert not screen._help_open and screen._confirmation == confirmation
        assert actions.has_focus and actions.highlighted_option is not None
        assert actions.highlighted_option.id == "cancel" and not service.saved
        await pilot.press("escape")
        assert screen._confirmation is None and not screen._help_open
        assert screen.query_one(SettingsOptionList).has_focus
        assert isinstance(pilot.app.screen, SettingsScreen) and not service.saved


@pytest.mark.asyncio
async def test_detail_view_contains_selected_description_and_can_scroll() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"skill_paths", "enter", "f1")
        detail = screen.query_one("#settings-help", NoMarkupStatic)
        assert detail.has_focus and detail.can_focus
        assert "Empty uses built-in skill locations" in str(detail.content)
        await pilot.press("down", "escape")
        assert screen.query_one("#settings-entries").has_focus


@pytest.mark.asyncio
async def test_checklist_action_rows_have_no_membership_and_hints_follow_cursor() -> (
    None
):
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        checklist = screen.query_one(SettingsChecklist)
        actions = focus_action(screen, "add-pattern")
        await pilot.pause()
        line = actions.render_line(0).text
        assert "Add pattern" in line and "[ ]" not in line
        assert all(row.id != "add-pattern" for row in checklist.options)
        assert "Add pattern" in str(
            screen.query_one("#settings-hint", NoMarkupStatic).content
        )
        checklist.focus()
        await pilot.pause()
        assert "Toggle" in str(
            screen.query_one("#settings-hint", NoMarkupStatic).content
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["skill_paths", "inventory_tools"])
async def test_wp4_pointer_add_help_delete_reset_and_back(path: str) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*path, "enter")
        actions = screen.query_one("#settings-actions", OptionList)
        await pilot.click(actions, offset=(5, 0), times=2)
        assert screen._editing == path and not service.saved
        editor = screen.query_one("#settings-input", Input)
        editor.value = "custom-*"
        await pilot.click(
            "#settings-hint",
            offset=pointer_shortcut_offset(
                screen, "#settings-hint", "activate_focused"
            ),
        )
        assert screen._editing is None and not service.saved
        await pilot.pause()
        body = screen.query_one(
            "#settings-checklist" if path == "inventory_tools" else "#settings-entries",
            OptionList,
        )
        body.focus()
        body.highlighted = body.option_count - 1
        await pilot.pause()
        await pilot.click("#settings-help", offset=(1, 1))
        assert screen._help_open
        await pilot.click(
            "#settings-hint",
            offset=pointer_shortcut_offset(screen, "#settings-hint", "close"),
        )
        assert not screen._help_open and body.has_focus
        await pilot.pause()
        await pilot.click(
            "#settings-hint",
            offset=pointer_shortcut_offset(screen, "#settings-hint", "delete_item"),
        )
        assert screen._confirmation is not None
        choices = screen.query_one("#settings-confirmation-actions", OptionList)
        await pilot.click(choices, offset=(3, 0))
        assert screen._confirmation is None and not service.saved
        await pilot.pause()
        await pilot.click(
            "#settings-hint",
            offset=pointer_shortcut_offset(screen, "#settings-hint", "delete_item"),
        )
        await pilot.click(choices, offset=(3, 1), times=2)
        assert screen._confirmation is None and not service.saved
        if path == "inventory_tools":
            await pilot.pause()
            await pilot.click(
                "#settings-help",
                offset=pointer_shortcut_offset(
                    screen, "#settings-help", "remove_override"
                ),
            )
            assert screen._confirmation is not None
            await pilot.click(choices, offset=(3, 1), times=2)
            await pilot.pause()
            assert service.saved == [{"enabled_tools": None, "disabled_tools": None}]
        else:
            assert screen._list_draft == []
            await pilot.click(
                "#settings-hint",
                offset=pointer_shortcut_offset(screen, "#settings-hint", "close"),
            )
            assert screen._expanded is None


@pytest.mark.asyncio
async def test_review_help_reopen_restores_full_geometry() -> None:
    async with Harness(FakeService()).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press("f1")
        height = screen.query_one("#settings-help").region.height
        assert height > 4
        await pilot.press("escape", "f1")
        assert screen._help_open
        assert screen.query_one("#settings-help").region.height == height
