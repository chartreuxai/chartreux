"""Keyboard navigation, focus, tab order, escape, filter, reachability, link, and pointer tests for the settings app suite."""

from __future__ import annotations

from typing import cast

import pytest
from textual import events
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from chartreux.cli.textual_ui.screens.settings import (
    SettingsOptionList,
    SettingsScreen,
    _draft_values,
)
from chartreux.cli.textual_ui.screens.status_line_settings import (
    StatusLineSettingsScreen,
)
from chartreux.ui.settings_service import SettingsSaveOutcome, SettingsService
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.cli.textual_ui._settings_app_support import (
    FakeService,
    Harness,
    focus_action,
    navigation_payload,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "group",
    [
        "settings-options",
        "settings-entries",
        "settings-choices",
        "settings-checklist",
        "settings-actions",
    ],
)
@pytest.mark.parametrize("count", [0, 1, 30])
@pytest.mark.parametrize("last", [False, True])
async def test_wp3_bounded_keyboard_groups(group: str, count: int, last: bool) -> None:
    service = FakeService()
    path = (
        "agent.identity"
        if group == "settings-choices"
        else "inventory_tools"
        if group == "settings-checklist"
        else "skill_paths"
    )
    next(
        field for field in service.snapshot.fields if field.path == "skill_paths"
    ).effective_value = [f"value-{i}" for i in range(count)]
    service.snapshot.inventories["tools"] = [f"member-{i}" for i in range(count)]
    next(
        item for item in service.snapshot.catalog if item.path == "agent.identity"
    ).choices = tuple(f"choice-{i}" for i in range(count))
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        if group != "settings-options":
            catalog.highlighted = catalog.get_option_index(path)
            await pilot.press("enter")
        options = screen.query_one(f"#{group}", OptionList)
        if group == "settings-options":
            # Include disabled headings at both ends and between selectable rows.
            screen._replace_group(
                options,
                [Option("heading", disabled=True)]
                + [Option("same", id=f"row:{i}") for i in range(count)]
                + [Option("heading", disabled=True)],
            )
        options.focus()
        await pilot.press("end" if last else "home")
        before = options.highlighted
        payload = navigation_payload(screen)
        await pilot.press(
            *(
                ["down", "j", "pagedown", "end"]
                if last
                else ["up", "k", "pageup", "home"]
            )
        )
        assert options.highlighted == before
        assert options.has_focus
        assert navigation_payload(screen) == payload
        assert not service.saved
        if group == "settings-options" and count > 1:
            await pilot.press("home", "down")
            assert options.highlighted == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "path,body_id",
    [
        ("skill_paths", "settings-entries"),
        ("agent.identity", "settings-choices"),
        ("inventory_tools", "settings-checklist"),
    ],
)
async def test_wp3_tab_order_and_row_scroll_reentry(
    reverse: bool, path: str, body_id: str
) -> None:
    service = FakeService()
    next(
        field for field in service.snapshot.fields if field.path == "skill_paths"
    ).effective_value = [f"value-{i}" for i in range(40)]
    next(
        item for item in service.snapshot.catalog if item.path == "agent.identity"
    ).choices = tuple(f"choice-{i}" for i in range(40))
    service.snapshot.inventories["tools"] = [f"member-{i}" for i in range(40)]
    next(
        field for field in service.snapshot.fields if field.path == "agent.identity"
    ).effective_value = "choice-39"
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index(path)
        await pilot.press("enter")
        body = screen.query_one(f"#{body_id}", OptionList)
        if path == "agent.identity":
            await pilot.pause()
            assert body.highlighted == 39 and body.scroll_y > 0
        body.highlighted = 25
        await pilot.pause()
        body.scroll_to(y=20, animate=False, immediate=True, force=True)
        await pilot.pause()
        selected = body.highlighted_option
        scroll = body.scroll_y
        assert selected is not None and scroll > 0
        payload = navigation_payload(screen)
        groups = ["settings-options", body_id] + (
            [] if path == "agent.identity" else ["settings-actions"]
        )
        index = groups.index(body_id)
        direction = -1 if reverse else 1
        for _ in groups:
            await pilot.press("shift+tab" if reverse else "tab")
            index = (index + direction) % len(groups)
            assert screen.focused is not None and screen.focused.id == groups[index]
        await pilot.pause()
        assert (
            body.highlighted_option is not None
            and body.highlighted_option.id == selected.id
        )
        assert body.scroll_y == scroll
        screen._refresh_options()
        await pilot.pause()
        assert body.has_focus and body.scroll_y == scroll
        assert navigation_payload(screen) == payload
        assert not service.saved
        # Catalog typing is suppressed throughout expansion, even when focused.
        catalog.focus()
        await pilot.press("x", "backspace")
        assert catalog._query == ""
        assert navigation_payload(screen) == payload
        body.focus()
        await pilot.press("f1", "tab")
        assert not screen._help_open and catalog.has_focus
        await pilot.press("shift+tab")
        assert screen.focused is not None and screen.focused.id == groups[-1]
        assert not service.saved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selected,new_ids,expected",
    [
        (0, ["b", "c", "d"], "b"),
        (1, ["a", "c", "d"], "c"),
        (3, ["a", "b", "c"], "c"),
        (2, ["b", "c", "d"], "c"),
        (1, ["a", "d"], "a"),
        (2, ["new0", "new1"], "new1"),
        (1, [], None),
    ],
)
async def test_wp3_position_repair(
    selected: int, new_ids: list[str], expected: str | None
) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one(SettingsOptionList)
        screen._replace_group(
            options,
            [Option("duplicate", id=identifier) for identifier in ["a", "b", "c", "d"]],
        )
        options.highlighted = selected
        screen._replace_group(
            options,
            [Option("heading", disabled=True)]
            + [Option("duplicate", id=identifier) for identifier in new_ids],
        )
        await pilot.pause()
        assert (
            options.highlighted_option.id if options.highlighted_option else None
        ) == expected
        assert options.has_focus and not service.saved
        if not new_ids:
            screen._replace_group(options, [Option("new", id="new")])
            await pilot.pause()
            assert (
                options.highlighted_option is not None
                and options.highlighted_option.id == "new"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("opener", ["entries", "actions"])
async def test_wp3_nested_openers_and_exact_escape(opener: str) -> None:
    service = FakeService()
    values = [f"value-{i}" for i in range(40)]
    next(
        field for field in service.snapshot.fields if field.path == "skill_paths"
    ).effective_value = [value for value in values]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index("skill_paths")
        await pilot.pause()
        catalog.scroll_to(y=2, animate=False, immediate=True, force=True)
        expansion_position = screen._capture_group_position(catalog)
        await pilot.press("enter")
        entries = screen.query_one("#settings-entries", OptionList)
        entries.highlighted = 25
        await pilot.pause()
        entries.scroll_to(y=20, animate=False, immediate=True, force=True)
        await pilot.pause()
        entry_id = entries.highlighted_option.id if entries.highlighted_option else None
        entry_scroll = entries.scroll_y
        if opener == "actions":
            await pilot.press("tab", "home")
        origin = screen.focused
        assert origin is not None
        origin_position = screen._capture_group_position(origin)
        payload = navigation_payload(screen)
        await pilot.press("enter", "tab", "shift+tab")
        editor = screen.query_one("#settings-input", Input)
        assert editor.has_focus
        editor.value = "temporary"
        await pilot.pause()
        editor.cursor_position = 2
        await pilot.press("f1")
        assert screen._help_open
        item = next(item for item in screen.catalog if item.path == "skill_paths")
        screen._open_confirmation(
            "discard", item, None, "Inspect nested confirmation", "Discard"
        )
        await pilot.press("escape")
        assert screen._confirmation is None and screen._help_open
        assert (
            screen.query_one("#settings-help").has_focus and screen._editing is not None
        )
        await pilot.press("escape")
        assert (
            not screen._help_open and editor.has_focus and editor.cursor_position == 2
        )
        await pilot.press("escape")
        await pilot.pause()
        assert (
            screen._editing is None
            and origin.has_focus
            and screen._expanded == "skill_paths"
        )
        assert origin_position is not None
        assert screen._capture_group_position(origin) == origin_position
        assert (
            entries.highlighted_option is not None
            and entries.highlighted_option.id == entry_id
        )
        assert entries.scroll_y == entry_scroll
        assert navigation_payload(screen) == payload and not service.saved
        # A dirty discard cancellation returns to this exact entry/action.
        screen._list_draft = (
            [*screen._list_draft, screen._new_draft_entry("new")]
            if screen._list_draft
            else []
        )
        await pilot.press("escape")
        assert screen._confirmation is not None
        await pilot.press("escape")
        await pilot.pause()
        assert origin.has_focus and screen._expanded == "skill_paths"
        assert screen._capture_group_position(origin) == origin_position
        # Confirm discard unwinds expansion, not the filter or Settings itself.
        await pilot.press("escape", "down", "enter")
        await pilot.pause()
        assert screen._expanded is None and catalog.has_focus
        assert screen._capture_group_position(catalog) == expansion_position
        assert not service.saved


@pytest.mark.asyncio
async def test_wp3_confirmation_over_editor_and_help_group_exit() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.press(*"skill_paths", "enter", "tab", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        editor = screen.query_one("#settings-input", Input)
        editor.value = "untouched"
        editor.cursor_position = 3
        item = next(item for item in screen.catalog if item.path == "skill_paths")
        screen._open_confirmation(
            "discard", item, None, "Confirmation over editor", "Discard"
        )
        await pilot.press("tab", "shift+tab", "escape")
        assert (
            editor.has_focus
            and editor.value == "untouched"
            and editor.cursor_position == 3
        )
        assert screen._editing is not None
        await pilot.press("escape")
        actions = screen.query_one("#settings-actions", OptionList)
        assert actions.has_focus
        await pilot.press("down", "f1", "shift+tab")
        assert actions.has_focus and not screen._help_open
        assert (
            actions.highlighted_option is not None
            and actions.highlighted_option.id == "apply"
        )
        await pilot.press("f1")
        screen.query_one("#settings-entries").focus()
        await pilot.pause()
        assert not screen._help_open and screen.query_one("#settings-entries").has_focus
        assert not service.saved


@pytest.mark.asyncio
async def test_wp3_filter_opener_repair_and_escape_purity() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index("skill_paths")
        await pilot.pause()
        catalog.scroll_to(y=2, animate=False, immediate=True, force=True)
        await pilot.pause()
        opener = screen._capture_group_position(catalog)
        payload = navigation_payload(screen)
        await pilot.press(*"agent.identity")
        assert (
            catalog.highlighted_option is not None
            and catalog.highlighted_option.id == "agent.identity"
        )
        await pilot.press("escape")
        await pilot.pause()
        assert (
            catalog._query == "" and screen._capture_group_position(catalog) == opener
        )
        assert navigation_payload(screen) == payload and not service.saved
        await pilot.press("escape")
        assert not isinstance(pilot.app.screen, SettingsScreen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body_id",
    [("agent.identity", "settings-choices"), ("inventory_tools", "settings-checklist")],
)
async def test_wp3_dirty_collection_confirmation_and_help_openers(
    path: str, body_id: str
) -> None:
    service = FakeService()
    next(
        item for item in service.snapshot.catalog if item.path == "agent.identity"
    ).choices = ("default", "other")
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index(path)
        await pilot.press("enter", "down", "space")
        body = screen.query_one(f"#{body_id}", OptionList)
        opener = screen._capture_group_position(body)
        payload = navigation_payload(screen)
        await pilot.press("f1")
        item = next(item for item in screen.catalog if item.path == path)
        screen._open_confirmation(
            "discard", item, None, "Inspect confirmation over help", "Discard"
        )
        await pilot.press("escape")
        assert screen._help_open and screen.query_one("#settings-help").has_focus
        await pilot.press("escape")
        assert body.has_focus and screen._capture_group_position(body) == opener
        await pilot.press("escape")
        assert screen._confirmation is not None
        await pilot.press("escape")
        assert body.has_focus and screen._capture_group_position(body) == opener
        assert navigation_payload(screen) == payload and not service.saved
        await pilot.press("escape", "down", "enter")
        assert screen._expanded is None and catalog.has_focus and not service.saved


@pytest.mark.asyncio
async def test_wp3_busy_escape_owns_press_before_confirmation_help_editor() -> None:
    import asyncio

    class DelayedService(FakeService):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def save(
            self, changed_leaves: dict[str, object], expected_revision: str | None
        ) -> SettingsSaveOutcome:
            self.started.set()
            await self.release.wait()
            return await super().save(changed_leaves, expected_revision)

    service = DelayedService()
    service.outcome = SettingsSaveOutcome(
        "not_saved", "unchanged", error="write failed"
    )
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.press(*"skill_paths", "enter", "tab", "enter", "f1")
        screen = cast(SettingsScreen, pilot.app.screen)
        item = next(item for item in screen.catalog if item.path == "skill_paths")
        screen._open_confirmation(
            "discard", item, None, "Nested busy confirmation", "Discard"
        )
        worker = screen.run_worker(screen._write(item, []), group="settings-write")
        await service.started.wait()
        await pilot.press("escape")
        assert screen._busy and screen._confirmation is not None and screen._help_open
        assert screen._editing == item.path and screen._expanded == item.path
        assert "Wait for save" in str(
            screen.query_one("#settings-hint", NoMarkupStatic).content
        )
        service.release.set()
        await worker.wait()
        await pilot.press("escape")
        assert (
            screen._confirmation is None
            and screen._help_open
            and screen._editing == item.path
        )


@pytest.mark.asyncio
async def test_wp3_catalog_jk_filter_text_not_movement() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        await pilot.press("j", "k")
        assert catalog._query == ""
        await pilot.press("a", "j", "k")
        assert catalog._query == "ajk"
        assert not service.saved


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["status_line", "theme"])
async def test_wp3_child_and_link_return_identity_scroll(path: str) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index(path)
        await pilot.pause()
        catalog.scroll_to(y=2, animate=False, immediate=True, force=True)
        await pilot.pause()
        scroll = catalog.scroll_y
        await pilot.press("enter")
        if path == "status_line":
            assert isinstance(pilot.app.screen, StatusLineSettingsScreen)
            await pilot.press("escape")
            assert pilot.app.screen is screen
        else:
            assert screen.return_state == ("", path, int(scroll), "settings-options")
            # The host's four-field tuple is the only state passed to a new screen.
            restored = SettingsScreen(cast(SettingsService, service), service.snapshot)
            restored.return_state = screen.return_state
            pilot.app.push_screen(restored)
            await pilot.pause()
            screen = restored
            catalog = screen.query_one(SettingsOptionList)
        await pilot.pause()
        assert catalog.has_focus
        assert (
            catalog.highlighted_option is not None
            and catalog.highlighted_option.id == path
        )
        assert catalog.scroll_y == scroll and not service.saved


@pytest.mark.asyncio
async def test_wp3_action_bookmarks_are_collection_context_local() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        actions = screen.query_one("#settings-actions", OptionList)
        catalog.highlighted = catalog.get_option_index("tool_paths")
        await pilot.press("enter", "tab", "down", "escape")
        catalog.highlighted = catalog.get_option_index("skill_paths")
        await pilot.press("enter", "tab")
        assert (
            actions.highlighted_option is not None
            and actions.highlighted_option.id == "add-item"
        )
        await pilot.press("escape")
        catalog.highlighted = catalog.get_option_index("tool_paths")
        await pilot.press("enter", "tab")
        assert (
            actions.highlighted_option is not None
            and actions.highlighted_option.id == "apply"
        )
        assert not service.saved


@pytest.mark.asyncio
async def test_focused_group_owns_commands_and_expanded_catalog_does_not_filter() -> (
    None
):
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.press(*"skill_paths", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        screen._list_draft = screen._make_draft(["draft-entry"])
        screen._refresh_options()
        catalog = screen.query_one("#settings-options", SettingsOptionList)
        catalog._query = ""
        screen._filter("")
        catalog.highlighted = next(
            i
            for i, row in enumerate(catalog.options)
            if row.id == "ask_confirmation_on_exit"
        )
        entries = screen.query_one("#settings-entries", OptionList)
        assert entries.has_focus and screen._current_item().path == "skill_paths"  # type: ignore[union-attr]
        catalog.focus()
        await pilot.pause()
        assert screen._current_item().path == "ask_confirmation_on_exit"  # type: ignore[union-attr]
        await pilot.press("x", "ctrl+d")
        assert catalog._query == "" and screen._confirmation is None
        assert _draft_values(screen._list_draft) == ["draft-entry"]
        focus_action(screen, "add-item")
        await pilot.pause()
        assert screen._current_item().path == "skill_paths"  # type: ignore[union-attr]
        await pilot.press("enter")
        assert screen._editing == "skill_paths"
        assert not service.saved


@pytest.mark.asyncio
async def test_responsive_browser_preserves_filter_selection_and_focus() -> None:
    async with Harness(FakeService()).run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        content = screen.query_one("#settings-content")
        assert content.region.width == 92 and content.region.height == 30
        assert content.border_title == "Settings"
        await pilot.press(*"timeout")
        options = screen.query_one(SettingsOptionList)
        assert options.highlighted_option is not None
        selected = options.highlighted_option.id
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert content.region.size == screen.size
        assert content.has_class("fullscreen") and not content.styles.border
        assert options.highlighted_option is not None
        assert options._query == "timeout" and options.highlighted_option.id == selected
        assert options.has_focus
        await pilot.resize_terminal(100, 32)
        await pilot.pause()
        assert not content.has_class("fullscreen")
        assert options.highlighted_option is not None
        assert options.highlighted_option.id == selected and options.has_focus


@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
@pytest.mark.asyncio
async def test_focused_row_does_not_add_virtual_rows(size: tuple[int, int]) -> None:
    async with Harness(FakeService()).run_test(size=size) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one(SettingsOptionList)
        initial_option_count = options.option_count
        initial_highlight = options.highlighted_option
        assert initial_highlight is not None
        assert options.virtual_size.height == initial_option_count

        await pilot.press("down")
        await pilot.pause()
        moved_highlight = options.highlighted_option
        assert moved_highlight is not None
        assert moved_highlight.id != initial_highlight.id
        assert options.option_count == initial_option_count
        assert options.virtual_size.height == initial_option_count

        await pilot.press("up")
        await pilot.pause()
        restored_highlight = options.highlighted_option
        assert restored_highlight is not None
        assert restored_highlight.id == initial_highlight.id
        assert options.option_count == initial_option_count
        assert options.virtual_size.height == initial_option_count


@pytest.mark.asyncio
async def test_empty_filter_is_focusable_and_does_not_activate() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"unfindable_setting_zzzz")
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one(SettingsOptionList)
        assert options.highlighted_option is not None
        assert options.highlighted_option.id == "empty"
        assert not options.highlighted_option.disabled
        assert "No matching settings" in str(options.highlighted_option.prompt)
        hint = screen.query_one("#settings-hint", NoMarkupStatic)
        assert "Open" not in str(hint.content)
        assert "Clear filter" in str(hint.content)
        await pilot.press("enter")
        assert service.saved == []
        await pilot.press("escape")
        assert options._query == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("dirty", [False, True])
async def test_space_on_compact_catalog_boolean_uses_navigation_guard(
    dirty: bool,
) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index("skill_paths")
        await pilot.press("enter")
        if dirty:
            focus_action(screen, "add-item")
            await pilot.press("enter", *"new-entry", "enter")
        catalog.focus()
        catalog.highlighted = catalog.get_option_index("ask_confirmation_on_exit")
        await pilot.pause()
        assert screen._expanded == "skill_paths" and catalog.has_focus
        assert "Tab/Shift+Tab" in str(
            screen.query_one("#settings-hint", NoMarkupStatic).content
        )
        await pilot.press("space")
        if dirty:
            assert screen._confirmation is not None and not service.saved
            await pilot.press("escape")
            assert _draft_values(screen._list_draft) == ["new-entry"]
            await pilot.press("space", "down", "enter")
        await pilot.pause()
        assert screen._expanded is None
        assert service.saved == [{"ask_confirmation_on_exit": True}]


@pytest.mark.asyncio
async def test_flat_sections_filter_and_navigation() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = next(
            screen
            for screen in pilot.app.screen_stack
            if isinstance(screen, SettingsScreen)
        )
        options = screen.query_one(SettingsOptionList)
        assert len([option for option in options.options if option.disabled]) == 8
        assert all(
            option.id not in {"write_all_settings", "open_config_file"}
            for option in options.options
        )
        assert all(
            "CONFIG FILE" not in str(option.prompt) for option in options.options
        )
        assert "ADVANCED & TOOLS" in str(
            next(
                option.prompt
                for option in options.options
                if option.disabled and "ADVANCED & TOOLS" in str(option.prompt)
            )
        )
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "autocopy_to_clipboard"
        )
        await pilot.press("down", "up", "j", "k")
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "autocopy_to_clipboard"
        )
        await pilot.press(*"timeout")
        assert "timeout" in str(
            screen.query_one("#settings-filter", NoMarkupStatic).content
        )
        assert all(not option.disabled for option in options.options)
        assert options.highlighted_option and "timeout" in str(
            options.highlighted_option.id
        )
        await pilot.press("backspace")
        assert options._query == "timeou"


@pytest.mark.asyncio
async def test_escape_clears_search_before_closing() -> None:
    async with Harness(FakeService()).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"timeout")
        options = screen.query_one(SettingsOptionList)
        assert options._query == "timeout"
        await pilot.press("escape")
        assert pilot.app.screen is screen
        assert options._query == ""
        assert "type to filter" in str(
            screen.query_one("#settings-filter", NoMarkupStatic).content
        )
        await pilot.press("escape")
        assert pilot.app.screen is not screen


@pytest.mark.asyncio
async def test_link_titles_are_plain_text() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        titles = {
            item.command: str(screen._row(item))
            for item in screen.catalog
            if item.command is not None
        }
        assert titles == {
            "/theme": "  Theme    ",
            "/log-level": "  Log level    ",
            "/mcp": "  MCP servers    ",
            "/providers": "  Provider Settings    ",
            "/web-search": "  Web Search    ",
            "/proxy-setup": "  Proxy setup    ",
        }


@pytest.mark.asyncio
async def test_settings_catalog_has_no_config_file_rows() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        assert all(item.group != "Config File" for item in screen.catalog)
        assert all(item.command != "/open-config-file" for item in screen.catalog)
        options = screen.query_one(SettingsOptionList)
        assert all(option.id != "open_config_file" for option in options.options)
        assert all(option.id != "write_all_settings" for option in options.options)


@pytest.mark.asyncio
async def test_full_app_settings_keyboard_reachability_at_minimum_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tests.conftest import build_test_chartreux_app
    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator):
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app = build_test_chartreux_app()
    async with app.run_test(size=(60, 18)) as pilot:
        await app._session_ready.wait()
        await pilot.pause(0.1)
        assert await app._handle_command("/settings")
        await pilot.pause()
        screen = next(
            screen for screen in app.screen_stack if isinstance(screen, SettingsScreen)
        )
        options = screen.query_one(SettingsOptionList)
        before = options.highlighted
        await pilot.press("down")
        assert options.highlighted != before
        await pilot.press("escape")
        await pilot.pause()
        assert not any(
            isinstance(screen, SettingsScreen) for screen in app.screen_stack
        )


@pytest.mark.asyncio
async def test_repeated_settings_commands_share_pending_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from tests.conftest import build_test_chartreux_app

    app = build_test_chartreux_app()
    gate = asyncio.Event()
    wait = AsyncMock(side_effect=lambda: gate.wait())
    monkeypatch.setattr(app, "_wait_for_settings", wait)
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app._show_settings()
        await app._show_settings()
        await pilot.pause()
        wait.assert_awaited_once()
        assert app._settings_worker is not None
        gate.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_settings_link_opens_target_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from chartreux.ui.widgets.theme_picker import ThemePickerApp
    from tests.conftest import build_test_chartreux_app
    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator):
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        await pilot.pause(0.1)
        assert await app._handle_command("/settings")
        await pilot.pause()
        await pilot.press(*"/theme", "enter")
        await pilot.pause(0.2)
        assert not any(
            isinstance(screen, SettingsScreen) for screen in app.screen_stack
        )
        assert app.query(ThemePickerApp)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert isinstance(app.screen, SettingsScreen)
        restored = app.screen.query_one(SettingsOptionList)
        assert restored._query == "/theme"
        assert restored.highlighted_option is not None
        assert app.screen.focused is restored
        await pilot.press("escape", "escape")
        await pilot.pause()
        assert not isinstance(app.screen, SettingsScreen)


@pytest.mark.asyncio
async def test_settings_log_level_link_returns_after_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from chartreux.cli.textual_ui.app import BottomApp
    from tests.conftest import build_test_chartreux_app
    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator):
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        await pilot.pause(0.1)
        assert await app._handle_command("/settings")
        await pilot.pause()
        await pilot.press(*"log_level", "enter")
        await pilot.pause(0.2)
        assert app._current_bottom_app == BottomApp.LogLevelPicker
        assert app.focused is not None
        assert app.focused.id == "loglevelpicker-session"
        await pilot.press("tab", "tab", "tab")
        assert app.focused is not None
        assert app.focused.id == "loglevelpicker-apply"
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert app._current_bottom_app == BottomApp.Input
        assert isinstance(app.screen, SettingsScreen)


@pytest.mark.asyncio
async def test_provider_escape_from_overview_dismisses_screen() -> None:
    from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
    from tests.conftest import build_test_chartreux_app

    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        assert await app._handle_command("/providers")
        for _ in range(50):
            if isinstance(app.screen, ProviderWorkbenchScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, ProviderWorkbenchScreen)
        assert app.screen.state is None
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ProviderWorkbenchScreen)


@pytest.mark.asyncio
async def test_provider_escape_from_deeper_step_goes_back_then_dismisses() -> None:
    from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
    from tests.conftest import build_test_chartreux_app

    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        assert await app._handle_command("/providers")
        for _ in range(50):
            if isinstance(app.screen, ProviderWorkbenchScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, ProviderWorkbenchScreen)
        screen = app.screen
        await pilot.press("enter")
        await pilot.pause()
        assert screen.state is not None
        await pilot.press("escape")
        await pilot.pause()
        assert screen.state is None
        assert app.screen is screen
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not screen


@pytest.mark.asyncio
async def test_workbench_escape_unwinds_before_dismissal() -> None:
    from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
    from tests.conftest import build_test_chartreux_app

    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        assert await app._handle_command("/providers")
        for _ in range(50):
            if isinstance(app.screen, ProviderWorkbenchScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, ProviderWorkbenchScreen)
        screen = app.screen
        await pilot.press("enter")
        assert screen.state is not None
        await pilot.press("escape")
        assert app.screen is screen and screen.state is None
        await pilot.press("escape")
        for _ in range(50):
            if app.screen is not screen:
                break
            await pilot.pause()
        assert app.screen is not screen
        assert screen._dismissed
        assert await app._handle_command("/providers")
        for _ in range(50):
            if isinstance(app.screen, ProviderWorkbenchScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, ProviderWorkbenchScreen)
        assert app.screen is not screen
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "models/providers"])
async def test_settings_provider_link_returns_to_settings(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    from types import SimpleNamespace

    from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
    from tests.conftest import build_test_chartreux_app
    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator):
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        assert await app._handle_command("/settings")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        settings = app.screen
        await pilot.press(*query)
        options = settings.query_one(SettingsOptionList)
        options.highlighted = options.get_option_index("models/providers")
        options.scroll_to_highlight()
        await pilot.pause()
        scroll = options.scroll_y
        if not query:
            assert scroll > 0
        await pilot.press("enter")
        for _ in range(50):
            if isinstance(app.screen, ProviderWorkbenchScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, ProviderWorkbenchScreen)
        await pilot.press("escape")
        for _ in range(50):
            if isinstance(app.screen, SettingsScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        options = app.screen.query_one(SettingsOptionList)
        assert options._query == query
        assert options.highlighted_option is not None
        assert options.highlighted_option.id == "models/providers"
        assert app.screen.focused is options
        assert options.scroll_y == scroll


@pytest.mark.asyncio
async def test_wp4_hover_and_wheel_only_affect_pointed_region() -> None:
    service = FakeService()
    next(
        f for f in service.snapshot.fields if f.path == "skill_paths"
    ).effective_value = [f"item-{i}" for i in range(40)]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index("skill_paths")
        await pilot.press("enter")
        entries = screen.query_one("#settings-entries", OptionList)
        await pilot.pause()
        before = navigation_payload(screen)
        selected = entries.highlighted_option
        catalog_scroll = catalog.scroll_y
        for offset in [(3, 0), (3, 2)]:
            await pilot.hover(entries, offset=offset)
        assert entries.highlighted_option is selected
        assert entries.has_focus and navigation_payload(screen) == before
        x, y = entries.region.x + 3, entries.region.y + 1
        for _ in range(4):
            screen._forward_event(
                events.MouseScrollDown(None, x, y, 0, 1, 0, False, False, False)
            )
        await pilot.pause()
        assert entries.scroll_y > 0 and catalog.scroll_y == catalog_scroll
        assert navigation_payload(screen) == before and not service.saved
