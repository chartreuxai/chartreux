from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Literal, cast

import pytest
from textual import events
from textual.app import App
from textual.content import Content
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from chartreux.app_server._resources import _inventory_item_states
from chartreux.app_server.protocol import (
    InventoryItemStateWire,
    SettingDescriptorWire,
    SettingLeafWire,
    SettingsReadResponse,
)
from chartreux.cli.textual_ui.screens.settings import (
    ConfirmationText,
    SettingsChecklist,
    SettingsHints,
    SettingsOptionList,
    SettingsScreen,
    _draft_values,
    inventory_item_state,
    inventory_name_matches,
    parse_setting_value,
    toggle_inventory_name,
)
from chartreux.cli.textual_ui.screens.status_line_settings import (
    StatusLineOptionList,
    StatusLineSettingsResult,
    StatusLineSettingsScreen,
)
from chartreux.core.config.models import StatusLineConfig
from chartreux.core.config.settings_catalog import (
    DEFERRED_SETTINGS,
    EDITABLE_BY_PATH,
    LINK_SETTINGS,
    VISIBLE_SETTINGS,
)
from chartreux.core.utils.matching import name_matches
from chartreux.ui.settings_service import SettingsSaveOutcome, SettingsService
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


def focus_action(screen: SettingsScreen, identifier: str) -> OptionList:
    actions = screen.query_one("#settings-actions", OptionList)
    actions.highlighted = next(
        i for i, row in enumerate(actions.options) if row.id == identifier
    )
    actions.focus()
    return actions


def navigation_payload(screen: SettingsScreen) -> object:
    return (
        _draft_values(screen._list_draft),
        screen._enum_draft,
        {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        },
        screen.snapshot.model_dump(),
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
        "system_prompt_id"
        if group == "settings-choices"
        else "inventory_tools"
        if group == "settings-checklist"
        else "agent_paths"
    )
    next(
        field for field in service.snapshot.fields if field.path == "agent_paths"
    ).effective_value = [f"value-{i}" for i in range(count)]
    service.snapshot.inventories["tools"] = [f"member-{i}" for i in range(count)]
    next(
        item for item in service.snapshot.catalog if item.path == "system_prompt_id"
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
        ("agent_paths", "settings-entries"),
        ("system_prompt_id", "settings-choices"),
        ("inventory_tools", "settings-checklist"),
    ],
)
async def test_wp3_tab_order_and_row_scroll_reentry(
    reverse: bool, path: str, body_id: str
) -> None:
    service = FakeService()
    next(
        field for field in service.snapshot.fields if field.path == "agent_paths"
    ).effective_value = [f"value-{i}" for i in range(40)]
    next(
        item for item in service.snapshot.catalog if item.path == "system_prompt_id"
    ).choices = tuple(f"choice-{i}" for i in range(40))
    service.snapshot.inventories["tools"] = [f"member-{i}" for i in range(40)]
    next(
        field for field in service.snapshot.fields if field.path == "system_prompt_id"
    ).effective_value = "choice-39"
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index(path)
        await pilot.press("enter")
        body = screen.query_one(f"#{body_id}", OptionList)
        if path == "system_prompt_id":
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
            [] if path == "system_prompt_id" else ["settings-actions"]
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
        field for field in service.snapshot.fields if field.path == "agent_paths"
    ).effective_value = [value for value in values]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index("agent_paths")
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
        item = next(item for item in screen.catalog if item.path == "agent_paths")
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
            and screen._expanded == "agent_paths"
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
        assert origin.has_focus and screen._expanded == "agent_paths"
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
        await pilot.press(*"agent_paths", "enter", "tab", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        editor = screen.query_one("#settings-input", Input)
        editor.value = "untouched"
        editor.cursor_position = 3
        item = next(item for item in screen.catalog if item.path == "agent_paths")
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
        catalog.highlighted = catalog.get_option_index("agent_paths")
        await pilot.pause()
        catalog.scroll_to(y=2, animate=False, immediate=True, force=True)
        await pilot.pause()
        opener = screen._capture_group_position(catalog)
        payload = navigation_payload(screen)
        await pilot.press(*"system_prompt")
        assert (
            catalog.highlighted_option is not None
            and catalog.highlighted_option.id == "system_prompt_id"
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
    [
        ("system_prompt_id", "settings-choices"),
        ("inventory_tools", "settings-checklist"),
    ],
)
async def test_wp3_dirty_collection_confirmation_and_help_openers(
    path: str, body_id: str
) -> None:
    service = FakeService()
    next(
        item for item in service.snapshot.catalog if item.path == "system_prompt_id"
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
        await pilot.press(*"agent_paths", "enter", "tab", "enter", "f1")
        screen = cast(SettingsScreen, pilot.app.screen)
        item = next(item for item in screen.catalog if item.path == "agent_paths")
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
        catalog.highlighted = catalog.get_option_index("agent_paths")
        await pilot.press("enter", "tab", "down", "escape")
        catalog.highlighted = catalog.get_option_index("skill_paths")
        await pilot.press("enter", "tab")
        assert (
            actions.highlighted_option is not None
            and actions.highlighted_option.id == "add-item"
        )
        await pilot.press("escape")
        catalog.highlighted = catalog.get_option_index("agent_paths")
        await pilot.press("enter", "tab")
        assert (
            actions.highlighted_option is not None
            and actions.highlighted_option.id == "apply"
        )
        assert not service.saved


def test_draft_tokens_unique_monotonic_and_reconciled_by_occurrence() -> None:
    service = FakeService()
    screen = SettingsScreen(cast(SettingsService, service), service.snapshot)
    entries = screen._make_draft(["same", "other", "same"])
    assert len({entry.token for entry in entries}) == 3
    reconciled = screen._reconcile_draft(entries, ["same", "same", "new"])
    assert reconciled[:2] == [entries[0], entries[2]]
    assert reconciled[2].token > max(entry.token for entry in entries)
    assert screen._make_draft(["same"])[0].token > reconciled[2].token


@pytest.mark.asyncio
async def test_list_edit_preserves_identity_and_string_payload() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.press(*"agent_paths", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        item = next(item for item in screen.catalog if item.path == "agent_paths")
        screen._list_draft = screen._make_draft(["same", "same"])
        first, second = screen._list_draft
        screen._refresh_options()
        screen._begin_list_input(item, str(second.token))
        editor = screen.query_one("#settings-input", Input)
        editor.value = "edited"
        await pilot.press("enter")
        assert screen._list_draft == [first, type(second)(second.token, "edited")]
        options = screen.query_one("#settings-entries", OptionList)
        assert options.highlighted_option is not None
        assert options.highlighted_option.id == f"list:{item.path}:{second.token}"
        screen._save_list(item)
        await pilot.pause()
        assert service.saved == [{item.path: ["same", "edited"]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("inventory", [False, True])
async def test_duplicate_deletion_targets_token(position: int, inventory: bool) -> None:
    service = FakeService()
    path = "inventory_tools" if inventory else "agent_paths"
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.press(*path, "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        entries = screen._make_draft(["duplicate-*"] * 3)
        token = entries[position].token
        if inventory:
            screen._inventory_draft = {"enabled": [], "disabled": entries}
            screen._render_checklist(highlight=f"\x00pattern:disabled:{token}")
            checklist = screen.query_one(SettingsChecklist)
            assert (
                checklist.get_option_at_index(position + 2).id
                == f"pattern:{path}:disabled:{token}"
            )
        else:
            screen._list_draft = entries
            screen._refresh_options()
            options = screen.query_one("#settings-entries", OptionList)
            options.highlighted = next(
                i
                for i, row in enumerate(options.options)
                if row.id == f"list:{path}:{token}"
            )
        screen.action_delete_item()
        assert screen._confirmation is not None
        screen._dismiss_confirmation(apply=True)
        remaining = (
            screen._inventory_draft["disabled"]
            if inventory and screen._inventory_draft
            else screen._list_draft
        )
        assert remaining is not None
        assert remaining == entries[:position] + entries[position + 1 :]
        survivor = remaining[min(position, len(remaining) - 1)]
        group = screen.query_one(
            "#settings-checklist" if inventory else "#settings-entries", OptionList
        )
        assert group.highlighted_option is not None
        assert group.highlighted_option.id == (
            f"pattern:{path}:disabled:{survivor.token}"
            if inventory
            else f"list:{path}:{survivor.token}"
        )
        assert group.has_focus
        assert not service.saved
        item = next(item for item in screen.catalog if item.path == path)
        if inventory:
            screen._save_inventory(item)
        else:
            screen._save_list(item)
        await pilot.pause()
        assert service.saved == [
            {"disabled_tools" if inventory else path: ["duplicate-*"] * 2}
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("inventory", [False, True])
async def test_token_changes_alone_are_clean(inventory: bool) -> None:
    service = FakeService()
    path = "inventory_tools" if inventory else "agent_paths"
    field = next(
        field
        for field in service.snapshot.fields
        if field.path == ("disabled_tools" if inventory else path)
    )
    field.effective_value = ["same", "same"]
    field.saved_value = ["same", "same"]
    field.saved_explicit = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.press(*path, "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        item = next(item for item in screen.catalog if item.path == path)
        if inventory:
            assert screen._inventory_draft is not None
            screen._inventory_draft["disabled"] = screen._make_draft(["same", "same"])
        else:
            screen._list_draft = screen._make_draft(["same", "same"])
        assert not screen._draft_is_dirty(item)
        await pilot.press("escape")
        assert screen._expanded is None and screen._confirmation is None
        assert not service.saved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body_id",
    [
        ("agent_paths", "settings-entries"),
        ("inventory_tools", "settings-checklist"),
        ("system_prompt_id", "settings-choices"),
    ],
)
async def test_expanded_groups_and_minimum_size_layout(path: str, body_id: str) -> None:
    service = FakeService()
    if path == "agent_paths":
        next(
            field for field in service.snapshot.fields if field.path == path
        ).effective_value = [f"entry-{i}" for i in range(40)]
    elif path == "inventory_tools":
        service.snapshot.inventories["tools"] = [f"member-{i}" for i in range(40)]
    else:
        next(
            item for item in service.snapshot.catalog if item.path == path
        ).choices = tuple(f"choice-{i}" for i in range(40))
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one("#settings-options", SettingsOptionList)
        catalog.highlighted = next(
            i for i, row in enumerate(catalog.options) if row.id == path
        )
        await pilot.press("enter")
        await pilot.pause()
        body = screen.query_one(f"#{body_id}", OptionList)
        label = screen.query_one("#settings-active-editor", NoMarkupStatic)
        actions = screen.query_one("#settings-actions", OptionList)
        assert body.has_focus and body.option_count == 40
        assert all(
            row.id is None
            or row.id == "empty"
            or row.id in {item.path for item in screen.catalog}
            for row in catalog.options
        )
        assert "Editing" in str(label.content) and "draft" in str(label.content)
        assert catalog.has_class("compact-catalog") and body.has_class("expanded-body")
        for widget in (
            catalog,
            label,
            body,
            screen.query_one("#settings-help"),
            screen.query_one("#settings-hint"),
        ):
            assert widget.display and widget.region.height >= 1
            assert 0 <= widget.region.y < widget.region.bottom <= 24
        assert catalog.highlighted is not None
        assert (
            catalog.scroll_y
            <= catalog.highlighted
            < catalog.scroll_y + catalog.scrollable_content_region.height
        )
        body.highlighted = body.option_count - 1
        await pilot.pause()
        assert body.scroll_y > 0
        if path != "system_prompt_id":
            assert actions.display and actions.option_count == 2
            assert actions.region.bottom <= screen.query_one("#settings-help").region.y
            focus_action(
                screen, "add-pattern" if path == "inventory_tools" else "add-item"
            )
            await pilot.press("enter")
            await pilot.pause()
            editor = screen.query_one("#settings-editor")
            assert editor.region.y == screen.query_one("#settings-filter").region.bottom
            assert screen.query_one("#settings-input", Input).has_focus
            assert all(
                not screen.query_one(f"#{identifier}").display
                for identifier in (
                    "settings-options",
                    body_id,
                    "settings-actions",
                    "settings-active-editor",
                )
            )
            assert editor.region.bottom <= screen.query_one("#settings-help").region.y
        assert not service.saved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body_id",
    [
        ("agent_paths", "settings-entries"),
        ("inventory_tools", "settings-checklist"),
        ("system_prompt_id", "settings-choices"),
    ],
)
async def test_empty_collection_state_rows_are_focusable_and_inert(
    path: str, body_id: str
) -> None:
    service = FakeService()
    service.snapshot.inventories["tools"] = []
    if path == "system_prompt_id":
        next(
            item for item in service.snapshot.catalog if item.path == path
        ).choices = ()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.press(*path, "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        body = screen.query_one(f"#{body_id}", OptionList)
        assert body.has_focus and body.option_count == 1
        assert body.highlighted_option and str(body.highlighted_option.id).startswith(
            "state:"
        )
        before = (screen._list_draft, screen._enum_draft, screen._inventory_draft)
        await pilot.press("enter", "space", "ctrl+d")
        assert (
            screen._expanded == path
            and screen._editing is None
            and screen._confirmation is None
        )
        assert (
            screen._list_draft,
            screen._enum_draft,
            screen._inventory_draft,
        ) == before
        assert not service.saved


@pytest.mark.asyncio
async def test_focused_group_owns_commands_and_expanded_catalog_does_not_filter() -> (
    None
):
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.press(*"agent_paths", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        screen._list_draft = screen._make_draft(["draft-entry"])
        screen._refresh_options()
        catalog = screen.query_one("#settings-options", SettingsOptionList)
        catalog._query = ""
        screen._filter("")
        catalog.highlighted = next(
            i for i, row in enumerate(catalog.options) if row.id == "show_greeting"
        )
        entries = screen.query_one("#settings-entries", OptionList)
        assert entries.has_focus and screen._current_item().path == "agent_paths"  # type: ignore[union-attr]
        catalog.focus()
        await pilot.pause()
        assert screen._current_item().path == "show_greeting"  # type: ignore[union-attr]
        await pilot.press("x", "ctrl+d")
        assert catalog._query == "" and screen._confirmation is None
        assert _draft_values(screen._list_draft) == ["draft-entry"]
        focus_action(screen, "add-item")
        await pilot.pause()
        assert screen._current_item().path == "agent_paths"  # type: ignore[union-attr]
        await pilot.press("enter")
        assert screen._editing == "agent_paths"
        assert not service.saved


class FakeService:
    def __init__(self) -> None:
        inventory_states: dict[
            Literal["tools", "skills", "agents"], dict[str, InventoryItemStateWire]
        ] = {
            category: {
                name: InventoryItemStateWire(
                    effective=True, default_effective=True, pattern_driven=False
                )
                for name in names
            }
            for category, names in cast(
                dict[Literal["tools", "skills", "agents"], list[str]],
                {
                    "tools": ["bash", "read_file"],
                    "skills": ["review", "search"],
                    "agents": ["worker", "reviewer"],
                },
            ).items()
        }
        self.snapshot = SettingsReadResponse(
            catalog=[
                SettingDescriptorWire.model_validate(asdict(item))
                for item in (*VISIBLE_SETTINGS, *DEFERRED_SETTINGS, *LINK_SETTINGS)
            ],
            fields=[
                SettingLeafWire(
                    path=item.path,
                    effective_value=StatusLineConfig().model_dump(mode="json")[
                        item.path.split(".")[1]
                    ]
                    if item.path.startswith("status_line.")
                    else False
                    if item.kind == "bool"
                    else 2
                    if item.kind in {"int", "float"}
                    else []
                    if item.kind == "list"
                    else "",
                    origin="default",
                    saved_explicit=False,
                )
                for item in EDITABLE_BY_PATH.values()
            ],
            inventories={
                "tools": ["bash", "read_file"],
                "skills": ["review", "search"],
                "agents": ["worker", "reviewer"],
            },
            inventory_states=inventory_states,
            user_layer="user",
            user_revision="revision",
        )
        self.saved: list[dict[str, object]] = []
        self.revisions: list[str | None] = []
        self.outcome: SettingsSaveOutcome | None = None

    async def read(self) -> SettingsReadResponse:
        return self.snapshot

    async def save(
        self, changed_leaves: dict[str, object], expected_revision: str | None
    ) -> SettingsSaveOutcome:
        self.saved.append(dict(changed_leaves))
        self.revisions.append(expected_revision)
        if self.outcome:
            return self.outcome
        fields = []
        for field in self.snapshot.fields:
            if field.path in changed_leaves:
                value = changed_leaves[field.path]
                field = field.model_copy(
                    update={
                        "effective_value": value
                        if value is not None
                        else StatusLineConfig().model_dump(mode="json")[
                            field.path.split(".")[1]
                        ]
                        if field.path.startswith("status_line.")
                        else ([] if isinstance(field.effective_value, list) else False),
                        "saved_explicit": value is not None,
                        "saved_value": value,
                        "origin": "user" if value is not None else "default",
                    }
                )
            fields.append(field)
        self.snapshot = self.snapshot.model_copy(
            update={"fields": fields, "user_revision": f"revision-{len(self.saved)}"}
        )
        self.snapshot.inventory_states = _inventory_item_states(
            self.snapshot.inventories,
            {field.path: field.effective_value for field in fields},
        )
        return SettingsSaveOutcome("saved", "applied", self.snapshot)


class Harness(App[None]):
    def __init__(self, service: FakeService) -> None:
        super().__init__()
        self.service = service

    def on_mount(self) -> None:
        self.push_screen(
            SettingsScreen(cast(SettingsService, self.service), self.service.snapshot)
        )


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
@pytest.mark.parametrize("dirty", [False, True])
async def test_space_on_compact_catalog_boolean_uses_navigation_guard(
    dirty: bool,
) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index("agent_paths")
        await pilot.press("enter")
        if dirty:
            focus_action(screen, "add-item")
            await pilot.press("enter", *"new-entry", "enter")
        catalog.focus()
        catalog.highlighted = catalog.get_option_index("show_greeting")
        await pilot.pause()
        assert screen._expanded == "agent_paths" and catalog.has_focus
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
        assert service.saved == [{"show_greeting": True}]


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
        assert len([option for option in options.options if option.disabled]) == 7
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
            and options.highlighted_option.id == "show_greeting"
        )
        await pilot.press("down", "up", "j", "k")
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "show_greeting"
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
async def test_bool_toggle_and_help_on_highlight() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        help_row = screen.query_one("#settings-help", NoMarkupStatic)
        assert "startup greeting" in str(help_row.content)
        assert "Not Set" in str(help_row.content)
        assert "saves the toggle to user settings" in str(help_row.content)
        await pilot.press("down")
        await pilot.pause()
        assert "clipboard" in str(help_row.content)
        await pilot.press("up", "enter")
        await pilot.pause()
        assert service.saved == [{"show_greeting": True}]
        assert screen._display_value(service.snapshot.catalog[0]) == "[■]"
        assert "Saved user value: True" in str(help_row.content)
        await pilot.press("space")
        await pilot.pause()
        assert service.saved[-1] == {"show_greeting": False}


@pytest.mark.asyncio
async def test_mouse_click_selects_and_double_click_saves_boolean() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one(SettingsOptionList)
        assert "double-click activates" in str(
            screen.query_one("#settings-filter", NoMarkupStatic).content
        )
        await pilot.click(options, offset=(5, 1))
        await pilot.pause()
        assert options.highlighted_option is not None
        assert options.highlighted_option.id == "show_greeting"
        assert options.has_focus
        assert service.saved == []
        await pilot.click(options, offset=(5, 1), times=2)
        await pilot.pause()
        assert service.saved == [{"show_greeting": True}]


@pytest.mark.asyncio
async def test_string_inline_input_filter_suspension_and_cancel() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"displayed_workdir", "enter")
        await pilot.pause()
        editor = screen.query_one("#settings-input", Input)
        assert editor.has_focus
        await pilot.press("j", "k")
        assert editor.value == "jk"
        assert screen.query_one(SettingsOptionList)._query == "displayed_workdir"
        await pilot.press("escape")
        await pilot.pause()
        assert not screen.query_one("#settings-editor").display
        assert not service.saved
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("j", "k", "enter")
        await pilot.pause()
        assert service.saved == [{"displayed_workdir": "jk"}]
        assert not screen.query_one("#settings-editor").display


def test_string_input_strips_whitespace() -> None:
    descriptor = SettingDescriptorWire.model_validate(
        asdict(EDITABLE_BY_PATH["displayed_workdir"])
    )
    assert parse_setting_value(descriptor, "  project  ") == "project"


@pytest.mark.asyncio
async def test_inline_editor_replaces_list_without_overlay_and_survives_resize() -> (
    None
):
    async with Harness(FakeService()).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one(SettingsOptionList)
        options.highlighted = next(
            i
            for i, row in enumerate(options.options)
            if row.id == "session_logging.save_dir"
        )
        await pilot.press("enter")
        await pilot.pause()
        editor = screen.query_one("#settings-input", Input)
        assert editor.has_focus
        assert screen.query_one("#settings-editor").display
        assert not options.display
        assert editor.region.y >= screen.query_one("#settings-editor-label").region.y
        await pilot.resize_terminal(100, 32)
        await pilot.pause()
        assert editor.has_focus and editor.value == ""


@pytest.mark.asyncio
async def test_numeric_inline_recovery_commit_and_provenance() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"api_timeout", "enter")
        await pilot.pause()
        editor = screen.query_one("#settings-input", Input)
        assert editor.value == "2"
        await pilot.press("ctrl+u", "x", "enter")
        assert editor.value == "x"
        assert editor.has_class("-invalid")
        await pilot.press("ctrl+u", "5")
        assert not editor.has_class("-invalid")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"api_timeout": 5.0}]
        assert "Saved user value: 5.0" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
        await pilot.press("ctrl+r", "down", "enter")
        await pilot.pause()
        assert service.saved[-1] == {"api_timeout": None}


@pytest.mark.parametrize("value", ["nan", "inf", "abc"])
def test_numeric_parse_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        parse_setting_value(
            SettingDescriptorWire.model_validate(
                asdict(EDITABLE_BY_PATH["api_timeout"])
            ),
            value,
        )


@pytest.mark.asyncio
async def test_conflict_keeps_old_row() -> None:
    service = FakeService()
    service.outcome = SettingsSaveOutcome("not_saved", "unchanged", error="conflict")
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        screen = next(
            screen
            for screen in pilot.app.screen_stack
            if isinstance(screen, SettingsScreen)
        )
        assert screen._needs_refresh
        await pilot.press("enter")
        await pilot.pause()
        assert "close and reopen Settings" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
        assert screen._display_value(service.snapshot.catalog[0]) == "[ ]"
        assert service.saved == [{"show_greeting": True}]


@pytest.mark.asyncio
async def test_save_failure_persists_on_navigation_and_success_resolves_it() -> None:
    service = FakeService()
    service.outcome = SettingsSaveOutcome("not_saved", "unchanged", error="disk full")
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press("enter")
        await pilot.pause()
        assert "✗ Failed:" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
        await pilot.press("down")
        await pilot.press("up")
        assert "disk full" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
        service.outcome = None
        await pilot.press("space")
        await pilot.pause()
        assert "✗ Failed:" not in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
        assert service.saved[-1] == {"show_greeting": True}


@pytest.mark.asyncio
async def test_unrelated_save_does_not_clear_failure() -> None:
    service = FakeService()
    service.outcome = SettingsSaveOutcome("not_saved", "unchanged", error="disk full")
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press("enter")
        await pilot.pause()
        service.outcome = None
        await pilot.press("down", "enter")
        await pilot.pause()
        await pilot.press("up")
        assert "✗ Failed:" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
        await pilot.press("space")
        await pilot.pause()
        assert "✗ Failed:" not in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )


@pytest.mark.asyncio
async def test_detail_view_contains_selected_description_and_can_scroll() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"agent_paths", "enter", "f1")
        detail = screen.query_one("#settings-help", NoMarkupStatic)
        assert detail.has_focus and detail.can_focus
        assert "Empty uses built-in agent locations" in str(detail.content)
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
async def test_view_only_prevents_edit_and_remove() -> None:
    service = FakeService()
    service.snapshot.user_revision = None
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "ctrl+r")
        await pilot.pause()
        assert not service.saved


@pytest.mark.asyncio
async def test_ctrl_r_removes_saved_user_override() -> None:
    service = FakeService()
    service.snapshot.fields[0] = service.snapshot.fields[0].model_copy(
        update={
            "saved_explicit": True,
            "saved_value": True,
            "effective_value": True,
            "origin": "user",
        }
    )
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+r", "down", "enter")
        await pilot.pause()
        assert service.saved == [{"show_greeting": None}]


@pytest.mark.asyncio
async def test_trust_store_default_no_then_confirm() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"enable_system_trust_store", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        assert screen._confirmation is not None
        assert "Cancel preserves" in str(
            screen.query_one("#settings-confirmation-text", NoMarkupStatic).content
        )
        assert not service.saved
        assert "y" not in str(
            screen.query_one("#settings-hint", NoMarkupStatic).content
        )
        await pilot.press("y", "n")
        assert screen._confirmation is not None and not service.saved
        await pilot.press("enter")
        assert screen._confirmation is None
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        assert service.saved == [{"enable_system_trust_store": True}]


@pytest.mark.asyncio
async def test_confirmation_cancel_is_visible_and_click_cancels_without_save() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"enable_system_trust_store", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        choices = screen.query_one("#settings-confirmation-actions", OptionList)
        assert choices.highlighted_option is not None
        assert choices.highlighted_option.id == "cancel"
        assert "[Cancel]" in str(choices.render_line(0))
        cancel_prompt = choices.get_option_at_index(0).prompt
        assert isinstance(cancel_prompt, Content)
        assert cancel_prompt.plain == "[Cancel]"
        assert cancel_prompt.spans[0].style == "$foreground"
        help_text = str(screen.query_one("#settings-help", NoMarkupStatic).content)
        assert "highlighted confirmation action" in help_text
        assert "saves the toggle" not in help_text
        await pilot.click(choices, offset=(5, 0))
        await pilot.pause()
        assert screen._confirmation is None
        assert service.saved == []
        assert "saves the toggle" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )


@pytest.mark.asyncio
async def test_confirmation_overflow_is_keyboard_reachable() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"enable_system_trust_store", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        assert screen._confirmation is not None
        confirmation = screen.query_one(
            "#settings-confirmation-scroll", ConfirmationText
        )
        screen.query_one("#settings-confirmation-text", NoMarkupStatic).update(
            "\n".join(f"Consequence {i}" for i in range(30))
        )
        await pilot.pause()
        await pilot.press("tab", "shift+tab")
        assert screen.query_one("#settings-confirmation-actions").has_focus
        await pilot.press("pagedown", "pagedown")
        await pilot.pause()
        assert confirmation.scroll_offset.y > 0, (
            confirmation.virtual_size,
            confirmation.size,
            confirmation.max_scroll_y,
        )
        assert not service.saved


@pytest.mark.parametrize("category", ["tools", "skills", "agents"])
@pytest.mark.asyncio
async def test_checklist_glyphs_do_not_rely_on_color(
    category: Literal["tools", "skills", "agents"],
) -> None:
    service = FakeService()
    for field in service.snapshot.fields:
        if field.path == f"disabled_{category}":
            field.saved_value = [service.snapshot.inventories[category][1]]
            field.saved_explicit = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*f"inventory_{category}", "enter")
        await pilot.pause()
        checklist = cast(SettingsScreen, pilot.app.screen).query_one(SettingsChecklist)
        checked = list(checklist.render_line(0))
        unchecked = list(checklist.render_line(1))
        assert "".join(segment.text for segment in checked).startswith("▸ [■]")
        assert "".join(segment.text for segment in unchecked).startswith("  [ ]")
        assert checked[0].style is not None
        assert checked[0].style.reverse and checked[0].style.bold
        assert unchecked[0].style is not None
        assert not unchecked[0].style.reverse


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
async def test_commit_existing_default_pins_explicit_value() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"displayed_workdir", "enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"displayed_workdir": ""}]
        field = next(
            field
            for field in service.snapshot.fields
            if field.path == "displayed_workdir"
        )
        assert field.saved_explicit and field.saved_value == ""


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
async def test_enum_choice_editor() -> None:
    service = FakeService()
    service.snapshot.catalog.append(
        SettingDescriptorWire(
            path="ui_color_scheme",
            label="Theme",
            description="Choose a theme.",
            kind="enum",
            group="Interface",
            choices=("light", "dark"),
        )
    )
    service.snapshot.fields.append(
        SettingLeafWire(
            path="ui_color_scheme",
            effective_value="dark",
            origin="default",
            saved_explicit=False,
        )
    )
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"ui_color_scheme", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one("#settings-choices", OptionList)
        choice = options.highlighted_option
        assert choice is not None and choice.id == "choice:ui_color_scheme:dark"
        assert (
            screen.query_one("#settings-options", SettingsOptionList)._query
            == "ui_color_scheme"
        )
        assert "Accept selected" in str(
            screen.query_one("#settings-hint", NoMarkupStatic).content
        )
        await pilot.press("escape")
        await pilot.pause()
        assert not options.display
        assert not any(
            str(row.id).startswith("choice:")
            for row in screen.query_one("#settings-options", OptionList).options
        )
        await pilot.press("enter", "up", "space", "down", "enter")
        await pilot.pause()
        assert service.saved == [{"ui_color_scheme": "light"}]


@pytest.mark.asyncio
async def test_list_add_edit_delete_discard_and_empty_help() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"agent_paths", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one("#settings-entries", OptionList)
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "state:entries"
        )
        assert "Empty uses built-in agent locations" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
        focus_action(screen, "add-item")
        await pilot.press("enter")
        await pilot.pause()
        assert screen.query_one("#settings-input", Input).has_focus
        await pilot.press(*"  re:^tool  ", "escape")
        await pilot.pause()
        assert not service.saved
        assert not screen.query_one("#settings-editor").display
        assert screen._expanded == "agent_paths"
        await pilot.press("escape")
        assert screen._expanded is None
        await pilot.press("enter")
        focus_action(screen, "add-item")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press(*"  tool-*  ", "enter")
        await pilot.pause()
        assert _draft_values(screen._list_draft) == ["tool-*"] and not service.saved
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"agent_paths": ["tool-*"]}]
        assert screen._expanded is None
        await pilot.press("enter")
        assert (
            options.highlighted_option
            and screen._list_draft
            and options.highlighted_option.id
            == f"list:agent_paths:{screen._list_draft[0].token}"
        )
        focus_action(screen, "add-item")
        await pilot.press("enter")
        await pilot.press(*"  other  ", "enter")
        assert (
            _draft_values(screen._list_draft) == ["tool-*", "other"]
            and len(service.saved) == 1
        )
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved[-1] == {"agent_paths": ["tool-*", "other"]}
        await pilot.press("enter", "home", "ctrl+d", "down", "enter")
        assert (
            _draft_values(screen._list_draft) == ["other"] and len(service.saved) == 2
        )
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved[-1] == {"agent_paths": ["other"]}
        await pilot.press("enter", "enter", "ctrl+u", "space", "enter")
        assert screen.query_one("#settings-input", Input).has_class("-invalid")
        assert (
            _draft_values(screen._list_draft) == ["other"] and len(service.saved) == 3
        )
        await pilot.press("escape", "ctrl+d", "down", "enter")
        assert _draft_values(screen._list_draft) == [] and len(service.saved) == 3
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved[-1] == {"agent_paths": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_navigation_from_dirty_list_requires_explicit_discard(
    size: tuple[int, int],
) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=size) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one(SettingsOptionList)
        options.highlighted = options.get_option_index("agent_paths")
        await pilot.press("enter", "tab", "enter")
        await pilot.press(*"new-entry", "enter")
        assert _draft_values(screen._list_draft) == ["new-entry"]

        await pilot.press("tab")
        assert options.has_focus
        options.highlighted = next(
            index
            for index, row in enumerate(options.options)
            if row.id == "show_greeting"
        )
        options.focus()
        await pilot.press("enter")
        assert screen._confirmation is not None
        await pilot.press("escape")
        assert _draft_values(screen._list_draft) == ["new-entry"]
        assert options.has_focus

        await pilot.press("enter")
        assert screen._confirmation is not None
        await pilot.press("down", "enter")
        await pilot.pause()
        assert screen._expanded is None
        assert screen._list_draft is None
        assert service.saved == [{"show_greeting": True}]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_apply_list_draft_then_continue_to_highlighted_setting(
    size: tuple[int, int],
) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=size) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one(SettingsOptionList)
        options.highlighted = options.get_option_index("agent_paths")
        await pilot.press("enter", "tab", "enter")
        await pilot.press(*"new-entry", "enter")
        await pilot.press("tab")
        assert options.has_focus
        options.highlighted = next(
            index
            for index, row in enumerate(options.options)
            if row.id == "show_greeting"
        )
        options.focus()
        await pilot.press("enter", "escape")
        assert _draft_values(screen._list_draft) == ["new-entry"]

        await pilot.press("shift+tab", "down", "enter")
        await pilot.pause()
        assert service.saved == [{"agent_paths": ["new-entry"]}]
        assert options.highlighted_option is not None
        assert options.highlighted_option.id == "show_greeting"
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved[-1] == {"show_greeting": True}


@pytest.mark.parametrize("name", ["bash", "read_file", "BASH"])
@pytest.mark.parametrize(
    "entries",
    [[], ["Bash"], ["READ_*"], ["re:^read_.*"], ["re:["], ["  "], ["re:.*file"]],
)
def test_inventory_matcher_agrees_with_server(name: str, entries: list[str]) -> None:
    assert inventory_name_matches(name, entries) == name_matches(name, entries)
    state = _inventory_item_states(
        {"tools": [name], "skills": [], "agents": []}, {"enabled_tools": entries}
    )["tools"][name]
    assert inventory_item_state(name, entries, []) == (
        state.effective,
        state.pattern_driven,
    )


def test_toggle_inventory_mode_and_last_item() -> None:
    assert toggle_inventory_name("bash", [], [], is_enabled=True) == ([], ["bash"])
    assert toggle_inventory_name("bash", [], ["bash"], is_enabled=False) == ([], [])
    assert toggle_inventory_name("bash", ["bash"], ["read_file"], is_enabled=True) == (
        [],
        ["read_file", "bash"],
    )
    assert toggle_inventory_name("bash", ["bash"], ["BASH"], is_enabled=False) == (
        ["bash"],
        [],
    )
    assert toggle_inventory_name("bash", ["read_file"], ["BASH"], is_enabled=False) == (
        ["read_file", "bash"],
        [],
    )
    for category in ("skills", "agents"):
        assert toggle_inventory_name(
            "bash", ["bash"], ["bash"], is_enabled=True, category=category
        ) == ([], ["bash"])
    assert toggle_inventory_name("bash", ["bash"], [], is_enabled=True) == (
        [],
        ["bash"],
    )
    assert toggle_inventory_name("bash", ["read_file"], [], is_enabled=False) == (
        ["read_file", "bash"],
        [],
    )


@pytest.mark.parametrize("category", ["tools", "skills", "agents"])
@pytest.mark.parametrize(
    ("enabled", "disabled", "effective", "pattern_driven"),
    [
        (["bash"], ["BASH"], False, False),
        (["bash"], ["bas*"], False, True),
        (["bas*"], ["bash"], False, True),
        (["bas*"], ["b*"], False, True),
        (["read_*"], ["bas*"], False, True),
        (["bash"], ["re:^bas.*"], False, True),
        (["bash"], ["read_*"], True, False),
    ],
)
def test_inventory_overlapping_filters_agree_with_server(
    category: Literal["tools", "skills", "agents"],
    enabled: list[str],
    disabled: list[str],
    effective: bool,
    pattern_driven: bool,
) -> None:
    if category != "tools":
        effective = inventory_name_matches("bash", enabled)
        pattern_driven = any(
            entry.lower() != "bash" and inventory_name_matches("bash", [entry])
            for entry in enabled
        )
    state = _inventory_item_states(
        {category: ["bash"]},
        {f"enabled_{category}": enabled, f"disabled_{category}": disabled},
    )[category]["bash"]
    assert (state.effective, state.pattern_driven) == (effective, pattern_driven)
    assert inventory_item_state("bash", enabled, disabled, category=category) == (
        effective,
        pattern_driven,
    )


@pytest.mark.parametrize(
    ("enabled", "disabled", "locked"),
    [
        (["bash", "read_file"], ["BASH"], False),
        (["bash", "read_file"], ["bas*"], True),
        (["bas*", "read_file"], ["bash"], True),
        (["bas*", "read_file"], ["b*"], True),
    ],
)
@pytest.mark.asyncio
async def test_inventory_overlapping_tools_save_and_reopen(
    enabled: list[str], disabled: list[str], locked: bool
) -> None:
    service = FakeService()
    for field in service.snapshot.fields:
        if field.path in {"enabled_tools", "disabled_tools"}:
            values = enabled if field.path == "enabled_tools" else disabled
            field.effective_value = list(values)
            field.saved_value = list(values)
            field.saved_explicit = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.selected == ["read_file"]
        assert screen._inventory_item_state("bash") == (False, locked)
        await pilot.press("space")
        assert ("bash" in checklist.selected) is not locked
        await pilot.press("down", "space")
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [
            {"enabled_tools": enabled[:-1]}
            if locked
            else {"enabled_tools": ["bash"], "disabled_tools": []}
        ]
        assert (
            service.snapshot.inventory_states["tools"]["bash"].effective is not locked
        )
        await pilot.press("enter")
        await pilot.pause()
        assert checklist.selected == ([] if locked else ["bash"])
        assert screen._inventory_item_state("bash") == (not locked, locked)


@pytest.mark.asyncio
async def test_inventory_unchanged_default_does_not_write() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert screen._expanded is None
        assert service.saved == []


@pytest.mark.asyncio
async def test_inventory_toggle_commit_discard_and_pattern_lock() -> None:
    service = FakeService()
    for field in service.snapshot.fields:
        if field.path == "enabled_tools":
            field.effective_value = ["bash", "read_*"]
            field.saved_value = ["bash", "read_*"]
            field.saved_explicit = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.selected == ["bash", "read_file"]
        assert (
            "pattern"
            in str(screen.query_one("#settings-help", NoMarkupStatic).content).lower()
        )
        await pilot.press("down", "space")
        assert checklist.selected == ["bash", "read_file"]
        await pilot.press("up", "space", "escape")
        assert screen._confirmation is not None and not service.saved
        assert "Cancel preserves both saved lists and the current draft" in str(
            screen.query_one("#settings-confirmation-text", NoMarkupStatic).content
        )
        await pilot.press("enter")  # Cancel discard; Apply explicitly persists.
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert screen._expanded is None
        assert service.saved == [{"enabled_tools": ["read_*"]}]


@pytest.mark.asyncio
async def test_inventory_last_allowed_item_restores_default() -> None:
    service = FakeService()
    for field in service.snapshot.fields:
        if field.path == "enabled_agents":
            field.effective_value = ["worker"]
            field.saved_value = ["worker"]
            field.saved_explicit = True
        if field.path == "disabled_agents":
            field.effective_value = ["worker"]
            field.saved_value = ["worker"]
            field.saved_explicit = True
    service.snapshot.inventory_states["agents"]["worker"].effective = True
    service.snapshot.inventory_states["agents"]["worker"].default_effective = False
    service.snapshot.inventory_states["agents"]["reviewer"].effective = False
    service.snapshot.inventory_states["agents"]["reviewer"].default_effective = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_agents", "enter")
        await pilot.pause()
        checklist = cast(SettingsScreen, pilot.app.screen).query_one(SettingsChecklist)
        assert checklist.selected == ["worker"]
        await pilot.press("space")
        assert checklist.selected == ["reviewer"]
        focus_action(cast(SettingsScreen, pilot.app.screen), "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"enabled_agents": []}]
        service.snapshot.inventory_states["agents"]["worker"].effective = False
        service.snapshot.inventory_states["agents"]["reviewer"].effective = True
        await pilot.press("enter")
        await pilot.pause()
        assert checklist.selected == ["reviewer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_inventory_add_pattern_space_does_not_activate(
    size: tuple[int, int],
) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_skills", "enter", "space")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        checklist = screen.query_one(SettingsChecklist)
        actions = focus_action(screen, "add-pattern")
        assert (
            actions.highlighted_option
            and actions.highlighted_option.id == "add-pattern"
        )
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["review"]}
        assert screen._inventory_draft is not None
        draft = {key: list(values) for key, values in screen._inventory_draft.items()}
        snapshot = service.snapshot.model_dump()
        expanded = screen._expanded
        selected = list(checklist.selected)
        highlighted = checklist.highlighted

        await pilot.press("space")
        await pilot.pause()

        assert pilot.app.screen is screen
        assert screen._expanded == expanded
        assert screen._editing is None
        assert checklist.display and actions.has_focus
        assert not screen.query_one("#settings-editor").display
        assert checklist.highlighted == highlighted
        assert checklist.selected == selected
        assert screen._inventory_draft == draft
        assert service.snapshot.model_dump() == snapshot
        assert not service.saved and not service.revisions

        await pilot.press("enter")
        await pilot.pause()
        assert screen.query_one("#settings-input", Input).has_focus
        assert screen.query_one("#settings-editor").display
        assert screen._inventory_draft == draft
        assert service.snapshot.model_dump() == snapshot
        assert not service.saved and not service.revisions


@pytest.mark.asyncio
async def test_inventory_pattern_add_delete_and_reset() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_skills", "enter", "space")
        screen = cast(SettingsScreen, pilot.app.screen)
        focus_action(screen, "add-pattern")
        await pilot.press("enter")
        await pilot.pause()
        assert screen.query_one("#settings-input", Input).has_focus
        await pilot.press(*"re:^my-", "enter")
        await pilot.pause()
        assert service.saved == []
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["review", "re:^my-"]}
        checklist = screen.query_one(SettingsChecklist)
        added = (screen._inventory_draft or {})["disabled"][-1]
        assert checklist.highlighted is not None
        assert (
            checklist.get_option_at_index(checklist.highlighted).value
            == f"\x00pattern:disabled:{added.token}"
        )
        assert (
            checklist.scroll_y
            <= (checklist.highlighted or 0)
            < checklist.scroll_y + checklist.scrollable_content_region.height
        )
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"disabled_skills": ["review", "re:^my-"]}]
        await pilot.press("enter", "down", "down")
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.get_option_at_index(
            checklist.highlighted or 0
        ).value.startswith("\x00pattern:")
        await pilot.press("ctrl+d", "down", "enter")
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["review"]}
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved[-1] == {"disabled_skills": ["review"]}
        await pilot.press("enter", "ctrl+r", "enter")
        assert len(service.saved) == 2
        await pilot.press("ctrl+r", "down", "enter")
        await pilot.pause()
        assert service.saved[-1] == {"enabled_skills": None, "disabled_skills": None}
        assert all(
            not field.saved_explicit
            for field in service.snapshot.fields
            if field.path in {"enabled_skills", "disabled_skills"}
        )


@pytest.mark.asyncio
async def test_pattern_add_highlights_and_reveals_new_row_in_long_checklist() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        assert screen._inventory_draft is not None
        screen._inventory_draft["disabled"] = screen._make_draft([
            f"existing-{index}-*" for index in range(40)
        ])
        screen._render_checklist()
        focus_action(screen, "add-pattern")
        await pilot.press("enter", *"new-pattern-*", "enter")
        await pilot.pause()
        entry = screen._inventory_draft["disabled"][-1]
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.highlighted is not None
        assert checklist.get_option_at_index(checklist.highlighted).value == (
            f"\x00pattern:disabled:{entry.token}"
        )
        assert checklist.scroll_y > 0
        assert (
            checklist.scroll_y
            <= checklist.highlighted
            < checklist.scroll_y + checklist.scrollable_content_region.height
        )
        assert not service.saved


@pytest.mark.asyncio
async def test_inventory_default_consecutive_unchecks_stay_unchecked() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter", "space", "down", "space")
        screen = cast(SettingsScreen, pilot.app.screen)
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["bash", "read_file"]}
        assert screen.query_one(SettingsChecklist).selected == []
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"disabled_tools": ["bash", "read_file"]}]


@pytest.mark.asyncio
async def test_inventory_pattern_deletion_recomputes_and_unlocks() -> None:
    service = FakeService()
    field = next(
        field for field in service.snapshot.fields if field.path == "disabled_tools"
    )
    field.saved_value = ["read_*"]
    field.saved_explicit = True
    field.effective_value = ["read_*"]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.selected == ["bash"]
        await pilot.press("down", "space")
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["read_*"]}
        await pilot.press("down", "ctrl+d", "down", "enter")
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": []}
        assert checklist.selected == ["bash", "read_file"]
        await pilot.press("down", "space")
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["read_file"]}


@pytest.mark.asyncio
async def test_inventory_project_values_not_copied_to_user() -> None:
    service = FakeService()
    field = next(
        field for field in service.snapshot.fields if field.path == "disabled_tools"
    )
    field.effective_value = ["project_*"]
    field.origin = "project"
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": []}
        await pilot.press("space")
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"disabled_tools": ["bash"]}]
        assert "project_*" not in str(service.saved)


@pytest.mark.asyncio
async def test_inventory_reset_uses_original_target_after_highlight_moves() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools")
        screen = cast(SettingsScreen, pilot.app.screen)
        screen.action_remove_override()
        assert screen._confirmation is not None
        options = screen.query_one(SettingsOptionList)
        options._query = "inventory_"
        screen._refresh_options()
        options.highlighted = next(
            i
            for i, option in enumerate(options.options)
            if option.id == "inventory_skills"
        )
        await pilot.press("down", "enter")
        await pilot.pause()
        assert service.saved == [{"enabled_tools": None, "disabled_tools": None}]


@pytest.mark.asyncio
async def test_inventory_reset_closes_open_pattern_editor() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        focus_action(screen, "add-pattern")
        await pilot.press("enter")
        assert screen.query_one("#settings-input", Input).has_focus
        screen.action_remove_override()
        await pilot.pause()
        assert not screen.query_one("#settings-editor").display
        await pilot.press("down", "enter")
        await pilot.pause()
        assert not screen.query_one("#settings-editor").display
        assert service.saved == [{"enabled_tools": None, "disabled_tools": None}]


@pytest.mark.asyncio
async def test_inventory_save_failure_visible_in_help() -> None:
    service = FakeService()
    service.outcome = SettingsSaveOutcome(
        "not_saved", "unchanged", error="invalid re: pattern"
    )
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_agents", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        focus_action(screen, "add-pattern")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press(*"re:[", "enter")
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        assert service.saved == [{"disabled_agents": ["re:["]}]
        assert "invalid re: pattern" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )


@pytest.mark.asyncio
async def test_prompt_enum_runtime_choices() -> None:
    service = FakeService()
    for item in service.snapshot.catalog:
        if item.path == "system_prompt_id":
            item.choices = ("cli", "custom-prompt", "explore")
    for field in service.snapshot.fields:
        if field.path == "system_prompt_id":
            field.effective_value = "custom-prompt"
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"system_prompt_id", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one("#settings-choices", OptionList)
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "choice:system_prompt_id:custom-prompt"
        )
        await pilot.press("up", "space", "down", "enter")
        await pilot.pause()
        assert service.saved == [{"system_prompt_id": "cli"}]


@pytest.mark.asyncio
async def test_invalid_numeric_input_stays_open_without_write() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"api_timeout", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        editor = screen.query_one("#settings-input", Input)
        await pilot.press("ctrl+u", "x", "enter")
        await pilot.pause()
        assert editor in screen.query(Input)
        assert editor.has_class("-invalid")
        assert "Error:" in str(
            screen.query_one("#settings-editor-error", NoMarkupStatic).content
        )
        assert not service.saved
        await pilot.press("escape")
        await pilot.pause()
        assert not screen.query_one("#settings-editor").display


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["not_saved", "exception"])
async def test_failed_list_save_retains_editable_draft_and_retry(failure: str) -> None:
    class FailingService(FakeService):
        async def save(
            self, changed_leaves: dict[str, object], expected_revision: str | None
        ) -> SettingsSaveOutcome:
            if failure == "exception" and self.outcome is not None:
                raise ValueError("invalid list")
            return await super().save(changed_leaves, expected_revision)

    service = FailingService()
    service.outcome = SettingsSaveOutcome(
        "not_saved", "unchanged", error="invalid list"
    )
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"agent_paths", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        item = next(item for item in screen.catalog if item.path == "agent_paths")
        draft = ["my-agent", "other-agent"]
        screen._list_draft = screen._make_draft(draft)
        screen._refresh_options()
        screen._save_list(item)
        await pilot.pause()
        assert screen._expanded == item.path
        assert _draft_values(screen._list_draft) == draft
        options = screen.query_one(SettingsOptionList)
        assert options.editing
        assert screen.query_one("#settings-entries").has_focus
        assert "invalid list" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
        assert not screen.fields[item.path].saved_explicit
        service.outcome = None
        screen._list_draft = screen._make_draft(["my-agent"])
        screen._save_list(item)
        await pilot.pause()
        assert service.saved[-1] == {item.path: ["my-agent"]}
        assert screen._expanded is None and screen._list_draft is None


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
        item = next(item for item in parent.catalog if item.path == "show_greeting")
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
        item = next(item for item in parent.catalog if item.path == "show_greeting")
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


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["enter", "space", "click", "double-click"])
async def test_wp4_membership_is_draft_only(gesture: str) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        checklist = screen.query_one(SettingsChecklist)
        if "click" in gesture:
            await pilot.click(
                checklist, offset=(5, 0), times=2 if gesture == "double-click" else 1
            )
        else:
            await pilot.press(gesture)
        assert _draft_values((screen._inventory_draft or {})["disabled"]) == ["bash"]
        assert checklist.has_focus
        assert not service.saved
        actions = focus_action(screen, "apply")
        await pilot.press("space")
        assert not service.saved and screen._editing is None
        await pilot.click(actions, offset=(5, 1), times=2)
        await pilot.pause()
        assert service.saved == [{"disabled_tools": ["bash"]}]
        assert screen._expanded is None


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["enter", "space", "click"])
@pytest.mark.parametrize("row", ["pattern", "locked", "state"])
async def test_wp4_inert_inventory_rows(row: str, gesture: str) -> None:
    service = FakeService()
    field = next(f for f in service.snapshot.fields if f.path == "disabled_tools")
    if row != "state":
        field.saved_value = ["read_*"]
        field.saved_explicit = True
    else:
        service.snapshot.inventories["tools"] = []
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        checklist = screen.query_one(SettingsChecklist)
        checklist.highlighted = 2 if row == "pattern" else 1 if row == "locked" else 0
        await pilot.pause()
        before = navigation_payload(screen)
        if gesture == "click":
            await pilot.click(checklist, offset=(5, checklist.highlighted or 0))
        else:
            await pilot.press(gesture)
        assert navigation_payload(screen) == before
        assert not service.saved


@pytest.mark.asyncio
@pytest.mark.parametrize("pointer", [False, True])
async def test_wp4_radio_accepts_selected_not_cursor(pointer: bool) -> None:
    service = FakeService()
    descriptor = next(
        d for d in service.snapshot.catalog if d.path == "system_prompt_id"
    )
    descriptor.choices = ("first", "second")
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"system_prompt_id", "enter")
        choices = screen.query_one("#settings-choices", OptionList)
        if pointer:
            await pilot.click(choices, offset=(5, 0), times=2)
        else:
            await pilot.press("space")
        assert screen._enum_draft == "first" and not service.saved
        await pilot.press("down")
        assert choices.highlighted == 1
        if pointer:
            assert "Accept selected" in str(
                screen.query_one("#settings-hint", NoMarkupStatic).content
            )
            await pilot.click("#settings-hint", offset=(8, 0))
        else:
            await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"system_prompt_id": "first"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("pointer", [False, True])
@pytest.mark.parametrize(
    "path",
    ["agent_paths", "displayed_workdir", "auto_compact_threshold", "show_greeting"],
)
async def test_wp4_editor_and_scalar_pointer_parity(path: str, pointer: bool) -> None:
    service = FakeService()
    if path == "agent_paths":
        next(f for f in service.snapshot.fields if f.path == path).effective_value = [
            "original"
        ]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index(path)
        await pilot.pause()
        if pointer:
            catalog.scroll_to_highlight()
            await pilot.pause()
            y = int(catalog.highlighted or 0) - int(catalog.scroll_y)
            await pilot.click(catalog, offset=(5, y))
            assert catalog.has_focus and not service.saved
            await pilot.click(catalog, offset=(5, y), times=2)
        else:
            await pilot.press("enter")
        if path == "show_greeting":
            await pilot.pause()
            assert service.saved == [{path: True}]
            return
        if path == "agent_paths":
            entries = screen.query_one("#settings-entries", OptionList)
            if pointer:
                await pilot.click(entries, offset=(5, 0))
                assert entries.has_focus and screen._editing is None
                await pilot.click(entries, offset=(5, 0), times=2)
            else:
                await pilot.press("enter")
        editor = screen.query_one("#settings-input", Input)
        assert editor.has_focus
        editor.value = "3" if path == "auto_compact_threshold" else "replacement value"
        if pointer:
            await pilot.click("#settings-hint", offset=(8, 0))
        else:
            await pilot.press("enter")
        await pilot.pause()
        if path == "agent_paths":
            assert (
                _draft_values(screen._list_draft) == ["replacement value"]
                and not service.saved
            )
            actions = focus_action(screen, "apply")
            if pointer:
                await pilot.click(actions, offset=(5, 1), times=2)
            else:
                await pilot.press("enter")
            await pilot.pause()
        assert service.saved == [
            {
                path: ["replacement value"]
                if path == "agent_paths"
                else 3
                if path == "auto_compact_threshold"
                else "replacement value"
            }
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("failure", ["not_saved", "conflict", "exception", "unknown"])
async def test_wp4_inventory_failure_retains_identity_opener_and_scroll(
    reset: bool, failure: str
) -> None:
    class FailingService(FakeService):
        async def save(
            self, changed_leaves: dict, expected_revision: str | None
        ) -> SettingsSaveOutcome:
            if failure == "exception":
                self.saved.append(changed_leaves)
                raise RuntimeError("transport failed")
            return await super().save(changed_leaves, expected_revision)

    service = FailingService()
    service.outcome = (
        SettingsSaveOutcome("saved", "applied")
        if failure == "unknown"
        else SettingsSaveOutcome(
            "not_saved",
            "unchanged",
            error="conflict" if failure == "conflict" else "write failed",
        )
    )
    service.snapshot.inventories["tools"] = [f"member-{i}" for i in range(30)]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter", "enter")
        checklist = screen.query_one(SettingsChecklist)
        checklist.highlighted = 20
        await pilot.pause()
        checklist.scroll_to(y=18, animate=False, immediate=True, force=True)
        await pilot.pause()
        position = screen._capture_group_position(checklist)
        draft = {
            side: list(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        }
        identities = [o.id for o in checklist.options]
        if reset:
            focus_action(screen, "add-pattern")
            await pilot.press("enter", "ctrl+r")
            assert screen._editing is None  # Explicit editor-reset remains supported.
            await pilot.press("down", "enter")
        else:
            focus_action(screen, "apply")
            await pilot.press("enter")
        await pilot.pause()
        assert len(service.saved) == 1 and not screen._busy
        if failure == "unknown":
            assert screen._inventory_draft is None and screen._expanded is None
            assert screen._needs_refresh and "current state unknown" in screen._error
        else:
            assert (
                screen._inventory_draft == draft
                and screen._expanded == "inventory_tools"
            )
            assert [o.id for o in checklist.options] == identities
            assert position is not None and checklist.scroll_y == position.scroll_y
            actions = screen.query_one("#settings-actions", OptionList)
            assert actions.has_focus and actions.highlighted_option is not None
            assert actions.highlighted_option.id == (
                "add-pattern" if reset else "apply"
            )
            assert screen._needs_refresh is (failure == "conflict")
            if failure == "conflict":
                before = navigation_payload(screen)
                screen._toggle_inventory("member-1")
                screen._save_inventory(
                    next(d for d in screen.catalog if d.path == screen._expanded)
                )
                assert navigation_payload(screen) == before and len(service.saved) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["inventory_tools", "agent_paths"])
async def test_wp4_apply_reserves_busy_before_worker_and_blocks_mutation(
    path: str,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class DelayedService(FakeService):
        async def save(
            self, changed_leaves: dict, expected_revision: str | None
        ) -> SettingsSaveOutcome:
            started.set()
            await release.wait()
            return await super().save(changed_leaves, expected_revision)

    service = DelayedService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*path, "enter")
        if path == "inventory_tools":
            await pilot.press("space")
        else:
            screen._list_draft = screen._make_draft(["pending"])
            screen._refresh_options()
        actions = focus_action(screen, "apply")
        item = next(d for d in screen.catalog if d.path == path)
        screen._apply_collection(item)
        screen._apply_collection(item)  # No await: second activation precedes startup.
        assert screen._busy and "Wait for save" in str(
            screen.query_one("#settings-hint", NoMarkupStatic).content
        )
        await started.wait()
        before = navigation_payload(screen)
        await pilot.press("escape", "enter", "space", "ctrl+r", "ctrl+d", "f1")
        await pilot.click(actions, offset=(5, 1), times=2)
        screen._toggle_inventory("bash")
        assert pilot.app.screen is screen and screen._expanded == path
        assert navigation_payload(screen) == before and not service.saved
        release.set()
        await pilot.app.workers.wait_for_complete()
        assert len(service.saved) == 1 and screen._expanded is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner", ["help", "confirmation", "editor", "view_only", "needs_refresh"]
)
async def test_wp4_mutation_guards_cover_keyboard_and_pointer(owner: str) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        checklist = screen.query_one(SettingsChecklist)
        if owner == "help":
            await pilot.press("f1")
        elif owner == "confirmation":
            await pilot.press("ctrl+r")
        elif owner == "editor":
            focus_action(screen, "add-pattern")
            await pilot.press("enter")
        elif owner == "view_only":
            screen.snapshot.user_revision = None
        else:
            screen._needs_refresh = True
        before = navigation_payload(screen)
        screen._toggle_inventory("bash")
        checklist.action_select()
        screen._apply_collection(
            next(d for d in screen.catalog if d.path == "inventory_tools")
        )
        screen.action_delete_item()
        if owner != "editor":
            screen.action_remove_override()
        if checklist.display:
            await pilot.click(checklist, offset=(5, 0), times=2)
        assert navigation_payload(screen) == before and not service.saved


def pointer_shortcut_offset(
    screen: SettingsScreen, identifier: str, action: str
) -> tuple[int, int]:
    widget = screen.query_one(identifier, SettingsHints)
    for y in range(widget.region.height):
        for x in range(widget.region.width):
            if widget.pointer_action_at(x, y) == action:
                return x, y
    raise AssertionError(f"No visible pointer target for {action} in {identifier}")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["agent_paths", "inventory_tools"])
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
async def test_wp4_hover_and_wheel_only_affect_pointed_region() -> None:
    service = FakeService()
    next(
        f for f in service.snapshot.fields if f.path == "agent_paths"
    ).effective_value = [f"item-{i}" for i in range(40)]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index("agent_paths")
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


@pytest.mark.asyncio
async def test_wp4_unfiltered_apply_double_click_does_not_activate_new_catalog_row() -> (
    None
):
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index("agent_paths")
        await pilot.press("enter")
        screen._list_draft = screen._make_draft(["new"])
        screen._refresh_options()
        actions = focus_action(screen, "apply")
        await pilot.pause()
        await pilot.click(actions, offset=(5, 1), times=2)
        await pilot.pause()
        assert service.saved == [{"agent_paths": ["new"]}]
        assert (
            screen._expanded is None
            and screen._editing is None
            and screen._confirmation is None
        )


@pytest.mark.asyncio
async def test_review_confirmation_apply_closes_help_before_save() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        item = next(item for item in screen.catalog if item.path == "show_greeting")
        screen._open_confirmation("value", item, True, "Confirm change", "Apply")
        await pilot.press("f1")
        assert screen._help_open
        await pilot.click("#settings-confirmation-actions", offset=(3, 1), times=2)
        await pilot.pause()
        assert not screen._help_open and screen._confirmation is None
        assert service.saved == [{"show_greeting": True}]


@pytest.mark.asyncio
async def test_review_boolean_override_pointer_reset_at_minimum_size() -> None:
    service = FakeService()
    field = next(f for f in service.snapshot.fields if f.path == "show_greeting")
    field.saved_explicit = True
    field.effective_value = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"show_greeting", "f1")
        await pilot.click(
            "#settings-help",
            offset=pointer_shortcut_offset(screen, "#settings-help", "remove_override"),
        )
        await pilot.pause()
        assert not screen._help_open and screen._confirmation is not None
        await pilot.click("#settings-confirmation-actions", offset=(3, 1))
        await pilot.pause()
        assert service.saved == [{"show_greeting": None}]


@pytest.mark.asyncio
async def test_review_narrow_enum_accept_pointer_target() -> None:
    service = FakeService()
    next(
        i for i in service.snapshot.catalog if i.path == "system_prompt_id"
    ).choices = ("first", "second")
    async with Harness(service).run_test(size=(50, 20)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"system_prompt_id", "enter")
        await pilot.click("#settings-choices", offset=(3, 1))
        assert screen._enum_draft == "second" and not service.saved
        await pilot.click(
            "#settings-hint",
            offset=pointer_shortcut_offset(
                screen, "#settings-hint", "activate_focused"
            ),
        )
        await pilot.pause()
        assert service.saved == [{"system_prompt_id": "second"}]


@pytest.mark.asyncio
async def test_review_added_pattern_bookmark_reentry() -> None:
    service = FakeService()
    field = next(f for f in service.snapshot.fields if f.path == "disabled_tools")
    field.saved_value = [f"existing-{index}-*" for index in range(40)]
    field.saved_explicit = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        focus_action(screen, "add-pattern")
        await pilot.press("enter")
        screen.query_one("#settings-input", Input).value = "custom-*"
        await pilot.press("enter")
        checklist = screen.query_one(SettingsChecklist)
        opener = checklist.highlighted_option
        assert opener is not None and str(opener.id).startswith("pattern:")
        added = (screen._inventory_draft or {})["disabled"][-1]
        assert opener.id == f"pattern:inventory_tools:disabled:{added.token}"
        scroll = checklist.scroll_y
        assert scroll > 0
        await pilot.press("shift+tab")
        assert checklist.has_focus and checklist.highlighted_option is opener
        assert checklist.scroll_y == scroll
        assert checklist.highlighted is not None
        assert (
            checklist.scroll_y
            <= checklist.highlighted
            < checklist.scroll_y + checklist.scrollable_content_region.height
        )
        await pilot.press("shift+tab", "tab")
        assert checklist.has_focus and checklist.highlighted_option is opener
        assert checklist.scroll_y == scroll
        assert not service.saved


@pytest.mark.asyncio
async def test_review_double_cancel_preserves_pattern_opener() -> None:
    service = FakeService()
    field = next(f for f in service.snapshot.fields if f.path == "disabled_tools")
    field.saved_value = ["long-pattern-" * 40]
    field.saved_explicit = True
    service.snapshot.inventories["tools"] = [f"member-{i}" for i in range(20)]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        checklist = screen.query_one(SettingsChecklist)
        checklist.highlighted = checklist.option_count - 1
        await pilot.pause()
        # Inspect earlier members without changing the selected pattern opener.
        checklist.scroll_to(y=0, animate=False, immediate=True, force=True)
        await pilot.pause()
        opener = checklist.highlighted_option
        assert opener is not None and str(opener.id).startswith("pattern:")
        payload = navigation_payload(screen)
        await pilot.press("ctrl+d")
        assert screen._confirmation is not None
        await pilot.click("#settings-confirmation-actions", offset=(3, 0), times=2)
        await pilot.pause()
        assert screen._confirmation is None and checklist.has_focus
        assert checklist.highlighted_option is opener
        assert navigation_payload(screen) == payload and not service.saved


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


@pytest.mark.asyncio
async def test_review_clean_enum_enter_collapses_without_override() -> None:
    service = FakeService()
    next(
        i for i in service.snapshot.catalog if i.path == "system_prompt_id"
    ).choices = ("first", "second")
    next(
        f for f in service.snapshot.fields if f.path == "system_prompt_id"
    ).effective_value = "first"
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"system_prompt_id", "enter", "down", "enter")
        assert screen._expanded is None and not service.saved
        assert not screen.fields["system_prompt_id"].saved_explicit
