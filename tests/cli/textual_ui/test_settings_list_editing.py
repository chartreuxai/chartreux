"""List editing draft tokens, identity, deletion, collection state, and add/edit/delete/apply tests for the settings app suite."""

from __future__ import annotations

from typing import cast

import pytest
from textual.widgets import Input, OptionList

from chartreux.cli.textual_ui.screens.settings import (
    SettingsChecklist,
    SettingsOptionList,
    SettingsScreen,
    _draft_values,
)
from chartreux.ui.settings_service import SettingsService
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.cli.textual_ui._settings_app_support import (
    FakeService,
    Harness,
    focus_action,
)


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
        await pilot.press(*"skill_paths", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        item = next(item for item in screen.catalog if item.path == "skill_paths")
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
    path = "inventory_tools" if inventory else "skill_paths"
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
    path = "inventory_tools" if inventory else "skill_paths"
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
        ("skill_paths", "settings-entries"),
        ("inventory_tools", "settings-checklist"),
        ("agent.identity", "settings-choices"),
    ],
)
async def test_expanded_groups_and_minimum_size_layout(path: str, body_id: str) -> None:
    service = FakeService()
    if path == "skill_paths":
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
        if path != "agent.identity":
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
        ("skill_paths", "settings-entries"),
        ("inventory_tools", "settings-checklist"),
        ("agent.identity", "settings-choices"),
    ],
)
async def test_empty_collection_state_rows_are_focusable_and_inert(
    path: str, body_id: str
) -> None:
    service = FakeService()
    service.snapshot.inventories["tools"] = []
    if path == "agent.identity":
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
async def test_list_add_edit_delete_discard_and_empty_help() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"skill_paths", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one("#settings-entries", OptionList)
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "state:entries"
        )
        assert "Empty uses built-in skill locations" in str(
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
        assert screen._expanded == "skill_paths"
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
        assert service.saved == [{"skill_paths": ["tool-*"]}]
        assert screen._expanded is None
        await pilot.press("enter")
        assert (
            options.highlighted_option
            and screen._list_draft
            and options.highlighted_option.id
            == f"list:skill_paths:{screen._list_draft[0].token}"
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
        assert service.saved[-1] == {"skill_paths": ["tool-*", "other"]}
        await pilot.press("enter", "home", "ctrl+d", "down", "enter")
        assert (
            _draft_values(screen._list_draft) == ["other"] and len(service.saved) == 2
        )
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved[-1] == {"skill_paths": ["other"]}
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
        assert service.saved[-1] == {"skill_paths": []}


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
        options.highlighted = options.get_option_index("skill_paths")
        await pilot.press("enter", "tab", "enter")
        await pilot.press(*"new-entry", "enter")
        await pilot.press("tab")
        assert options.has_focus
        options.highlighted = next(
            index
            for index, row in enumerate(options.options)
            if row.id == "ask_confirmation_on_exit"
        )
        options.focus()
        await pilot.press("enter", "escape")
        assert _draft_values(screen._list_draft) == ["new-entry"]

        await pilot.press("shift+tab", "down", "enter")
        await pilot.pause()
        assert service.saved == [{"skill_paths": ["new-entry"]}]
        assert options.highlighted_option is not None
        assert options.highlighted_option.id == "ask_confirmation_on_exit"
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved[-1] == {"ask_confirmation_on_exit": True}
