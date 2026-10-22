"""Confirmation dialogs, dirty discard, conflicts, save failures, apply and mutation guards, and override removal tests for the settings app suite."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from textual.content import Content
from textual.widgets import OptionList

from chartreux.cli.textual_ui.screens.settings import (
    ConfirmationText,
    SettingsChecklist,
    SettingsOptionList,
    SettingsScreen,
    _draft_values,
)
from chartreux.ui.settings_service import SettingsSaveOutcome
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.cli.textual_ui._settings_app_support import (
    FakeService,
    Harness,
    focus_action,
    navigation_payload,
    pointer_shortcut_offset,
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
        assert service.saved == [{"autocopy_to_clipboard": True}]


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
        assert service.saved[-1] == {"autocopy_to_clipboard": True}


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
        assert service.saved == [{"autocopy_to_clipboard": None}]


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
        options.highlighted = options.get_option_index("skill_paths")
        await pilot.press("enter", "tab", "enter")
        await pilot.press(*"new-entry", "enter")
        assert _draft_values(screen._list_draft) == ["new-entry"]

        await pilot.press("tab")
        assert options.has_focus
        options.highlighted = next(
            index
            for index, row in enumerate(options.options)
            if row.id == "ask_confirmation_on_exit"
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
        assert service.saved == [{"ask_confirmation_on_exit": True}]


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
        await pilot.press(*"skill_paths", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        item = next(item for item in screen.catalog if item.path == "skill_paths")
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
@pytest.mark.parametrize("path", ["inventory_tools", "skill_paths"])
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


@pytest.mark.asyncio
async def test_wp4_unfiltered_apply_double_click_does_not_activate_new_catalog_row() -> (
    None
):
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        catalog = screen.query_one(SettingsOptionList)
        catalog.highlighted = catalog.get_option_index("skill_paths")
        await pilot.press("enter")
        screen._list_draft = screen._make_draft(["new"])
        screen._refresh_options()
        actions = focus_action(screen, "apply")
        await pilot.pause()
        await pilot.click(actions, offset=(5, 1), times=2)
        await pilot.pause()
        assert service.saved == [{"skill_paths": ["new"]}]
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
        item = next(
            item for item in screen.catalog if item.path == "ask_confirmation_on_exit"
        )
        screen._open_confirmation("value", item, True, "Confirm change", "Apply")
        await pilot.press("f1")
        assert screen._help_open
        await pilot.click("#settings-confirmation-actions", offset=(3, 1), times=2)
        await pilot.pause()
        assert not screen._help_open and screen._confirmation is None
        assert service.saved == [{"ask_confirmation_on_exit": True}]


@pytest.mark.asyncio
async def test_review_boolean_override_pointer_reset_at_minimum_size() -> None:
    service = FakeService()
    field = next(
        f for f in service.snapshot.fields if f.path == "ask_confirmation_on_exit"
    )
    field.saved_explicit = True
    field.effective_value = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"ask_confirmation_on_exit", "f1")
        await pilot.click(
            "#settings-help",
            offset=pointer_shortcut_offset(screen, "#settings-help", "remove_override"),
        )
        await pilot.pause()
        assert not screen._help_open and screen._confirmation is not None
        await pilot.click("#settings-confirmation-actions", offset=(3, 1))
        await pilot.pause()
        assert service.saved == [{"ask_confirmation_on_exit": None}]
