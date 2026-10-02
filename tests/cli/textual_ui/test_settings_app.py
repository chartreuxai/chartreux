from __future__ import annotations

from dataclasses import asdict
from typing import Literal, cast

import pytest
from textual.app import App
from textual.content import Content
from textual.widgets import Input, OptionList

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
    SettingsOptionList,
    SettingsScreen,
    inventory_item_state,
    inventory_name_matches,
    parse_setting_value,
    toggle_inventory_name,
)
from chartreux.core.config.settings_catalog import (
    DEFERRED_SETTINGS,
    EDITABLE_BY_PATH,
    LINK_SETTINGS,
    VISIBLE_SETTINGS,
)
from chartreux.core.utils.matching import name_matches
from chartreux.ui.settings_service import SettingsSaveOutcome, SettingsService
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


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
                    effective_value=False
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
                        else ([] if isinstance(field.effective_value, list) else False),
                        "saved_explicit": value is not None,
                        "saved_value": value,
                        "origin": "user" if value is not None else "default",
                    }
                )
            fields.append(field)
        self.snapshot = self.snapshot.model_copy(update={"fields": fields})
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
    async with Harness(FakeService()).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one(SettingsOptionList)
        hint = screen.query_one("#settings-hint", NoMarkupStatic)
        assert "Ctrl+D" not in str(hint.content)
        await pilot.press("f1")
        assert screen.query_one("#settings-help").has_focus
        assert "Ctrl+D" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
        await pilot.press("escape")
        assert options.has_focus


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
        assert screen.query_one(SettingsOptionList).has_focus


@pytest.mark.asyncio
async def test_checklist_action_rows_have_no_membership_and_hints_follow_cursor() -> (
    None
):
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter", "down", "down")
        checklist = screen.query_one(SettingsChecklist)
        line = "".join(segment.text for segment in checklist.render_line(2))
        assert "Add Pattern" in line and "[ ]" not in line
        assert "Add pattern" in str(
            screen.query_one("#settings-hint", NoMarkupStatic).content
        )
        await pilot.press("up")
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
        await pilot.press("tab")
        assert confirmation.has_focus
        for _ in range(18):
            await pilot.press("down")
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
async def test_settings_provider_link_returns_to_settings(
    monkeypatch: pytest.MonkeyPatch,
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
        await pilot.press(*"models/providers", "enter")
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
        assert options._query == "models/providers"
        assert options.highlighted_option is not None
        assert options.highlighted_option.id == "models/providers"
        assert app.screen.focused is options


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
        options = screen.query_one(SettingsOptionList)
        choice = options.highlighted_option
        assert choice is not None and choice.id == "choice:ui_color_scheme:dark"
        assert options._query == "ui_color_scheme"
        assert "Save choice to user settings" in str(
            screen.query_one("#settings-hint", NoMarkupStatic).content
        )
        await pilot.press("escape")
        await pilot.pause()
        assert not any(str(row.id).startswith("choice:") for row in options.options)
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
        options = screen.query_one(SettingsOptionList)
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "list:agent_paths:add"
        )
        assert "Empty uses built-in agent locations" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )
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
        await pilot.press("enter", "enter")
        await pilot.pause()
        await pilot.press(*"  tool-*  ", "enter")
        await pilot.pause()
        assert screen._list_draft == ["tool-*"] and not service.saved
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert service.saved == [{"agent_paths": ["tool-*"]}]
        assert screen._expanded is None
        await pilot.press("enter")
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "list:agent_paths:0"
        )
        await pilot.press("down", "enter")
        await pilot.press(*"  other  ", "enter")
        assert screen._list_draft == ["tool-*", "other"] and len(service.saved) == 1
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert service.saved[-1] == {"agent_paths": ["tool-*", "other"]}
        await pilot.press("enter", "ctrl+d", "down", "enter")
        assert screen._list_draft == ["other"] and len(service.saved) == 2
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert service.saved[-1] == {"agent_paths": ["other"]}
        await pilot.press("enter", "enter", "ctrl+u", "space", "enter")
        assert screen.query_one("#settings-input", Input).has_class("-invalid")
        assert screen._list_draft == ["other"] and len(service.saved) == 3
        await pilot.press("escape", "ctrl+d", "down", "enter")
        assert screen._list_draft == [] and len(service.saved) == 3
        options.highlighted = next(
            i
            for i, row in enumerate(options.options)
            if row.id == "list:agent_paths:apply"
        )
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
        await pilot.press(*"agent_paths", "enter", "enter")
        await pilot.press(*"new-entry", "enter")
        assert screen._list_draft == ["new-entry"]

        options._query = ""
        screen._filter("")
        options.highlighted = next(
            index
            for index, row in enumerate(options.options)
            if row.id == "show_greeting"
        )
        await pilot.press("enter")
        assert screen._confirmation is not None
        await pilot.press("escape")
        assert screen._list_draft == ["new-entry"]
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
        await pilot.press(*"agent_paths", "enter", "enter")
        await pilot.press(*"new-entry", "enter")
        options._query = ""
        screen._filter("")
        options.highlighted = next(
            index
            for index, row in enumerate(options.options)
            if row.id == "show_greeting"
        )
        await pilot.press("enter", "escape")
        assert screen._list_draft == ["new-entry"]

        options.highlighted = next(
            index
            for index, row in enumerate(options.options)
            if row.id == "list:agent_paths:apply"
        )
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"agent_paths": ["new-entry"]}]

        options.highlighted = next(
            index
            for index, row in enumerate(options.options)
            if row.id == "show_greeting"
        )
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
        await pilot.press("down", "space", "enter")
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
        await pilot.press(*"inventory_tools", "enter", "enter")
        await pilot.pause()
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
        await pilot.press("enter", "down", "enter")
        assert screen._expanded is None
        await pilot.press("enter", "enter")
        await pilot.pause()
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
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"enabled_agents": []}]
        service.snapshot.inventory_states["agents"]["worker"].effective = False
        service.snapshot.inventory_states["agents"]["reviewer"].effective = True
        await pilot.press("enter")
        await pilot.pause()
        assert checklist.selected == ["reviewer"]


@pytest.mark.asyncio
async def test_inventory_pattern_add_delete_and_reset() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(
            *"inventory_skills", "enter", "space", "down", "down", "enter"
        )
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        assert screen.query_one("#settings-input", Input).has_focus
        await pilot.press(*"re:^my-", "enter")
        await pilot.pause()
        assert service.saved == []
        assert screen._inventory_draft == {
            "enabled": [],
            "disabled": ["review", "re:^my-"],
        }
        await pilot.press("up", "enter")
        await pilot.pause()
        assert service.saved == [{"disabled_skills": ["review", "re:^my-"]}]
        await pilot.press("enter", "down", "down")
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.get_option_at_index(
            checklist.highlighted or 0
        ).value.startswith("\x00pattern:")
        await pilot.press("ctrl+d", "down", "enter")
        assert screen._inventory_draft == {"enabled": [], "disabled": ["review"]}
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
async def test_inventory_default_consecutive_unchecks_stay_unchecked() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter", "space", "down", "space")
        screen = cast(SettingsScreen, pilot.app.screen)
        assert screen._inventory_draft == {
            "enabled": [],
            "disabled": ["bash", "read_file"],
        }
        assert screen.query_one(SettingsChecklist).selected == []
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
        assert screen._inventory_draft == {"enabled": [], "disabled": ["read_*"]}
        await pilot.press("down", "ctrl+d", "down", "enter")
        assert screen._inventory_draft == {"enabled": [], "disabled": []}
        assert checklist.selected == ["bash", "read_file"]
        await pilot.press("down", "space")
        assert screen._inventory_draft == {"enabled": [], "disabled": ["read_file"]}


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
        assert screen._inventory_draft == {"enabled": [], "disabled": []}
        await pilot.press("space", "enter")
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
        await pilot.press(*"inventory_tools", "enter", "down", "down", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        assert screen.query("#settings-input")
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
        await pilot.press(*"inventory_agents", "enter", "down", "down", "enter")
        await pilot.pause()
        await pilot.press(*"re:[", "enter", "up", "enter")
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
        options = screen.query_one(SettingsOptionList)
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
