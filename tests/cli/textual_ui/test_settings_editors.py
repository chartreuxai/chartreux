"""Inline editor tests for the settings app suite (bool, string, numeric, enum, list, pointer parity, shell permissions)."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import cast

import pytest
from textual.app import App
from textual.widgets import Input, OptionList

from chartreux.app_server.protocol import SettingDescriptorWire, SettingLeafWire
from chartreux.cli.textual_ui.screens.settings import (
    SettingsChecklist,
    SettingsOptionList,
    SettingsScreen,
    _draft_values,
    parse_setting_value,
)
from chartreux.core.config.settings_catalog import EDITABLE_BY_PATH
from chartreux.ui.settings_service import SettingsService
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.cli.textual_ui._settings_app_support import (
    FakeService,
    Harness,
    focus_action,
    pointer_shortcut_offset,
)


@pytest.mark.asyncio
async def test_bool_toggle_and_help_on_highlight() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        help_row = screen.query_one("#settings-help", NoMarkupStatic)
        assert "clipboard" in str(help_row.content)
        assert "Not Set" in str(help_row.content)
        assert "saves the toggle to user settings" in str(help_row.content)
        await pilot.press("down")
        await pilot.pause()
        assert "closing the conversation" in str(help_row.content)
        await pilot.press("up", "enter")
        await pilot.pause()
        assert service.saved == [{"autocopy_to_clipboard": True}]
        assert screen._display_value(service.snapshot.catalog[0]) == "[■]"
        assert "Saved user value: True" in str(help_row.content)
        await pilot.press("space")
        await pilot.pause()
        assert service.saved[-1] == {"autocopy_to_clipboard": False}


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
        assert options.highlighted_option.id == "autocopy_to_clipboard"
        assert options.has_focus
        assert service.saved == []
        await pilot.click(options, offset=(5, 1), times=2)
        await pilot.pause()
        assert service.saved == [{"autocopy_to_clipboard": True}]


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
async def test_view_only_prevents_edit_and_remove() -> None:
    service = FakeService()
    service.snapshot.user_revision = None
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "ctrl+r")
        await pilot.pause()
        assert not service.saved


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
async def test_identity_enum_runtime_choices() -> None:
    service = FakeService()
    for item in service.snapshot.catalog:
        if item.path == "agent.identity":
            item.choices = ("coder", "orchestrator", "researcher")
    for field in service.snapshot.fields:
        if field.path == "agent.identity":
            field.effective_value = "orchestrator"
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"agent.identity", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        options = screen.query_one("#settings-choices", OptionList)
        assert (
            options.highlighted_option
            and options.highlighted_option.id == "choice:agent.identity:orchestrator"
        )
        await pilot.press("up", "space", "down", "enter")
        await pilot.pause()
        assert service.saved == [{"agent.identity": "coder"}]


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
@pytest.mark.parametrize("pointer", [False, True])
async def test_wp4_radio_accepts_selected_not_cursor(pointer: bool) -> None:
    service = FakeService()
    descriptor = next(d for d in service.snapshot.catalog if d.path == "agent.identity")
    descriptor.choices = ("first", "second")
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"agent.identity", "enter")
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
        assert service.saved == [{"agent.identity": "first"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("pointer", [False, True])
@pytest.mark.parametrize(
    "path",
    [
        "skill_paths",
        "displayed_workdir",
        "auto_compact_threshold",
        "ask_confirmation_on_exit",
    ],
)
async def test_wp4_editor_and_scalar_pointer_parity(path: str, pointer: bool) -> None:
    service = FakeService()
    if path == "skill_paths":
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
        if path == "ask_confirmation_on_exit":
            await pilot.pause()
            assert service.saved == [{path: True}]
            return
        if path == "skill_paths":
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
        if path == "skill_paths":
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
                if path == "skill_paths"
                else 3
                if path == "auto_compact_threshold"
                else "replacement value"
            }
        ]


@pytest.mark.asyncio
async def test_review_narrow_enum_accept_pointer_target() -> None:
    service = FakeService()
    next(i for i in service.snapshot.catalog if i.path == "agent.identity").choices = (
        "first",
        "second",
    )
    async with Harness(service).run_test(size=(50, 20)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"agent.identity", "enter")
        await pilot.click("#settings-choices", offset=(3, 1))
        assert screen._enum_draft == "second" and not service.saved
        await pilot.click(
            "#settings-hint",
            offset=pointer_shortcut_offset(
                screen, "#settings-hint", "activate_focused"
            ),
        )
        await pilot.pause()
        assert service.saved == [{"agent.identity": "second"}]


@pytest.mark.asyncio
async def test_review_clean_enum_enter_collapses_without_override() -> None:
    service = FakeService()
    next(i for i in service.snapshot.catalog if i.path == "agent.identity").choices = (
        "first",
        "second",
    )
    next(
        f for f in service.snapshot.fields if f.path == "agent.identity"
    ).effective_value = "first"
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"agent.identity", "enter", "down", "enter")
        assert screen._expanded is None and not service.saved
        assert not screen.fields["agent.identity"].saved_explicit


class FakeAuthorityResource:
    """Minimal P5 resource double for the Shell Permissions panel."""

    def __init__(self) -> None:
        from chartreux.app_server.protocol import (
            DangerousCommandAuthorityReadResponse,
            DangerousCommandGrantScopeWire,
            DangerousCommandGrantWire,
            DangerousCommandRevocationOutcome,
            DangerousCommandRevokeResponse,
        )

        self.inspection = DangerousCommandAuthorityReadResponse(
            project="/home/example/projects/shop",
            live_authority_generation=1,
            live_grants=[
                DangerousCommandGrantWire(
                    operation_id="git.push",
                    scope=DangerousCommandGrantScopeWire(
                        git_dir="/home/example/projects/shop/.git",
                        work_tree="/home/example/projects/shop",
                    ),
                )
            ],
            accepted_saved_grants=[],
            saved_operations=[],
            disk_revision="rev-1",
            blocking_denylist=[],
        )
        self.revoke = DangerousCommandRevokeResponse(
            live=DangerousCommandRevocationOutcome(status="removed", generation=2),
            saved=DangerousCommandRevocationOutcome(status="skipped"),
            live_authority_generation=2,
        )
        self.revoke_calls: list[dict[str, object]] = []

    async def read_dangerous_command_authority(self):
        return self.inspection

    async def revoke_dangerous_command_grant(
        self,
        *,
        operation_id: str,
        expected_live_generation: int,
        expected_disk_revision: str | None,
        user_initiated: bool,
    ):
        self.revoke_calls.append({
            "operation_id": operation_id,
            "expected_live_generation": expected_live_generation,
            "expected_disk_revision": expected_disk_revision,
            "user_initiated": user_initiated,
        })
        return self.revoke


class ShellPermissionsHarness(App[None]):
    def __init__(self, service: FakeService, shell_permissions) -> None:
        super().__init__()
        self.service = service
        self.shell_permissions = shell_permissions

    def on_mount(self) -> None:
        self.push_screen(
            SettingsScreen(
                cast(SettingsService, self.service),
                self.service.snapshot,
                shell_permissions=self.shell_permissions,
            )
        )


@pytest.mark.asyncio
async def test_shell_permissions_row_uses_label_and_panel_help() -> None:
    from chartreux.ui.shell_permissions import ShellPermissionsService

    resource = FakeAuthorityResource()
    async with ShellPermissionsHarness(
        FakeService(), ShellPermissionsService(resource)
    ).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        item = next(item for item in screen.catalog if item.path == "shell_permissions")
        assert item.control == "shell_permissions"
        assert item.kind == "deferred"
        # The row shows the human label, not the bare path.
        row = screen._row(item, selected=False).plain
        assert row.split("    ")[0].strip() == "Shell Permissions"
        options = screen.query_one(SettingsOptionList)
        options.highlighted = options.get_option_index("shell_permissions")
        await pilot.pause()
        help_text = str(screen.query_one("#settings-help", NoMarkupStatic).content)
        assert "Shell Permissions panel" in help_text
        assert ("Enter", "Open") in screen._available_actions()


@pytest.mark.asyncio
async def test_shell_permissions_enter_opens_panel_and_return_restores_state() -> None:
    from chartreux.cli.textual_ui.widgets.shell_permissions_app import (
        ShellPermissionsApp,
    )
    from chartreux.ui.shell_permissions import ShellPermissionsService

    resource = FakeAuthorityResource()
    app = ShellPermissionsHarness(FakeService(), ShellPermissionsService(resource))
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"shell")
        await pilot.pause()
        options = screen.query_one(SettingsOptionList)
        assert options._query == "shell"
        assert options.highlighted_option is not None
        assert str(options.highlighted_option.id) == "shell_permissions"
        await pilot.press("enter")
        await pilot.pause()
        panel = pilot.app.screen
        assert isinstance(panel, ShellPermissionsApp)
        # The panel inspected through the service before it can be used.
        deadline = asyncio.get_event_loop().time() + 2.0
        while panel._inspection is None:
            assert asyncio.get_event_loop().time() < deadline
            await pilot.pause(0.01)
        assert panel._inspection is not None
        grants = panel.query_one("#shell-permissions-grants", OptionList)
        row_ids = [str(row.id) for row in grants.options if row.id is not None]
        assert "live:git.push" in row_ids
        # Escape returns to Settings with focus, filter, and selection intact.
        await pilot.press("escape")
        await pilot.pause()
        assert pilot.app.screen is screen
        assert options._query == "shell"
        assert options.highlighted_option is not None
        assert str(options.highlighted_option.id) == "shell_permissions"
        assert options.has_focus


@pytest.mark.asyncio
async def test_shell_permissions_unavailable_without_a_service() -> None:
    async with Harness(FakeService()).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        assert screen.shell_permissions is None
        await pilot.press(*"shell")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        # No panel is pushed; the row reports the unavailable service.
        assert pilot.app.screen is screen
        help_text = str(screen.query_one("#settings-help", NoMarkupStatic).content)
        assert "Shell Permissions is unavailable" in help_text
