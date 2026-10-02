from __future__ import annotations

from typing import cast

from textual.pilot import Pilot

from chartreux.app_server.protocol import (
    InventoryItemStateWire,
    SettingDescriptorWire,
    SettingLeafWire,
)
from chartreux.cli.textual_ui.screens.settings import SettingsScreen
from chartreux.ui.settings_service import SettingsService
from tests.cli.textual_ui.test_settings_app import FakeService
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp
from tests.snapshots.snap_compare import SnapCompare


class SettingsSnapshotApp(BaseSnapshotTestApp):
    async def on_mount(self) -> None:
        await super().on_mount()
        service = FakeService()
        self.push_screen(
            SettingsScreen(cast(SettingsService, service), service.snapshot)
        )


def test_settings_browser(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp", terminal_size=(80, 24)
    )


def test_settings_list_draft(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=[*"agent_paths", "enter", "enter", *"my-agent", "enter"],
    )


def test_settings_detail_view(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=["f1"],
    )


def test_settings_no_matches(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=list("unfindable_setting_zzzz"),
    )


def test_settings_modal_browser(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp", terminal_size=(100, 32)
    )


def test_settings_filtered(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=list("timeout"),
    )


def test_settings_bool_toggled(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=["enter"],
    )


def test_settings_string_inline(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=[*"displayed_workdir", "enter"],
    )


def test_settings_enum_expanded(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        screen = cast(SettingsScreen, pilot.app.screen)
        screen.catalog.append(
            SettingDescriptorWire(
                path="ui_color_scheme",
                label="Theme",
                description="Choose a theme.",
                kind="enum",
                group="Interface",
                choices=("light", "dark"),
            )
        )
        screen.fields["ui_color_scheme"] = SettingLeafWire(
            path="ui_color_scheme",
            effective_value="dark",
            origin="default",
            saved_explicit=False,
        )
        screen._refresh_options()
        await pilot.press(*"ui_color_scheme", "enter")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


def test_settings_checklist_expanded(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        screen = cast(SettingsScreen, pilot.app.screen)
        screen.snapshot.inventories["tools"] = ["bash", "read_file", "write_file"]
        screen.snapshot.inventory_states["tools"] = {
            "bash": InventoryItemStateWire(
                effective=True, default_effective=True, pattern_driven=False
            ),
            "read_file": InventoryItemStateWire(
                effective=False, default_effective=True, pattern_driven=False
            ),
            "write_file": InventoryItemStateWire(
                effective=False, default_effective=True, pattern_driven=False
            ),
        }
        screen.fields["enabled_tools"].saved_value = ["bash", "tool-*", "re:^custom_"]
        screen.fields["enabled_tools"].saved_explicit = True
        screen.fields["enabled_tools"].effective_value = [
            "bash",
            "tool-*",
            "re:^custom_",
        ]
        screen._refresh_options()
        await pilot.press(*"inventory_tools", "enter")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


def test_settings_number_invalid(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=[*"api_timeout", "enter", "ctrl+u", "x", "enter"],
    )


def test_settings_highlight_help(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=["down"],
    )


def test_settings_confirmation(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=[*"enable_system_trust_store", "enter"],
    )


def test_settings_link_help(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        press=[*"/theme"],
    )


def test_settings_provenance(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        screen = next(
            screen
            for screen in pilot.app.screen_stack
            if isinstance(screen, SettingsScreen)
        )
        field = screen.fields["show_greeting"]
        screen.fields["show_greeting"] = field.model_copy(
            update={
                "saved_explicit": True,
                "saved_value": True,
                "effective_value": False,
                "origin": "environment",
            }
        )
        screen._refresh_options()
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_settings.py:SettingsSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )
