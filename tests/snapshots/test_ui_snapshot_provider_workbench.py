"""Stable 80x24 views of the experimental Provider Settings surface."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, OptionList

from chartreux.core.model_catalog.loader import load_catalog
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp
from tests.snapshots.snap_compare import SnapCompare
from tests.ui.providers.test_workbench import setup


@pytest.fixture(autouse=True)
def _provider_snapshot_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the snapshot palette independent of the caller's NO_COLOR setting."""
    monkeypatch.delenv("NO_COLOR", raising=False)


def test_models_catalog_view(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        screen = cast(ProviderWorkbenchScreen, pilot.app.screen)
        screen._open_catalog()
        screen._select_catalog("model:a")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


def test_default_presets(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        await pilot.press("enter")
        screen = cast(ProviderWorkbenchScreen, pilot.app.screen)
        assert screen.state
        screen.state.set_role_preset("other", "a", "low")
        screen._open_presets()
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


class ProviderWorkbenchSnapshotApp(BaseSnapshotTestApp):
    async def on_mount(self) -> None:
        await super().on_mount()
        screen, _services = setup()
        self.push_screen(screen)


def test_provider_browser(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
    )


def test_expanded_provider_checklist(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        await pilot.press("enter")
        screen = cast(ProviderWorkbenchScreen, pilot.app.screen)
        screen._select_action("models")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


def test_model_detail_editor(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        await pilot.press("enter")
        screen = cast(ProviderWorkbenchScreen, pilot.app.screen)
        screen._select_action("models")
        screen._open_detail("a")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


def test_add_connection_stage(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        screen = cast(ProviderWorkbenchScreen, pilot.app.screen)
        screen.snapshot = load_catalog(Path("/nonexistent/chartreux-models.toml"))
        screen._refresh_browser()
        browser = screen.query_one("#wb-providers", OptionList)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "mistral"
        )
        await pilot.press("enter")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


def test_discard_confirmation(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        await pilot.press("enter")
        screen = cast(ProviderWorkbenchScreen, pilot.app.screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter")
        screen._select_action("discard")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


def test_protocol_draft_radios(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        await pilot.press("enter")
        screen = cast(ProviderWorkbenchScreen, pilot.app.screen)
        screen._select_action("style")
        await pilot.press("down")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


def test_invalid_credential_field(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        await pilot.press("enter")
        screen = cast(ProviderWorkbenchScreen, pilot.app.screen)
        screen._select_action("key")
        await pilot.press("enter")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


def test_add_models_stage(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        screen = cast(ProviderWorkbenchScreen, pilot.app.screen)
        screen.snapshot = load_catalog(Path("/nonexistent/chartreux-models.toml"))
        screen._refresh_browser()
        browser = screen.query_one("#wb-providers", OptionList)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "mistral"
        )
        await pilot.press("enter")
        screen._connection_action("continue")
        await pilot.pause()

    assert snap_compare(
        "test_ui_snapshot_provider_workbench.py:ProviderWorkbenchSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )
