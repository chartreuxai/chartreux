from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.pilot import Pilot

from chartreux.cli.textual_ui.widgets.proxy_setup_app import ProxySetupApp
from chartreux.core.proxy_setup import get_current_proxy_settings, set_proxy_var
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp
from tests.snapshots.snap_compare import SnapCompare


class ProxySetupTestApp(BaseSnapshotTestApp):
    async def on_mount(self) -> None:
        await super().on_mount()
        await self._switch_to_proxy_setup_app()


class PrePopulatedProxySetupTestApp(BaseSnapshotTestApp):
    async def on_mount(self) -> None:
        set_proxy_var("HTTP_PROXY", "http://old-proxy:8080")
        set_proxy_var("HTTPS_PROXY", "https://old-proxy:8443")
        await super().on_mount()
        await self._switch_to_proxy_setup_app()


@pytest.mark.asyncio
async def test_proxy_setup_rejects_invalid_batch_without_invoking_setters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_proxy = Mock()
    unset_proxy = Mock()
    monkeypatch.setattr("chartreux.app_server._resources.set_proxy_var", set_proxy)
    monkeypatch.setattr("chartreux.app_server._resources.unset_proxy_var", unset_proxy)

    async with ProxySetupTestApp().run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"http://proxy.example.com:8080", "down")
        await pilot.press(*"invalid-proxy:8443", "enter")
        await pilot.pause()

        proxy_setup = pilot.app.query_one(ProxySetupApp)
        assert proxy_setup.inputs["HTTP_PROXY"].value == "http://proxy.example.com:8080"
        assert proxy_setup.inputs["HTTPS_PROXY"].value == "invalid-proxy:8443"
        assert "HTTPS_PROXY must start" in str(
            proxy_setup.query_one("#proxysetup-error", NoMarkupStatic).content
        )

    set_proxy.assert_not_called()
    unset_proxy.assert_not_called()


@pytest.mark.asyncio
async def test_proxy_setup_saves_valid_batch_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_proxy = Mock()
    monkeypatch.setattr("chartreux.app_server._resources.set_proxy_var", set_proxy)

    async with ProxySetupTestApp().run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"http://proxy.example.com:8080", "down")
        await pilot.press(*"https://proxy.example.com:8443", "enter")
        await pilot.pause()

        assert not list(pilot.app.query(ProxySetupApp))

    assert set_proxy.call_args_list == [
        (("HTTP_PROXY", "http://proxy.example.com:8080"),),
        (("HTTPS_PROXY", "https://proxy.example.com:8443"),),
    ]


def test_snapshot_proxy_setup_initial_empty(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)

    assert snap_compare(
        "test_ui_snapshot_proxy_setup.py:ProxySetupTestApp",
        terminal_size=(100, 36),
        run_before=run_before,
    )


def test_snapshot_proxy_setup_initial_with_values(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)

    assert snap_compare(
        "test_ui_snapshot_proxy_setup.py:PrePopulatedProxySetupTestApp",
        terminal_size=(100, 36),
        run_before=run_before,
    )


def test_snapshot_proxy_setup_save_new_values(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)
        await pilot.press(*"http://proxy.example.com:8080")
        await pilot.press("down")
        await pilot.press(*"https://proxy.example.com:8443")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.2)

    assert snap_compare(
        "test_ui_snapshot_proxy_setup.py:ProxySetupTestApp",
        terminal_size=(100, 36),
        run_before=run_before,
    )

    settings = get_current_proxy_settings()
    assert settings["HTTP_PROXY"] == "http://proxy.example.com:8080"
    assert settings["HTTPS_PROXY"] == "https://proxy.example.com:8443"


def test_snapshot_proxy_setup_edit_existing_values(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)
        await pilot.press("ctrl+u")
        await pilot.press(*"http://new-proxy:9090")
        await pilot.press("down")
        await pilot.press("ctrl+u")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.2)

    assert snap_compare(
        "test_ui_snapshot_proxy_setup.py:PrePopulatedProxySetupTestApp",
        terminal_size=(100, 36),
        run_before=run_before,
    )

    settings = get_current_proxy_settings()
    assert settings["HTTP_PROXY"] == "http://new-proxy:9090"
    assert settings["HTTPS_PROXY"] is None


def test_snapshot_proxy_setup_cancel_discards_changes(
    snap_compare: SnapCompare,
) -> None:

    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)
        await pilot.press(*"http://should-not-save:8080")
        await pilot.pause(0.1)
        await pilot.press("escape")
        await pilot.pause(0.2)

    assert snap_compare(
        "test_ui_snapshot_proxy_setup.py:ProxySetupTestApp",
        terminal_size=(100, 36),
        run_before=run_before,
    )

    settings = get_current_proxy_settings()
    assert settings["HTTP_PROXY"] is None


def test_snapshot_proxy_setup_save_error(
    snap_compare: SnapCompare, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_error(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("chartreux.core.proxy_setup.set_key", raise_error)

    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)
        await pilot.press(*"http://proxy:8080")
        await pilot.press("enter")
        await pilot.pause(0.2)

    assert snap_compare(
        "test_ui_snapshot_proxy_setup.py:ProxySetupTestApp",
        terminal_size=(100, 36),
        run_before=run_before,
    )
