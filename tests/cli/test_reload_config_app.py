from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server import AppServerConnectionClosed
from chartreux.cli.textual_ui.widgets.messages import WarningMessage
from chartreux.cli.textual_ui.widgets.reload_message import ReloadConfigMessage
from tests.conftest import (
    build_test_chartreux_app,
    build_test_vibe_config,
    stub_config_reload,
)


@pytest.mark.asyncio
async def test_reload_config_picks_up_disk_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app(
        config=build_test_vibe_config(autocopy_to_clipboard=False)
    )
    reloaded = build_test_vibe_config(autocopy_to_clipboard=True)
    stub_config_reload(monkeypatch, reloaded)

    async with app.run_test():
        assert app.config.autocopy_to_clipboard is False
        await app._reload_config()
        assert app.config.autocopy_to_clipboard is True
        messages = list(app.query(ReloadConfigMessage))
        assert len(messages) == 1
        assert messages[0].get_content().startswith("Configuration reloaded")
        assert messages[0]._spinner_timer is None


@pytest.mark.asyncio
async def test_reload_warns_when_launch_metadata_is_not_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()

    async with app.run_test():
        monkeypatch.setattr(
            app.app_server.resources.config,
            "reload",
            AsyncMock(
                return_value=SimpleNamespace(
                    stripped_history_images=0, launch_metadata_persisted=False
                )
            ),
        )

        await app._reload_config()

        warning = app.query_one(WarningMessage)
        assert "applied for this session" in warning._message
        assert "could not be saved" in warning._message


@pytest.mark.asyncio
async def test_delayed_reload_shows_spinner_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_reload(*_args: object, **_kwargs: object) -> SimpleNamespace:
        started.set()
        await release.wait()
        return SimpleNamespace(
            stripped_history_images=0, launch_metadata_persisted=True
        )

    async with app.run_test() as pilot:
        await app._ensure_loading_widget()
        turn_loading = app._loading_widget
        assert turn_loading is not None
        monkeypatch.setattr(app.app_server.resources.config, "reload", delayed_reload)
        task = asyncio.create_task(app._reload_config())
        await started.wait()
        await pilot.pause()

        message = app.query_one(ReloadConfigMessage)
        assert message._is_spinning
        assert message.get_content() == "Reloading configuration..."
        assert app._loading_widget is turn_loading
        assert turn_loading.parent is not None

        release.set()
        await task

        assert not message._is_spinning
        assert message._spinner_timer is None
        assert len(list(app.query(ReloadConfigMessage))) == 1
        assert app._loading_widget is turn_loading


@pytest.mark.asyncio
async def test_unmounting_spinning_reload_message_cleans_up_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_reload(*_args: object, **_kwargs: object) -> SimpleNamespace:
        started.set()
        await release.wait()
        return SimpleNamespace(
            stripped_history_images=0, launch_metadata_persisted=True
        )

    async with app.run_test() as pilot:
        monkeypatch.setattr(app.app_server.resources.config, "reload", delayed_reload)
        task = asyncio.create_task(app._reload_config())
        await started.wait()
        await pilot.pause()

        message = app.query_one(ReloadConfigMessage)
        assert message._is_spinning
        timer = message._spinner_timer
        assert timer is not None
        assert timer._active

        await message.remove()

        assert message._spinner_timer is None
        assert not list(app.query(ReloadConfigMessage))

        release.set()
        await task

        await app._reload_config()
        replacement = app.query_one(ReloadConfigMessage)
        assert not replacement._is_spinning
        assert replacement._spinner_timer is None


@pytest.mark.asyncio
async def test_failed_reload_settles_spinner_to_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()

    async with app.run_test():
        monkeypatch.setattr(
            app.app_server.resources.config,
            "reload",
            AsyncMock(side_effect=RuntimeError("bad config")),
        )

        await app._reload_config()

        message = app.query_one(ReloadConfigMessage)
        assert not message._is_spinning
        assert message._spinner_timer is None
        assert message.get_content() == "Failed to reload config: bad config"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [asyncio.CancelledError(), AppServerConnectionClosed("disconnected")]
)
async def test_cancelled_or_disconnected_reload_removes_spinner(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    app = build_test_chartreux_app()

    async with app.run_test():
        monkeypatch.setattr(
            app.app_server.resources.config, "reload", AsyncMock(side_effect=failure)
        )

        with pytest.raises(type(failure)):
            await app._reload_config()

        assert not list(app.query(ReloadConfigMessage))
