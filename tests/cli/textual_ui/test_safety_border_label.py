from __future__ import annotations

import pytest

from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from tests.conftest import build_test_chartreux_app, build_test_vibe_config


def _border_label(app) -> str:
    return (app.query_one("#input-box").border_title or "").strip()


@pytest.mark.asyncio
async def test_startup_smart_level_sets_tint_and_border_label() -> None:
    app = build_test_chartreux_app(
        config=build_test_vibe_config(safety={"default_level": "smart"})
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()
        await pilot.pause()
        container = app.query_one("#input-container")
        assert container.has_class("safety-smart")
        assert _border_label(app) == "safety: smart"


@pytest.mark.asyncio
async def test_apply_sets_tint_and_label_while_container_field_is_none() -> None:
    # Regression: the initial apply runs in _mount_after_session_ready before
    # _complete_mount assigns _chat_input_container, so the field is None.
    app = build_test_chartreux_app(
        config=build_test_vibe_config(safety={"default_level": "smart"})
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        app._chat_input_container = None
        container = app.query_one("#input-container")
        container.remove_class("safety-smart")
        await app._apply_safety_level_ui("safe")
        assert _border_label(app) == "safety: safe"
        container.remove_class("safety-safe")
        applied = await app._apply_safety_level_ui("smart")
        assert applied is True
        assert container.has_class("safety-smart")
        assert _border_label(app) == "safety: smart"


@pytest.mark.asyncio
async def test_invalid_level_clears_border_label_and_tint() -> None:
    app = build_test_chartreux_app(
        config=build_test_vibe_config(safety={"default_level": "smart"})
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        await app._apply_safety_level_ui("smart")
        assert _border_label(app) == "safety: smart"
        container = app.query_one("#input-container")
        applied = await app._apply_safety_level_ui("not-a-level")
        assert applied is True
        assert not container.has_class("safety-smart")
        assert _border_label(app) == ""


@pytest.mark.asyncio
async def test_dangerous_level_mounts_banner_and_sets_label() -> None:
    app = build_test_chartreux_app(
        config=build_test_vibe_config(safety={"default_level": "smart"})
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        container = app.query_one("#input-container")
        applied = await app._apply_safety_level_ui("dangerous")
        assert applied is True
        assert container.has_class("safety-dangerous")
        assert _border_label(app) == "safety: dangerous"
        assert app.query("#dangerous-banner")
        # Leaving dangerous removes the banner and relabels.
        applied = await app._apply_safety_level_ui("safe")
        assert applied is True
        assert not app.query("#dangerous-banner")
        assert container.has_class("safety-safe")
        assert _border_label(app) == "safety: safe"


@pytest.mark.asyncio
async def test_apply_skips_quietly_when_container_not_mounted() -> None:
    app = build_test_chartreux_app(
        config=build_test_vibe_config(safety={"default_level": "smart"})
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        container = app.query(ChatInputContainer).first()
        assert container is not None
        await container.remove()
        await pilot.pause()
        # Guarded query returns None: no exception, non-dangerous apply still True.
        applied = await app._apply_safety_level_ui("smart")
        assert applied is True
