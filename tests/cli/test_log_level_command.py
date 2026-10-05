from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest
import pytest_asyncio
from textual.widget import Widget
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

from chartreux.cli.textual_ui.widgets.log_level_picker import (
    LogLevelPickerApp,
    _build_row,
)
from chartreux.cli.textual_ui.widgets.messages import UserCommandMessage
from chartreux.observability.logging import (
    _ChartreuxFileHandler,
    get_effective_log_level,
    get_log_level_chain,
    get_session_override,
    init_file_logging,
    logger as vibe_logger,
    set_session_override,
)
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


@pytest_asyncio.fixture(autouse=True)
async def _snapshot_event_loop_wake() -> None:
    install_snapshot_wake()


async def _wait_until(pilot, predicate: Callable[[], bool], *, tries: int = 50) -> bool:
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return predicate()


def _has_command_message(app, *needles: str) -> bool:
    return any(
        all(needle in message._content for needle in needles)
        for message in app.query(UserCommandMessage)
    )


@pytest.fixture(autouse=True)
def _clear_session_override(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    set_session_override(None)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("DEBUG_MODE", raising=False)
    yield
    set_session_override(None)


@pytest.mark.asyncio
async def test_bare_opens_picker_panel() -> None:
    config = build_test_vibe_config()
    agent_loop = build_test_agent_loop(config=config)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        handled = await app._handle_command("/log-level")
        await pilot.pause()
        assert app.query(LogLevelPickerApp)

    assert handled is True


@pytest.mark.asyncio
async def test_badge_focus_cycle_is_symmetric_and_navigation_is_inert() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        original = (picker._session_level, picker._config_level)
        config = app.app_server.resources.config.current.log_level
        targets = [
            "loglevelpicker-session",
            "loglevelpicker-config",
            "loglevelpicker-options",
            "loglevelpicker-apply",
        ]
        assert cast(Widget, app.focused).id == targets[0]
        for target in [*targets[1:], targets[0]]:
            await pilot.press("tab")
            assert cast(Widget, app.focused).id == target
        for target in [*reversed(targets[1:]), targets[0]]:
            await pilot.press("shift+tab")
            assert cast(Widget, app.focused).id == target
        assert (picker._session_level, picker._config_level) == original
        assert get_session_override() is None
        assert app.app_server.resources.config.current.log_level == config
        await pilot.press("enter", "home", "up", "k")
        options = picker.query_one(OptionList)
        assert options.highlighted == 0
        await pilot.press("down", "down")
        session_row = cast(Option, options.highlighted_option).id
        await pilot.press("shift+tab", "shift+tab")
        assert picker._highlighted_level == session_row
        await pilot.press("tab", "enter", "end", "down", "j")
        assert options.highlighted == options.option_count - 1
        config_row = cast(Option, options.highlighted_option).id
        await pilot.press("shift+tab", "shift+tab", "enter")
        assert cast(Option, options.highlighted_option).id == session_row
        await pilot.press("shift+tab", "enter")
        assert cast(Option, options.highlighted_option).id == config_row
        assert (picker._session_level, picker._config_level) == original


def test_highlighted_log_badges_distinguish_set_and_unset() -> None:
    row = _build_row(
        "DEBUG",
        is_highlighted=True,
        focused_badge="session",
        effective_level="DEBUG",
        session_level="DEBUG",
        config_level=None,
    )
    assert "● session" in row.plain
    assert "○ config" in row.plain
    assert "[" not in row.plain
    unhighlighted = _build_row(
        "DEBUG",
        is_highlighted=False,
        focused_badge="config",
        effective_level="DEBUG",
        session_level="DEBUG",
        config_level=None,
    )
    assert unhighlighted.plain == row.plain


@pytest.mark.asyncio
async def test_picker_escape_confirms_dirty_draft() -> None:
    config = build_test_vibe_config()
    app = build_test_chartreux_app(agent_loop=build_test_agent_loop(config=config))

    async with app.run_test() as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        assert get_session_override() is None
        picker._session_level = "DEBUG"
        await pilot.press("escape")
        await pilot.pause()
        assert app.query(LogLevelPickerApp)
        assert app.focused is picker.query_one("#loglevelpicker-keep", Button)
        await pilot.press("escape")
        await pilot.pause()
        assert app.query(LogLevelPickerApp)
        await pilot.press("escape")
        await pilot.pause()
        picker.query_one("#loglevelpicker-confirm-discard", Button).press()
        await pilot.pause()
        assert not app.query(LogLevelPickerApp)
        assert get_session_override() is None


@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
@pytest.mark.asyncio
async def test_discard_confirmation_fits_and_restores_log_level_picker(
    size: tuple[int, int],
) -> None:
    config = build_test_vibe_config()
    app = build_test_chartreux_app(agent_loop=build_test_agent_loop(config=config))

    async with app.run_test(size=size) as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        options = picker.query_one("#loglevelpicker-options", OptionList)
        highlighted = options.highlighted_option
        assert highlighted is not None

        await pilot.press("tab", "enter", "space", "escape")
        await pilot.pause()

        cancel = picker.query_one("#loglevelpicker-keep", Button)
        discard = picker.query_one("#loglevelpicker-confirm-discard", Button)
        help_widget = picker.query_one("#loglevelpicker-help", Static)
        assert picker._confirming_discard
        assert not options.display
        assert cancel.region.bottom <= picker.region.bottom
        assert discard.region.bottom <= picker.region.bottom
        assert help_widget.region.bottom <= picker.region.bottom
        assert app.focused is cancel
        assert picker._config_level == picker._highlighted_level

        await pilot.press("tab")
        assert app.focused is discard
        await pilot.press("ctrl+s")
        assert app.query(LogLevelPickerApp)
        assert picker._confirming_discard

        await pilot.press("escape")
        await pilot.pause()
        assert not picker._confirming_discard
        assert options.display
        assert options.has_focus
        assert options.highlighted_option is highlighted
        assert picker._focused_badge == "config"
        assert picker._config_level == picker._highlighted_level
        assert not picker.query_one("#loglevelpicker-discard").display
        assert "Navigate" in str(help_widget.content)


@pytest.mark.parametrize("scope_key", ["enter", "space"])
@pytest.mark.parametrize("apply_key", ["enter", "space", "ctrl+s", "click"])
@pytest.mark.asyncio
async def test_activation_edits_only_focused_scope_until_explicit_apply(
    apply_key: str, scope_key: str
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        original_config = picker._config_level
        await pilot.press(scope_key)
        assert picker.query_one(OptionList).has_focus
        await pilot.press("home", "enter")
        assert picker._session_level == "DEBUG"
        assert picker._config_level == original_config
        assert get_session_override() is None
        assert app.query(LogLevelPickerApp)
        await pilot.press("shift+tab", scope_key)
        assert picker.query_one(OptionList).has_focus
        await pilot.press("end", "enter")
        assert picker._session_level == "DEBUG"
        assert picker._config_level == "CRITICAL"
        assert app.app_server.resources.config.current.log_level == original_config
        await pilot.press("enter")
        assert picker._config_level is None
        await pilot.press("enter", "tab")
        assert app.query(LogLevelPickerApp)
        assert get_session_override() is None
        if apply_key == "click":
            await pilot.click("#loglevelpicker-apply")
        else:
            await pilot.press(apply_key)
        assert await _wait_until(pilot, lambda: not app.query(LogLevelPickerApp))
        assert get_session_override() == "DEBUG"
        assert app.app_server.resources.config.current.log_level == "CRITICAL"


@pytest.mark.asyncio
async def test_pointer_badges_edit_the_clicked_scope_without_saving() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        options = picker.query_one(OptionList)
        original_config = picker._config_level
        # DEBUG is not the effective row, so its badges begin at columns 14 and 25.
        await pilot.click(options, offset=(16, 0))
        assert picker._session_level == "DEBUG"
        assert picker._config_level == original_config
        # The Active label now shifts the badges seven columns to the right.
        await pilot.click(options, offset=(34, 0))
        assert picker._focused_badge == "config"
        assert picker._config_level == "DEBUG"
        assert get_session_override() is None
        assert app.app_server.resources.config.current.log_level == original_config
        await pilot.press("escape", "escape")
        assert options.has_focus
        assert picker._focused_badge == "config"
        assert picker._highlighted_level == "DEBUG"
        await pilot.click("#loglevelpicker-session")
        assert options.has_focus
        assert picker._focused_badge == "session"
        assert picker._session_level == "DEBUG"


@pytest.mark.asyncio
async def test_discard_cancel_restores_apply_target() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        await pilot.press("enter", "home", "space", "tab", "escape")
        assert cast(Widget, app.focused).id == "loglevelpicker-keep"
        await pilot.press("enter")
        assert cast(Widget, app.focused).id == "loglevelpicker-apply"
        assert picker._session_level == "DEBUG"
        assert get_session_override() is None


@pytest.mark.asyncio
async def test_picker_session_only_shows_feedback() -> None:
    config = build_test_vibe_config()
    agent_loop = build_test_agent_loop(config=config)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        picker.post_message(
            LogLevelPickerApp.Applied(
                session_level="DEBUG", config_level=None, config_cleared=False
            )
        )
        assert await _wait_until(
            pilot, lambda: _has_command_message(app, "DEBUG", "session")
        )

    assert get_session_override() == "DEBUG"


@pytest.mark.asyncio
async def test_picker_global_shows_feedback_and_persists() -> None:
    config = build_test_vibe_config()
    agent_loop = build_test_agent_loop(config=config)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        picker.post_message(
            LogLevelPickerApp.Applied(
                session_level=None, config_level="ERROR", config_cleared=False
            )
        )
        assert await _wait_until(
            pilot, lambda: _has_command_message(app, "ERROR", "config.toml")
        )

    assert app.app_server.resources.config.current.log_level == "ERROR"


@pytest.mark.asyncio
async def test_config_change_applies_when_no_session_override(tmp_path: Path) -> None:
    log_file = tmp_path / "chartreux.log"
    init_file_logging(log_file, target_logger=vibe_logger)
    try:
        config = build_test_vibe_config()
        agent_loop = build_test_agent_loop(config=config)
        app = build_test_chartreux_app(agent_loop=agent_loop)

        async with app.run_test() as pilot:
            await app.app_server.resources.config.update({"log_level": "ERROR"})
            await pilot.pause()
            assert get_session_override() is None
            assert get_effective_log_level() == "ERROR"
    finally:
        vibe_logger.handlers = [
            h
            for h in vibe_logger.handlers
            if not (
                isinstance(h, _ChartreuxFileHandler) and h.baseFilename == str(log_file)
            )
        ]


@pytest.mark.asyncio
async def test_config_level_applied_at_mount(tmp_path: Path) -> None:
    log_file = tmp_path / "chartreux.log"
    init_file_logging(log_file, target_logger=vibe_logger)
    try:
        config = build_test_vibe_config(log_level="ERROR")
        agent_loop = build_test_agent_loop(config=config)
        app = build_test_chartreux_app(agent_loop=agent_loop)

        async with app.run_test() as pilot:
            await pilot.pause()
            chain = get_log_level_chain()
            assert chain.config == "ERROR"
            assert chain.session is None
            assert get_effective_log_level() == "ERROR"
    finally:
        vibe_logger.handlers = [
            h
            for h in vibe_logger.handlers
            if not (
                isinstance(h, _ChartreuxFileHandler) and h.baseFilename == str(log_file)
            )
        ]


@pytest.mark.asyncio
async def test_picker_config_cleared_removes_persisted_value() -> None:
    config = build_test_vibe_config()
    agent_loop = build_test_agent_loop(config=config)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await app.app_server.resources.config.update({"log_level": "ERROR"})
        await pilot.pause()
        assert app.app_server.resources.config.current.log_level == "ERROR"

        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        picker.post_message(
            LogLevelPickerApp.Applied(
                session_level=None, config_level=None, config_cleared=True
            )
        )
        assert await _wait_until(
            pilot, lambda: _has_command_message(app, "config.toml cleared")
        )

    assert app.app_server.resources.config.current.log_level is None


@pytest.mark.asyncio
async def test_picker_session_cleared_shows_feedback() -> None:
    config = build_test_vibe_config()
    agent_loop = build_test_agent_loop(config=config)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    set_session_override("DEBUG")
    async with app.run_test() as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        picker.post_message(
            LogLevelPickerApp.Applied(
                session_level=None, config_level=None, config_cleared=False
            )
        )
        assert await _wait_until(
            pilot, lambda: _has_command_message(app, "session override cleared")
        )

    assert get_session_override() is None


@pytest.mark.asyncio
async def test_picker_config_write_failure_surfaces_error(monkeypatch) -> None:
    # When the inline (idle) config write fails, the picker must surface an
    # error instead of reporting success.
    config = build_test_vibe_config()
    agent_loop = build_test_agent_loop(config=config)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(app, "_persist_config_changes", _boom)

    async with app.run_test() as pilot:
        await app._handle_command("/log-level")
        await pilot.pause()
        picker = app.query_one(LogLevelPickerApp)
        picker.post_message(
            LogLevelPickerApp.Applied(
                session_level=None, config_level="ERROR", config_cleared=False
            )
        )
        await pilot.pause()

        assert "disk on fire" in app._recovery_issues["log-level-save"][1]
        assert app.query_one("#recovery-notice").display
        assert "/log-level" in app._recovery_issues["log-level-save"][1]
        success = app.query(UserCommandMessage)
        assert not any("config.toml" in m._content for m in success)
