from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server.models import AgentStatsSnapshot, SessionLogSummary
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.context_progress import ContextProgress
from chartreux.cli.textual_ui.widgets.messages import UserCommandMessage
from tests.conftest import build_test_chartreux_app

_SPENT_TOKENS = 50_000
_CONTEXT_WINDOW = 200_000


@pytest.fixture
def chartreux_app() -> ChartreuxApp:
    return build_test_chartreux_app()


def _set_session_log(
    chartreux_app: ChartreuxApp, *, enabled: bool, persisted: bool
) -> None:
    state = chartreux_app.app_server.resources.runtime._state
    state.session_log = SessionLogSummary(
        enabled=enabled,
        persisted=persisted,
        session_id="test-session-123" if enabled else None,
    )


def _fill_context_gauge(chartreux_app: ChartreuxApp) -> None:
    """Put the gauge where a conversation would have left it."""
    state = chartreux_app.app_server.resources.runtime._state
    state.stats = AgentStatsSnapshot(
        context_tokens=_SPENT_TOKENS, session_prompt_tokens=_SPENT_TOKENS
    )
    state.context_window = _CONTEXT_WINDOW
    chartreux_app._refresh_context_progress()


def _empty_the_context(chartreux_app: ChartreuxApp) -> None:
    """Stand in for the runtime read `AppServerSession.clear_history` performs.

    That read is what actually refreshes this cache, and is pinned by
    `test_clear_history_refreshes_the_cached_context_gauge`. These tests own the
    other half: that the widget re-reads the cache instead of waiting for an event.
    """
    chartreux_app.app_server.resources.runtime._state.stats = AgentStatsSnapshot()


@pytest.mark.asyncio
async def test_clear_history_shows_resume_hint_when_session_persisted(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        _set_session_log(chartreux_app, enabled=True, persisted=True)
        chartreux_app.app_server.clear_history = AsyncMock()
        chartreux_app._reset_message_widgets = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()
        chartreux_app._handle_user_message = AsyncMock()

        await chartreux_app._clear_history()

        mounted = chartreux_app._mount_and_scroll.call_args_list
        assert any(
            isinstance(args.args[0], UserCommandMessage)
            and "chartreux --resume" in args.args[0]._content
            for args in mounted
        ), "Expected resume hint in mounted UserCommandMessage"
        chartreux_app._handle_user_message.assert_not_called()


@pytest.mark.asyncio
async def test_clear_history_omits_resume_hint_when_logging_disabled(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        _set_session_log(chartreux_app, enabled=False, persisted=False)
        chartreux_app.app_server.clear_history = AsyncMock()
        chartreux_app._reset_message_widgets = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()
        chartreux_app._handle_user_message = AsyncMock()

        await chartreux_app._clear_history()

        mounted = chartreux_app._mount_and_scroll.call_args_list
        for args in mounted:
            if isinstance(args.args[0], UserCommandMessage):
                assert "chartreux --resume" not in args.args[0]._content
        chartreux_app._handle_user_message.assert_not_called()


@pytest.mark.asyncio
async def test_clear_history_dispatches_prompt_when_args_provided(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        _set_session_log(chartreux_app, enabled=True, persisted=True)
        chartreux_app.app_server.clear_history = AsyncMock()
        chartreux_app._reset_message_widgets = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()
        chartreux_app._handle_user_message = AsyncMock()

        await chartreux_app._clear_history("fix the tests")

        chartreux_app._handle_user_message.assert_awaited_once_with("fix the tests")


@pytest.mark.asyncio
async def test_clear_history_does_not_dispatch_when_clear_fails(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        _set_session_log(chartreux_app, enabled=True, persisted=True)
        chartreux_app.app_server.clear_history = AsyncMock(
            side_effect=RuntimeError("server down")
        )
        chartreux_app._reset_message_widgets = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()
        chartreux_app._handle_user_message = AsyncMock()

        await chartreux_app._clear_history("fix the tests")

        chartreux_app._handle_user_message.assert_not_called()


@pytest.mark.asyncio
async def test_clear_history_empties_the_context_gauge(
    chartreux_app: ChartreuxApp,
) -> None:
    """*Prepare*: Spend context, so the gauge reads well above zero.
    *Do*: Clear the history.
    *Assert*: The gauge drops immediately, without waiting for the next turn.
    """
    async with chartreux_app.run_test():
        # Prepare
        _set_session_log(chartreux_app, enabled=True, persisted=True)
        _fill_context_gauge(chartreux_app)
        assert (
            chartreux_app.query_one(ContextProgress).tokens.current_tokens
            == _SPENT_TOKENS
        )
        chartreux_app.app_server.clear_history = AsyncMock(
            side_effect=lambda: _empty_the_context(chartreux_app)
        )
        chartreux_app._reset_message_widgets = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()
        chartreux_app._handle_user_message = AsyncMock()

        # Do
        await chartreux_app._clear_history()

        # Assert
        widget = chartreux_app.query_one(ContextProgress)
        assert widget.tokens.current_tokens == 0
        assert widget.tokens.max_tokens == _CONTEXT_WINDOW


@pytest.mark.asyncio
async def test_clear_history_refreshes_banner(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await chartreux_app.app_server.resources.runtime.wait_until_ready()
        await chartreux_app._startup_command_availability_ready.wait()
        await pilot.pause()
        chartreux_app.app_server.clear_history = AsyncMock()
        chartreux_app._reset_message_widgets = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()
        chartreux_app._refresh_banner = MagicMock()

        await chartreux_app._clear_history()

        chartreux_app._refresh_banner.assert_called_once_with()


@pytest.mark.asyncio
async def test_clear_history_leaves_the_gauge_alone_when_clear_fails(
    chartreux_app: ChartreuxApp,
) -> None:
    """A failed clear keeps the conversation, so the gauge must keep measuring it."""
    async with chartreux_app.run_test():
        _set_session_log(chartreux_app, enabled=True, persisted=True)
        _fill_context_gauge(chartreux_app)
        chartreux_app.app_server.clear_history = AsyncMock(
            side_effect=RuntimeError("server down")
        )
        chartreux_app._reset_message_widgets = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()

        await chartreux_app._clear_history()

        assert (
            chartreux_app.query_one(ContextProgress).tokens.current_tokens
            == _SPENT_TOKENS
        )


@pytest.mark.asyncio
async def test_clear_history_resets_terminal_title(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test():
        # The fresh session carries no title, so the tab must reset.
        _set_session_log(chartreux_app, enabled=True, persisted=True)
        chartreux_app.app_server.clear_history = AsyncMock()
        chartreux_app._reset_message_widgets = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()
        chartreux_app._handle_user_message = AsyncMock()
        chartreux_app._terminal_notifier.set_default_title = MagicMock()

        await chartreux_app._clear_history()

        chartreux_app._terminal_notifier.set_default_title.assert_called_once_with("")
