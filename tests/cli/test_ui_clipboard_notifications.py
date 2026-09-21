from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.selection import Selection
from textual.widget import Widget

from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.ui.clipboard import ClipboardCopyResult, copy_selection_to_clipboard


class ClipboardSelectionWidget(Widget):
    def __init__(self, selected_text: str) -> None:
        super().__init__()
        self._selected_text = selected_text

    @property
    def text_selection(self) -> Selection | None:
        return Selection(None, None)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        return (self._selected_text, "\n")


@pytest.mark.asyncio
async def test_ui_clipboard_notification_does_not_crash_on_markup_text(
    monkeypatch: pytest.MonkeyPatch, chartreux_app: ChartreuxApp
) -> None:
    async with chartreux_app.run_test(notifications=True) as pilot:
        await chartreux_app.mount(ClipboardSelectionWidget("[/]"))
        with patch("chartreux.ui.clipboard.copy_to_clipboard", return_value=True):
            copy_selection_to_clipboard(chartreux_app)

        await pilot.pause(0.1)
        notifications = list(chartreux_app._notifications)
        assert notifications
        notification = notifications[-1]
        assert notification.markup is False
        assert "Selection copied to clipboard" in notification.message


def test_clipboard_notice_omits_hint_when_copy_is_verified(
    chartreux_app: ChartreuxApp,
) -> None:
    result = ClipboardCopyResult(text="hello", verified=True)

    assert chartreux_app._clipboard_notice_message(result) == "Copied to clipboard"


def test_clipboard_notice_uses_generic_hint_when_copy_is_unverified(
    chartreux_app: ChartreuxApp,
) -> None:
    result = ClipboardCopyResult(text="hello", verified=False)

    assert chartreux_app._clipboard_notice_message(result) == (
        "Copied · if paste fails, hold Shift (Option in iTerm2, Fn in Terminal.app) "
        "while selecting for native copy"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("verified", [True, False])
async def test_action_copy_selection_shows_clipboard_notice(
    chartreux_app: ChartreuxApp, verified: bool
) -> None:
    async with chartreux_app.run_test() as pilot:
        await chartreux_app.mount(ClipboardSelectionWidget("hello"))
        with (
            patch("chartreux.ui.clipboard.copy_to_clipboard", return_value=verified),
            patch.object(chartreux_app._inline_notice, "show") as show_notice,
        ):
            chartreux_app.action_copy_selection()
            await pilot.pause(0.1)

        show_notice.assert_called_once()
        message = show_notice.call_args.args[0]
        assert ("if paste fails" in message) is not verified
