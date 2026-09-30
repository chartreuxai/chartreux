from __future__ import annotations

import pytest

from chartreux.cli.textual_ui.app import ChartreuxApp


@pytest.mark.asyncio
async def test_inline_notice_show_displays_message(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        chartreux_app._inline_notice.show("Selection copied to clipboard", timeout=10)
        await pilot.pause(0.05)

        assert chartreux_app._inline_notice.display is True


@pytest.mark.asyncio
async def test_warning_notice_does_not_start_dismissal_timer(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        notice = chartreux_app._inline_notice
        notice.show("Warning: Unsaved changes", timeout=0.01)
        await pilot.pause(0.05)
        assert notice.display
        assert notice._hide_timer is None
        assert str(notice.render()) == "! Warning: Unsaved changes"
        notice.show("Connection refused", severity="error", timeout=0.01)
        assert str(notice.render()) == "✗ Failed: Connection refused"
        await pilot.pause(0.05)
        assert notice.display and notice._hide_timer is None
        notice.show("Failed: Could not save", timeout=0.01)
        await pilot.pause(0.05)
        assert notice.display and notice._hide_timer is None
        notice.hide()
