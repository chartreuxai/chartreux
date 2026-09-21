from __future__ import annotations

import pytest

from chartreux.cli.textual_ui.app import ChartreuxApp


@pytest.mark.asyncio
async def test_inline_notice_show_displays_message(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        chartreux_app._inline_notice.show("Selection copied to clipboard", timeout=10)
        await pilot.pause(0.05)

        assert chartreux_app._inline_notice.display is True
