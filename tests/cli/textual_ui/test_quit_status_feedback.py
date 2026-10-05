from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from chartreux.cli.textual_ui.quit_manager import (
    ExitConsequencesScreen,
    QuitConfirmKey,
    QuitManager,
)
from chartreux.cli.textual_ui.widgets.session_status_line import (
    SessionStatusLine,
    SessionStatusState,
)


class _QuitStatusApp(App[None]):
    def compose(self) -> ComposeResult:
        yield SessionStatusLine(SessionStatusState(cwd=Path("/work")))


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["Ctrl+C", "Ctrl+D", "/exit"])
async def test_quit_confirmation_is_plain_status_feedback(key: QuitConfirmKey) -> None:
    app = _QuitStatusApp()
    manager = QuitManager(app)
    async with app.run_test(size=(80, 24)) as pilot:
        line = app.query_one(SessionStatusLine)
        original = line.render().plain
        manager.request_confirmation(key, "No active work")
        await pilot.pause()
        assert line.render().plain == f"Press {key} again to quit (No active work)"
        assert manager.is_confirmed(key)
        manager.cancel_confirmation()
        await pilot.pause()
        assert line.render().plain == original
        assert not manager.is_confirmed(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_exit_consequences_scroll_and_controls_survive_resize(
    size: tuple[int, int],
) -> None:
    app = _QuitStatusApp()
    consequences = "\n".join(f"Consequence {index}" for index in range(60))
    results: list[bool | None] = []
    async with app.run_test(size=size) as pilot:
        dialog = ExitConsequencesScreen(consequences)
        app.push_screen(dialog, callback=results.append)
        await pilot.pause()
        assert dialog.focused is dialog.query_one("#exit-cancel", Button)
        scroll = dialog.query_one("#exit-consequences", VerticalScroll)
        content = scroll.query_one(Static)
        assert consequences in str(content.content)
        for viewport in (size, (120, 36) if size == (80, 24) else (80, 24)):
            await pilot.resize_terminal(*viewport)
            await pilot.press("shift+tab")
            assert dialog.focused is scroll
            await pilot.press("home")
            await pilot.pause()
            assert scroll.scroll_y == 0
            await pilot.press("end")
            await pilot.pause()
            assert scroll.max_scroll_y > 0
            assert scroll.scroll_y == scroll.max_scroll_y
            assert content.region.bottom <= scroll.content_region.bottom
            await pilot.press("tab")
            assert dialog.focused is dialog.query_one("#exit-cancel", Button)
            assert results == []
            for control in ("#exit-cancel", "#exit-confirm"):
                button = dialog.query_one(control, Button)
                assert app.screen.region.contains_region(button.region)
                assert button.region.height == 3
        assert await pilot.click("#exit-cancel")
        await pilot.pause()
        assert results == [False]
