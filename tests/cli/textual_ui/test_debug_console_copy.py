from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from chartreux.cli.textual_ui.widgets.debug_console import _LogView


class _CopyHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.copied: list[str] = []

    def copy_to_clipboard(self, text: str) -> None:
        self.copied.append(text)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                "Click row to select · c Copy selected", id="debug-console-footer"
            )
            self.log_view = _LogView(
                load_page=lambda: None, has_more=lambda: False, id="debug-console-log"
            )
            yield self.log_view


@pytest.mark.asyncio
async def test_log_click_selects_without_copying_until_explicit_action() -> None:
    app = _CopyHost()
    async with app.run_test(size=(80, 10)) as pilot:
        message = "INFO " + "log-entry " * 15
        app.log_view.write_line(f"[bold]INFO[/bold] {message[5:]}")
        await pilot.pause()

        await pilot.click(app.log_view, offset=(1, 0))
        assert app.copied == []
        assert app.log_view._selected_line == 0
        assert "c Copy selected" in str(
            app.query_one("#debug-console-footer", Static).render()
        )
        for visual_line in (0, 1):
            assert any(
                segment.style is not None and segment.style.reverse
                for segment in app.log_view.render_line(visual_line)._segments
            )

        await pilot.press("c")
        assert app.copied == [message]
