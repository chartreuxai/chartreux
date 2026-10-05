from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from chartreux.app_server.models import DebugLogEntry, DebugLogPage
from chartreux.cli.textual_ui.widgets.debug_console import DebugConsole, _LogView


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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_keyboard_selection_is_bounded_and_copies_logical_entries(size) -> None:
    app = _CopyHost()
    async with app.run_test(size=size) as pilot:
        view = app.log_view
        duplicate = "duplicate " * 30
        for line in (duplicate, duplicate, "last"):
            view.write_line(line, scroll_end=False)
        view.focus()
        await pilot.press("down", "c")
        assert view._selected_line == 0
        assert app.copied == [duplicate]
        await pilot.press("down")
        assert view._selected_line == 1
        view.prepend_lines(["older"])
        assert view._selected_line == 2
        await pilot.press("c", "j", "j")
        assert app.copied == [duplicate, duplicate]
        assert view._selected_line == 3
        await pilot.press("k", "up", "up", "up")
        assert view._selected_line == 0
        assert any(
            segment.style is not None and segment.style.reverse
            for segment in view.render_line(0)._segments
        )


class _PollingHost(_CopyHost):
    def compose(self) -> ComposeResult:
        self.theme_variables.update({
            "text-muted": "#888888",
            "foreground": "#ffffff",
            "error": "#ff0000",
        })
        self.source = AsyncMock()
        self.source.read_logs.return_value = DebugLogPage(entries=[], has_more=False)
        self.debug_console = DebugConsole(self.source)
        yield self.debug_console


@pytest.mark.asyncio
async def test_polling_keeps_duplicate_entries_distinct_and_selection_stationary() -> (
    None
):
    app = _PollingHost()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        stamp = datetime.now(UTC)

        def entry(identity: str) -> DebugLogEntry:
            return DebugLogEntry(
                id=identity,
                timestamp=stamp,
                ppid=1,
                pid=2,
                level="INFO",
                message="same text " * 20,
                raw_line="",
            )

        app.source.read_logs.return_value = DebugLogPage(
            entries=[entry("second"), entry("first")], has_more=False, cursor=2
        )
        await app.debug_console._poll_latest()
        view = app.debug_console.query_one(_LogView)
        assert len(view._lines) == 2
        assert view._lines[0] == view._lines[1]
        view.focus()
        await pilot.press("down", "down", "c")
        assert view._selected_line == 1
        reading_position = view.scroll_y
        app.source.read_logs.return_value = DebugLogPage(
            entries=[entry("third"), entry("second"), entry("first")], has_more=False
        )
        await app.debug_console._poll_latest()
        assert len(view._lines) == 3
        assert view._selected_line == 1
        assert view.scroll_y == reading_position
        await pilot.press("c")
        assert len(app.copied) == 2 and app.copied[0] == app.copied[1]
        await app.debug_console._poll_latest()
        assert len(view._lines) == 3
