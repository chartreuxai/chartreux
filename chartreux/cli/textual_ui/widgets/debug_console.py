from __future__ import annotations

import bisect
from collections.abc import Callable
from typing import ClassVar, Protocol

from rich.markup import escape
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.cache import LRUCache
from textual.containers import Vertical
from textual.geometry import Size
from textual.message import Message
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Static

from chartreux.app_server.models import DebugLogEntry, DebugLogPage
from chartreux.observability.logging import decode_log_message

LOG_LEVEL_ROLES: dict[str, str] = {
    "DEBUG": "$text-muted",
    "INFO": "$foreground",
    "WARNING": "$warning",
    "ERROR": "$error",
    "CRITICAL": "$error",
}

DEFAULT_LOG_PAGE_SIZE = 30
DEBUG_DOCK_MIN_WIDTH = 120
DEBUG_DOCK_WIDTH = 40
LOG_POLL_INTERVAL = 0.5
_EMPTY_STYLE = Style()


class DebugLogSource(Protocol):
    async def read_logs(self, *, limit: int = 100, offset: int = 0) -> DebugLogPage: ...


class _LogView(ScrollView, can_focus=True):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("c", "copy_selected_line", "Copy selected log line", show=False),
        Binding("up,k", "select_previous", "Previous log row", show=False),
        Binding("down,j", "select_next", "Next log row", show=False),
        Binding("tab", "leave_console(1)", "Next focus group", show=False),
        Binding("shift+tab", "leave_console(-1)", "Previous focus group", show=False),
    ]

    def __init__(
        self,
        load_page: Callable[[], None],
        has_more: Callable[[], bool],
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._lines: list[str] = []
        self._selected_line: int | None = None
        self._wrap_counts: list[int] = []
        self._wrap_prefix: list[int] = [0]
        self._total_visual: int = 0
        self._cached_width: int = 0
        self._render_line_cache: LRUCache[int, Strip] = LRUCache(1024)
        self._load_page = load_page
        self._has_more = has_more

    def focus(self, scroll_visible: bool = True) -> _LogView:
        if isinstance(self.parent, DebugConsole):
            self.parent._suspend_input_focusability()
        super().focus(scroll_visible=scroll_visible)
        return self

    def on_focus(self) -> None:
        # ChatTextArea normally reclaims focus on blur. Suspend that while the
        # console owns focus, including Screen's direct Tab/pointer focus route.
        if isinstance(self.parent, DebugConsole):
            self.parent._suspend_input_focusability()

    def action_leave_console(self, direction: int) -> None:
        if isinstance(self.parent, DebugConsole):
            if self.parent.has_class("-fullscreen"):
                return
            self.parent._restore_input_focusability()
        if direction > 0:
            self.screen.focus_next()
        else:
            self.screen.focus_previous()

    def on_blur(self) -> None:
        if isinstance(self.parent, DebugConsole) and not self.parent.has_class(
            "-fullscreen"
        ):
            self.parent._restore_input_focusability()

    def _wrap_markup(self, markup: str) -> int:
        """Return the number of visual lines this markup produces at current width."""
        width = self._cached_width
        if width <= 0:
            return 1

        text = Text.from_markup(markup, style=self.rich_style)

        return len(text.wrap(self.app.console, width))

    def _recompute_prefix(self) -> None:
        self._wrap_prefix = [0]
        for count in self._wrap_counts:
            self._wrap_prefix.append(self._wrap_prefix[-1] + count)
        self._total_visual = self._wrap_prefix[-1]

    def _reflow(self) -> None:
        """Re-wrap all lines at current widget width."""
        width = self.size.width
        if width <= 0:
            return
        anchor = max(0, bisect.bisect_right(self._wrap_prefix, int(self.scroll_y)) - 1)
        fragment = int(self.scroll_y) - self._wrap_prefix[anchor]
        self._cached_width = width
        self._render_line_cache.clear()
        self._wrap_counts = [self._wrap_markup(m) for m in self._lines]
        self._recompute_prefix()
        self.virtual_size = Size(width, self._total_visual)
        if anchor < len(self._lines):
            self.scroll_to(
                y=self._wrap_prefix[anchor]
                + min(fragment, self._wrap_counts[anchor] - 1),
                animate=False,
                immediate=True,
            )

    def write_line(self, markup: str, scroll_end: bool | None = None) -> None:
        at_bottom = self.is_vertical_scroll_end
        width = self._cached_width or self.size.width
        self._cached_width = width

        self._lines.append(markup)
        count = self._wrap_markup(markup)
        self._wrap_counts.append(count)
        self._wrap_prefix.append(self._wrap_prefix[-1] + count)
        self._total_visual += count
        self.virtual_size = Size(width, self._total_visual)

        if scroll_end or (
            scroll_end is None and at_bottom and self._selected_line is None
        ):
            self.scroll_end(animate=False, immediate=True, x_axis=False)

    def prepend_lines(self, markups: list[str]) -> None:
        if not markups:
            return
        width = self._cached_width or self.size.width
        self._cached_width = width

        new_counts = [self._wrap_markup(m) for m in markups]
        new_visual = sum(new_counts)

        self._lines[0:0] = markups
        if self._selected_line is not None:
            self._selected_line += len(markups)
        self._wrap_counts[0:0] = new_counts
        self._recompute_prefix()
        self._render_line_cache.clear()
        self.virtual_size = Size(width, self._total_visual)
        self.scroll_to(y=self.scroll_y + new_visual, animate=False, immediate=True)

    def render_line(self, y: int) -> Strip:
        _, scroll_y = self.scroll_offset
        abs_y = scroll_y + y
        width = self.size.width
        wrap_width = self._cached_width or width
        rich_style = self.rich_style

        if abs_y >= self._total_visual:
            return Strip.blank(width, rich_style)
        if abs_y in self._render_line_cache:
            return self._render_line_cache[abs_y]

        logical_idx = bisect.bisect_right(self._wrap_prefix, abs_y) - 1
        text = Text.from_markup(self._lines[logical_idx], style=rich_style)
        wrapped = text.wrap(self.app.console, wrap_width)

        if logical_idx == self._selected_line:
            for line_text in wrapped:
                line_text.stylize(Style(bold=True, reverse=True))

        base = self._wrap_prefix[logical_idx]
        for i, line_text in enumerate(wrapped):
            segments = [
                segment
                if segment.style is not None
                else Segment(segment.text, _EMPTY_STYLE, segment.control)
                for segment in line_text.render(self.app.console)
            ]
            strip = Strip(segments, line_text.cell_len)
            strip = strip.crop_extend(0, width, rich_style)
            self._render_line_cache[base + i] = strip

        try:
            return self._render_line_cache[abs_y]
        except KeyError:
            return Strip.blank(width, rich_style)

    def notify_style_update(self) -> None:
        super().notify_style_update()
        self._render_line_cache.clear()

    def on_resize(self, event: events.Resize) -> None:
        if event.size.width != self._cached_width:
            self._reflow()

    def on_click(self, event: events.Click) -> None:
        _, scroll_y = self.scroll_offset
        visual_y = scroll_y + event.y
        logical_idx = bisect.bisect_right(self._wrap_prefix, visual_y) - 1
        if 0 <= logical_idx < len(self._lines):
            self._select_line(logical_idx)
            self.focus()
            event.stop()

    def _select_line(self, index: int) -> None:
        self._selected_line = index
        self._render_line_cache.clear()
        self.refresh()

    def _move_selection(self, delta: int) -> None:
        if not self._lines:
            return
        if self._selected_line is None:
            index = bisect.bisect_right(self._wrap_prefix, int(self.scroll_y)) - 1
        else:
            index = self._selected_line + delta
        index = max(0, min(index, len(self._lines) - 1))
        self._select_line(index)
        top = self._wrap_prefix[index]
        bottom = self._wrap_prefix[index + 1]
        if top < self.scroll_y or bottom - top >= self.size.height:
            self.scroll_to(y=top, animate=False, immediate=True)
        elif bottom > self.scroll_y + self.size.height:
            self.scroll_to(y=bottom - self.size.height, animate=False, immediate=True)
        self._try_load_previous()

    def action_select_previous(self) -> None:
        self._move_selection(-1)

    def action_select_next(self) -> None:
        self._move_selection(1)

    def action_copy_selected_line(self) -> None:
        if self._selected_line is None:
            self.app.notify("Select a log row before copying", timeout=2.0)
            return
        plain = Text.from_markup(self._lines[self._selected_line]).plain
        self.app.copy_to_clipboard(plain)
        self.app.notify("Log row copied", timeout=2.0)

    def _try_load_previous(self) -> None:
        if not self._has_more() or self.scroll_y > 0:
            return
        self._load_page()

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        super()._on_mouse_scroll_up(event)
        self._try_load_previous()

    def action_scroll_up(self) -> None:
        super().action_scroll_up()
        self._try_load_previous()

    def action_page_up(self) -> None:
        super().action_page_up()
        self._try_load_previous()

    def action_scroll_home(self) -> None:
        super().action_scroll_home()
        self._try_load_previous()


class _ConsoleFooter(Static):
    def action_copy(self) -> None:
        if isinstance(self.parent, DebugConsole):
            self.parent.action_copy()

    def action_close(self) -> None:
        if isinstance(self.parent, DebugConsole):
            self.parent.action_close()


class DebugConsole(Vertical):
    class Closed(Message):
        pass

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close debug console", show=False),
        Binding("c", "copy", "Copy selected log row", show=False),
    ]

    def action_copy(self) -> None:
        if self._log_view is not None:
            self._log_view.action_copy_selected_line()

    def action_close(self) -> None:
        self.post_message(self.Closed())

    @property
    def owns_interaction(self) -> bool:
        focused = self.app.screen.focused
        return self.screen is self.app.screen and (
            focused is self or (focused is not None and self in focused.ancestors)
        )

    def __init__(
        self, log_source: DebugLogSource, page_size: int = DEFAULT_LOG_PAGE_SIZE
    ) -> None:
        super().__init__(id="debug-console")
        self._log_source = log_source
        self._log_view: _LogView | None = None
        self._cursor: int | None = None
        self._has_more: bool = True
        self._page_size = page_size
        self._seen_entry_ids: set[str] = set()
        self._reading = False
        self._state_row: Static | None = None
        self._input_focus_target: Widget | None = None
        self._input_can_focus: bool | None = None

    def compose(self) -> ComposeResult:
        yield Static("Debug console", id="debug-console-header")
        self._state_row = Static(
            "Running: Loading debug logs", id="debug-console-state"
        )
        self._state_row.styles.height = 1
        self._state_row.styles.width = "100%"
        self._state_row.styles.color = self.app.theme_variables["text-muted"]
        yield self._state_row
        self._log_view = _LogView(
            load_page=self._schedule_load_page,
            has_more=lambda: self._has_more and self._cursor is not None,
            id="debug-console-log",
        )
        yield self._log_view
        footer = _ConsoleFooter(
            "↑/↓ j/k Select row · [@click=copy]c Copy selected[/] · "
            "[@click=close]Esc / Ctrl+\\ Close[/]",
            id="debug-console-footer",
        )
        footer.styles.height = "auto"
        yield footer

    def on_mount(self) -> None:
        self._update_geometry()
        self._schedule_load_page()
        self.set_interval(LOG_POLL_INTERVAL, self._schedule_poll)

    def on_resize(self, event: events.Resize) -> None:
        self._update_geometry()

    def _update_geometry(self) -> None:
        fullscreen = self.app.size.width < DEBUG_DOCK_MIN_WIDTH
        promoted = fullscreen and not self.has_class("-fullscreen")
        self.set_class(fullscreen, "-fullscreen")
        self.styles.width = self.app.size.width if fullscreen else DEBUG_DOCK_WIDTH
        if promoted:
            self._suspend_input_focusability()
            self.set_timer(0.05, self._focus_fullscreen_log)
        elif not fullscreen and not self.owns_interaction:
            self._restore_input_focusability()

    def _suspend_input_focusability(self) -> None:
        if self._input_focus_target is None:
            inputs = self.app.query("#input")
            if inputs:
                self._input_focus_target = inputs.first()
                self._input_can_focus = self._input_focus_target.can_focus
                self._input_focus_target.can_focus = False

    def on_unmount(self) -> None:
        self._restore_input_focusability()

    def _restore_input_focusability(self) -> None:
        if self._input_focus_target is not None and self._input_can_focus is not None:
            self._input_focus_target.can_focus = self._input_can_focus
        self._input_focus_target = None
        self._input_can_focus = None

    def _focus_fullscreen_log(self) -> None:
        if not self.is_mounted or self.screen is not self.app.screen:
            return
        if self._log_view is None or not self.has_class("-fullscreen"):
            return
        if (
            self.owns_interaction
            or self.screen.focused is None
            or self.screen.focused is self._input_focus_target
        ):
            self._log_view.focus()

    def _show_state(self, text: str | None, *, failed: bool = False) -> None:
        if self._state_row is None:
            return
        self._state_row.display = text is not None
        if text is not None:
            self._state_row.update(text)
            self._state_row.styles.color = self.app.theme_variables[
                "error" if failed else "text-muted"
            ]

    def _update_loaded_state(self) -> None:
        if self._log_view is not None and not self._log_view._lines:
            self._show_state(
                "Info: No debug logs yet. New logs appear here automatically."
            )
        else:
            self._show_state(None)

    def _schedule_load_page(self) -> None:
        if self._reading:
            return
        self.run_worker(self._load_page())

    async def _load_page(self) -> None:
        if self._log_view is None or self._reading:
            return
        self._reading = True
        try:
            result = await self._log_source.read_logs(
                limit=self._page_size, offset=self._cursor or 0
            )
            self._cursor = result.cursor
            self._has_more = result.has_more
            entries = [
                entry
                for entry in result.entries
                if entry.id not in self._seen_entry_ids
            ]
            self._seen_entry_ids.update(entry.id for entry in entries)
            markups = [self._format_entry(entry) for entry in reversed(entries)]
            self._log_view.prepend_lines(markups)
            self._update_loaded_state()
            self._fill_viewport()
        except Exception as exc:
            self._show_state(
                f"Failed: Could not load debug logs ({exc}). Existing logs remain "
                "visible; close and reopen the console to retry.",
                failed=True,
            )
        finally:
            self._reading = False

    def _schedule_poll(self) -> None:
        if self._reading:
            return
        self.run_worker(self._poll_latest())

    async def _poll_latest(self) -> None:
        if self._log_view is None or self._reading:
            return
        self._reading = True
        try:
            result = await self._log_source.read_logs(limit=500)
            entries = [
                entry
                for entry in result.entries
                if entry.id not in self._seen_entry_ids
            ]
            if not entries:
                self._update_loaded_state()
                return
            self._seen_entry_ids.update(entry.id for entry in entries)
            if self._cursor is not None:
                self._cursor += len(entries)
            for entry in reversed(entries):
                self._log_view.write_line(self._format_entry(entry))
            self._update_loaded_state()
        except Exception as exc:
            self._show_state(
                f"Failed: Could not refresh debug logs ({exc}). Existing logs "
                "remain visible; close and reopen the console to retry.",
                failed=True,
            )
        finally:
            self._reading = False

    def _fill_viewport(self) -> None:
        """Load enough logs to fill the viewport, then scroll to the bottom."""
        if self._log_view is None or not self._has_more:
            return
        self.call_after_refresh(self._check_and_fill)

    def _check_and_fill(self) -> None:
        if self._log_view is None or not self._has_more:
            return
        if self._log_view.virtual_size.height <= self._log_view.size.height:
            self._schedule_load_page()

    def _format_entry(self, entry: DebugLogEntry) -> str:
        role = LOG_LEVEL_ROLES.get(entry.level, "$text-muted").lstrip("$")
        color = self.app.theme_variables[role]
        muted = self.app.theme_variables["text-muted"]
        ts = entry.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        message = decode_log_message(entry.message)
        safe_message = escape(message)
        return f"[{muted}]{ts}[/] [{color}]{entry.level:<8}[/] {safe_message}"
