"""List widgets for the provider workbench with an injected host protocol."""

from __future__ import annotations

from typing import Any, ClassVar

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.strip import Strip
from textual.widgets import OptionList

from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.providers.workbench_types import CURSOR_GUTTER, WorkbenchWidgetHost
from chartreux.ui.widgets.checklist import CHECKBOX_SEGMENTS, Checklist
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class DetailHelp(NoMarkupStatic):
    """Scrollable selected-item text reachable by Tab when a summary truncates."""

    def __init__(
        self, *args: Any, host: WorkbenchWidgetHost | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._host = host

    def on_focus(self, event: events.Focus) -> None:
        if self._host is not None:
            self._host.schedule_update_help()

    can_focus = True

    def focus_on_click(self) -> bool:
        return False

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if self._host is not None and self._host.confirm_open():
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_down(event)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self._host is not None and self._host.confirm_open():
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_up(event)

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("down", "scroll_down", "Scroll down", show=False),
        Binding("up", "scroll_up", "Scroll up", show=False),
    ]


def _bounded_cursor(widget: OptionList, direction: int) -> None:
    """Move to the next enabled row without wrapping or leaving the group."""
    anchor = widget.highlighted
    if anchor is None:
        anchor = -1 if direction > 0 else widget.option_count
    for index in range(
        anchor + direction, widget.option_count if direction > 0 else -1, direction
    ):
        if not widget.get_option_at_index(index).disabled:
            widget.highlighted = index
            return


class WorkbenchList(NavigableOptionList):
    """Two-cell cursor gutter, independent of each row's value."""

    def __init__(
        self, *args: Any, host: WorkbenchWidgetHost | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._host = host

    def focus(self, scroll_visible: bool = True) -> WorkbenchList:
        super().focus(scroll_visible=scroll_visible)
        host = self._host
        if host is not None:
            if self.id != "wb-confirm-actions":
                host.record_help_focus(self.id)
            host.update_help()
        return self

    BINDINGS: ClassVar[list[BindingType]] = [
        *NavigableOptionList.BINDINGS,
        Binding("enter", "select", "Select", priority=True, show=False),
    ]

    def focus_on_click(self) -> bool:
        host = self._host
        if host is not None:
            return host.pointer_focus(self.id)
        return True

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if (
            self._host is not None
            and self._host.confirm_open()
            and self.id != "wb-confirm-actions"
        ):
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_down(event)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if (
            self._host is not None
            and self._host.confirm_open()
            and self.id != "wb-confirm-actions"
        ):
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_up(event)

    async def _on_click(self, event: events.Click) -> None:
        # A double click on a provider can land its second click on a control
        # exposed by the first click. The first click already activates the row;
        # later clicks in that gesture must not activate the destination view.
        host = self._host
        if event.chain > 1 or (
            host is not None
            and (host.consume_help_click(self.id) or not host.can_activate(self.id))
        ):
            event.stop()
            event.prevent_default()
            return
        if self.id == "wb-protocol" and host is not None:
            index = event.style.meta.get("option")
            if index is not None:
                option = self.get_option_at_index(index)
                if not option.disabled and option.id:
                    self.highlighted = index
                    host.select_protocol(str(option.id))
            event.stop()
            event.prevent_default()
            return
        await super()._on_click(event)
        event.stop()
        event.prevent_default()

    def action_select(self) -> None:
        host = self._host
        if host is not None:
            if not host.can_activate(self.id):
                return
            if self.id == "wb-protocol":
                host.accept_protocol()
                return
        super().action_select()

    def action_cursor_down(self) -> None:
        _bounded_cursor(self, 1)

    def action_cursor_up(self) -> None:
        _bounded_cursor(self, -1)

    def action_page_down(self) -> None:
        if self.id == "wb-confirm-actions":
            host = self._host
            if host is None:
                self.screen.query_one("#wb-confirm", VerticalScroll).scroll_page_down(
                    animate=False
                )
            else:
                host.page_confirm(1)
            return
        for _ in range(max(1, self.scrollable_content_region.height)):
            _bounded_cursor(self, 1)

    def action_page_up(self) -> None:
        if self.id == "wb-confirm-actions":
            host = self._host
            if host is None:
                self.screen.query_one("#wb-confirm", VerticalScroll).scroll_page_up(
                    animate=False
                )
            else:
                host.page_confirm(-1)
            return
        for _ in range(max(1, self.scrollable_content_region.height)):
            _bounded_cursor(self, -1)

    def on_key(self, event: events.Key) -> None:
        host = self._host
        if event.key == "space" and host is not None and not host.can_activate(self.id):
            event.stop()
            event.prevent_default()
            return
        if (
            event.key == "space"
            and self.id == "wb-detail-fields"
            and host is not None
            and self.highlighted_option is not None
            and self.highlighted_option.id == "images"
        ):
            host.toggle_detail_images()
            event.stop()
            event.prevent_default()
            return
        if self.id == "wb-protocol" and event.key == "space" and host is not None:
            if self.highlighted_option and self.highlighted_option.id:
                host.select_protocol(str(self.highlighted_option.id))
            event.stop()
            event.prevent_default()

    def on_focus(self, event: events.Focus) -> None:
        host = self._host
        if host is not None:
            if self.id != "wb-confirm-actions":
                host.record_help_focus(self.id)
            host.schedule_update_help()

    def render_line(self, y: int) -> Strip:
        line = super().render_line(y)
        try:
            index, _ = self._lines[self.scroll_offset.y + y]
        except IndexError:
            return line
        if line.cell_length < CURSOR_GUTTER:
            return line
        option = self.get_option_at_index(index)
        current = self.has_focus and self.highlighted == index and not option.disabled
        marker = f"{chrome_glyph('cursor')} " if current else "  "
        style = (list(line)[0].style or self.rich_style) + Style(meta={"option": index})
        return Strip([
            Segment(marker, style),
            *list(line.crop(0, line.cell_length - CURSOR_GUTTER)),
        ])


class BrowserList(WorkbenchList):
    """Type to filter while browsing, without stealing keys from an editor."""

    def on_key(self, event: events.Key) -> None:
        host = self._host
        if host is None:
            return
        if host.is_busy():
            event.stop()
            event.prevent_default()
            return
        if event.key in {"j", "k", "space"} and not host.get_filter():
            return
        if event.key == "backspace":
            host.set_filter(host.get_filter()[:-1])
        elif (char := event.character) and len(char) == 1 and char.isprintable():
            host.set_filter(host.get_filter() + char)
        else:
            return
        host.refresh_browser()
        event.stop()
        event.prevent_default()


class ModelChecklist(Checklist):
    """Space toggles a deployment; Enter opens its metadata editor."""

    def __init__(
        self, *args: Any, host: WorkbenchWidgetHost | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._host = host

    def focus_on_click(self) -> bool:
        host = self._host
        if host is not None:
            return host.pointer_focus(self.id)
        return True

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if self._host is not None and self._host.confirm_open():
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_down(event)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self._host is not None and self._host.confirm_open():
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_up(event)

    def _toggle_user_selection(self, value: str) -> None:
        host = self._host
        if host is None or not host.can_activate(self.id):
            return
        self.toggle(value)
        # Accept the user intent in this turn, before a queued Help/Back key can
        # change ownership. Queued programmatic SelectedChanged stays guarded.
        host.selection_changed(self)

    def action_select(self) -> None:
        host = self._host
        if host is not None and not host.can_activate(self.id):
            return
        if self.id != "wb-models":
            super().action_select()
        elif self.highlighted is not None:
            option = self.get_option_at_index(self.highlighted)
            if not option.disabled and not option.value.startswith("\x00"):
                self._toggle_user_selection(option.value)

    def on_focus(self, event: events.Focus) -> None:
        if self._host is not None:
            self._host.schedule_update_help()

    def render_line(self, y: int) -> Strip:
        line = super().render_line(y)
        index = self.scroll_offset.y + y
        if index >= self.option_count:
            return line
        segments = list(line)
        if not segments:
            return line
        style = segments[0].style or self.rich_style
        value = self.get_option_at_index(index).value
        if value.startswith("\x00") and len(segments) >= CHECKBOX_SEGMENTS:
            segments[1:CHECKBOX_SEGMENTS] = [Segment("   ", style)]
        if self.has_focus and index == self.highlighted:
            segments[0] = Segment(f"{chrome_glyph('cursor')} ", style)
        line = Strip(segments)
        if self.id == "wb-models" and not value.startswith("\x00"):
            details = " [Details]"
            details_style = style + Style(meta={"model_details": index})
            line = Strip([
                *line.crop_extend(
                    0,
                    max(0, self.scrollable_content_region.width - len(details)),
                    style,
                ),
                Segment(details, details_style),
            ])
        return line

    async def _on_click(self, event: events.Click) -> None:
        if self.id != "wb-models":
            return
        event.stop()
        event.prevent_default()
        host = self._host
        if (
            host is None
            or host.consume_help_click(self.id)
            or not host.can_activate(self.id)
            or event.chain > 1
        ):
            return
        details_index = event.style.meta.get("model_details")
        index = (
            details_index
            if details_index is not None
            else event.style.meta.get("option")
        )
        if index is None:
            return
        option = self.get_option_at_index(index)
        if option.disabled or option.value.startswith("\x00"):
            return
        self.highlighted = index
        self.focus()
        if details_index is not None:
            self.action_detail()
        else:
            self._toggle_user_selection(option.value)

    def action_cursor_down(self) -> None:
        if self.id == "wb-models":
            _bounded_cursor(self, 1)
        else:
            super().action_cursor_down()

    def action_cursor_up(self) -> None:
        if self.id == "wb-models":
            _bounded_cursor(self, -1)
        else:
            super().action_cursor_up()

    def action_page_down(self) -> None:
        for _ in range(max(1, self.scrollable_content_region.height)):
            _bounded_cursor(self, 1)

    def action_page_up(self) -> None:
        for _ in range(max(1, self.scrollable_content_region.height)):
            _bounded_cursor(self, -1)

    def on_key(self, event: events.Key) -> None:
        if event.key == "space" and self.highlighted is not None and self.option_count:
            value = self.get_option_at_index(self.highlighted).value
            if isinstance(value, str) and value.startswith("\x00"):
                event.stop()
                event.prevent_default()
                return
        # Normal keys continue through SelectionList's key bindings; calling
        # Widget.on_key here is invalid because SelectionList has no such
        # superclass handler.
        return

    BINDINGS: ClassVar[list[BindingType]] = [
        # Process user inclusion intent before a following priority Help/Back key.
        Binding("space", "select", "Toggle option", priority=True, show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("enter", "detail", "Details", priority=True, show=False),
    ]

    def action_detail(self) -> None:
        host = self._host
        if host is not None and host.can_activate(self.id):
            if self.id == "wb-image-support":
                return
            if (
                self.id == "wb-models"
                and self.highlighted is not None
                and self.get_option_at_index(self.highlighted).value == "\x00empty"
            ):
                return
            if self.highlighted is not None:
                value = self.get_option_at_index(self.highlighted).value
                if not value.startswith("\x00"):
                    host.open_detail(value)
