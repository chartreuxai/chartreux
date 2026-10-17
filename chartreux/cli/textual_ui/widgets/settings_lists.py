"""Value/draft helpers and option-list widgets for the settings screen."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
import re
from typing import Any, ClassVar, Literal, Protocol

from pydantic import JsonValue
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.strip import Strip
from textual.widgets import OptionList, SelectionList

from chartreux.app_server.protocol import SettingDescriptorWire
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.widgets.checklist import Checklist
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList


@dataclass(frozen=True)
class _DraftEntry:
    token: int
    value: str


@dataclass(frozen=True)
class _ListEditorContext:
    mode: Literal["edit", "add-item", "add-pattern"]
    token: int | None = None


def _draft_values(entries: list[_DraftEntry] | None) -> list[str]:
    return [entry.value for entry in entries or []]


def inventory_name_matches(name: str, entries: list[str]) -> bool:
    """Mirror the server's case-insensitive full-name glob and re: matching."""
    for raw in entries:
        if not (entry := (raw or "").strip()):
            continue
        if entry.startswith("re:"):
            try:
                if re.fullmatch(entry[3:], name, flags=re.IGNORECASE) is not None:
                    return True
            except re.error:
                continue
        elif fnmatch(name.lower(), entry.lower()):
            return True
    return False


def inventory_item_state(
    name: str,
    enabled: list[str],
    disabled: list[str],
    *,
    category: Literal["tools", "skills", "agents"] = "tools",
) -> tuple[bool, bool]:
    active = enabled + disabled if category == "tools" else enabled or disabled
    effective = (
        (not enabled or inventory_name_matches(name, enabled))
        and not inventory_name_matches(name, disabled)
        if category == "tools"
        else inventory_name_matches(name, enabled)
        if enabled
        else not inventory_name_matches(name, disabled)
    )
    pattern_driven = any(
        entry.lower() != name.lower() and inventory_name_matches(name, [entry])
        for entry in active
    )
    return effective, pattern_driven


def toggle_inventory_name(
    name: str,
    enabled: list[str],
    disabled: list[str],
    *,
    is_enabled: bool,
    category: Literal["tools", "skills", "agents"] = "tools",
) -> tuple[list[str], list[str]]:
    enabled, disabled = list(enabled), list(disabled)
    if category == "tools":
        if is_enabled:
            enabled = [entry for entry in enabled if entry.lower() != name.lower()]
            # An empty allowlist restores defaults, so keep this name blocked.
            if not enabled and not inventory_name_matches(name, disabled):
                disabled.append(name)
        else:
            disabled = [entry for entry in disabled if entry.lower() != name.lower()]
            if enabled and not inventory_name_matches(name, enabled):
                enabled.append(name)
        return enabled, disabled
    target = enabled if enabled else disabled
    if is_enabled:
        if enabled:
            target[:] = [entry for entry in target if entry.lower() != name.lower()]
        else:
            target.append(name)
    elif enabled:
        target.append(name)
    else:
        target[:] = [entry for entry in target if entry.lower() != name.lower()]
    return enabled, disabled


def parse_setting_value(descriptor: SettingDescriptorWire, text: str) -> JsonValue:
    """Parse numeric text without relaxing descriptor domain constraints."""
    try:
        if descriptor.kind == "int":
            value: JsonValue = int(text.strip())
        elif descriptor.kind == "float":
            value = float(text.strip())
        else:
            value = text.strip()
        descriptor.validate_value(value)
    except (ValueError, OverflowError) as exc:
        bound = descriptor.minimum
        rule = (
            f" ({'greater than' if descriptor.exclusive_minimum else 'at least'} {bound})"
            if bound is not None
            else ""
        )
        raise ValueError(f"Enter a valid {descriptor.kind} value{rule}.") from exc
    return value


DOUBLE_CLICK = 2


class SettingsListHost(Protocol):
    """Screen surface the list widgets act on; the settings screen implements it."""

    _busy: bool
    _help_open: bool
    _pointer_command_started: bool

    # Read-only on the host: the widgets only inspect the editor state; the
    # owning controller (and the screen's forwarding property) may rebind it.
    @property
    def _editing(self) -> str | None: ...

    # Read-only on the host: the widgets only inspect the confirmation state;
    # the owning controller (and the screen's forwarding property) may rebind
    # it.
    @property
    def _confirmation(self) -> tuple[str, SettingDescriptorWire, Any] | None: ...

    # Read-only on the host: the widgets only inspect the inventory draft; the
    # owning controller (and the screen's forwarding property) may rebind it.
    @property
    def _inventory_draft(self) -> dict[str, list[_DraftEntry]] | None: ...

    def _begin_pointer_event(self, event: events.Click) -> bool: ...
    def _can_mutate(self, *, editor: bool = False) -> bool: ...
    def _update_help(self) -> None: ...
    def _update_hint(self) -> None: ...
    def _toggle_inventory(self, name: str) -> bool: ...
    def _render_checklist(self, *, highlight: str | None = None) -> None: ...
    def _select_enum_draft(self) -> None: ...
    def _mark_cursor(self) -> None: ...


class ConfirmationText(VerticalScroll):
    """Focusable scroll target for long confirmation consequences."""

    can_focus = True
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "scroll_up", show=False),
        Binding("down", "scroll_down", show=False),
        Binding("pageup", "page_up", show=False),
        Binding("pagedown", "page_down", show=False),
    ]


class _BoundedOptionList(OptionList):
    """Settings-local movement never wraps or transfers focus."""

    def __init__(
        self, *args: Any, host: SettingsListHost | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._host = host

    def focus_on_click(self) -> bool:
        # Focus only after the pointer command's ownership guard has run.
        return False

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        event.stop()

    def on_show(self, event: events.Show | None = None) -> None:
        # OptionList's default Show handler scrolls *after* layout callbacks,
        # overwriting the immediate opener's restored viewport.
        if event is not None:
            event.prevent_default()

    def _move_bounded(self, direction: int, distance: int = 1) -> None:
        valid = [i for i, option in enumerate(self.options) if not option.disabled]
        if not valid:
            self.highlighted = None
            return
        current = self.highlighted
        if current not in valid:
            self.highlighted = valid[0 if direction > 0 else -1]
            return
        ordinal = valid.index(current)
        self.highlighted = valid[
            max(0, min(len(valid) - 1, ordinal + direction * distance))
        ]

    def action_cursor_down(self) -> None:
        self._move_bounded(1)

    def action_cursor_up(self) -> None:
        self._move_bounded(-1)

    def action_first(self) -> None:
        self.highlighted = next(
            (i for i, option in enumerate(self.options) if not option.disabled), None
        )

    def action_last(self) -> None:
        self.highlighted = next(
            (
                i
                for i in reversed(range(self.option_count))
                if not self.options[i].disabled
            ),
            None,
        )

    def action_page_down(self) -> None:
        self._move_bounded(1, max(1, self.scrollable_content_region.height))

    def action_page_up(self) -> None:
        self._move_bounded(-1, max(1, self.scrollable_content_region.height))


class SettingsConfirmationList(_BoundedOptionList):
    """Page keys inspect consequences without leaving confirmation actions."""

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        host = self._host
        if host is None or host._busy:
            return
        if event.chain != 1:
            return
        index = event.style.meta.get("option")
        if index is not None and not self.get_option_at_index(index).disabled:
            self.highlighted = index
            self.focus()
            host._pointer_command_started = True
            self.action_select()

    def action_page_down(self) -> None:
        self.screen.query_one(
            "#settings-confirmation-scroll", ConfirmationText
        ).action_page_down()

    def action_page_up(self) -> None:
        self.screen.query_one(
            "#settings-confirmation-scroll", ConfirmationText
        ).action_page_up()


class SettingsChecklist(_BoundedOptionList, Checklist):
    """Membership activation changes only the collection draft."""

    COMPONENT_CLASSES: ClassVar[set[str]] = SelectionList.COMPONENT_CLASSES

    BINDINGS: ClassVar[list[BindingType]] = [
        *SelectionList.BINDINGS,
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("enter", "select", "Toggle draft", show=False, priority=True),
    ]

    def render_line(self, y: int) -> Strip:
        line = super().render_line(y)
        index = self.scroll_offset.y + y
        if index >= self.option_count:
            return line
        option = self.get_option_at_index(index)
        segments = list(line)
        if not segments:
            return line
        focused = self.has_focus and self.highlighted == index
        style = Style(reverse=True, bold=True) if focused else Style()
        if option.value.startswith(("\x00pattern:", "\x00state:")):
            # Replace the membership column on action rows with ordinary spacing.
            text = "".join(segment.text for segment in segments)
            text = "     " + text[5:]
            segments = [
                Segment(
                    text,
                    (segments[0].style or self.rich_style)
                    + style
                    + Style(meta={"option": index}),
                )
            ]
        elif focused:
            segments = [
                Segment(segment.text, (segment.style or self.rich_style) + style)
                for segment in segments
            ]
        if self.highlighted == index:
            text = "".join(segment.text for segment in segments)
            cursor_style = (
                (segments[0].style or self.rich_style)
                + style
                + Style(meta={"option": index})
            )
            segments = (
                [
                    Segment(f"{chrome_glyph('cursor')} ", cursor_style),
                    Segment(text[2:], cursor_style),
                ]
                if focused
                else [
                    Segment(f"{chrome_glyph('cursor')} ", cursor_style),
                    *list(Strip(segments).crop(2, len(text))),
                ]
            )
        return Strip(segments).extend_cell_length(
            self.size.width, self.rich_style + style
        )

    def _on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        event.prevent_default()
        # A queued highlight from before a draft refresh may no longer exist.
        if (
            event.option_index >= self.option_count
            or self.get_option_at_index(event.option_index) is not event.option
        ):
            event.stop()
            return
        super()._on_option_list_option_highlighted(event)

    def on_selection_list_selection_highlighted(
        self, event: SelectionList.SelectionHighlighted
    ) -> None:
        if self._host is not None:
            self._host._update_help()
            self._host._update_hint()

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        host = self._host
        if (
            host is None
            or not host._begin_pointer_event(event)
            or not host._can_mutate()
        ):
            return
        index = event.style.meta.get("option")
        if index is not None and not self.get_option_at_index(index).disabled:
            self.highlighted = index
            self.focus()
            if event.chain == 1:
                self.action_select()

    def action_select(self) -> None:
        host = self._host
        if host is None or not host._can_mutate():
            return
        if self.highlighted is not None:
            value = self.get_option_at_index(self.highlighted).value
            if value.startswith(("\x00pattern:", "\x00state:")):
                return
            if host._inventory_draft is not None:
                if host._toggle_inventory(value):
                    host._render_checklist(highlight=value)
                return
        super().action_select()

    def on_key(self, event: events.Key) -> None:
        if event.key == "space" and self.highlighted is not None:
            value = self.get_option_at_index(self.highlighted).value
            if value.startswith(("\x00pattern:", "\x00state:")):
                event.stop()
                event.prevent_default()


class SettingsGroupList(_BoundedOptionList, NavigableOptionList):
    """Separate collection navigation surface with focus-dependent hints."""

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        event.stop()

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        host = self._host
        if host is None:
            return
        if (
            host._busy
            or not host._begin_pointer_event(event)
            or host._help_open
            or host._confirmation is not None
            or host._editing is not None
        ):
            return
        index = event.style.meta.get("option")
        if index is None or self.get_option_at_index(index).disabled:
            return
        self.highlighted = index
        self.focus()
        if self.id == "settings-choices":
            if event.chain == 1:
                host._select_enum_draft()
        elif self.id == "settings-actions":
            if event.chain == 1:
                host._pointer_command_started = True
                self.action_select()
        elif event.chain == DOUBLE_CLICK:
            self.action_select()

    def on_focus(self) -> None:
        if self._host is not None:
            self.screen.call_later(self._host._mark_cursor)

    def on_blur(self) -> None:
        if self._host is not None:
            self.screen.call_later(self._host._mark_cursor)


class SettingsOptionList(_BoundedOptionList, NavigableOptionList):
    """Keep the list focused while printable keys edit the search query."""

    def __init__(
        self,
        on_filter: Callable[[str], None],
        *,
        host: SettingsListHost | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(host=host, **kwargs)
        self._query = ""
        self.on_filter = on_filter
        self.editing = False

    def on_key(self, event: events.Key) -> None:
        if self.editing or event.key == "space":
            return
        if event.key in {"j", "k"} and not self._query:
            return
        if event.key == "backspace":
            self._query = self._query[:-1]
        elif (
            (char := event.character) is not None
            and len(char) == 1
            and char.isprintable()
        ):
            self._query += char
        else:
            return
        self.on_filter(self._query)
        event.stop()
        event.prevent_default()

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        host = self._host
        if host is not None:
            if not host._begin_pointer_event(event):
                return
            if (
                host._busy
                or host._help_open
                or host._editing is not None
                or host._confirmation is not None
            ):
                return
        index = event.style.meta.get("option")
        if index is not None and not self.get_option_at_index(index).disabled:
            self.highlighted = index
            self.focus()
            if event.chain == DOUBLE_CLICK:
                self.action_select()
