"""Searchable, flat settings browser with inline single-leaf editing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
import re
from typing import Any, ClassVar, Literal

from pydantic import JsonValue
from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.visual import VisualType
from textual.widget import Widget
from textual.widgets import Input, OptionList, SelectionList
from textual.widgets.option_list import Option, OptionDoesNotExist
from textual.widgets.selection_list import Selection

from chartreux.app_server.protocol import (
    STATUS_LINE_PATHS,
    SettingDescriptorWire,
    SettingsReadResponse,
)
from chartreux.cli.textual_ui.screens.status_line_settings import (
    StatusLineSettingsResult,
    StatusLineSettingsScreen,
)
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.settings_service import SettingsService
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.checklist import Checklist
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from chartreux.ui.widgets.vscode_compat import VscodeCompatInput

DOUBLE_CLICK = 2
MIN_MODAL_WIDTH = 84
MIN_MODAL_HEIGHT = 28
MIN_SHORTCUTS_WITH_OVERFLOW = 2


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


class ConfirmationText(VerticalScroll):
    """Focusable scroll target for long confirmation consequences."""

    can_focus = True
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "scroll_up", show=False),
        Binding("down", "scroll_down", show=False),
        Binding("pageup", "page_up", show=False),
        Binding("pagedown", "page_down", show=False),
    ]


@dataclass(frozen=True)
class _GroupPosition:
    group: str
    context: str
    identities: tuple[str, ...]
    ordinal: int
    scroll_x: float
    scroll_y: float
    cursor_position: int | None = None


class _BoundedOptionList(OptionList):
    """Settings-local movement never wraps or transfers focus."""

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
        screen = self.screen
        if not isinstance(screen, SettingsScreen) or screen._busy:
            return
        if event.chain != 1:
            return
        index = event.style.meta.get("option")
        if index is not None and not self.get_option_at_index(index).disabled:
            self.highlighted = index
            self.focus()
            screen._pointer_command_started = True
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
        if isinstance(self.screen, SettingsScreen):
            self.screen._update_help()
            self.screen._update_hint()

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        screen = self.screen
        if (
            not isinstance(screen, SettingsScreen)
            or not screen._begin_pointer_event(event)
            or not screen._can_mutate()
        ):
            return
        index = event.style.meta.get("option")
        if index is not None and not self.get_option_at_index(index).disabled:
            self.highlighted = index
            self.focus()
            if event.chain == 1:
                self.action_select()

    def action_select(self) -> None:
        if not isinstance(self.screen, SettingsScreen) or not self.screen._can_mutate():
            return
        if self.highlighted is not None:
            value = self.get_option_at_index(self.highlighted).value
            if value.startswith(("\x00pattern:", "\x00state:")):
                return
            screen = self.screen
            if (
                isinstance(screen, SettingsScreen)
                and screen._inventory_draft is not None
            ):
                if screen._toggle_inventory(value):
                    screen._render_checklist(highlight=value)
                return
        super().action_select()

    def on_key(self, event: events.Key) -> None:
        if event.key == "space" and self.highlighted is not None:
            value = self.get_option_at_index(self.highlighted).value
            if value.startswith(("\x00pattern:", "\x00state:")):
                event.stop()
                event.prevent_default()


GROUPS = (
    "Interface",
    "Prompts & Compaction",
    "Project Context",
    "Subagents",
    "Session History",
    "Network",
    "Advanced & Tools",
)
LONG_LIST_HEADING_THRESHOLD = 100


def format_value(value: JsonValue) -> str:
    if value is None:
        return "—"
    if value == "":
        return '""'
    return str(value)


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


class SettingsGroupList(_BoundedOptionList, NavigableOptionList):
    """Separate collection navigation surface with focus-dependent hints."""

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        event.stop()

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        screen = self.screen
        if not isinstance(screen, SettingsScreen):
            return
        if (
            screen._busy
            or not screen._begin_pointer_event(event)
            or screen._help_open
            or screen._confirmation is not None
            or screen._editing is not None
        ):
            return
        index = event.style.meta.get("option")
        if index is None or self.get_option_at_index(index).disabled:
            return
        self.highlighted = index
        self.focus()
        if self.id == "settings-choices":
            if event.chain == 1:
                screen._select_enum_draft()
        elif self.id == "settings-actions":
            if event.chain == 1:
                screen._pointer_command_started = True
                self.action_select()
        elif event.chain == DOUBLE_CLICK:
            self.action_select()

    def on_focus(self) -> None:
        if isinstance(self.screen, SettingsScreen):
            self.screen.call_later(self.screen._mark_cursor)

    def on_blur(self) -> None:
        if isinstance(self.screen, SettingsScreen):
            self.screen.call_later(self.screen._mark_cursor)


class SettingsOptionList(_BoundedOptionList, NavigableOptionList):
    """Keep the list focused while printable keys edit the search query."""

    def __init__(self, on_filter: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
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
        screen = self.screen
        if isinstance(screen, SettingsScreen):
            if not screen._begin_pointer_event(event):
                return
            if (
                screen._busy
                or screen._help_open
                or screen._editing is not None
                or screen._confirmation is not None
            ):
                return
        index = event.style.meta.get("option")
        if index is not None and not self.get_option_at_index(index).disabled:
            self.highlighted = index
            self.focus()
            if event.chain == DOUBLE_CLICK:
                self.action_select()


class SettingsHints(NoMarkupStatic):
    """Clickable shortcut text does not add keyboard focus stops."""

    targets: list[tuple[int, int, str]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.targets = []

    def pointer_action_at(self, x: int, y: int) -> str | None:
        return next(
            (action for start, end, action in self.targets if start <= x < end), None
        )

    async def on_click(self, event: events.Click) -> None:
        action = self.pointer_action_at(event.x, event.y)
        if action and event.chain == 1:
            event.stop()
            if isinstance(self.screen, SettingsScreen):
                self.screen._pointer_command_started = True
            await self.screen.run_action(action)


class SettingsHelp(SettingsHints):
    """The detail region opens help; named mutation shortcuts stay explicit."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up,k", "scroll_up", show=False),
        Binding("down,j", "scroll_down", show=False),
        Binding("pageup", "page_up", show=False),
        Binding("pagedown", "page_down", show=False),
        Binding("home", "scroll_home", show=False),
        Binding("end", "scroll_end", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield NoMarkupStatic(self.content, id="settings-help-text")

    def update(self, content: VisualType = "", *, layout: bool = True) -> None:
        super().update(content, layout=layout)
        if self.is_mounted:
            self.query_one("#settings-help-text", NoMarkupStatic).update(
                content, layout=layout
            )

    async def on_click(self, event: events.Click) -> None:
        event.prevent_default()
        # Child text clicks bubble with child-local coordinates.
        action = self.pointer_action_at(
            event.screen_x - self.region.x, event.screen_y - self.region.y
        )
        if action and event.chain == 1:
            event.stop()
            if isinstance(self.screen, SettingsScreen):
                self.screen._pointer_command_started = True
            await self.screen.run_action(action)

    def pointer_action_at(self, x: int, y: int) -> str | None:
        child = self.query_one("#settings-help-text", NoMarkupStatic)
        line = "".join(
            segment.text for segment in child.render_line(y + int(self.scroll_y))
        )
        for key, action in (("Ctrl+R", "remove_override"), ("Ctrl+D", "delete_item")):
            index = line.find(key)
            if index >= 0 and cell_len(line[:index]) <= x < cell_len(
                line[:index]
            ) + len(key):
                return action
        return "help"


class SettingsScreen(ModalScreen[str | None]):
    """Full-screen browser; link commands are returned to the host on close."""

    SCOPED_CSS = False
    CSS_PATH = "settings.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("tab", "next_group", show=False, priority=True),
        Binding("shift+tab", "previous_group", show=False, priority=True),
        Binding("escape", "close", "Close", show=False, priority=True),
        Binding("ctrl+r", "remove_override", "Remove override", show=False),
        Binding("ctrl+d", "delete_item", "Delete item", show=False),
        Binding("f1", "help", "Help", show=False),
    ]

    def __init__(
        self, service: SettingsService, snapshot: SettingsReadResponse
    ) -> None:
        super().__init__(id="settings-screen")
        self.service = service
        self.snapshot = snapshot
        self.catalog = snapshot.catalog
        self.fields = {field.path: field for field in snapshot.fields}
        self._busy = False
        self._pointer_command_started = False
        self._needs_refresh = False
        self._cursor: str | None = None
        self._expanded: str | None = None
        self._next_draft_token = 0
        self._list_editor: _ListEditorContext | None = None
        self._list_draft: list[_DraftEntry] | None = None
        self._enum_draft: str | None = None
        self._inventory_draft: dict[str, list[_DraftEntry]] | None = None
        self._editing: str | None = None
        self._error = ""
        self._error_path: str | None = None
        self._unresolved: dict[str, str] = {}
        self._confirmation: tuple[str, SettingDescriptorWire, Any] | None = None
        self._confirmation_return_focus: _GroupPosition | None = None
        self._positions: dict[tuple[str, str], _GroupPosition] = {}
        self._group_contents_context: dict[str, str] = {}
        self._openers: dict[str, _GroupPosition] = {}
        self._pending_navigation: SettingDescriptorWire | None = None
        self._help_open = False
        self._child_returning = False
        self._help_return_focus: _GroupPosition | None = None
        self.return_state: tuple[str, str | None, int, str | None] | None = None

    def _group_context(self, group: str) -> str:
        return (
            "catalog"
            if group == "settings-options"
            else self._expanded or self._editing or "catalog"
        )

    def _visible_focus_groups(self) -> list[Widget]:
        if self._editing is not None or self._confirmation is not None:
            return []
        identifiers = [
            "settings-options",
            "settings-entries",
            "settings-choices",
            "settings-checklist",
            "settings-actions",
        ]
        if self._help_open:
            identifiers.append("settings-help")
        return [
            widget
            for identifier in identifiers
            if (widget := self.query_one(f"#{identifier}")).display and widget.can_focus
        ]

    # Keep identity-neighbour repair aligned with ProviderWorkbenchScreen's
    # _capture_group_position/_restore_group_position and MCPApp's equivalent.
    def _capture_group_position(
        self, widget: Widget | None = None
    ) -> _GroupPosition | None:
        widget = widget or self.focused
        if widget is None or widget.id is None:
            return None
        identities = (
            tuple(
                str(option.id)
                for option in widget.options
                if not option.disabled and option.id is not None
            )
            if isinstance(widget, OptionList)
            else ()
        )
        selected = widget.highlighted_option if isinstance(widget, OptionList) else None
        ordinal = (
            identities.index(str(selected.id))
            if selected and str(selected.id) in identities
            else 0
        )
        position = _GroupPosition(
            widget.id,
            self._group_context(widget.id),
            identities,
            ordinal,
            widget.scroll_x,
            widget.scroll_y,
            widget.cursor_position if isinstance(widget, Input) else None,
        )
        self._positions[(position.group, position.context)] = position
        return position

    def _restore_group_position(
        self, position: _GroupPosition | None, *, restore_focus: bool = True
    ) -> None:
        if position is None:
            return
        widget = self.query_one(f"#{position.group}")
        if isinstance(widget, OptionList):
            valid = [
                i for i, option in enumerate(widget.options) if not option.disabled
            ]
            ids = {str(widget.options[i].id): i for i in valid}
            for ordinal in sorted(
                range(len(position.identities)),
                key=lambda i: (abs(i - position.ordinal), i < position.ordinal),
            ):
                if position.identities[ordinal] in ids:
                    widget.highlighted = ids[position.identities[ordinal]]
                    break
            else:
                widget.highlighted = (
                    valid[min(position.ordinal, len(valid) - 1)] if valid else None
                )
        if restore_focus and widget.display and widget.can_focus:
            widget.focus(scroll_visible=False)

        def finish_restore() -> None:
            # Input's focus handler moves its caret; restore only after that handler.
            if (
                restore_focus
                and isinstance(widget, Input)
                and position.cursor_position is not None
            ):
                widget.cursor_position = position.cursor_position
            widget.scroll_to(
                x=position.scroll_x,
                y=position.scroll_y,
                animate=False,
                force=True,
                immediate=True,
            )

        self.call_after_refresh(finish_restore)

    def _remember_opener(self, owner: str) -> None:
        if (position := self._capture_group_position()) is not None:
            self._openers[owner] = position

    def _cycle_group(self, direction: int) -> None:
        groups = self._visible_focus_groups()
        if not groups:
            return
        self._capture_group_position()
        index = (
            groups.index(self.focused)
            if self.focused in groups
            else (-1 if direction > 0 else 0)
        )
        target = groups[(index + direction) % len(groups)]
        position = self._positions.get((
            target.id or "",
            self._group_context(target.id or ""),
        ))
        if self._help_open and target.id != "settings-help":
            self._close_help(restore=False)
        if position is not None:
            self._restore_group_position(position)
        else:
            target.focus()

    def action_next_group(self) -> None:
        self._cycle_group(1)

    def action_previous_group(self) -> None:
        self._cycle_group(-1)

    def _new_draft_entry(self, value: str) -> _DraftEntry:
        entry = _DraftEntry(self._next_draft_token, value)
        self._next_draft_token += 1
        return entry

    def _make_draft(self, values: list[str]) -> list[_DraftEntry]:
        return [self._new_draft_entry(value) for value in values]

    def _reconcile_draft(
        self, entries: list[_DraftEntry], values: list[str]
    ) -> list[_DraftEntry]:
        remaining = list(entries)
        result = []
        for value in values:
            index = next(
                (i for i, entry in enumerate(remaining) if entry.value == value), None
            )
            result.append(
                remaining.pop(index)
                if index is not None
                else self._new_draft_entry(value)
            )
        return result

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-content"):
            yield NoMarkupStatic("Settings", id="settings-title")
            pending = NoMarkupStatic(
                "Action pending · finish editing, then Esc to answer",
                id="settings-pending-action",
            )
            pending.display = False
            yield pending
            yield NoMarkupStatic(
                "Filter: type to filter · Click selects; double-click activates",
                id="settings-filter",
            )
            yield SettingsOptionList(self._filter, id="settings-options")
            yield NoMarkupStatic("", id="settings-active-editor")
            yield SettingsGroupList(id="settings-entries", classes="expanded-body")
            yield SettingsGroupList(id="settings-choices", classes="expanded-body")
            yield SettingsChecklist(id="settings-checklist", classes="expanded-body")
            yield SettingsGroupList(id="settings-actions")
            with Vertical(id="settings-editor"):
                yield NoMarkupStatic("", id="settings-editor-label")
                yield VscodeCompatInput(
                    id="settings-input", validate_on=["blur", "submitted"]
                )
                yield NoMarkupStatic("", id="settings-editor-error")
            with Vertical(id="settings-confirmation"):
                with ConfirmationText(id="settings-confirmation-scroll"):
                    yield NoMarkupStatic("", id="settings-confirmation-text")
                yield SettingsConfirmationList(id="settings-confirmation-actions")
            yield SettingsHelp("", id="settings-help")
            yield SettingsHints("", id="settings-hint")

    def on_mount(self) -> None:
        if getattr(self.app, "_pending_callbacks", None) or getattr(
            self.app, "_pending_local_question", None
        ):
            self.query_one("#settings-pending-action").display = True
        self.query_one("#settings-checklist", SettingsChecklist).display = False
        self._resize_surface()
        self._refresh_options()
        if self.return_state is not None:
            self.call_after_refresh(self._restore_return_state)
        else:
            self.query_one("#settings-options", SettingsOptionList).focus()

    def _restore_return_state(self) -> None:
        state = self.return_state
        if state is None:
            return
        query, selected, scroll_y, focused_id = state
        options = self.query_one("#settings-options", SettingsOptionList)
        options._query = query
        self._filter(query)
        for index in range(options.option_count):
            if options.get_option_at_index(index).id == selected:
                options.highlighted = index
                break
        target = self.query_one(f"#{focused_id}") if focused_id else options
        target.focus(scroll_visible=False)
        self.call_after_refresh(
            options.scroll_to, y=scroll_y, animate=False, force=True, immediate=True
        )

    def _restore_child_state(self) -> None:
        self._restore_return_state()
        self._restore_group_position(self._openers.pop("child", None))
        self.return_state = None
        self._child_returning = False
        self._update_help()

    def on_resize(self, event: events.Resize) -> None:
        position = self._capture_group_position()
        self._resize_surface()
        self._restore_group_position(position, restore_focus=False)

    def _resize_surface(self) -> None:
        content = self.query_one("#settings-content", Vertical)
        width, height = self.size
        fullscreen = width < MIN_MODAL_WIDTH or height < MIN_MODAL_HEIGHT
        content.set_class(fullscreen, "fullscreen")
        content.styles.width = "100%" if fullscreen else min(92, width - 2)
        content.styles.height = "100%" if fullscreen else max(1, height - 2)
        content.border_title = "" if fullscreen else "Settings"
        self._update_hint()
        self.call_after_refresh(self._mark_cursor)

    def _available_actions(self) -> list[tuple[str, str]]:  # noqa: PLR0911, PLR0912
        options = self.query_one("#settings-options", SettingsOptionList)
        item = self._current_item()
        if self._busy:
            return [("Esc", "Wait for save")]
        if self._help_open and (
            self._confirmation is None or self.query_one("#settings-help").has_focus
        ):
            return [("↑↓/PgUp/PgDn", "Scroll"), ("F1", "Hide help"), ("Esc", "Back")]
        if self._confirmation is not None:
            choices = self.query_one("#settings-confirmation-actions", OptionList)
            choice = choices.highlighted_option
            return [
                (
                    "Enter",
                    "Cancel" if choice is None or choice.id == "cancel" else "Confirm",
                ),
                ("↑↓", "Choose"),
                ("Esc", "Cancel"),
            ]
        if self._editing is not None:
            return [
                (
                    "Enter",
                    "Add to draft"
                    if self._list_editor is not None
                    else "Save to user settings",
                ),
                ("Esc", "Cancel"),
            ]
        action_list = self.query_one("#settings-actions", OptionList)
        if action_list.has_focus:
            selected_action = action_list.highlighted_option
            return [
                (
                    "Enter",
                    "Add pattern"
                    if selected_action and selected_action.id == "add-pattern"
                    else "Add item"
                    if selected_action and selected_action.id == "add-item"
                    else "Save to user settings",
                ),
                ("Esc", "Back"),
            ]
        if self.query_one("#settings-checklist", SettingsChecklist).has_focus:
            checklist = self.query_one("#settings-checklist", SettingsChecklist)
            selected = (
                checklist.get_option_at_index(checklist.highlighted).value
                if checklist.highlighted is not None
                else None
            )
            if selected is not None and selected.startswith("\x00state:"):
                return [("Esc", "Back")]
            actions = [("Esc", "Back")]
            if selected is not None and selected.startswith("\x00pattern:"):
                actions.insert(0, ("Ctrl+D", "Delete pattern"))
            elif selected is not None and not self._inventory_item_state(selected)[1]:
                actions.insert(0, ("Enter", "Toggle draft"))
                actions.insert(1, ("Space", "Toggle"))
            return actions
        if self.query_one("#settings-entries").has_focus and item is not None:
            options = self.query_one("#settings-entries", OptionList)
            actions = [("Enter", "Edit item"), ("Esc", "Back")]
            identifier = (
                str(options.highlighted_option.id) if options.highlighted_option else ""
            )
            if identifier.startswith("state:"):
                return [("Esc", "Back")]
            if identifier.rsplit(":", 1)[-1].isdigit():
                actions.insert(1, ("Ctrl+D", "Delete item"))
            return actions
        if self.query_one("#settings-choices").has_focus and item is not None:
            if not item.choices:
                return [("Esc", "Back")]
            return [
                ("Enter", "Accept selected"),
                ("Space", "Select locally"),
                ("Esc", "Back"),
            ]
        if item is None:
            return [("Esc", "Clear filter" if options._query else "Back")]
        action = (
            "Save toggle to user settings"
            if item.kind == "bool" and not self.snapshot.view_only
            else "Open"
        )
        actions = [("Enter", action), ("Esc", "Back")]
        if item.kind == "bool" and not self.snapshot.view_only:
            actions.insert(1, ("Space", "Save toggle"))
        if item is not None and not self.snapshot.view_only:
            field = self.fields.get(item.path)
            if (
                item.control == "toggle_inventory"
                or (
                    item.control == "status_line"
                    and any(
                        self.fields[path].saved_explicit
                        for path in STATUS_LINE_PATHS
                        if path in self.fields
                    )
                )
                or (field and field.saved_explicit)
            ):
                actions.insert(-1, ("Ctrl+R", "Remove override"))
        return actions

    def _update_hint(self) -> None:
        if not self.is_mounted:
            return
        actions = self._available_actions()
        group_hint = ("Tab/Shift+Tab", "Groups")
        cycling = (
            not self._busy
            and self._editing is None
            and self._confirmation is None
            and len(self._visible_focus_groups()) > 1
        )
        if cycling:
            actions.insert(-1, group_hint)
        content = self.query_one("#settings-content", Vertical)
        available = max(1, (content.region.width or min(self.size.width, 92)) - 4)

        def render(pairs: list[tuple[str, str]]) -> str:
            return "  ".join(f"{shortcut(key)} {label}" for key, label in pairs)

        visible = actions
        if (
            len(shortcut_hint(render(actions)).plain) > available
            and len(actions) > MIN_SHORTCUTS_WITH_OVERFLOW
        ):
            visible = [
                actions[0],
                group_hint if cycling else ("F1", "Help"),
                actions[-1],
            ]
        if (
            len(shortcut_hint(render(visible)).plain) > available
            and len(visible) > MIN_SHORTCUTS_WITH_OVERFLOW
        ):
            primary = visible[0]
            visible = [
                primary if primary[0] == "Enter" or not cycling else group_hint,
                visible[-1],
            ]
        text = shortcut_hint(render(visible))
        commands = {
            "Enter": "activate_focused",
            "Space": "select_draft",
            "Esc": "close",
            "F1": "help",
            "Ctrl+D": "delete_item",
            "Ctrl+R": "remove_override",
        }
        hint = self.query_one("#settings-hint", SettingsHints)
        hint.targets = []
        start = 0
        for key, label in visible:
            part = shortcut_hint(f"{shortcut(key)} {label}").plain
            hint.targets.append((start, start + cell_len(part), commands.get(key, "")))
            start += cell_len(part) + 2
        hint.update(text)

    def _close_help(self, *, restore: bool = True) -> None:
        help_widget = self.query_one("#settings-help", NoMarkupStatic)
        self._help_open = False
        help_widget.remove_class("details-open")
        help_widget.can_focus = False
        opener = self._help_return_focus
        self._help_return_focus = None
        if restore:
            self._restore_group_position(opener)
        self._update_help()
        self._update_hint()

    def action_help(self) -> None:
        if self._busy:
            return
        if self._help_open:
            self._close_help()
            return
        help_widget = self.query_one("#settings-help", NoMarkupStatic)
        self._help_return_focus = self._capture_group_position()
        help_widget.can_focus = True
        help_widget.add_class("details-open")
        self._help_open = True
        self._update_help()
        help_widget.focus()
        self._update_hint()

    def _filter(self, query: str) -> None:
        if query and "filter" not in self._openers:
            self._remember_opener("filter")
        self.query_one("#settings-filter", NoMarkupStatic).update(
            f"Filter: {query or 'type to filter'}"
        )
        self._refresh_options()
        if not query:
            self._restore_group_position(
                self._openers.pop("filter", None),
                restore_focus=self.query_one("#settings-options").has_focus,
            )

    def _display_value(self, item: SettingDescriptorWire) -> str:  # noqa: PLR0911
        if item.control == "status_line":
            field = self.fields.get("status_line.segments")
            segments = field.effective_value if field else []
            return f"{len(segments) if isinstance(segments, list) else 0} segments"
        if item.control == "toggle_inventory":
            category = item.inventory or "tools"
            names = self.snapshot.inventories.get(category, [])
            states = self.snapshot.inventory_states.get(category, {})
            return (
                f"[{sum(state.effective for state in states.values())}/{len(names)} on]"
            )
        if item.kind in {"link", "deferred"}:
            return ""
        if self._editing == item.path:
            return ""
        field = self.fields.get(item.path)
        if item.kind == "list":
            field = self.fields.get(item.path)
            value = field.effective_value if field else None
            if isinstance(value, list):
                return f"[{len(value)} items]"
        if item.kind == "bool":
            return (
                f"[{chrome_glyph('checked')}]"
                if field and field.effective_value is True
                else f"[{chrome_glyph('unchecked')}]"
            )
        return format_value(field.effective_value if field else None)

    def _row(self, item: SettingDescriptorWire, *, selected: bool = False) -> Content:
        return Content.assemble(
            (
                f"{chrome_glyph('cursor')} " if selected else "  ",
                "$primary bold"
                if self.query_one("#settings-options", SettingsOptionList).has_focus
                else "$text-muted",
            ),
            (
                f"[{chrome_glyph('checked')}] "
                if self.fields.get(item.path)
                and self.fields[item.path].effective_value is True
                else f"[{chrome_glyph('unchecked')}] ",
                "$text-muted",
            )
            if item.kind == "bool"
            else "",
            item.label
            if item.control in {"toggle_inventory", "status_line"}
            or item.kind == "link"
            else item.path,
            "    ",
            ("" if item.kind == "bool" else self._display_value(item), "$text-muted"),
        )

    def _refresh_options(self, *, preserve: bool = True) -> None:
        options = self.query_one("#settings-options", SettingsOptionList)
        position = self._capture_group_position(options) if preserve else None
        query = options._query.strip().lower()
        rows: list[Option] = []
        if query:
            scored: list[tuple[int, int, SettingDescriptorWire]] = []
            for index, item in enumerate(self.catalog):
                name = item.path.lower()
                description = item.description.lower()
                if query in name or query in item.label.lower():
                    score = 3 if name.startswith(query) else 2
                elif query in description:
                    score = 1
                else:
                    continue
                scored.append((-score, index, item))
            rows.extend(
                Option(self._row(item), id=item.path) for _, _, item in sorted(scored)
            )
        else:
            use_long_list_headings = len(self.catalog) >= LONG_LIST_HEADING_THRESHOLD
            for group in GROUPS:
                items = [item for item in self.catalog if item.group == group]
                if items:
                    rows.append(
                        Option(
                            Content.assemble((
                                f"── {group.upper()} ──",
                                "$foreground bold",
                            ))
                            if use_long_list_headings
                            else group.upper(),
                            disabled=True,
                        )
                    )
                    rows.extend(Option(self._row(item), id=item.path) for item in items)
        if not any(not row.disabled for row in rows):
            rows.append(Option(Content.assemble("  No matching settings"), id="empty"))
        options.clear_options()
        options.add_options(rows)
        options.highlighted = next(
            (i for i, row in enumerate(rows) if not row.disabled), None
        )
        self._restore_group_position(position, restore_focus=False)
        self._refresh_collection()
        self._mark_cursor()

    def _collection_visibility(self, *, hidden: bool = False) -> None:
        active = self._expanded is not None and not hidden
        inventory = self._inventory_draft is not None
        item = next(
            (item for item in self.catalog if item.path == self._expanded), None
        )
        catalog = self.query_one("#settings-options", SettingsOptionList)
        was_compact = catalog.has_class("compact-catalog")
        catalog.set_class(active, "compact-catalog")
        if active and not was_compact:

            def reveal_catalog() -> None:
                if catalog.has_class("compact-catalog"):
                    catalog.scroll_to_highlight()

            self.call_after_refresh(reveal_catalog)
        self.query_one("#settings-active-editor").display = active
        self.query_one("#settings-entries").display = (
            active and not inventory and item is not None and item.kind == "list"
        )
        self.query_one("#settings-choices").display = (
            active and item is not None and item.kind == "enum"
        )
        self.query_one("#settings-checklist").display = active and inventory
        self.query_one("#settings-actions").display = (
            active and item is not None and item.kind != "enum"
        )

    def _refresh_collection(self) -> None:
        self._collection_visibility(
            hidden=self._editing is not None or self._confirmation is not None
        )
        item = next(
            (item for item in self.catalog if item.path == self._expanded), None
        )
        if item is None:
            return
        self.query_one("#settings-active-editor", NoMarkupStatic).update(
            f"Editing {item.label} — draft"
        )
        if self._inventory_draft is not None:
            actions = [("add-pattern", "Add pattern"), ("apply", "Apply changes")]
        elif item.kind == "list":
            entries = self.query_one("#settings-entries", OptionList)
            self._replace_group(
                entries,
                [
                    Option(entry.value, id=f"list:{item.path}:{entry.token}")
                    for entry in self._list_draft or []
                ]
                or [Option("No items", id="state:entries")],
            )
            actions = [("add-item", "Add item"), ("apply", "Apply changes")]
        else:
            choices = self.query_one("#settings-choices", OptionList)
            self._replace_group(
                choices,
                [
                    Option(choice, id=f"choice:{item.path}:{choice}")
                    for choice in item.choices
                ]
                or [Option("No choices", id="state:choices")],
            )
            actions = []
        self._replace_group(
            self.query_one("#settings-actions", OptionList),
            [Option(label, id=identifier) for identifier, label in actions],
        )

    def _position_before_rebuild(self, options: OptionList) -> _GroupPosition | None:
        group = options.id or ""
        context = self._group_context(group)
        position = (
            self._capture_group_position(options)
            if self._group_contents_context.get(group) == context
            else self._positions.get((group, context))
        )
        self._group_contents_context[group] = context
        return position

    def _replace_group(self, options: OptionList, rows: list[Option]) -> None:
        position = self._position_before_rebuild(options)
        options.clear_options()
        options.add_options(rows)
        if position is None:
            options.highlighted = next(
                (i for i, row in enumerate(rows) if not row.disabled), None
            )
        self._restore_group_position(position, restore_focus=False)

    def _mark_cursor(self) -> None:
        for identifier in (
            "settings-options",
            "settings-entries",
            "settings-choices",
            "settings-actions",
        ):
            self._mark_group_cursor(self.query_one(f"#{identifier}", OptionList))
        self._update_help()
        self._update_hint()

    def _mark_group_cursor(self, options: OptionList) -> None:
        item_id = (
            str(options.highlighted_option.id)
            if options.highlighted_option and options.highlighted_option.id
            else None
        )
        paths = (
            (self._cursor, item_id)
            if options.id == "settings-options"
            else (str(option.id) for option in options.options if option.id)
        )
        for path in paths:
            if path is None:
                continue
            item = next((entry for entry in self.catalog if entry.path == path), None)
            if item is not None:
                prompt = self._row(item, selected=path == item_id)
            elif path == "empty":
                prompt = Content.assemble(
                    (
                        f"{chrome_glyph('cursor')} " if path == item_id else "  ",
                        "$primary bold" if options.has_focus else "$text-muted",
                    ),
                    "No matching settings",
                )
            elif path.startswith("choice:"):
                _, parent, choice = path.split(":", 2)
                prompt = Content.assemble(
                    (
                        f"{chrome_glyph('cursor')} " if path == item_id else "  ",
                        "$primary bold" if options.has_focus else "$text-muted",
                    ),
                    (
                        chrome_glyph("radio_selected")
                        if self._enum_draft == choice
                        else chrome_glyph("radio_empty"),
                        "$foreground",
                    ),
                    " ",
                    choice,
                )
            elif path.startswith("list:"):
                _, parent, index = path.split(":", 2)
                label = next(
                    (
                        entry.value
                        for entry in self._list_draft or []
                        if entry.token == int(index)
                    ),
                    "",
                )
                prompt = Content.assemble(
                    (
                        f"{chrome_glyph('cursor')} " if path == item_id else "  ",
                        "$primary bold" if options.has_focus else "$text-muted",
                    ),
                    "  ",
                    label,
                )
            elif path in {"add-item", "add-pattern", "apply"} or path.startswith(
                "state:"
            ):
                label = {
                    "add-item": "Add item",
                    "add-pattern": "Add pattern",
                    "apply": "Apply changes",
                    "state:entries": "No items",
                    "state:choices": "No choices",
                }.get(path, "")
                prompt = Content.assemble(
                    (
                        f"{chrome_glyph('cursor')} " if path == item_id else "  ",
                        "$primary bold" if options.has_focus else "$text-muted",
                    ),
                    label,
                )
            else:
                continue
            if path == item_id and options.has_focus:
                prompt = Content.assemble((
                    str(prompt).ljust(options.scrollable_content_region.width),
                    "bold reverse",
                ))
            try:
                options.replace_option_prompt(path, prompt)
            except OptionDoesNotExist:
                pass
        if options.id == "settings-options":
            self._cursor = item_id
        self._update_help()
        self._update_hint()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if self.is_mounted:
            if self._help_open and event.widget.id in {
                "settings-options",
                "settings-entries",
                "settings-choices",
                "settings-checklist",
                "settings-actions",
            }:
                destination = self._positions.get((
                    event.widget.id,
                    self._group_context(event.widget.id),
                ))
                self._close_help(restore=False)
                self._restore_group_position(destination)
            self.call_after_refresh(self._mark_cursor)

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        if self.is_mounted:
            self.call_after_refresh(self._mark_cursor)

    def _current_item(self) -> SettingDescriptorWire | None:
        focused = self.focused
        if self._editing is not None:
            path = self._editing
        elif self._confirmation is not None:
            path = self._confirmation[1].path
        elif focused is not None and focused.id in {
            "settings-entries",
            "settings-choices",
            "settings-checklist",
            "settings-actions",
        }:
            path = self._expanded or ""
        else:
            option = self.query_one(
                "#settings-options", SettingsOptionList
            ).highlighted_option
            path = str(option.id) if option and option.id else ""
            if (
                focused is not None
                and focused.id == "settings-help"
                and self._help_return_focus is not None
                and self._help_return_focus.group != "settings-options"
            ):
                path = self._expanded or path
        return next((item for item in self.catalog if item.path == path), None)

    def _help_with_feedback(self, detail: str) -> Content:
        item = self._current_item()
        path = item.path if item is not None else None
        feedback = self._unresolved.get(path or "", "") or (
            self._error if self._error_path is None or self._error_path == path else ""
        )
        if not feedback:
            return Content(detail)
        style = (
            "$error"
            if feedback.startswith(f"{chrome_glyph('error')} Failed:")
            else "$warning"
            if feedback.startswith("! Warning:")
            else "$success"
            if feedback.startswith(f"{chrome_glyph('success')} Saved:")
            else "$primary"
            if feedback.startswith(f"{chrome_glyph('running')} Running:")
            else "$text-muted"
        )
        return Content.assemble((feedback, style), "\n", detail)

    def _update_help(self) -> None:
        if self._confirmation is not None:
            self.query_one("#settings-help", NoMarkupStatic).update(
                self._help_with_feedback(
                    "Enter activates the highlighted confirmation action; Esc cancels."
                )
            )
            return
        item = self._current_item()
        if self._help_open:
            help_widget = self.query_one("#settings-help", NoMarkupStatic)
            self._help_open = False
            self._update_help()
            self._help_open = True
            detail = help_widget.content
            help_widget.update(
                Content.assemble(
                    detail if isinstance(detail, Content) else str(detail),
                    f"\nType to filter; {chrome_glyph('vertical')}/jk move; Tab/Shift+Tab cycle groups (except in editors or confirmations); Enter opens or accepts; Space toggles or selects; Ctrl+R removes override; Ctrl+D deletes an item; Esc goes back. List and inventory drafts require Apply changes.",
                )
            )
            return
        if item is None:
            self.query_one("#settings-help", NoMarkupStatic).update(
                self._help_with_feedback(
                    "No matching settings. Esc clears the filter."
                    if self.query_one("#settings-options", SettingsOptionList)._query
                    else ""
                )
            )
            return
        if item.control == "status_line":
            count = sum(
                self.fields[path].saved_explicit
                for path in STATUS_LINE_PATHS
                if path in self.fields
            )
            self.query_one("#settings-help", NoMarkupStatic).update(
                self._help_with_feedback(
                    item.description
                    + "\nOpen to edit a draft; Apply changes saves to user settings."
                    + f"\nSaved user overrides: {count}/4 · Ctrl+R restores inheritance."
                )
            )
            return
        if item.control == "toggle_inventory":
            draft = self._inventory_draft
            detail = ""
            checklist = self.query_one("#settings-checklist", SettingsChecklist)
            if (
                draft is not None
                and checklist.display
                and checklist.highlighted is not None
            ):
                value = checklist.get_option_at_index(checklist.highlighted).value
                if value.startswith(("\x00pattern:", "\x00state:")):
                    detail = "\nPattern entries are read-only here; Ctrl+D removes the highlighted entry."
                elif self._pattern_driven(item.inventory or "tools", value):
                    detail = "\nThis state comes from a pattern entry below. Remove that entry with Ctrl+D to toggle this item."
            help_widget = self.query_one("#settings-help", NoMarkupStatic)
            help_widget.set_class(self._inventory_draft is not None, "inventory-help")
            help_widget.update(
                self._help_with_feedback(
                    "Ctrl+R resets both lists.\n"
                    + item.description
                    + "\nEmptying the allow-only list returns to default mode (except disabled entries)."
                    + "\nPattern-controlled items cannot toggle; Ctrl+D removes the pattern entry."
                    + "\nChanges here are a draft; Apply changes saves to user settings."
                    + detail
                )
            )
            return
        self.query_one("#settings-help", NoMarkupStatic).remove_class("inventory-help")
        field = self.fields.get(item.path)
        saved = (
            format_value(field.saved_value)
            if field and field.saved_explicit
            else "Not Set"
        )
        effective = format_value(field.effective_value) if field else "—"
        origin = field.origin if field else "unknown"
        timing = (
            f"\nApplies: {item.timing.replace('-', ' ')}"
            + (" (expected)" if not item.timing_verified else "")
            if item.kind not in {"link", "deferred"}
            else ""
        )
        if item.kind == "bool":
            save_scope = (
                "\nEnter or Space saves the toggle to user settings immediately."
            )
        elif item.kind == "enum":
            save_scope = (
                "\nSpace selects locally; Enter saves the selected choice to user settings."
                if self._expanded == item.path
                else "\nOpen to choose; Enter in the choice list saves to user settings."
            )
        elif item.kind == "list":
            save_scope = (
                "\nItem edits stay in a draft; Apply changes saves to user settings."
            )
        elif item.kind not in {"link", "deferred"}:
            save_scope = "\nSubmitting the editor saves to user settings immediately."
        else:
            save_scope = ""
        self.query_one("#settings-help", NoMarkupStatic).update(
            self._help_with_feedback(
                item.description
                + (f"\n{item.empty}" if item.kind == "list" else "")
                + save_scope
                + f"\nSaved user value: {saved}  Effective: {effective} ({origin})"
                + timing
            )
        )

    def on_selection_list_selection_highlighted(
        self, event: SelectionList.SelectionHighlighted
    ) -> None:
        self._update_help()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_list.id == "settings-confirmation-actions":
            self._update_hint()
            return
        if (
            self._error.startswith(f"{chrome_glyph('success')} Saved:")
            and not self._busy
            and not self._child_returning
            and (event.option.id != "status_line" or self._error_path != "status_line")
        ):
            self._error = ""
        self._mark_cursor()

    def _begin_pointer_event(self, event: events.Click) -> bool:
        if event.chain == 1:
            self._pointer_command_started = False
        return not self._pointer_command_started

    def _can_mutate(self, *, editor: bool = False) -> bool:
        return not (
            self._busy
            or self._help_open
            or self._confirmation is not None
            or (self._editing is not None and not editor)
            or self.snapshot.view_only
            or self._needs_refresh
        )

    def action_activate_focused(self) -> None:
        if self._busy or self._help_open:
            return
        if self._editing is not None:
            editor = self.query_one("#settings-input", Input)
            self.on_input_submitted(Input.Submitted(editor, editor.value))
        elif isinstance(self.focused, OptionList):
            self.focused.action_select()

    def _select_enum_draft(self) -> None:
        if not self._can_mutate():
            return
        choices = self.query_one("#settings-choices", OptionList)
        option = choices.highlighted_option
        if choices.has_focus and option and str(option.id).startswith("choice:"):
            self._enum_draft = str(option.id).split(":", 2)[2]
            self._refresh_options()

    def action_select_draft(self) -> None:
        if not self._can_mutate():
            return
        if self.query_one("#settings-choices").has_focus:
            self._select_enum_draft()
        elif isinstance(self.focused, SettingsChecklist):
            self.focused.action_select()
        elif (
            self.query_one("#settings-options").has_focus
            and (item := self._current_item()) is not None
            and item.kind == "bool"
        ):
            self.action_activate_focused()

    def _apply_collection(self, item: SettingDescriptorWire) -> None:
        if not self._can_mutate() or self._expanded != item.path:
            return
        self._capture_group_position()
        if self._inventory_draft is not None:
            self._save_inventory(item)
        else:
            self._save_list(item)

    def _accept_enum_draft(self, item: SettingDescriptorWire) -> None:
        if self._enum_draft not in item.choices:
            return
        if self._draft_is_dirty(item):
            self._start_write(item, {item.path: self._enum_draft})
        else:
            self._collapse(restore_opener=False)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:  # noqa: PLR0911
        if event.option_list.id == "settings-confirmation-actions":
            self._dismiss_confirmation(apply=event.option.id == "apply")
            return
        if (
            self._busy
            or self._help_open
            or self._editing is not None
            or self._confirmation is not None
            or event.option.id is None
        ):
            return
        if event.option_list.id != "settings-options" and not self._can_mutate():
            return
        option_id = str(event.option.id)
        if option_id == "empty" or option_id.startswith("state:"):
            return
        if event.option_list.id == "settings-actions":
            item = next(
                (entry for entry in self.catalog if entry.path == self._expanded), None
            )
            if item is not None:
                if option_id in {"add-item", "add-pattern"}:
                    self._begin_list_input(item, "add")
                elif option_id == "apply":
                    self._apply_collection(item)
            return
        if option_id.startswith("list:"):
            _, path, index = option_id.split(":", 2)
            item = next((entry for entry in self.catalog if entry.path == path), None)
            if item is not None and self._expanded == path:
                self._begin_list_input(item, index)
            return
        if option_id.startswith("choice:"):
            _, path, _ = option_id.split(":", 2)
            item = next((entry for entry in self.catalog if entry.path == path), None)
            if item is not None:
                self._accept_enum_draft(item)
            return
        item = next((item for item in self.catalog if item.path == option_id), None)
        if item is None:
            return
        if self._expanded is not None:
            expanded = next(
                entry for entry in self.catalog if entry.path == self._expanded
            )
            if self._draft_is_dirty(expanded):
                self._pending_navigation = item
                self._open_confirmation(
                    "discard",
                    expanded,
                    None,
                    f"Discard edits to {expanded.label}? Cancel preserves the current draft; Discard continues to {item.label}.",
                    "Discard edits",
                )
                return
            self._collapse()
        self._open_setting(item)

    def _draft_is_dirty(self, item: SettingDescriptorWire) -> bool:
        field = self.fields.get(item.path)
        if (
            item.kind == "list"
            and self._list_draft is not None
            and self._inventory_draft is None
        ):
            return _draft_values(self._list_draft) != (
                field.effective_value if field else []
            )
        if item.kind == "enum" and self._enum_draft is not None:
            return self._enum_draft != (field.effective_value if field else None)
        if self._inventory_draft is not None:
            enabled, disabled = self._inventory_values(item.inventory or "tools")
            return (
                _draft_values(self._inventory_draft["enabled"]) != enabled
                or _draft_values(self._inventory_draft["disabled"]) != disabled
            )
        return False

    def _open_setting(self, item: SettingDescriptorWire) -> None:
        if item.control == "status_line":
            self._remember_opener("child")
            options = self.query_one("#settings-options", SettingsOptionList)
            state = (
                options._query,
                str(options.highlighted_option.id)
                if options.highlighted_option
                else None,
                int(options.scroll_y),
                self.focused.id if self.focused is not None else None,
            )

            def returned(result: StatusLineSettingsResult | None) -> None:
                if result is not None:
                    if result.snapshot is not None:
                        self.snapshot = result.snapshot.model_copy(
                            update={"user_revision": result.revision}
                        )
                        self.catalog = self.snapshot.catalog
                        self.fields = {
                            field.path: field for field in self.snapshot.fields
                        }
                    else:
                        self.snapshot = self.snapshot.model_copy(
                            update={"user_revision": None}
                        )
                    self._needs_refresh = self._needs_refresh or result.needs_refresh
                    # Inspection-only returns do not reconcile a prior save warning.
                    if result.feedback.startswith((
                        f"{chrome_glyph('success')} Saved:",
                        "! Warning:",
                        "Failed:",
                    )):
                        self._error_path = item.path
                        self._error = result.feedback
                        self._unresolved.pop(item.path, None)
                        if (
                            result.feedback.startswith("! Warning:")
                            or result.needs_refresh
                        ):
                            self._unresolved[item.path] = result.feedback
                self.return_state = state
                self._child_returning = True
                self._refresh_options()
                self.call_after_refresh(self._restore_child_state)

            self.app.push_screen(
                StatusLineSettingsScreen(
                    self.service, self.snapshot, needs_refresh=self._needs_refresh
                ),
                returned,
            )
        elif item.kind == "link":
            options = self.query_one("#settings-options", SettingsOptionList)
            self.return_state = (
                options._query,
                str(options.highlighted_option.id)
                if options.highlighted_option
                else None,
                int(options.scroll_y),
                self.focused.id if self.focused is not None else None,
            )
            self.dismiss(item.command)
        elif item.kind == "deferred":
            self._error_path = item.path
            self._error = "Info: Open config file to edit this setting."
            self._update_help()
        elif self.snapshot.view_only:
            self._error_path = item.path
            self._error = f"{chrome_glyph('error')} User configuration is unavailable (view only)."
            self._update_help()
        elif self._needs_refresh:
            self._error_path = item.path
            self._error = "Failed: Could not save settings; close and reopen Settings to reconcile external changes."
            self._unresolved[item.path] = self._error
            self._update_help()
        elif item.kind == "bool":
            self._toggle_bool(item)
        elif item.kind in {"list", "enum"} or item.control == "toggle_inventory":
            self._expand(item)
        else:
            self._begin_input(item)

    def _toggle_bool(self, item: SettingDescriptorWire) -> None:
        if not self._can_mutate():
            return
        current = self.fields.get(item.path)
        value = not (current.effective_value is True if current else False)
        if item.risk == "needs-confirmation" and value:
            self._open_confirmation(
                "trust",
                item,
                True,
                "Trust OS certificate authorities for API connections? Cancel preserves the current trust setting.",
                "Trust authorities",
            )
        else:
            self._start_write(item, {item.path: value})

    def _expand(self, item: SettingDescriptorWire) -> None:
        self._remember_opener("expansion")
        self._expanded = item.path
        field = self.fields.get(item.path)
        value = field.effective_value if field else None
        if item.kind == "enum":
            self._enum_draft = str(value) if value is not None else None
        if item.kind == "list":
            self._list_draft = self._make_draft(
                [entry for entry in value if isinstance(entry, str)]
                if isinstance(value, list)
                else []
            )
        if item.control == "toggle_inventory":
            self._open_checklist(item)
            return
        options = self.query_one("#settings-options", SettingsOptionList)
        options.editing = True
        body_id = "settings-entries" if item.kind == "list" else "settings-choices"
        remembered = self._positions.get((body_id, item.path))
        self._refresh_options()
        body = self.query_one(f"#{body_id}", OptionList)
        ids = [row.id for row in body.options]
        target = f"choice:{item.path}:{value}"
        if remembered is None and target in ids:
            body.highlighted = ids.index(target)
        body.focus(scroll_visible=remembered is None)
        if remembered is None:
            self.call_after_refresh(body.scroll_to_highlight)

    def _save_list(self, item: SettingDescriptorWire) -> None:
        if not self._can_mutate():
            return
        values = _draft_values(self._list_draft)
        saved = self.fields.get(item.path)
        original = saved.effective_value if saved is not None else []
        if values != original:
            self._start_write(item, {item.path: values})
        else:
            self._collapse(restore_opener=False)

    def _inventory_values(self, category: str) -> tuple[list[str], list[str]]:
        def values(path: str) -> list[str]:
            field = self.fields.get(path)
            value = field.saved_value if field and field.saved_explicit else None
            return (
                [entry for entry in value if isinstance(entry, str)]
                if isinstance(value, list)
                else []
            )

        return values(f"enabled_{category}"), values(f"disabled_{category}")

    def _inventory_item_state(self, name: str) -> tuple[bool, bool]:
        draft = self._inventory_draft
        if draft is None:
            return False, False
        item = next(item for item in self.catalog if item.path == self._expanded)
        return inventory_item_state(
            name,
            _draft_values(draft["enabled"]),
            _draft_values(draft["disabled"]),
            category=item.inventory or "tools",
        )

    def _pattern_driven(
        self, category: Literal["tools", "skills", "agents"], name: str
    ) -> bool:
        return self._inventory_item_state(name)[1]

    def _effective_inventory_item(
        self, category: Literal["tools", "skills", "agents"], name: str
    ) -> bool:
        return self._inventory_item_state(name)[0]

    def _render_checklist(self, *, highlight: str | None = None) -> None:
        item = next(entry for entry in self.catalog if entry.path == self._expanded)
        category = item.inventory or "tools"
        draft = self._inventory_draft
        if draft is None:
            return
        enabled, disabled = draft["enabled"], draft["disabled"]
        names = self.snapshot.inventories.get(category, [])
        checklist = self.query_one("#settings-checklist", SettingsChecklist)
        position = self._position_before_rebuild(checklist)
        checklist.highlighted = None
        checklist.clear_options()
        checklist.add_options(
            [
                Selection(
                    f"  {name}"
                    + (
                        " (pattern-controlled)"
                        if self._pattern_driven(category, name)
                        else ""
                    ),
                    name,
                    self._effective_inventory_item(category, name),
                    id=f"inventory:{category}:{name}",
                )
                for name in names
            ]
            + [
                Selection(
                    f"  {entry.value} ({side} pattern)",
                    f"\x00pattern:{side}:{entry.token}",
                    False,
                    id=f"pattern:{item.path}:{side}:{entry.token}",
                )
                for side, values in (("enabled", enabled), ("disabled", disabled))
                for entry in values
                if entry.value.lower() not in {name.lower() for name in names}
            ]
            + (
                []
                if names
                or any(
                    entry.value.lower() not in {name.lower() for name in names}
                    for entry in enabled + disabled
                )
                else [
                    Selection(
                        "  No members", "\x00state:members", False, id="state:members"
                    )
                ]
            )
        )
        if position is None:
            checklist.highlighted = 0 if checklist.option_count else None
        self._restore_group_position(position, restore_focus=False)
        if highlight is not None:
            checklist.highlighted = next(
                (
                    index
                    for index in range(checklist.option_count)
                    if checklist.get_option_at_index(index).value == highlight
                ),
                checklist.highlighted,
            )
            self._capture_group_position(checklist)
        self._update_help()

    def _toggle_inventory(self, name: str) -> bool:
        if not self._can_mutate():
            return False
        draft = self._inventory_draft
        if draft is None:
            return False
        item = next(item for item in self.catalog if item.path == self._expanded)
        category = item.inventory or "tools"
        if self._pattern_driven(category, name):
            return False
        is_enabled = self._effective_inventory_item(category, name)
        enabled, disabled = toggle_inventory_name(
            name,
            _draft_values(draft["enabled"]),
            _draft_values(draft["disabled"]),
            is_enabled=is_enabled,
            category=category,
        )
        draft["enabled"] = self._reconcile_draft(draft["enabled"], enabled)
        draft["disabled"] = self._reconcile_draft(draft["disabled"], disabled)
        return True

    def _open_checklist(self, item: SettingDescriptorWire) -> None:
        checklist = self.query_one("#settings-checklist", SettingsChecklist)
        enabled, disabled = self._inventory_values(item.inventory or "tools")
        self._inventory_draft = {
            "enabled": self._make_draft(enabled),
            "disabled": self._make_draft(disabled),
        }
        self._render_checklist()
        options = self.query_one("#settings-options", SettingsOptionList)
        options.editing = True
        self._refresh_collection()
        checklist.focus()
        self._update_hint()
        self._update_help()

    def _save_inventory(self, item: SettingDescriptorWire) -> None:
        if not self._can_mutate():
            return
        draft = self._inventory_draft
        if draft is None:
            return
        category = item.inventory or "tools"
        saved_enabled, saved_disabled = self._inventory_values(category)
        changes = {
            f"{side}_{category}": _draft_values(draft[side])
            for side, saved in (
                ("enabled", saved_enabled),
                ("disabled", saved_disabled),
            )
            if _draft_values(draft[side]) != saved
        }
        if changes:
            self._start_write(item, changes)
        else:
            self._collapse(restore_opener=False)

    def _collapse(self, *, restore_opener: bool = True) -> None:
        for group in self._visible_focus_groups():
            self._capture_group_position(group)
        path = self._expanded
        catalog_position = self._positions.get(("settings-options", "catalog"))
        opener = self._openers.pop("expansion", None)
        self._expanded = None
        self._list_draft = None
        self._enum_draft = None
        self._inventory_draft = None
        self._list_editor = None
        self._collection_visibility()
        options = self.query_one("#settings-options", SettingsOptionList)
        options.focus()
        options.editing = False
        if path is not None:
            self._refresh_options()
        self._restore_group_position(opener if restore_opener else catalog_position)

    def _begin_list_input(self, item: SettingDescriptorWire, index: str) -> None:
        entries = self._list_draft or []
        if index == "add":
            self._list_editor = _ListEditorContext(
                "add-pattern" if item.control == "toggle_inventory" else "add-item"
            )
            value, label = "", "Add item"
        else:
            token = int(index)
            position = next(
                (i for i, entry in enumerate(entries) if entry.token == token), None
            )
            if position is None:
                return
            self._list_editor = _ListEditorContext("edit", token)
            value, label = entries[position].value, f"Edit item {position + 1}"
        self._show_editor(item, value, label)

    def _begin_input(self, item: SettingDescriptorWire) -> None:
        field = self.fields.get(item.path)
        value = (
            ""
            if field is None or field.effective_value is None
            else str(field.effective_value)
        )
        self._show_editor(item, value, item.label)

    def _show_editor(self, item: SettingDescriptorWire, value: str, label: str) -> None:
        self._remember_opener("editor")
        self._editing = item.path
        self.query_one("#settings-options", SettingsOptionList).editing = True
        self.query_one("#settings-options", SettingsOptionList).display = False
        self._collection_visibility(hidden=True)
        self.query_one("#settings-editor-label", NoMarkupStatic).update(
            f"{label} — Enter adds to draft; Apply changes saves; Esc cancels"
            if self._list_editor is not None
            else f"{label} — Enter saves to user settings immediately; Esc cancels"
        )
        self.query_one("#settings-editor-error", NoMarkupStatic).update("")
        editor = self.query_one("#settings-input", Input)
        editor.value = value
        editor.remove_class("-invalid")
        self.query_one("#settings-editor").display = True
        editor.focus()
        self._update_hint()

    def _cancel_input(self) -> None:
        list_input = self._list_editor is not None
        self._editing = None
        self.query_one("#settings-editor").display = False
        self.query_one("#settings-editor-error", NoMarkupStatic).update("")
        self.query_one("#settings-input", Input).remove_class("-invalid")
        self._collection_visibility()
        checklist = self.query_one("#settings-checklist", SettingsChecklist)
        if list_input and self._inventory_draft is not None:
            self.query_one("#settings-options", SettingsOptionList).display = True
            checklist.focus()
        else:
            options = self.query_one("#settings-options", SettingsOptionList)
            options.display = True
            options.editing = self._expanded is not None
            if list_input:
                self.query_one("#settings-entries").focus()
            else:
                options.focus()
        self._list_editor = None
        self._restore_group_position(self._openers.pop("editor", None))
        self._update_hint()

    def on_input_submitted(self, event: Input.Submitted) -> None:  # noqa: PLR0911
        if self._editing is None or not self._can_mutate(editor=True):
            return
        item = next(entry for entry in self.catalog if entry.path == self._editing)
        if self._list_editor is not None:
            if self._list_editor.mode == "add-pattern":
                text = event.value.strip()
                draft = self._inventory_draft
                if draft is None or not text:
                    self._cancel_input()
                    return
                side = "enabled" if draft["enabled"] else "disabled"
                if text not in _draft_values(draft[side]):
                    draft[side].append(self._new_draft_entry(text))
                entry = next(entry for entry in draft[side] if entry.value == text)
                self._cancel_input()
                self._render_checklist(highlight=f"\x00pattern:{side}:{entry.token}")
                checklist = self.query_one("#settings-checklist", SettingsChecklist)

                def reveal_added_pattern() -> None:
                    checklist.scroll_to_highlight()
                    self._capture_group_position(checklist)

                checklist.call_after_refresh(reveal_added_pattern)
                return
            values = list(self._list_draft or [])
            text = event.value.strip()
            if not text:
                event.input.add_class("-invalid")
                self.query_one("#settings-editor-error", NoMarkupStatic).update(
                    "Error: Enter a nonempty item; use Ctrl+D to delete an existing item."
                )
                return
            token = self._list_editor.token
            if self._list_editor.mode == "edit":
                position = next(
                    (i for i, entry in enumerate(values) if entry.token == token), None
                )
                if position is None:
                    self._cancel_input()
                    return
                entry = _DraftEntry(values[position].token, text)
                values[position] = entry
            else:
                entry = self._new_draft_entry(text)
                values.append(entry)
            self._cancel_input()
            self._list_draft = values
            self._refresh_options()
            options = self.query_one("#settings-entries", OptionList)
            ids = [row.id for row in options.options]
            target = f"list:{item.path}:{entry.token}"
            if target in ids:
                options.highlighted = ids.index(target)
            return
        try:
            value = parse_setting_value(item, event.value)
        except ValueError as exc:
            event.input.add_class("-invalid")
            self.query_one("#settings-editor-error", NoMarkupStatic).update(
                f"Error: {exc}"
            )
            return
        self._cancel_input()
        self._start_write(item, {item.path: value})

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._editing is None or not event.input.has_class("-invalid"):
            return
        item = next(entry for entry in self.catalog if entry.path == self._editing)
        if self._list_editor is not None and item.kind == "list":
            if not event.value.strip():
                return
            event.input.remove_class("-invalid")
            self.query_one("#settings-editor-error", NoMarkupStatic).update("")
            return
        try:
            parse_setting_value(item, event.value)
        except ValueError:
            return
        event.input.remove_class("-invalid")
        self.query_one("#settings-editor-error", NoMarkupStatic).update("")

    def _open_confirmation(
        self,
        kind: str,
        item: SettingDescriptorWire,
        value: Any,
        message: str,
        action: str,
    ) -> None:
        self._confirmation_return_focus = self._capture_group_position()
        self._confirmation = (kind, item, value)
        self.query_one("#settings-options", SettingsOptionList).display = False
        self._collection_visibility(hidden=True)
        self.query_one("#settings-confirmation-text", NoMarkupStatic).update(message)
        choices = self.query_one("#settings-confirmation-actions", OptionList)
        choices.clear_options()
        choices.add_options([
            Option(Content.assemble(("[Cancel]", "$foreground")), id="cancel"),
            Option(
                Content.assemble((
                    f"[{action}]",
                    "$error"
                    if kind in {"delete", "pattern", "reset", "discard", "override"}
                    else "$foreground",
                )),
                id="apply",
            ),
        ])
        self.query_one("#settings-confirmation").display = True
        choices.highlighted = 0
        choices.focus()
        self._update_help()
        self._update_hint()

    def _dismiss_confirmation(self, *, apply: bool = False) -> None:
        if self._busy:
            return
        if (
            apply
            and (self.snapshot.view_only or self._needs_refresh)
            and self._confirmation is not None
            and self._confirmation[0] != "discard"
        ):
            return
        if apply and self._help_open:
            self._close_help()
        pending = self._confirmation
        self._confirmation = None
        self.query_one("#settings-confirmation").display = False
        options = self.query_one("#settings-options", SettingsOptionList)
        options.display = self._editing is None
        self._collection_visibility(hidden=self._editing is not None)
        self._restore_group_position(self._confirmation_return_focus)
        self._confirmation_return_focus = None
        self._update_help()
        self._update_hint()
        if not apply or pending is None:
            self._pending_navigation = None
            return
        kind, item, value = pending
        if kind == "pattern":
            side, token = value
            if self._inventory_draft is not None:
                self._inventory_draft[side] = [
                    entry
                    for entry in self._inventory_draft[side]
                    if entry.token != token
                ]
                self._render_checklist()
        elif kind == "delete":
            entries = self._list_draft or []
            self._list_draft = [entry for entry in entries if entry.token != value]
            self._refresh_options()
        elif kind == "discard":
            self._collapse()
            target = self._pending_navigation
            self._pending_navigation = None
            if target is not None:
                options = self.query_one("#settings-options", SettingsOptionList)
                ids = [row.id for row in options.options]
                if target.path in ids:
                    options.highlighted = ids.index(target.path)
                self._open_setting(target)
        elif kind == "status-reset":
            self._start_write(item, value)
        elif kind == "reset":
            category = item.inventory or "tools"
            self._start_write(
                item, {f"enabled_{category}": None, f"disabled_{category}": None}
            )
        else:
            self._start_write(item, {item.path: value})

    def on_key(self, event: events.Key) -> None:
        if self._confirmation is not None:
            return
        if event.key == "space" and self._editing is None:
            self.action_select_draft()
            event.stop()
            event.prevent_default()

    def action_delete_item(self) -> None:  # noqa: PLR0911
        if not self._can_mutate() or self._expanded is None:
            return
        checklist = self.query_one("#settings-checklist", SettingsChecklist)
        if self._inventory_draft is not None and checklist.has_focus:
            value = checklist.get_option_at_index(checklist.highlighted or 0).value
            if value.startswith("\x00pattern:"):
                _, side, index = value.rsplit(":", 2)
                item = next(
                    entry for entry in self.catalog if entry.path == self._expanded
                )
                target = next(
                    (
                        entry.value
                        for entry in self._inventory_draft[side]
                        if entry.token == int(index)
                    ),
                    None,
                )
                if target is None:
                    return
                self._open_confirmation(
                    "pattern",
                    item,
                    (side, int(index)),
                    f"Delete pattern {target} from {side}_{item.inventory}? Cancel preserves the pattern and current draft.",
                    "Delete pattern",
                )
            return
        options = self.query_one("#settings-entries", OptionList)
        if not options.has_focus:
            return
        option = options.highlighted_option
        identifier = str(option.id) if option and option.id else ""
        if not identifier.startswith(f"list:{self._expanded}:"):
            return
        index = identifier.rsplit(":", 1)[-1]
        if not index.isdigit():
            return
        item = next(entry for entry in self.catalog if entry.path == self._expanded)
        target = next(
            (
                entry.value
                for entry in self._list_draft or []
                if entry.token == int(index)
            ),
            None,
        )
        if target is None:
            return
        self._open_confirmation(
            "delete",
            item,
            int(index),
            f"Delete {target} from the {item.label} draft? Apply changes will save the shortened list. Cancel preserves the item and stored list.",
            "Delete item",
        )

    def action_remove_override(self) -> None:  # noqa: PLR0911
        if self._help_open and not self._busy and self._confirmation is None:
            self._close_help()
        if (
            self._busy
            or self._help_open
            or self._needs_refresh
            or self._confirmation is not None
        ):
            return
        item = self._current_item()
        if item is not None and item.control == "status_line":
            self._error_path = item.path
            if self.snapshot.view_only:
                self._error = f"{chrome_glyph('error')} User configuration is unavailable (view only)."
                self._update_help()
                return
            changes = {
                path: None
                for path in STATUS_LINE_PATHS
                if path in self.fields and self.fields[path].saved_explicit
            }
            if not changes:
                self._error = "Info: Status line has no user override."
                self._update_help()
                return
            self._open_confirmation(
                "status-reset",
                item,
                changes,
                "Remove saved status line overrides? Restores inheritance, not necessarily defaults. Cancel preserves saved values.",
                "Remove overrides",
            )
            return
        if item is not None and item.control == "toggle_inventory":
            if self.snapshot.view_only:
                return
            opener = (
                self._openers.get("editor")
                if self._editing is not None
                else self._capture_group_position()
            )
            if self._editing is not None:
                self._cancel_input()
            self._open_confirmation(
                "reset",
                item,
                None,
                f"Reset {item.label}? Clears enabled and disabled {item.inventory} lists. Cancel preserves both saved lists and the current draft.",
                "Reset lists",
            )
            self._confirmation_return_focus = opener
            return
        if item is None or item.kind in {"link", "deferred"}:
            return
        if self.snapshot.view_only:
            self._error_path = item.path
            self._error = f"{chrome_glyph('error')} User configuration is unavailable (view only)."
            self._update_help()
        elif not (field := self.fields.get(item.path)) or not field.saved_explicit:
            self._error_path = item.path
            self._error = f"Info: {item.label} has no user override."
            self._update_help()
        else:
            if self._editing is not None:
                self._cancel_input()
            self._open_confirmation(
                "override",
                item,
                None,
                f"Remove saved override for {item.label}? The effective value will fall back to its other sources. Cancel preserves the saved override.",
                "Remove override",
            )

    def _start_write(
        self,
        item: SettingDescriptorWire,
        changes: Mapping[str, JsonValue | list[str] | None],
    ) -> None:
        if not self._can_mutate():
            return
        # Reserve synchronously: queued activations cannot race worker startup.
        self._busy = True
        self._error_path = item.path
        self._error = f"{chrome_glyph('running')} Running: Saving {item.label}"
        self._update_help()
        self._update_hint()
        self.run_worker(
            self._write_changes(item, changes, reserved=True), group="settings-write"
        )

    async def _write(
        self, item: SettingDescriptorWire, value: JsonValue | list[str] | None
    ) -> None:
        await self._write_changes(item, {item.path: value})

    async def _write_changes(
        self,
        item: SettingDescriptorWire,
        changes: Mapping[str, JsonValue | list[str] | None],
        *,
        reserved: bool = False,
    ) -> None:
        if self._busy and not reserved:
            return
        self._busy = True
        self._error_path = item.path
        self._error = f"{chrome_glyph('running')} Running: Saving {item.label}"
        self._update_help()
        self._update_hint()
        try:
            if self._needs_refresh:
                self._error_path = item.path
                self._error = f"{chrome_glyph('error')} Failed: Could not save settings; close and reopen Settings to reconcile external changes."
                self._unresolved[item.path] = self._error
                self._update_help()
                return
            outcome = await self.service.save(
                {
                    path: list[JsonValue](value) if isinstance(value, list) else value
                    for path, value in changes.items()
                },
                self.snapshot.user_revision,
            )
            if outcome.persistence == "not_saved":
                if outcome.error == "conflict":
                    self._needs_refresh = True
                self._error_path = item.path
                self._error = f"{chrome_glyph('error')} Failed: Could not save {item.label}: {outcome.error or 'write failed'}."
                self._unresolved[item.path] = self._error
                self._update_help()
                return
            # End drafts only after persistence, including saved-but-unknown.
            if self._expanded == item.path:
                self._collapse(restore_opener=False)
            if outcome.snapshot is None:
                self._needs_refresh = True
                self._error = "! Warning: Settings saved; current state unknown. Reopen Settings before editing."
                self._unresolved[item.path] = self._error
                self._update_help()
                return
            self.snapshot = outcome.snapshot
            self.catalog = self.snapshot.catalog
            self.fields = {field.path: field for field in self.snapshot.fields}
            self._unresolved.pop(item.path, None)
            self._error = f"{chrome_glyph('success')} Saved: {item.label} updated"
            if outcome.shadowed:
                self._error = (
                    "! Warning: Saved but shadowed by a higher layer: "
                    + ", ".join(outcome.shadowed)
                )
            if (
                outcome.persistence == "durability_uncertain"
                or outcome.application == "failed"
            ):
                self._error = "! Warning: Settings saved, but durability or application is uncertain."
            if self._error.startswith("! Warning:"):
                self._unresolved[item.path] = self._error
            self._refresh_options()
        except Exception as exc:
            self._error_path = item.path
            self._error = (
                f"{chrome_glyph('error')} Failed: Could not save {item.label}: {exc}"
            )
            self._unresolved[item.path] = self._error
            self._update_help()
        finally:
            self._busy = False
            if (
                self._expanded is None
                and self._editing is None
                and self._confirmation is None
            ):
                self.query_one("#settings-options", SettingsOptionList).focus()
            self._update_hint()

    def action_close(self) -> None:
        if self._busy:
            self._update_hint()
            return
        if self._help_open and self.query_one("#settings-help").has_focus:
            self.action_help()
        elif self._confirmation is not None:
            self._dismiss_confirmation()
        elif self._help_open:
            self.action_help()
        elif self._editing is not None:
            self._cancel_input()
        elif self._expanded is not None:
            item = next(entry for entry in self.catalog if entry.path == self._expanded)
            if self._inventory_draft is not None:
                dirty = self._draft_is_dirty(item)
                if dirty:
                    self._pending_navigation = None
                    self._open_confirmation(
                        "discard",
                        item,
                        None,
                        f"Discard edits to enabled and disabled {item.inventory} lists? Unsaved inventory changes will be lost. Cancel preserves both saved lists and the current draft.",
                        "Discard edits",
                    )
                    return
            elif self._list_draft is not None:
                field = self.fields.get(item.path)
                if _draft_values(self._list_draft) != (
                    field.effective_value if field else []
                ):
                    self._pending_navigation = None
                    self._open_confirmation(
                        "discard",
                        item,
                        None,
                        f"Discard edits to {item.label}? Unsaved list changes will be lost. Cancel preserves the saved list and the current draft.",
                        "Discard edits",
                    )
                    return
            elif item.kind == "enum" and self._draft_is_dirty(item):
                self._pending_navigation = None
                self._open_confirmation(
                    "discard",
                    item,
                    None,
                    f"Discard edits to {item.label}? Cancel preserves the current draft.",
                    "Discard edits",
                )
                return
            self._collapse()
        elif (
            options := self.query_one("#settings-options", SettingsOptionList)
        )._query:
            options._query = ""
            self._filter("")
        else:
            self.dismiss(None)
