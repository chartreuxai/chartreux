"""Searchable, flat settings browser with inline single-leaf editing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fnmatch import fnmatch
import re
from typing import Any, ClassVar, Literal

from pydantic import JsonValue
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Input, OptionList, SelectionList
from textual.widgets.option_list import Option, OptionDoesNotExist

from chartreux.app_server.protocol import SettingDescriptorWire, SettingsReadResponse
from chartreux.cli.textual_ui.settings_service import SettingsService
from chartreux.cli.textual_ui.widgets.vscode_compat import VscodeCompatInput
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.checklist import Checklist
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

DOUBLE_CLICK = 2
MIN_MODAL_WIDTH = 84
MIN_MODAL_HEIGHT = 28
ADD_PATTERN = "\x00add-pattern"
MIN_SHORTCUTS_WITH_OVERFLOW = 2


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


class SettingsChecklist(Checklist):
    """Multi-select editor with Enter committing rather than toggling."""

    BINDINGS: ClassVar[list[BindingType]] = [
        *SelectionList.BINDINGS,
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("enter", "commit", "Save", show=False, priority=True),
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
        if option.value == ADD_PATTERN or option.value.startswith("\x00pattern:"):
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

    def on_selection_list_selection_highlighted(
        self, event: SelectionList.SelectionHighlighted
    ) -> None:
        if isinstance(self.screen, SettingsScreen):
            self.screen._update_help()
            self.screen._update_hint()

    def action_commit(self) -> None:
        screen = self.screen
        if isinstance(screen, SettingsScreen):
            screen._commit_checklist()

    def action_select(self) -> None:
        if self.highlighted is not None:
            value = self.get_option_at_index(self.highlighted).value
            if value == ADD_PATTERN:
                self.action_commit()
                return
            if value.startswith("\x00pattern:"):
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
            if value == ADD_PATTERN or value.startswith("\x00pattern:"):
                if value == ADD_PATTERN:
                    self.action_commit()
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


class SettingsOptionList(NavigableOptionList):
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
        index = event.style.meta.get("option")
        if index is not None and not self.get_option_at_index(index).disabled:
            self.highlighted = index
            if event.chain >= DOUBLE_CLICK:
                self.action_select()


class SettingsScreen(ModalScreen[str | None]):
    """Full-screen browser; link commands are returned to the host on close."""

    SCOPED_CSS = False
    CSS_PATH = "settings.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
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
        self._needs_refresh = False
        self._cursor: str | None = None
        self._expanded: str | None = None
        self._list_edit_index: int | None = None
        self._list_draft: list[str] | None = None
        self._enum_draft: str | None = None
        self._inventory_draft: dict[str, list[str]] | None = None
        self._editing: str | None = None
        self._error = ""
        self._error_path: str | None = None
        self._unresolved: dict[str, str] = {}
        self._confirmation: tuple[str, SettingDescriptorWire, Any] | None = None
        self._pending_navigation: SettingDescriptorWire | None = None
        self._help_open = False
        self._help_return_focus: Widget | None = None
        self.return_state: tuple[str, str | None, int, str | None] | None = None

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
            yield SettingsChecklist(id="settings-checklist")
            with Vertical(id="settings-editor"):
                yield NoMarkupStatic("", id="settings-editor-label")
                yield VscodeCompatInput(
                    id="settings-input", validate_on=["blur", "submitted"]
                )
                yield NoMarkupStatic("", id="settings-editor-error")
            with Vertical(id="settings-confirmation"):
                with ConfirmationText(id="settings-confirmation-scroll"):
                    yield NoMarkupStatic("", id="settings-confirmation-text")
                yield OptionList(id="settings-confirmation-actions")
            yield NoMarkupStatic("", id="settings-help")
            yield NoMarkupStatic("", id="settings-hint")

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
            self.query_one(SettingsOptionList).focus()

    def _restore_return_state(self) -> None:
        state = self.return_state
        if state is None:
            return
        query, selected, scroll_y, focused_id = state
        options = self.query_one(SettingsOptionList)
        options._query = query
        self._filter(query)
        for index in range(options.option_count):
            if options.get_option_at_index(index).id == selected:
                options.highlighted = index
                break
        options.scroll_to(y=scroll_y, animate=False, force=True, immediate=True)
        target = self.query_one(f"#{focused_id}") if focused_id else options
        target.focus()

    def on_resize(self, event: events.Resize) -> None:
        self._resize_surface()

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
        options = self.query_one(SettingsOptionList)
        item = self._current_item()
        if self._busy:
            return [("Esc", "Wait for save")]
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
        if self._help_open:
            return [("F1", "Hide help"), ("Esc", "Back")]
        if self._editing is not None:
            return [
                (
                    "Enter",
                    "Add to draft"
                    if self._list_edit_index is not None
                    else "Save to user settings",
                ),
                ("Esc", "Cancel"),
            ]
        if self.query_one(SettingsChecklist).display:
            checklist = self.query_one(SettingsChecklist)
            selected = (
                checklist.get_option_at_index(checklist.highlighted).value
                if checklist.highlighted is not None
                else None
            )
            actions = [
                (
                    "Enter",
                    "Add pattern"
                    if selected == ADD_PATTERN
                    else "Save to user settings",
                ),
                ("Esc", "Back"),
            ]
            if selected is not None and selected.startswith("\x00pattern:"):
                actions.insert(1, ("Ctrl+D", "Delete pattern"))
            elif (
                selected is not None
                and selected != ADD_PATTERN
                and not self._inventory_item_state(selected)[1]
            ):
                actions.insert(1, ("Space", "Toggle"))
            return actions
        if self._expanded is not None and item is not None and item.kind == "list":
            actions = [("Enter", "Edit item"), ("Esc", "Back")]
            identifier = (
                str(options.highlighted_option.id) if options.highlighted_option else ""
            )
            if identifier.endswith(":apply"):
                actions[0] = ("Enter", "Save to user settings")
            elif identifier.endswith(":add"):
                actions[0] = ("Enter", "Add item")
            elif identifier.rsplit(":", 1)[-1].isdigit():
                actions.insert(1, ("Ctrl+D", "Delete item"))
            return actions
        if self._expanded is not None and item is not None and item.kind == "enum":
            return [
                ("Enter", "Save choice to user settings"),
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
            if item.control == "toggle_inventory" or (field and field.saved_explicit):
                actions.insert(-1, ("Ctrl+R", "Remove override"))
        return actions

    def _update_hint(self) -> None:
        if not self.is_mounted:
            return
        actions = self._available_actions()
        content = self.query_one("#settings-content", Vertical)
        available = max(1, (content.region.width or min(self.size.width, 92)) - 4)

        def render(pairs: list[tuple[str, str]]) -> str:
            return "  ".join(f"{shortcut(key)} {label}" for key, label in pairs)

        visible = actions
        if (
            len(shortcut_hint(render(actions)).plain) > available
            and len(actions) > MIN_SHORTCUTS_WITH_OVERFLOW
        ):
            visible = [actions[0], ("F1", "Help"), actions[-1]]
        if (
            len(shortcut_hint(render(visible)).plain) > available
            and len(visible) > MIN_SHORTCUTS_WITH_OVERFLOW
        ):
            visible = [visible[0], visible[-1]]
        self.query_one("#settings-hint", NoMarkupStatic).update(
            shortcut_hint(render(visible))
        )

    def action_help(self) -> None:
        help_widget = self.query_one("#settings-help", NoMarkupStatic)
        if self._help_open:
            help_widget.styles.max_height = 4
            help_widget.remove_class("details-open")
            if self._help_return_focus is not None:
                self._help_return_focus.focus()
            help_widget.can_focus = False
            self._help_return_focus = None
        else:
            self._help_return_focus = self.focused
            help_widget.can_focus = True
            help_widget.add_class("details-open")
        self._help_open = not self._help_open
        self._update_help()
        if self._help_open:
            help_widget.focus()
        self._update_hint()

    def _filter(self, query: str) -> None:
        self.query_one("#settings-filter", NoMarkupStatic).update(
            f"Filter: {query or 'type to filter'}"
        )
        self._refresh_options(preserve=False)

    def _display_value(self, item: SettingDescriptorWire) -> str:
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
                if self.query_one(SettingsOptionList).has_focus
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
            if item.control == "toggle_inventory" or item.kind == "link"
            else item.path,
            "    ",
            ("" if item.kind == "bool" else self._display_value(item), "$text-muted"),
        )

    def _refresh_options(self, *, preserve: bool = True) -> None:
        options = self.query_one(SettingsOptionList)
        previous = (
            str(options.highlighted_option.id)
            if preserve and options.highlighted_option
            else None
        )
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
        if self._expanded is not None:
            parent = next(
                (item for item in self.catalog if item.path == self._expanded), None
            )
            ids = [row.id for row in rows]
            if parent is not None and parent.path in ids:
                position = ids.index(parent.path) + 1
                if parent.kind == "list":
                    for index, value in enumerate(self._list_draft or []):
                        rows.insert(
                            position,
                            Option(
                                Content.assemble(
                                    "  ", ("  ", "$text-muted"), (value, "$foreground")
                                ),
                                id=f"list:{parent.path}:{index}",
                            ),
                        )
                        position += 1
                    rows.insert(
                        position,
                        Option(
                            Content.assemble("    Add item"),
                            id=f"list:{parent.path}:add",
                        ),
                    )
                    rows.insert(
                        position + 1,
                        Option(
                            Content.assemble("    Apply changes"),
                            id=f"list:{parent.path}:apply",
                        ),
                    )
                for choice in parent.choices:
                    rows.insert(
                        position,
                        Option(
                            Content.assemble(
                                "  ",
                                (
                                    chrome_glyph("radio_selected")
                                    if self._enum_draft == choice
                                    else chrome_glyph("radio_empty"),
                                    "$foreground",
                                ),
                                " ",
                                (choice, "$foreground"),
                            ),
                            id=f"choice:{parent.path}:{choice}",
                        ),
                    )
                    position += 1
        if not any(not row.disabled for row in rows):
            rows.append(Option(Content.assemble("  No matching settings"), id="empty"))
        options.clear_options()
        options.add_options(rows)
        ids = [row.id for row in rows]
        options.highlighted = (
            ids.index(previous)
            if previous is not None and previous in ids
            else next((i for i, row in enumerate(rows) if not row.disabled), None)
        )
        self._cursor = None
        self._mark_cursor()

    def _mark_cursor(self) -> None:
        options = self.query_one(SettingsOptionList)
        item_id = (
            str(options.highlighted_option.id)
            if options.highlighted_option and options.highlighted_option.id
            else None
        )
        for path in (self._cursor, item_id):
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
                label = (
                    "Apply changes"
                    if index == "apply"
                    else "Add item"
                    if index == "add"
                    else (self._list_draft or [])[int(index)]
                )
                prompt = Content.assemble(
                    (
                        f"{chrome_glyph('cursor')} " if path == item_id else "  ",
                        "$primary bold" if options.has_focus else "$text-muted",
                    ),
                    "  ",
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
        self._cursor = item_id
        self._update_help()
        self._update_hint()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if self.is_mounted:
            self.call_after_refresh(self._mark_cursor)

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        if self.is_mounted:
            self.call_after_refresh(self._mark_cursor)

    def _current_item(self) -> SettingDescriptorWire | None:
        option = self.query_one(SettingsOptionList).highlighted_option
        path = str(option.id) if option and option.id else ""
        if path.startswith(("choice:", "list:")):
            path = path.split(":", 2)[1]
        if self._expanded is not None and self.query_one(SettingsChecklist).display:
            path = self._expanded
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
                    f"\nType to filter; {chrome_glyph('vertical')}/jk move; Enter opens or accepts; Space toggles or selects; Ctrl+R removes override; Ctrl+D deletes an item; Esc goes back. List and inventory drafts require Apply changes.",
                )
            )
            return
        if item is None:
            self.query_one("#settings-help", NoMarkupStatic).update(
                self._help_with_feedback(
                    "No matching settings. Esc clears the filter."
                    if self.query_one(SettingsOptionList)._query
                    else ""
                )
            )
            return
        if item.control == "toggle_inventory":
            draft = self._inventory_draft
            detail = ""
            checklist = self.query_one(SettingsChecklist)
            if (
                draft is not None
                and checklist.display
                and checklist.highlighted is not None
            ):
                value = checklist.get_option_at_index(checklist.highlighted).value
                if value.startswith("\x00pattern:"):
                    detail = "\nPattern entries are read-only here; Ctrl+D removes the highlighted entry."
                elif value != ADD_PATTERN and self._pattern_driven(
                    item.inventory or "tools", value
                ):
                    detail = "\nThis state comes from a pattern entry below. Remove that entry with Ctrl+D to toggle this item."
            help_widget = self.query_one("#settings-help", NoMarkupStatic)
            help_widget.set_class(self._inventory_draft is not None, "inventory-help")
            help_widget.update(
                self._help_with_feedback(
                    item.description
                    + "\nEmptying the allow-only list returns to default mode (except disabled entries)."
                    + "\nPattern-controlled items cannot toggle; Ctrl+D removes the pattern entry. Ctrl+R resets both lists."
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
        ):
            self._error = ""
        self._mark_cursor()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "settings-confirmation-actions":
            self._dismiss_confirmation(apply=event.option.id == "apply")
            return
        if self._busy or self._confirmation is not None or event.option.id is None:
            return
        option_id = str(event.option.id)
        if option_id.startswith("list:"):
            _, path, index = option_id.split(":", 2)
            item = next((entry for entry in self.catalog if entry.path == path), None)
            if item is not None and self._expanded == path:
                if index == "apply":
                    self._save_list(item)
                else:
                    self._begin_list_input(item, index)
            return
        if option_id.startswith("choice:"):
            _, path, _ = option_id.split(":", 2)
            item = next((entry for entry in self.catalog if entry.path == path), None)
            if item is not None and self._enum_draft in item.choices:
                choice = self._enum_draft
                self._collapse()
                self.run_worker(self._write(item, choice), group="settings-write")
            return
        item = self._current_item()
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
        if item.kind == "list" and self._list_draft is not None:
            return self._list_draft != (field.effective_value if field else [])
        if item.kind == "enum" and self._enum_draft is not None:
            return self._enum_draft != (field.effective_value if field else None)
        if self._inventory_draft is not None:
            enabled, disabled = self._inventory_values(item.inventory or "tools")
            return (
                self._inventory_draft["enabled"] != enabled
                or self._inventory_draft["disabled"] != disabled
            )
        return False

    def _open_setting(self, item: SettingDescriptorWire) -> None:
        if item.kind == "link":
            options = self.query_one(SettingsOptionList)
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
        elif item.kind == "bool":
            self._toggle_bool(item)
        elif item.kind in {"list", "enum"} or item.control == "toggle_inventory":
            self._expand(item)
        else:
            self._begin_input(item)

    def _toggle_bool(self, item: SettingDescriptorWire) -> None:
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
            self.run_worker(self._write(item, value), group="settings-write")

    def _expand(self, item: SettingDescriptorWire) -> None:
        self._expanded = item.path
        field = self.fields.get(item.path)
        value = field.effective_value if field else None
        if item.kind == "enum":
            self._enum_draft = str(value) if value is not None else None
        if item.kind == "list":
            self._list_draft = (
                [entry for entry in value if isinstance(entry, str)]
                if isinstance(value, list)
                else []
            )
        if item.control == "toggle_inventory":
            self._open_checklist(item)
            return
        options = self.query_one(SettingsOptionList)
        options.editing = True
        self._refresh_options()
        ids = [row.id for row in options.options]
        if item.kind == "list":
            target = (
                f"list:{item.path}:0" if self._list_draft else f"list:{item.path}:add"
            )
        else:
            target = f"choice:{item.path}:{value}"
        if target in ids:
            options.highlighted = ids.index(target)

    def _save_list(self, item: SettingDescriptorWire) -> None:
        values = list(self._list_draft or [])
        saved = self.fields.get(item.path)
        original = saved.effective_value if saved is not None else []
        self._collapse()
        if values != original:
            self.run_worker(self._write(item, values), group="settings-write")

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
            draft["enabled"],
            draft["disabled"],
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
        checklist = self.query_one(SettingsChecklist)
        checklist.clear_options()
        checklist.add_options(
            [
                (
                    f"  {name}"
                    + (
                        " (pattern-controlled)"
                        if self._pattern_driven(category, name)
                        else ""
                    ),
                    name,
                    self._effective_inventory_item(category, name),
                )
                for name in names
            ]
            + [
                (f"  {entry} ({side} pattern)", f"\x00pattern:{side}:{index}", False)
                for side, values in (("enabled", enabled), ("disabled", disabled))
                for index, entry in enumerate(values)
                if entry.lower() not in {name.lower() for name in names}
            ]
            + [("  + Add Pattern", ADD_PATTERN, False)]
        )
        checklist.highlighted = next(
            (
                index
                for index in range(checklist.option_count)
                if checklist.get_option_at_index(index).value == highlight
            ),
            0,
        )
        self._update_help()

    def _toggle_inventory(self, name: str) -> bool:
        draft = self._inventory_draft
        if draft is None:
            return False
        item = next(item for item in self.catalog if item.path == self._expanded)
        category = item.inventory or "tools"
        if self._pattern_driven(category, name):
            return False
        is_enabled = self._effective_inventory_item(category, name)
        draft["enabled"], draft["disabled"] = toggle_inventory_name(
            name,
            draft["enabled"],
            draft["disabled"],
            is_enabled=is_enabled,
            category=category,
        )
        return True

    def _open_checklist(self, item: SettingDescriptorWire) -> None:
        checklist = self.query_one(SettingsChecklist)
        enabled, disabled = self._inventory_values(item.inventory or "tools")
        self._inventory_draft = {"enabled": enabled, "disabled": disabled}
        self._render_checklist()
        options = self.query_one(SettingsOptionList)
        options.editing = True
        options.styles.height = 1  # Keep the parent row above its checklist.
        checklist.display = True
        checklist.highlighted = 0
        checklist.focus()
        self._update_hint()
        self._update_help()

    def _commit_checklist(self) -> None:
        checklist = self.query_one(SettingsChecklist)
        if not checklist.display or self._busy or self._expanded is None:
            return
        item = next(item for item in self.catalog if item.path == self._expanded)
        if checklist.highlighted == checklist.option_count - 1:
            self._begin_list_input(item, "add")
            return
        self._save_inventory(item)

    def _save_inventory(self, item: SettingDescriptorWire) -> None:
        draft = self._inventory_draft
        if draft is None:
            return
        category = item.inventory or "tools"
        saved_enabled, saved_disabled = self._inventory_values(category)
        changes = {
            f"{side}_{category}": list(draft[side])
            for side, saved in (
                ("enabled", saved_enabled),
                ("disabled", saved_disabled),
            )
            if draft[side] != saved
        }
        self._collapse()
        if changes:
            self.run_worker(self._write_changes(item, changes), group="settings-write")

    def _collapse(self) -> None:
        path = self._expanded
        self._expanded = None
        self._list_draft = None
        self._enum_draft = None
        self._inventory_draft = None
        self._list_edit_index = None
        checklist = self.query_one(SettingsChecklist)
        if checklist.display:
            checklist.display = False
            self.query_one(SettingsOptionList).styles.height = "1fr"
        options = self.query_one(SettingsOptionList)
        options.editing = False
        if path is not None:
            self._refresh_options()
            ids = [row.id for row in options.options]
            if path in ids:
                options.highlighted = ids.index(path)

    def _begin_list_input(self, item: SettingDescriptorWire, index: str) -> None:
        values = self._list_draft or []
        self._list_edit_index = len(values) if index == "add" else int(index)
        value = (
            values[self._list_edit_index] if self._list_edit_index < len(values) else ""
        )
        self._show_editor(
            item,
            value,
            "Add item" if index == "add" else f"Edit item {self._list_edit_index + 1}",
        )

    def _begin_input(self, item: SettingDescriptorWire) -> None:
        field = self.fields.get(item.path)
        value = (
            ""
            if field is None or field.effective_value is None
            else str(field.effective_value)
        )
        self._show_editor(item, value, item.label)

    def _show_editor(self, item: SettingDescriptorWire, value: str, label: str) -> None:
        self._editing = item.path
        self.query_one(SettingsOptionList).editing = True
        self.query_one(SettingsOptionList).display = False
        self.query_one(SettingsChecklist).display = False
        self.query_one("#settings-editor-label", NoMarkupStatic).update(
            f"{label} — Enter adds to draft; Apply changes saves; Esc cancels"
            if self._list_edit_index is not None
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
        list_input = self._list_edit_index is not None
        self._editing = None
        self.query_one("#settings-editor").display = False
        self.query_one("#settings-editor-error", NoMarkupStatic).update("")
        self.query_one("#settings-input", Input).remove_class("-invalid")
        checklist = self.query_one(SettingsChecklist)
        if checklist.display or (list_input and self._inventory_draft is not None):
            self.query_one(SettingsOptionList).display = True
            checklist.display = True
            checklist.focus()
        else:
            options = self.query_one(SettingsOptionList)
            options.display = True
            options.editing = self._expanded is not None
            options.focus()
        self._list_edit_index = None
        self._update_hint()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._editing is None or self._busy:
            return
        item = next(entry for entry in self.catalog if entry.path == self._editing)
        if self._list_edit_index is not None:
            if item.control == "toggle_inventory":
                text = event.value.strip()
                draft = self._inventory_draft
                if draft is None or not text:
                    self._cancel_input()
                    return
                side = "enabled" if draft["enabled"] else "disabled"
                if text not in draft[side]:
                    draft[side].append(text)
                self._cancel_input()
                self._render_checklist(highlight=ADD_PATTERN)
                return
            values = list(self._list_draft or [])
            text = event.value.strip()
            if not text:
                event.input.add_class("-invalid")
                self.query_one("#settings-editor-error", NoMarkupStatic).update(
                    "Error: Enter a nonempty item; use Ctrl+D to delete an existing item."
                )
                return
            edited_index = self._list_edit_index
            if edited_index < len(values):
                values[edited_index] = text
            else:
                values.append(text)
            self._cancel_input()
            self._list_draft = values
            self._refresh_options()
            options = self.query_one(SettingsOptionList)
            ids = [row.id for row in options.options]
            target = f"list:{item.path}:{edited_index}"
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
        self.run_worker(self._write(item, value), group="settings-write")

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._editing is None or not event.input.has_class("-invalid"):
            return
        item = next(entry for entry in self.catalog if entry.path == self._editing)
        if self._list_edit_index is not None and item.kind == "list":
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
        self._confirmation = (kind, item, value)
        self.query_one(SettingsOptionList).display = False
        self.query_one(SettingsChecklist).display = False
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
        pending = self._confirmation
        self._confirmation = None
        self.query_one("#settings-confirmation").display = False
        options = self.query_one(SettingsOptionList)
        options.display = True
        checklist = self.query_one(SettingsChecklist)
        if (
            pending
            and (
                pending[0] in {"reset", "pattern"}
                or (pending[0] == "discard" and self._inventory_draft is not None)
            )
            and self._expanded is not None
        ):
            checklist.display = True
            options.styles.height = 1
            checklist.focus()
        else:
            options.focus()
        self._update_help()
        self._update_hint()
        if not apply or pending is None:
            self._pending_navigation = None
            return
        kind, item, value = pending
        if kind == "pattern":
            side, index = value
            if self._inventory_draft is not None:
                self._inventory_draft[side].pop(index)
                self._render_checklist()
        elif kind == "delete":
            values = list(self._list_draft or [])
            values.pop(value)
            self._list_draft = values
            self._refresh_options()
        elif kind == "discard":
            self._collapse()
            target = self._pending_navigation
            self._pending_navigation = None
            if target is not None:
                options = self.query_one(SettingsOptionList)
                ids = [row.id for row in options.options]
                if target.path in ids:
                    options.highlighted = ids.index(target.path)
                self._open_setting(target)
        elif kind == "reset":
            category = item.inventory or "tools"
            self._collapse()
            self.run_worker(
                self._write_changes(
                    item, {f"enabled_{category}": None, f"disabled_{category}": None}
                ),
                group="settings-write",
            )
        else:
            if self._expanded is not None:
                self._collapse()
            self.run_worker(self._write(item, value), group="settings-write")

    def on_key(self, event: events.Key) -> None:
        if self._confirmation is not None:
            return
        if event.key == "space" and self._editing is None:
            options = self.query_one(SettingsOptionList)
            if options.has_focus and self._expanded is not None:
                option = options.highlighted_option
                identifier = str(option.id) if option and option.id else ""
                if identifier.startswith("choice:"):
                    self._enum_draft = identifier.split(":", 2)[2]
                    self._refresh_options()
                    event.stop()
                    event.prevent_default()
                    return
            if options.has_focus and self._expanded is None:
                option = options.highlighted_option
                if (
                    option is not None
                    and (item := self._current_item())
                    and item.kind == "bool"
                ):
                    self.on_option_list_option_selected(
                        OptionList.OptionSelected(
                            options, option, options.highlighted or 0
                        )
                    )
                    event.stop()
                    event.prevent_default()

    def action_delete_item(self) -> None:
        if self._busy or self._editing is not None or self._expanded is None:
            return
        checklist = self.query_one(SettingsChecklist)
        if self._inventory_draft is not None and checklist.display:
            value = checklist.get_option_at_index(checklist.highlighted or 0).value
            if value.startswith("\x00pattern:"):
                _, side, index = value.rsplit(":", 2)
                item = next(
                    entry for entry in self.catalog if entry.path == self._expanded
                )
                target = self._inventory_draft[side][int(index)]
                self._open_confirmation(
                    "pattern",
                    item,
                    (side, int(index)),
                    f"Delete pattern {target} from {side}_{item.inventory}? Cancel preserves the pattern and current draft.",
                    "Delete pattern",
                )
            return
        options = self.query_one(SettingsOptionList)
        option = options.highlighted_option
        identifier = str(option.id) if option and option.id else ""
        if not identifier.startswith(f"list:{self._expanded}:"):
            return
        index = identifier.rsplit(":", 1)[-1]
        if not index.isdigit():
            return
        item = next(entry for entry in self.catalog if entry.path == self._expanded)
        values = list(self._list_draft or [])
        self._open_confirmation(
            "delete",
            item,
            int(index),
            f"Delete {values[int(index)]} from the {item.label} draft? Apply changes will save the shortened list. Cancel preserves the item and stored list.",
            "Delete item",
        )

    def action_remove_override(self) -> None:
        if self._busy or self._confirmation is not None:
            return
        item = self._current_item()
        if item is not None and item.control == "toggle_inventory":
            if self.snapshot.view_only:
                return
            if self._editing is not None:
                self._cancel_input()
            self._open_confirmation(
                "reset",
                item,
                None,
                f"Reset {item.label}? Clears enabled and disabled {item.inventory} lists. Cancel preserves both saved lists and the current draft.",
                "Reset lists",
            )
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

    async def _write(
        self, item: SettingDescriptorWire, value: JsonValue | list[str] | None
    ) -> None:
        await self._write_changes(item, {item.path: value})

    async def _write_changes(
        self,
        item: SettingDescriptorWire,
        changes: Mapping[str, JsonValue | list[str] | None],
    ) -> None:
        if self._busy:
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
            self.query_one(SettingsOptionList).focus()
            self._update_hint()

    def action_close(self) -> None:
        if self._busy:
            return
        if self._confirmation is not None:
            self._dismiss_confirmation()
        elif self._help_open:
            self.action_help()
        elif self._editing is not None:
            self._cancel_input()
        elif self._expanded is not None:
            item = next(entry for entry in self.catalog if entry.path == self._expanded)
            if self._inventory_draft is not None:
                saved_enabled, saved_disabled = self._inventory_values(
                    item.inventory or "tools"
                )
                dirty = (
                    self._inventory_draft["enabled"] != saved_enabled
                    or self._inventory_draft["disabled"] != saved_disabled
                )
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
                if self._list_draft != (field.effective_value if field else []):
                    self._pending_navigation = None
                    self._open_confirmation(
                        "discard",
                        item,
                        None,
                        f"Discard edits to {item.label}? Unsaved list changes will be lost. Cancel preserves the saved list and the current draft.",
                        "Discard edits",
                    )
                    return
            self._collapse()
        elif (options := self.query_one(SettingsOptionList))._query:
            options._query = ""
            self._filter("")
        else:
            self.dismiss(None)
