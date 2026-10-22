"""Searchable, flat settings browser with inline single-leaf editing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import JsonValue
from rich.cells import cell_len
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Input, OptionList, SelectionList
from textual.widgets.option_list import Option, OptionDoesNotExist

from chartreux.app_server.protocol import (
    STATUS_LINE_PATHS,
    SettingDescriptorWire,
    SettingsReadResponse,
)
from chartreux.cli.textual_ui.screens.settings_collections import (
    SettingsCollectionsController,
)
from chartreux.cli.textual_ui.screens.settings_confirmation import (
    SettingsConfirmationController,
)
from chartreux.cli.textual_ui.screens.settings_editor import SettingsEditorController
from chartreux.cli.textual_ui.screens.settings_write import SettingsWriteController
from chartreux.cli.textual_ui.screens.status_line_settings import (
    StatusLineSettingsResult,
    StatusLineSettingsScreen,
)
from chartreux.cli.textual_ui.widgets.settings_help import SettingsHelp, SettingsHints
from chartreux.cli.textual_ui.widgets.settings_lists import (
    ConfirmationText,
    SettingsChecklist,
    SettingsConfirmationList,
    SettingsGroupList,
    SettingsOptionList,
    _draft_values,
    _DraftEntry,
    _ListEditorContext,
    inventory_item_state,  # noqa: F401
    inventory_name_matches,  # noqa: F401
    parse_setting_value,  # noqa: F401
    toggle_inventory_name,  # noqa: F401
)
from chartreux.cli.textual_ui.widgets.shell_permissions_app import (
    ShellPermissionsApp,
    ShellPermissionsResult,
)
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.settings_service import SettingsService
from chartreux.ui.shell_permissions import ShellPermissionsService
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from chartreux.ui.widgets.vscode_compat import VscodeCompatInput

MIN_MODAL_WIDTH = 84
MIN_MODAL_HEIGHT = 28
MIN_SHORTCUTS_WITH_OVERFLOW = 2


@dataclass(frozen=True)
class _GroupPosition:
    group: str
    context: str
    identities: tuple[str, ...]
    ordinal: int
    scroll_x: float
    scroll_y: float
    cursor_position: int | None = None


GROUPS = (
    "Interface",
    "Identity & Orchestration",
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
        self,
        service: SettingsService,
        snapshot: SettingsReadResponse,
        *,
        shell_permissions: ShellPermissionsService | None = None,
    ) -> None:
        super().__init__(id="settings-screen")
        self.service = service
        self.snapshot = snapshot
        self.shell_permissions = shell_permissions
        self.catalog = snapshot.catalog
        self.fields = {field.path: field for field in snapshot.fields}
        self._busy = False
        self._pointer_command_started = False
        self._needs_refresh = False
        self._cursor: str | None = None
        self._expanded: str | None = None
        self._error = ""
        self._error_path: str | None = None
        self._unresolved: dict[str, str] = {}
        self._positions: dict[tuple[str, str], _GroupPosition] = {}
        self._group_contents_context: dict[str, str] = {}
        self._openers: dict[str, _GroupPosition] = {}
        self._help_open = False
        self._child_returning = False
        self._help_return_focus: _GroupPosition | None = None
        self.return_state: tuple[str, str | None, int, str | None] | None = None
        # The write controller owns the save path; the screen is its late-bound
        # host and keeps the _start_write/_write/_write_changes delegates.
        self._write_controller = SettingsWriteController(self)
        # The collections controller owns the draft token/sequence state and
        # the collection edit/save/expand/collapse surface; the screen is its
        # late-bound host, forwards the moved draft fields through the
        # properties below, and keeps the collection delegates.
        self._collections = SettingsCollectionsController(self)
        # The editor controller owns the inline editor surface and the
        # _editing/_list_editor state; the screen is its late-bound host,
        # forwards the moved fields through the properties below, and keeps
        # the editor delegates (including the on_input_* handlers).
        self._editor = SettingsEditorController(self)
        # The confirmation controller owns the confirmation surface and the
        # _confirmation/_pending_navigation state (_confirmation_return_focus
        # is internal); the screen is its late-bound host, forwards the moved
        # fields through the properties below, and keeps the confirmation
        # delegates (including action_delete_item/action_remove_override).
        self._confirmation_controller = SettingsConfirmationController(self)

    @property
    def _list_draft(self) -> list[_DraftEntry] | None:
        return self._collections._list_draft

    @_list_draft.setter
    def _list_draft(self, value: list[_DraftEntry] | None) -> None:
        self._collections._list_draft = value

    @property
    def _enum_draft(self) -> str | None:
        return self._collections._enum_draft

    @property
    def _inventory_draft(self) -> dict[str, list[_DraftEntry]] | None:
        return self._collections._inventory_draft

    @_inventory_draft.setter
    def _inventory_draft(self, value: dict[str, list[_DraftEntry]] | None) -> None:
        self._collections._inventory_draft = value

    @property
    def _list_editor(self) -> _ListEditorContext | None:
        return self._editor._list_editor

    @_list_editor.setter
    def _list_editor(self, value: _ListEditorContext | None) -> None:
        self._editor._list_editor = value

    @property
    def _editing(self) -> str | None:
        return self._editor._editing

    @property
    def _confirmation(self) -> tuple[str, SettingDescriptorWire, Any] | None:
        return self._confirmation_controller._confirmation

    @property
    def _pending_navigation(self) -> SettingDescriptorWire | None:
        return self._confirmation_controller._pending_navigation

    @_pending_navigation.setter
    def _pending_navigation(self, value: SettingDescriptorWire | None) -> None:
        self._confirmation_controller._pending_navigation = value

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
        return self._collections.new_draft_entry(value)

    def _make_draft(self, values: list[str]) -> list[_DraftEntry]:
        return self._collections.make_draft(values)

    def _reconcile_draft(
        self, entries: list[_DraftEntry], values: list[str]
    ) -> list[_DraftEntry]:
        return self._collections.reconcile_draft(entries, values)

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
            yield SettingsOptionList(self._filter, id="settings-options", host=self)
            yield NoMarkupStatic("", id="settings-active-editor")
            yield SettingsGroupList(
                id="settings-entries", classes="expanded-body", host=self
            )
            yield SettingsGroupList(
                id="settings-choices", classes="expanded-body", host=self
            )
            yield SettingsChecklist(
                id="settings-checklist", classes="expanded-body", host=self
            )
            yield SettingsGroupList(id="settings-actions", host=self)
            with Vertical(id="settings-editor"):
                yield NoMarkupStatic("", id="settings-editor-label")
                yield VscodeCompatInput(
                    id="settings-input", validate_on=["blur", "submitted"]
                )
                yield NoMarkupStatic("", id="settings-editor-error")
            with Vertical(id="settings-confirmation"):
                with ConfirmationText(id="settings-confirmation-scroll"):
                    yield NoMarkupStatic("", id="settings-confirmation-text")
                yield SettingsConfirmationList(
                    id="settings-confirmation-actions", host=self
                )
            yield SettingsHelp("", id="settings-help", host=self)
            yield SettingsHints("", id="settings-hint", host=self)

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
            if item.control in {"toggle_inventory", "status_line", "shell_permissions"}
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
            if feedback.startswith(chrome_glyph("success"))
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
        if item.control == "shell_permissions":
            self.query_one("#settings-help", NoMarkupStatic).update(
                self._help_with_feedback(
                    item.description
                    + "\nEnter opens the Shell Permissions panel: live session "
                    "grants, saved project grants, and blocking denylist "
                    "entries, with per-operation revocation."
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
            roster = item.inventory == "identities"
            help_widget.update(
                self._help_with_feedback(
                    (
                        "Ctrl+R resets the denylist.\n"
                        + item.description
                        + "\nEmptying the denylist makes every identity launchable."
                        if roster
                        else "Ctrl+R resets both lists.\n"
                        + item.description
                        + "\nEmptying the allow-only list returns to default mode "
                        "(except disabled entries)."
                    )
                    + "\nPattern-controlled items cannot toggle; Ctrl+D removes the "
                    "pattern entry."
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
        self._collections.select_enum_draft()

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
        self._collections.apply_collection(item)

    def _accept_enum_draft(self, item: SettingDescriptorWire) -> None:
        self._collections.accept_enum_draft(item)

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
        return self._collections.draft_is_dirty(item)

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
        elif item.control == "shell_permissions":
            self._open_shell_permissions(item)
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

    def _open_shell_permissions(self, item: SettingDescriptorWire) -> None:
        if self.shell_permissions is None:
            self._error_path = item.path
            self._error = (
                f"{chrome_glyph('error')} Failed: Shell Permissions is "
                "unavailable for this session."
            )
            self._update_help()
            return
        self._remember_opener("child")
        options = self.query_one("#settings-options", SettingsOptionList)
        state = (
            options._query,
            str(options.highlighted_option.id) if options.highlighted_option else None,
            int(options.scroll_y),
            self.focused.id if self.focused is not None else None,
        )

        def shell_permissions_returned(result: ShellPermissionsResult | None) -> None:
            if result is not None and result.feedback.startswith((
                chrome_glyph("success"),
                "! Warning:",
                "Failed:",
            )):
                self._error_path = item.path
                self._error = result.feedback
                self._unresolved.pop(item.path, None)
                if result.feedback.startswith("! Warning:"):
                    self._unresolved[item.path] = result.feedback
            self.return_state = state
            self._child_returning = True
            self._refresh_options()
            self.call_after_refresh(self._restore_child_state)

        self.app.push_screen(
            ShellPermissionsApp(self.shell_permissions), shell_permissions_returned
        )

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
        self._collections.expand(item)

    def _save_list(self, item: SettingDescriptorWire) -> None:
        self._collections.save_list(item)

    def _inventory_values(self, category: str) -> tuple[list[str], list[str]]:
        return self._collections.inventory_values(category)

    def _inventory_item_state(self, name: str) -> tuple[bool, bool]:
        return self._collections.inventory_item_state(name)

    def _pattern_driven(
        self, category: Literal["tools", "skills", "identities"], name: str
    ) -> bool:
        return self._collections.pattern_driven(category, name)

    def _effective_inventory_item(
        self, category: Literal["tools", "skills", "identities"], name: str
    ) -> bool:
        return self._collections.effective_inventory_item(category, name)

    def _render_checklist(self, *, highlight: str | None = None) -> None:
        self._collections.render_checklist(highlight=highlight)

    def _toggle_inventory(self, name: str) -> bool:
        return self._collections.toggle_inventory(name)

    def _open_checklist(self, item: SettingDescriptorWire) -> None:
        self._collections.open_checklist(item)

    def _save_inventory(self, item: SettingDescriptorWire) -> None:
        self._collections.save_inventory(item)

    def _collapse(self, *, restore_opener: bool = True) -> None:
        self._collections.collapse(restore_opener=restore_opener)

    def _begin_list_input(self, item: SettingDescriptorWire, index: str) -> None:
        self._editor.begin_list_input(item, index)

    def _begin_input(self, item: SettingDescriptorWire) -> None:
        self._editor.begin_input(item)

    def _show_editor(self, item: SettingDescriptorWire, value: str, label: str) -> None:
        self._editor.show_editor(item, value, label)

    def _cancel_input(self) -> None:
        self._editor.cancel_input()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._editor.submit(event)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._editor.value_changed(event)

    def _open_confirmation(
        self,
        kind: str,
        item: SettingDescriptorWire,
        value: Any,
        message: str,
        action: str,
    ) -> None:
        self._confirmation_controller.open_confirmation(
            kind, item, value, message, action
        )

    def _dismiss_confirmation(self, *, apply: bool = False) -> None:
        self._confirmation_controller.dismiss_confirmation(apply=apply)

    def on_key(self, event: events.Key) -> None:
        if self._confirmation is not None:
            return
        if event.key == "space" and self._editing is None:
            self.action_select_draft()
            event.stop()
            event.prevent_default()

    def action_delete_item(self) -> None:
        self._confirmation_controller.delete_item()

    def action_remove_override(self) -> None:
        self._confirmation_controller.remove_override()

    def _start_write(
        self,
        item: SettingDescriptorWire,
        changes: Mapping[str, JsonValue | list[str] | None],
    ) -> None:
        self._write_controller.start_write(item, changes)

    async def _write(
        self, item: SettingDescriptorWire, value: JsonValue | list[str] | None
    ) -> None:
        await self._write_controller.write(item, value)

    async def _write_changes(
        self,
        item: SettingDescriptorWire,
        changes: Mapping[str, JsonValue | list[str] | None],
        *,
        reserved: bool = False,
    ) -> None:
        await self._write_controller.write_changes(item, changes, reserved=reserved)

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
                    roster = item.inventory == "identities"
                    self._open_confirmation(
                        "discard",
                        item,
                        None,
                        "Discard edits to the task identity denylist? Unsaved roster changes will be lost. Cancel preserves the saved denylist and the current draft."
                        if roster
                        else f"Discard edits to enabled and disabled {item.inventory} lists? Unsaved inventory changes will be lost. Cancel preserves both saved lists and the current draft.",
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
