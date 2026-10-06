"""Draft-only state cyclers for the composite status line setting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import JsonValue
from rich.cells import cell_len
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from chartreux.app_server.config import StatusLineConfigView
from chartreux.app_server.models import UsageWindowSummary
from chartreux.app_server.protocol import STATUS_LINE_PATHS, SettingsReadResponse
from chartreux.cli.textual_ui.widgets.session_status_line import (
    SessionStatusState,
    format_status_line,
)
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.settings_service import SettingsService
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

DOUBLE_CLICK = 2
MIN_MODAL_WIDTH = 84
MIN_MODAL_HEIGHT = 28
MIN_SHORTCUTS_WITH_OVERFLOW = 2
SEGMENTS = (
    "directory",
    "pid",
    "model",
    "context",
    "git-branch",
    "spend-today",
    "spend-week",
    "spend-month",
    "background-jobs",
)
LABELS = {
    "directory": "Directory",
    "pid": "Process ID",
    "model": "Model",
    "context": "Context",
    "git-branch": "Git branch",
    "spend-today": "Spend today",
    "spend-week": "Spend week",
    "spend-month": "Spend month",
    "background-jobs": "Background jobs",
    "separator": "Separator",
    "apply": "Apply changes",
    "back": "Back",
}
DESCRIPTIONS = {
    "directory": "Show the directory name or home-abbreviated path. Always enabled.",
    "pid": "Show the current process ID.",
    "model": "Show the active provider/model identity.",
    "context": "Show tokens, alone or with percentage of the compaction threshold. Always enabled.",
    "git-branch": "Show the current Git branch or repository state.",
    "spend-today": "Show recorded USD spend across all projects today (local calendar day); the Usage browser's Current project filter never changes this scope. + means some cost is unknown; Unknown means none is priced; — means unavailable.",
    "spend-week": "Show recorded USD spend across all projects this local calendar week (Monday start); the Usage browser's Current project filter never changes this scope. + means some cost is unknown; Unknown means none is priced; — means unavailable.",
    "spend-month": "Show recorded USD spend across all projects this local calendar month; the Usage browser's Current project filter never changes this scope. + means some cost is unknown; Unknown means none is priced; — means unavailable.",
    "background-jobs": "Show Jobs N: active managed shell jobs in this root session, including jobs created by children that remain active after child completion. Shows Jobs 0 when none are active.",
    "separator": "Separate enabled segments with spaces or pipes. Fixed position; cannot reorder.",
    "apply": "Save only changed settings in one revision-checked batch.",
    "back": "Return to Settings; unsaved changes require discard confirmation.",
}
VIEW_ONLY = "User configuration is unavailable (view only)."


@dataclass(frozen=True)
class StatusLineSettingsResult:
    snapshot: SettingsReadResponse | None
    revision: str | None
    feedback: str
    needs_refresh: bool


@dataclass(frozen=True)
class _GroupPosition:
    group: str
    context: str
    identities: tuple[str, ...]
    ordinal: int
    scroll_x: float
    scroll_y: float


class StatusLineOptionList(NavigableOptionList):
    """Bounded configuration rows; clicks select, double clicks cycle."""

    def focus_on_click(self) -> bool:
        return False

    def on_show(self, event: events.Show | None = None) -> None:
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
        self.highlighted = valid[
            max(0, min(len(valid) - 1, valid.index(current) + direction * distance))
        ]

    def action_cursor_down(self) -> None:
        self._move_bounded(1)

    def action_cursor_up(self) -> None:
        self._move_bounded(-1)

    def action_page_down(self) -> None:
        self._move_bounded(1, max(1, self.scrollable_content_region.height))

    def action_page_up(self) -> None:
        self._move_bounded(-1, max(1, self.scrollable_content_region.height))

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        screen = self.screen
        if not isinstance(screen, StatusLineSettingsScreen):
            return
        confirmation = self.id == "status-line-settings-confirmation-actions"
        if not screen._begin_pointer_event(event) or screen._busy:
            return
        if not confirmation and screen._confirmation:
            return
        if not confirmation and (screen._help_open or screen._details_open):
            return
        index = event.style.meta.get("option")
        if index is not None and not self.get_option_at_index(index).disabled:
            self.highlighted = index
            self.focus()
            action_list = self.id != "status-line-settings-options"
            if (action_list and event.chain == 1) or (
                not action_list and event.chain == DOUBLE_CLICK
            ):
                screen._pointer_command_started = True
                self.action_select()

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        event.stop()


class StatusLineActions(StatusLineOptionList):
    """Fixed Enter-only commands, also available by single click."""


class StatusLineHints(NoMarkupStatic):
    targets: list[tuple[int, int, str]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.targets = []

    async def on_click(self, event: events.Click) -> None:
        action = next(
            (action for start, end, action in self.targets if start <= event.x < end),
            None,
        )
        screen = self.screen
        if action and isinstance(screen, StatusLineSettingsScreen):
            event.stop()
            if screen._begin_pointer_event(event) and event.chain == 1:
                screen._pointer_command_started = True
                await screen.run_action(action)


class StatusLineDetails(VerticalScroll):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up,k", "scroll_up", show=False),
        Binding("down,j", "scroll_down", show=False),
        Binding("pageup", "page_up", show=False),
        Binding("pagedown", "page_down", show=False),
    ]


class StatusLineSettingsScreen(ModalScreen[StatusLineSettingsResult]):
    CSS_PATH = "status_line_settings.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("tab", "next_group", show=False, priority=True),
        Binding("shift+tab", "previous_group", show=False, priority=True),
        Binding("escape", "close", show=False, priority=True),
        Binding("space", "cycle", show=False, priority=True),
        Binding("left_square_bracket,alt+up", "move_up", show=False, priority=True),
        Binding(
            "right_square_bracket,alt+down", "move_down", show=False, priority=True
        ),
        Binding("ctrl+r", "remove_override", show=False),
        Binding("d", "details", show=False),
        Binding("f1", "help", show=False),
    ]

    def __init__(
        self,
        service: SettingsService,
        snapshot: SettingsReadResponse,
        *,
        needs_refresh: bool = False,
    ) -> None:
        super().__init__(id="status-line-settings-screen")
        self.service = service
        self.snapshot = snapshot
        fields = {field.path: field for field in snapshot.fields}
        self.opening = StatusLineConfigView.model_validate({
            path.split(".")[1]: fields[path].effective_value
            for path in STATUS_LINE_PATHS
        })
        self.draft = self.opening.model_copy(deep=True)
        self.order = list(self.draft.segments) + [
            name for name in SEGMENTS if name not in self.draft.segments
        ]
        self._busy = False
        self._needs_refresh = needs_refresh
        self._confirmation: str | None = None
        self._help_open = False
        self._details_open = False
        self._positions: dict[tuple[str, str], _GroupPosition] = {}
        self._openers: dict[str, _GroupPosition] = {}
        self._active_group = "status-line-settings-options"
        self._pointer_command_started = False
        self._feedback = VIEW_ONLY if snapshot.view_only else ""
        example_start = datetime(2024, 1, 1, tzinfo=UTC)
        example_usage = {
            window: UsageWindowSummary(
                start_local=example_start,
                end_local=end,
                start_utc=example_start,
                end_utc=end,
                timezone="UTC",
                requests=3,
                known_cost_usd=cost,
                has_known_cost=True,
            )
            for window, end, cost in (
                ("day", datetime(2024, 1, 2, tzinfo=UTC), 1.23),
                ("week", datetime(2024, 1, 8, tzinfo=UTC), 12.34),
                ("month", datetime(2024, 2, 1, tzinfo=UTC), 45.67),
            )
        }
        self._example = SessionStatusState(
            cwd=Path("/home/example/projects/chartreux"),
            home_directory=Path("/home/example"),
            pid=4242,
            model_identity="Example/model",
            context_tokens=12345,
            auto_compact_threshold=100000,
            branch="main",
            branch_status="branch",
            active_background_job_count=2,
            usage_day=example_usage["day"],
            usage_week=example_usage["week"],
            usage_month=example_usage["month"],
            ascii_chrome=fields.get("ascii_chrome") is not None
            and fields["ascii_chrome"].effective_value is True,
        )

    @property
    def dirty(self) -> bool:
        return self.draft != self.opening

    def compose(self) -> ComposeResult:
        with Vertical(id="status-line-settings-content"):
            yield NoMarkupStatic("Status line", id="status-line-settings-title")
            pending = NoMarkupStatic(
                "Action pending · finish editing, then Esc to answer",
                id="status-line-settings-pending-action",
            )
            pending.display = False
            yield pending
            yield NoMarkupStatic(
                "Segment                 State       Required / Fixed",
                id="status-line-settings-heading",
            )
            yield StatusLineOptionList(id="status-line-settings-options")
            yield StatusLineActions(
                Option("Apply changes", id="apply"),
                Option("Back", id="back"),
                id="status-line-settings-actions",
            )
            with Vertical(id="status-line-settings-confirmation"):
                yield NoMarkupStatic("", id="status-line-settings-confirmation-text")
                yield StatusLineActions(id="status-line-settings-confirmation-actions")
            with StatusLineDetails(id="status-line-settings-details"):
                yield NoMarkupStatic("", id="status-line-settings-detail-text")
            yield NoMarkupStatic("", id="status-line-settings-feedback")
            yield NoMarkupStatic(
                "Preview · example · draft", id="status-line-settings-preview-label"
            )
            yield NoMarkupStatic("", id="status-line-settings-preview")
            controls = StatusLineHints(
                "Move up  Move down  Reset  Details  Help  Back",
                id="status-line-settings-controls",
            )
            controls.targets = [
                (0, 7, "move_up"),
                (9, 18, "move_down"),
                (20, 25, "remove_override"),
                (27, 34, "details"),
                (36, 40, "help"),
                (42, 46, "close"),
            ]
            yield controls
            yield StatusLineHints("", id="status-line-settings-hint")

    def on_mount(self) -> None:
        self.query_one("#status-line-settings-confirmation").display = False
        self.query_one("#status-line-settings-pending-action").display = bool(
            getattr(self.app, "_pending_callbacks", None)
            or getattr(self.app, "_pending_local_question", None)
        )
        self._render_rows("directory")
        self._resize_surface()
        self.query_one(StatusLineOptionList).focus()

    def on_resize(self) -> None:
        position = self._capture_group_position()
        self._resize_surface()
        self._restore_group_position(position, restore_focus=False)

    def _resize_surface(self) -> None:
        content = self.query_one("#status-line-settings-content", Vertical)
        width, height = self.size
        fullscreen = width < MIN_MODAL_WIDTH or height < MIN_MODAL_HEIGHT
        content.set_class(fullscreen, "fullscreen")
        content.styles.width = "100%" if fullscreen else min(92, width - 2)
        content.styles.height = "100%" if fullscreen else height - 2
        content.border_title = "" if fullscreen else "Status line"
        self.call_after_refresh(self._refresh_editor_layout)

    def _refresh_editor_layout(self) -> None:
        self._mark_cursor()
        self._update_details()
        self._update_preview()

    def _begin_pointer_event(self, event: events.Click) -> bool:
        if event.chain == 1:
            self._pointer_command_started = False
        return not self._pointer_command_started

    def _visible_focus_groups(self) -> list[Widget]:
        if self._confirmation:
            return []
        groups: list[Widget] = [
            self.query_one("#status-line-settings-options"),
            self.query_one("#status-line-settings-actions"),
        ]
        if self._help_open or self._details_open:
            groups.append(self.query_one("#status-line-settings-details"))
        return [widget for widget in groups if widget.display and widget.can_focus]

    def _capture_group_position(
        self, widget: Widget | None = None
    ) -> _GroupPosition | None:
        widget = widget or self.focused
        if widget is None or widget.id is None:
            return None
        identities = (
            tuple(str(row.id) for row in widget.options if not row.disabled)
            if isinstance(widget, OptionList)
            else ()
        )
        selected = widget.highlighted_option if isinstance(widget, OptionList) else None
        position = _GroupPosition(
            widget.id,
            "status_line",
            identities,
            identities.index(str(selected.id))
            if selected and str(selected.id) in identities
            else 0,
            widget.scroll_x,
            widget.scroll_y,
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
            valid = [i for i, row in enumerate(widget.options) if not row.disabled]
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
        self.call_after_refresh(
            lambda: widget.scroll_to(
                x=position.scroll_x,
                y=position.scroll_y,
                animate=False,
                force=True,
                immediate=True,
            )
        )

    def _cycle_group(self, direction: int) -> None:
        groups = self._visible_focus_groups()
        if not groups:
            return
        self._capture_group_position()
        index = groups.index(self.focused) if self.focused in groups else -1
        target = groups[(index + direction) % len(groups)]
        position = self._positions.get((target.id or "", "status_line"))
        if target.id != "status-line-settings-details":
            self._close_inspection(restore=False)
        if position:
            self._restore_group_position(position)
        else:
            target.focus()

    def action_next_group(self) -> None:
        self._cycle_group(1)

    def action_previous_group(self) -> None:
        self._cycle_group(-1)

    def _selected(self) -> str:
        option = self.query_one(f"#{self._active_group}", OptionList).highlighted_option
        return str(option.id) if option else "directory"

    def _state(self, name: str) -> str:
        if name == "directory":
            return self.draft.directory_style.title()
        if name == "context":
            return (
                "Tokens + %"
                if self.draft.context_style == "tokens-percent"
                else "Tokens"
            )
        if name == "separator":
            return self.draft.separator.title()
        return "On" if name in self.draft.segments else "Off"

    def _row_prompt(self, name: str, *, selected: bool = False) -> Content:
        options = self.query_one(
            "#status-line-settings-actions"
            if name in {"apply", "back"}
            else "#status-line-settings-options",
            OptionList,
        )
        required = (
            "Required"
            if name in {"directory", "context"}
            else "Fixed"
            if name == "separator"
            else ""
        )
        state = self._state(name) if name not in {"apply", "back"} else ""
        cursor = chrome_glyph("cursor") if selected else " "
        label_width = min(23, max(12, options.content_size.width - 25))
        text = f"{cursor} {LABELS[name]:<{label_width}} {state:<11} {required}"
        if selected and options.has_focus:
            # The list supplies the full-row block cursor; suppress inline colors.
            return Content(text)
        style = (
            "$text-muted"
            if name in SEGMENTS and name not in self.draft.segments
            else "$foreground"
        )
        return Content.assemble((text, style))

    def _mark_cursor(self) -> None:
        width = self.query_one("#status-line-settings-content").content_size.width
        label_width = min(23, max(12, width - 25))
        self.query_one("#status-line-settings-heading", NoMarkupStatic).update(
            f"  {'Segment':<{label_width}} {'State':<11} Constraint"
        )
        for options in (
            self.query_one("#status-line-settings-options", OptionList),
            self.query_one("#status-line-settings-actions", OptionList),
        ):
            for option in options.options:
                if option.id is not None:
                    name = str(option.id)
                    options.replace_option_prompt(
                        name,
                        self._row_prompt(
                            name, selected=option == options.highlighted_option
                        ),
                    )

        actions = self.query_one(
            "#status-line-settings-confirmation-actions", OptionList
        )
        for option in actions.options:
            if option.id == "confirm":
                text = (
                    "[Discard edits]"
                    if self._confirmation == "discard"
                    else "[Remove overrides]"
                )
                prompt = (
                    Content(text)
                    if actions.has_focus and actions.highlighted_option == option
                    else Content.assemble((text, "$error"))
                )
                actions.replace_option_prompt("confirm", prompt)

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if event.widget.id in {
            "status-line-settings-options",
            "status-line-settings-actions",
        }:
            self._active_group = event.widget.id
            if self._help_open or self._details_open:
                self._close_inspection(restore=False)
        self._mark_cursor()
        self._update_details()

    def on_descendant_blur(self) -> None:
        self._mark_cursor()

    def _render_rows(self, selected: str | None = None) -> None:
        options = self.query_one("#status-line-settings-options", OptionList)
        position = self._capture_group_position(options)
        selected = selected or (
            str(options.highlighted_option.id)
            if options.highlighted_option
            else "directory"
        )
        rows = [
            Option(self._row_prompt(name), id=name)
            for name in (*self.order, "separator")
        ]
        options.clear_options()
        options.add_options(rows)
        options.highlighted = next(
            i for i, row in enumerate(rows) if row.id == selected
        )
        if position and position.identities:
            self._restore_group_position(position, restore_focus=False)
        self._mark_cursor()
        self._update_details()
        self._update_preview()

    def _update_preview(self) -> None:
        widget = self.query_one("#status-line-settings-preview", NoMarkupStatic)
        widget.update(
            format_status_line(self._example, self.draft, widget.content_size.width)
        )

    def _update_details(self) -> None:
        name = self._selected()
        details = self.query_one("#status-line-settings-details", StatusLineDetails)
        expanded = self._help_open or self._details_open
        details.set_class(expanded, "expanded")
        details.can_focus = expanded
        if not expanded:
            details.scroll_to(y=0, animate=False, immediate=True)
        text = (
            f"{LABELS[name]} · {self._state(name) if name not in {'apply', 'back'} else 'Action'}\n"
            + DESCRIPTIONS[name]
            + "\n"
            + (
                "Off-row positions last only this session; only enabled order is saved."
                if name in SEGMENTS and name not in self.draft.segments
                else "Draft only · Apply changes saves; Ctrl+R restores inheritance."
            )
        )
        if self._details_open:
            if self._feedback:
                text += "\n" + self._feedback
            text += "\n" + "\n".join(
                f"{LABELS[key]}: {value}"
                for key, value in DESCRIPTIONS.items()
                if key in SEGMENTS
            )
        if self._help_open:
            text += (
                "\nUp/Down or j/k: bounded navigation. Tab/Shift+Tab: groups. Enter/Space: cycle configuration (never save)."
                "\n[/] or Alt+Up/Down: move segment; Separator and actions stay fixed."
                "\nEnter on Apply: save changed leaves. Enter on Back: return."
                "\nCtrl+R: remove saved overrides (confirmed). d: details. F1: help."
                "\nEsc: cancel confirmation, help, details, then confirm discard or return."
            )
        self.query_one("#status-line-settings-detail-text", NoMarkupStatic).update(text)
        self.query_one("#status-line-settings-feedback", NoMarkupStatic).update(
            Content.assemble((
                self._feedback,
                "$error"
                if self._feedback.startswith("Failed:")
                else "$primary"
                if self._busy
                else "$text-muted",
            ))
        )
        self._update_hint()

    def _update_hint(self) -> None:
        if self._busy:
            actions = [("Esc", "Wait for save")]
        elif self._confirmation:
            actions = [("Enter", "Choose"), ("Esc", "Cancel")]
        elif self._help_open or self._details_open:
            actions = [("↑↓", "Scroll"), ("Esc", "Back")]
        elif self.snapshot.view_only or self._needs_refresh:
            actions = [("d", "Details"), ("F1", "Help"), ("Esc", "Back")]
        else:
            name = self._selected()
            actions = [
                (
                    "Enter",
                    "Save"
                    if name == "apply"
                    else "Back"
                    if name == "back"
                    else "Cycle",
                )
            ]
            if name in SEGMENTS:
                actions += [("Space", "Cycle"), ("[/]", "Reorder")]
            actions += [("d", "Details"), ("F1", "Help"), ("Esc", "Back")]

        def render(pairs: list[tuple[str, str]]) -> Content:
            return shortcut_hint(
                "  ".join(f"{shortcut(key)} {label}" for key, label in pairs)
            )

        widget = self.query_one("#status-line-settings-hint", NoMarkupStatic)
        if (
            len(render(actions).plain) > widget.content_size.width
            and len(actions) > MIN_SHORTCUTS_WITH_OVERFLOW
        ):
            actions = [actions[0], ("F1", "Help"), actions[-1]]
        widget.update(render(actions))
        commands = {
            "Esc": "close",
            "d": "details",
            "F1": "help",
            "Space": "cycle",
            "Enter": "activate_focused",
        }
        hint = self.query_one("#status-line-settings-hint", StatusLineHints)
        hint.targets = []
        offset = 0
        for key, label in actions:
            width = cell_len(render([(key, label)]).plain)
            if key in commands:
                hint.targets.append((offset, offset + width, commands[key]))
            offset += width + 2

    def action_activate_focused(self) -> None:
        if isinstance(self.focused, OptionList):
            self.focused.action_select()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_list.id in {
            "status-line-settings-options",
            "status-line-settings-actions",
        }:
            if event.option_list.has_focus:
                self._active_group = event.option_list.id
            self._mark_cursor()
            self._update_details()
        elif event.option_list.id == "status-line-settings-confirmation-actions":
            self._mark_cursor()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option_list.id == "status-line-settings-confirmation-actions":
            pending = self._confirmation
            self._cancel_confirmation()
            if event.option.id == "confirm":
                if pending == "discard":
                    self._dismiss()
                elif pending == "reset" and self._can_edit():
                    self._busy = True
                    self._update_hint()
                    self.run_worker(
                        self._save(self._reset_changes()), group="status-line-save"
                    )
            return
        if self._busy or self._confirmation or self._help_open or self._details_open:
            return
        self._active_group = event.option_list.id or self._active_group
        if event.option.id == "apply":
            self.action_apply()
        elif event.option.id == "back":
            self.action_close()
        else:
            self.action_cycle()

    def _can_edit(self) -> bool:
        if self._busy or self._confirmation or self._help_open or self._details_open:
            return False
        if self.snapshot.view_only:
            self._feedback = VIEW_ONLY
            self._update_details()
            return False
        if self._needs_refresh:
            self._feedback = (
                "Close and reopen Settings to reconcile current state before saving."
            )
            self._update_details()
            return False
        return True

    def action_cycle(self) -> None:
        if not self._can_edit():
            return
        name = self._selected()
        if name == "directory":
            self.draft.directory_style = (
                "path" if self.draft.directory_style == "name" else "name"
            )
        elif name == "context":
            self.draft.context_style = (
                "tokens"
                if self.draft.context_style == "tokens-percent"
                else "tokens-percent"
            )
        elif name == "separator":
            self.draft.separator = "space" if self.draft.separator == "pipe" else "pipe"
        elif name in SEGMENTS:
            enabled = set(self.draft.segments)
            enabled.symmetric_difference_update({name})
            self.draft.segments = [key for key in self.order if key in enabled]
        else:
            return  # Space on actions never saves or navigates.
        self._render_rows(name)

    def _move(self, delta: int) -> None:
        if not self._can_edit() or (name := self._selected()) not in self.order:
            return
        index = self.order.index(name)
        target = index + delta
        if 0 <= target < len(self.order):
            self.order[index], self.order[target] = (
                self.order[target],
                self.order[index],
            )
            self.draft.segments = [
                key for key in self.order if key in self.draft.segments
            ]
            self._render_rows(name)

    def action_move_up(self) -> None:
        self._move(-1)

    def action_move_down(self) -> None:
        self._move(1)

    def action_apply(self) -> None:
        if not self._can_edit():
            return
        before = self.opening.model_dump(mode="json", by_alias=False)
        changes = {
            f"status_line.{key}": value
            for key, value in self.draft.model_dump(mode="json", by_alias=False).items()
            if value != before[key]
        }
        if not changes:
            self._feedback = "Info: No changes to save."
            self._update_details()
            return
        self._busy = True
        self._update_hint()
        self.run_worker(self._save(changes), group="status-line-save")

    def _reset_changes(self) -> dict[str, JsonValue | None]:
        return {
            field.path: None
            for field in self.snapshot.fields
            if field.path in STATUS_LINE_PATHS and field.saved_explicit
        }

    def action_remove_override(self) -> None:
        if not self._can_edit():
            return
        if not self._reset_changes():
            self._feedback = "Info: Status line has no user override."
            self._update_details()
            return
        self._confirm(
            "reset",
            "Remove saved status line overrides? Restores inheritance, not necessarily defaults. Unsaved draft changes will be replaced. Cancel preserves the draft.",
        )

    def _confirm(self, kind: str, text: str) -> None:
        if (position := self._capture_group_position()) is not None:
            self._openers["confirmation"] = position
        self._confirmation = kind
        self.query_one("#status-line-settings-options").display = False
        self.query_one("#status-line-settings-actions").display = False
        self.query_one("#status-line-settings-confirmation").display = True
        self.query_one(
            "#status-line-settings-confirmation-text", NoMarkupStatic
        ).update(text)
        actions = self.query_one(
            "#status-line-settings-confirmation-actions", OptionList
        )
        actions.clear_options()
        actions.add_options([
            Option(Content("[Cancel]"), id="cancel"),
            Option(
                Content.assemble((
                    "[Discard edits]" if kind == "discard" else "[Remove overrides]",
                    "$error",
                )),
                id="confirm",
            ),
        ])
        actions.highlighted = 0
        actions.focus()
        self._update_details()

    def _cancel_confirmation(self) -> None:
        self._confirmation = None
        self.query_one("#status-line-settings-confirmation").display = False
        self.query_one("#status-line-settings-options").display = True
        self.query_one("#status-line-settings-actions").display = True
        self._restore_group_position(self._openers.pop("confirmation", None))
        self._update_details()

    async def _save(self, changes: Mapping[str, JsonValue | None]) -> None:
        self._feedback = "Running: Saving Status line"
        self._update_details()
        try:
            outcome = await self.service.save(changes, self.snapshot.user_revision)
            if outcome.persistence == "not_saved":
                self._needs_refresh = outcome.error == "conflict"
                self._feedback = f"Failed: {outcome.error or 'write failed'}."
                if self._needs_refresh:
                    self._feedback += " Close and reopen Settings before retrying."
                return
            feedback = f"{chrome_glyph('success')} Saved: Status line updated"
            warnings = []
            if outcome.shadowed:
                warnings.append(
                    "shadowed by a higher layer: " + ", ".join(outcome.shadowed)
                )
            if outcome.persistence == "durability_uncertain":
                warnings.append("durability uncertain")
            if outcome.application == "failed":
                warnings.append(
                    "application failed"
                    + (f": {outcome.error}" if outcome.error else "")
                )
            if outcome.snapshot is None:
                warnings.append("current state unknown; reopen Settings before editing")
            if warnings:
                feedback = "! Warning: Settings saved; " + "; ".join(warnings)
            self.dismiss(
                StatusLineSettingsResult(
                    outcome.snapshot,
                    outcome.snapshot.user_revision if outcome.snapshot else None,
                    feedback,
                    outcome.snapshot is None,
                )
            )
        except Exception as exc:
            self._feedback = f"Failed: {exc}"
        finally:
            self._busy = False
            if self.is_mounted:
                self._update_details()

    def _close_inspection(self, *, restore: bool = True) -> None:
        self._help_open = False
        self._details_open = False
        self._update_details()
        opener = self._openers.pop("inspection", None)
        self._openers.pop("help", None)
        if restore:
            self._restore_group_position(opener)

    def _focus_inspection(self) -> None:
        if self._help_open or self._details_open:
            self.query_one("#status-line-settings-details").focus()
        else:
            self._restore_group_position(self._openers.pop("inspection", None))

    def action_details(self) -> None:
        if not self._busy and not self._confirmation and not self._help_open:
            if not self._details_open:
                if (position := self._capture_group_position()) is not None:
                    self._openers["inspection"] = position
            self._details_open = not self._details_open
            self._update_details()
            self._focus_inspection()

    def action_help(self) -> None:
        if not self._busy and not self._confirmation:
            if not self._help_open:
                if (position := self._capture_group_position()) is not None:
                    self._openers["help"] = position
                    if not self._details_open:
                        self._openers["inspection"] = position
            self._help_open = not self._help_open
            self._update_details()
            if not self._help_open:
                self._restore_group_position(self._openers.pop("help", None))
                if not self._details_open:
                    self._openers.pop("inspection", None)
            else:
                self._focus_inspection()

    def _dismiss(self) -> None:
        self.dismiss(
            StatusLineSettingsResult(
                self.snapshot,
                self.snapshot.user_revision,
                self._feedback,
                self._needs_refresh,
            )
        )

    def action_close(self) -> None:
        if self._busy:
            return
        if self._confirmation:
            self._cancel_confirmation()
        elif self._help_open:
            self.action_help()
        elif self._details_open:
            self.action_details()
        elif self.dirty:
            self._confirm(
                "discard",
                "Discard status line edits? Unsaved changes will be lost. Cancel preserves the current draft.",
            )
        else:
            self._dismiss()
