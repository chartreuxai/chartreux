"""Inline editor controller for the settings screen (W5 11.4c, stage 11)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Input, OptionList

from chartreux.app_server.protocol import SettingDescriptorWire
from chartreux.cli.textual_ui.widgets.settings_lists import (
    SettingsChecklist,
    SettingsOptionList,
    _draft_values,
    _DraftEntry,
    _ListEditorContext,
    parse_setting_value,
)
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

if TYPE_CHECKING:
    from chartreux.cli.textual_ui.screens.settings import SettingsScreen


class SettingsEditorController:
    """Owns the settings screen's inline editor surface.

    Plain object (no mixins): the screen constructs it in ``__init__`` and is
    its late-bound host — every collaboration is ``self._screen._method(...)``
    at call time, never a constructor-captured bound method. The moved state
    (``_editing``/``_list_editor``) lives here; the screen forwards both
    through same-name getter properties so direct-attribute consumers (tests,
    the screen's own hint/guard surface) see no change. Surface-global state
    (``_busy``/``_expanded``/``_openers``/``catalog``/``fields``) stays on the
    screen and is reached through the host, as do the collections and write
    delegates (``_new_draft_entry``/``_render_checklist``/``_start_write``).

    ``_ListEditorContext`` imports from ``widgets/settings_lists`` (it moved
    there with this controller): importing it from ``screens.settings`` would
    create a controller⇄screen cycle.

    Pinned invariant, moved verbatim: ``submit``'s add-pattern branch cancels
    the editor through the screen's ``_cancel_input`` delegate *before*
    re-rendering the checklist with the reveal callback (ordering + patch
    seam).
    """

    def __init__(self, screen: SettingsScreen) -> None:
        self._screen = screen
        self._list_editor: _ListEditorContext | None = None
        self._editing: str | None = None

    def begin_list_input(self, item: SettingDescriptorWire, index: str) -> None:
        entries = self._screen._list_draft or []
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
        self.show_editor(item, value, label)

    def begin_input(self, item: SettingDescriptorWire) -> None:
        field = self._screen.fields.get(item.path)
        value = (
            ""
            if field is None or field.effective_value is None
            else str(field.effective_value)
        )
        self.show_editor(item, value, item.label)

    def show_editor(self, item: SettingDescriptorWire, value: str, label: str) -> None:
        self._screen._remember_opener("editor")
        self._editing = item.path
        self._screen.query_one("#settings-options", SettingsOptionList).editing = True
        self._screen.query_one("#settings-options", SettingsOptionList).display = False
        self._screen._collection_visibility(hidden=True)
        self._screen.query_one("#settings-editor-label", NoMarkupStatic).update(
            f"{label} — Enter adds to draft; Apply changes saves; Esc cancels"
            if self._list_editor is not None
            else f"{label} — Enter saves to user settings immediately; Esc cancels"
        )
        self._screen.query_one("#settings-editor-error", NoMarkupStatic).update("")
        editor = self._screen.query_one("#settings-input", Input)
        editor.value = value
        editor.remove_class("-invalid")
        self._screen.query_one("#settings-editor").display = True
        editor.focus()
        self._screen._update_hint()

    def cancel_input(self) -> None:
        list_input = self._list_editor is not None
        self._editing = None
        self._screen.query_one("#settings-editor").display = False
        self._screen.query_one("#settings-editor-error", NoMarkupStatic).update("")
        self._screen.query_one("#settings-input", Input).remove_class("-invalid")
        self._screen._collection_visibility()
        checklist = self._screen.query_one("#settings-checklist", SettingsChecklist)
        if list_input and self._screen._inventory_draft is not None:
            self._screen.query_one(
                "#settings-options", SettingsOptionList
            ).display = True
            checklist.focus()
        else:
            options = self._screen.query_one("#settings-options", SettingsOptionList)
            options.display = True
            options.editing = self._screen._expanded is not None
            if list_input:
                self._screen.query_one("#settings-entries").focus()
            else:
                options.focus()
        self._list_editor = None
        self._screen._restore_group_position(self._screen._openers.pop("editor", None))
        self._screen._update_hint()

    def submit(self, event: Input.Submitted) -> None:  # noqa: PLR0911
        if self._editing is None or not self._screen._can_mutate(editor=True):
            return
        item = next(
            entry for entry in self._screen.catalog if entry.path == self._editing
        )
        if self._list_editor is not None:
            if self._list_editor.mode == "add-pattern":
                text = event.value.strip()
                draft = self._screen._inventory_draft
                if draft is None or not text:
                    self._screen._cancel_input()
                    return
                side = "enabled" if draft["enabled"] else "disabled"
                if text not in _draft_values(draft[side]):
                    draft[side].append(self._screen._new_draft_entry(text))
                entry = next(entry for entry in draft[side] if entry.value == text)
                self._screen._cancel_input()
                self._screen._render_checklist(
                    highlight=f"\x00pattern:{side}:{entry.token}"
                )
                checklist = self._screen.query_one(
                    "#settings-checklist", SettingsChecklist
                )

                def reveal_added_pattern() -> None:
                    checklist.scroll_to_highlight()
                    self._screen._capture_group_position(checklist)

                checklist.call_after_refresh(reveal_added_pattern)
                return
            values = list(self._screen._list_draft or [])
            text = event.value.strip()
            if not text:
                event.input.add_class("-invalid")
                self._screen.query_one("#settings-editor-error", NoMarkupStatic).update(
                    "Error: Enter a nonempty item; use Ctrl+D to delete an existing item."
                )
                return
            token = self._list_editor.token
            if self._list_editor.mode == "edit":
                position = next(
                    (i for i, entry in enumerate(values) if entry.token == token), None
                )
                if position is None:
                    self._screen._cancel_input()
                    return
                entry = _DraftEntry(values[position].token, text)
                values[position] = entry
            else:
                entry = self._screen._new_draft_entry(text)
                values.append(entry)
            self._screen._cancel_input()
            self._screen._list_draft = values
            self._screen._refresh_options()
            options = self._screen.query_one("#settings-entries", OptionList)
            ids = [row.id for row in options.options]
            target = f"list:{item.path}:{entry.token}"
            if target in ids:
                options.highlighted = ids.index(target)
            return
        try:
            value = parse_setting_value(item, event.value)
        except ValueError as exc:
            event.input.add_class("-invalid")
            self._screen.query_one("#settings-editor-error", NoMarkupStatic).update(
                f"Error: {exc}"
            )
            return
        self._screen._cancel_input()
        self._screen._start_write(item, {item.path: value})

    def value_changed(self, event: Input.Changed) -> None:
        if self._editing is None or not event.input.has_class("-invalid"):
            return
        item = next(
            entry for entry in self._screen.catalog if entry.path == self._editing
        )
        if self._list_editor is not None and item.kind == "list":
            if not event.value.strip():
                return
            event.input.remove_class("-invalid")
            self._screen.query_one("#settings-editor-error", NoMarkupStatic).update("")
            return
        try:
            parse_setting_value(item, event.value)
        except ValueError:
            return
        event.input.remove_class("-invalid")
        self._screen.query_one("#settings-editor-error", NoMarkupStatic).update("")
