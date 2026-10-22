"""Confirmation controller for the settings screen (W5 11.4d, stage 11)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.content import Content
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from chartreux.app_server.protocol import STATUS_LINE_PATHS, SettingDescriptorWire
from chartreux.app_server.protocol.config import TASK_IDENTITY_DENYLIST_PATH
from chartreux.cli.textual_ui.widgets.settings_lists import (
    SettingsChecklist,
    SettingsOptionList,
)
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

if TYPE_CHECKING:
    from chartreux.cli.textual_ui.screens.settings import SettingsScreen, _GroupPosition


class SettingsConfirmationController:
    """Owns the settings screen's confirmation surface.

    Plain object (no mixins): the screen constructs it in ``__init__`` and is
    its late-bound host — every collaboration is ``self._screen._method(...)``
    at call time, never a constructor-captured bound method. The moved state
    (``_confirmation``/``_confirmation_return_focus``/``_pending_navigation``)
    lives here; the screen forwards ``_confirmation`` (getter) and
    ``_pending_navigation`` (getter+setter — the screen-resident navigation
    flows ``on_option_list_option_selected``/``action_close`` write it) through
    same-name properties. ``_confirmation_return_focus`` is internal: the
    post-open overwrite in ``remove_override`` moves with the controller.
    Surface-global state (``_busy``/``_help_open``/``_needs_refresh``/
    ``_error``/``_error_path``/``_editing``/``snapshot``/``catalog``/``fields``/
    ``_openers``) stays on the screen and is reached through the host, as do
    the editor/collections/write delegates (``_cancel_input``/``_collapse``/
    ``_render_checklist``/``_refresh_options``/``_open_setting``/
    ``_start_write``).

    Pinned invariants, moved verbatim: ``dismiss_confirmation``'s apply
    dispatch stays one five-way kind switch over controller-external effects
    (never split per kind); ``remove_override``'s post-``open_confirmation``
    overwrite of ``_confirmation_return_focus`` and its editor-opener capture
    keep their ordering. The 11.3-converted ``SettingsConfirmationList``
    callbacks resolve against the screen's ``_confirmation`` forwarder and
    ``_dismiss_confirmation`` delegate by name — names unchanged.
    """

    def __init__(self, screen: SettingsScreen) -> None:
        self._screen = screen
        self._confirmation: tuple[str, SettingDescriptorWire, Any] | None = None
        self._confirmation_return_focus: _GroupPosition | None = None
        self._pending_navigation: SettingDescriptorWire | None = None

    def open_confirmation(
        self,
        kind: str,
        item: SettingDescriptorWire,
        value: Any,
        message: str,
        action: str,
    ) -> None:
        self._confirmation_return_focus = self._screen._capture_group_position()
        self._confirmation = (kind, item, value)
        self._screen.query_one("#settings-options", SettingsOptionList).display = False
        self._screen._collection_visibility(hidden=True)
        self._screen.query_one("#settings-confirmation-text", NoMarkupStatic).update(
            message
        )
        choices = self._screen.query_one("#settings-confirmation-actions", OptionList)
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
        self._screen.query_one("#settings-confirmation").display = True
        choices.highlighted = 0
        choices.focus()
        self._screen._update_help()
        self._screen._update_hint()

    def dismiss_confirmation(self, *, apply: bool = False) -> None:
        if self._screen._busy:
            return
        if (
            apply
            and (self._screen.snapshot.view_only or self._screen._needs_refresh)
            and self._confirmation is not None
            and self._confirmation[0] != "discard"
        ):
            return
        if apply and self._screen._help_open:
            self._screen._close_help()
        pending = self._confirmation
        self._confirmation = None
        self._screen.query_one("#settings-confirmation").display = False
        options = self._screen.query_one("#settings-options", SettingsOptionList)
        options.display = self._screen._editing is None
        self._screen._collection_visibility(hidden=self._screen._editing is not None)
        self._screen._restore_group_position(self._confirmation_return_focus)
        self._confirmation_return_focus = None
        self._screen._update_help()
        self._screen._update_hint()
        if not apply or pending is None:
            self._pending_navigation = None
            return
        kind, item, value = pending
        if kind == "pattern":
            side, token = value
            if self._screen._inventory_draft is not None:
                self._screen._inventory_draft[side] = [
                    entry
                    for entry in self._screen._inventory_draft[side]
                    if entry.token != token
                ]
                self._screen._render_checklist()
        elif kind == "delete":
            entries = self._screen._list_draft or []
            self._screen._list_draft = [
                entry for entry in entries if entry.token != value
            ]
            self._screen._refresh_options()
        elif kind == "discard":
            self._screen._collapse()
            target = self._pending_navigation
            self._pending_navigation = None
            if target is not None:
                options = self._screen.query_one(
                    "#settings-options", SettingsOptionList
                )
                ids = [row.id for row in options.options]
                if target.path in ids:
                    options.highlighted = ids.index(target.path)
                self._screen._open_setting(target)
        elif kind == "status-reset":
            self._screen._start_write(item, value)
        elif kind == "reset":
            category = item.inventory or "tools"
            changes = (
                {TASK_IDENTITY_DENYLIST_PATH: None}
                if category == "identities"
                else {f"enabled_{category}": None, f"disabled_{category}": None}
            )
            self._screen._start_write(item, changes)
        else:
            self._screen._start_write(item, {item.path: value})

    def delete_item(self) -> None:  # noqa: PLR0911
        if not self._screen._can_mutate() or self._screen._expanded is None:
            return
        checklist = self._screen.query_one("#settings-checklist", SettingsChecklist)
        if self._screen._inventory_draft is not None and checklist.has_focus:
            value = checklist.get_option_at_index(checklist.highlighted or 0).value
            if value.startswith("\x00pattern:"):
                _, side, index = value.rsplit(":", 2)
                item = next(
                    entry
                    for entry in self._screen.catalog
                    if entry.path == self._screen._expanded
                )
                target = next(
                    (
                        entry.value
                        for entry in self._screen._inventory_draft[side]
                        if entry.token == int(index)
                    ),
                    None,
                )
                if target is None:
                    return
                source = (
                    "the task identity denylist"
                    if item.inventory == "identities"
                    else f"{side}_{item.inventory}"
                )
                self.open_confirmation(
                    "pattern",
                    item,
                    (side, int(index)),
                    f"Delete pattern {target} from {source}? Cancel preserves the pattern and current draft.",
                    "Delete pattern",
                )
            return
        options = self._screen.query_one("#settings-entries", OptionList)
        if not options.has_focus:
            return
        option = options.highlighted_option
        identifier = str(option.id) if option and option.id else ""
        if not identifier.startswith(f"list:{self._screen._expanded}:"):
            return
        index = identifier.rsplit(":", 1)[-1]
        if not index.isdigit():
            return
        item = next(
            entry
            for entry in self._screen.catalog
            if entry.path == self._screen._expanded
        )
        target = next(
            (
                entry.value
                for entry in self._screen._list_draft or []
                if entry.token == int(index)
            ),
            None,
        )
        if target is None:
            return
        self.open_confirmation(
            "delete",
            item,
            int(index),
            f"Delete {target} from the {item.label} draft? Apply changes will save the shortened list. Cancel preserves the item and stored list.",
            "Delete item",
        )

    def remove_override(self) -> None:  # noqa: PLR0911
        if (
            self._screen._help_open
            and not self._screen._busy
            and self._confirmation is None
        ):
            self._screen._close_help()
        if (
            self._screen._busy
            or self._screen._help_open
            or self._screen._needs_refresh
            or self._confirmation is not None
        ):
            return
        item = self._screen._current_item()
        if item is not None and item.control == "status_line":
            self._screen._error_path = item.path
            if self._screen.snapshot.view_only:
                self._screen._error = f"{chrome_glyph('error')} User configuration is unavailable (view only)."
                self._screen._update_help()
                return
            changes = {
                path: None
                for path in STATUS_LINE_PATHS
                if path in self._screen.fields
                and self._screen.fields[path].saved_explicit
            }
            if not changes:
                self._screen._error = "Info: Status line has no user override."
                self._screen._update_help()
                return
            self.open_confirmation(
                "status-reset",
                item,
                changes,
                "Remove saved status line overrides? Restores inheritance, not necessarily defaults. Cancel preserves saved values.",
                "Remove overrides",
            )
            return
        if item is not None and item.control == "toggle_inventory":
            if self._screen.snapshot.view_only:
                return
            opener = (
                self._screen._openers.get("editor")
                if self._screen._editing is not None
                else self._screen._capture_group_position()
            )
            if self._screen._editing is not None:
                self._screen._cancel_input()
            self.open_confirmation(
                "reset",
                item,
                None,
                f"Reset {item.label}? Clears the task identity denylist. Cancel preserves the saved denylist and the current draft."
                if item.inventory == "identities"
                else f"Reset {item.label}? Clears enabled and disabled {item.inventory} lists. Cancel preserves both saved lists and the current draft.",
                "Reset lists",
            )
            self._confirmation_return_focus = opener
            return
        if item is None or item.kind in {"link", "deferred"}:
            return
        if self._screen.snapshot.view_only:
            self._screen._error_path = item.path
            self._screen._error = f"{chrome_glyph('error')} User configuration is unavailable (view only)."
            self._screen._update_help()
        elif (
            not (field := self._screen.fields.get(item.path))
            or not field.saved_explicit
        ):
            self._screen._error_path = item.path
            self._screen._error = f"Info: {item.label} has no user override."
            self._screen._update_help()
        else:
            if self._screen._editing is not None:
                self._screen._cancel_input()
            self.open_confirmation(
                "override",
                item,
                None,
                f"Remove saved override for {item.label}? The effective value will fall back to its other sources. Cancel preserves the saved override.",
                "Remove override",
            )
