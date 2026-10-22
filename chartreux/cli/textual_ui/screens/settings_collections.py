"""Draft/collection controller for the settings screen (W5 11.4b, stage 11)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from textual.widgets import OptionList
from textual.widgets.selection_list import Selection

from chartreux.app_server.protocol import SettingDescriptorWire
from chartreux.app_server.protocol.config import TASK_IDENTITY_DENYLIST_PATH
from chartreux.cli.textual_ui.widgets.settings_lists import (
    SettingsChecklist,
    SettingsOptionList,
    _draft_values,
    _DraftEntry,
    inventory_item_state,
    toggle_inventory_name,
)

if TYPE_CHECKING:
    from chartreux.cli.textual_ui.screens.settings import SettingsScreen

InventoryCategory = Literal["tools", "skills", "identities"]
_ListCategory = Literal["tools", "skills", "agents"]


def _list_state_category(category: InventoryCategory) -> _ListCategory:
    """The identity roster's denylist state uses the tools-shaped matcher.

    A denylist-only list behaves exactly like the tools inventory with an
    always-empty allow list: effective is ``not denied`` and pattern entries
    come from the single denylist side.
    """
    if category == "identities":
        return "tools"
    return cast(_ListCategory, category)


class SettingsCollectionsController:
    """Owns the settings screen's draft tokens and collection edit surface.

    Plain object (no mixins): the screen constructs it in ``__init__`` and is
    its late-bound host — every collaboration is ``self._screen._method(...)``
    at call time, never a constructor-captured bound method. The moved state
    (``_next_draft_token``/``_list_draft``/``_enum_draft``/``_inventory_draft``)
    lives here; the screen forwards the draft fields through same-name
    properties so direct-attribute consumers (tests, widgets) see no change.
    Surface-global state (``_busy``/``_expanded``/``_positions``/``_openers``/
    ``snapshot``/``catalog``/``fields``) stays on the screen and is read/written
    through the host; ``_list_editor`` is owned by the editor controller and
    reached through the screen's forwarding property (``collapse`` clears it).

    The draft token machinery is fully constructed in ``__init__`` with zero
    mount/app dependencies (bare-instance gate
    ``test_draft_tokens_unique_monotonic_and_reconciled_by_occurrence``).
    """

    def __init__(self, screen: SettingsScreen) -> None:
        self._screen = screen
        self._next_draft_token = 0
        self._list_draft: list[_DraftEntry] | None = None
        self._enum_draft: str | None = None
        self._inventory_draft: dict[str, list[_DraftEntry]] | None = None

    def new_draft_entry(self, value: str) -> _DraftEntry:
        entry = _DraftEntry(self._next_draft_token, value)
        self._next_draft_token += 1
        return entry

    def make_draft(self, values: list[str]) -> list[_DraftEntry]:
        return [self.new_draft_entry(value) for value in values]

    def reconcile_draft(
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
                else self.new_draft_entry(value)
            )
        return result

    def draft_is_dirty(self, item: SettingDescriptorWire) -> bool:
        field = self._screen.fields.get(item.path)
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
            enabled, disabled = self.inventory_values(item.inventory or "tools")
            return (
                _draft_values(self._inventory_draft["enabled"]) != enabled
                or _draft_values(self._inventory_draft["disabled"]) != disabled
            )
        return False

    def select_enum_draft(self) -> None:
        if not self._screen._can_mutate():
            return
        choices = self._screen.query_one("#settings-choices", OptionList)
        option = choices.highlighted_option
        if choices.has_focus and option and str(option.id).startswith("choice:"):
            self._enum_draft = str(option.id).split(":", 2)[2]
            self._screen._refresh_options()

    def accept_enum_draft(self, item: SettingDescriptorWire) -> None:
        if self._enum_draft not in item.choices:
            return
        if self.draft_is_dirty(item):
            self._screen._start_write(item, {item.path: self._enum_draft})
        else:
            self.collapse(restore_opener=False)

    def apply_collection(self, item: SettingDescriptorWire) -> None:
        if not self._screen._can_mutate() or self._screen._expanded != item.path:
            return
        self._screen._capture_group_position()
        if self._inventory_draft is not None:
            self.save_inventory(item)
        else:
            self.save_list(item)

    def save_list(self, item: SettingDescriptorWire) -> None:
        if not self._screen._can_mutate():
            return
        values = _draft_values(self._list_draft)
        saved = self._screen.fields.get(item.path)
        original = saved.effective_value if saved is not None else []
        if values != original:
            self._screen._start_write(item, {item.path: values})
        else:
            self.collapse(restore_opener=False)

    def inventory_values(self, category: str) -> tuple[list[str], list[str]]:
        if category == "identities":
            field = self._screen.fields.get(TASK_IDENTITY_DENYLIST_PATH)
            value = field.saved_value if field and field.saved_explicit else None
            return (
                [],
                [entry for entry in value if isinstance(entry, str)]
                if isinstance(value, list)
                else [],
            )

        def values(path: str) -> list[str]:
            field = self._screen.fields.get(path)
            value = field.saved_value if field and field.saved_explicit else None
            return (
                [entry for entry in value if isinstance(entry, str)]
                if isinstance(value, list)
                else []
            )

        return values(f"enabled_{category}"), values(f"disabled_{category}")

    def inventory_item_state(self, name: str) -> tuple[bool, bool]:
        draft = self._inventory_draft
        if draft is None:
            return False, False
        item = next(
            item for item in self._screen.catalog if item.path == self._screen._expanded
        )
        return inventory_item_state(
            name,
            _draft_values(draft["enabled"]),
            _draft_values(draft["disabled"]),
            category=_list_state_category(item.inventory or "tools"),
        )

    def pattern_driven(self, category: InventoryCategory, name: str) -> bool:
        return self.inventory_item_state(name)[1]

    def effective_inventory_item(self, category: InventoryCategory, name: str) -> bool:
        return self.inventory_item_state(name)[0]

    def render_checklist(self, *, highlight: str | None = None) -> None:
        item = next(
            entry
            for entry in self._screen.catalog
            if entry.path == self._screen._expanded
        )
        category = item.inventory or "tools"
        draft = self._inventory_draft
        if draft is None:
            return
        enabled, disabled = draft["enabled"], draft["disabled"]
        names = self._screen.snapshot.inventories.get(category, [])
        checklist = self._screen.query_one("#settings-checklist", SettingsChecklist)
        position = self._screen._position_before_rebuild(checklist)
        checklist.highlighted = None
        checklist.clear_options()
        checklist.add_options(
            [
                Selection(
                    f"  {name}"
                    + (
                        " (pattern-controlled)"
                        if self.pattern_driven(category, name)
                        else ""
                    ),
                    name,
                    self.effective_inventory_item(category, name),
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
        self._screen._restore_group_position(position, restore_focus=False)
        if highlight is not None:
            checklist.highlighted = next(
                (
                    index
                    for index in range(checklist.option_count)
                    if checklist.get_option_at_index(index).value == highlight
                ),
                checklist.highlighted,
            )
            self._screen._capture_group_position(checklist)
        self._screen._update_help()

    def toggle_inventory(self, name: str) -> bool:
        if not self._screen._can_mutate():
            return False
        draft = self._inventory_draft
        if draft is None:
            return False
        item = next(
            item for item in self._screen.catalog if item.path == self._screen._expanded
        )
        category = item.inventory or "tools"
        if self.pattern_driven(category, name):
            return False
        is_enabled = self.effective_inventory_item(category, name)
        enabled, disabled = toggle_inventory_name(
            name,
            _draft_values(draft["enabled"]),
            _draft_values(draft["disabled"]),
            is_enabled=is_enabled,
            category=_list_state_category(category),
        )
        draft["enabled"] = self.reconcile_draft(draft["enabled"], enabled)
        draft["disabled"] = self.reconcile_draft(draft["disabled"], disabled)
        return True

    def open_checklist(self, item: SettingDescriptorWire) -> None:
        checklist = self._screen.query_one("#settings-checklist", SettingsChecklist)
        enabled, disabled = self.inventory_values(item.inventory or "tools")
        self._inventory_draft = {
            "enabled": self.make_draft(enabled),
            "disabled": self.make_draft(disabled),
        }
        self.render_checklist()
        options = self._screen.query_one("#settings-options", SettingsOptionList)
        options.editing = True
        self._screen._refresh_collection()
        checklist.focus()
        self._screen._update_hint()
        self._screen._update_help()

    def save_inventory(self, item: SettingDescriptorWire) -> None:
        if not self._screen._can_mutate():
            return
        draft = self._inventory_draft
        if draft is None:
            return
        category = item.inventory or "tools"
        saved_enabled, saved_disabled = self.inventory_values(category)
        if category == "identities":
            denylist = _draft_values(draft["disabled"])
            if denylist != saved_disabled:
                self._screen._start_write(item, {TASK_IDENTITY_DENYLIST_PATH: denylist})
            else:
                self.collapse(restore_opener=False)
            return
        changes = {
            f"{side}_{category}": _draft_values(draft[side])
            for side, saved in (
                ("enabled", saved_enabled),
                ("disabled", saved_disabled),
            )
            if _draft_values(draft[side]) != saved
        }
        if changes:
            self._screen._start_write(item, changes)
        else:
            self.collapse(restore_opener=False)

    def expand(self, item: SettingDescriptorWire) -> None:
        self._screen._remember_opener("expansion")
        self._screen._expanded = item.path
        field = self._screen.fields.get(item.path)
        value = field.effective_value if field else None
        if item.kind == "enum":
            self._enum_draft = str(value) if value is not None else None
        if item.kind == "list":
            self._list_draft = self.make_draft(
                [entry for entry in value if isinstance(entry, str)]
                if isinstance(value, list)
                else []
            )
        if item.control == "toggle_inventory":
            self.open_checklist(item)
            return
        options = self._screen.query_one("#settings-options", SettingsOptionList)
        options.editing = True
        body_id = "settings-entries" if item.kind == "list" else "settings-choices"
        remembered = self._screen._positions.get((body_id, item.path))
        self._screen._refresh_options()
        body = self._screen.query_one(f"#{body_id}", OptionList)
        ids = [row.id for row in body.options]
        target = f"choice:{item.path}:{value}"
        if remembered is None and target in ids:
            body.highlighted = ids.index(target)
        body.focus(scroll_visible=remembered is None)
        if remembered is None:
            self._screen.call_after_refresh(body.scroll_to_highlight)

    def collapse(self, *, restore_opener: bool = True) -> None:
        for group in self._screen._visible_focus_groups():
            self._screen._capture_group_position(group)
        path = self._screen._expanded
        catalog_position = self._screen._positions.get(("settings-options", "catalog"))
        opener = self._screen._openers.pop("expansion", None)
        self._screen._expanded = None
        self._list_draft = None
        self._enum_draft = None
        self._inventory_draft = None
        self._screen._list_editor = None
        self._screen._collection_visibility()
        options = self._screen.query_one("#settings-options", SettingsOptionList)
        options.focus()
        options.editing = False
        if path is not None:
            self._screen._refresh_options()
        self._screen._restore_group_position(
            opener if restore_opener else catalog_position
        )
