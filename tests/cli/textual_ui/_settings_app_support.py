"""Shared test-support surface for the settings app suite (W10, stage 19e).

The definitions here are the shared module-level helper surface of
``tests/cli/textual_ui/test_settings_app.py``, relocated AST-verbatim in
the 19e phase-(i) helper-relocation commit: the settings service double
(``FakeService``), the settings screen harness (``Harness``), and the
shared helpers ``focus_action``, ``navigation_payload`` and
``pointer_shortcut_offset``. No test bodies, fixtures, or marks live
here; the test population stays in ``test_settings_app.py`` until the
phase-(ii) concern moves. Importers: ``test_settings_app.py`` itself
plus the seam file ``tests/snapshots/test_ui_snapshot_settings.py``
(which imports ``FakeService`` and ``focus_action``). The single-concern
harnesses ``StatusLineHarness`` (status line concern) and
``FakeAuthorityResource`` / ``ShellPermissionsHarness`` (shell
permissions block) stay in ``test_settings_app.py`` and move with their
concerns in phase (ii).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal, cast

from textual.app import App
from textual.widgets import OptionList

from chartreux.app_server._resources import _inventory_item_states
from chartreux.app_server.protocol import (
    InventoryItemStateWire,
    SettingDescriptorWire,
    SettingLeafWire,
    SettingsReadResponse,
)
from chartreux.cli.textual_ui.screens.settings import (
    SettingsHints,
    SettingsScreen,
    _draft_values,
)
from chartreux.core.config.models import StatusLineConfig
from chartreux.core.config.settings_catalog import (
    DEFERRED_SETTINGS,
    EDITABLE_BY_PATH,
    LINK_SETTINGS,
    TASK_DENYLIST_SETTING,
    VISIBLE_SETTINGS,
)
from chartreux.ui.settings_service import SettingsSaveOutcome, SettingsService

# Real enum defaults for the fake leaves so displays stay representative.
FAKE_ENUM_DEFAULTS = {
    "agent.identity": "coder",
    "agent.workflow": "minimal",
    "agent.orchestration_strategy": "none",
    "compaction_prompt_id": "compact",
    "status_line.directory_style": "name",
    "status_line.context_style": "tokens-percent",
    "status_line.separator": "pipe",
}

FAKE_INVENTORIES: dict[Literal["tools", "skills", "identities"], list[str]] = {
    "tools": ["bash", "read_file"],
    "skills": ["review", "search"],
    "identities": ["coder", "researcher"],
}


def focus_action(screen: SettingsScreen, identifier: str) -> OptionList:
    actions = screen.query_one("#settings-actions", OptionList)
    actions.highlighted = next(
        i for i, row in enumerate(actions.options) if row.id == identifier
    )
    actions.focus()
    return actions


def navigation_payload(screen: SettingsScreen) -> object:
    return (
        _draft_values(screen._list_draft),
        screen._enum_draft,
        {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        },
        screen.snapshot.model_dump(),
    )


class FakeService:
    def __init__(self) -> None:
        inventory_states: dict[
            Literal["tools", "skills", "identities"], dict[str, InventoryItemStateWire]
        ] = {
            category: {
                name: InventoryItemStateWire(
                    effective=True, default_effective=True, pattern_driven=False
                )
                for name in names
            }
            for category, names in cast(
                dict[Literal["tools", "skills", "identities"], list[str]],
                FAKE_INVENTORIES,
            ).items()
        }
        self.snapshot = SettingsReadResponse(
            catalog=[
                SettingDescriptorWire.model_validate(asdict(item))
                for item in (*VISIBLE_SETTINGS, *DEFERRED_SETTINGS, *LINK_SETTINGS)
            ],
            fields=[
                SettingLeafWire(
                    path=item.path,
                    effective_value=StatusLineConfig().model_dump(mode="json")[
                        item.path.split(".")[1]
                    ]
                    if item.path.startswith("status_line.")
                    else FAKE_ENUM_DEFAULTS.get(item.path, "")
                    if item.kind == "enum"
                    else False
                    if item.kind == "bool"
                    else 2
                    if item.kind in {"int", "float"}
                    else []
                    if item.kind == "list"
                    else "",
                    origin="default",
                    saved_explicit=False,
                )
                for item in EDITABLE_BY_PATH.values()
            ]
            + [
                SettingLeafWire(
                    path=TASK_DENYLIST_SETTING.path,
                    effective_value=[],
                    origin="default",
                    saved_explicit=False,
                )
            ],
            inventories=dict(FAKE_INVENTORIES),
            inventory_states=inventory_states,
            user_layer="user",
            user_revision="revision",
        )
        self.saved: list[dict[str, object]] = []
        self.revisions: list[str | None] = []
        self.outcome: SettingsSaveOutcome | None = None

    async def read(self) -> SettingsReadResponse:
        return self.snapshot

    async def save(
        self, changed_leaves: dict[str, object], expected_revision: str | None
    ) -> SettingsSaveOutcome:
        self.saved.append(dict(changed_leaves))
        self.revisions.append(expected_revision)
        if self.outcome:
            return self.outcome
        fields = []
        for field in self.snapshot.fields:
            if field.path in changed_leaves:
                value = changed_leaves[field.path]
                field = field.model_copy(
                    update={
                        "effective_value": value
                        if value is not None
                        else StatusLineConfig().model_dump(mode="json")[
                            field.path.split(".")[1]
                        ]
                        if field.path.startswith("status_line.")
                        else ([] if isinstance(field.effective_value, list) else False),
                        "saved_explicit": value is not None,
                        "saved_value": value,
                        "origin": "user" if value is not None else "default",
                    }
                )
            fields.append(field)
        self.snapshot = self.snapshot.model_copy(
            update={"fields": fields, "user_revision": f"revision-{len(self.saved)}"}
        )
        self.snapshot.inventory_states = _inventory_item_states(
            self.snapshot.inventories,
            {field.path: field.effective_value for field in fields},
        )
        return SettingsSaveOutcome("saved", "applied", self.snapshot)


class Harness(App[None]):
    def __init__(self, service: FakeService) -> None:
        super().__init__()
        self.service = service

    def on_mount(self) -> None:
        self.push_screen(
            SettingsScreen(cast(SettingsService, self.service), self.service.snapshot)
        )


def pointer_shortcut_offset(
    screen: SettingsScreen, identifier: str, action: str
) -> tuple[int, int]:
    widget = screen.query_one(identifier, SettingsHints)
    for y in range(widget.region.height):
        for x in range(widget.region.width):
            if widget.pointer_action_at(x, y) == action:
                return x, y
    raise AssertionError(f"No visible pointer target for {action} in {identifier}")
