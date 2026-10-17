"""Structural types and value tables for the provider workbench."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from textual.widgets import SelectionList

CURSOR_GUTTER = 2


class WorkbenchView(StrEnum):
    """The one authoritative view state used by the workbench."""

    PROVIDERS = "providers"
    CHOOSE = "choose"
    CONNECTION = "connection"
    ACTIONS = "actions"
    MODELS = "models"
    CATALOG = "catalog"
    GLOBAL_CATALOG = "catalog"
    DEPLOYMENTS = "deployments"
    DETAIL = "detail"
    MODEL_EDITOR = "detail"
    PROVIDER_MODELS = "models"
    EDITOR = "editor"
    PROTOCOL = "protocol"
    PICKER = "picker"
    ACTIVE_PICKER = "picker"
    PRESETS = "presets"
    PRESET_EDITOR = "preset-editor"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class _GroupPosition:
    view: WorkbenchView
    group: str
    context: tuple[str, ...]
    identities: tuple[str, ...]
    ordinal: int
    scroll_x: float
    scroll_y: float
    cursor_position: int | None = None


@dataclass
class NavigationFrame:
    """State needed to return to the exact opener after a nested view."""

    view: WorkbenchView
    focus_id: str | None = None
    selected_id: str | None = None
    scroll: int = 0
    filter_text: str = ""
    model_filter: str | None = None
    detail_cursor: str | None = None
    stage: str | None = None
    provider_id: str | None = None
    context: tuple[str, ...] = ()
    position: _GroupPosition | None = None


VIEW_GROUPS: dict[WorkbenchView, tuple[str, ...]] = {
    WorkbenchView.PROVIDERS: ("providers", "root-actions"),
    WorkbenchView.CHOOSE: ("choose",),
    WorkbenchView.ACTIONS: ("actions", "provider-operations"),
    WorkbenchView.CONNECTION: ("actions", "connection-actions"),
    WorkbenchView.MODELS: ("models", "models-actions"),
    WorkbenchView.CATALOG: ("catalog", "catalog-filter", "catalog-actions"),
    WorkbenchView.DEPLOYMENTS: ("picker",),
    WorkbenchView.DETAIL: ("detail-fields", "detail-actions"),
    WorkbenchView.EDITOR: ("editor",),
    WorkbenchView.PROTOCOL: ("protocol",),
    WorkbenchView.PICKER: ("picker",),
    WorkbenchView.PRESETS: ("presets", "presets-actions"),
    WorkbenchView.PRESET_EDITOR: ("preset-editor", "preset-editor-actions"),
    WorkbenchView.CONFIRM: ("confirm",),
}

VIEW_CONTROLS: tuple[str, ...] = tuple(
    dict.fromkeys(name for groups in VIEW_GROUPS.values() for name in groups)
) + ("detail", "connection-form")

UNDERLAY_CONTROLS: tuple[str, ...] = (
    "title",
    "pending-action",
    "filter",
    "count",
    "help",
    "hint",
    "field-error",
) + tuple(name for name in VIEW_CONTROLS if name != "confirm")

FIELDS: dict[str, str] = {
    "base": "api_base",
    "style": "api_style",
    "env": "api_key_env_var",
}

DESCRIPTIONS: dict[str, str] = {
    "name": "Unique provider name for this new connection. Enter accepts it into the draft.",
    "base": "API base URL used for listing and inference. Enter accepts into the draft.",
    "style": "API style: openai, openai-responses, or anthropic.",
    "env": "Environment variable name for this provider's credential (not a key).",
    "key": "Enter saves the masked key separately; saved keys are not undone by Discard.",
    "discover": "List available model IDs. Discovery does not test inference.",
    "models": "Click a model row or press Space to toggle inclusion (never delete); Enter opens its details.",
    "apply": "Save all catalog edits for every provider and role preset to disk in one batch.",
    "discard": "Discard all pending catalog edits across providers and role presets; saved keys remain saved.",
}


class WorkbenchWidgetHost(Protocol):
    """Screen surface the workbench widgets act on; the screen implements it."""

    def schedule_update_help(self) -> None: ...
    def update_help(self) -> None: ...
    def record_help_focus(self, group_id: str | None) -> None: ...
    def confirm_open(self) -> bool: ...
    def pointer_focus(self, group_id: str | None) -> bool: ...
    def can_activate(self, group_id: str | None) -> bool: ...
    def consume_help_click(self, group_id: str | None) -> bool: ...
    def select_protocol(self, value: str) -> None: ...
    def accept_protocol(self) -> None: ...
    def page_confirm(self, direction: int) -> None: ...
    def is_busy(self) -> bool: ...
    def get_filter(self) -> str: ...
    def set_filter(self, text: str) -> None: ...
    def refresh_browser(self) -> None: ...
    def selection_changed(self, widget: SelectionList) -> None: ...
    def toggle_detail_images(self) -> None: ...
    def open_detail(self, name: str) -> None: ...
