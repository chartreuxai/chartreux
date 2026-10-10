"""Provider Settings for onboarding and in-session catalog management."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from math import isfinite
import re
from typing import ClassVar, Literal, cast

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.color import Color
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Input, Label, OptionList, SelectionList
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

from chartreux.core.dispatch.schema import DispatchMode
from chartreux.core.dispatch.session import bind_snapshot_policy
from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    CatalogValidationError,
    CatalogWriter,
    CatalogWriteResult,
    ConfigService,
    CredentialService,
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
    DiscoveryService,
    ModelEdits,
    OptionalEdit,
    ProviderDraft,
    ProviderWorkbenchResult,
    TLSConfig,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot, project_catalog_changes
from chartreux.core.model_catalog.matching import match_discovered_model
from chartreux.core.model_catalog.presets import (
    FULLY_CUSTOM,
    MISTRAL,
    PRESETS,
    ProviderPreset,
)
from chartreux.core.model_catalog.schema import (
    BaseModelDefinition,
    DeploymentDefinition,
    ProviderDefinition,
    valid_provider_name,
)
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.providers.management_state import (
    ConnectionDraft,
    CredentialStatusResolver,
    ManagementState,
    PendingModel,
    credential_status,
    usable_canonical_models,
)
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.checklist import CHECKBOX_SEGMENTS, Checklist
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

CURSOR_GUTTER = 2


class DetailHelp(NoMarkupStatic):
    """Scrollable selected-item text reachable by Tab when a summary truncates."""

    def on_focus(self, event: events.Focus) -> None:
        if isinstance(self.screen, ProviderWorkbenchScreen):
            self.screen.call_after_refresh(self.screen._update_help)

    can_focus = True

    def focus_on_click(self) -> bool:
        return False

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if isinstance(self.screen, ProviderWorkbenchScreen) and self.screen._confirm:
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_down(event)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if isinstance(self.screen, ProviderWorkbenchScreen) and self.screen._confirm:
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_up(event)

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("down", "scroll_down", "Scroll down", show=False),
        Binding("up", "scroll_up", "Scroll up", show=False),
    ]


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


def _bounded_cursor(widget: OptionList, direction: int) -> None:
    """Move to the next enabled row without wrapping or leaving the group."""
    anchor = widget.highlighted
    if anchor is None:
        anchor = -1 if direction > 0 else widget.option_count
    for index in range(
        anchor + direction, widget.option_count if direction > 0 else -1, direction
    ):
        if not widget.get_option_at_index(index).disabled:
            widget.highlighted = index
            return


class WorkbenchList(NavigableOptionList):
    """Two-cell cursor gutter, independent of each row's value."""

    def focus(self, scroll_visible: bool = True) -> WorkbenchList:
        super().focus(scroll_visible=scroll_visible)
        if isinstance(self.screen, ProviderWorkbenchScreen):
            if self.id != "wb-confirm-actions":
                self.screen._help_focus_id = self.id
            self.screen._update_help()
        return self

    BINDINGS: ClassVar[list[BindingType]] = [
        *NavigableOptionList.BINDINGS,
        Binding("enter", "select", "Select", priority=True, show=False),
    ]

    def focus_on_click(self) -> bool:
        if isinstance(self.screen, ProviderWorkbenchScreen):
            return self.screen._pointer_focus(self.id)
        return True

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if (
            isinstance(self.screen, ProviderWorkbenchScreen)
            and self.screen._confirm
            and self.id != "wb-confirm-actions"
        ):
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_down(event)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if (
            isinstance(self.screen, ProviderWorkbenchScreen)
            and self.screen._confirm
            and self.id != "wb-confirm-actions"
        ):
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_up(event)

    async def _on_click(self, event: events.Click) -> None:
        # A double click on a provider can land its second click on a control
        # exposed by the first click. The first click already activates the row;
        # later clicks in that gesture must not activate the destination view.
        if event.chain > 1 or (
            isinstance(self.screen, ProviderWorkbenchScreen)
            and (
                self.screen._consume_help_click(self.id)
                or not self.screen._can_activate(self.id)
            )
        ):
            event.stop()
            event.prevent_default()
            return
        if self.id == "wb-protocol" and isinstance(
            self.screen, ProviderWorkbenchScreen
        ):
            index = event.style.meta.get("option")
            if index is not None:
                option = self.get_option_at_index(index)
                if not option.disabled and option.id:
                    self.highlighted = index
                    self.screen._select_protocol(str(option.id))
            event.stop()
            event.prevent_default()
            return
        await super()._on_click(event)
        event.stop()
        event.prevent_default()

    def action_select(self) -> None:
        if isinstance(self.screen, ProviderWorkbenchScreen):
            if not self.screen._can_activate(self.id):
                return
            if self.id == "wb-protocol":
                self.screen._accept_protocol()
                return
        super().action_select()

    def action_cursor_down(self) -> None:
        _bounded_cursor(self, 1)

    def action_cursor_up(self) -> None:
        _bounded_cursor(self, -1)

    def action_page_down(self) -> None:
        if self.id == "wb-confirm-actions":
            self.screen.query_one("#wb-confirm", VerticalScroll).scroll_page_down(
                animate=False
            )
            return
        for _ in range(max(1, self.scrollable_content_region.height)):
            _bounded_cursor(self, 1)

    def action_page_up(self) -> None:
        if self.id == "wb-confirm-actions":
            self.screen.query_one("#wb-confirm", VerticalScroll).scroll_page_up(
                animate=False
            )
            return
        for _ in range(max(1, self.scrollable_content_region.height)):
            _bounded_cursor(self, -1)

    def on_key(self, event: events.Key) -> None:
        screen = self.screen
        if (
            event.key == "space"
            and isinstance(screen, ProviderWorkbenchScreen)
            and not screen._can_activate(self.id)
        ):
            event.stop()
            event.prevent_default()
            return
        if (
            event.key == "space"
            and self.id == "wb-detail-fields"
            and isinstance(screen, ProviderWorkbenchScreen)
            and self.highlighted_option is not None
            and self.highlighted_option.id == "images"
        ):
            screen._toggle_detail_images()
            event.stop()
            event.prevent_default()
            return
        if (
            self.id == "wb-protocol"
            and event.key == "space"
            and isinstance(screen, ProviderWorkbenchScreen)
        ):
            if self.highlighted_option and self.highlighted_option.id:
                screen._select_protocol(str(self.highlighted_option.id))
            event.stop()
            event.prevent_default()

    def on_focus(self, event: events.Focus) -> None:
        if isinstance(self.screen, ProviderWorkbenchScreen):
            if self.id != "wb-confirm-actions":
                self.screen._help_focus_id = self.id
            self.screen.call_after_refresh(self.screen._update_help)

    def render_line(self, y: int) -> Strip:
        line = super().render_line(y)
        try:
            index, _ = self._lines[self.scroll_offset.y + y]
        except IndexError:
            return line
        if line.cell_length < CURSOR_GUTTER:
            return line
        option = self.get_option_at_index(index)
        current = self.has_focus and self.highlighted == index and not option.disabled
        marker = f"{chrome_glyph('cursor')} " if current else "  "
        style = (list(line)[0].style or self.rich_style) + Style(meta={"option": index})
        return Strip([
            Segment(marker, style),
            *list(line.crop(0, line.cell_length - CURSOR_GUTTER)),
        ])


class BrowserList(WorkbenchList):
    """Type to filter while browsing, without stealing keys from an editor."""

    def on_key(self, event: events.Key) -> None:
        screen = self.screen
        if not isinstance(screen, ProviderWorkbenchScreen):
            return
        if screen._busy:
            event.stop()
            event.prevent_default()
            return
        if event.key in {"j", "k", "space"} and not screen.filter_text:
            return
        if event.key == "backspace":
            screen.filter_text = screen.filter_text[:-1]
        elif (char := event.character) and len(char) == 1 and char.isprintable():
            screen.filter_text += char
        else:
            return
        screen._refresh_browser()
        event.stop()
        event.prevent_default()


class ModelChecklist(Checklist):
    """Space toggles a deployment; Enter opens its metadata editor."""

    def focus_on_click(self) -> bool:
        if isinstance(self.screen, ProviderWorkbenchScreen):
            return self.screen._pointer_focus(self.id)
        return True

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if isinstance(self.screen, ProviderWorkbenchScreen) and self.screen._confirm:
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_down(event)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if isinstance(self.screen, ProviderWorkbenchScreen) and self.screen._confirm:
            event.stop()
            event.prevent_default()
            return
        super()._on_mouse_scroll_up(event)

    def _toggle_user_selection(self, value: str) -> None:
        screen = self.screen
        if not isinstance(screen, ProviderWorkbenchScreen) or not screen._can_activate(
            self.id
        ):
            return
        self.toggle(value)
        # Accept the user intent in this turn, before a queued Help/Back key can
        # change ownership. Queued programmatic SelectedChanged stays guarded.
        screen.on_selection_list_selected_changed(SelectionList.SelectedChanged(self))

    def action_select(self) -> None:
        if isinstance(
            self.screen, ProviderWorkbenchScreen
        ) and not self.screen._can_activate(self.id):
            return
        if self.id != "wb-models":
            super().action_select()
        elif self.highlighted is not None:
            option = self.get_option_at_index(self.highlighted)
            if not option.disabled and not option.value.startswith("\x00"):
                self._toggle_user_selection(option.value)

    def on_focus(self, event: events.Focus) -> None:
        if isinstance(self.screen, ProviderWorkbenchScreen):
            self.screen.call_after_refresh(self.screen._update_help)

    def render_line(self, y: int) -> Strip:
        line = super().render_line(y)
        index = self.scroll_offset.y + y
        if index >= self.option_count:
            return line
        segments = list(line)
        if not segments:
            return line
        style = segments[0].style or self.rich_style
        value = self.get_option_at_index(index).value
        if value.startswith("\x00") and len(segments) >= CHECKBOX_SEGMENTS:
            segments[1:CHECKBOX_SEGMENTS] = [Segment("   ", style)]
        if self.has_focus and index == self.highlighted:
            segments[0] = Segment(f"{chrome_glyph('cursor')} ", style)
        line = Strip(segments)
        if self.id == "wb-models" and not value.startswith("\x00"):
            details = " [Details]"
            details_style = style + Style(meta={"model_details": index})
            line = Strip([
                *line.crop_extend(
                    0,
                    max(0, self.scrollable_content_region.width - len(details)),
                    style,
                ),
                Segment(details, details_style),
            ])
        return line

    async def _on_click(self, event: events.Click) -> None:
        if self.id != "wb-models":
            return
        event.stop()
        event.prevent_default()
        if (
            not isinstance(self.screen, ProviderWorkbenchScreen)
            or self.screen._consume_help_click(self.id)
            or not self.screen._can_activate(self.id)
            or event.chain > 1
        ):
            return
        details_index = event.style.meta.get("model_details")
        index = (
            details_index
            if details_index is not None
            else event.style.meta.get("option")
        )
        if index is None:
            return
        option = self.get_option_at_index(index)
        if option.disabled or option.value.startswith("\x00"):
            return
        self.highlighted = index
        self.focus()
        if details_index is not None:
            self.action_detail()
        else:
            self._toggle_user_selection(option.value)

    def action_cursor_down(self) -> None:
        if self.id == "wb-models":
            _bounded_cursor(self, 1)
        else:
            super().action_cursor_down()

    def action_cursor_up(self) -> None:
        if self.id == "wb-models":
            _bounded_cursor(self, -1)
        else:
            super().action_cursor_up()

    def action_page_down(self) -> None:
        for _ in range(max(1, self.scrollable_content_region.height)):
            _bounded_cursor(self, 1)

    def action_page_up(self) -> None:
        for _ in range(max(1, self.scrollable_content_region.height)):
            _bounded_cursor(self, -1)

    def on_key(self, event: events.Key) -> None:
        if event.key == "space" and self.highlighted is not None and self.option_count:
            value = self.get_option_at_index(self.highlighted).value
            if isinstance(value, str) and value.startswith("\x00"):
                event.stop()
                event.prevent_default()
                return
        # Normal keys continue through SelectionList's key bindings; calling
        # Widget.on_key here is invalid because SelectionList has no such
        # superclass handler.
        return

    BINDINGS: ClassVar[list[BindingType]] = [
        # Process user inclusion intent before a following priority Help/Back key.
        Binding("space", "select", "Toggle option", priority=True, show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("enter", "detail", "Details", priority=True, show=False),
    ]

    def action_detail(self) -> None:
        screen = self.screen
        if isinstance(screen, ProviderWorkbenchScreen) and screen._can_activate(
            self.id
        ):
            if self.id == "wb-image-support":
                return
            if (
                self.id == "wb-models"
                and self.highlighted is not None
                and self.get_option_at_index(self.highlighted).value == "\x00empty"
            ):
                return
            if self.highlighted is not None:
                value = self.get_option_at_index(self.highlighted).value
                if not value.startswith("\x00"):
                    screen._open_detail(value)


class ProviderWorkbenchScreen(ModalScreen[ProviderWorkbenchResult]):
    """Full-screen management browser; the host opens it with push_screen_wait."""

    DEFAULT_CSS = """
    ProviderWorkbenchScreen { width: 100%; height: 100%; background: $background; }
    #workbench { width: 100%; height: 100%; padding: 0 1; }
    #wb-title { height: 1; color: $foreground; text-style: bold; }
    #wb-pending-action { height: 1; width: 100%; color: $warning; text-overflow: ellipsis; }
    #wb-filter, #wb-count, #wb-hint { height: 1; color: $text-muted; }
    #wb-providers { height: 1fr; min-height: 2; border: none; }
    #wb-catalog-filter { height: auto; max-height: 6; overflow-y: auto; border: none; }
    #wb-providers > .option-list--option-disabled { color: $text-muted; }
    #wb-actions, #wb-choose, #wb-picker, #wb-catalog, #wb-presets, #wb-preset-editor, #wb-detail-fields,
    #wb-protocol { height: 1fr; min-height: 2; border: none; }
    #wb-models { height: 1fr; min-height: 2; border: none; }
    #wb-root-actions { height: auto; max-height: 8; min-height: 2; border: none; }
    #wb-actions.wb-management-fields { height: 4; min-height: 4; }
    #wb-provider-operations { height: 1fr; min-height: 2; border: none; }
    #wb-models-actions { height: 5; min-height: 5; border: none; }
    #wb-help { height: 2; overflow-y: auto; color: $text-muted; }
    #wb-confirm-host {
        overlay: screen; layer: overlay; width: 80%; max-width: 60;
        height: 11; background: transparent;
    }
    #wb-confirm {
        width: 100%; height: 100%;
        overflow-y: auto; border: solid $foreground-muted; background: $surface;
        padding: 0 1;
    }
    #wb-confirm-text { height: auto; }
    #wb-confirm-actions { height: auto; max-height: 3; border: none; }
    #wb-confirm-help { height: 1; color: $text-muted; }
    .wb-confirm-underlay { opacity: 40%; }
    .wb-group, #wb-providers, #wb-actions, #wb-choose, #wb-picker, #wb-catalog,
    #wb-presets, #wb-preset-editor, #wb-detail-fields, #wb-protocol, #wb-confirm-actions, #wb-models, #wb-models-actions,
    #wb-connection-actions, #wb-detail-actions, #wb-image-support {
        background: $background; padding: 0; border: none;
        &:focus { background-tint: transparent; }
        & > .option-list--option-hover { background: transparent; }
        & > .option-list--option-highlighted { background: transparent; color: $foreground; text-style: none; }
        &:focus > .option-list--option-highlighted { background: $foreground; color: $background; text-style: bold; }
    }
    #wb-models > .selection-list--button-selected,
    #wb-image-support > .selection-list--button-selected { color: $foreground; background: transparent; }
    #wb-image-support { height: auto; min-height: 1; border: none; }
    #wb-editor { height: 1fr; min-height: 4; }
    #wb-editor Label { height: 1; }
    #wb-editor Input { height: 3; border: solid $foreground-muted; }
    #wb-editor Input:focus { border: solid $primary; }
    #wb-field-error { height: 1; }
    #wb-field-error.has-error { color: $error; }
    #wb-editor Input.-invalid { border: solid $error; }
    #wb-connection-form { height: 1fr; overflow-y: auto; }
    #wb-connection-form .connection-row { height: 1; width: 100%; }
    #wb-connection-form .connection-row Label {
        width: 24; height: 1;
    }
    #wb-connection-form .connection-row Input {
        width: 1fr; height: 1; border: none; background: $surface;
    }
    #wb-connection-form Input:focus { background: $primary-background; }
    #wb-connection-style { width: 1fr; height: 3; border: none; }
    #wb-connection-actions { height: auto; min-height: 2; border: none; }
    #wb-detail { height: 1fr; overflow-y: auto; }
    #wb-detail-actions { height: auto; min-height: 2; border: none; }
    #wb-catalog-actions, #wb-presets-actions, #wb-preset-editor-actions {
        height: auto; min-height: 2; border: none;
    }
    #wb-detail .price-row { height: 1; width: 100%; }
    #wb-detail .price-row Label {
        width: 22;
        height: 1;
    }
    #wb-detail .price-row Input {
        width: 1fr;
        height: 1;
        border: none;
        background: $surface;
    }
    #wb-detail .price-row Input:focus { background: $primary-background; }
    #wb-detail .price-row Input.-invalid { background: $error 30%; }
    #wb-detail .price-error { height: 1; color: $error; }
    #wb-detail .detail-field-row { height: 1; width: 100%; }
    #wb-detail .detail-field-row Label {
        width: 22;
        height: 1;
    }
    #wb-detail .detail-field-row Input {
        width: 1fr;
        height: 1;
        border: none;
        background: $surface;
    }
    #wb-detail .detail-field-row Input:focus { background: $primary-background; }
    #wb-detail .detail-field-row Input.-invalid { background: $error 30%; }
    #wb-detail .detail-error { height: 1; color: $error; }
    #wb-detail-supported-thinking { height: auto; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", "Back", priority=True, show=False),
        Binding("f1", "help", "Help", show=False),
        Binding("tab", "cycle_group(1)", "Next group", priority=True, show=False),
        Binding(
            "shift+tab", "cycle_group(-1)", "Previous group", priority=True, show=False
        ),
    ]
    # Reading order and primary body for each view. Auxiliary groups remain in
    # the inventory even when their backing forms are never displayed.
    VIEW_GROUPS: ClassVar[dict[WorkbenchView, tuple[str, ...]]] = {
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
    VIEW_CONTROLS: ClassVar[tuple[str, ...]] = tuple(
        dict.fromkeys(name for groups in VIEW_GROUPS.values() for name in groups)
    ) + ("detail", "connection-form")
    UNDERLAY_CONTROLS: ClassVar[tuple[str, ...]] = (
        "title",
        "pending-action",
        "filter",
        "count",
        "help",
        "hint",
        "field-error",
    ) + tuple(name for name in VIEW_CONTROLS if name != "confirm")

    MIN_WIDTH = 48
    MIN_HEIGHT = 24
    FIELDS: ClassVar[dict[str, str]] = {
        "base": "api_base",
        "style": "api_style",
        "env": "api_key_env_var",
    }
    DESCRIPTIONS: ClassVar[dict[str, str]] = {
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

    def __init__(  # noqa: PLR0913, PLR0915 - host-owned dependencies and runtime state
        self,
        *,
        snapshot: CatalogSnapshot,
        discovery: DiscoveryService,
        catalog_writer: CatalogWriter,
        credentials: CredentialService,
        config: ConfigService,
        credential_resolver: CredentialStatusResolver,
        tls: TLSConfig | None = None,
        mode: Literal["management", "onboarding"] = "management",
        initial_view: Literal["providers", "presets"] = "providers",
        on_models_saved: Callable[
            [frozenset[str], frozenset[str], frozenset[str]], None
        ]
        | None = None,
        session_only_credential_envs: set[str] | None = None,
        allowed_models: Sequence[str] = (),
        thinking_overrides: Mapping[str, str] | None = None,
        selected_model: str | None = None,
    ) -> None:
        super().__init__()
        self.on_models_saved = on_models_saved
        self.snapshot = snapshot
        self.discovery_service = discovery
        self.catalog_writer = catalog_writer
        self.credentials = credentials
        self.config = config
        self.allowed_models = tuple(allowed_models)
        self.thinking_overrides = dict(thinking_overrides or {})
        self.selected_model = selected_model
        self.credential_resolver = credential_resolver
        self.tls = tls or TLSConfig()
        self.mode: Literal["management", "onboarding"] = mode
        self.initial_view = initial_view
        self._setup_ready = False
        self._models_view = False
        self._model_filter: str | None = None
        self._customize_presets = False
        self._preset_role: str | None = None
        self._preset_field: str | None = None
        self._preset_draft: tuple[str, str] | None = None
        self._key_after_env = False
        self._after_commit: str | None = None
        self._protocol_picker = False
        self.filter_text = ""
        self.state: ManagementState | None = None
        self._editing: str | None = None
        self._detail: str | None = None
        self._detail_preview_wire: str | None = None
        self._detail_cursor: str | None = None
        self._detail_scroll = 0
        self._detail_original: tuple[object, ...] | None = None
        self._detail_edit_row: str | None = None
        self._confirm: str | None = None
        self._confirm_opener: str | None = None
        self._connection_approved = False
        self._connection_scopes: list[str] = []
        self._protocol_value: str | None = None
        self._help_open = False
        self._help_opener: NavigationFrame | None = None
        self._presets_opener: NavigationFrame | None = None
        self._add_opener: NavigationFrame | None = None
        self._editor_opener: NavigationFrame | None = None
        self._confirm_position: _GroupPosition | None = None
        self._resize_frame: NavigationFrame | None = None
        self._navigation_epoch = 0
        self._unmounted = False
        self._resize_owner: tuple[str | None, str | None, bool] | None = None
        self._models_list_open = False
        self._provider_open = False
        self._message = ""
        self._unresolved: dict[
            str, tuple[str, Literal["running", "success", "error", "warning", "info"]]
        ] = {}
        self._feedback_kind: Literal[
            "running", "success", "error", "warning", "info"
        ] = "info"
        self._changed = False
        self._warning: str | None = None
        self._busy = False
        self._commit_task: asyncio.Task[None] | None = None
        self._commit_return: NavigationFrame | None = None
        self._write_task: (
            asyncio.Task[CatalogWriteResult | CatalogValidationError] | None
        ) = None
        self._reload_failed = False
        self._mistral_overwrite_approved = False
        self._dismissed = False
        self._model_sync = False
        self._browser_cursor: str | None = None
        self._provider_visible_count = (0, 0)
        self._stage: str | None = None
        self._draft_session = 0
        self._add: ProviderDraft | None = None
        self._add_checkpoint: ProviderDraft | None = None
        self._connection_pending: dict[str, PendingModel] | None = None
        self._add_state: ManagementState | None = None
        self._pre_add_state: ManagementState | None = None
        self._picker = False
        self._view = WorkbenchView.PROVIDERS
        self._navigation: list[NavigationFrame] = []
        self._positions: dict[
            tuple[WorkbenchView, str, tuple[str, ...]], _GroupPosition
        ] = {}
        self._group_contents_context: dict[
            str, tuple[WorkbenchView, tuple[str, ...]]
        ] = {}
        self._model_row_identities: tuple[str, ...] = ()
        self._return_focus: str | None = None
        self._help_focus_id: str | None = None
        self._hint_targets: list[tuple[int, int, str]] | None = None
        self._restore_scroll: int | None = None
        self._help_dismiss_click: str | None = None
        self._feedback_token = 0
        self._credential_input_id = "wb-input"
        self._saved_credential_envs: set[str] = set()
        # Keep the host's runtime provenance by reference across screen instances.
        self._session_only_credential_envs = (
            session_only_credential_envs
            if session_only_credential_envs is not None
            else set()
        )
        self._onboarding_presets_seeded = False

    def compose(self) -> ComposeResult:  # noqa: PLR0915 - declarative widget tree
        with Vertical(id="workbench"):
            yield NoMarkupStatic("Provider Settings", id="wb-title")
            pending = NoMarkupStatic(
                "Action pending · finish editing, then Esc to answer",
                id="wb-pending-action",
            )
            pending.display = False
            yield pending
            yield NoMarkupStatic("Filter: type to filter", id="wb-filter")
            yield NoMarkupStatic("", id="wb-count")
            yield BrowserList(id="wb-providers")
            yield WorkbenchList(id="wb-root-actions", classes="wb-group")
            yield WorkbenchList(id="wb-choose")
            yield WorkbenchList(id="wb-picker")
            yield WorkbenchList(id="wb-catalog-filter")
            yield WorkbenchList(id="wb-catalog")
            yield WorkbenchList(id="wb-catalog-actions", classes="wb-group")
            yield WorkbenchList(id="wb-presets")
            yield WorkbenchList(id="wb-presets-actions", classes="wb-group")
            yield WorkbenchList(id="wb-preset-editor")
            yield WorkbenchList(id="wb-preset-editor-actions", classes="wb-group")
            yield WorkbenchList(id="wb-detail-fields")
            yield WorkbenchList(id="wb-protocol")
            yield WorkbenchList(id="wb-actions")
            yield WorkbenchList(id="wb-provider-operations", classes="wb-group")
            yield ModelChecklist(id="wb-models")
            yield WorkbenchList(id="wb-models-actions")
            with VerticalScroll(id="wb-detail"):
                yield NoMarkupStatic("", id="wb-detail-model-id")
                yield NoMarkupStatic("", id="wb-detail-deployment-id")
                yield NoMarkupStatic("", id="wb-detail-supported-thinking")
                for field, label, placeholder in (
                    ("thinking", "Thinking", "Known thinking level"),
                    ("temperature", "Temperature", "Blank clears"),
                    ("auto_compact_threshold", "Compaction threshold", "Blank clears"),
                ):
                    with Horizontal(classes="detail-field-row"):
                        yield Label(label, id=f"label-detail-{field}")
                        yield Input(id=f"detail-{field}", placeholder=placeholder)
                    error = NoMarkupStatic(
                        "", id=f"detail-error-{field}", classes="detail-error"
                    )
                    error.display = False
                    yield error
                yield NoMarkupStatic(
                    "Deployment image support · Space toggles",
                    id="wb-image-support-label",
                )
                yield ModelChecklist(id="wb-image-support")
                for name, label in (
                    ("input", "Input price"),
                    ("output", "Output price"),
                    ("cached_input", "Cached input price"),
                ):
                    with Horizontal(classes="price-row"):
                        yield Label(label, id=f"label-{name}")
                        yield Input(
                            id=f"price-{name}", placeholder="Blank clears; 0 means free"
                        )
                    error = NoMarkupStatic(
                        "", id=f"error-{name}", classes="price-error"
                    )
                    error.display = False
                    yield error
            yield WorkbenchList(id="wb-detail-actions")
            with Vertical(id="wb-editor"):
                yield Label("Edit value", id="wb-editor-label")
                yield Input(id="wb-input")
            yield NoMarkupStatic("", id="wb-field-error", classes="field-error")
            with VerticalScroll(id="wb-connection-form"):
                with Horizontal(classes="connection-row"):
                    yield Label("Provider name", id="wb-connection-name-label")
                    yield Input(id="wb-connection-name")
                with Horizontal(classes="connection-row"):
                    yield Label("API base URL", id="wb-connection-base-label")
                    yield Input(id="wb-connection-base")
                with Horizontal(classes="connection-row"):
                    yield Label("API style", id="wb-connection-style-label")
                    yield WorkbenchList(id="wb-connection-style")
                with Horizontal(classes="connection-row"):
                    yield Label("Credential env var", id="wb-connection-env-label")
                    yield Input(id="wb-connection-env")
                with Horizontal(classes="connection-row"):
                    yield Label("API key", id="wb-connection-key-label")
                    yield Input(id="wb-connection-key", password=True)
            yield WorkbenchList(id="wb-connection-actions")
            confirm_host = Container(id="wb-confirm-host")
            confirm_host.display = False
            with confirm_host:
                with VerticalScroll(id="wb-confirm"):
                    yield NoMarkupStatic("", id="wb-confirm-text")
                    yield WorkbenchList(id="wb-confirm-actions")
                    yield NoMarkupStatic(
                        f"{chrome_glyph('vertical')} Choose · Enter Confirm · Esc Keep editing",
                        id="wb-confirm-help",
                    )
            yield DetailHelp("", id="wb-help")
            yield NoMarkupStatic("", id="wb-hint")
        yield NoMarkupStatic(
            "Terminal too small — need 48x24. Resize to continue.", id="wb-small"
        )

    def on_resize(self, event: events.Resize) -> None:
        if self.is_mounted:
            too_small = (
                event.size.width < self.MIN_WIDTH or event.size.height < self.MIN_HEIGHT
            )
            workbench = self.query_one("#workbench")
            if too_small and workbench.display:
                self._resize_frame = self._frame()
                self._resize_owner = (self._confirm, self._editing, self._help_open)
            workbench.display = not too_small
            self.query_one("#wb-small").display = too_small
            if self._busy or too_small:
                return
            if not too_small:
                if self._models_view:
                    self._refresh_catalog(preserve_filter_cursor=True)
                elif self._models_list_open:
                    self._refresh_models()
                elif self._stage == "connection":
                    if not self._editing and not self._confirm and not self._detail:
                        self._refresh_add_connection()
                elif self._provider_open or self._stage == "models":
                    if not self._editing and not self._confirm and not self._detail:
                        self._refresh_actions()
                else:
                    self._refresh_browser()
                self._update_help()
                frame, self._resize_frame = self._resize_frame, None
                if (
                    frame is not None
                    and frame.view == self._view
                    and frame.stage == self._stage
                    and self._resize_owner
                    == (self._confirm, self._editing, self._help_open)
                ):
                    self._restore_transient_opener(frame)
                    if frame.focus_id:
                        self._defer_focus(self.query_one(f"#{frame.focus_id}"))

    def on_mount(self) -> None:
        if getattr(self.app, "_pending_callbacks", None) or getattr(
            self.app, "_pending_local_question", None
        ):
            self.query_one("#wb-pending-action").display = True
        self.query_one("#wb-small").display = False
        for name in self.VIEW_CONTROLS:
            widget = self.query_one(f"#wb-{name}")
            widget.display = False
            if isinstance(widget, OptionList):
                widget.add_class("wb-group")
        self._refresh_browser()
        self._activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
        if self.initial_view == "presets":
            self._open_presets()

    def _can_focus_group(self, group_id: str | None) -> bool:
        if self._unmounted or self._busy or self._dismissed or group_id is None:
            return False
        widget = self.query_one(f"#{group_id}")
        if (
            not widget.can_focus
            or not widget.display
            or not widget.visible
            or widget.disabled
            or any(not ancestor.display for ancestor in widget.ancestors)
        ):
            return False
        if self._confirm:
            return group_id == "wb-confirm-actions"
        return True

    def _can_activate(self, group_id: str | None) -> bool:
        """Resolve interaction ownership before pointer focus can dismiss help."""
        if not self._can_focus_group(group_id):
            return False
        if self._confirm:
            return group_id == "wb-confirm-actions"
        if self._help_open:
            return False
        if self._editing or self._view == WorkbenchView.EDITOR:
            return group_id == "wb-input"
        return True

    def _pointer_focus(self, group_id: str | None) -> bool:
        if not self._can_focus_group(group_id):
            return False
        if self._help_open and group_id != "wb-help":
            self._help_dismiss_click = group_id
            self._help_open = False
            self._help_opener = None
            self._update_help()
        return True

    def _consume_help_click(self, group_id: str | None) -> bool:
        dismissed = self._help_dismiss_click == group_id
        self._help_dismiss_click = None
        return dismissed

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if self._confirm and event.widget.id != "wb-confirm-actions":
            self.set_focus(self.query_one("#wb-confirm-actions"), scroll_visible=False)

    def on_click(self, event: events.Click) -> None:  # noqa: PLR0912
        if event.chain > 1:
            return
        if event.widget is self.query_one("#wb-help"):
            event.stop()
            event.prevent_default()
            self.action_help()
            return
        hint = self.query_one("#wb-hint", NoMarkupStatic)
        if event.widget is not hint:
            return
        event.stop()
        event.prevent_default()
        x = event.screen_x - hint.region.x
        key = next(
            (key for start, end, key in self._hint_targets or [] if start <= x < end),
            None,
        )
        if key == "Esc":
            self.action_back()
        elif key == "F1":
            self.action_help()
        elif key in {"Tab", "Shift+Tab"}:
            self.action_cycle_group(1 if key == "Tab" else -1)
        elif key == "Enter":
            if self._protocol_picker and self._can_activate("wb-protocol"):
                self._accept_protocol()
            elif self._editing and self._can_activate("wb-input"):
                self._accept_editor()
            elif isinstance(self.focused, (WorkbenchList, ModelChecklist)):
                if isinstance(self.focused, ModelChecklist):
                    self.focused.action_detail()
                else:
                    self.focused.action_select()
        elif key == "Space":
            if isinstance(self.focused, ModelChecklist):
                self.focused.action_select()
            elif isinstance(self.focused, WorkbenchList) and self._can_activate(
                self.focused.id
            ):
                option = self.focused.highlighted_option
                if option and option.id:
                    if self.focused.id == "wb-protocol":
                        self._select_protocol(str(option.id))
                    elif (
                        self.focused.id == "wb-detail-fields" and option.id == "images"
                    ):
                        self._toggle_detail_images()

    def _focus_groups(self) -> list[str]:
        groups = {
            WorkbenchView.PROVIDERS: ["wb-providers", "wb-root-actions"],
            WorkbenchView.ACTIONS: ["wb-actions", "wb-provider-operations"],
            WorkbenchView.CONNECTION: ["wb-actions", "wb-connection-actions"],
            WorkbenchView.MODELS: ["wb-models", "wb-models-actions", "wb-help"],
            WorkbenchView.CATALOG: [
                "wb-catalog-filter",
                "wb-catalog",
                "wb-catalog-actions",
                "wb-help",
            ],
            WorkbenchView.DETAIL: ["wb-detail-fields", "wb-detail-actions"],
            WorkbenchView.PRESETS: ["wb-presets", "wb-presets-actions"],
            WorkbenchView.PRESET_EDITOR: [
                "wb-preset-editor",
                "wb-preset-editor-actions",
            ],
        }.get(self._view, [])
        if self._help_open and "wb-help" not in groups:
            groups = [*groups, "wb-help"]
        return [
            group
            for group in groups
            if self.query_one(f"#{group}").can_focus
            and self.query_one(f"#{group}").display
            and self.query_one(f"#{group}").visible
            and all(
                ancestor.display for ancestor in self.query_one(f"#{group}").ancestors
            )
            and not self.query_one(f"#{group}").disabled
        ]

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "cycle_group":
            if self._busy or self._confirm:
                return True
            return (
                bool(self._focus_groups())
                and not self._editing
                and self._view != WorkbenchView.EDITOR
            )
        return super().check_action(action, parameters)

    def action_cycle_group(self, direction: int) -> None:
        if self._busy:
            return
        if self._confirm:
            self.query_one("#wb-confirm-actions").focus()
            return
        groups = self._focus_groups()
        if not groups:
            return
        self._navigation_epoch += 1
        current = self.focused.id if self.focused else None
        index = (
            groups.index(current) if current in groups else (-1 if direction > 0 else 0)
        )
        self._capture_group_position()
        target = self.query_one(f"#{groups[(index + direction) % len(groups)]}")
        if target.id != "wb-help":
            self._help_open = False
            self._help_opener = None
        position = self._positions.get((
            self._view,
            target.id or "",
            self._group_context(self._view, target.id or ""),
        ))
        if position is not None:
            self._restore_group_position(position, restore_focus=True)
        else:
            target.focus(scroll_visible=False)
        self._update_help()

    def _group_context(  # noqa: PLR0911 - each view owns its semantic context
        self, view: WorkbenchView, group_id: str
    ) -> tuple[str, ...]:
        if group_id == "wb-help" and self._help_focus_id:
            origin = self._group_contents_context.get(self._help_focus_id)
            if origin:
                return (origin[0].value, self._help_focus_id, *origin[1])
        provider = self.state.provider_id if self.state else ""
        if view in {
            WorkbenchView.ACTIONS,
            WorkbenchView.CONNECTION,
            WorkbenchView.MODELS,
        }:
            if self._add is not None and self._stage == "connection":
                provider = self._add.provider_id or f"draft:{self._draft_session}"
            return (provider, self._stage or "management")
        if view == WorkbenchView.CATALOG:
            return (self._model_filter or "all",) if group_id == "wb-catalog" else ()
        if view == WorkbenchView.DETAIL:
            return (provider, self._detail_preview_wire or self._detail or "")
        if view == WorkbenchView.PRESET_EDITOR:
            return (self._preset_role or "",)
        if view in {WorkbenchView.PICKER, WorkbenchView.DEPLOYMENTS}:
            return (
                "preset"
                if self._preset_field
                else "deployment"
                if self._picker
                else "model",
                self._preset_role or provider,
                self._preset_field or self._detail or self._detail_cursor or "",
            )
        if view == WorkbenchView.EDITOR:
            group = self._help_focus_id or "wb-actions"
            origin = self._group_contents_context.get(group)
            if origin:
                return (origin[0].value, group, *origin[1], self._editing or "")
            if self._navigation:
                opener = self._navigation[-1]
                return (
                    opener.view.value,
                    opener.focus_id or "",
                    *opener.context,
                    self._editing or "",
                )
        return ()

    def _row_identity(self, widget: OptionList, index: int) -> str | None:
        option = widget.get_option_at_index(index)
        if isinstance(widget, SelectionList):
            if widget.id == "wb-models" and index < len(self._model_row_identities):
                return self._model_row_identities[index]
            return str(cast(Selection[str], option).value)
        return str(option.id) if option.id is not None else None

    # Keep the identity-neighbour repair algorithm aligned with SettingsScreen's
    # _capture_group_position/_restore_group_position (and MCPApp's equivalent).
    def _capture_group_position(
        self, widget: Widget | None = None
    ) -> _GroupPosition | None:
        widget = widget or self.focused
        if widget is None or widget.id is None:
            return None
        owner = self._group_contents_context.get(widget.id)
        if isinstance(widget, Input) or widget.id == "wb-help":
            if not widget.display:
                return None
            view = WorkbenchView.EDITOR if isinstance(widget, Input) else self._view
            owner = (view, self._group_context(view, widget.id))
            self._group_contents_context[widget.id] = owner
        if owner is None:
            if not widget.display:
                return None
            owner = (self._view, self._group_context(self._view, widget.id))
            self._group_contents_context[widget.id] = owner
        view, context = owner
        identities = (
            tuple(
                identity
                for index, option in enumerate(widget.options)
                if not option.disabled
                and (identity := self._row_identity(widget, index)) is not None
            )
            if isinstance(widget, OptionList)
            else ()
        )
        selected = (
            self._row_identity(widget, widget.highlighted)
            if isinstance(widget, OptionList) and widget.highlighted is not None
            else None
        )
        position = _GroupPosition(
            view,
            widget.id,
            context,
            identities,
            identities.index(selected) if selected in identities else 0,
            widget.scroll_x,
            widget.scroll_y,
            widget.cursor_position if isinstance(widget, Input) else None,
        )
        self._positions[(view, widget.id, context)] = position
        return position

    def _position_before_rebuild(
        self, view: WorkbenchView, group: str
    ) -> _GroupPosition | None:
        widget = self.query_one(f"#{group}")
        context = self._group_context(view, group)
        owner = self._group_contents_context.get(group)
        previous = self._capture_group_position(widget) if owner is not None else None
        position = self._positions.get((view, group, context))
        if owner != (view, context) and isinstance(widget, OptionList):
            widget.highlighted = None
            if group == "wb-catalog" and position is None and previous is not None:
                position = replace(previous, view=view, context=context)
        self._group_contents_context[group] = (view, context)
        return position

    def _restore_group_position(
        self, position: _GroupPosition | None, *, restore_focus: bool = False
    ) -> None:
        if position is None:
            return
        widget = self.query_one(f"#{position.group}")
        if self._group_contents_context.get(position.group) != (
            position.view,
            position.context,
        ):
            return
        if isinstance(widget, OptionList):
            valid = [
                i for i, option in enumerate(widget.options) if not option.disabled
            ]
            ids = {self._row_identity(widget, i): i for i in valid}
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
        if (
            restore_focus
            and self._view == position.view
            and widget.display
            and widget.can_focus
            and self._can_focus_group(position.group)
        ):
            if isinstance(widget, WorkbenchList) and widget.id != "wb-confirm-actions":
                self._help_focus_id = widget.id
            self.set_focus(widget, scroll_visible=False)

        focused = self.focused
        view, epoch = self._view, self._navigation_epoch
        owner = (self._confirm, self._editing, self._help_open, self._busy)

        def finish_restore() -> None:
            if self._unmounted:
                return
            if (
                self._dismissed
                or self._navigation_epoch != epoch
                or self._view != view
                or self.focused is not focused
                or (self._confirm, self._editing, self._help_open, self._busy) != owner
            ):
                return
            if not widget.display or any(
                not ancestor.display for ancestor in widget.ancestors
            ):
                return
            if self._group_contents_context.get(position.group) != (
                position.view,
                position.context,
            ):
                return
            if self._group_context(position.view, position.group) != position.context:
                return
            if (
                restore_focus
                and widget.has_focus
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
            if isinstance(widget, OptionList):
                widget.scroll_to_highlight()

        self.call_after_refresh(finish_restore)

    def _frame(self) -> NavigationFrame:
        focused = self.focused
        selected: str | None = None
        scroll = 0
        if focused is not None:
            selected_index = getattr(focused, "highlighted", None)
            if selected_index is not None:
                if isinstance(focused, SelectionList):
                    selected_option = focused.get_option_at_index(selected_index)
                    selected = str(cast(Selection[str], selected_option).value)
                else:
                    selected_option = getattr(focused, "highlighted_option", None)
                    if selected_option is not None and selected_option.id is not None:
                        selected = str(selected_option.id)
                scroll = int(getattr(getattr(focused, "scroll_offset", None), "y", 0))
        position = self._capture_group_position()
        return NavigationFrame(
            self._view,
            focused.id if focused is not None else None,
            selected,
            scroll,
            self.filter_text,
            self._model_filter,
            self._detail_cursor,
            self._stage,
            self.state.provider_id if self.state is not None else None,
            position.context if position else (),
            position,
        )

    @property
    def view(self) -> WorkbenchView:
        """Current view for diagnostics and keyboard-flow tests."""
        return self._view

    def _push_view(self, view: WorkbenchView) -> None:
        self._navigation_epoch += 1
        self._navigation.append(self._frame())
        self._view = view

    def _activate_view(
        self,
        view: WorkbenchView,
        *,
        focus_id: str | None = None,
        selected_id: str | None = None,
    ) -> None:
        """Display one view and restore focus only after all controls are updated."""
        self._navigation_epoch += 1
        self._capture_group_position()
        self._view = view
        primary = self._sync_view()
        self._restore_view_focus(
            focus_id or ("wb-input" if primary == "editor" else f"wb-{primary}"),
            selected_id=selected_id,
        )

    def _defer_focus(self, widget: Widget) -> None:
        view, stage, epoch = self._view, self._stage, self._navigation_epoch
        context = self._group_context(view, widget.id or "")
        owner = (self._confirm, self._editing, self._help_open, self._busy)

        def restore() -> None:
            if not self.is_mounted or self._dismissed or self._unmounted:
                return
            if not widget.display or any(
                not ancestor.display for ancestor in widget.ancestors
            ):
                return
            if (
                self._navigation_epoch == epoch
                and self._view == view
                and self._stage == stage
                and (self._confirm, self._editing, self._help_open, self._busy) == owner
                and self._group_context(view, widget.id or "") == context
            ):
                widget.focus()

        self.call_after_refresh(restore)

    def _restore_view_focus(
        self, widget_id: str, *, selected_id: str | None = None
    ) -> None:
        """Restore an opener only after the shared projection has been applied."""
        if not widget_id.startswith("wb-"):
            widget_id = f"wb-{widget_id}"
        if self._confirm:
            widget_id = "wb-confirm-actions"
        widget = self.query_one(f"#{widget_id}")
        if (
            not widget.display
            or not widget.visible
            or any(not ancestor.display for ancestor in widget.ancestors)
        ):
            return
        if selected_id is not None and isinstance(widget, OptionList):
            ids = [
                str(cast(Selection[str], option).value)
                if isinstance(widget, SelectionList)
                else str(option.id)
                for option in widget.options
            ]
            if selected_id in ids:
                widget.highlighted = ids.index(selected_id)
        if getattr(widget, "can_focus", False) or hasattr(widget, "focus"):
            self.set_focus(widget)
        if selected_id is None:
            self._restore_group_position(
                self._positions.get((
                    self._view,
                    widget_id,
                    self._group_context(self._view, widget_id),
                ))
            )
        if frame_scroll := getattr(self, "_restore_scroll", None):
            try:
                widget.scroll_y = frame_scroll
            except (AttributeError, TypeError):
                pass

    def _pop_view(self) -> bool:
        if not self._navigation:
            return False
        frame = self._navigation.pop()
        # Provider typing is a local browser filter; keep changes made while
        # a nested model/editor view was open.
        self._model_filter = frame.model_filter
        self._detail_cursor = frame.detail_cursor
        self._stage = frame.stage
        if frame.view == WorkbenchView.CONNECTION:
            self._models_list_open = False
            if self.state is not None:
                self.state.begin_discovery()
        self._view = frame.view
        self._return_focus = frame.focus_id
        self._restore_frame(frame)
        return True

    def _restore_frame(self, frame: NavigationFrame) -> None:
        self._restore_scroll = frame.scroll
        if (
            frame.view != WorkbenchView.CONNECTION
            and frame.provider_id is not None
            and self.state is not None
            and frame.provider_id in self.state.catalog.providers
        ):
            self.state.for_provider(frame.provider_id)
        if frame.view == WorkbenchView.PROVIDERS:
            self._refresh_browser()
            self._activate_view(
                frame.view,
                focus_id=frame.focus_id or "wb-providers",
                selected_id=frame.selected_id,
            )
        elif frame.view == WorkbenchView.CATALOG:
            self._refresh_catalog()
            self._activate_view(
                frame.view,
                focus_id=(
                    frame.focus_id
                    if frame.focus_id in {"wb-catalog", "wb-catalog-filter"}
                    or (
                        frame.focus_id == "wb-catalog-actions"
                        and self.state
                        and self.state.dirty
                    )
                    else "wb-catalog"
                ),
                selected_id=frame.selected_id,
            )
        elif frame.view == WorkbenchView.MODELS:
            self._refresh_models()
            self._activate_view(
                frame.view,
                focus_id=(
                    frame.focus_id
                    if frame.focus_id in {"wb-models", "wb-models-actions"}
                    else "wb-models"
                ),
                selected_id=frame.selected_id,
            )
        elif frame.view in {WorkbenchView.ACTIONS, WorkbenchView.CONNECTION}:
            if frame.view == WorkbenchView.CONNECTION:
                self._refresh_add_connection(highlight=frame.selected_id)
            else:
                self._refresh_actions(highlight=frame.selected_id)
            self._activate_view(
                frame.view,
                focus_id=frame.focus_id or "wb-actions",
                selected_id=frame.selected_id,
            )
        else:
            # Nested pickers, editors, confirmations, and preset selection all
            # retain their widgets; restore their recorded control directly.
            self._activate_view(
                frame.view, focus_id=frame.focus_id, selected_id=frame.selected_id
            )
        if (
            frame.view in {WorkbenchView.MODELS, WorkbenchView.CATALOG}
            and frame.focus_id
        ):
            widget = self.query_one(f"#{frame.focus_id}")
            widget.scroll_to(y=frame.scroll, animate=False, force=True)
        self._restore_group_position(frame.position)
        self._restore_scroll = 0

    def _refresh_browser(self) -> None:  # noqa: PLR0914
        browser = self.query_one("#wb-providers", BrowserList)
        position = self._position_before_rebuild(
            WorkbenchView.PROVIDERS, "wb-providers"
        )
        current = (
            str(browser.highlighted_option.id)
            if browser.highlighted_option
            else self._browser_cursor
        )
        query = self.filter_text.strip()
        sensitive = any(char.isupper() for char in query)
        catalog = (
            self.state.catalog if self.state is not None else self.snapshot.catalog
        )
        names = sorted(catalog.providers)
        configured = [
            name
            for name in names
            if name in self.snapshot.overlaid_providers
            or name not in SHIPPED_CATALOG.providers
            or bool(
                catalog.providers[name].api_key_env_var
                and self.credential_resolver(catalog.providers[name].api_key_env_var)
            )
        ]
        available = [name for name in names if name not in configured]
        names = configured + available
        visible = [
            name
            for name in names
            if (query in name if sensitive else query.lower() in name.lower())
            or (
                query in self._friendly(name)
                if sensitive
                else query.lower() in self._friendly(name).lower()
            )
        ]
        browser.clear_options()
        browser.add_option(Option("PROVIDERS", id="\x00heading", disabled=True))
        if not visible:
            browser.add_option(
                Option(
                    "No matching providers" if query else "No configured providers",
                    id="\x00empty",
                )
            )
        for name in visible:
            count = self._runnable_provider_count(name)
            status = credential_status(
                self._provider_env(name), self.credential_resolver
            )
            provider = catalog.providers[name]
            if provider.disabled:
                status = "Disabled"
            elif not self._provider_has_enabled_model(name):
                status += "; No enabled models"
            if self.state is not None and isinstance(
                self.state.discoveries.get(name), DiscoveryError
            ):
                status += "; Discovery Failed"
            browser.add_option(
                Option(
                    self._summary(
                        f"{self._friendly(name)}"
                        f"{'  Pending' if self.state and name in self.state.staged_providers else ''}"
                        f"  {status}  {count} runnable model"
                        f"{'s' if count != 1 else ''}"
                    ),
                    id=name,
                )
            )
        root_actions = self.query_one("#wb-root-actions", WorkbenchList)
        action_position = self._position_before_rebuild(
            WorkbenchView.PROVIDERS, "wb-root-actions"
        )
        root_actions.clear_options()
        root_actions.add_option(Option("ACTIONS", id="\x00actions", disabled=True))
        actions = [("\x00add", "Add custom provider"), ("\x00models", "Model catalog")]
        actions.append(("\x00presets", "Choose default presets"))
        if self.state is not None and self.state.dirty:
            actions.extend([
                ("\x00apply", "Apply all catalog edits to disk"),
                ("\x00discard", "Discard all pending catalog edits"),
            ])
        for key, label in actions:
            root_actions.add_option(Option(label, id=key))
        root_actions.highlighted = 1
        self._restore_group_position(action_position)
        ids = [str(option.id) for option in browser.options]
        preferred: str | None = None
        if (
            position is None
            and self.mode == "onboarding"
            and "mistral" in visible
            and self._mistral_needs_setup("mistral")
        ):
            preferred = "mistral"
        current_option = next(
            (option for option in browser.options if str(option.id) == current), None
        )
        current_actionable = bool(
            current_option
            and not current_option.disabled
            and current not in {"\x00heading", "\x00actions", "\x00empty"}
        )
        target = current if current_actionable else preferred
        browser.highlighted = ids.index(target) if target in ids else 1
        self.query_one("#wb-filter", NoMarkupStatic).update(
            f"Filter: {query or 'type to filter'}  ·  {self._provider_count_label(len(visible), len(names))}"
        )
        self._provider_visible_count = (len(visible), len(names))
        self.query_one("#wb-count", NoMarkupStatic).display = False
        self._restore_group_position(position)
        self._update_help()

    @staticmethod
    def _friendly(provider_id: str) -> str:
        return provider_id

    def _visible_provider_count(self) -> str:
        visible, total = self._provider_visible_count
        return self._provider_count_label(visible, total)

    @staticmethod
    def _provider_count_label(visible: int, total: int) -> str:
        if visible == total:
            return f"{total} provider{'s' if total != 1 else ''}"
        return f"{visible} of {total} providers"

    def _provider_env(self, name: str) -> str:
        provider = (
            self.state.catalog.providers.get(name) if self.state is not None else None
        ) or self.snapshot.catalog.providers.get(name)
        if self.state is not None:
            connection = self.state.connections.get(name)
            if connection is not None:
                return connection.api_key_env_var
        if self.state is not None and self.state.provider_id == name:
            return self.state.connection.api_key_env_var
        return provider.api_key_env_var if provider is not None else ""

    def _credential_ready_for_provider(self, name: str) -> bool:
        env_var = self._provider_env(name)
        return (
            not env_var
            or env_var in self._saved_credential_envs
            or bool(self.credential_resolver(env_var))
        )

    def _credential_value(self, env_var: str) -> str | None:
        return (
            "saved"
            if env_var in self._saved_credential_envs
            else self.credential_resolver(env_var)
        )

    def _api_key_status(self, env_var: str) -> str:
        if env_var in self._saved_credential_envs:
            return "Saved"
        if env_var and self.credential_resolver(env_var):
            return "From environment"
        return "Not set"

    def _runnable_provider_count(self, name: str) -> int:
        catalog = (
            self.state.catalog if self.state is not None else self.snapshot.catalog
        )
        provider = catalog.providers.get(name)
        if (
            provider is None
            or provider.disabled
            or not self._credential_ready_for_provider(name)
        ):
            return 0
        count = 0
        for base, model in catalog.models.items():
            if model.disabled:
                continue
            for dep in model.deployments:
                if dep.provider != name:
                    continue
                if self._deployment_enabled(base, dep):
                    count += 1
        if self.state is not None:
            count += sum(
                item.enabled
                for item in self.state.pending_by_provider.get(name, {}).values()
            )
        return count

    def _mistral_needs_setup(self, name: str) -> bool:
        shipped = SHIPPED_CATALOG.providers.get(name)
        return (
            name == "mistral"
            and name not in self.snapshot.overlaid_providers
            and self.snapshot.catalog.providers.get(name) == shipped
            and not self._credential_ready_for_provider(name)
        )

    def _deployment_enabled(self, base: str, deployment: DeploymentDefinition) -> bool:
        enabled = not deployment.disabled
        if self.state is not None:
            enabled = self.state.enabled_by_deployment.get(
                (base, deployment.provider), enabled
            )
        return enabled

    def _provider_has_enabled_model(self, name: str) -> bool:
        catalog = (
            self.state.catalog if self.state is not None else self.snapshot.catalog
        )
        configured = any(
            not model.disabled
            and dep.provider == name
            and self._deployment_enabled(base, dep)
            for base, model in catalog.models.items()
            for dep in model.deployments
        )
        pending = (
            any(
                item.enabled
                for item in self.state.pending_by_provider.get(name, {}).values()
            )
            if self.state is not None
            else False
        )
        return configured or pending

    def _summary(self, text: str, *, reserve: int = 8) -> str:
        width = max(1, self.size.width - reserve)
        if cell_len(text) <= width:
            return text
        result = ""
        for char in text:
            if cell_len(result + char) > width - 1:
                break
            result += char
        return result + chrome_glyph("truncation")

    def _open_catalog(
        self,
        provider: str | None = None,
        focus: str | None = None,
        *,
        push: bool = True,
    ) -> None:
        if push and self._view != WorkbenchView.CATALOG:
            self._push_view(WorkbenchView.CATALOG)
        self._models_view = True
        self._model_filter = provider
        self._detail_cursor = focus
        self._refresh_catalog(preserve_filter_cursor=False)
        self._activate_view(
            WorkbenchView.CATALOG,
            focus_id="wb-catalog",
            selected_id=f"model:{focus}" if focus else None,
        )

    def _catalog_model_summary(
        self, name: str, definition: BaseModelDefinition | None
    ) -> str:
        state = self.state
        catalog = state.catalog if state else self.snapshot.catalog
        deployments = definition.deployments if definition is not None else ()
        entries: list[tuple[str, bool, bool]] = []
        for deployment in deployments:
            provider = catalog.providers.get(deployment.provider)
            enabled = bool(
                definition is not None
                and not definition.disabled
                and self._deployment_enabled(name, deployment)
            )
            runnable = bool(
                enabled
                and provider is not None
                and not provider.disabled
                and self._credential_ready_for_provider(deployment.provider)
            )
            entries.append((deployment.provider, enabled, runnable))
        if state is not None:
            entries.extend(
                (
                    provider_id,
                    True,
                    bool(
                        (provider := catalog.providers.get(provider_id)) is not None
                        and not provider.disabled
                        and self._credential_ready_for_provider(provider_id)
                    ),
                )
                for provider_id, pending_by_wire in state.pending_by_provider.items()
                for item in pending_by_wire.values()
                if item.canonical_name == name and item.enabled
            )
        providers = (
            ", ".join(sorted({provider for provider, _, _ in entries})) or "pending"
        )
        roles = sorted(
            role
            for role, entry in catalog.roles.items()
            if name == (state.preset(role)[0] if state else entry.model)
        )
        enabled_count = sum(enabled for _, enabled, _ in entries)
        runnable_count = sum(runnable for _, _, runnable in entries)
        role_summary = ", ".join(roles) or "no presets"
        return self._summary(
            f"{name} ({providers}) · {enabled_count} enabled, "
            f"{runnable_count} runnable · {role_summary}"
        )

    def _refresh_catalog(self, *, preserve_filter_cursor: bool = True) -> None:  # noqa: PLR0914
        filters = self.query_one("#wb-catalog-filter", NavigableOptionList)
        filter_position = self._position_before_rebuild(
            WorkbenchView.CATALOG, "wb-catalog-filter"
        )
        position = self._position_before_rebuild(WorkbenchView.CATALOG, "wb-catalog")
        current_filter = f"filter:{self._model_filter or ''}"
        state = self.state
        previous_filter = (
            str(filters.highlighted_option.id)
            if preserve_filter_cursor
            and filters.highlighted_option
            and filters.highlighted_option.id is not None
            else None
        )
        previous_filter_index = filters.highlighted or 1
        filters.clear_options()
        filters.add_option(
            Option("PROVIDER FILTER", id="\x00provider-filter", disabled=True)
        )
        filters.add_option(Option("All Providers", id="filter:"))
        provider_ids = set(
            state.catalog.providers if state else self.snapshot.catalog.providers
        )
        if state is not None:
            provider_ids.add(state.provider_id)
            provider_ids.update(state.pending_by_provider)
        for provider in sorted(provider_ids):
            filters.add_option(Option(f"{provider}", id=f"filter:{provider}"))
        filter_ids = [str(option.id) for option in filters.options]
        filter_target = previous_filter or current_filter
        filters.highlighted = (
            filter_ids.index(filter_target)
            if filter_target in filter_ids
            else min(previous_filter_index, len(filter_ids) - 1)
        )
        catalog = self.query_one("#wb-catalog", NavigableOptionList)
        previous = (
            str(catalog.highlighted_option.id)
            if catalog.highlighted_option and catalog.highlighted_option.id is not None
            else None
        )
        previous_index = catalog.highlighted or 0
        catalog.clear_options()
        actions = self.query_one("#wb-catalog-actions", WorkbenchList)
        action_position = self._position_before_rebuild(
            WorkbenchView.CATALOG, "wb-catalog-actions"
        )
        actions.clear_options()
        if state is not None and state.dirty:
            actions.add_option(
                Option("Apply all catalog edits to disk", id="\x00apply")
            )
            actions.add_option(
                Option("Discard all pending catalog edits", id="\x00discard")
            )
        actions.highlighted = 0 if actions.option_count else None
        self._restore_group_position(action_position)
        models = state.catalog.models if state else self.snapshot.catalog.models
        pending = (
            {
                item.canonical_name
                for items in state.pending_by_provider.values()
                for item in items.values()
                if item.enabled
            }
            if state
            else set()
        )
        shown = 0
        for name in sorted(models.keys() | pending):
            definition = models.get(name)
            deployments = definition.deployments if definition else ()
            if (
                self._model_filter
                and not any(dep.provider == self._model_filter for dep in deployments)
                and not (
                    state
                    and any(
                        item.canonical_name == name
                        for item in state.pending_by_provider.get(
                            self._model_filter, {}
                        ).values()
                    )
                )
            ):
                continue
            shown += 1
            catalog.add_option(
                Option(
                    self._catalog_model_summary(name, definition), id=f"model:{name}"
                )
            )
        if not shown:
            catalog.add_option(Option("No models configured", id="\x00empty"))
        ids = [str(option.id) for option in catalog.options]
        target = previous or (
            f"model:{self._detail_cursor}" if self._detail_cursor else None
        )
        catalog.highlighted = (
            ids.index(target)
            if target in ids
            else (min(previous_index, len(ids) - 1) if ids else None)
        )
        self._restore_group_position(position)
        if preserve_filter_cursor:
            self._restore_group_position(filter_position)
        self._update_help()

    def _catalog_model_owners(self, name: str) -> list[str]:
        """Return every live provider that owns a catalog or pending model."""
        catalog = self.state.catalog if self.state else self.snapshot.catalog
        definition = catalog.models.get(name)
        owners = (
            {dep.provider for dep in definition.deployments} if definition else set()
        )
        if self.state:
            owners.update(
                provider_id
                for provider_id, entries in self.state.pending_by_provider.items()
                if any(
                    item.canonical_name == name and item.enabled
                    for item in entries.values()
                )
            )
        if self._model_filter:
            owners = {owner for owner in owners if owner == self._model_filter}
        return sorted(owners)

    def _select_catalog(self, key: str) -> None:
        if self._busy:
            return
        if key.startswith("filter:"):
            self._model_filter = key.removeprefix("filter:") or None
            self._detail_cursor = None
            self._refresh_catalog()
        elif key == "\x00provider-filter":
            return
        elif key == "\x00apply":
            self._request_commit()
        elif key == "\x00discard":
            self._confirm = "discard"
            self._update_help()
        elif key.startswith("model:"):
            name = key.removeprefix("model:")
            if self.mode == "onboarding":
                self.selected_model = name
            self._detail_cursor = name
            providers = self._catalog_model_owners(name)
            provider = providers[0] if len(providers) == 1 else None
            if provider:
                if (
                    not self._navigation
                    or self._navigation[-1].view != WorkbenchView.CATALOG
                ):
                    self._push_view(WorkbenchView.CATALOG)
                self._expand(provider)
                self._open_detail(name, push=False)
            elif len(providers) > 1:
                self._open_deployment_picker(name, providers)
            else:
                self._message = "This model has no live provider owner to edit; open its provider draft first."
                self._update_help()

    def _open_deployment_picker(self, name: str, providers: list[str]) -> None:
        """Choose the deployment that owns an ambiguous global model row."""
        self._push_view(WorkbenchView.DEPLOYMENTS)
        self._picker = True
        self._detail_cursor = name
        position = self._position_before_rebuild(WorkbenchView.DEPLOYMENTS, "wb-picker")
        picker = self.query_one("#wb-picker", NavigableOptionList)
        picker.clear_options()
        for provider in providers:
            picker.add_option(
                Option(f"{provider}/{name}", id=f"deployment:{provider}:{name}")
            )
        picker.highlighted = 0
        self._restore_group_position(position)
        self._activate_view(WorkbenchView.DEPLOYMENTS, focus_id="wb-picker")

    def _request_commit(self) -> None:
        if self._busy:
            return
        if self.state is None:
            first = next(iter(self.snapshot.catalog.providers), None)
            if first is None:
                self._set_feedback("catalog", "Configure a provider first.", "warning")
                return
            self.state = ManagementState.from_snapshot(self.snapshot, first)
        if self.mode == "onboarding":
            self._open_presets()
            return
        self._apply(self.state)

    def _open_presets(self) -> None:
        # Guided setup advances a stage; management presets are a nested view.
        self._presets_opener = self._frame() if self._stage is None else None
        if self.state is None:
            first = next(iter(self.snapshot.catalog.providers), None)
            if first is None:
                self._set_feedback("preset", "Connect a provider first.", "warning")
                return
            self.state = ManagementState.from_snapshot(self.snapshot, first)
        if self.mode == "onboarding":
            self._seed_onboarding_presets()
        self._stage = "presets" if self.mode == "onboarding" else None
        self._add = None
        self._preset_field = None
        self._preset_role = None
        self._customize_presets = False
        self._refresh_presets()
        self._activate_view(WorkbenchView.PRESETS, focus_id="wb-presets")
        self._defer_focus(self.query_one("#wb-presets", WorkbenchList))

    def _seed_onboarding_presets(self) -> None:
        """Seed usable roles; shipped dispatch slots resolve through these bindings.

        A single canonical model (even at different thinking levels) needs no
        explicit slot overlay. Keep the saved mode, including one-model orchestration.
        """
        state = self.state
        self._onboarding_presets_seeded = False
        if state is None:
            return
        changes = state._build()
        try:
            candidate = project_catalog_changes(state.snapshot, changes)
        except (ValueError, TypeError):
            candidate = state.snapshot
        configured = {
            provider_id
            for provider_id, provider in state.catalog.providers.items()
            if provider_id in self.snapshot.overlaid_providers
            or provider_id not in SHIPPED_CATALOG.providers
            or bool(
                provider.api_key_env_var
                and self._credential_value(provider.api_key_env_var)
            )
        }
        candidates = [
            (name, thinking)
            for name, definition in sorted(state.catalog.models.items())
            if (self.selected_model is None or name == self.selected_model)
            and not definition.disabled
            and any(
                deployment.provider in configured
                and not state.catalog.providers[deployment.provider].disabled
                for deployment in definition.deployments
            )
            for thinking in state.preset_thinking_levels(name)
            if state._preset_readiness(
                name,
                thinking,
                self._credential_value,
                candidate=candidate,
                allowed_models=self.allowed_models,
                thinking_overrides=self.thinking_overrides,
            )
            is None
        ]
        if not candidates:
            reason = None
            if self.selected_model is not None:
                reason = state._preset_readiness(
                    self.selected_model,
                    state.preset_thinking_levels(self.selected_model)[0]
                    if self.selected_model in state.catalog.models
                    and state.preset_thinking_levels(self.selected_model)
                    else "medium",
                    self._credential_value,
                    candidate=candidate,
                    allowed_models=self.allowed_models,
                    thinking_overrides=self.thinking_overrides,
                )
            self._set_feedback(
                "preset",
                f"Selected model {self.selected_model!r} is not ready: {reason}"
                if reason
                else "No configured, enabled model is ready. Connect a provider, enable a model, and provide its credential.",
                "warning",
            )
            return
        self._onboarding_presets_seeded = True
        for role in state.catalog.roles:
            model, thinking = state.preset(role)
            if (
                state._preset_readiness(
                    model,
                    thinking,
                    self._credential_value,
                    role=role,
                    candidate=candidate,
                    allowed_models=self.allowed_models,
                    thinking_overrides=self.thinking_overrides,
                )
                is None
            ):
                continue
            replacement = next(
                (candidate for candidate in candidates if candidate[1] == thinking),
                next(
                    (candidate for candidate in candidates if candidate[0] == model),
                    candidates[0],
                ),
            )
            state.set_role_preset(role, *replacement)

    def _refresh_presets(self, *, highlight: str | None = None) -> None:
        state = self.state
        if state is None:
            return
        rows = self.query_one("#wb-presets", WorkbenchList)
        position = self._position_before_rebuild(WorkbenchView.PRESETS, "wb-presets")
        previous = highlight or (
            str(rows.highlighted_option.id) if rows.highlighted_option else None
        )
        rows.clear_options()
        rows.add_option(Option("DEFAULT PRESETS", id="heading", disabled=True))
        summary = self.mode == "onboarding" and not self._customize_presets
        if summary:
            models = sorted({state.preset(role)[0] for role in state.catalog.roles})
            rows.add_option(
                Option(
                    self._summary(
                        f"Roles bound automatically to {', '.join(models)}"
                        if self._onboarding_presets_seeded
                        else "Role presets need repair; no ready configured model found"
                    ),
                    id="summary",
                    disabled=True,
                )
            )
            rows.add_option(Option("Customize role presets", id="customize"))
        for dispatch_mode in DispatchMode:
            label = (
                "Standalone — direct implementation"
                if dispatch_mode == DispatchMode.STANDALONE
                else "Orchestrated — including with one model"
            )
            marker = chrome_glyph(
                "radio_selected"
                if state._dispatch.mode == dispatch_mode
                else "radio_empty"
            )
            rows.add_option(
                Option(
                    self._summary(f"{marker} {label}"), id=f"mode:{dispatch_mode.value}"
                )
            )
        rows.add_option(
            Option("Mode changes apply next session", id="next-session", disabled=True)
        )
        ordered_roles = [
            role
            for role in ("orchestrator", "worker", "scout", "heavy")
            if role in state.catalog.roles
        ]
        ordered_roles.extend(sorted(set(state.catalog.roles) - set(ordered_roles)))
        for role in () if summary else ordered_roles:
            model, thinking = state.preset(role)
            title = {
                "orchestrator": "Main assistant (@orchestrator)",
                "worker": "Worker (@worker)",
                "scout": "Scout (@scout)",
                "heavy": "Heavy (@heavy)",
            }.get(role, f"@{role}")
            rows.add_option(
                Option(
                    self._summary(f"{title}  {model} · thinking {thinking}"),
                    id=f"preset:{role}",
                )
            )
        actions = self.query_one("#wb-presets-actions", WorkbenchList)
        action_position = self._position_before_rebuild(
            WorkbenchView.PRESETS, "wb-presets-actions"
        )
        actions.clear_options()
        actions.add_option(
            Option(
                "Finish setup" if self.mode == "onboarding" else "Save presets",
                id="finish",
            )
        )
        actions.add_option(Option("Add another provider", id="add-another"))
        actions.highlighted = 0
        self._restore_group_position(action_position)
        ids = [str(option.id) for option in rows.options]
        rows.highlighted = (
            ids.index(previous) if previous in ids else (2 if summary else 1)
        )
        if highlight is None:
            self._restore_group_position(position)
        self._update_help()

    def _select_preset_action(self, key: str) -> None:  # noqa: PLR0911, PLR0912
        state = self.state
        if state is None or self._busy:
            return
        if key == "customize":
            self._customize_presets = True
            self._refresh_presets(highlight="preset:orchestrator")
            return
        if key.startswith("mode:"):
            state.dispatch_mode = DispatchMode(key.partition(":")[2])
            self._set_feedback(
                "preset", "Mode staged; save to apply next session.", "info"
            )
            self._refresh_presets(highlight=key)
            return
        if key == "add-another":
            if state.dirty:
                self._after_commit = "another"
                self._start_commit(state, create=False)
            else:
                self._start_add()
            return
        if key == "finish":
            if self.mode == "onboarding":
                validation = state.validate(
                    mode="onboarding",
                    credential_resolver=self._credential_value,
                    allowed_models=self.allowed_models,
                    thinking_overrides=self.thinking_overrides,
                )
                if validation.errors:
                    self._customize_presets = True
                    self._set_feedback("preset", "; ".join(validation.errors), "error")
                    self._refresh_presets(
                        highlight=f"preset:{(validation.unresolved_roles or validation.unusable_roles or ('orchestrator',))[0]}"
                    )
                    self.query_one("#wb-presets").focus()
                    return
            else:
                for role, pair in state.role_presets.items():
                    definition = state.catalog.roles[role]
                    if pair == (definition.model, definition.thinking):
                        continue
                    error = state.validate_preset(
                        role,
                        self._credential_value,
                        allowed_models=self.allowed_models,
                        thinking_overrides=self.thinking_overrides,
                    )
                    if error:
                        self._set_feedback("preset", error, "error")
                        self._refresh_presets(highlight=f"preset:{role}")
                        self.query_one("#wb-presets").focus()
                        return
            if state.dirty:
                self._after_commit = (
                    "finish" if self.mode == "onboarding" else "presets"
                )
                self._start_commit(state, create=False)
            elif self.mode == "onboarding":
                self._setup_ready = True
                self._finish()
            else:
                self._set_feedback("preset", "Presets are already saved.", "info")
            return
        field, _, role = key.partition(":")
        if field != "preset" or role not in state.catalog.roles:
            return
        self._push_view(WorkbenchView.PRESET_EDITOR)
        self._preset_role = role
        self._preset_draft = state.preset(role)
        self._preset_field = None
        self._refresh_preset_editor()
        self._activate_view(WorkbenchView.PRESET_EDITOR, focus_id="wb-preset-editor")
        self._defer_focus(self.query_one("#wb-preset-editor", WorkbenchList))

    def _refresh_preset_editor(self, *, highlight: str | None = None) -> None:
        if self._preset_role is None or self._preset_draft is None:
            return
        rows = self.query_one("#wb-preset-editor", WorkbenchList)
        position = self._position_before_rebuild(
            WorkbenchView.PRESET_EDITOR, "wb-preset-editor"
        )
        previous = highlight or (
            str(rows.highlighted_option.id) if rows.highlighted_option else "model"
        )
        model, thinking = self._preset_draft
        supported = self.state.preset_thinking_levels(model) if self.state else ()
        rows.clear_options()
        rows.add_option(Option(self._summary(f"Model  {model}"), id="model"))
        rows.add_option(
            Option(
                self._summary(
                    f"Thinking  {thinking}{' (unsupported)' if supported and thinking not in supported else ''}"
                ),
                id="thinking",
            )
        )
        actions = self.query_one("#wb-preset-editor-actions", WorkbenchList)
        action_position = self._position_before_rebuild(
            WorkbenchView.PRESET_EDITOR, "wb-preset-editor-actions"
        )
        actions.clear_options()
        actions.add_option(Option("Apply model and thinking", id="apply"))
        actions.add_option(Option("Cancel", id="cancel"))
        actions.highlighted = 0
        self._restore_group_position(action_position)
        ids = [str(option.id) for option in rows.options]
        rows.highlighted = ids.index(previous) if previous in ids else 0
        if highlight is None:
            self._restore_group_position(position)

    def _select_preset_editor(self, key: str) -> None:
        if (
            self.state is None
            or self._preset_role is None
            or self._preset_draft is None
        ):
            return
        if key == "cancel":
            self._close_preset_editor()
            return
        if key == "apply":
            model, thinking = self._preset_draft
            supported = self.state.preset_thinking_levels(model)
            if thinking not in supported:
                self._set_feedback(
                    "preset",
                    f"@{self._preset_role}: {model} supports {', '.join(supported) or 'no thinking levels'}; choose a supported level before Apply.",
                    "error",
                )
                self._refresh_preset_editor(highlight="thinking")
                self.query_one("#wb-preset-editor").focus()
                return
            role = self._preset_role
            self.state.set_role_preset(role, model, thinking)
            self._close_preset_editor()
            return
        if key not in {"model", "thinking"}:
            return
        model, current_thinking = self._preset_draft
        supported = (
            self.state.preset_thinking_levels(model) if key == "thinking" else ()
        )
        if key == "thinking" and not supported:
            self._set_feedback(
                "preset",
                f"@{self._preset_role}: {model} has no enabled deployment. Choose a model first.",
                "warning",
            )
            return
        self._push_view(WorkbenchView.PICKER)
        self._preset_field = key
        position = self._position_before_rebuild(WorkbenchView.PICKER, "wb-picker")
        picker = self.query_one("#wb-picker", WorkbenchList)
        picker.clear_options()
        if key == "model":
            names = set(self.state.catalog.models) | {
                item.canonical_name
                for pending in self.state.pending_by_provider.values()
                for item in pending.values()
                if item.enabled
            }
            for name in sorted(names):
                picker.add_option(
                    Option(f"{name}{'  (current)' if name == model else ''}", id=name)
                )
        else:
            if current_thinking not in supported:
                picker.add_option(
                    Option(
                        f"Current {current_thinking} (unsupported) — choose a supported level",
                        id="\x00unsupported",
                        disabled=True,
                    )
                )
            for level in supported:
                picker.add_option(
                    Option(
                        f"{level}{'  (current)' if level == current_thinking else ''}",
                        id=level,
                    )
                )
        picker.highlighted = next(
            (i for i, option in enumerate(picker.options) if not option.disabled), None
        )
        self._restore_group_position(position)
        self._activate_view(WorkbenchView.PICKER, focus_id="wb-picker")
        self._defer_focus(picker)

    def _close_preset_editor(self) -> None:
        self._capture_group_position()
        role = self._preset_role
        self._preset_field = None
        self._preset_draft = None
        self._preset_role = None
        self._refresh_presets(highlight=f"preset:{role}")
        if not self._pop_view():
            self._activate_view(
                WorkbenchView.PRESETS,
                focus_id="wb-presets",
                selected_id=f"preset:{role}",
            )

    def _choose_preset_value(self, value: str) -> None:
        if (
            self.state is None
            or self._preset_role is None
            or self._preset_field is None
        ):
            return
        role, field = self._preset_role, self._preset_field
        model, thinking = self._preset_draft or self.state.preset(role)
        candidate = value if field == "model" else model
        self._preset_draft = (candidate, value if field == "thinking" else thinking)
        if field == "model":
            supported = self.state.preset_thinking_levels(candidate)
            if supported and thinking not in supported:
                self._set_feedback(
                    "preset",
                    f"@{role}: {candidate} supports {', '.join(supported)}; choose a supported thinking level before Apply.",
                    "warning",
                )
        self._capture_group_position()
        self._preset_field = None
        self._refresh_preset_editor(highlight=field)
        if not self._pop_view():
            self._activate_view(
                WorkbenchView.PRESET_EDITOR,
                focus_id="wb-preset-editor",
                selected_id=field,
            )

    def _refresh_actions(self, *, highlight: str | None = None) -> None:
        if self._stage == "models":
            self._refresh_add_models(highlight=highlight)
            return
        actions = self.query_one("#wb-actions", NavigableOptionList)
        position = self._position_before_rebuild(WorkbenchView.ACTIONS, "wb-actions")
        operations = self.query_one("#wb-provider-operations", WorkbenchList)
        operation_position = self._position_before_rebuild(
            WorkbenchView.ACTIONS, "wb-provider-operations"
        )
        previous_operation = highlight or (
            str(operations.highlighted_option.id)
            if operations.highlighted_option
            else None
        )
        operations.clear_options()
        previous = highlight or (
            str(actions.highlighted_option.id) if actions.highlighted_option else None
        )
        actions.clear_options()
        state = self.state
        if state is None:
            return
        for key, label, value in (
            ("base", "API base", state.connection.api_base or "Not set"),
            ("style", "API style", state.connection.api_style),
            (
                "env",
                "Credential env var",
                state.connection.api_key_env_var or "Not required",
            ),
            (
                "key",
                "API key —",
                self._api_key_status(state.connection.api_key_env_var),
            ),
            (
                "discover",
                "Discover models",
                "Retry" if isinstance(state.discovery, DiscoveryError) else "",
            ),
            ("models", "Models", f"{len(state.model_rows())} available"),
        ):
            target = operations if key in {"discover", "models"} else actions
            target.add_option(Option(self._summary(f"{label}  {value}"), id=key))
        ids = [str(option.id) for option in actions.options]
        actions.highlighted = ids.index(previous) if previous in ids else 0
        if highlight is None or highlight not in ids:
            self._restore_group_position(position)
        actions = operations
        previous = previous_operation
        for wire, item in state.pending.items():
            if item.enabled:
                outcome = match_discovered_model(state.catalog, state.provider_id, wire)
                if outcome.kind in {
                    "base_exists_other_provider",
                    "occupied_slot",
                    "multiple_matches",
                }:
                    reason = item.decision or (
                        "existing provider deployment"
                        if outcome.kind == "occupied_slot"
                        else outcome.kind
                    )
                    actions.add_option(
                        Option(
                            f"Resolve {wire} {chrome_glyph('forward')} {item.canonical_name} ({reason})",
                            id=f"collision:{wire}",
                        )
                    )
        actions.add_option(
            Option(
                f"Apply all catalog edits to disk{' *' if state.dirty else ''}",
                id="apply",
            )
        )
        actions.add_option(Option("Choose default presets", id="presets"))
        actions.add_option(Option("Discard all pending catalog edits", id="discard"))
        if self._reload_failed:
            actions.add_option(Option("Retry Runtime Reload", id="retry-reload"))
        ids = [str(option.id) for option in actions.options]
        actions.highlighted = ids.index(previous) if previous in ids else 0
        if highlight is None or highlight not in ids:
            self._restore_group_position(operation_position)
        self._update_help()

    def _refresh_models(self, *, highlight: str | None = None) -> None:
        state = self.state
        if state is None:
            return
        models = self.query_one("#wb-models", ModelChecklist)
        position = self._position_before_rebuild(WorkbenchView.MODELS, "wb-models")
        action_position = self._position_before_rebuild(
            WorkbenchView.MODELS, "wb-models-actions"
        )
        previous = highlight or (
            models.get_option_at_index(models.highlighted).value
            if models.highlighted is not None and models.option_count
            else None
        )
        previous_index = models.highlighted or 0
        self._model_sync = True
        models.clear_options()
        rows = state.model_rows()
        configured = state.configured()
        self._model_row_identities = tuple(
            f"canonical:{name}"
            if name in configured and configured[name].name == wire
            else f"wire:{wire}"
            for name, wire, _enabled, _found in rows
        )
        model_actions = self.query_one("#wb-models-actions", WorkbenchList)
        previous_action = (
            str(model_actions.highlighted_option.id)
            if model_actions.highlighted_option
            else None
        )
        model_actions.clear_options()
        if self._stage == "models":
            for label, action in (
                ("Retry discovery", "retry-discovery"),
                ("Edit connection", "edit-connection"),
                ("Add model manually", "manual"),
                ("Save and add another provider", "add-another"),
                ("Save and continue to presets", "continue-presets"),
            ):
                model_actions.add_option(Option(label, id=action))
            action_ids = [str(option.id) for option in model_actions.options]
            model_actions.highlighted = (
                action_ids.index(previous_action)
                if previous_action in action_ids
                else 0
            )
        for name, wire, enabled, found in rows:
            models.add_option(
                Selection(
                    self._summary(
                        f"{state.provider_id}/{wire} ({name}){' — not discovered' if not found else ''}"
                    ),
                    name,
                    enabled,
                )
            )
        if not rows:
            models.add_option(Selection("No models configured", "\x00empty", False))
        ids = [str(cast(Selection[str], option).value) for option in models.options]
        models.highlighted = (
            ids.index(previous)
            if previous in ids
            else min(previous_index, len(ids) - 1)
        )
        if highlight is None:
            self._restore_group_position(position)
        self._restore_group_position(action_position)
        self._model_sync = False
        self._update_help()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_list.id == "wb-providers" and event.option.id is not None:
            self._browser_cursor = str(event.option.id)
        self._update_help()

    def on_option_list_option_selected(  # noqa: PLR0911, PLR0912, PLR0915
        self, event: OptionList.OptionSelected
    ) -> None:
        if not self._can_activate(event.option_list.id) or event.option.id is None:
            return
        if self._confirm:
            if event.option_list.id == "wb-confirm-actions":
                if event.option.id == "accept":
                    self.action_confirm_yes()
                elif event.option.id == "continue":
                    self.action_confirm_continue()
                elif event.option.id == "save-detail":
                    self.action_confirm_detail_save()
                elif event.option.id == "discard-detail":
                    self.action_confirm_detail_discard()
                else:
                    self.action_confirm_no()
            return
        key = str(event.option.id)
        if event.option_list.id in {"wb-providers", "wb-root-actions"}:
            if key == "\x00empty":
                return
            if key == "\x00add":
                self._choose_preset(FULLY_CUSTOM)
                return
            if key == "\x00models":
                self._open_catalog()
                return
            if key == "\x00presets":
                self._open_presets()
                return
            if key == "\x00finish":
                self._request_commit()
                return
            if key == "\x00apply":
                self._request_commit()
                return
            if key == "\x00discard":
                self._confirm = "discard"
                self._update_help()
                return
            if key == "mistral" and self._mistral_needs_setup(key):
                self._choose_preset(MISTRAL)
                return
            self._expand(key)
        elif event.option_list.id == "wb-choose":
            if key.startswith("existing:"):
                self._stage = None
                self._expand(key.removeprefix("existing:"))
            else:
                preset = next(p for p in PRESETS if p.id == key)
                self._choose_preset(preset)
        elif event.option_list.id == "wb-protocol":
            self._select_protocol(key)
        elif event.option_list.id in {"wb-presets", "wb-presets-actions"}:
            self._select_preset_action(key)
        elif event.option_list.id in {"wb-preset-editor", "wb-preset-editor-actions"}:
            self._select_preset_editor(key)
        elif event.option_list.id == "wb-detail-fields":
            self._select_detail_field(key)
        elif event.option_list.id in {
            "wb-catalog",
            "wb-catalog-filter",
            "wb-catalog-actions",
        }:
            self._select_catalog(key)
        elif event.option_list.id == "wb-picker":
            if self._preset_field is not None:
                self._choose_preset_value(key)
                return
            if key == "\x00repair":
                self._close_picker()
                self._open_catalog()
                return
            if key.startswith("deployment:"):
                _prefix, provider, name = key.split(":", 2)
                self._picker = False
                if (
                    self._navigation
                    and self._navigation[-1].view == WorkbenchView.DEPLOYMENTS
                ):
                    frame = self._navigation[-1]
                    self._navigation[-1] = replace(frame, view=WorkbenchView.CATALOG)
                self._expand(provider)
                self._open_detail(name, push=False)
                return
            if key not in {"\x00empty", "\x00unavailable"}:
                self._set_feedback(
                    "picker", "Choose a preset field to edit.", "warning"
                )
        elif event.option_list.id in {"wb-actions", "wb-provider-operations"}:
            if self._stage == "connection":
                self._connection_action(key)
            else:
                self._select_action(key)
        elif event.option_list.id == "wb-connection-style":
            if self._add and key in {"openai", "openai-responses", "anthropic"}:
                self._add = replace(self._add, api_style=key)
                self._refresh_connection_form()
                self.set_focus(self.query_one("#wb-connection-env", Input))
        elif event.option_list.id == "wb-connection-actions":
            if key == "continue":
                self._connection_action("continue")
        elif event.option_list.id == "wb-detail-actions" and key == "save-detail":
            self._save_detail()
        elif event.option_list.id == "wb-models-actions":
            if key in {"create", "add-another", "continue-presets"}:
                self._advance_models("continue-presets" if key == "create" else key)
            elif self.state is not None:
                if key == "retry-discovery":
                    self.run_worker(
                        self._discovery_request(self.state),
                        group="workbench-discovery",
                        exclusive=True,
                    )
                else:
                    self._select_add_action(key, self.state)

    def _start_add(self) -> None:
        self._add_opener = (
            self._frame() if self._view == WorkbenchView.PROVIDERS else None
        )
        self._pre_add_state = deepcopy(self.state) if self.state is not None else None
        self._provider_open = False
        self._stage = "choose"
        self._view = WorkbenchView.CHOOSE
        self._add = None
        self._add_checkpoint = None
        self._add_state = None
        if self.is_mounted:
            self.query_one("#wb-connection-key", Input).value = ""
        if self._feedback_kind not in {"error", "warning"}:
            self._message = ""
        choose = self.query_one("#wb-choose", NavigableOptionList)
        choose.clear_options()
        for name in sorted(self.snapshot.catalog.providers):
            choose.add_option(
                Option(
                    f"Existing: {self._friendly(name)} ({name})", id=f"existing:{name}"
                )
            )
        for preset in PRESETS:
            choose.add_option(Option(f"{preset.name}", id=preset.id))
        choose.highlighted = 0
        self._activate_view(WorkbenchView.CHOOSE, focus_id="wb-choose")
        self._update_help()

    def _choose_preset(self, preset: ProviderPreset) -> None:
        if self._view == WorkbenchView.PROVIDERS:
            self._add_opener = self._frame()
        if self.state is not None and self._add is None and self._pre_add_state is None:
            self._pre_add_state = deepcopy(self.state)
        name = (
            ""
            if preset is FULLY_CUSTOM
            else preset.id
            if preset is MISTRAL
            else preset.name
        )
        existing = (
            self.snapshot.catalog.providers.get("mistral")
            if preset is MISTRAL and "mistral" in self.snapshot.overlaid_providers
            else None
        )
        self._mistral_overwrite_approved = False
        self._draft_session += 1
        self._add = ProviderDraft(
            preset.id,
            "",
            name,
            existing.api_base if existing else preset.api_base or "",
            existing.api_style if existing else preset.api_style or "openai",
            existing.api_key_env_var if existing else preset.api_key_env_var or "",
            None,
            existing.backend if existing else preset.backend,
            existing.reasoning_field_name if existing else preset.reasoning_field_name,
            dict(existing.extra_headers) if existing else {},
        )
        self._add_checkpoint = deepcopy(self._add)
        if self.is_mounted:
            self.query_one("#wb-connection-key", Input).value = ""
        self._stage = "connection"
        self._view = WorkbenchView.CONNECTION
        self._refresh_add_connection()
        self._activate_view(WorkbenchView.CONNECTION, focus_id="wb-actions")

    def _refresh_add_connection(
        self, *, highlight: str | None = None, selected_id: str | None = None
    ) -> None:
        draft = self._add
        if draft is None:
            return
        actions = self.query_one("#wb-actions", NavigableOptionList)
        position = self._position_before_rebuild(WorkbenchView.CONNECTION, "wb-actions")
        previous = (
            highlight
            or selected_id
            or (
                str(actions.highlighted_option.id)
                if actions.highlighted_option
                else None
            )
        )
        actions.clear_options()
        for key, label, value in (
            ("name", "Name", draft.name or "Not set"),
            ("base", "API base", draft.api_base or "Not set"),
            ("style", "API style", draft.api_style),
            ("env", "Credential env var", draft.api_key_env_var or "Not required"),
        ):
            actions.add_option(Option(self._summary(f"{label}  {value}"), id=key))
        actions.add_option(
            Option(
                self._summary(
                    f"API key — {self._api_key_status(draft.api_key_env_var)}"
                ),
                id="key",
            )
        )
        ids = [str(option.id) for option in actions.options]
        actions.highlighted = ids.index(previous) if previous in ids else 0
        self._refresh_connection_form()
        if highlight is None and selected_id is None:
            self._restore_group_position(position)
        self._update_help()

    def _refresh_connection_form(self) -> None:
        draft = self._add
        if draft is None or not self.is_mounted:
            return
        for widget_id, value in (
            ("#wb-connection-name", draft.name),
            ("#wb-connection-base", draft.api_base),
            ("#wb-connection-env", draft.api_key_env_var),
        ):
            self.query_one(widget_id, Input).value = value
        style = self.query_one("#wb-connection-style", WorkbenchList)
        style.clear_options()
        for value in ("openai", "openai-responses", "anthropic"):
            style.add_option(
                Option(
                    f"{value}{'  (current)' if value == draft.api_style else ''}",
                    id=value,
                )
            )
        style.highlighted = next(
            (
                i
                for i, option in enumerate(style.options)
                if option.id == draft.api_style
            ),
            0,
        )
        self._refresh_connection_actions()

    def _refresh_connection_actions(self) -> None:
        draft = self._add
        if draft is None:
            return
        actions = self.query_one("#wb-connection-actions", WorkbenchList)
        position = self._position_before_rebuild(
            WorkbenchView.CONNECTION, "wb-connection-actions"
        )
        current = actions.highlighted_option.id if actions.highlighted_option else None
        actions.clear_options()
        actions.add_option(Option("Save and configure models", id="continue"))
        ids = [option.id for option in actions.options]
        actions.highlighted = ids.index(current) if current in ids else len(ids) - 1
        self._restore_group_position(position)

    def _connection_action(self, key: str) -> None:
        if self._busy:
            return
        draft = self._add
        if draft is None:
            return
        if key == "style":
            self._open_protocol_picker(draft.api_style)
            return
        if key == "continue":
            try:
                provider_id = (
                    draft.provider_id
                    if draft.preset == "existing"
                    else self._new_provider_id(draft.name)
                )
                definition = ProviderDefinition.model_validate({
                    **(
                        self.snapshot.catalog.providers[provider_id].model_dump()
                        if provider_id == "mistral"
                        and provider_id in self.snapshot.catalog.providers
                        else {}
                    ),
                    "api_base": draft.api_base,
                    "api_style": draft.api_style,
                    "api_key_env_var": draft.api_key_env_var,
                    "backend": draft.backend,
                    "reasoning_field_name": draft.reasoning_field_name,
                    "extra_headers": dict(draft.extra_headers),
                })
                if draft.backend == "mistral" and not draft.api_base.rstrip(
                    "/"
                ).endswith("/v1"):
                    raise ValueError("Mistral requires an API base ending in /v1.")
            except ValueError as exc:
                self._set_feedback("field", str(exc), "error")
                field = (
                    "name"
                    if not draft.name.strip() or "name" in str(exc).lower()
                    else "base"
                    if "base" in str(exc).lower() or "url" in str(exc).lower()
                    else "env"
                )
                self._refresh_add_connection(highlight=field)
                self._update_help()
                return
            if self._add_state and self._add_state.provider_id != provider_id:
                self._set_feedback(
                    "field",
                    "Keep the original name to preserve selected models, or create this provider first and add another.",
                    "error",
                )
                self._refresh_add_connection(highlight="name")
                self._update_help()
                return
            if self._pre_add_state is None and self.state is not None:
                self._pre_add_state = deepcopy(self.state)
            self._add = replace(draft, provider_id=provider_id, name=provider_id)
            if self._view == WorkbenchView.CONNECTION and (
                not self._navigation
                or self._navigation[-1].view != WorkbenchView.CONNECTION
            ):
                self._push_view(WorkbenchView.CONNECTION)
            if self.state is None:
                self.state = ManagementState.from_snapshot(self.snapshot, provider_id)
            if provider_id in self.snapshot.catalog.providers:
                self.state.for_provider(provider_id)
                self.state.set_connection(ConnectionDraft.from_definition(definition))
            else:
                self.state.stage_provider(provider_id, definition)
            self._add_state = self.state
            existing_edit = draft.preset == "existing"
            self._connection_pending = (
                deepcopy(self.state.pending) if existing_edit else None
            )
            self._after_commit = "models"
            if existing_edit:
                self._connection_approved = True
            self._start_commit(self.state, create=not existing_edit)
            return
        if key in {"name", *self.FIELDS, "key"}:
            self._open_connection_editor(key, draft)

    def _open_connection_editor(self, key: str, draft: ProviderDraft) -> None:
        if key == "key" and not draft.api_key_env_var:
            self._key_after_env = True
            key = "env"
        self._editing = key
        editor = self.query_one("#wb-input", Input)
        editor.password = key == "key"
        editor.value = (
            "" if key == "key" else str(getattr(draft, self.FIELDS.get(key, key)))
        )
        if key == "env" and self._key_after_env and not editor.value:
            normalized = re.sub(r"[^A-Z0-9]+", "_", draft.name.upper()).strip("_")
            editor.value = f"{normalized or 'PROVIDER'}_API_KEY"
        self.query_one("#wb-editor-label", Label).update(
            {
                "name": "Provider Name *",
                "base": "API Base *",
                "style": "API Style *",
                "env": "Credential Env Var (for API key)"
                if self._key_after_env
                else "Credential Env Var",
                "key": "API Key *",
            }[key]
        )
        self._sync_view()
        self._restore_view_focus("wb-input")
        self._update_help()

    def _refresh_protocol(self) -> None:
        picker = self.query_one("#wb-protocol", NavigableOptionList)
        current = (
            str(picker.highlighted_option.id) if picker.highlighted_option else None
        )
        values = ("openai", "openai-responses", "anthropic")
        picker.clear_options()
        for value in values:
            picker.add_option(
                Option(
                    f"{chrome_glyph('radio_selected') if value == self._protocol_value else chrome_glyph('radio_empty')} {value}",
                    id=value,
                )
            )
        picker.highlighted = (
            values.index(current)
            if current in values
            else values.index(self._protocol_value or "openai")
        )

    def _select_protocol(self, value: str) -> None:
        if self._can_activate("wb-protocol"):
            self._protocol_value = value
            self._refresh_protocol()

    def _accept_protocol(self) -> None:
        if not self._can_activate("wb-protocol"):
            return
        value = self._protocol_value
        self._protocol_picker = False
        if value and self._stage == "connection" and self._add:
            self._add = replace(self._add, api_style=value)
            self._refresh_add_connection(highlight="style")
        elif value and self.state:
            self.state.set_connection(replace(self.state.connection, api_style=value))
            self._refresh_actions(highlight="style")
        if not self._pop_view():
            self._activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
        self._update_help()

    def _open_protocol_picker(self, current: str) -> None:
        if not self._protocol_picker:
            self._push_view(WorkbenchView.PROTOCOL)
        self._protocol_value = current
        self._refresh_protocol()
        self._protocol_picker = True
        self._activate_view(WorkbenchView.PROTOCOL, focus_id="wb-protocol")
        self._update_help()

    def _new_provider_id(self, name: str) -> str:
        provider_id = valid_provider_name(name)
        if (
            self.state is not None
            and provider_id in self.state.staged_providers
            and (self._add_state is None or self._add_state.provider_id != provider_id)
        ):
            raise ValueError(
                f"Provider name {provider_id!r} already exists in the draft."
            )
        if provider_id in self.snapshot.catalog.providers and not (
            provider_id == "mistral" and self._add and self._add.preset == "mistral"
        ):
            raise ValueError(f"Provider name {provider_id!r} already exists.")
        return provider_id

    def _refresh_add_models(self, *, highlight: str | None = None) -> None:
        state = self.state
        if state is None:
            return
        actions = self.query_one("#wb-actions", NavigableOptionList)
        position = self._position_before_rebuild(WorkbenchView.ACTIONS, "wb-actions")
        previous = highlight or (
            str(actions.highlighted_option.id) if actions.highlighted_option else None
        )
        actions.clear_options()
        status = (
            "Discovery Failed"
            if isinstance(state.discovery, DiscoveryError)
            else f"Discovered {len(state.discovery.models)} models"
            if isinstance(state.discovery, DiscoveryResult)
            else f"Discovering models{chrome_glyph('running')}"
        )
        actions.add_option(Option(f"{status}", id="status", disabled=True))
        for key, label in (
            ("discover", "Retry"),
            ("edit-connection", "Edit Connection"),
            ("edit-key", "Edit Key"),
            ("manual", "Add Model Manually"),
            ("models", f"Models ({len(state.model_rows())})"),
        ):
            if key != "edit-key" or state.connection.api_key_env_var:
                actions.add_option(Option(f"{label}", id=key))
        for wire, item in state.pending.items():
            if item.enabled:
                outcome = match_discovered_model(state.catalog, state.provider_id, wire)
                if outcome.kind in {
                    "base_exists_other_provider",
                    "occupied_slot",
                    "multiple_matches",
                }:
                    reason = item.decision or (
                        "existing provider deployment"
                        if outcome.kind == "occupied_slot"
                        else outcome.kind
                    )
                    actions.add_option(
                        Option(
                            f"Resolve {wire} {chrome_glyph('forward')} {item.canonical_name} ({reason})",
                            id=f"collision:{wire}",
                        )
                    )
        actions.add_option(Option("Save and add another provider", id="add-another"))
        actions.add_option(
            Option("Save and choose default presets", id="continue-presets")
        )
        if self._reload_failed:
            actions.add_option(Option("Retry Runtime Reload", id="retry-reload"))
        ids = [str(option.id) for option in actions.options]
        actions.highlighted = ids.index(previous) if previous in ids else 1
        if highlight is None:
            self._restore_group_position(position)
        self._update_help()

    def _advance_models(self, action: str) -> None:
        if self.state is None or action not in {"add-another", "continue-presets"}:
            return
        destination = "another" if action == "add-another" else "presets"
        if not self.state.dirty:
            if destination == "another":
                self._start_add()
            else:
                self._open_presets()
            return
        self._after_commit = destination
        self._start_commit(self.state, create=False)

    def _close_picker(self) -> None:
        self._picker = False
        if self._pop_view():
            self._update_help()
            return
        view = (
            WorkbenchView.CATALOG
            if self._models_view
            else WorkbenchView.ACTIONS
            if self.state
            else WorkbenchView.PROVIDERS
        )
        self._activate_view(view)
        self._update_help()

    def _expand(self, provider_id: str, *, committed: bool = False) -> None:
        if self._busy and not committed:
            return
        if not committed and self._view == WorkbenchView.PROVIDERS:
            self._push_view(WorkbenchView.PROVIDERS)
        if self.state is not None:
            self.state.begin_discovery()
            if provider_id in self.state.catalog.providers:
                self.state.for_provider(provider_id)
            else:
                self.state = ManagementState.from_snapshot(self.snapshot, provider_id)
        else:
            self.state = ManagementState.from_snapshot(self.snapshot, provider_id)
        self._confirm = None
        if self._feedback_kind not in {"error", "warning"}:
            self._message = ""
        self._connection_approved = False
        self._models_list_open = False
        self._provider_open = True
        self._refresh_actions()
        self._activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
        self._update_help()

    def _select_add_action(self, key: str, state: ManagementState) -> bool:
        if key == "retry-reload":
            self._retry_reload()
            return True
        if key in {"create", "add-another", "continue-presets"}:
            self._advance_models("continue-presets" if key == "create" else key)
        elif key == "edit-connection":
            self._begin_saved_connection_edit(state)
        elif key == "edit-key":
            self._select_action("key")
        elif key == "manual":
            self._editing = "manual"
            editor = self.query_one("#wb-input", Input)
            editor.password = False
            editor.value = ""
            self.query_one("#wb-editor-label", Label).update("Model ID *")
            self._sync_view()
            self._restore_view_focus("wb-input")
            self._update_help()
        else:
            return key == "status"
        return True

    def _begin_saved_connection_edit(self, state: ManagementState) -> None:
        connection = state.connection
        self._draft_session += 1
        self._add = ProviderDraft(
            "existing",
            state.provider_id,
            state.provider_id,
            connection.api_base,
            connection.api_style,
            connection.api_key_env_var,
            None,
            connection.backend,
            connection.reasoning_field_name,
            dict(connection.extra_headers),
        )
        self._add_checkpoint = deepcopy(self._add)
        self._add_state = state
        state.begin_discovery()
        self._stage = "connection"
        self._view = WorkbenchView.CONNECTION
        self._models_list_open = False
        self._refresh_add_connection()
        self._activate_view(WorkbenchView.CONNECTION, focus_id="wb-actions")

    def _select_action(self, key: str) -> None:  # noqa: PLR0912, PLR0915
        state = self.state
        if state is None or self._busy:
            return
        if self._stage == "models" and self._select_add_action(key, state):
            return
        if key == "retry-reload":
            self._retry_reload()
            return
        if key == "presets":
            self._open_presets()
            return
        if key == "style":
            self._open_protocol_picker(state.connection.api_style)
            return
        if key in self.FIELDS or key == "key":
            if key == "key" and not state.connection.api_key_env_var:
                self._message = "Set a credential environment variable first."
                self._update_help()
                return
            if key == "key":
                self._credential_input_id = "wb-input"
            self._editing = key
            field = self.FIELDS.get(key)
            editor = self.query_one("#wb-input", Input)
            editor.password = key == "key"
            editor.value = str(getattr(state.connection, field)) if field else ""
            self.query_one("#wb-editor-label", Label).update(
                {
                    "base": "API Base",
                    "style": "API Style",
                    "env": "Credential Env Var",
                    "key": "API Key *",
                }[key]
            )
            self._sync_view()
            self._restore_view_focus("wb-input")
        elif key == "discover":
            self.run_worker(
                self._discovery_request(state),
                group="workbench-discovery",
                exclusive=True,
            )
        elif key == "models":
            if self._view != WorkbenchView.MODELS:
                self._push_view(WorkbenchView.MODELS)
            self._models_list_open = True
            self._view = WorkbenchView.MODELS
            self._refresh_models()
            self._activate_view(WorkbenchView.MODELS, focus_id="wb-models")
        elif key.startswith("collision:"):
            wire = key.partition(":")[2]
            item = state.pending[wire]
            outcome = match_discovered_model(state.catalog, state.provider_id, wire)
            if outcome.kind == "multiple_matches":
                self._message = f"{wire} already matches multiple deployments; cannot resolve safely."
            elif outcome.kind == "base_exists_other_provider" and item.decision is None:
                self._confirm = f"existing:{wire}"
            else:
                self._editing = f"canonical:{wire}"
                editor = self.query_one("#wb-input", Input)
                editor.password = False
                editor.value = item.canonical_name
                self.query_one("#wb-editor-label", Label).update(
                    "Separate Canonical Name"
                )
                self._sync_view()
                self._restore_view_focus("wb-input")
        elif key == "apply":
            self._apply(state)
        elif key == "discard":
            if state.dirty:
                self._confirm = "discard"
            else:
                self._expand(state.provider_id)
        self._update_help()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not self._can_activate(event.input.id):
            return
        if event.input.id == "wb-input":
            self._accept_editor()
        elif event.input.id in {
            "wb-connection-name",
            "wb-connection-base",
            "wb-connection-env",
            "wb-connection-key",
        }:
            self._accept_connection_form_field(event.input)
        elif event.input.id in {
            "detail-thinking",
            "detail-temperature",
            "detail-auto_compact_threshold",
            "price-input",
            "price-output",
            "price-cached_input",
        }:
            fields = (
                "detail-thinking",
                "detail-temperature",
                "detail-auto_compact_threshold",
                "price-input",
                "price-output",
                "price-cached_input",
            )
            current = event.input.id or ""
            index = fields.index(current)
            self.set_focus(
                self.query_one(f"#{fields[index + 1]}", Input)
                if index + 1 < len(fields)
                else self.query_one("#wb-detail-actions", WorkbenchList)
            )

    def _field_error(self, reason: str) -> None:
        self.query_one("#wb-input", Input).add_class("-invalid")
        error = self.query_one("#wb-field-error", NoMarkupStatic)
        error.add_class("has-error")
        error.update(f"Error: {reason}")
        self._set_feedback("field", reason, "error")

    def _accept_connection_form_field(self, input_widget: Input) -> None:
        if self._add is None:
            return
        input_id = input_widget.id or ""
        field = {
            "wb-connection-name": "name",
            "wb-connection-base": "api_base",
            "wb-connection-env": "api_key_env_var",
        }.get(input_id)
        if field:
            self._add = replace(self._add, **{field: input_widget.value.strip()})
            if field == "api_key_env_var":
                self._refresh_connection_form()
            next_widget: Input | WorkbenchList
            if input_id == "wb-connection-name":
                next_widget = self.query_one("#wb-connection-base", Input)
            elif input_id == "wb-connection-base":
                next_widget = self.query_one("#wb-connection-style", WorkbenchList)
            elif self._add.api_key_env_var:
                next_widget = self.query_one("#wb-connection-key", Input)
            else:
                next_widget = self.query_one("#wb-connection-actions", WorkbenchList)
            self.set_focus(next_widget)
        elif input_widget.id == "wb-connection-key":
            actions = self.query_one("#wb-connection-actions", WorkbenchList)
            if self._add.api_key_env_var:
                actions.highlighted = 0
            self.set_focus(actions)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "wb-input":
            event.input.remove_class("-invalid")
            error = self.query_one("#wb-field-error", NoMarkupStatic)
            error.remove_class("has-error")
            error.update("")
        elif event.input.id in {
            "detail-thinking",
            "detail-temperature",
            "detail-auto_compact_threshold",
        }:
            event.input.remove_class("-invalid")
            field = event.input.id.removeprefix("detail-")
            error = self.query_one(f"#detail-error-{field}", NoMarkupStatic)
            error.update("")
            error.display = False
        elif (
            event.input.id
            in {"wb-connection-name", "wb-connection-base", "wb-connection-env"}
            and self._add
        ):
            field = {
                "wb-connection-name": "name",
                "wb-connection-base": "api_base",
                "wb-connection-env": "api_key_env_var",
            }[event.input.id]
            self._add = replace(self._add, **{field: event.input.value})
            if event.input.id == "wb-connection-env":
                self._refresh_connection_actions()
        elif event.input.id == "wb-connection-key" and self._add:
            self._refresh_connection_actions()

    def _accept_add_connection(self, key: str) -> None:
        assert self._add is not None
        editor = self.query_one("#wb-input", Input)
        value = editor.value.strip()
        try:
            draft = replace(self._add, **{self.FIELDS.get(key, key): value})
            if key == "name":
                self._new_provider_id(value)
            if key in self.FIELDS and key != "base":
                ProviderDefinition.model_validate({
                    "api_base": draft.api_base or "https://placeholder.invalid",
                    "api_style": draft.api_style,
                    "api_key_env_var": draft.api_key_env_var,
                })
            if key == "base" and value:
                ProviderDefinition.model_validate({"api_base": value})
        except ValueError as exc:
            self._field_error(f"Invalid {key}: {exc}")
            return
        self._add = draft
        if key == "env" and self._key_after_env:
            if not value:
                self._field_error("Enter an environment variable name for the API key.")
                return
            self._key_after_env = False
            self._editing = None
            self._refresh_add_connection(highlight="key")
            self._connection_action("key")
            return
        self._set_feedback("field", "", "info")
        self._editing = None
        editor.value = ""
        editor.remove_class("-invalid")
        self._sync_view()
        self._refresh_add_connection(highlight=key)
        self._return_from_editor()

    def _accept_manual_model(self, state: ManagementState) -> None:
        editor = self.query_one("#wb-input", Input)
        wire = editor.value.strip()
        if not wire or "@" in wire or "/" in wire:
            self._field_error("Enter a model name without @ or /.")
            return
        state.discovery_generation += 1  # A late probe must not replace manual entries.
        if wire not in {row[1] for row in state.model_rows()}:
            previous = (
                state.discovery.models
                if isinstance(state.discovery, DiscoveryResult)
                else ()
            )
            state.discovery = DiscoveryResult((*previous, DiscoveryItem(wire)))
        state.select(wire)
        if self.mode == "onboarding":
            item = state.pending.get(wire)
            self.selected_model = item.canonical_name if item else wire
        self._set_feedback("field", "", "info")
        self._editing = None
        editor.value = ""
        self._sync_view()
        self._refresh_models(highlight=wire)
        self._refresh_add_models(highlight="manual")
        self._return_from_editor()

    def _accept_editor(self) -> None:  # noqa: PLR0911
        if self._busy:
            return
        state = self.state
        key = self._editing
        if key is None:
            return
        if key.startswith("detail:"):
            field = key.removeprefix("detail:")
            widget_id = (
                f"#price-{field}"
                if field in {"input", "output", "cached_input"}
                else f"#detail-{field}"
            )
            target = self.query_one(widget_id, Input)
            target.value = self.query_one("#wb-input", Input).value.strip()
            target.remove_class("-invalid")
            error_id = (
                f"#error-{field}"
                if field in {"input", "output", "cached_input"}
                else f"#detail-error-{field}"
            )
            error = self.query_one(error_id, NoMarkupStatic)
            error.update("")
            error.display = False
            self._editing = None
            self._detail_edit_row = None
            self._refresh_detail_fields(highlight=field)
            self._activate_view(
                WorkbenchView.DETAIL, focus_id="wb-detail-fields", selected_id=field
            )
            self._return_from_editor()
            return
        if self._stage == "connection" and self._add is not None and key != "key":
            self._accept_add_connection(key)
            return
        if key == "manual" and state is not None:
            self._accept_manual_model(state)
            return
        editor = self.query_one("#wb-input", Input)
        value = editor.value.strip()
        if key == "key":
            self._accept_credential(value, state)
            return
        if state is None:
            return
        if key.startswith("canonical:"):
            wire = key.partition(":")[2]
            item = state.pending[wire]
            state.pending[wire] = replace(item, canonical_name=value)
            self._refresh_models(highlight=value)
        else:
            try:
                candidate = replace(state.connection, **{self.FIELDS[key]: value})
                ProviderDefinition.model_validate({
                    **self.snapshot.catalog.providers[state.provider_id].model_dump(),
                    "api_base": candidate.api_base,
                    "api_style": candidate.api_style,
                    "api_key_env_var": candidate.api_key_env_var,
                })
            except ValueError as exc:
                self._field_error(f"Invalid {key}: {exc}")
                return
            state.set_connection(candidate)
        self._set_feedback("field", "", "info")
        self._editing = None
        editor.value = ""
        editor.remove_class("-invalid")
        self._sync_view()
        self._refresh_actions()
        self._refresh_browser()
        self._return_from_editor()

    def _accept_credential(self, value: str, state: ManagementState | None) -> None:
        if not value:
            self._field_error("Enter a key before saving.")
            return
        env = (
            self._add.api_key_env_var
            if self._stage and self._add
            else state.connection.api_key_env_var
            if state
            else ""
        )
        if env and (
            env in self._saved_credential_envs or self.credential_resolver(env)
        ):
            self._confirm = "shared-key"
            self._update_help()
            return
        self._save_key(value)

    def _persisted_credential_value(self, env_var: str) -> str | None:
        if env_var in self._session_only_credential_envs:
            return None
        return self._credential_value(env_var)

    def _bound_roster_identities(self, snapshot: CatalogSnapshot) -> frozenset[str]:
        """Distinct canonical identities the snapshot's slots would bind."""
        return bind_snapshot_policy(
            snapshot, allowed_models=self.allowed_models
        ).bound_canonical_identities

    def _notify_models_saved(
        self, previously_usable: frozenset[str], roster_before: frozenset[str]
    ) -> None:
        if self.on_models_saved is not None:
            usable = usable_canonical_models(
                self.snapshot,
                self._persisted_credential_value,
                allowed_models=self.allowed_models,
                thinking_overrides=self.thinking_overrides,
            )
            self.on_models_saved(
                roster_before,
                self._bound_roster_identities(self.snapshot),
                usable - previously_usable,
            )

    def _save_key(self, value: str) -> None:
        previously_usable = usable_canonical_models(
            self.snapshot,
            self._persisted_credential_value,
            allowed_models=self.allowed_models,
            thinking_overrides=self.thinking_overrides,
        )
        roster_before = self._bound_roster_identities(self.snapshot)
        state = self.state
        env = (
            self._add.api_key_env_var
            if self._stage and self._add
            else state.connection.api_key_env_var
            if state
            else ""
        )
        if not env:
            return
        result = self.credentials.save_key(env, value)
        if result.status != "invalid_env_var":
            self._saved_credential_envs.add(env)
        if result.status == "session_only":
            self._session_only_credential_envs.add(env)
        if result.status == "saved":
            self._session_only_credential_envs.discard(env)
            self._notify_models_saved(previously_usable, roster_before)
        self._unresolved.pop("field", None)
        self._confirm = None
        self._editing = None
        editor = self.query_one("#wb-input", Input)
        credential_input = self.query_one(f"#{self._credential_input_id}", Input)
        editor.value = ""
        self._sync_view()
        if self._stage == "connection":
            self._refresh_add_connection(highlight="key")
            self.set_focus(self.query_one("#wb-actions", WorkbenchList))
        else:
            self.set_focus(self.query_one("#wb-actions", NavigableOptionList))
        feedback_kind = cast(
            Literal["running", "success", "error", "warning", "info"],
            {"saved": "success", "session_only": "warning", "invalid_env_var": "error"}[
                result.status
            ],
        )
        self._set_feedback(
            "credential",
            {
                "saved": "Key saved.",
                "session_only": "Key available for this session only.",
                "invalid_env_var": "Failed to save key: invalid environment variable.",
            }[result.status],
            feedback_kind,
        )
        if result.status != "invalid_env_var":
            credential_input.value = ""
            self._changed = True
            if self._stage == "models" and state:
                state.begin_discovery()
        if self._stage == "connection":
            self._refresh_connection_actions()
        elif self._stage == "models":
            self._refresh_add_models(highlight="edit-key")
        self._refresh_browser()
        self._return_from_editor()
        self._update_help()

    @staticmethod
    def _is_chat_model(item: DiscoveryItem) -> bool:
        value = f"{item.wire_id} {item.display_label or ''}".casefold()
        return not any(
            marker in value
            for marker in (
                "embed",
                "moderation",
                "ocr",
                "voxtral",
                "audio",
                "transcri",
                "whisper",
                "tts",
                "image-gen",
                "video",
                "rerank",
                "guard",
                "realtime",
                "labs",
                "experimental",
            )
        )

    def _chat_discovery(
        self,
        state: ManagementState,
        result: DiscoveryResult,
        *,
        provider_id: str | None = None,
    ) -> DiscoveryResult:
        """Show one chat-capable wire ID per canonical deployment or unmatched ID."""
        provider_id = provider_id or state.provider_id
        groups: dict[str, list[DiscoveryItem]] = {}
        for item in result.models:
            if item.wire_id and self._is_chat_model(item):
                match = match_discovered_model(state.catalog, provider_id, item.wire_id)
                base = match.existing_base if match.kind == "existing" else None
                groups.setdefault(base or item.wire_id, []).append(item)
        chosen = []
        for base, items in groups.items():
            model = state.catalog.models.get(base)
            names = (
                {dep.name for dep in model.deployments if dep.provider == provider_id}
                if model
                else set()
            )
            chosen.append(
                next((item for item in items if item.wire_id in names), items[0])
            )
        return DiscoveryResult(
            tuple(sorted(chosen, key=lambda item: item.wire_id.casefold())),
            result.diagnostics,
        )

    def _discovery_request(
        self, state: ManagementState
    ) -> partial[Coroutine[None, None, None]]:
        provider_id, generation = state.begin_discovery()
        return partial(
            self._discover,
            state,
            request=(
                provider_id,
                generation,
                self._stage,
                deepcopy(state.connection),
                self._add.name
                if self._stage == "models" and self._add
                else self._friendly(provider_id),
            ),
        )

    async def _discover(
        self,
        state: ManagementState,
        *,
        request: tuple[str, int, str | None, ConnectionDraft, str] | None = None,
    ) -> None:
        if self._unmounted or self._dismissed or not self.is_mounted or self._busy:
            return
        if request is None:
            if self.state is not state:
                return
            provider_id, generation = state.begin_discovery()
            stage, connection = self._stage, deepcopy(state.connection)
            display_name = (
                self._add.name
                if stage == "models" and self._add
                else self._friendly(provider_id)
            )
        else:
            provider_id, generation, stage, connection, display_name = request
        running_message = (
            f"Discovering models for {provider_id}{chrome_glyph('running')}"
        )
        self._set_feedback("discovery", running_message, "running")
        feedback_token = self._feedback_token
        provider = ProviderDraft(
            None,
            provider_id,
            display_name,
            connection.api_base,
            connection.api_style,
            connection.api_key_env_var,
            None,
            connection.backend,
            connection.reasoning_field_name,
            connection.extra_headers,
        )
        try:
            result = await self.discovery_service(
                provider,
                self.credential_resolver(connection.api_key_env_var)
                if connection.api_key_env_var
                else None,
                self.tls,
            )
        except Exception:
            result = DiscoveryError(
                "connection",
                f"Discovery failed for {provider_id}; retry or check the connection.",
            )
        if isinstance(result, DiscoveryResult):
            result = self._chat_discovery(state, result, provider_id=provider_id)
        rejected = self._stage != stage or self._unmounted or not self.is_mounted
        if not rejected:
            rejected = (
                self._busy
                or self._dismissed
                or self.state is not state
                or state.provider_id != provider_id
                or not state.accept_discovery(provider_id, generation, result)
            )
        if rejected:
            if (
                self.is_mounted
                and not self._unmounted
                and self._feedback_token == feedback_token
                and self._message == running_message
            ):
                self._set_feedback("discovery", "", "info")
                self._update_help()
            return
        self._set_feedback(
            "discovery",
            f"Discovery Failed for {provider_id}: " + result.message
            if isinstance(result, DiscoveryError)
            else f"Discovered {len(result.models)} models for {provider_id}; inference not tested.",
            "error" if isinstance(result, DiscoveryError) else "info",
        )
        self._refresh_browser()
        self._refresh_actions()
        if self._models_list_open or self.query_one("#wb-models").display:
            self._refresh_models()
        self._update_help()

    def on_selection_list_selection_highlighted(
        self, event: SelectionList.SelectionHighlighted
    ) -> None:
        self._update_help()

    def _remember_onboarding_model_selection(
        self, name: str, selected: set[str]
    ) -> None:
        if self.mode == "onboarding" and name in selected:
            self.selected_model = name

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        if self._busy or self._model_sync or self.state is None:
            return
        if event.selection_list.id == "wb-models" and (
            not self._can_activate("wb-models")
            or self._group_contents_context.get("wb-models")
            != (
                WorkbenchView.MODELS,
                self._group_context(WorkbenchView.MODELS, "wb-models"),
            )
        ):
            return
        if (
            event.selection_list.id == "wb-models"
            and "\x00empty" in event.selection_list.selected
        ):
            event.selection_list.deselect("\x00empty")
            return
        if event.selection_list.id == "wb-image-support":
            return
        state = self.state
        selected = set(event.selection_list.selected)
        rows = state.model_rows()
        if all((name in selected) == enabled for name, _wire, enabled, _found in rows):
            return  # SelectionList emits queued events for programmatic row refreshes.
        original_enabled = dict(state.enabled)
        original_pending = dict(state.pending)
        for name, wire, enabled, _found in rows:
            if (name in selected) == enabled:
                continue
            self._remember_onboarding_model_selection(name, selected)
            if name in state.configured():
                state.toggle(name, name in selected)
            elif wire in state.pending:
                state.pending[wire] = replace(
                    state.pending[wire], enabled=name in selected
                )
            elif name in selected:
                state.select(wire)
        validation = state.validate(completion=False)
        if validation.errors:
            self._set_feedback("models", "; ".join(validation.errors), "error")
            state.enabled = original_enabled
            state.pending = original_pending
            models = self.query_one("#wb-models", ModelChecklist)
            self._model_sync = True
            for name, _wire, enabled, _found in rows:
                if enabled:
                    models.select(name)
                else:
                    models.deselect(name)
            self._model_sync = False
        else:
            self._set_feedback(
                "models",
                "; ".join(validation.warnings),
                "warning" if validation.warnings else "info",
            )
        self._refresh_actions()
        self._refresh_browser()
        self._update_help()

    @staticmethod
    def _canonical_thinking_baseline(
        state: ManagementState, name: str, item: PendingModel | None
    ) -> str:
        model = state.catalog.models.get(name)
        if model is not None:
            return model.thinking
        if item is not None:
            outcome = match_discovered_model(
                state.catalog, state.provider_id, item.wire_name
            )
            if outcome.proposed is not None:
                return outcome.proposed.definition.thinking
        return "medium"

    def _open_detail(self, name: str, *, push: bool = True) -> None:  # noqa: PLR0914, PLR0915 - form setup
        state = self.state
        if state is None:
            return
        self._detail_preview_wire = None
        if (
            self._stage == "models"
            and name not in state.configured()
            and not any(item.canonical_name == name for item in state.pending.values())
            and isinstance(state.discovery, DiscoveryResult)
            and any(item.wire_id == name for item in state.discovery.models)
        ):
            # Opening a discovered deployment is a read-only preview until Save.
            self._detail_preview_wire = name
        if (
            self._detail_preview_wire is None
            and name not in state.configured()
            and not any(item.canonical_name == name for item in state.pending.values())
        ):
            self._message = "Enable this discovered model before editing details."
            self._update_help()
            return
        self._detail = name
        if push and self._view != WorkbenchView.DETAIL:
            self._push_view(WorkbenchView.DETAIL)
        self._view = WorkbenchView.DETAIL
        models = self.query_one("#wb-models", ModelChecklist)
        self._detail_cursor = name
        self._detail_scroll = models.scroll_offset.y
        item = next(
            (item for item in state.pending.values() if item.canonical_name == name),
            None,
        )
        if item is None and self._detail_preview_wire is not None:
            item = PendingModel(self._detail_preview_wire, name, False)
        edits = item.edits if item else state.edits.get(name, ModelEdits())
        model = state.catalog.models.get(name)
        deployment = (
            next(
                (
                    candidate
                    for candidate in model.deployments
                    if candidate.provider == state.provider_id
                ),
                None,
            )
            if model is not None
            else None
        )
        if deployment is None and item is not None:
            deployment = DeploymentDefinition(
                provider=state.provider_id, name=item.wire_name
            )
        if deployment is None:
            self._message = "This model has no deployment in the selected provider."
            self._detail = None
            self._pop_view()
            self._update_help()
            return
        self.query_one("#wb-detail-model-id", NoMarkupStatic).update(
            f"Model: {name} · Deployment: {state.provider_id}/{deployment.name}"
        )
        self.query_one("#wb-detail-deployment-id").display = False
        self.query_one("#wb-image-support-label").display = False
        supported = deployment.supported_thinking_levels
        self.query_one("#wb-detail-supported-thinking", NoMarkupStatic).update(
            "Supported thinking levels: "
            + (", ".join(supported) if supported is not None else "provider default")
        )
        thinking_baseline = self._canonical_thinking_baseline(state, name, item)
        thinking_value = (
            edits.thinking.value
            if edits.thinking.state == "set"
            else ""
            if edits.thinking.state == "cleared"
            else thinking_baseline
        )
        temperature_baseline = model.temperature if model is not None else None
        temperature_value = (
            edits.temperature.value
            if edits.temperature.state == "set"
            else None
            if edits.temperature.state == "cleared"
            else temperature_baseline
        )
        threshold_value = (
            edits.auto_compact_threshold.value
            if edits.auto_compact_threshold.state == "set"
            else None
            if edits.auto_compact_threshold.state == "cleared"
            else deployment.auto_compact_threshold
        )
        for field, value in (
            ("thinking", thinking_value),
            ("temperature", temperature_value),
            ("auto_compact_threshold", threshold_value),
        ):
            widget = self.query_one(f"#detail-{field}", Input)
            widget.remove_class("-invalid")
            widget.value = "" if value is None else str(value)
            error = self.query_one(f"#detail-error-{field}", NoMarkupStatic)
            error.update("")
            error.display = False
        images = self.query_one("#wb-image-support", ModelChecklist)
        self._model_sync = True
        images.clear_options()
        image_edit = edits.supports_images
        image_value = (
            bool(image_edit.value)
            if image_edit.state == "set"
            else False
            if image_edit.state == "cleared"
            else deployment.supports_images
        )
        images.add_option(
            Selection("Images · Space toggles", "supports-images", image_value)
        )
        self._model_sync = False
        for field in ("input", "output", "cached_input"):
            edit: OptionalEdit[float] = getattr(edits, f"{field}_price")
            previous = getattr(deployment.prices, field) if deployment else None
            value = (
                edit.value
                if edit.state == "set"
                else None
                if edit.state == "cleared"
                else previous
            )
            self.query_one(f"#price-{field}", Input).remove_class("-invalid")
            error = self.query_one(f"#error-{field}", NoMarkupStatic)
            error.update("")
            error.display = False
            self.query_one(f"#price-{field}", Input).value = (
                "" if value is None else str(value)
            )
        self._detail_original = self._detail_form_snapshot()
        detail_actions = self.query_one("#wb-detail-actions", WorkbenchList)
        action_position = self._position_before_rebuild(
            WorkbenchView.DETAIL, "wb-detail-actions"
        )
        detail_actions.clear_options()
        detail_actions.add_option(
            Option("Accept model edits into catalog draft", id="save-detail")
        )
        detail_actions.highlighted = 0
        self._restore_group_position(action_position)
        self._refresh_detail_fields()
        self._activate_view(WorkbenchView.DETAIL, focus_id="wb-detail-fields")
        self._update_help()

    def _refresh_detail_fields(self, *, highlight: str | None = None) -> None:
        rows = self.query_one("#wb-detail-fields", WorkbenchList)
        position = self._position_before_rebuild(
            WorkbenchView.DETAIL, "wb-detail-fields"
        )
        previous = highlight or (
            str(rows.highlighted_option.id) if rows.highlighted_option else "thinking"
        )
        rows.clear_options()
        fields = (
            ("thinking", "Default thinking", "#detail-thinking"),
            ("temperature", "Temperature", "#detail-temperature"),
            (
                "auto_compact_threshold",
                "Auto compact threshold",
                "#detail-auto_compact_threshold",
            ),
            ("input", "Input price", "#price-input"),
            ("output", "Output price", "#price-output"),
            ("cached_input", "Cached input price", "#price-cached_input"),
        )
        for key, label, widget_id in fields:
            widget = self.query_one(widget_id, Input)
            value = widget.value or "Not set"
            status = "Error: " if widget.has_class("-invalid") else ""
            rows.add_option(Option(self._summary(f"{status}{label}  {value}"), id=key))
        images = self.query_one("#wb-image-support", ModelChecklist)
        enabled = "supports-images" in images.selected
        rows.add_option(
            Option(
                f"Image support  {'Enabled' if enabled else 'Disabled'} · Space toggles",
                id="images",
            )
        )
        ids = [str(option.id) for option in rows.options]
        rows.highlighted = ids.index(previous) if previous in ids else 0
        if highlight is None:
            self._restore_group_position(position)

    def _toggle_detail_images(self) -> None:
        images = self.query_one("#wb-image-support", ModelChecklist)
        if "supports-images" in images.selected:
            images.deselect("supports-images")
        else:
            images.select("supports-images")
        self._refresh_detail_fields(highlight="images")
        self.query_one("#wb-detail-fields", WorkbenchList).focus()

    def _select_detail_field(self, key: str) -> None:
        if key == "save-detail":
            self._save_detail()
            return
        if key == "images":
            self._toggle_detail_images()
            return
        widget_id = (
            f"#price-{key}"
            if key in {"input", "output", "cached_input"}
            else f"#detail-{key}"
        )
        widget = self.query_one(widget_id, Input)
        self._detail_edit_row = key
        self._editing = f"detail:{key}"
        editor = self.query_one("#wb-input", Input)
        editor.password = False
        editor.value = widget.value
        self.query_one("#wb-editor-label", Label).update(key.replace("_", " ").title())
        self._update_help()
        editor.focus()

    def _detail_form_snapshot(self) -> tuple[object, ...]:
        fields = (
            "thinking",
            "temperature",
            "auto_compact_threshold",
            "input",
            "output",
            "cached_input",
        )
        values = tuple(
            self.query_one(f"#detail-{field}", Input).value for field in fields[:3]
        )
        prices = tuple(
            self.query_one(f"#price-{field}", Input).value for field in fields[3:]
        )
        images = tuple(
            sorted(self.query_one("#wb-image-support", ModelChecklist).selected)
        )
        return (*values, *prices, images)

    def _save_detail(self) -> None:  # noqa: PLR0911, PLR0912, PLR0914, PLR0915 - validation and draft save
        if self._busy:
            return
        state = self.state
        name = self._detail
        if state is None or name is None:
            return
        item = next(
            (item for item in state.pending.values() if item.canonical_name == name),
            None,
        )
        if item is None and self._detail_preview_wire is not None:
            item = PendingModel(self._detail_preview_wire, name, False)
        edits = item.edits if item else state.edits.get(name, ModelEdits())
        model = state.catalog.models.get(name)
        deployment = (
            next(
                (
                    candidate
                    for candidate in model.deployments
                    if candidate.provider == state.provider_id
                ),
                None,
            )
            if model is not None
            else None
        )
        if deployment is None and item is not None:
            deployment = DeploymentDefinition(
                provider=state.provider_id, name=item.wire_name
            )
        if deployment is None:
            self._set_detail_error(
                "thinking", "This model has no deployment in the selected provider."
            )
            return

        thinking_widget = self.query_one("#detail-thinking", Input)
        thinking = thinking_widget.value.strip()
        if not thinking:
            self._set_detail_error("thinking", "Thinking level cannot be blank.")
            return
        temperature_widget = self.query_one("#detail-temperature", Input)
        temperature_raw = temperature_widget.value.strip()
        try:
            temperature = float(temperature_raw) if temperature_raw else None
            if temperature is not None and not isfinite(temperature):
                raise ValueError
        except ValueError:
            self._set_detail_error("temperature", "Use a finite number or leave blank.")
            return
        threshold_widget = self.query_one("#detail-auto_compact_threshold", Input)
        threshold_raw = threshold_widget.value.strip()
        try:
            threshold = float(threshold_raw) if threshold_raw else None
            if threshold is not None and (not isfinite(threshold) or threshold <= 0):
                raise ValueError
        except ValueError:
            self._set_detail_error(
                "auto_compact_threshold", "Use a positive finite number or leave blank."
            )
            return
        try:
            BaseModelDefinition.model_validate({
                "thinking": thinking,
                "temperature": temperature,
                "deployments": [deployment.model_dump()],
                "disabled": model.disabled if model is not None else False,
            })
            if (
                deployment.supported_thinking_levels is not None
                and thinking not in deployment.supported_thinking_levels
            ):
                raise ValueError("Thinking level is not supported by this deployment.")
        except ValueError as exc:
            self._set_detail_error("thinking", str(exc))
            return
        for field in ("thinking", "temperature", "auto_compact_threshold"):
            widget = self.query_one(f"#detail-{field}", Input)
            widget.remove_class("-invalid")
            error = self.query_one(f"#detail-error-{field}", NoMarkupStatic)
            error.update("")
            error.display = False

        old_temperature = (
            edits.temperature.value
            if edits.temperature.state == "set"
            else None
            if edits.temperature.state == "cleared"
            else model.temperature
            if model is not None
            else None
        )
        temperature_edit = edits.temperature
        if temperature != old_temperature:
            temperature_edit = (
                OptionalEdit.cleared()
                if temperature is None
                else OptionalEdit.set(temperature)
            )
        old_thinking = (
            edits.thinking.value
            if edits.thinking.state == "set"
            else ""
            if edits.thinking.state == "cleared"
            else self._canonical_thinking_baseline(state, name, item)
        )
        thinking_edit = (
            edits.thinking if thinking == old_thinking else OptionalEdit.set(thinking)
        )
        old_threshold = (
            edits.auto_compact_threshold.value
            if edits.auto_compact_threshold.state == "set"
            else None
            if edits.auto_compact_threshold.state == "cleared"
            else deployment.auto_compact_threshold
        )
        threshold_edit = edits.auto_compact_threshold
        if threshold != old_threshold:
            threshold_edit = (
                OptionalEdit.cleared()
                if threshold is None
                else OptionalEdit.set(threshold)
            )
        image_value = (
            "supports-images"
            in self.query_one("#wb-image-support", ModelChecklist).selected
        )
        old_image_value = (
            bool(edits.supports_images.value)
            if edits.supports_images.state == "set"
            else False
            if edits.supports_images.state == "cleared"
            else deployment.supports_images
        )
        image_edit = (
            edits.supports_images
            if image_value == old_image_value
            else OptionalEdit.set(image_value)
        )
        updates: dict[str, OptionalEdit[float]] = {}
        for field in ("input", "output", "cached_input"):
            widget = self.query_one(f"#price-{field}", Input)
            raw = widget.value.strip()
            try:
                value = float(raw) if raw else None
                if value is not None and (not isfinite(value) or value < 0):
                    raise ValueError
            except ValueError:
                widget.add_class("-invalid")
                error = self.query_one(f"#error-{field}", NoMarkupStatic)
                error.update("Error: use a non-negative finite number.")
                error.display = True
                self._refresh_detail_fields(highlight=field)
                self.query_one("#wb-detail-fields", WorkbenchList).focus()
                self._set_feedback(
                    "detail", f"{field} price must be non-negative and finite.", "error"
                )
                self._update_help()
                return
            widget.remove_class("-invalid")
            error = self.query_one(f"#error-{field}", NoMarkupStatic)
            error.update("")
            error.display = False
            previous = getattr(deployment.prices, field) if deployment else None
            updates[f"{field}_price"] = (
                OptionalEdit()
                if value == previous
                else OptionalEdit.cleared()
                if value is None
                else OptionalEdit.set(value)
            )
        replacement = replace(
            edits,
            **updates,
            thinking=thinking_edit,
            temperature=temperature_edit,
            supports_images=image_edit,
            auto_compact_threshold=threshold_edit,
        )
        if item:
            if item.wire_name not in state.pending:
                state.pending[item.wire_name] = item
            state.set_pending_edits(item.wire_name, replacement)
        else:
            state.set_edits(name, replacement)
        if self._feedback_kind not in {"error", "warning"}:
            self._message = ""
        self._close_detail()
        self._refresh_actions()
        self._refresh_browser()

    def _set_detail_error(self, field: str, message: str) -> None:
        widget = self.query_one(f"#detail-{field}", Input)
        widget.add_class("-invalid")
        error = self.query_one(f"#detail-error-{field}", NoMarkupStatic)
        error.update(f"Error: {message}")
        error.display = True
        self._refresh_detail_fields(highlight=field)
        self.query_one("#wb-detail-fields", WorkbenchList).focus()
        self._set_feedback("detail", message, "error")
        self._update_help()

    def _detail_is_modified(self) -> bool:
        if self._detail_original is None or not self._detail:
            return False
        return self._detail_form_snapshot() != self._detail_original

    def _detail_context(self) -> str:
        state, name = self.state, self._detail
        if state is None or name is None:
            return "Model details"
        deployment = state.configured().get(name)
        wire = (
            deployment.name
            if deployment
            else next(
                (
                    item.wire_name
                    for item in state.pending.values()
                    if item.canonical_name == name
                ),
                name,
            )
        )
        return self._summary(
            f"Model details: {name} · {state.provider_id}/{wire}", reserve=4
        )

    def _close_detail(self) -> None:
        self._detail = None
        self._detail_edit_row = None
        self._detail_preview_wire = None
        self._detail_original = None
        if self._pop_view():
            return
        self._refresh_models(highlight=self._detail_cursor)
        models = self.query_one("#wb-models", ModelChecklist)
        models.scroll_y = self._detail_scroll
        if self._models_view:
            self._refresh_catalog()
            self._activate_view(WorkbenchView.CATALOG, focus_id="wb-catalog")
        else:
            self._activate_view(
                WorkbenchView.MODELS
                if self._models_list_open
                else WorkbenchView.ACTIONS
            )
        self._update_help()

    def _apply(self, state: ManagementState) -> None:
        self._start_commit(state, create=False)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in self.query("Input, SelectionList, OptionList"):
            widget.disabled = busy
        if busy:
            self._feedback_kind = "running"
        if not busy:
            too_small = (
                self.size.width < self.MIN_WIDTH or self.size.height < self.MIN_HEIGHT
            )
            self.query_one("#workbench").display = not too_small
            self.query_one("#wb-small").display = too_small
        self._update_help()

    def _start_commit(  # noqa: PLR0911, PLR0912
        self, state: ManagementState, *, create: bool
    ) -> None:
        if self._busy or state is not self.state or self._dismissed:
            return
        if self._reload_failed:
            self._message = "Retry Runtime Reload before saving more catalog changes."
            self._update_help()
            return
        if self._detail is not None:
            self._save_detail()
            if self._detail is not None:
                return
        if self._editing is not None:
            self._accept_editor()
            if self._editing is not None:
                return
        try:
            changes = state.changes(
                mode="management" if self.mode == "onboarding" else self.mode,
                credential_resolver=self._credential_value,
            )
            if (
                self._after_commit == "models"
                and self._add is not None
                and self._add.preset == "existing"
            ):
                # Editing a saved connection is its own checkpoint. Keep model
                # selections in the draft until the Models forward action.
                changes = replace(changes, models={}, roles=None)
            if create:
                if self._add is None:
                    return
                provider = self.snapshot.catalog.providers.get(state.provider_id)
                is_mistral_preset = (
                    state.provider_id == "mistral" and self._add.preset == "mistral"
                )
                if provider is not None and not is_mistral_preset:
                    raise ValueError("Provider already exists; choose another name.")
                connection_changed = provider is not None and (
                    state.connection != ConnectionDraft.from_definition(provider)
                )
                if is_mistral_preset and provider is not None:
                    customized = state.provider_id in self.snapshot.overlaid_providers
                    if (
                        customized
                        and connection_changed
                        and not self._mistral_overwrite_approved
                    ):
                        self._confirm = "overwrite-mistral"
                        self._update_help()
                        return
                if provider is None or connection_changed:
                    # A new provider persists its full staged definition, and a
                    # changed preset overrides the shipped one. An unchanged
                    # preset over an existing definition persists nothing: the
                    # provider already lives in the base catalog, and writing
                    # its table into the overlay would flip overlaid_providers
                    # provenance and block first-run materialization of the
                    # default template.
                    changes = replace(
                        changes,
                        provider={
                            **state.catalog.providers[state.provider_id].model_dump(),
                            **changes.provider_patches.get(state.provider_id, {}),
                        },
                    )
            elif not state.dirty:
                self._feedback_kind = "info"
                self._message = "No catalog changes to apply."
                self._update_help()
                return
            if not self._connection_approved:
                affected = [
                    name
                    for name, connection in state.connections.items()
                    if name in self.snapshot.overlaid_providers
                    and name in self.snapshot.catalog.providers
                    and not (create and name == state.provider_id)
                    and connection
                    != connection.from_definition(self.snapshot.catalog.providers[name])
                ]
                if affected:
                    self._connection_scopes = affected
                    self._confirm = "replace-connection"
                    self._update_help()
                    return
            # The draft is mutable, including nested deployment patches. Never hand
            # any of its containers to the writer thread.
            changes = deepcopy(
                replace(changes, expected_revision=self.snapshot.revision)
            )
        except ValueError as exc:
            self._set_feedback("catalog", str(exc), "error")
            return
        self._message = f"Saving{chrome_glyph('running')}"
        self._feedback_kind = "running"
        self._mistral_overwrite_approved = False
        self._connection_approved = False
        self._commit_return = self._frame()
        self._set_busy(True)
        self._commit_task = asyncio.create_task(
            self._commit(changes, state, create=create)
        )

    async def _commit(  # noqa: PLR0912, PLR0915
        self, changes: CatalogChanges, state: ManagementState, *, create: bool
    ) -> None:
        previously_usable = usable_canonical_models(
            self.snapshot,
            self._persisted_credential_value,
            allowed_models=self.allowed_models,
            thinking_overrides=self.thinking_overrides,
        )
        roster_before = self._bound_roster_identities(self.snapshot)
        write = asyncio.create_task(
            asyncio.to_thread(self.catalog_writer.apply_changes, changes)
        )
        self._write_task = write
        saved = False
        try:
            # Shield the thread operation even when a screen worker is cancelled.
            while True:
                try:
                    result = await asyncio.shield(write)
                    break
                except asyncio.CancelledError:
                    if write.done():
                        result = write.result()
                        break
            if isinstance(result, CatalogValidationError):
                self._set_feedback(
                    "catalog",
                    f"Failed to {'create' if create else 'apply'}: {result.message}",
                    "error",
                )
                return
            saved = True
            self._changed |= result.changed
            self.snapshot = result.snapshot
            state.begin_discovery()
            self.state = ManagementState.from_snapshot(
                result.snapshot, state.provider_id
            )
            if result.changed:
                self._notify_models_saved(previously_usable, roster_before)
            if create:
                self._stage = None
                self._add = None
                self._add_checkpoint = None
                self._add_state = None
                self._pre_add_state = None
                self._navigation.clear()
                self._provider_open = False
                self._models_list_open = False
                self._view = WorkbenchView.PROVIDERS
                self._sync_view()
                self._browser_cursor = state.provider_id
                self.query_one("#wb-providers", BrowserList).highlighted = None
            await self._reload_catalog()
            if not self._reload_failed:
                self._set_feedback(
                    "catalog",
                    "Provider created." if create else "Saved catalog changes.",
                    "success",
                )
            self._refresh_browser()
            if create:
                self.state = None
                self._view = WorkbenchView.PROVIDERS
                self._refresh_browser()
                browser = self.query_one("#wb-providers", BrowserList)
                ids = [str(option.id) for option in browser.options]
                if state.provider_id in ids:
                    browser.highlighted = ids.index(state.provider_id)
                self._activate_view(
                    WorkbenchView.PROVIDERS,
                    focus_id="wb-providers",
                    selected_id=state.provider_id,
                )
                browser.focus()
                self._defer_focus(browser)
            else:
                self._refresh_actions(highlight="apply")
                self._refresh_models()
                if self._models_view:
                    self._refresh_catalog()
        except asyncio.CancelledError:
            if saved:
                self._reload_failed = True
                self._warning = "Runtime reload failed."
                self._message = "Saved; Reload Failed."
            else:
                raise
        except Exception:
            if saved:
                self._reload_failed = True
                self._warning = "Runtime reload failed."
                self._message = "Saved; Reload Failed."
            else:
                self._set_feedback(
                    "catalog",
                    "Failed to create provider. Draft preserved."
                    if create
                    else "Failed to apply catalog changes. Draft preserved.",
                    "error",
                )
        finally:
            if self._message.startswith(("Failed", "Saved; Reload Failed")):
                self._feedback_kind = "error"
            self._write_task = None
            self._set_busy(False)
            if saved and create and not self._reload_failed:
                browser = self.query_one("#wb-providers", BrowserList)
                browser.focus()
                self._defer_focus(browser)
            frame = self._commit_return
            self._commit_return = None
            if frame is not None and (not create or not saved):
                self._restore_frame(frame)
            destination = self._after_commit
            self._after_commit = None
            if saved and not self._reload_failed:
                if destination == "models":
                    self.state = ManagementState.from_snapshot(
                        self.snapshot, state.provider_id
                    )
                    if self._connection_pending is not None:
                        self.state.pending = self._connection_pending
                        self._connection_pending = None
                    self._stage = "models"
                    self._models_list_open = True
                    self._refresh_models()
                    self._activate_view(WorkbenchView.MODELS, focus_id="wb-models")
                    self._defer_focus(self.query_one("#wb-models", ModelChecklist))
                    self.run_worker(
                        self._discovery_request(self.state),
                        group="workbench-discovery",
                        exclusive=True,
                    )
                elif destination == "another":
                    self._start_add()
                elif destination == "presets":
                    self._open_presets()
                elif destination == "finish":
                    self._setup_ready = True
                    self._finish()

    async def _reload_catalog(self) -> None:
        try:
            reload = await self.config.reload_catalog_and_config()
        except Exception:
            reload = None
        if reload is None or reload.snapshot is None:
            self._reload_failed = True
            self._warning = (
                reload.message if reload is not None else None
            ) or "Runtime reload failed."
            self._set_feedback("reload", "Saved; Reload Failed.", "error")
        else:
            self._reload_failed = False
            self._warning = None
            self.snapshot = reload.snapshot
            if self.state:
                self.state = ManagementState.from_snapshot(
                    reload.snapshot, self.state.provider_id
                )
            self._set_feedback("reload", "Runtime reloaded.", "success")
        self._refresh_actions()
        self._update_help()

    def _retry_reload(self) -> None:
        if self._busy or not self._reload_failed:
            return
        self._message = f"Reloading{chrome_glyph('running')}"
        self._feedback_kind = "running"
        self._set_busy(True)
        self._commit_task = asyncio.create_task(self._retry_reload_task())

    async def _retry_reload_task(self) -> None:
        try:
            await self._reload_catalog()
        finally:
            self._set_busy(False)

    async def on_unmount(self) -> None:
        self._unmounted = True
        if self._commit_task is not None and not self._commit_task.done():
            await asyncio.shield(self._commit_task)

    def _restore_confirmation_focus(self) -> None:
        """Return focus to the control that opened a completed confirmation."""
        opener_id = self._confirm_opener
        self._confirm_opener = None
        if opener_id:
            opener = self.query(f"#{opener_id}")
            if opener and opener.first().display:
                opener.first().focus()
                self._restore_group_position(self._confirm_position, restore_focus=True)
                self._confirm_position = None
                return
        for widget_id in ("wb-catalog", "wb-actions", "wb-providers"):
            widget = self.query_one(f"#{widget_id}")
            if widget.display and getattr(widget, "can_focus", False):
                widget.focus()
                return

    def action_confirm_yes(self) -> None:
        if self._busy:
            return
        if not self._confirm:
            return
        decision = self._confirm
        self._confirm = None
        self._sync_view()
        state = self.state
        if decision == "replace-connection" and state:
            self._connection_approved = True
            self._start_commit(state, create=self._stage == "models")
        elif decision == "overwrite-mistral" and state:
            self._mistral_overwrite_approved = True
            self._start_commit(state, create=True)
        elif decision == "shared-key":
            self._save_key(
                self.query_one(f"#{self._credential_input_id}", Input).value.strip()
            )
        elif decision in {"add-discard", "add-discard-simple"}:
            if self.state:
                self.state.begin_discovery()
            self.state = self._pre_add_state
            self._pre_add_state = None
            self._add_state = None
            self._add = None
            self._add_checkpoint = None
            self._stage = None
            self._navigation.clear()
            self._provider_open = False
            self._models_list_open = False
            self._view = WorkbenchView.PROVIDERS
            self._refresh_browser()
            self._activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
            self._restore_transient_opener(self._add_opener)
            self._add_opener = None
        elif decision == "discard" and state:
            state.begin_discovery()
            self.state = ManagementState.from_snapshot(self.snapshot, state.provider_id)
            self._refresh_actions()
            self._refresh_browser()
            if self._models_view:
                self._refresh_catalog()
        elif decision == "close":
            self._finish()
        elif decision.startswith("existing:") and state:
            wire = decision.partition(":")[2]
            item = state.pending[wire]
            state.pending[wire] = replace(item, decision="add_existing")
            self._refresh_actions(highlight=f"collision:{wire}")
        self._update_help()
        if decision == "discard" or decision.startswith("existing:"):
            self._restore_confirmation_focus()

    def action_confirm_continue(self) -> None:
        if self._busy or self._confirm != "add-discard" or self._add_state is None:
            return
        self._confirm = None
        self._sync_view()
        self._confirm_opener = None
        self._connection_action("continue")

    def action_confirm_detail_save(self) -> None:
        if self._busy or self._confirm != "detail-discard":
            return
        self._confirm = None
        self._sync_view()
        self._confirm_opener = None
        self._save_detail()

    def action_confirm_detail_discard(self) -> None:
        if self._busy or self._confirm != "detail-discard":
            return
        self._confirm = None
        self._sync_view()
        self._confirm_opener = None
        self._close_detail()

    def action_confirm_no(self) -> None:
        if self._busy:
            return
        if not self._confirm:
            return
        decision = self._confirm
        self._confirm = None
        self._sync_view()
        if decision == "shared-key":
            self._restore_view_focus(self._credential_input_id)
        elif self._confirm_opener:
            self._restore_view_focus(self._confirm_opener)
        self._restore_group_position(self._confirm_position, restore_focus=True)
        self._confirm_position = None
        self._confirm_opener = None
        if decision.startswith("existing:") and self.state:
            wire = decision.partition(":")[2]
            item = self.state.pending[wire]
            self.state.pending[wire] = replace(item, decision="separate")
            self._select_action(f"collision:{wire}")
        self._update_help()

    def _leave_add(self) -> None:
        if self.state:
            self.state.begin_discovery()
        self.state = self._pre_add_state
        self._pre_add_state = None
        self._add_state = None
        self._add = None
        self._add_checkpoint = None
        self._stage = None
        self._models_list_open = False
        self._provider_open = False
        self._navigation.clear()
        self._view = WorkbenchView.PROVIDERS
        self._activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
        self._restore_transient_opener(self._add_opener)
        self._add_opener = None
        self._update_help()

    def _back_add(self) -> None:
        if self._stage == "models":
            if self.state:
                self._begin_saved_connection_edit(self.state)
        elif self._stage == "connection":
            self._view = WorkbenchView.CONNECTION
            draft = self._add
            checkpoint = self._add_checkpoint
            changed = bool(
                draft is not None
                and checkpoint is not None
                and (draft.name, draft.api_base, draft.api_style, draft.api_key_env_var)
                != (
                    checkpoint.name,
                    checkpoint.api_base,
                    checkpoint.api_style,
                    checkpoint.api_key_env_var,
                )
            )
            if changed:
                self._confirm = (
                    "add-discard"
                    if self._add_state is not None
                    else "add-discard-simple"
                )
            else:
                self._leave_add()
        elif self._stage == "choose":
            self._leave_add()

    def _restore_transient_opener(self, frame: NavigationFrame | None) -> None:
        if frame is None:
            return
        self._sync_view()
        self._restore_view_focus(frame.focus_id or "wb-providers")
        self._restore_group_position(frame.position, restore_focus=True)

    def _return_from_editor(self) -> None:
        opener = self._editor_opener
        self._editor_opener = None
        self._restore_transient_opener(opener)

    def action_help(self) -> None:
        if self._busy or self._confirm:
            return
        if self._help_open:
            self._help_open = False
            opener = self._help_opener
            self._help_opener = None
            self._update_help()
            self._restore_transient_opener(opener)
        else:
            self._help_opener = self._frame()
            self._help_open = True
            self._update_help()

    def action_back(self) -> None:  # noqa: PLR0911, PLR0912, PLR0915
        if self._busy:
            self._update_help()
            return
        if self._confirm:
            self.action_confirm_no()
            return
        if self._help_open:
            self.action_help()
            return
        if self._editing:
            self._key_after_env = False
            row = self._detail_edit_row
            self._editing = None
            self._detail_edit_row = None
            self.query_one("#wb-input", Input).value = ""
            if row:
                self._refresh_detail_fields(highlight=row)
            self._return_from_editor()
            self._update_help()
            return
        if self._preset_field is not None:
            self._capture_group_position()
            field = self._preset_field
            self._preset_field = None
            self._refresh_preset_editor(highlight=field)
            if not self._pop_view():
                self._activate_view(
                    WorkbenchView.PRESET_EDITOR,
                    focus_id="wb-preset-editor",
                    selected_id=field,
                )
            return
        if self._view == WorkbenchView.PRESET_EDITOR:
            self._close_preset_editor()
            return
        if self._view == WorkbenchView.PRESETS:
            if self.mode == "onboarding" and self._customize_presets:
                self._customize_presets = False
                self._refresh_presets(highlight="customize")
                self.query_one("#wb-presets").focus()
                return
            self._stage = None
            opener = self._presets_opener
            self._presets_opener = None
            if opener and opener.focus_id:
                self._restore_frame(opener)
            else:
                self._refresh_browser()
                self._activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
            return
        if (
            self.filter_text
            and not self._provider_open
            and not self._stage
            and not self._models_view
            and not self._picker
        ):
            self.filter_text = ""
            self._refresh_browser()
            return
        if self._protocol_picker:
            self._protocol_picker = False
            if not self._pop_view():
                self._activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
            return
        if self._can_clear_catalog_filter():
            self._model_filter = None
            self._detail_cursor = None
            self._refresh_catalog()
            self.set_focus(self.query_one("#wb-catalog-filter", NavigableOptionList))
            self._update_help()
            return
        if self._picker:
            self._close_picker()
            return
        if self._models_view and not self._detail and not self._editing:
            self._models_view = False
            if not self._pop_view():
                self._provider_open = False
                self._activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
            self._update_help()
            return
        if self._detail:
            if self._detail_is_modified():
                self._confirm = "detail-discard"
            else:
                self._close_detail()
        elif self._stage == "models" and self._models_list_open:
            self._models_list_open = False
            if not self._pop_view():
                self._activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
        elif self._stage:
            self._back_add()
        elif self.filter_text and self.state is None:
            self.filter_text = ""
            self._refresh_browser()
        elif self._models_list_open:
            self._models_list_open = False
            if not self._pop_view():
                self._activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
        elif self.state:
            if self.query_one("#wb-actions").display:
                if not self.state.dirty:
                    self.state.begin_discovery()
                    self.state = None
                self._provider_open = False
                self._activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
            elif self.state.dirty:
                self._confirm = "close"
            else:
                self.state = None
                self._finish()
        else:
            self._finish()
        self._update_help()

    def _can_clear_catalog_filter(self) -> bool:
        return bool(
            self._models_view
            and self._view == WorkbenchView.CATALOG
            and self._model_filter
            and not self._detail
            and not self._editing
            and not self._picker
        )

    def _finish(self) -> None:
        if self._busy:
            return
        self._dismissed = True
        if self.state:
            self.state.begin_discovery()
        self.dismiss(
            ProviderWorkbenchResult(
                "completed"
                if (self._setup_ready if self.mode == "onboarding" else self._changed)
                else "cancelled",
                changed=self._changed,
                warning=self._warning,
            )
        )

    def _confirmation_content(self) -> tuple[str, str]:  # noqa: PLR0911
        decision = self._confirm
        state = self.state
        provider = (
            state.provider_id
            if state
            else self._add.name
            if self._add
            else "this provider"
        )
        if decision == "shared-key":
            env = (
                self._add.api_key_env_var
                if self._stage and self._add
                else state.connection.api_key_env_var
                if state
                else ""
            )
            affected = [
                name
                for name, definition in self.snapshot.catalog.providers.items()
                if definition.api_key_env_var == env
            ]
            if provider not in affected:
                affected.insert(0, provider)
            return (
                f"Replace saved credential {env} for {', '.join(affected)}? "
                "Keep editing preserves the saved credential and your unsaved key entry.",
                "Replace credential",
            )
        if decision == "replace-connection":
            return (
                f"Replace customized connections for {', '.join(self._connection_scopes)} and apply catalog changes "
                "(all providers and global roles)? Keep editing preserves the saved connections and catalog draft.",
                "Replace connection",
            )
        if decision == "overwrite-mistral":
            return (
                "Replace customized Mistral connection and apply all catalog changes "
                "(all providers and global roles)? Keep editing preserves the customized connection and catalog draft; saved credentials remain saved.",
                "Replace connection",
            )
        if decision and decision.startswith("existing:"):
            wire = decision.partition(":")[2]
            return (
                f"Add {wire} as a deployment of the existing canonical model? "
                "Keep editing preserves the pending model for a separate name.",
                "Use existing model",
            )
        if decision == "add-discard":
            return (
                f"Discard the new provider {provider} draft? Existing catalog edits across providers and global roles stay preserved. "
                "Saved credentials remain saved. Keep editing preserves this draft; Continue setup returns to Models.",
                "Discard edits",
            )
        if decision == "add-discard-simple":
            return (
                f"Unsaved provider draft for {provider}. Discard it? Saved credentials remain saved. Keep editing preserves your edits.",
                "Discard edits",
            )
        if decision == "detail-discard":
            return (
                "Unsaved model edits are still local to this form. Keep editing, save them into the catalog draft, or discard them.",
                "Discard model edits",
            )
        if decision == "close":
            return (
                "Discard all catalog edits across providers and global roles and close? "
                "Saved credentials remain saved. Keep editing preserves your edits and leaves the workbench open.",
                "Discard and close",
            )
        return (
            "Discard all catalog edits across providers and global roles? "
            "Saved credentials remain saved. Keep editing preserves your edits.",
            "Discard all pending changes",
        )

    def _visibility_projection(self) -> tuple[str, set[str]]:
        """Resolve body, replacement editor, and overlay without changing state."""
        groups = self.VIEW_GROUPS[self._view]
        if self._editing:
            groups = self.VIEW_GROUPS[WorkbenchView.EDITOR]
        elif self._detail:
            groups = self.VIEW_GROUPS[WorkbenchView.DETAIL]
        primary = groups[0]
        visible = set(groups)
        if primary == "models" and self._stage != "models":
            visible.discard("models-actions")
        if primary == "catalog" and not (self.state and self.state.dirty):
            visible.discard("catalog-actions")
        if self._confirm:
            visible.add("confirm")
        else:
            visible.update(("help", "hint"))
        return primary, visible

    def _apply_visibility(self, visible: set[str]) -> None:
        """The only view-control display writer; pending-action is host-owned."""
        for name in (*self.VIEW_CONTROLS, "help", "hint"):
            self.query_one(f"#wb-{name}").display = name in visible
        self.query_one("#wb-actions").set_class(
            self._view == WorkbenchView.ACTIONS, "wb-management-fields"
        )
        self.query_one("#wb-confirm-host").display = "confirm" in visible
        for name in self.UNDERLAY_CONTROLS:
            widget = self.query_one(f"#wb-{name}")
            widget.set_class(
                bool(self._confirm and widget.display), "wb-confirm-underlay"
            )

    def _sync_view(self) -> str:
        if self._editing and self._editor_opener is None:
            self._editor_opener = self._frame()
        primary, visible = self._visibility_projection()
        opening_confirmation = bool(
            self._confirm and not self.query_one("#wb-confirm").display
        )
        if self._confirm:
            confirm = self.query_one("#wb-confirm")
            confirm.border_title = "Confirm change"
            host = self.query_one("#wb-confirm-host")
            width = min(60, max(1, int(self.size.width * 0.8)))
            host.styles.offset = (
                max(0, (self.size.width - width) // 2),
                max(0, (self.size.height - 10) // 2),
            )
            if opening_confirmation:
                self._confirm_opener = self.focused.id if self.focused else None
                self._confirm_position = self._capture_group_position()
                message, label = self._confirmation_content()
                self.query_one("#wb-confirm-text", NoMarkupStatic).update(message)
                actions = self.query_one("#wb-confirm-actions", NavigableOptionList)
                actions.clear_options()
                if self._confirm == "detail-discard":
                    actions.add_option(Option("Keep editing", id="cancel"))
                    actions.add_option(
                        Option(
                            "Accept model edits into catalog draft", id="save-detail"
                        )
                    )
                    actions.add_option(
                        Option("Discard model edits", id="discard-detail")
                    )
                elif self._confirm == "add-discard":
                    actions.add_option(Option("Keep editing", id="cancel"))
                    actions.add_option(Option("Continue setup", id="continue"))
                else:
                    actions.add_option(Option("Keep editing", id="cancel"))
                if self._confirm != "detail-discard":
                    actions.add_option(
                        Option(
                            f"! {label}"
                            if label.startswith(("Discard", "Replace"))
                            else label,
                            id="accept",
                        )
                    )
                actions.highlighted = 0
        self._apply_visibility(visible)
        if opening_confirmation:
            self._restore_view_focus("wb-confirm-actions")
        context = (
            "Unsaved provider draft"
            if self._confirm in {"add-discard", "add-discard-simple"}
            else "Unsaved model edits"
            if self._confirm == "detail-discard"
            else "Confirm change"
            if self._confirm
            else self._detail_context()
            if self._detail
            else f"Edit {self._editing} for {self.state.provider_id if self.state else self._add.name if self._add else 'provider'}"
            if self._editing
            else f"API style for {self.state.provider_id if self.state else self._add.name if self._add else 'provider'}"
            if self._protocol_picker
            else "Choose deployment to edit"
            if self._view == WorkbenchView.DEPLOYMENTS
            else f"Choose {self._preset_field} for {self._preset_role}"
            if self._preset_field and self._preset_role
            else f"Edit {self._preset_role.replace('orchestrator', 'Main assistant').title()} preset"
            if self._view == WorkbenchView.PRESET_EDITOR and self._preset_role
            else "Choose a default preset"
            if self._picker
            else "Choose default presets"
            if self._view == WorkbenchView.PRESETS
            else f"Models: {self._model_filter or 'all providers'}"
            if self._models_view
            else "Choose provider type"
            if self._stage == "choose"
            else f"Models for {self.state.provider_id} · click/Space toggles inclusion · Enter edits details"
            if self._models_list_open and self.state
            else f"Provider: {self.state.provider_id}"
            if self._provider_open and self.state
            else f"New provider: {self._add.name or 'unnamed'} — {self._stage}"
            if self._stage and self._add
            else (
                f"Filter: {self.filter_text or 'type to filter'}  ·  "
                f"{self._visible_provider_count()}"
            )
        )
        self.query_one("#wb-filter", NoMarkupStatic).update(context)
        self.query_one("#wb-count").display = False
        self._group_contents_context["wb-help"] = (
            self._view,
            self._group_context(self._view, "wb-help"),
        )
        return primary

    def _set_feedback(
        self,
        operation: str,
        message: str,
        kind: Literal["running", "success", "error", "warning", "info"],
    ) -> None:
        self._feedback_token += 1
        if kind in {"error", "warning"}:
            self._unresolved[operation] = (message, kind)
        elif kind in {"success", "info"}:
            self._unresolved.pop(operation, None)
        if self._unresolved and kind not in {"error", "warning"}:
            self._message, self._feedback_kind = next(
                reversed(self._unresolved.values())
            )
        else:
            self._message, self._feedback_kind = message, kind
        self._update_help()

    def _feedback(self) -> Text:
        message = self._message
        kind = self._feedback_kind
        if self._unresolved:
            message, kind = next(reversed(self._unresolved.values()))
        if (
            message.startswith("Failed")
            or message.startswith("Discovery Failed")
            or message.startswith("Saved; Reload Failed")
        ):
            kind = "error"
        if not message:
            return Text()
        if kind == "running":
            operation = (
                message
                .removesuffix(chrome_glyph("running"))
                .removesuffix("…")
                .removesuffix("...")
            )
            return Text.assemble(
                (
                    f"{chrome_glyph('running')} ",
                    Style(
                        color=Color.parse(
                            self.app.theme_variables["primary"]
                        ).rich_color
                    ),
                ),
                f"Running: {operation}",
            )
        if kind == "success":
            what = message.removesuffix(".").removesuffix(" saved")
            if message == "Key saved.":
                what = "Credential"
            elif message == "Saved catalog changes.":
                what = "Catalog changes"
            return Text.assemble(
                (
                    f"{chrome_glyph('success')} ",
                    Style(
                        color=Color.parse(
                            self.app.theme_variables["success"]
                        ).rich_color
                    ),
                ),
                f"Saved: {what}",
            )
        if kind == "error":
            reason = (
                message
                .removeprefix("Failed to ")
                .removeprefix("Failed: ")
                .removeprefix("Discovery Failed: ")
            )
            return Text.assemble(
                (
                    f"{chrome_glyph('error')} ",
                    Style(
                        color=Color.parse(self.app.theme_variables["error"]).rich_color
                    ),
                ),
                f"Failed: {reason}",
            )
        if kind == "warning":
            return Text.assemble(
                (
                    f"{chrome_glyph('warning')} ",
                    Style(
                        color=Color.parse(
                            self.app.theme_variables["warning"]
                        ).rich_color
                    ),
                ),
                f"Warning: {message}",
            )
        return Text(message)

    def _root_option(self) -> Option | None:
        group = (
            "#wb-root-actions"
            if self.query_one("#wb-root-actions").has_focus
            or self._help_focus_id == "wb-root-actions"
            else "#wb-providers"
        )
        return self.query_one(group, OptionList).highlighted_option

    def _root_enter_hint(self) -> str:  # noqa: PLR0911
        option = self._root_option()
        if option is not None and option.id == "\x00models":
            return "Open model catalog"
        if option is not None and option.id == "\x00finish":
            return "Finish setup"
        if option is not None and option.id == "\x00add":
            return "Add custom provider"
        if option is not None and option.id == "\x00apply":
            return "Apply changes"
        if option is not None and option.id == "\x00discard":
            return "Discard changes"
        if option is not None and option.id in {
            "\x00empty",
            "\x00heading",
            "\x00actions",
        }:
            return "No action"
        catalog = self.state.catalog if self.state else self.snapshot.catalog
        if option is not None and option.id in catalog.providers:
            return "Set up" if self._mistral_needs_setup(str(option.id)) else "Manage"
        return "Select"

    def _root_escape_hint(self) -> str:
        if self.filter_text:
            return "Clear filter"
        return "Cancel setup" if self.mode == "onboarding" else "Close"

    def _update_help(self) -> None:  # noqa: PLR0912, PLR0914, PLR0915
        if not self.is_mounted or self._unmounted:
            return
        primary = self._sync_view()
        state = self.state
        if self._busy:
            description = (
                "Saving; wait to close"
                if "Saving" in self._message
                else "Operation in progress; wait to close"
            )
        elif self._confirm:
            if self._confirm == "add-discard":
                description = "Choose Keep editing, Continue setup, or Discard edits."
            elif self._confirm == "detail-discard":
                description = "Choose Keep editing, Accept model edits into the catalog draft, or Discard model edits."
            else:
                description = "Choose Cancel to keep the current state."
        elif self._editing:
            description = self.DESCRIPTIONS.get(
                self._editing, "Enter accepts this field into the draft."
            )
        elif self._detail:
            save_destination = (
                "Save and continue to presets persists the draft."
                if self._stage == "models"
                else "Apply all catalog edits to disk later."
            )
            description = (
                f"{self._detail}: {chrome_glyph('vertical')} move; Enter edits a field; "
                "Space toggles images. Tab moves to acceptance actions. Accept model edits returns to the opener; "
                f"{save_destination}"
            )
        elif primary == "models" and state:
            if self.query_one("#wb-models-actions").has_focus:
                action = self.query_one(
                    "#wb-models-actions", WorkbenchList
                ).highlighted_option
                descriptions = {
                    "retry-discovery": "Enter retries model discovery.",
                    "edit-connection": "Enter edits the provider connection.",
                    "manual": "Enter adds a model manually.",
                    "add-another": "Save selected models, then connect another provider.",
                    "continue-presets": "Save selected models, then choose default presets.",
                }
                description = (
                    descriptions.get(
                        str(action.id) if action else "", "Choose a model action."
                    )
                    + " Tab/Shift+Tab cycles models, actions, and help."
                )
                model_value = ""
            else:
                models = self.query_one("#wb-models", ModelChecklist)
                name = (
                    models.get_option_at_index(models.highlighted).value
                    if models.highlighted is not None and models.option_count
                    else ""
                )
                row = next((row for row in state.model_rows() if row[0] == name), None)
                synthetic = {"\x00empty": "No models configured."}
                if name in synthetic:
                    description = synthetic[name]
                else:
                    description = (
                        f"{state.provider_id}/{row[1]} ({row[0]}). Click row or Space toggles inclusion; click [Details] or Enter opens details."
                        if row
                        else "No models configured."
                    )
                    if row and not row[3]:
                        description += " Configured but not returned by discovery; remains editable."
                model_value = name
        elif primary == "presets":
            description = "Each preset has one model and thinking level. Enter edits the selected pair; Tab moves to Save/Add another actions; Esc returns to providers."
        elif primary == "preset-editor":
            description = "Edit Model and Thinking in this preset; Tab moves to Apply/Cancel actions. Apply accepts into the draft, never disk; Esc cancels."
        elif primary == "catalog":
            description = ""
            filter_focused = (
                self._help_focus_id == "wb-catalog-filter"
                or self.query_one("#wb-catalog-filter").has_focus
            )
            if filter_focused:
                description = (
                    f"Provider filter. {chrome_glyph('vertical')} choose a provider; Tab moves to models; "
                    + (
                        "Esc clears this filter first."
                        if self._model_filter
                        else "Esc goes back."
                    )
                )
                option = None
            else:
                group = (
                    "#wb-catalog-actions"
                    if self._help_focus_id == "wb-catalog-actions"
                    or self.query_one("#wb-catalog-actions").has_focus
                    else "#wb-catalog"
                )
                option = self.query_one(group, NavigableOptionList).highlighted_option
            catalog = state.catalog if state else self.snapshot.catalog
            model = (
                catalog.models.get(str(option.id).removeprefix("model:"))
                if option and option.id
                else None
            )
            description = (
                description
                if filter_focused
                else (
                    f"Model: {str(option.id).removeprefix('model:')}. "
                    + "; ".join(
                        f"{dep.provider}/{dep.name}: {dep.prices.model_dump(exclude_none=True) or 'no prices'}"
                        for dep in (model.deployments if model is not None else ())
                    )
                    + ". Enter edits model."
                    if option and str(option.id).startswith("model:")
                    else self.DESCRIPTIONS[str(option.id).removeprefix("\x00")]
                    if option and option.id in {"\x00apply", "\x00discard"}
                    else "Select a provider filter or a model to inspect."
                )
            )
        elif primary == "picker":
            picker = self.query_one("#wb-picker", NavigableOptionList)
            option = picker.highlighted_option
            description = (
                f"Enter selects {option.id} for {self._preset_role}; Esc returns to presets."
                if self._preset_field and option and option.id
                else "Enter opens the selected deployment's model details."
                if self._view == WorkbenchView.DEPLOYMENTS
                else "No models configured."
            )
        elif primary == "actions" and (state or self._add):
            group = self._help_focus_id
            if group not in {
                "wb-actions",
                "wb-provider-operations",
                "wb-connection-actions",
            }:
                group = "wb-actions"
            option = self.query_one(f"#{group}", NavigableOptionList).highlighted_option
            key = str(option.id) if option and option.id else ""
            description = self.DESCRIPTIONS.get(
                key, "Enter accepts fields into the catalog draft; only Apply saves it."
            )
            if key in {"name", "base", "style", "env"}:
                connection = (
                    self._add
                    if self._stage == "connection" and self._add
                    else state.connection
                    if state
                    else None
                )
                value = (
                    self._add.name
                    if key == "name" and self._stage == "connection" and self._add
                    else state.provider_id
                    if key == "name" and state
                    else getattr(connection, self.FIELDS[key], None)
                )
                display_value = value or ("Not required" if key == "env" else "Not set")
                description = f"{display_value}. {description}"
        elif primary == "providers":
            option = self._root_option()
            catalog = self.state.catalog if self.state else self.snapshot.catalog
            if option and option.id in catalog.providers:
                action = (
                    "sets up connection and deployments"
                    if self._mistral_needs_setup(str(option.id))
                    else "manages connection and deployments"
                )
                description = f"{option.id}: Enter {action}."
            elif option and option.id == "\x00models":
                description = "Enter opens the model catalog."
            elif option and option.id == "\x00presets":
                description = "Enter opens default presets. Choose one model and thinking level per role."
            elif option and option.id == "\x00add":
                description = "Enter sets up a custom provider."
            elif option and option.id == "\x00apply":
                description = (
                    "Enter saves all pending provider and role-preset edits to disk."
                )
            elif option and option.id == "\x00discard":
                description = "Enter confirms discarding all pending provider and role-preset edits."
            elif option and option.id in {"\x00empty", "\x00heading", "\x00actions"}:
                description = "No action available."
            else:
                description = "Choose a provider or action."
        else:
            description = (
                "Enter selects the current item; Escape returns to the previous view."
            )
        if self._help_open:
            description = f"Shortcuts: {chrome_glyph('vertical')} move; Space toggles a deployment; Enter opens or accepts; Esc returns; F1 closes help."
        feedback = self._feedback()
        help_text = Text()
        if feedback:
            help_text.append_text(feedback)
        if description and not self._busy:
            if feedback:
                help_text.append("  ")
            help_text.append(description)
        self.query_one("#wb-help", NoMarkupStatic).update(help_text)
        model_option = (
            self.query_one("#wb-models", ModelChecklist).highlighted_option
            if primary == "models"
            else None
        )
        model_value = (
            str(cast(Selection[str], model_option).value)
            if model_option is not None
            else ""
        )
        model_bindings = (
            [("↑↓", "Move"), ("Enter", "Select"), ("Esc", "Back")]
            if self.query_one("#wb-models-actions").has_focus
            else [
                ("↑↓", "Move"),
                ("Space", "Toggle"),
                ("Enter", "Details"),
                ("Esc", "Back"),
            ]
            if model_value != "\x00empty"
            else [("Esc", "Back")]
        )
        picker_option = (
            self.query_one("#wb-picker", NavigableOptionList).highlighted_option
            if primary == "picker"
            else None
        )
        picker_bindings = (
            [("Enter", "Open Models"), ("Esc", "Back")]
            if picker_option is not None and picker_option.id == "\x00repair"
            else [("↑↓", "Move"), ("Enter", "Select"), ("Esc", "Back")]
        )
        catalog_bindings = (
            [
                ("↑↓", "Filter providers"),
                ("Tab", "Models"),
                ("Shift+Tab", "Models"),
                ("Esc", "Clear filter" if self._model_filter else "Back"),
            ]
            if self._help_focus_id == "wb-catalog-filter"
            or self.query_one("#wb-catalog-filter").has_focus
            else [
                ("↑↓", "Move"),
                ("Tab", "Provider filter"),
                ("Shift+Tab", "Provider filter"),
                (
                    "Enter",
                    "Select"
                    if self.query_one("#wb-catalog-actions").has_focus
                    else "Edit",
                ),
                ("Esc", "Back"),
            ]
        )
        groups = self._focus_groups()
        traversal: list[tuple[str, str]] = []
        if primary in {"models", "catalog"} and groups:
            labels = {
                "wb-models": "Models",
                "wb-models-actions": "Actions",
                "wb-catalog-filter": "Filter",
                "wb-catalog": "Models",
                "wb-catalog-actions": "Actions",
                "wb-help": "Help",
            }
            current = self.focused.id if self.focused else None
            index = groups.index(current) if current in groups else 0
            traversal = [
                ("Tab", labels[groups[(index + 1) % len(groups)]]),
                ("Shift+Tab", labels[groups[(index - 1) % len(groups)]]),
            ]
            target_bindings = (
                model_bindings if primary == "models" else catalog_bindings
            )
            target_bindings[:] = traversal + [
                item for item in target_bindings if item[0] not in {"Tab", "Shift+Tab"}
            ]
            if current == "wb-help":
                target_bindings[:] = traversal + [("↑↓", "Scroll"), ("Esc", "Back")]
            help_text.append(
                " Tab moves forward and Shift+Tab moves backward through "
                + ", ".join(labels[group] for group in groups)
                + "; each group remembers its position."
            )
            self.query_one("#wb-help", NoMarkupStatic).update(help_text)

        detail_bindings = [
            ("↑↓", "Move"),
            ("Space", "Toggle images"),
            (
                "Enter",
                "Accept edits"
                if self.query_one("#wb-detail-actions").has_focus
                else "Edit field",
            ),
            ("Esc", "Back"),
        ]
        bindings = (
            [("Esc", "Wait to close")]
            if self._busy
            else [("Enter", "Select"), ("Esc", "Cancel")]
            if self._confirm
            else [("Space", "Select"), ("Enter", "Accept field"), ("Esc", "Back")]
            if self._protocol_picker
            else [
                ("Enter", "Save API key" if self._editing == "key" else "Accept field"),
                ("Esc", "Back"),
            ]
            if self._editing
            else detail_bindings
            if self._detail
            else model_bindings
            if primary == "models"
            else picker_bindings
            if primary == "picker"
            else catalog_bindings
            if primary == "catalog"
            else [
                ("↑↓", "Move"),
                ("Enter", self._root_enter_hint()),
                ("Esc", self._root_escape_hint()),
            ]
            if primary == "providers"
            else [("↑↓", "Move"), ("Enter", "Select"), ("Esc", "Back")]
        )
        if (
            primary
            in {"providers", "actions", "detail-fields", "presets", "preset-editor"}
            and groups
            and not self._confirm
            and not self._busy
            and not self._editing
        ):
            labels = {
                "wb-providers": "Providers",
                "wb-root-actions": "Actions",
                "wb-actions": "Fields",
                "wb-provider-operations": "Operations",
                "wb-connection-actions": "Actions",
                "wb-detail-fields": "Fields",
                "wb-detail-actions": "Actions",
                "wb-presets": "Roles",
                "wb-presets-actions": "Actions",
                "wb-preset-editor": "Fields",
                "wb-preset-editor-actions": "Actions",
                "wb-help": "Help",
            }
            current = self.focused.id if self.focused else None
            index = groups.index(current) if current in groups else 0
            traversal = [
                ("Tab", labels[groups[(index + 1) % len(groups)]]),
                ("Shift+Tab", labels[groups[(index - 1) % len(groups)]]),
            ]
            bindings = traversal + bindings
        width = max(1, self.size.width - 2)
        shown = bindings

        def hint_content(items: list[tuple[str, str]]) -> Content:
            return shortcut_hint(
                "  ".join(f"{shortcut(key)} {label}" for key, label in items)
            )

        if hint_content(bindings).cell_length > width:
            escape_binding = next(
                (item for item in bindings if item[0] == "Esc"), ("Esc", "Back")
            )
            if (
                primary
                in {"catalog", "models", "detail-fields", "presets", "preset-editor"}
                and groups
                and not self._confirm
                and not self._busy
            ):
                shown = traversal + [escape_binding]
            else:
                shown = [
                    next(
                        (item for item in bindings if item[0] == "Enter"), bindings[0]
                    ),
                    escape_binding,
                    ("F1", "Help"),
                ]
            if hint_content(shown).cell_length > width:
                shown = shown[:2]
        self._hint_targets = []
        offset = 0
        for key, label in shown:
            width = hint_content([(key, label)]).cell_length
            self._hint_targets.append((offset, offset + width, key))
            offset += width + 2
        self.query_one("#wb-hint", NoMarkupStatic).update(hint_content(shown))
