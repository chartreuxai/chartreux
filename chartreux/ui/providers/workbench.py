"""Provider Settings for onboarding and in-session catalog management."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import ClassVar, Literal, cast

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Input, Label, OptionList, SelectionList
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

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
    ProviderDraft,
    ProviderWorkbenchResult,
    TLSConfig,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
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
)
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.providers import workbench_types as _workbench_types
from chartreux.ui.providers.management_state import (
    ConnectionDraft,
    CredentialStatusResolver,
    ManagementState,
    PendingModel,
)
from chartreux.ui.providers.workbench_browser import WorkbenchBrowser
from chartreux.ui.providers.workbench_connection import WorkbenchConnection
from chartreux.ui.providers.workbench_flow_policy import WorkbenchFlowPolicy
from chartreux.ui.providers.workbench_model_editor import WorkbenchModelEditor
from chartreux.ui.providers.workbench_navigation import WorkbenchNavigation
from chartreux.ui.providers.workbench_operations import WorkbenchOperations
from chartreux.ui.providers.workbench_presentation import WorkbenchPresentation
from chartreux.ui.providers.workbench_presets import WorkbenchPresets
from chartreux.ui.providers.workbench_types import (
    CURSOR_GUTTER as CURSOR_GUTTER,
    NavigationFrame as NavigationFrame,
    WorkbenchView as WorkbenchView,
    _GroupPosition as _GroupPosition,
)
from chartreux.ui.providers.workbench_widgets import (
    BrowserList as BrowserList,
    DetailHelp as DetailHelp,
    ModelChecklist as ModelChecklist,
    WorkbenchList as WorkbenchList,
    _bounded_cursor as _bounded_cursor,
)
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class ProviderWorkbenchScreen(  # noqa: PLR0904 - widget host surface and delegates
    ModalScreen[ProviderWorkbenchResult]
):
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

    MIN_WIDTH = 48
    MIN_HEIGHT = 24
    VIEW_GROUPS: ClassVar[dict[WorkbenchView, tuple[str, ...]]] = (
        _workbench_types.VIEW_GROUPS
    )
    VIEW_CONTROLS: ClassVar[tuple[str, ...]] = _workbench_types.VIEW_CONTROLS
    UNDERLAY_CONTROLS: ClassVar[tuple[str, ...]] = _workbench_types.UNDERLAY_CONTROLS
    FIELDS: ClassVar[dict[str, str]] = _workbench_types.FIELDS
    DESCRIPTIONS: ClassVar[dict[str, str]] = _workbench_types.DESCRIPTIONS

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
        # Presentation collaborator (workbench_presentation.py); state stays
        # on the screen and the presenter mutates it through host=self.
        self._presentation = WorkbenchPresentation(host=self)
        # Navigation collaborator (workbench_navigation.py); the view stack and
        # focus state stay on the screen and the controller mutates them
        # through host=self.
        self._navigation_controller = WorkbenchNavigation(host=self)
        # Browser collaborator (workbench_browser.py); the provider browser and
        # catalog state stay on the screen and the controller mutates them
        # through host=self.
        self._browser_controller = WorkbenchBrowser(host=self)
        # Connection collaborator (workbench_connection.py); the add/edit
        # connection state stays on the screen and the controller mutates
        # it through host=self.
        self._connection_controller = WorkbenchConnection(host=self)
        # Model editor collaborator (workbench_model_editor.py); the detail
        # state stays on the screen and the controller mutates it through
        # host=self.
        self._model_editor_controller = WorkbenchModelEditor(host=self)
        # Presets collaborator (workbench_presets.py); the presets state
        # stays on the screen and the controller mutates it through
        # host=self.
        self._presets_controller = WorkbenchPresets(host=self)
        # Flow policy collaborator (workbench_flow_policy.py); the policy
        # owns the onboarding mode decisions only — the mode and
        # ManagementState stay on the screen, and the policy reads
        # host.mode at decision time (tests write screen.mode after
        # construction).
        self._flow = WorkbenchFlowPolicy(host=self)
        # Operations collaborator (workbench_operations.py); the commit,
        # reload, and confirmation state stays on the screen and the
        # controller mutates it through host=self.
        self._operations = WorkbenchOperations(host=self)

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
            yield BrowserList(id="wb-providers", host=self)
            yield WorkbenchList(id="wb-root-actions", classes="wb-group", host=self)
            yield WorkbenchList(id="wb-choose", host=self)
            yield WorkbenchList(id="wb-picker", host=self)
            yield WorkbenchList(id="wb-catalog-filter", host=self)
            yield WorkbenchList(id="wb-catalog", host=self)
            yield WorkbenchList(id="wb-catalog-actions", classes="wb-group", host=self)
            yield WorkbenchList(id="wb-presets", host=self)
            yield WorkbenchList(id="wb-presets-actions", classes="wb-group", host=self)
            yield WorkbenchList(id="wb-preset-editor", host=self)
            yield WorkbenchList(
                id="wb-preset-editor-actions", classes="wb-group", host=self
            )
            yield WorkbenchList(id="wb-detail-fields", host=self)
            yield WorkbenchList(id="wb-protocol", host=self)
            yield WorkbenchList(id="wb-actions", host=self)
            yield WorkbenchList(
                id="wb-provider-operations", classes="wb-group", host=self
            )
            yield ModelChecklist(id="wb-models", host=self)
            yield WorkbenchList(id="wb-models-actions", host=self)
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
                yield ModelChecklist(id="wb-image-support", host=self)
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
            yield WorkbenchList(id="wb-detail-actions", host=self)
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
                    yield WorkbenchList(id="wb-connection-style", host=self)
                with Horizontal(classes="connection-row"):
                    yield Label("Credential env var", id="wb-connection-env-label")
                    yield Input(id="wb-connection-env")
                with Horizontal(classes="connection-row"):
                    yield Label("API key", id="wb-connection-key-label")
                    yield Input(id="wb-connection-key", password=True)
            yield WorkbenchList(id="wb-connection-actions", host=self)
            confirm_host = Container(id="wb-confirm-host")
            confirm_host.display = False
            with confirm_host:
                with VerticalScroll(id="wb-confirm"):
                    yield NoMarkupStatic("", id="wb-confirm-text")
                    yield WorkbenchList(id="wb-confirm-actions", host=self)
                    yield NoMarkupStatic(
                        f"{chrome_glyph('vertical')} Choose · Enter Confirm · Esc Keep editing",
                        id="wb-confirm-help",
                    )
            yield DetailHelp("", id="wb-help", host=self)
            yield NoMarkupStatic("", id="wb-hint")
        yield NoMarkupStatic(
            "Terminal too small — need 48x24. Resize to continue.", id="wb-small"
        )

    # WorkbenchWidgetHost surface (workbench_types.py): the widgets in
    # workbench_widgets.py reach the screen only through these late-bound
    # calls, so instance patches on the screen methods stay interceptable.
    def schedule_update_help(self) -> None:
        self.call_after_refresh(self._update_help)

    def update_help(self) -> None:
        self._update_help()

    def record_help_focus(self, group_id: str | None) -> None:
        if group_id != "wb-confirm-actions":
            self._help_focus_id = group_id

    def confirm_open(self) -> bool:
        return self._confirm is not None

    def pointer_focus(self, group_id: str | None) -> bool:
        return self._pointer_focus(group_id)

    def can_activate(self, group_id: str | None) -> bool:
        return self._can_activate(group_id)

    def consume_help_click(self, group_id: str | None) -> bool:
        return self._consume_help_click(group_id)

    def select_protocol(self, value: str) -> None:
        self._select_protocol(value)

    def accept_protocol(self) -> None:
        self._accept_protocol()

    def page_confirm(self, direction: int) -> None:
        confirm = self.query_one("#wb-confirm", VerticalScroll)
        if direction > 0:
            confirm.scroll_page_down(animate=False)
        else:
            confirm.scroll_page_up(animate=False)

    def is_busy(self) -> bool:
        return self._busy

    def get_filter(self) -> str:
        return self.filter_text

    def set_filter(self, text: str) -> None:
        self.filter_text = text

    def refresh_browser(self) -> None:
        self._refresh_browser()

    def selection_changed(self, widget: SelectionList) -> None:
        self.on_selection_list_selected_changed(SelectionList.SelectedChanged(widget))

    def toggle_detail_images(self) -> None:
        self._toggle_detail_images()

    def open_detail(self, name: str) -> None:
        self._open_detail(name)

    # Presentation delegates (workbench_presentation.py): the presenter is a
    # plain object constructed in __init__ with host=self; all state stays on
    # the screen and the presenter mutates it through the host. The same-name
    # delegates keep every internal call site on self._method() attribute
    # resolution, so instance patches on the screen methods stay interceptable.
    def _update_help(self) -> None:
        self._presentation.update_help()

    def _sync_view(self) -> str:
        return self._presentation.sync_view()

    def _visibility_projection(self) -> tuple[str, set[str]]:
        return self._presentation.visibility_projection()

    def _apply_visibility(self, visible: set[str]) -> None:
        self._presentation.apply_visibility(visible)

    def _set_feedback(
        self,
        operation: str,
        message: str,
        kind: Literal["running", "success", "error", "warning", "info"],
    ) -> None:
        self._presentation.set_feedback(operation, message, kind)

    def _feedback(self) -> Text:
        return self._presentation.feedback()

    def _confirmation_content(self) -> tuple[str, str]:
        return self._presentation.confirmation_content()

    def _summary(self, text: str, *, reserve: int = 8) -> str:
        return self._presentation.summary(text, reserve=reserve)

    def _detail_context(self) -> str:
        return self._presentation.detail_context()

    def _root_option(self) -> Option | None:
        return self._presentation.root_option()

    def _root_enter_hint(self) -> str:
        return self._presentation.root_enter_hint()

    def _root_escape_hint(self) -> str:
        return self._presentation.root_escape_hint()

    # Navigation delegates (workbench_navigation.py): the controller is a plain
    # object constructed in __init__ with host=self; the view stack, positions,
    # and focus state stay on the screen and the controller mutates them
    # through the host. The same-name delegates keep every internal call site
    # on self._method() attribute resolution, so instance patches on the
    # screen methods stay interceptable.

    def _can_focus_group(self, group_id: str | None) -> bool:
        return self._navigation_controller.can_focus_group(group_id)

    def _can_activate(self, group_id: str | None) -> bool:
        return self._navigation_controller.can_activate(group_id)

    def _pointer_focus(self, group_id: str | None) -> bool:
        return self._navigation_controller.pointer_focus(group_id)

    def _consume_help_click(self, group_id: str | None) -> bool:
        return self._navigation_controller.consume_help_click(group_id)

    def _focus_groups(self) -> list[str]:
        return self._navigation_controller.focus_groups()

    def action_cycle_group(self, direction: int) -> None:
        self._navigation_controller.cycle_group_body(direction)

    def _group_context(self, view: WorkbenchView, group_id: str) -> tuple[str, ...]:
        return self._navigation_controller.group_context(view, group_id)

    def _row_identity(self, widget: OptionList, index: int) -> str | None:
        return self._navigation_controller.row_identity(widget, index)

    def _capture_group_position(
        self, widget: Widget | None = None
    ) -> _GroupPosition | None:
        return self._navigation_controller.capture_group_position(widget)

    def _position_before_rebuild(
        self, view: WorkbenchView, group: str
    ) -> _GroupPosition | None:
        return self._navigation_controller.position_before_rebuild(view, group)

    def _restore_group_position(
        self, position: _GroupPosition | None, *, restore_focus: bool = False
    ) -> None:
        self._navigation_controller.restore_group_position(
            position, restore_focus=restore_focus
        )

    def _frame(self) -> NavigationFrame:
        return self._navigation_controller.frame()

    def _push_view(self, view: WorkbenchView) -> None:
        self._navigation_controller.push_view(view)

    def _activate_view(
        self,
        view: WorkbenchView,
        *,
        focus_id: str | None = None,
        selected_id: str | None = None,
    ) -> None:
        self._navigation_controller.activate_view(
            view, focus_id=focus_id, selected_id=selected_id
        )

    def _defer_focus(self, widget: Widget) -> None:
        self._navigation_controller.defer_focus(widget)

    def _restore_view_focus(
        self, widget_id: str, *, selected_id: str | None = None
    ) -> None:
        self._navigation_controller.restore_view_focus(
            widget_id, selected_id=selected_id
        )

    def _pop_view(self) -> bool:
        return self._navigation_controller.pop_view()

    def _restore_frame(self, frame: NavigationFrame) -> None:
        self._navigation_controller.restore_frame(frame)

    def _restore_transient_opener(self, frame: NavigationFrame | None) -> None:
        self._navigation_controller.restore_transient_opener(frame)

    def _return_from_editor(self) -> None:
        self._navigation_controller.return_from_editor()

    def action_help(self) -> None:
        self._navigation_controller.help_body()

    def action_back(self) -> None:
        self._navigation_controller.back_body()

    def _can_clear_catalog_filter(self) -> bool:
        return self._navigation_controller.can_clear_catalog_filter()

    def on_resize(self, event: events.Resize) -> None:
        self._navigation_controller.resize_body(event)

    # Browser delegates (workbench_browser.py): the controller is a plain
    # object constructed in __init__ with host=self; the browser filter, cursor,
    # and catalog state stay on the screen and the controller mutates them
    # through the host. The same-name delegates keep every internal call site
    # on self._method() attribute resolution, so instance patches on the
    # screen methods stay interceptable. _friendly and _provider_count_label
    # were staticmethods on the screen; their delegates are plain methods that
    # forward to the controller staticmethods.

    def _refresh_browser(self) -> None:
        self._browser_controller.refresh_browser()

    def _friendly(self, provider_id: str) -> str:
        return self._browser_controller.friendly(provider_id)

    def _visible_provider_count(self) -> str:
        return self._browser_controller.visible_provider_count()

    def _provider_count_label(self, visible: int, total: int) -> str:
        return self._browser_controller.provider_count_label(visible, total)

    def _provider_env(self, name: str) -> str:
        return self._browser_controller.provider_env(name)

    def _credential_ready_for_provider(self, name: str) -> bool:
        return self._browser_controller.credential_ready_for_provider(name)

    def _credential_value(self, env_var: str) -> str | None:
        return self._browser_controller.credential_value(env_var)

    def _api_key_status(self, env_var: str) -> str:
        return self._browser_controller.api_key_status(env_var)

    def _runnable_provider_count(self, name: str) -> int:
        return self._browser_controller.runnable_provider_count(name)

    def _mistral_needs_setup(self, name: str) -> bool:
        return self._browser_controller.mistral_needs_setup(name)

    def _deployment_enabled(self, base: str, deployment: DeploymentDefinition) -> bool:
        return self._browser_controller.deployment_enabled(base, deployment)

    def _provider_has_enabled_model(self, name: str) -> bool:
        return self._browser_controller.provider_has_enabled_model(name)

    def _open_catalog(
        self,
        provider: str | None = None,
        focus: str | None = None,
        *,
        push: bool = True,
    ) -> None:
        self._browser_controller.open_catalog(provider, focus, push=push)

    def _catalog_model_summary(
        self, name: str, definition: BaseModelDefinition | None
    ) -> str:
        return self._browser_controller.catalog_model_summary(name, definition)

    def _refresh_catalog(self, *, preserve_filter_cursor: bool = True) -> None:
        self._browser_controller.refresh_catalog(
            preserve_filter_cursor=preserve_filter_cursor
        )

    def _catalog_model_owners(self, name: str) -> list[str]:
        return self._browser_controller.catalog_model_owners(name)

    def _select_catalog(self, key: str) -> None:
        self._browser_controller.select_catalog(key)

    def _open_deployment_picker(self, name: str, providers: list[str]) -> None:
        self._browser_controller.open_deployment_picker(name, providers)

    # Connection delegates (workbench_connection.py): the controller is a plain
    # object constructed in __init__ with host=self; the add/edit connection
    # state stays on the screen and the controller mutates it through the
    # host. The same-name delegates keep every internal call site on
    # self._method() attribute resolution, so instance patches on the screen
    # methods stay interceptable; the four on_* handlers stay on the screen as
    # thin delegates with their bodies moved. _is_chat_model was a staticmethod
    # on the screen; its delegate is a plain method that forwards to the
    # controller staticmethod. The TWP:78-79 _select_action instance-patch seam
    # rides this move: the controller reaches the select_action body only
    # through this delegate.

    def _start_add(self) -> None:
        self._connection_controller.start_add()

    def _choose_preset(self, preset: ProviderPreset) -> None:
        self._connection_controller.choose_preset(preset)

    def _refresh_add_connection(
        self, *, highlight: str | None = None, selected_id: str | None = None
    ) -> None:
        self._connection_controller.refresh_add_connection(
            highlight=highlight, selected_id=selected_id
        )

    def _refresh_connection_form(self) -> None:
        self._connection_controller.refresh_connection_form()

    def _refresh_connection_actions(self) -> None:
        self._connection_controller.refresh_connection_actions()

    def _connection_action(self, key: str) -> None:
        self._connection_controller.connection_action(key)

    def _open_connection_editor(self, key: str, draft: ProviderDraft) -> None:
        self._connection_controller.open_connection_editor(key, draft)

    def _refresh_protocol(self) -> None:
        self._connection_controller.refresh_protocol()

    def _select_protocol(self, value: str) -> None:
        self._connection_controller.select_protocol(value)

    def _accept_protocol(self) -> None:
        self._connection_controller.accept_protocol()

    def _open_protocol_picker(self, current: str) -> None:
        self._connection_controller.open_protocol_picker(current)

    def _new_provider_id(self, name: str) -> str:
        return self._connection_controller.new_provider_id(name)

    def _refresh_add_models(self, *, highlight: str | None = None) -> None:
        self._connection_controller.refresh_add_models(highlight=highlight)

    def _advance_models(self, action: str) -> None:
        self._connection_controller.advance_models(action)

    def _close_picker(self) -> None:
        self._connection_controller.close_picker()

    def _expand(self, provider_id: str, *, committed: bool = False) -> None:
        self._connection_controller.expand(provider_id, committed=committed)

    def _select_add_action(self, key: str, state: ManagementState) -> bool:
        return self._connection_controller.select_add_action(key, state)

    def _begin_saved_connection_edit(self, state: ManagementState) -> None:
        self._connection_controller.begin_saved_connection_edit(state)

    def _select_action(self, key: str) -> None:
        self._connection_controller.select_action(key)

    def _field_error(self, reason: str) -> None:
        self._connection_controller.field_error(reason)

    def _accept_connection_form_field(self, input_widget: Input) -> None:
        self._connection_controller.accept_connection_form_field(input_widget)

    def _accept_add_connection(self, key: str) -> None:
        self._connection_controller.accept_add_connection(key)

    def _accept_manual_model(self, state: ManagementState) -> None:
        self._connection_controller.accept_manual_model(state)

    def _accept_editor(self) -> None:
        self._connection_controller.accept_editor()

    def _accept_credential(self, value: str, state: ManagementState | None) -> None:
        self._connection_controller.accept_credential(value, state)

    def _persisted_credential_value(self, env_var: str) -> str | None:
        return self._connection_controller.persisted_credential_value(env_var)

    def _bound_roster_identities(self, snapshot: CatalogSnapshot) -> frozenset[str]:
        return self._connection_controller.bound_roster_identities(snapshot)

    def _notify_models_saved(
        self, previously_usable: frozenset[str], roster_before: frozenset[str]
    ) -> None:
        self._connection_controller.notify_models_saved(
            previously_usable, roster_before
        )

    def _save_key(self, value: str) -> None:
        self._connection_controller.save_key(value)

    def _is_chat_model(self, item: DiscoveryItem) -> bool:
        return self._connection_controller.is_chat_model(item)

    def _chat_discovery(
        self,
        state: ManagementState,
        result: DiscoveryResult,
        *,
        provider_id: str | None = None,
    ) -> DiscoveryResult:
        return self._connection_controller.chat_discovery(
            state, result, provider_id=provider_id
        )

    def _discovery_request(
        self, state: ManagementState
    ) -> partial[Coroutine[None, None, None]]:
        return self._connection_controller.discovery_request(state)

    async def _discover(
        self,
        state: ManagementState,
        *,
        request: tuple[str, int, str | None, ConnectionDraft, str] | None = None,
    ) -> None:
        await self._connection_controller.discover(state, request=request)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._connection_controller.input_submitted_body(event)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._connection_controller.input_changed_body(event)

    def on_selection_list_selection_highlighted(
        self, event: SelectionList.SelectionHighlighted
    ) -> None:
        self._connection_controller.selection_highlighted_body(event)

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        self._connection_controller.selection_changed_body(event)

    def _leave_add(self) -> None:
        self._connection_controller.leave_add()

    def _back_add(self) -> None:
        self._connection_controller.back_add()

    # Model editor delegates (workbench_model_editor.py): the controller is a
    # plain object constructed in __init__ with host=self; the detail state
    # stays on the screen and the controller mutates it through the host. The
    # same-name delegates keep every internal call site on self._method()
    # attribute resolution, so instance patches on the screen methods stay
    # interceptable. _canonical_thinking_baseline was a staticmethod on the
    # screen; its delegate is a plain method that forwards to the controller
    # staticmethod (the 18.2b/18.2c precedent; the second and last screen
    # staticmethod). The TWP:120-132 _save_detail dispatch spy rides this move:
    # on_option_list_option_selected resolves the screen attribute.

    def _canonical_thinking_baseline(
        self, state: ManagementState, name: str, item: PendingModel | None
    ) -> str:
        return self._model_editor_controller.canonical_thinking_baseline(
            state, name, item
        )

    def _open_detail(self, name: str, *, push: bool = True) -> None:
        self._model_editor_controller.open_detail(name, push=push)

    def _refresh_detail_fields(self, *, highlight: str | None = None) -> None:
        self._model_editor_controller.refresh_detail_fields(highlight=highlight)

    def _toggle_detail_images(self) -> None:
        self._model_editor_controller.toggle_detail_images()

    def _select_detail_field(self, key: str) -> None:
        self._model_editor_controller.select_detail_field(key)

    def _detail_form_snapshot(self) -> tuple[object, ...]:
        return self._model_editor_controller.detail_form_snapshot()

    def _save_detail(self) -> None:
        self._model_editor_controller.save_detail()

    def _set_detail_error(self, field: str, message: str) -> None:
        self._model_editor_controller.set_detail_error(field, message)

    def _detail_is_modified(self) -> bool:
        return self._model_editor_controller.detail_is_modified()

    def _close_detail(self) -> None:
        self._model_editor_controller.close_detail()

    # Presets delegates (workbench_presets.py): the controller is a plain
    # object constructed in __init__ with host=self; the presets state stays
    # on the screen and the controller mutates it through the host. The
    # same-name delegates keep every internal call site on self._method()
    # attribute resolution, so instance patches on the screen methods stay
    # interceptable. The presets bodies' mode decisions route through the
    # flow policy (workbench_flow_policy.py, constructed in __init__);
    # open_presets reaches the seeding through the host delegate
    # _seed_onboarding_presets (re-homed to the flow policy below). The
    # TWP:120-132 _select_preset_action and _select_preset_editor dispatch
    # spies ride this move: on_option_list_option_selected resolves the
    # screen attributes.

    def _open_presets(self) -> None:
        self._presets_controller.open_presets()

    def _refresh_presets(self, *, highlight: str | None = None) -> None:
        self._presets_controller.refresh_presets(highlight=highlight)

    def _select_preset_action(self, key: str) -> None:
        self._presets_controller.select_preset_action(key)

    def _refresh_preset_editor(self, *, highlight: str | None = None) -> None:
        self._presets_controller.refresh_preset_editor(highlight=highlight)

    def _select_preset_editor(self, key: str) -> None:
        self._presets_controller.select_preset_editor(key)

    def _close_preset_editor(self) -> None:
        self._presets_controller.close_preset_editor()

    def _choose_preset_value(self, value: str) -> None:
        self._presets_controller.choose_preset_value(value)

    # Flow policy delegates (workbench_flow_policy.py): the policy is a
    # plain object constructed in __init__ with host=self; it owns the
    # onboarding mode decisions only — the mode and ManagementState stay on
    # the screen, and the policy reads host.mode at decision time (tests
    # write screen.mode after construction). The same-name delegates keep
    # every internal call site on self._method() attribute resolution, so
    # instance patches on the screen methods stay interceptable.
    # _seed_onboarding_presets re-homes here as the policy's seed (the
    # seeding body; open_presets reaches it through the host) and
    # _remember_onboarding_model_selection re-homes here as
    # remember_selection (the connection controller's selection_changed_body
    # reaches it through the host).

    def _seed_onboarding_presets(self) -> None:
        self._flow.seed(self.state)

    def _remember_onboarding_model_selection(
        self, name: str, selected: set[str]
    ) -> None:
        self._flow.remember_selection(name, selected)

    # Operations delegates (workbench_operations.py): the controller is a
    # plain object constructed in __init__ with host=self; the commit,
    # reload, and confirmation state stays on the screen and the controller
    # mutates it through the host. The same-name delegates keep every
    # internal call site on self._method() attribute resolution, so instance
    # patches on the screen methods stay interceptable. on_unmount stays an
    # async handler whose body moved; the action_confirm_* delegates keep
    # the option-list dispatch and the TWP:78-79 _select_action seam
    # interceptable through action_confirm_no.

    def _apply(self, state: ManagementState) -> None:
        self._operations.apply(state)

    def _request_commit(self) -> None:
        self._operations.request_commit()

    def _set_busy(self, busy: bool) -> None:
        self._operations.set_busy(busy)

    def _start_commit(self, state: ManagementState, *, create: bool) -> None:
        self._operations.start_commit(state, create=create)

    async def _commit(
        self, changes: CatalogChanges, state: ManagementState, *, create: bool
    ) -> None:
        await self._operations.commit(changes, state, create=create)

    async def _reload_catalog(self) -> None:
        await self._operations.reload_catalog()

    def _retry_reload(self) -> None:
        self._operations.retry_reload()

    async def _retry_reload_task(self) -> None:
        await self._operations.retry_reload_task()

    async def on_unmount(self) -> None:
        await self._operations.unmount_body()

    def _restore_confirmation_focus(self) -> None:
        self._operations.restore_confirmation_focus()

    def action_confirm_yes(self) -> None:
        self._operations.confirm_yes_body()

    def action_confirm_continue(self) -> None:
        self._operations.confirm_continue_body()

    def action_confirm_detail_save(self) -> None:
        self._operations.confirm_detail_save_body()

    def action_confirm_detail_discard(self) -> None:
        self._operations.confirm_detail_discard_body()

    def action_confirm_no(self) -> None:
        self._operations.confirm_no_body()

    def _finish(self) -> None:
        self._operations.finish()

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

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "cycle_group":
            return self._navigation_controller.cycle_group_check()
        return super().check_action(action, parameters)

    @property
    def view(self) -> WorkbenchView:
        """Current view for diagnostics and keyboard-flow tests."""
        return self._view

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
