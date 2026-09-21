"""Host-neutral, modal provider-management flow.

The screen deliberately owns only transient draft state.  Hosts provide the three
persistence boundaries from :mod:`chartreux.ui.providers.contracts`, so it is
safe to mount in onboarding and the main TUI alike.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import ClassVar, Literal, cast

from rich.segment import Segment
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widgets import (
    Button,
    Input,
    Label,
    OptionList,
    Select,
    SelectionList,
    Static,
)
from textual.widgets.option_list import Option, OptionDoesNotExist
from textual.widgets.selection_list import Selection
from textual.worker import Worker, WorkerState

from chartreux.core.model_catalog.loader import CatalogLoadError, CatalogSnapshot
from chartreux.core.model_catalog.matching import MatchOutcome, match_discovered_model
from chartreux.core.model_catalog.presets import PRESETS, ProviderPreset
from chartreux.core.model_catalog.schema import ModelCatalog, ProviderDefinition
from chartreux.ui.providers.contracts import (
    ApiStyle,
    CatalogChanges,
    CatalogValidationError,
    CatalogWriter,
    ConfigService,
    CredentialService,
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
    DiscoveryService,
    ModelEdits,
    ModelSelectionDraft,
    OptionalEdit,
    ProviderDraft,
    ProviderFlowResult,
    TLSConfig,
)
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

Step = Literal[
    "overview",
    "choice",
    "form",
    "credential",
    "probe",
    "models",
    "review",
    "again",
    "picker",
]

DEFAULT_ACTIVE_MODEL_OPTION = "\x00default"


def provider_id_for_name(name: str, existing: Iterable[str] = ()) -> str:
    """Return a stable, collision-free ``<name>/default`` provider identifier."""
    stem = "-".join(name.casefold().strip().split()) or "provider"
    used = set(existing)
    candidate = f"{stem}/default"
    index = 2
    while candidate in used:
        candidate = f"{stem}-{index}/default"
        index += 1
    return candidate


def provider_urls(provider: ProviderDraft) -> tuple[str, str]:
    """Return listing and inference previews without normalizing gateway paths."""
    listing = (
        f"{provider.api_base}/v1/models"
        if provider.api_style == "anthropic"
        else f"{provider.api_base}/models"
    )
    if provider.backend == "mistral":
        # The Mistral SDK receives the origin; it appends its versioned endpoint.
        inference = provider.api_base.removesuffix("/v1")
    elif provider.api_style == "anthropic":
        inference = f"{provider.api_base}/v1/messages"
    elif provider.api_style == "openai-responses":
        inference = f"{provider.api_base}/responses"
    else:
        inference = f"{provider.api_base}/chat/completions"
    return listing, inference


def validate_provider_draft(provider: ProviderDraft) -> str | None:
    """Validate form facts not represented by the catalog schema."""
    if not provider.name.strip():
        return "Provider name is required."
    if not provider.api_base.startswith(("http://", "https://")):
        return "API base must be an HTTP(S) URL."
    if provider.backend == "mistral":
        if not provider.api_base.rstrip("/").endswith("/v1"):
            return "Mistral requires a versioned API base ending in /v1."
        if provider.reasoning_field_name != "reasoning_content":
            return "Mistral does not support a custom reasoning field."
    if (
        provider.api_key_env_var
        and not provider.api_key_env_var.replace("_", "a").isalnum()
    ):
        return "Credential environment variable is invalid."
    return None


class ChatModelSelectionList(SelectionList[str]):
    """Selection list with unambiguous empty and selected checkbox glyphs."""

    BINDINGS: ClassVar[list[BindingType]] = [
        *SelectionList.BINDINGS,
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def render_line(self, y: int) -> Strip:
        line = super().render_line(y)
        index = self.scroll_offset.y + y
        try:
            selected = self.get_option_at_index(index).value in self.selected
        except OptionDoesNotExist:
            return line
        replaced = False
        segments: list[Segment] = []
        for segment in line:
            if not replaced and segment.text == "X":
                segments.append(Segment("✓" if selected else " ", segment.style))
                replaced = True
            else:
                segments.append(segment)
        return Strip(segments)


class ProviderManagementScreen(ModalScreen[ProviderFlowResult]):
    """A fresh, full-screen provider flow with injected service boundaries."""

    DEFAULT_CSS = """
    ProviderManagementScreen { align: center middle; }
    #provider-management { width: 90%; height: 100%; padding: 0 2; border: round $primary; }
    #provider-management #flow-title { color: $primary; text-style: bold; }
    #provider-management #flow-content { height: 1fr; overflow-y: auto; }
    #provider-management #provider-actions { height: auto; }
    #provider-management #provider-shortcut-hint { color: $text-muted; height: 1; }
    #provider-management .primary-actions Button { text-style: bold; }
    #provider-management .destructive-actions { height: auto; margin-top: 1; }
    #provider-management .destructive-actions Button { color: $error; }
    #provider-management .field-help { color: $text-muted; }
    #provider-management .read-only-value { color: $text-muted; }
    #provider-management .advanced-actions { height: auto; layout: grid; grid-size: 3; }
    #provider-management .error { color: $error; }
    #provider-management .muted { color: $text-muted; }
    #provider-management Input {
        width: 1fr;
        height: 3;
        border: solid $foreground-muted;
    }
    #provider-management Input:focus { border: solid $primary; }
    #provider-management Input.-invalid { border: solid $error; }
    #provider-management #overview-provider,
    #provider-management #active-model { width: 100%; max-height: 50vh; border: none; }
    #provider-management #overview-provider:focus,
    #provider-management #active-model:focus { border: none; }
    #provider-management SelectionList { height: 1fr; }
    #provider-management #models > .selection-list--button { color: $foreground-muted; background: $surface; }
    #provider-management #models > .selection-list--button-highlighted { color: $foreground; background: $primary 20%; }
    #provider-management #models > .selection-list--button-selected { color: $success; background: $success 20%; text-style: bold; }
    #provider-management #models > .selection-list--button-selected-highlighted { color: $success; background: $success 35%; text-style: bold reverse; }
    #provider-management #actions { height: auto; layout: grid; grid-size: 3; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", "Back", show=False, priority=True),
        Binding("j", "picker_down", "Down", show=False),
        Binding("k", "picker_up", "Up", show=False),
    ]

    def __init__(
        self,
        *,
        discovery: DiscoveryService,
        catalog_writer: CatalogWriter,
        credentials: CredentialService,
        config: ConfigService,
        snapshot: CatalogSnapshot,
        management: bool = False,
        tls: TLSConfig | None = None,
        validate_selection: Callable[[str | None], str | None] | None = None,
        initial_active_model: str | None = None,
        initial_step: Step | None = None,
    ) -> None:
        super().__init__()
        self.discovery = discovery
        self.catalog_writer = catalog_writer
        self.credentials = credentials
        self.config = config
        self.snapshot = snapshot
        self.management = management
        self.tls = tls or TLSConfig()
        # A missing callback deliberately skips host-specific runtime validation.
        # ``None`` selection asks the host to validate its configured active model.
        self.validate_selection = validate_selection
        self.step: Step = initial_step or ("overview" if management else "choice")
        self.provider: ProviderDraft | None = None
        self._selected_models: dict[str, ModelSelectionDraft] = {}
        self._selected_wires: set[str] = set()
        self._detail_wire: str | None = None
        self._tag_orders: dict[str, tuple[str, ...]] = dict(snapshot.catalog.tags)
        self._review_inputs: dict[str, dict[str, str]] = {}
        self._overview_provider_id: str | None = None
        self._active_model_expression = initial_active_model
        self._invalid_inputs: set[str] = set()
        self._management_action: Literal["edit", "discover", "credential"] | None = None
        self.discovered: tuple[DiscoveryItem, ...] = ()
        self.error: str | None = None
        self.status: str | None = None
        self._changed = False
        self._probe_worker: Worker[DiscoveryResult | DiscoveryError] | None = None
        self._probe_generation = 0
        self._probe_context: tuple[int, ProviderDraft] | None = None
        self._credential_collision_warning: str | None = None
        self._credential_value: str | None = None
        self._warning: str | None = None
        self._dismissed = False
        self._composing_actions = False

    @property
    def catalog(self) -> ModelCatalog:
        return self.snapshot.catalog

    def compose(self) -> ComposeResult:
        with Vertical(id="provider-management"):
            yield NoMarkupStatic("Provider Management", id="flow-title")
            with Vertical(id="flow-content"):
                yield from self._step_widgets()
            with Vertical(id="provider-actions"):
                yield Static(
                    shortcut_hint(self._shortcut_hint()), id="provider-shortcut-hint"
                )
                self._composing_actions = True
                yield from self._step_widgets()
                self._composing_actions = False

    def _step_widgets(self) -> ComposeResult:  # noqa: PLR0912, PLR0915
        if self._composing_actions:
            yield from self._action_buttons()
            return
        if self.step == "overview":
            yield Label("Providers")
            providers = sorted(self.catalog.providers)
            yield NavigableOptionList(
                *[
                    Option(
                        self._option_text(
                            self._overview_label(provider_id),
                            provider_id == self._overview_provider_id,
                        ),
                        id=provider_id,
                    )
                    for provider_id in providers
                ],
                id="overview-provider",
            )
            if self.error:
                yield NoMarkupStatic(self.error, classes="error")
            if not self.catalog.providers:
                yield Static("No providers yet.")
            yield from self._buttons(
                ("add", "Add provider"),
                ("use", "Use provider"),
                ("edit", "Edit connection details"),
                ("discover", "Discover + add models"),
                ("credential", "Replace credential"),
                ("cancel", "Close"),
            )
        elif self.step == "choice":
            yield Label("Choose a provider")
            yield from self._buttons(
                ("mistral", "Use Mistral"),
                ("add", "Add a provider"),
                *[
                    (
                        self._onboarding_provider_button_id(provider_id),
                        f"Use {self._provider_choice_label(provider_id)}",
                    )
                    for provider_id in self._available_shipped_keyless_provider_ids()
                ],
                ("cancel", "Cancel"),
            )
        elif self.step == "form":
            current = self.provider
            preset_options = [(preset.name, preset.id) for preset in PRESETS]
            yield Label("Preset")
            yield Select(
                preset_options,
                value=current.preset
                if current and current.preset
                else "generic-openai",
                id="preset",
                prompt="Preset",
            )
            yield Label("Provider name *")
            yield Input(
                current.name if current else "",
                placeholder="Provider name",
                id="name",
                validate_on=["blur", "submitted"],
                disabled=bool(
                    current and current.provider_id in self.catalog.providers
                ),
            )
            if current and current.provider_id in self.catalog.providers:
                yield Static(
                    "Provider names are fixed after creation.", classes="muted"
                )
            yield Label("API base *")
            yield Input(
                current.api_base if current else "",
                placeholder="API base",
                id="api-base",
                validate_on=["blur", "submitted"],
            )
            yield Label("API style")
            yield Static(
                "API style controls how requests are formatted for the provider.",
                classes="muted",
            )
            yield Select(
                [
                    ("OpenAI-style", "openai"),
                    ("OpenAI Responses", "openai-responses"),
                    ("Anthropic-style", "anthropic"),
                ],
                value=current.api_style if current else "openai",
                id="api-style",
            )
            yield Label("Credential environment variable")
            yield Input(
                current.api_key_env_var if current else "",
                placeholder="API key environment variable (blank = no authentication)",
                id="env-var",
                validate_on=["blur", "submitted"],
            )
            yield Label("Reasoning field")
            yield Input(
                current.reasoning_field_name if current else "reasoning_content",
                placeholder="Reasoning field",
                id="reasoning-field",
                validate_on=["blur", "submitted"],
            )
            listing, inference = provider_urls(current) if current else ("", "")
            yield NoMarkupStatic(
                f"Provider ID: {current.provider_id if current else '<name>/default'}\n"
                f"Listing URL: {listing}\nInference URL: {inference}",
                classes="muted",
                id="urls",
            )
            if self.error:
                yield NoMarkupStatic(self.error, classes="error")
            if self.status:
                yield NoMarkupStatic(self.status, classes="muted")
            yield from self._buttons(
                ("continue", "Continue"), ("back", "Back"), ("cancel", "Cancel")
            )
        elif self.step == "credential":
            assert self.provider is not None
            yield Label("Credential")
            yield NoMarkupStatic(
                "No authentication is enabled."
                if not self.provider.api_key_env_var
                else (
                    f"Save key as {self.provider.api_key_env_var}. Keys are masked."
                    " If disk persistence is unavailable, the key remains available"
                    " only for this session."
                )
            )
            if self.provider.api_key_env_var:
                yield Label("API Key *")
                yield Input(
                    self._credential_value or "",
                    password=True,
                    placeholder="API key",
                    id="key",
                    validate_on=["blur", "submitted"],
                )
            if self.error:
                yield NoMarkupStatic(self.error, classes="error")
            yield from self._buttons(
                ("continue", "Continue"), ("back", "Back"), ("cancel", "Cancel")
            )
        elif self.step == "probe":
            yield Label("Discover models")
            yield NoMarkupStatic(
                self.error
                or "Ready to list models. Discovery success only means models were discovered; it does not prove inference support."
            )
            yield from self._buttons(
                ("retry", "Retry"),
                ("manual", "Manual entry"),
                ("edit-key", "Edit key"),
                ("back", "Back"),
                ("cancel", "Cancel"),
            )
        elif self.step == "models":
            picker_items = self._picker_items()
            yield Label("Search")
            yield Input(
                placeholder="Search model names",
                id="search",
                validate_on=["blur", "submitted"],
            )
            if any(not item.wire_id for item in self.discovered):
                yield Label("Wire ID")
                yield Input(
                    placeholder="Enter the provider wire name", id="manual-wire-name"
                )
            yield NoMarkupStatic(
                f"{len(picker_items)} of {len(picker_items)} chat models",
                id="model-count",
            )
            yield ChatModelSelectionList(
                *[
                    (
                        self._model_label(item),
                        item.wire_id,
                        item.wire_id in self._selected_wires
                        or item.wire_id in self._selected_models
                        or self._is_configured_model(item),
                    )
                    for item in picker_items
                ],
                id="models",
            )
            if self.error:
                yield NoMarkupStatic(self.error, classes="error")
            yield from self._buttons(
                ("continue", "Review selected models"),
                ("back", "Back"),
                ("cancel", "Cancel"),
            )
        elif self.step == "review":
            selection = self._review_selection()
            yield Label("Selected models")
            yield Select(
                [
                    (selection.wire_name, selection.wire_name)
                    for selection in self._selected_models.values()
                ],
                id="detail-model",
                prompt="Model details",
                value=selection.wire_name if selection else Select.NULL,
            )
            if selection is not None:
                yield from self._collision_widgets(selection)
            yield Label("Base Name")
            yield NoMarkupStatic(
                selection.base_name if selection else "", classes="read-only-value"
            )
            yield Label("Input Price Per Million Tokens")
            yield Input(
                self._review_value(
                    selection, "input-price", self._edit_value(selection, "input_price")
                ),
                placeholder="Input price per million tokens (blank = unknown)",
                id="input-price",
                validate_on=["blur", "submitted"],
            )
            yield Label("Output Price Per Million Tokens")
            yield Input(
                self._review_value(
                    selection,
                    "output-price",
                    self._edit_value(selection, "output_price"),
                ),
                placeholder="Output price per million tokens (blank = unknown)",
                id="output-price",
                validate_on=["blur", "submitted"],
            )
            yield Label("Cached Input Price Per Million Tokens")
            yield Input(
                self._review_value(
                    selection,
                    "cached-price",
                    self._edit_value(selection, "cached_input_price"),
                ),
                placeholder="Cached input price per million tokens (blank = unknown)",
                id="cached-price",
                validate_on=["blur", "submitted"],
            )
            yield Label("Aliases")
            yield Static(
                "Alternative names you can use to select this model.",
                classes="field-help",
            )
            yield Input(
                self._review_value(
                    selection, "aliases", self._edit_value(selection, "aliases")
                ),
                placeholder="Aliases, comma separated",
                id="aliases",
                validate_on=["blur", "submitted"],
            )
            yield Label("Tags")
            yield Static(
                "Groups that let you select related models together.",
                classes="field-help",
            )
            yield Input(
                self._review_value(
                    selection, "tags", self._edit_value(selection, "tags")
                ),
                placeholder="Tags, comma separated",
                id="tags",
                validate_on=["blur", "submitted"],
            )
            if self.error:
                yield NoMarkupStatic(self.error, classes="error")
            yield from self._buttons(
                ("save-details", "Save details"),
                ("clear-prices", "Clear prices"),
                ("clear-aliases", "Clear aliases"),
                ("clear-tags", "Clear tags"),
                ("bulk-tags", "Assign tags to all selected"),
                ("continue", "Save / Continue"),
                ("back", "Back"),
                ("cancel", "Cancel"),
            )
        elif self.step == "again":
            yield Label("Provider saved.")
            if self.error:
                yield NoMarkupStatic(self.error, classes="error")
            yield from self._buttons(
                ("add", "Add another provider"),
                ("picker", "Choose active model"),
                ("finish-unchanged", "Finish without changing active model"),
            )
        else:
            options = self._active_model_picker_options()
            unavailable = self._unavailable_model_labels()
            yield Label("Choose Active Model")
            if self.error:
                yield NoMarkupStatic(self.error, classes="error")
            yield NavigableOptionList(
                *[
                    Option(
                        self._option_text(
                            label,
                            value == self._active_model_expression
                            or (
                                value == DEFAULT_ACTIVE_MODEL_OPTION
                                and self._active_model_expression
                                not in {
                                    option_value
                                    for _option_label, option_value in self._active_model_options()
                                }
                            ),
                        ),
                        id=value,
                    )
                    for label, value in options
                ],
                id="active-model",
            )
            if unavailable:
                yield NoMarkupStatic(
                    "Unavailable (visible but not selectable):\n"
                    + "\n".join(unavailable),
                    classes="muted",
                )
            yield Static(
                "Labels identify deployments; saved values are canonical names, aliases, or @tags.",
                classes="muted",
            )
            yield from self._buttons(("finish", "Finish"), ("cancel", "Cancel"))

    def _shortcut_hint(self) -> str:
        if self.step == "overview":
            return (
                f"{shortcut('↑↓/jk')} Navigate  {shortcut('Enter')} Select  "
                f"{shortcut('Esc')} Close"
            )
        if self.step == "picker":
            return (
                f"{shortcut('↑↓/jk')} Navigate  {shortcut('Enter')} Select  "
                f"{shortcut('Esc')} Back"
            )
        if self.step == "models":
            return (
                f"{shortcut('↑↓/jk')} Navigate  {shortcut('Space/Enter')} Toggle  "
                f"{shortcut('Search: Enter')} Continue  {shortcut('Esc')} Back"
            )
        if self.step in {"form", "credential", "review"}:
            return f"{shortcut('Enter')} Continue  {shortcut('Esc')} Back"
        if self.step == "choice":
            return f"{shortcut('Esc')} Cancel"
        return f"{shortcut('Esc')} Back"

    @staticmethod
    def _option_text(label: str, is_current: bool) -> Text:
        """Render house-style current markers with dimmed row metadata."""
        primary, separator, metadata = label.partition("\t")
        text = Text(no_wrap=True)
        text.append("› " if is_current else "  ", style="green" if is_current else "")
        text.append(primary, style="bold" if is_current else "")
        if separator:
            text.append(f"  {metadata}", style="dim")
        return text

    def on_mount(self) -> None:
        self._focus_current_picker()

    def _focus_current_picker(self) -> None:
        if self.step == "overview" and self.query("#overview-provider"):
            option_list = self.query_one("#overview-provider", OptionList)
            providers = sorted(self.catalog.providers)
            if self._overview_provider_id in providers:
                option_list.highlighted = providers.index(self._overview_provider_id)
            elif providers:
                option_list.highlighted = 0
            option_list.focus()
        elif self.step == "models" and self.query("#models"):
            models = self.query_one("#models", SelectionList)
            if models.option_count:
                models.highlighted = 0
            models.focus()
        elif self.step == "again" and self.query("#picker"):
            self.query_one("#picker", Button).focus()
        elif self.step == "picker" and self.query("#active-model"):
            option_list = self.query_one("#active-model", OptionList)
            options = self._active_model_picker_options()
            selected = (
                self._active_model_expression
                if self._active_model_expression in {value for _label, value in options}
                else DEFAULT_ACTIVE_MODEL_OPTION
            )
            option_list.highlighted = next(
                index
                for index, (_label, value) in enumerate(options)
                if value == selected
            )
            option_list.focus()
        self._apply_invalid_input_state()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if not event.option.id:
            return
        if event.option_list.id == "overview-provider" and self.step == "overview":
            self._overview_provider_id = event.option.id
            self._show("overview")
        elif event.option_list.id == "active-model" and self.step == "picker":
            expression = (
                ""
                if event.option.id == DEFAULT_ACTIVE_MODEL_OPTION
                else event.option.id
            )
            self._active_model_expression = expression
            self._finish(expression)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit each single-line field through the same primary action as Continue."""
        if self.step in {"form", "credential", "models", "review"}:
            self._continue()

    def _review_selection(self) -> ModelSelectionDraft | None:
        """Return the selected detail row, defaulting to insertion order."""
        if self._detail_wire is not None:
            return self._selected_models.get(self._detail_wire)
        return next(iter(self._selected_models.values()), None)

    def _edit_value(self, selection: ModelSelectionDraft | None, field: str) -> str:
        if selection is None:
            return ""
        value = getattr(selection.edits, field).value
        if value is None:
            return ""
        return ", ".join(value) if isinstance(value, tuple) else str(value)

    def _review_value(
        self, selection: ModelSelectionDraft | None, field: str, default: str
    ) -> str:
        if selection is None:
            return default
        return self._review_inputs.get(selection.wire_name, {}).get(field, default)

    def _capture_review_inputs(self, wire: str | None = None) -> str | None:
        if self.step != "review":
            return wire
        current = self._review_selection()
        wire = wire or (current.wire_name if current else None)
        if wire is None or not self.query("#input-price"):
            return wire
        self._review_inputs[wire] = {
            field: self.query_one(f"#{field}", Input).value
            for field in (
                "input-price",
                "output-price",
                "cached-price",
                "aliases",
                "tags",
            )
        }
        return wire

    def _clear_review_input(self, selection: ModelSelectionDraft, *fields: str) -> None:
        values = self._review_inputs.setdefault(selection.wire_name, {})
        for field in fields:
            values[field] = ""

    def _collision_widgets(self, selection: ModelSelectionDraft) -> ComposeResult:
        outcome = match_discovered_model(
            self.catalog,
            self.provider_id,
            selection.wire_name,
            provider=self._provider_definition(),
        )
        if outcome.kind == "occupied_slot":
            yield Label(
                "This provider already occupies the proposed base. Choose an explicit resolution."
            )
            yield Select(
                [(f"Add as deployment to {outcome.existing_base}", "existing")],
                value="existing",
                id="collision-choice",
            )
        elif outcome.kind == "multiple_matches":
            yield Label(
                "Multiple configured deployments match this wire name. Choose its base."
            )
            yield Select(
                [(match.base_name, match.base_name) for match in outcome.matches],
                value=outcome.matches[0].base_name,
                id="collision-choice",
            )
        elif outcome.kind == "alias_collision":
            yield NoMarkupStatic(
                f"{selection.wire_name} is an alias of {outcome.existing_base}; it will be added to that base."
            )

    def _action_buttons(self) -> ComposeResult:
        buttons: tuple[tuple[str, str] | tuple[str, str, bool], ...]
        if self.step == "overview":
            unavailable = not self.catalog.providers
            selected_provider_id = self._overview_provider_id
            selected = (
                self.catalog.providers.get(selected_provider_id)
                if selected_provider_id
                else None
            )
            use_available = (
                selected is not None
                and selected_provider_id is not None
                and not self._is_configured_provider(selected_provider_id)
            )
            credential_available = bool(selected and selected.api_key_env_var) or (
                selected is None
                and len(self.catalog.providers) == 1
                and next(iter(self.catalog.providers.values())).api_key_env_var
            )
            buttons = (
                *((("use", "Use provider"),) if use_available else ()),
                ("add", "Add provider"),
                ("edit", "Edit connection details", unavailable),
                ("discover", "Discover + add models", unavailable),
                *(
                    (("credential", "Replace credential"),)
                    if credential_available
                    else ()
                ),
                ("cancel", "Close"),
            )
        elif self.step == "choice":
            buttons = (
                ("mistral", "Use Mistral"),
                ("add", "Add a provider"),
                *[
                    (
                        self._onboarding_provider_button_id(provider_id),
                        f"Use {self._provider_choice_label(provider_id)}",
                    )
                    for provider_id in self._available_shipped_keyless_provider_ids()
                ],
                ("cancel", "Cancel"),
            )
        elif self.step == "form":
            buttons = (("continue", "Continue"), ("back", "Back"), ("cancel", "Cancel"))
        elif self.step == "credential":
            buttons = (("continue", "Continue"), ("back", "Back"), ("cancel", "Cancel"))
        elif self.step == "probe":
            busy = self.error == "Discovering models…"
            buttons = (
                ("retry", "Retry", busy),
                ("manual", "Manual entry", busy),
                ("edit-key", "Edit key", busy),
                ("back", "Back"),
                ("cancel", "Cancel"),
            )
        elif self.step == "models":
            buttons = (
                ("continue", "Review selected models"),
                ("back", "Back"),
                ("cancel", "Cancel"),
            )
        elif self.step == "review":
            with Horizontal(classes="primary-actions"):
                yield Button("Save Models", id="continue")
                yield Button("Back", id="back")
                yield Button("Cancel", id="cancel")
            with Horizontal(classes="advanced-actions"):
                yield Button("Apply Tags To Selected", id="bulk-tags")
            with Horizontal(classes="destructive-actions"):
                yield Button("Clear Prices", id="clear-prices")
                yield Button("Clear Aliases", id="clear-aliases")
                yield Button("Clear Tags", id="clear-tags")
            return
        elif self.step == "again":
            buttons = (
                ("add", "Add another provider"),
                ("picker", "Choose active model"),
                ("finish-unchanged", "Finish without changing active model"),
            )
        else:
            buttons = (("cancel", "Cancel"),)
        yield from self._buttons(*buttons)

    def _buttons(
        self, *buttons: tuple[str, str] | tuple[str, str, bool]
    ) -> ComposeResult:
        if not self._composing_actions:
            return
        with Horizontal(id="actions"):
            for button in buttons:
                button_id, label = button[:2]
                disabled = button[-1] if isinstance(button[-1], bool) else False
                yield Button(label, id=button_id, disabled=disabled)

    @staticmethod
    def _is_chat_model(item: DiscoveryItem) -> bool:
        """Exclude known non-chat families from the interactive discovery picker."""
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

    def _configured_base(self, item: DiscoveryItem) -> str | None:
        outcome = match_discovered_model(
            self.catalog,
            self.provider_id,
            item.wire_id,
            provider=self._provider_definition(),
        )
        return (
            outcome.existing_base
            if outcome.kind in {"existing", "alias_collision"}
            else None
        )

    def _picker_items(self) -> tuple[DiscoveryItem, ...]:
        """Return one chat-capable row for each canonical configured or wire ID."""
        groups: dict[str, list[DiscoveryItem]] = {}
        for item in self.discovered:
            if item.wire_id and self._is_chat_model(item):
                groups.setdefault(
                    self._configured_base(item) or item.wire_id, []
                ).append(item)
        selected: list[DiscoveryItem] = []
        for base, items in groups.items():
            definition = self.catalog.models.get(base)
            deployment_names = (
                {
                    deployment.name
                    for deployment in definition.deployments
                    if deployment.provider == self.provider_id
                }
                if definition
                else set()
            )
            selected.append(
                next(
                    (item for item in items if item.wire_id in deployment_names),
                    items[0],
                )
            )
        return tuple(sorted(selected, key=lambda item: item.wire_id.casefold()))

    def _is_configured_model(self, item: DiscoveryItem) -> bool:
        return self._configured_base(item) is not None

    def _model_label(self, item: DiscoveryItem) -> str:
        """Render one canonical target with dim aliases as supporting metadata."""
        base = self._configured_base(item)
        if base is not None:
            aliases = self.catalog.models[base].aliases
            metadata = f"aliases: {', '.join(aliases)}" if aliases else base
            return f"{item.wire_id}\t{metadata}"
        return (
            f"{item.wire_id}\t{item.display_label}"
            if item.display_label and item.display_label != item.wire_id
            else item.wire_id
        )

    def _is_configured_provider(self, provider_id: str) -> bool:
        """Return whether a provider is user-configured rather than only shipped."""
        from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG

        definition = self.catalog.providers[provider_id]
        shipped_definition = SHIPPED_CATALOG.providers.get(provider_id)
        return (
            shipped_definition is None
            or definition != shipped_definition
            or bool(self._configured_credential(definition.api_key_env_var))
        )

    def _overview_label(self, provider_id: str) -> str:
        definition = self.catalog.providers[provider_id]
        friendly_name = provider_id.rsplit("/", 1)[0].replace("-", " ").title()
        model_count = sum(
            1
            for model in self.catalog.models.values()
            for deployment in model.deployments
            if deployment.provider == provider_id
        )
        if not definition.api_key_env_var:
            auth_status = "no authentication"
        elif self._configured_credential(definition.api_key_env_var):
            auth_status = "api key set"
        else:
            auth_status = "api key required"
        status = (
            "Configured" if self._is_configured_provider(provider_id) else "Available"
        )
        return (
            f"{friendly_name}\t{status}; {provider_id}, {definition.api_base}, "
            f"{model_count} models, {auth_status}"
        )

    @property
    def provider_id(self) -> str:
        return self.provider.provider_id if self.provider else "new/default"

    def _provider_definition(self) -> ProviderDefinition | None:
        if self.provider is None:
            return None
        return ProviderDefinition(
            api_base=self.provider.api_base,
            api_key_env_var=self.provider.api_key_env_var,
            api_style=self.provider.api_style,
            backend=self.provider.backend,
            reasoning_field_name=self.provider.reasoning_field_name,
        )

    @staticmethod
    def _onboarding_provider_button_id(provider_id: str) -> str:
        return f"use-provider-{provider_id.replace('/', '-')}"

    @staticmethod
    def _provider_choice_label(provider_id: str) -> str:
        name, separator, variant = provider_id.partition("/")
        return f"{name.title()} ({variant})" if separator else name.title()

    def _available_shipped_keyless_provider_ids(self) -> tuple[str, ...]:
        """Return unconfigured shipped providers that onboarding can use directly."""
        from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG

        if self.management:
            return ()
        return tuple(
            provider_id
            for provider_id, definition in sorted(self.catalog.providers.items())
            if (
                provider_id in SHIPPED_CATALOG.providers
                and not definition.api_key_env_var
                and not self._is_configured_provider(provider_id)
            )
        )

    def _adopt_existing_provider(self, provider_id: str | None = None) -> bool:
        """Select a shipped provider and continue through its required setup path."""
        if provider_id is not None:
            self._overview_provider_id = provider_id
        if not self._select_existing_provider():
            return False
        self._reset_unsaved_model_metadata()
        self._management_action = None
        assert self.provider is not None
        if self.provider.api_key_env_var:
            self._show("credential")
        else:
            self._show("probe")
            self._run_probe()
        return True

    def on_button_pressed(  # noqa: PLR0911, PLR0912, PLR0915
        self, event: Button.Pressed
    ) -> None:
        button = event.button.id
        if button in {"cancel", "Close"}:
            self.action_cancel()
            return
        if button == "back":
            self.action_back()
            return
        if button == "mistral":
            if not self.management and self._select_onboarding_mistral_provider():
                self._show("credential")
            else:
                self._apply_preset("mistral")
                self._show("form")
            return
        onboarding_provider_ids = self._available_shipped_keyless_provider_ids()
        if button in {
            self._onboarding_provider_button_id(provider_id)
            for provider_id in onboarding_provider_ids
        }:
            provider_id = next(
                provider_id
                for provider_id in onboarding_provider_ids
                if self._onboarding_provider_button_id(provider_id) == button
            )
            self._adopt_existing_provider(provider_id)
            return
        if button == "use":
            self._adopt_existing_provider()
            return
        if button == "add":
            self._management_action = None
            self._reset_unsaved_model_metadata()
            if self.step == "overview":
                self.provider = None
                self._show("choice")
            else:
                if self.provider is None:
                    self._apply_preset("generic-openai")
                self._show("form")
            return
        if button in {"edit", "discover", "credential"}:
            if not self.catalog.providers:
                self.error = "Select a provider first."
                self._show("overview")
                return
            if len(self.catalog.providers) > 1 and self._overview_provider_id is None:
                self.error = "Select a provider first."
                self._show("overview")
                return
            if not self._select_existing_provider():
                return
            self._reset_unsaved_model_metadata()
            self._management_action = cast(
                Literal["edit", "discover", "credential"], button
            )
            self._show(
                "credential"
                if button == "credential"
                else "form"
                if button == "edit"
                else "probe"
            )
            if button == "discover":
                self._run_probe()
            return
        if button == "continue":
            self._continue()
            return
        if button == "retry":
            self._run_probe()
            return
        if button == "edit-key":
            self._show("credential")
            return
        if button == "manual":
            self.discovered = (DiscoveryItem(""),)
            self._show("models")
            return
        if button == "clear-prices":
            self._clear_prices()
            return
        if button in {"clear-aliases", "clear-tags"}:
            self._clear_detail_metadata(
                "aliases" if button == "clear-aliases" else "tags"
            )
            return
        if button == "bulk-tags":
            self._assign_bulk_tags()
            return
        if button == "picker":
            self._show("picker")
            return
        if button == "finish-unchanged":
            if not self._validate_selection(None, step="again"):
                return
            self._dismissed = True
            self.dismiss(
                ProviderFlowResult(
                    "completed", changed=self._changed, warning=self._warning
                )
            )
            return
        if button == "finish":
            self._finish()

    def _continue(self) -> None:
        if self.step == "form":
            self._save_form()
        elif self.step == "credential":
            self._save_credential()
        elif self.step == "models":
            self._save_model_selection()
        elif self.step == "review":
            self._commit()

    def _save_form(self) -> None:
        self.status = None
        preset_id = self.query_one("#preset", Select).value
        preset = next((item for item in PRESETS if item.id == preset_id), None)
        name = self.query_one("#name", Input).value
        api_base = self.query_one("#api-base", Input).value
        style = self.query_one("#api-style", Select).value
        env_var = self.query_one("#env-var", Input).value
        reasoning = self.query_one("#reasoning-field", Input).value
        if not isinstance(preset_id, str) or not isinstance(style, str):
            self.error = "Choose a preset and API style."
            self._show("form")
            return
        previous = self.provider
        provider = ProviderDraft(
            preset_id,
            (
                previous.provider_id
                if previous is not None
                and previous.provider_id in self.catalog.providers
                else provider_id_for_name(name, self.catalog.providers)
            ),
            name,
            api_base,
            cast(ApiStyle, style),
            env_var,
            previous.key if previous else None,
            backend=preset.backend if preset else "generic",
            reasoning_field_name=reasoning,
        )
        identity_changed = previous and (
            provider.api_base != previous.api_base
            or provider.api_key_env_var != previous.api_key_env_var
            or provider.preset != previous.preset
        )
        if identity_changed:
            provider = replace(provider, key=None)
            self._credential_value = None
            self._credential_collision_warning = None
        self.error = validate_provider_draft(provider)
        self._invalid_inputs = self._form_invalid_inputs(provider)
        if self.error:
            self.provider = provider
            self._show("form")
            return
        provider = replace(
            provider,
            name=provider.name.strip(),
            api_base=provider.api_base.strip(),
            api_key_env_var=provider.api_key_env_var.strip(),
            reasoning_field_name=provider.reasoning_field_name.strip()
            or "reasoning_content",
        )
        if previous and (
            provider.api_base != previous.api_base
            or provider.api_key_env_var != previous.api_key_env_var
            or provider.preset != previous.preset
        ):
            provider = replace(provider, key=None)
            self._credential_value = None
            self._credential_collision_warning = None
        if not provider.api_key_env_var:
            provider = replace(provider, key=None)
            self._credential_value = None
        self._invalid_inputs.clear()
        self.provider = provider
        if self._management_action == "edit":
            if self.provider.api_key_env_var and self.provider.key is None:
                self._show("credential")
            else:
                self._save_connection_details()
            return
        self._show("credential")

    def _save_connection_details(self) -> None:
        """Persist a validated existing-provider connection edit without discovery."""
        assert self.provider is not None
        current = self.catalog.providers[self.provider.provider_id]
        values = {
            "api_base": self.provider.api_base,
            "api_style": self.provider.api_style,
            "api_key_env_var": self.provider.api_key_env_var,
            "backend": self.provider.backend,
            "reasoning_field_name": self.provider.reasoning_field_name,
        }
        patch = {
            name: value
            for name, value in values.items()
            if getattr(current, name) != value
        }
        if not patch:
            self.status = "No changes."
            self._show("form")
            return
        try:
            result = self.catalog_writer.apply_changes(
                CatalogChanges(self.provider.provider_id, patch)
            )
        except CatalogLoadError as failure:
            self.error = str(failure)
            self._show("form")
            return
        if isinstance(result, CatalogValidationError):
            self.error = result.message
            self._show("form")
            return
        self.snapshot = result.snapshot
        self._tag_orders = dict(self.catalog.tags)
        self._changed = self._changed or result.changed
        self.dismiss(
            ProviderFlowResult(
                "completed", changed=self._changed, warning=self._warning
            )
        )

    def _save_credential(self) -> None:  # noqa: PLR0911
        assert self.provider is not None
        if not self.provider.api_key_env_var:
            self.provider = replace(self.provider, key=None)
            self._credential_value = None
            if self._management_action == "credential":
                self.error = "Configure an API key environment variable before replacing a credential."
                self._show("credential")
                return
            self._show("probe")
            self._run_probe()
            return
        key = self.query_one("#key", Input).value
        if not key:
            self.error = (
                "An API key is required, or return and select no authentication."
            )
            self._show("credential")
            return
        self._credential_value = key
        shared_by = [
            provider_id
            for provider_id, definition in self.catalog.providers.items()
            if (
                provider_id != self.provider.provider_id
                and definition.api_key_env_var == self.provider.api_key_env_var
            )
        ]
        if (
            shared_by
            and self._credential_collision_warning != self.provider.api_key_env_var
        ):
            self._credential_collision_warning = self.provider.api_key_env_var
            self.error = (
                f"{self.provider.api_key_env_var} is also used by "
                f"{', '.join(sorted(shared_by))}. Continue again to replace the shared key."
            )
            self._show("credential")
            return
        result = self.credentials.save_key(self.provider.api_key_env_var, key)
        if result.status == "invalid_env_var":
            self.error = result.message or "Credential environment variable is invalid."
            self._show("credential")
            return
        self.provider = replace(self.provider, key=key)
        self._changed = True
        self._warning = (
            result.message or "Key is available for this session only."
            if result.status == "session_only"
            else self._warning
        )
        self.error = self._warning if result.status == "session_only" else None
        if self._management_action == "edit":
            self._save_connection_details()
            return
        if self._management_action == "credential":
            self.dismiss(
                ProviderFlowResult(
                    "completed", changed=self._changed, warning=self._warning
                )
            )
            return
        self._show("probe")
        self._run_probe()

    def _run_probe(self) -> None:
        assert self.provider is not None
        self._probe_generation += 1
        context = (self._probe_generation, self.provider)
        self._probe_context = context
        self.error = "Discovering models…"
        self._show("probe")
        self._probe_worker = self.run_worker(
            self.discovery(context[1], context[1].key, self.tls),
            exclusive=True,
            name="provider-discovery",
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if self._dismissed or event.worker is not self._probe_worker:
            return
        if event.state is not WorkerState.SUCCESS or self.step != "probe":
            return
        if self._probe_context is None or self._probe_context != (
            self._probe_generation,
            self.provider,
        ):
            return
        result = event.worker.result
        if isinstance(result, DiscoveryError):
            self.error = result.message
            self._show("probe")
        elif isinstance(result, DiscoveryResult):
            self.error = None
            self.discovered = result.models
            self._show("models")
        else:
            self.error = "Model discovery did not return a result."
            self._show("probe")

    def on_select_changed(self, event: Select.Changed) -> None:  # noqa: PLR0911
        """Apply editable form defaults immediately when a preset is selected."""
        if event.select.id == "overview-provider" and self.step == "overview":
            self._overview_provider_id = (
                event.value if isinstance(event.value, str) else None
            )
            return
        if event.select.id == "detail-model" and self.step == "review":
            current = self._review_selection()
            wire = event.value if isinstance(event.value, str) else None
            if current is None or wire == current.wire_name:
                return
            self._capture_review_inputs(current.wire_name)
            if not self._save_details(wire=current.wire_name, recompose=False):
                return
            self._detail_wire = wire
            self._show("review")
            return
        if self.step == "form" and event.select.id == "api-style":
            self._refresh_form_previews()
            return
        if event.select.id != "preset" or self.step != "form":
            return
        if self.provider is not None and event.value == self.provider.preset:
            return
        preset = next((item for item in PRESETS if item.id == event.value), None)
        if preset is None:
            return
        name = self.query_one("#name", Input)
        api_base = self.query_one("#api-base", Input)
        api_style = self.query_one("#api-style", Select)
        env_var = self.query_one("#env-var", Input)
        reasoning = self.query_one("#reasoning-field", Input)
        name.value = preset.name
        api_base.value = preset.api_base or ""
        if preset.api_style is not None:
            api_style.value = preset.api_style
        env_var.value = preset.api_key_env_var or self._suggest_env_var(preset.name)
        reasoning.value = preset.reasoning_field_name
        self._credential_value = None
        self._credential_collision_warning = None
        self.provider = (
            replace(
                self.provider,
                preset=preset.id,
                name=preset.name,
                api_base=preset.api_base or "",
                api_style=preset.api_style or "openai",
                api_key_env_var=env_var.value,
                key=None,
                backend=preset.backend,
                reasoning_field_name=preset.reasoning_field_name,
            )
            if self.provider
            else None
        )
        self._refresh_form_previews()

    @staticmethod
    def _suggest_env_var(name: str) -> str:
        """Return a non-binding credential variable suggestion for generic presets."""
        stem = "_".join(
            "".join(char if char.isalnum() else " " for char in name).split()
        )
        return f"{stem.upper()}_API_KEY" if stem else "API_KEY"

    def _refresh_form_previews(self) -> None:
        if self.step != "form" or not self.query("#urls"):
            return
        name = self.query_one("#name", Input).value.strip()
        current = self.provider
        provider = ProviderDraft(
            current.preset if current else None,
            (
                current.provider_id
                if current and current.provider_id in self.catalog.providers
                else provider_id_for_name(name, self.catalog.providers)
            ),
            name,
            self.query_one("#api-base", Input).value.strip(),
            cast(ApiStyle, self.query_one("#api-style", Select).value),
            self.query_one("#env-var", Input).value.strip(),
            None,
            backend=current.backend if current else "generic",
        )
        listing, inference = provider_urls(provider)
        self.query_one("#urls", Static).update(
            f"Provider ID: {provider.provider_id}\n"
            f"Listing URL: {listing}\nInference URL: {inference}"
        )

    def _form_invalid_inputs(self, provider: ProviderDraft) -> set[str]:
        """Return the field responsible for the current provider-form error."""
        if not provider.name.strip():
            return {"name"}
        if not provider.api_base.startswith(("http://", "https://")):
            return {"api-base"}
        if provider.backend == "mistral" and not provider.api_base.rstrip("/").endswith(
            "/v1"
        ):
            return {"api-base"}
        if (
            provider.backend == "mistral"
            and provider.reasoning_field_name != "reasoning_content"
        ):
            return {"reasoning-field"}
        if (
            provider.api_key_env_var
            and not provider.api_key_env_var.replace("_", "a").isalnum()
        ):
            return {"env-var"}
        return set()

    def _apply_invalid_input_state(self) -> None:
        for input_id in self._invalid_inputs:
            if self.query(f"#{input_id}"):
                self.query_one(f"#{input_id}", Input).add_class("-invalid")
        if self._invalid_inputs:
            input_id = next(iter(self._invalid_inputs))
            if self.query(f"#{input_id}"):
                self.query_one(f"#{input_id}", Input).focus()

    def _revalidate_form_after_error(self) -> None:
        """Give recovery feedback only after a submit or blur has shown an error."""
        assert self.provider is not None
        provider = replace(
            self.provider,
            name=self.query_one("#name", Input).value,
            api_base=self.query_one("#api-base", Input).value,
            api_key_env_var=self.query_one("#env-var", Input).value,
            reasoning_field_name=self.query_one("#reasoning-field", Input).value,
        )
        self.error = validate_provider_draft(provider)
        self._invalid_inputs = self._form_invalid_inputs(provider)
        for field in ("name", "api-base", "env-var", "reasoning-field"):
            input_widget = self.query_one(f"#{field}", Input)
            input_widget.set_class(field in self._invalid_inputs, "-invalid")
        for error_widget in self.query(Static):
            if not error_widget.has_class("error"):
                continue
            error_widget.update(self.error or "")
            error_widget.display = self.error is not None

    def _highlighted_model_label(self, item: DiscoveryItem, query: str) -> Text:
        label = self._model_label(item)
        text = Text(label, no_wrap=True)
        if not query:
            return text
        needle = (
            query
            if any(character.isupper() for character in query)
            else query.casefold()
        )
        haystack = label if query == needle else label.casefold()
        start = 0
        while (index := haystack.find(needle, start)) >= 0:
            text.stylize("reverse", index, index + len(query))
            start = index + len(query)
        return text

    def on_input_changed(self, event: Input.Changed) -> None:
        if self.step == "form" and event.input.id in {"name", "api-base"}:
            self._refresh_form_previews()
        if self.step == "form" and self._invalid_inputs:
            self._revalidate_form_after_error()
        if event.input.id != "search" or self.step != "models":
            return
        query = event.value
        needle = (
            query
            if any(character.isupper() for character in query)
            else query.casefold()
        )
        selection = self.query_one("#models", SelectionList)
        visible = {
            option.value for option in cast(list[Selection[str]], selection.options)
        }
        self._selected_wires.difference_update(visible - set(selection.selected))
        self._selected_wires.update(selection.selected)
        picker_items = self._picker_items()
        found = [
            item
            for item in picker_items
            if needle in (item.wire_id if query == needle else item.wire_id.casefold())
            or (
                item.display_label
                and needle
                in (
                    item.display_label
                    if query == needle
                    else item.display_label.casefold()
                )
            )
        ]
        selection.clear_options()
        selection.add_options(
            (
                self._highlighted_model_label(item, query),
                item.wire_id,
                item.wire_id in self._selected_wires
                or item.wire_id in self._selected_models
                or self._is_configured_model(item),
            )
            for item in found
        )
        self.query_one("#model-count", Static).update(
            f"{len(found)} of {len(picker_items)} chat models"
        )

    def _save_model_selection(self) -> None:
        selection_list = self.query_one("#models", SelectionList)
        visible = {
            option.value
            for option in cast(list[Selection[str]], selection_list.options)
        }
        self._selected_wires.difference_update(visible - set(selection_list.selected))
        self._selected_wires.update(selection_list.selected)
        selected = set(self._selected_wires)
        if "" in selected:
            selected.remove("")
        if any(not item.wire_id for item in self.discovered):
            manual_wire = self.query_one("#manual-wire-name", Input).value.strip()
            if manual_wire:
                selected.add(manual_wire)
            elif not selected:
                self.error = "Enter a wire name for the manual model."
                self._show("models")
                return
        retained = {
            wire: selection
            for wire, selection in self._selected_models.items()
            if wire in selected
        }
        for wire in selected - retained.keys():
            outcome = match_discovered_model(
                self.catalog,
                self.provider_id,
                wire,
                provider=self._provider_definition(),
            )
            base_name = (
                outcome.existing_base
                if outcome.kind in {"existing", "alias_collision"}
                else wire
            )
            assert base_name is not None
            retained[wire] = ModelSelectionDraft(wire, base_name)
        self._selected_models = retained
        self._selected_wires = set(retained)
        self._detail_wire = next(iter(self._selected_models), None)
        if not self._selected_models:
            self.error = "Select at least one model, or go back and cancel."
            self._show("models")
            return
        self.error = None
        self._show("review")

    def _save_details(self, *, wire: str | None = None, recompose: bool = True) -> bool:
        selected_wire = self.query_one("#detail-model", Select).value
        wire = wire or (selected_wire if isinstance(selected_wire, str) else None)
        if wire is None:
            wire = self._detail_wire
        self._capture_review_inputs(wire)
        if not isinstance(wire, str) or wire not in self._selected_models:
            self.error = "Choose a selected model to edit."
            self._show("review")
            return False
        selection = self._selected_models[wire]
        try:
            values = [
                self._price("#input-price"),
                self._price("#output-price"),
                self._price("#cached-price"),
            ]
        except ValueError:
            self.error = "Prices must be non-negative numbers; blank means unknown."
            self._show("review")
            return False
        aliases = self._csv("#aliases")
        tags = self._csv("#tags")
        edits = ModelEdits(
            *(
                self._optional_price(value, previous)
                for value, previous in zip(
                    values,
                    (
                        selection.edits.input_price,
                        selection.edits.output_price,
                        selection.edits.cached_input_price,
                    ),
                    strict=True,
                )
            ),
            aliases=self._optional_csv(aliases, selection.edits.aliases),
            tags=self._optional_csv(tags, selection.edits.tags),
        )
        base = selection.base_name
        outcome = match_discovered_model(
            self.catalog, self.provider_id, wire, provider=self._provider_definition()
        )
        if outcome.kind == "occupied_slot":
            base = cast(str, outcome.existing_base)
        elif outcome.kind == "multiple_matches":
            choice = self.query_one("#collision-choice", Select).value
            if isinstance(choice, str):
                base = choice
        self._selected_models[wire] = ModelSelectionDraft(wire, base, edits)
        self.error = None
        if recompose:
            self._show("review")
        return True

    @staticmethod
    def _optional_price(
        value: float | None, previous: OptionalEdit[float]
    ) -> OptionalEdit[float]:
        return OptionalEdit.set(value) if value is not None else previous

    @staticmethod
    def _optional_csv(
        value: tuple[str, ...], previous: OptionalEdit[tuple[str, ...]]
    ) -> OptionalEdit[tuple[str, ...]]:
        return OptionalEdit.set(value) if value else previous

    def _clear_prices(self) -> None:
        """Explicitly clear all selected deployment prices back to unknown."""
        selection = self._review_selection()
        if selection is None:
            return
        self._capture_review_inputs(selection.wire_name)
        self._selected_models[selection.wire_name] = replace(
            selection,
            edits=replace(
                selection.edits,
                input_price=OptionalEdit.cleared(),
                output_price=OptionalEdit.cleared(),
                cached_input_price=OptionalEdit.cleared(),
            ),
        )
        self._clear_review_input(
            selection, "input-price", "output-price", "cached-price"
        )
        self._show("review")

    def _clear_detail_metadata(self, field: Literal["aliases", "tags"]) -> None:
        selection = self._review_selection()
        if selection is None:
            return
        self._capture_review_inputs(selection.wire_name)
        self._selected_models[selection.wire_name] = replace(
            selection, edits=replace(selection.edits, **{field: OptionalEdit.cleared()})
        )
        self._clear_review_input(selection, field)
        self._show("review")

    def _price(self, selector: str) -> float | None:
        value = self.query_one(selector, Input).value.strip()
        if not value:
            return None
        price = float(value)
        if price < 0:
            raise ValueError
        return price

    def _assign_bulk_tags(self) -> None:
        """Apply the current ordered tags to every selected model, not prices or aliases."""
        current = self._review_selection()
        if current is not None:
            self._capture_review_inputs(current.wire_name)
        tags = self._csv("#tags")
        if not tags:
            self.error = "Enter at least one tag before bulk assignment."
            self._show("review")
            return
        for wire, selection in self._selected_models.items():
            self._selected_models[wire] = replace(
                selection, edits=replace(selection.edits, tags=OptionalEdit.set(tags))
            )
            # Programmatic edits supersede captured tag text without discarding
            # unrelated raw fields for the model currently being edited.
            self._review_inputs.setdefault(wire, {})["tags"] = ", ".join(tags)
        self.error = None
        self._show("review")

    def _csv(self, selector: str) -> tuple[str, ...]:
        return tuple(
            item.strip()
            for item in self.query_one(selector, Input).value.split(",")
            if item.strip()
        )

    def _commit(self) -> None:
        assert self.provider is not None
        if not self._save_details(recompose=False):
            return
        patches: dict[str, dict[str, object]] = {}
        tags: dict[str, tuple[str, ...]] = dict(self._tag_orders)
        for selection in self._selected_models.values():
            outcome: MatchOutcome = match_discovered_model(
                self.catalog,
                self.provider.provider_id,
                selection.wire_name,
                provider=self._provider_definition(),
            )
            if outcome.kind == "existing" and selection.edits == ModelEdits():
                continue
            base_name = (
                outcome.existing_base
                if outcome.kind in {"existing", "alias_collision"}
                else selection.base_name
            )
            assert base_name is not None
            patch = patches.setdefault(base_name, {"deployments": []})
            if outcome.kind == "new_model":
                assert outcome.proposed is not None
                patch["thinking"] = outcome.proposed.definition.thinking
            deployments = patch["deployments"]
            assert isinstance(deployments, list)
            deployment: dict[str, object] = {
                "provider": self.provider.provider_id,
                "name": selection.wire_name,
            }
            price_edits = (
                ("input", selection.edits.input_price),
                ("output", selection.edits.output_price),
                ("cached_input", selection.edits.cached_input_price),
            )
            if any(edit.state != "untouched" for _, edit in price_edits):
                inherited = (
                    outcome.matches[0].deployment.prices
                    if outcome.kind == "existing"
                    else None
                )
                prices = {
                    name: (
                        edit.value
                        if edit.state == "set"
                        else None
                        if edit.state == "cleared"
                        else getattr(inherited, name)
                        if inherited is not None
                        else None
                    )
                    for name, edit in price_edits
                }
                deployment["prices"] = {
                    name: value for name, value in prices.items() if value is not None
                }
            deployments.append(deployment)
            if selection.edits.aliases.state == "set":
                patch["aliases"] = list(selection.edits.aliases.value or ())
            if selection.edits.aliases.state == "cleared":
                patch["aliases"] = []
            if selection.edits.tags.state == "set":
                for tag in selection.edits.tags.value or ():
                    members = list(tags.get(tag, self.catalog.tags.get(tag, ())))
                    if base_name not in members:
                        members.append(base_name)
                    tags[tag] = tuple(members)
            if selection.edits.tags.state == "cleared":
                for tag in self.catalog.tags.keys() | tags.keys():
                    members = tags.get(tag, self.catalog.tags.get(tag, ()))
                    if base_name in members:
                        remaining = tuple(
                            member for member in members if member != base_name
                        )
                        if not remaining:
                            self.error = (
                                "Removing the final member of a tag is not supported."
                            )
                            self._show("review")
                            return
                        tags[tag] = remaining
        if any(not members for members in tags.values()):
            self.error = "Removing the final member of a tag is not supported."
            self._show("review")
            return
        current_provider = self.catalog.providers.get(self.provider.provider_id)
        provider_values = {
            "api_base": self.provider.api_base,
            "api_style": self.provider.api_style,
            "api_key_env_var": self.provider.api_key_env_var,
            "backend": self.provider.backend,
            "reasoning_field_name": self.provider.reasoning_field_name,
        }
        provider_patch = (
            provider_values
            if current_provider is None
            else {
                name: value
                for name, value in provider_values.items()
                if getattr(current_provider, name) != value
            }
        )
        self._apply_changes(
            provider_patch, patches, tags if tags != dict(self.catalog.tags) else {}
        )

    def _apply_changes(
        self,
        provider_patch: Mapping[str, object],
        patches: Mapping[str, Mapping[str, object]],
        tags: Mapping[str, tuple[str, ...]],
    ) -> None:
        """Persist a non-empty catalog change and advance on success."""
        if not provider_patch and not patches and not tags:
            self._show("again")
            return
        try:
            result = self.catalog_writer.apply_changes(
                CatalogChanges(self.provider_id, provider_patch, patches, tags or None)
            )
        except CatalogLoadError as failure:
            self.error = str(failure)
            self._show("review")
            return
        if isinstance(result, CatalogValidationError):
            self.error = result.message
            self._show("review")
            return
        self.snapshot = result.snapshot
        self._tag_orders = dict(self.catalog.tags)
        self._changed = self._changed or result.changed
        self._show("again")

    def _unavailable_model_labels(self) -> list[str]:
        """Render disabled or provider-incompatible entries without making them values."""
        unavailable: list[str] = []
        for base, definition in sorted(self.catalog.models.items()):
            if definition.disabled:
                unavailable.append(f"{base} (model disabled)")
                continue
            for deployment in definition.deployments:
                provider = self.catalog.providers[deployment.provider]
                if deployment.disabled or provider.disabled:
                    unavailable.append(
                        f"{deployment.provider}/{deployment.name} ({base}; unavailable)"
                    )
        return unavailable

    def _has_usable_deployment(self, base: str) -> bool:
        definition = self.catalog.models[base]
        return not definition.disabled and any(
            not deployment.disabled
            and not self.catalog.providers[deployment.provider].disabled
            for deployment in definition.deployments
        )

    def _active_model_options(self) -> list[tuple[str, str]]:
        """Return one selectable canonical row per model, plus usable tag expressions."""
        options: list[tuple[str, str]] = []
        for base, definition in sorted(self.catalog.models.items()):
            if definition.disabled or not self._has_usable_deployment(base):
                continue
            deployment = next(
                item
                for item in definition.deployments
                if not item.disabled
                and not self.catalog.providers[item.provider].disabled
            )
            provider_status = (
                "Configured"
                if self._is_configured_provider(deployment.provider)
                else "Available"
            )
            aliases = (
                f"aliases: {', '.join(definition.aliases)}; "
                if definition.aliases
                else ""
            )
            options.append((
                f"{deployment.provider}/{deployment.name} ({base})\t"
                f"{aliases}{provider_status}",
                base,
            ))
        options.extend(
            (f"@{tag}\ttag expression", f"@{tag}")
            for tag, members in sorted(self.catalog.tags.items())
            if any(self._has_usable_deployment(member) for member in members)
        )
        return options

    def _active_model_picker_options(self) -> list[tuple[str, str]]:
        """Include a visible Default row when no configured model is selectable."""
        options = self._active_model_options()
        if self._active_model_expression not in {value for _label, value in options}:
            current = self._active_model_expression or "catalog default"
            return [
                (f"Default\t(currently {current})", DEFAULT_ACTIVE_MODEL_OPTION),
                *options,
            ]
        return options

    def _validate_selection(self, expression: str | None, *, step: Step) -> bool:
        if self.validate_selection is None:
            return True
        try:
            error = self.validate_selection(expression)
        except Exception as failure:
            self.error = f"Could not validate the active model: {failure}"
            self._show(step)
            return False
        if error:
            self.error = error
            self._show(step)
            return False
        self.error = None
        return True

    def _finish(self, expression: str | None = None) -> None:
        if expression is None:
            expression = self._active_model_expression
        valid_expressions = {value for _label, value in self._active_model_options()}
        if expression != "" and (
            not isinstance(expression, str) or expression not in valid_expressions
        ):
            self.error = "Choose a usable canonical name, alias, or @tag."
            self._show("picker")
            return
        if not self._validate_selection(expression, step="picker"):
            return
        self.run_worker(
            self._persist_active(expression),
            exclusive=True,
            name="provider-active-model",
        )

    async def _persist_active(self, expression: str) -> None:
        reload_result = await self.config.reload_catalog_and_config()
        if reload_result.snapshot is None:
            self.error = reload_result.message or "Could not reload the catalog."
            self._show("picker")
            return
        result = await self.config.persist_active_model(expression)
        if not result.persisted:
            self.error = result.message or "Could not save the active model."
            self._show("picker")
            return
        self.dismiss(
            ProviderFlowResult(
                "completed", expression, self._changed, warning=self._warning
            )
        )

    def _apply_preset(self, preset_id: str) -> None:
        preset: ProviderPreset = next(item for item in PRESETS if item.id == preset_id)
        self._credential_value = None
        self._credential_collision_warning = None
        name = preset.name
        self.provider = ProviderDraft(
            preset.id,
            provider_id_for_name(name, self.catalog.providers),
            name,
            preset.api_base or "",
            preset.api_style or "openai",
            preset.api_key_env_var or "",
            None,
            backend=preset.backend,
            reasoning_field_name=preset.reasoning_field_name,
        )

    def _select_onboarding_mistral_provider(self) -> bool:
        """Select an existing Mistral backend for the onboarding shortcut."""
        provider_id = next(
            (
                candidate
                for candidate, definition in self.catalog.providers.items()
                if definition.backend == "mistral"
            ),
            None,
        )
        if provider_id is None:
            return False
        self._overview_provider_id = provider_id
        self._select_existing_provider()
        return True

    def _select_existing_provider(self) -> bool:
        provider_id = self._overview_provider_id
        if provider_id is None and len(self.catalog.providers) == 1:
            provider_id = next(iter(self.catalog.providers))
        if provider_id is None:
            self.error = "Select a provider first."
            self._show("overview")
            return False
        definition = self.catalog.providers[provider_id]
        previous_id = self.provider.provider_id if self.provider is not None else None
        if previous_id != provider_id:
            # A masked value belongs to the prior provider identity, never the
            # newly selected provider (including the onboarding Mistral shortcut).
            self._credential_value = None
            self._credential_collision_warning = None
        self.provider = ProviderDraft(
            (
                "mistral"
                if definition.backend == "mistral"
                else "generic-anthropic"
                if definition.api_style == "anthropic"
                else "generic-openai"
            ),
            provider_id,
            provider_id.rsplit("/", 1)[0],
            definition.api_base,
            definition.api_style,
            definition.api_key_env_var,
            self._configured_credential(definition.api_key_env_var),
            backend=definition.backend,
            reasoning_field_name=definition.reasoning_field_name,
            extra_headers=definition.extra_headers,
        )
        return True

    def _configured_credential(self, env_var: str) -> str | None:
        """Resolve an existing provider credential through the host boundary."""
        resolver = getattr(self.credentials, "resolve_key", None)
        value = resolver(env_var) if callable(resolver) and env_var else None
        return value if isinstance(value, str) else None

    def _reset_unsaved_model_metadata(self) -> None:
        """Discard draft model state before entering another management path."""
        self._selected_models = {}
        self._selected_wires = set()
        self._review_inputs = {}
        self._detail_wire = None
        self.discovered = ()
        self._credential_value = None
        self._credential_collision_warning = None
        self.error = None

    def _show(self, step: Step) -> None:
        if step != self.step:
            self._invalid_inputs.clear()
        if step == "form" and self.step == "again":
            self.provider = None
            self._selected_models = {}
            self._detail_wire = None
        self.step = step
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_current_picker)
        self.call_later(self._focus_current_picker)

    def action_picker_down(self) -> None:
        if self.step == "picker" and self.query("#active-model"):
            self.query_one("#active-model", OptionList).action_cursor_down()

    def action_picker_up(self) -> None:
        if self.step == "picker" and self.query("#active-model"):
            self.query_one("#active-model", OptionList).action_cursor_up()

    def action_back(self) -> None:
        if self.step == "models" and self.query("#search"):
            search = self.query_one("#search", Input)
            if search.value:
                search.value = ""
                return
        if self.step == "picker" and self.provider is None:
            self._show("overview" if self.management else "choice")
            return
        previous: dict[Step, Step] = {
            "choice": "overview" if self.management else "choice",
            "form": "choice",
            "credential": "form",
            "probe": "credential",
            "models": "probe",
            "review": "models",
            "again": "review",
            "picker": "again",
        }
        if self.step in {"overview", "choice"}:
            self.action_cancel()
        else:
            self._probe_generation += 1
            if self.step == "review":
                self._capture_review_inputs()
            if self._management_action == "edit" and self.step == "form":
                self._show("overview")
            elif self._management_action == "credential" and self.step == "credential":
                self._show("overview")
            elif self._management_action == "discover" and self.step == "probe":
                self._show("overview")
            else:
                self._show(previous[self.step])

    def action_cancel(self) -> None:
        self._dismissed = True
        self._probe_generation += 1
        if self._probe_worker is not None and not self._probe_worker.is_finished:
            self._probe_worker.cancel()
        self.dismiss(
            ProviderFlowResult(
                "cancelled", changed=self._changed, warning=self._warning
            )
        )


__all__ = [
    "ProviderManagementScreen",
    "provider_id_for_name",
    "provider_urls",
    "validate_provider_draft",
]
