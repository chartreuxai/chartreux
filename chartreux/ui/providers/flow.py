"""Host-neutral, modal provider-management flow.

The screen deliberately owns only transient draft state.  Hosts provide the three
persistence boundaries from :mod:`chartreux.ui.providers.contracts`, so it is
safe to mount in onboarding and the main TUI alike.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import ClassVar, Literal, cast

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, SelectionList, Static
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


class ProviderManagementScreen(ModalScreen[ProviderFlowResult]):
    """A fresh, full-screen provider flow with injected service boundaries."""

    DEFAULT_CSS = """
    ProviderManagementScreen { align: center middle; }
    #provider-management { width: 90%; height: 100%; padding: 0 2; border: round $primary; }
    #provider-management .error { color: $error; }
    #provider-management .muted { color: $text-muted; }
    #provider-management Input { width: 1fr; height: 1; border: none; }
    #provider-management SelectionList { height: 1fr; }
    #provider-management #actions { height: auto; layout: grid; grid-size: 3; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", "Back", show=False, priority=True),
        Binding("ctrl+c", "cancel", "Cancel", show=False),
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
    ) -> None:
        super().__init__()
        self.discovery = discovery
        self.catalog_writer = catalog_writer
        self.credentials = credentials
        self.config = config
        self.snapshot = snapshot
        self.management = management
        self.tls = tls or TLSConfig()
        self.step: Step = "overview" if management else "choice"
        self.provider: ProviderDraft | None = None
        self._selected_models: dict[str, ModelSelectionDraft] = {}
        self._selected_wires: set[str] = set()
        self._detail_wire: str | None = None
        self._detail_tag: str | None = None
        self._tag_orders: dict[str, tuple[str, ...]] = dict(snapshot.catalog.tags)
        self._overview_provider_id: str | None = None
        self._management_action: Literal["edit", "discover", "credential"] | None = None
        self.discovered: tuple[DiscoveryItem, ...] = ()
        self.error: str | None = None
        self._changed = False
        self._probe_worker: Worker[DiscoveryResult | DiscoveryError] | None = None
        self._probe_generation = 0
        self._probe_context: tuple[int, ProviderDraft] | None = None
        self._credential_collision_warning: str | None = None
        self._credential_value: str | None = None
        self._warning: str | None = None
        self._dismissed = False

    @property
    def catalog(self) -> ModelCatalog:
        return self.snapshot.catalog

    def compose(self) -> ComposeResult:
        with Vertical(id="provider-management"):
            yield Label("Provider management", id="flow-title")
            yield from self._step_widgets()

    def _step_widgets(self) -> ComposeResult:  # noqa: PLR0912, PLR0915
        if self.step == "overview":
            yield Label("Configured providers")
            yield Select(
                [
                    (provider_id, provider_id)
                    for provider_id in sorted(self.catalog.providers)
                ],
                value=self._overview_provider_id or Select.NULL,
                prompt="Select a provider to manage",
                id="overview-provider",
            )
            if not self.catalog.providers:
                yield Static("No providers yet.")
            yield from self._buttons(
                ("add", "Add provider"),
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
                ("cancel", "Cancel"),
            )
        elif self.step == "form":
            current = self.provider
            preset_options = [(preset.name, preset.id) for preset in PRESETS]
            yield Select(
                preset_options,
                value=current.preset
                if current and current.preset
                else "generic-openai",
                id="preset",
                prompt="Preset",
            )
            yield Input(
                current.name if current else "", placeholder="Provider name", id="name"
            )
            yield Input(
                current.api_base if current else "",
                placeholder="API base",
                id="api-base",
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
            yield Input(
                current.api_key_env_var if current else "",
                placeholder="API key environment variable (blank = no authentication)",
                id="env-var",
            )
            yield Input(
                current.reasoning_field_name if current else "reasoning_content",
                placeholder="Reasoning field",
                id="reasoning-field",
            )
            listing, inference = provider_urls(current) if current else ("", "")
            yield Static(
                f"Provider ID: {current.provider_id if current else '<name>/default'}\n"
                f"Listing URL: {listing}\nInference URL: {inference}",
                classes="muted",
                id="urls",
            )
            if self.error:
                yield Static(self.error, classes="error")
            yield from self._buttons(
                ("continue", "Continue"), ("back", "Back"), ("cancel", "Cancel")
            )
        elif self.step == "credential":
            assert self.provider is not None
            yield Label("Credential")
            yield Static(
                "No authentication is enabled."
                if not self.provider.api_key_env_var
                else (
                    f"Save key as {self.provider.api_key_env_var}. Keys are masked."
                    " If disk persistence is unavailable, the key remains available"
                    " only for this session."
                )
            )
            if self.provider.api_key_env_var:
                yield Input(
                    self._credential_value or "",
                    password=True,
                    placeholder="API key",
                    id="key",
                )
            if self.error:
                yield Static(self.error, classes="error")
            yield from self._buttons(
                ("continue", "Continue"), ("back", "Back"), ("cancel", "Cancel")
            )
        elif self.step == "probe":
            yield Label("Discover models")
            yield Static(
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
            yield Input(placeholder="Search wire IDs (case-insensitive)", id="search")
            if any(not item.wire_id for item in self.discovered):
                yield Input(
                    placeholder="Enter the provider wire name", id="manual-wire-name"
                )
            yield Static(
                f"{len(self.discovered)} models — type to filter", id="model-count"
            )
            yield SelectionList(
                *[
                    (
                        self._model_label(item),
                        item.wire_id,
                        item.wire_id in self._selected_wires
                        or item.wire_id in self._selected_models,
                    )
                    for item in sorted(self.discovered, key=lambda value: value.wire_id)
                ],
                id="models",
            )
            if self.error:
                yield Static(self.error, classes="error")
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
            yield Input(
                selection.base_name if selection else "",
                placeholder="Base name",
                id="base-name",
            )
            yield Input(
                self._edit_value(selection, "input_price"),
                placeholder="Input price per million tokens (blank = unknown)",
                id="input-price",
            )
            yield Input(
                self._edit_value(selection, "output_price"),
                placeholder="Output price per million tokens (blank = unknown)",
                id="output-price",
            )
            yield Input(
                self._edit_value(selection, "cached_input_price"),
                placeholder="Cached input price per million tokens (blank = unknown)",
                id="cached-price",
            )
            yield Input(
                self._edit_value(selection, "aliases"),
                placeholder="Aliases, comma separated",
                id="aliases",
            )
            yield Input(
                self._edit_value(selection, "tags"),
                placeholder="Tags, comma separated",
                id="tags",
            )
            tag_options = self._ordered_tag_options(selection)
            yield Select(
                [(tag, tag) for tag in tag_options],
                value=(
                    self._detail_tag
                    if self._detail_tag in tag_options
                    else (tag_options[0] if tag_options else Select.NULL)
                ),
                prompt="Tag whose members to reorder",
                id="ordered-tag",
            )
            if self.error:
                yield Static(self.error, classes="error")
            yield from self._buttons(
                ("save-details", "Save details"),
                ("clear-prices", "Clear prices"),
                ("clear-aliases", "Clear aliases"),
                ("clear-tags", "Clear tags"),
                ("tag-up", "Move tag up"),
                ("tag-down", "Move tag down"),
                ("bulk-tags", "Assign tags to all selected"),
                ("continue", "Save / Continue"),
                ("back", "Back"),
                ("cancel", "Cancel"),
            )
        elif self.step == "again":
            yield Label("Provider saved.")
            yield from self._buttons(
                ("add", "Add another provider"),
                ("picker", "Choose active model"),
                ("finish-unchanged", "Finish without changing active model"),
            )
        else:
            options = self._active_model_options()
            unavailable = self._unavailable_model_labels()
            yield Label("Choose active model")
            if self.error:
                yield Static(self.error, classes="error")
            yield Select(options, id="active-model", prompt="Active model")
            if unavailable:
                yield Static(
                    "Unavailable (visible but not selectable):\n"
                    + "\n".join(unavailable),
                    classes="muted",
                )
            yield Static(
                "Labels identify deployments; saved values are canonical names, aliases, or @tags.",
                classes="muted",
            )
            yield from self._buttons(("finish", "Finish"), ("cancel", "Cancel"))

    def _review_selection(self) -> ModelSelectionDraft | None:
        """Return the selected detail row, defaulting to insertion order."""
        if self._detail_wire is not None:
            return self._selected_models.get(self._detail_wire)
        return next(iter(self._selected_models.values()), None)

    def _ordered_tag_options(
        self, selection: ModelSelectionDraft | None
    ) -> tuple[str, ...]:
        """Return tags that can scope ordering for the current detail model."""
        if selection is None:
            return ()
        selected_tags = selection.edits.tags.value or ()
        return tuple(dict.fromkeys((*selected_tags, *self._tag_orders)))

    def _edit_value(self, selection: ModelSelectionDraft | None, field: str) -> str:
        if selection is None:
            return ""
        value = getattr(selection.edits, field).value
        if value is None:
            return ""
        return ", ".join(value) if isinstance(value, tuple) else str(value)

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
                [
                    (f"Add as deployment to {outcome.existing_base}", "existing"),
                    ("Use a new unambiguous base name", "new"),
                ],
                value="new",
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
            yield Label(
                f"{selection.wire_name} is an alias of {outcome.existing_base}; enter an unambiguous base name."
            )

    def _buttons(self, *buttons: tuple[str, str]) -> ComposeResult:
        with Horizontal(id="actions"):
            for button_id, label in buttons:
                yield Button(label, id=button_id)

    def _model_label(self, item: DiscoveryItem) -> str:
        outcome = match_discovered_model(
            self.catalog,
            self.provider_id,
            item.wire_id,
            provider=self._provider_definition(),
        )
        if outcome.kind == "existing":
            return f"{item.wire_id} — Already configured as {outcome.existing_base}"
        if outcome.kind == "alias_collision":
            return f"{item.wire_id} — alias of {outcome.existing_base}; choose an unambiguous base name"
        if outcome.kind == "occupied_slot":
            return f"{item.wire_id} — occupied slot; add to existing base or use an unambiguous name"
        if outcome.kind == "multiple_matches":
            return f"{item.wire_id} — multiple matches: {', '.join(match.base_name for match in outcome.matches)}"
        return f"{item.wire_id} — {outcome.proposed.base_name if outcome.proposed else item.wire_id}"

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
            self._apply_preset("mistral")
            self._show("form")
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
                self.error = "Select a configured provider first."
                self._show("overview")
                return
            self._select_existing_provider()
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
        if button == "save-details":
            self._save_details()
            return
        if button == "clear-prices":
            self._clear_prices()
            return
        if button in {"clear-aliases", "clear-tags"}:
            self._clear_detail_metadata(
                "aliases" if button == "clear-aliases" else "tags"
            )
            return
        if button in {"tag-up", "tag-down"}:
            self._move_tag(-1 if button == "tag-up" else 1)
            return
        if button == "bulk-tags":
            self._assign_bulk_tags()
            return
        if button == "picker":
            self._show("picker")
            return
        if button == "finish-unchanged":
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
        preset_id = self.query_one("#preset", Select).value
        preset = next((item for item in PRESETS if item.id == preset_id), None)
        name = self.query_one("#name", Input).value.strip()
        api_base = self.query_one("#api-base", Input).value.strip()
        style = self.query_one("#api-style", Select).value
        env_var = self.query_one("#env-var", Input).value.strip()
        reasoning = (
            self.query_one("#reasoning-field", Input).value.strip()
            or "reasoning_content"
        )
        if not isinstance(preset_id, str) or not isinstance(style, str):
            self.error = "Choose a preset and API style."
            self._show("form")
            return
        provider = ProviderDraft(
            preset_id,
            (
                self.provider.provider_id
                if self.provider is not None
                and self.provider.provider_id in self.catalog.providers
                else provider_id_for_name(name, self.catalog.providers)
            ),
            name,
            api_base,
            cast(ApiStyle, style),
            env_var,
            self.provider.key if self.provider else None,
            backend=preset.backend if preset else "generic",
            reasoning_field_name=reasoning,
        )
        self.error = validate_provider_draft(provider)
        if self.error:
            self._show("form")
            return
        self.provider = provider
        if self._management_action == "edit":
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
        self._changed = self._changed or result.changed
        self.dismiss(
            ProviderFlowResult(
                "completed", changed=self._changed, warning=self._warning
            )
        )

    def _save_credential(self) -> None:
        assert self.provider is not None
        if not self.provider.api_key_env_var:
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
            wire = event.value if isinstance(event.value, str) else None
            current = self._review_selection()
            self._detail_wire = wire
            if current is None or wire == current.wire_name:
                return
            self._show("review")
            return
        if event.select.id == "ordered-tag" and self.step == "review":
            self._detail_tag = event.value if isinstance(event.value, str) else None
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
        self.provider = (
            replace(
                self.provider,
                preset=preset.id,
                name=preset.name,
                api_base=preset.api_base or "",
                api_style=preset.api_style or "openai",
                api_key_env_var=env_var.value,
                backend=preset.backend,
                reasoning_field_name=preset.reasoning_field_name,
            )
            if self.provider
            else None
        )

    @staticmethod
    def _suggest_env_var(name: str) -> str:
        """Return a non-binding credential variable suggestion for generic presets."""
        stem = "_".join(
            "".join(char if char.isalnum() else " " for char in name).split()
        )
        return f"{stem.upper()}_API_KEY" if stem else "API_KEY"

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search" or self.step != "models":
            return
        query = event.value.casefold()
        selection = self.query_one("#models", SelectionList)
        visible = {
            option.value for option in cast(list[Selection[str]], selection.options)
        }
        self._selected_wires.difference_update(visible - set(selection.selected))
        self._selected_wires.update(selection.selected)
        found = [item for item in self.discovered if query in item.wire_id.casefold()]
        selection.clear_options()
        selection.add_options(
            (
                self._model_label(item),
                item.wire_id,
                item.wire_id in self._selected_wires
                or item.wire_id in self._selected_models,
            )
            for item in sorted(found, key=lambda value: value.wire_id)
        )
        self.query_one("#model-count", Static).update(
            f"{len(found)} models — type to filter"
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
            base_name = outcome.existing_base if outcome.kind == "existing" else wire
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

    def _save_details(self, *, recompose: bool = True) -> bool:
        wire = self.query_one("#detail-model", Select).value
        if not isinstance(wire, str):
            wire = self._detail_wire
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
        base = self.query_one("#base-name", Input).value.strip() or selection.base_name
        outcome = match_discovered_model(
            self.catalog, self.provider_id, wire, provider=self._provider_definition()
        )
        if outcome.kind == "occupied_slot":
            resolution = self.query_one("#collision-choice", Select).value
            if resolution == "existing":
                base = cast(str, outcome.existing_base)
            elif base == outcome.existing_base:
                self.error = (
                    "Enter an unambiguous new base name for this occupied slot."
                )
                self._show("review")
                return False
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
        self._selected_models[selection.wire_name] = replace(
            selection,
            edits=replace(
                selection.edits,
                input_price=OptionalEdit.cleared(),
                output_price=OptionalEdit.cleared(),
                cached_input_price=OptionalEdit.cleared(),
            ),
        )
        self._show("review")

    def _clear_detail_metadata(self, field: Literal["aliases", "tags"]) -> None:
        selection = self._review_selection()
        if selection is None:
            return
        self._selected_models[selection.wire_name] = replace(
            selection, edits=replace(selection.edits, **{field: OptionalEdit.cleared()})
        )
        self._show("review")

    def _price(self, selector: str) -> float | None:
        value = self.query_one(selector, Input).value.strip()
        if not value:
            return None
        price = float(value)
        if price < 0:
            raise ValueError
        return price

    def _move_tag(self, direction: int) -> None:
        """Move the detail model inside the explicitly selected tag's member order."""
        selection = self._review_selection()
        tag = self._detail_tag or self.query_one("#ordered-tag", Select).value
        if selection is None or not isinstance(tag, str):
            self.error = "Choose a tag whose members to reorder."
            self._show("review")
            return
        members = list(self._tag_orders.get(tag, self.catalog.tags.get(tag, ())))
        if selection.base_name not in members:
            members.append(selection.base_name)
        index = members.index(selection.base_name)
        other = index + direction
        if 0 <= other < len(members):
            members[index], members[other] = members[other], members[index]
        self._tag_orders[tag] = tuple(members)
        self._detail_tag = tag
        self.error = None
        self._show("review")

    def _assign_bulk_tags(self) -> None:
        """Apply the current ordered tags to every selected model, not prices or aliases."""
        tags = self._csv("#tags")
        if not tags:
            self.error = "Enter at least one tag before bulk assignment."
            self._show("review")
            return
        for wire, selection in self._selected_models.items():
            self._selected_models[wire] = replace(
                selection, edits=replace(selection.edits, tags=OptionalEdit.set(tags))
            )
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
        tags: dict[str, tuple[str, ...]] = (
            dict(self._tag_orders)
            if self._tag_orders != dict(self.catalog.tags)
            else {}
        )
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
                if outcome.kind == "existing"
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
        self._apply_changes(provider_patch, patches, tags)

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
        options: list[tuple[str, str]] = []
        for base, definition in sorted(self.catalog.models.items()):
            if definition.disabled:
                continue
            usable = self._has_usable_deployment(base)
            if usable:
                for deployment in definition.deployments:
                    if (
                        not deployment.disabled
                        and not self.catalog.providers[deployment.provider].disabled
                    ):
                        options.append((
                            f"{deployment.provider}/{deployment.name} ({base})",
                            base,
                        ))
                        break
                options.extend(
                    (f"{base} alias: {alias}", alias) for alias in definition.aliases
                )
        options.extend(
            (f"@{tag}", f"@{tag}")
            for tag, members in sorted(self.catalog.tags.items())
            if any(self._has_usable_deployment(member) for member in members)
        )
        return options

    def _finish(self) -> None:
        expression = self.query_one("#active-model", Select).value
        valid_expressions = {value for _label, value in self._active_model_options()}
        if not isinstance(expression, str) or expression not in valid_expressions:
            self.error = "Choose a usable canonical name, alias, or @tag."
            self._show("picker")
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

    def _select_existing_provider(self) -> None:
        provider_id = self._overview_provider_id or next(
            iter(self.catalog.providers), None
        )
        if provider_id is None:
            self.error = "Select a configured provider first."
            self._show("overview")
            return
        definition = self.catalog.providers[provider_id]
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

    def _configured_credential(self, env_var: str) -> str | None:
        """Resolve an existing provider credential through the host boundary."""
        resolver = getattr(self.credentials, "resolve_key", None)
        value = resolver(env_var) if callable(resolver) and env_var else None
        return value if isinstance(value, str) else None

    def _reset_unsaved_model_metadata(self) -> None:
        """Discard draft model state before entering another management path."""
        self._selected_models = {}
        self._selected_wires = set()
        self._detail_wire = None
        self._detail_tag = None
        self.discovered = ()
        self._credential_value = None
        self._credential_collision_warning = None
        self.error = None

    def _show(self, step: Step) -> None:
        if step == "form" and self.step == "again":
            self.provider = None
            self._selected_models = {}
            self._detail_wire = None
        self.step = step
        self.refresh(recompose=True)

    def action_back(self) -> None:
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
