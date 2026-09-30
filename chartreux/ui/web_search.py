"""Dedicated draft editor for the built-in web search tool."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import ClassVar, Literal, Protocol

from pydantic import JsonValue
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widgets import Button, Input, OptionList
from textual.widgets.option_list import Option

from chartreux.app_server.protocol import SettingsReadResponse
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.settings_service import SettingsReloadOutcome, SettingsSaveOutcome
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from chartreux.ui.widgets.vscode_compat import VscodeCompatInput

PREFIX = "tools.web_search."
MIN_MODAL_WIDTH = 84
MIN_MODAL_HEIGHT = 28
MAX_RESULTS = 20
STANDALONE_PROVIDERS = (
    ("mistral", "Mistral"),
    ("exa", "Exa"),
    ("brave", "Brave"),
    ("duckduckgo", "DuckDuckGo"),
)
ONBOARDING_PROVIDERS = STANDALONE_PROVIDERS[1:]
ADVANCED_FIELDS = (
    (
        "api_key_env_var",
        "Credential variable",
        "Empty uses the selected provider's default.",
    ),
    ("base_url", "Base URL", "Empty uses the selected provider's endpoint."),
    ("timeout", "Timeout (seconds)", "Must be greater than zero."),
    ("max_results", "Result limit", "Between 1 and 20."),
    ("model", "Mistral search model", "Only used by Mistral."),
)


class CredentialService(Protocol):
    def resolve_key(self, env_var: str) -> str | None: ...

    def save_key(self, env_var: str, key: str) -> object: ...


class SearchSettingsService(Protocol):
    async def read(self) -> SettingsReadResponse: ...

    async def save(
        self,
        changed_leaves: Mapping[str, JsonValue | None],
        expected_revision: str | None,
    ) -> SettingsSaveOutcome: ...

    async def retry_runtime(self) -> SettingsReloadOutcome: ...

    async def retry_ui(self) -> SettingsReloadOutcome: ...


class DraftChoiceList(NavigableOptionList):
    """Space chooses a radio value; Enter accepts the existing draft."""

    BINDINGS: ClassVar[list[BindingType]] = [
        *NavigableOptionList.BINDINGS,
        Binding("space", "select", "Choose", show=False),
        Binding("enter", "accept_draft", "Accept draft", show=False, priority=True),
    ]

    def action_accept_draft(self) -> None:
        if isinstance(self.screen, WebSearchScreen):
            self.screen.action_accept_provider_draft()

    def render_line(self, y: int) -> Strip:
        line = super().render_line(y)
        if not self.has_focus or self.scroll_offset.y + y != self.highlighted:
            return line
        focus_style = Style(reverse=True, bold=True)
        return Strip([
            Segment(
                segment.text,
                (segment.style or self.rich_style) + focus_style,
                segment.control,
            )
            for segment in line
        ])


type WebSearchExit = Literal["finish", "skip", "back"]


class WebSearchScreen(ModalScreen[WebSearchExit | None]):
    """Search settings draft; credentials persist through a separate action."""

    SCOPED_CSS = False
    CSS_PATH = "web_search.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", "Back", show=False, priority=True)
    ]

    def __init__(
        self,
        service: SearchSettingsService,
        snapshot: SettingsReadResponse,
        *,
        credentials: CredentialService,
        mode: Literal["standalone", "onboarding"] = "standalone",
    ) -> None:
        super().__init__(id="websearch-screen")
        if snapshot.web_search is None:
            raise ValueError("Web search settings are unavailable")
        self.service = service
        self.snapshot = snapshot
        self.credentials = credentials
        self.mode = mode
        self._leaving: WebSearchExit = "back"
        self._invalid_fields = set(snapshot.web_search.invalid_fields)
        self._fields = {
            field.path.removeprefix(PREFIX): field
            for field in snapshot.web_search.fields
        }
        self._draft = {
            name: ""
            if field.path in self._invalid_fields or field.effective_value is None
            else str(field.effective_value)
            for name, field in self._fields.items()
        }
        self._touched: set[str] = set()
        self._forced_reset: set[str] = set()
        self._busy = False
        self._needs_refresh = False
        self._runtime_failed = False
        self._ui_refresh_failed = False
        self._confirming = False
        self._advanced = bool(
            self._invalid_fields - {PREFIX + "provider", PREFIX + "permission"}
        )
        self._syncing = False
        self._provider_cursor: str | None = None
        self._message = ""
        self._credential_message = (
            "Save API key separately; saved keys survive Discard edits."
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="websearch-content"):
            yield NoMarkupStatic("Web search", id="websearch-title")
            yield NoMarkupStatic("", id="websearch-readiness")
            with VerticalScroll(id="websearch-form"):
                yield NoMarkupStatic("PROVIDER", classes="websearch-section")
                yield DraftChoiceList(id="websearch-providers")
                yield NoMarkupStatic("", id="websearch-current-provider")
                yield NoMarkupStatic("", id="websearch-provider-note")
                with Vertical(id="websearch-credential"):
                    yield NoMarkupStatic("API KEY", classes="websearch-section")
                    yield NoMarkupStatic("", id="websearch-selected-env")
                    with Horizontal(id="websearch-key-row"):
                        yield VscodeCompatInput(
                            placeholder="Paste API key; value is hidden",
                            password=True,
                            id="websearch-key",
                        )
                        yield Button("Save API key", id="websearch-save-key")
                    yield NoMarkupStatic("", id="websearch-key-result")
                yield Button("Show advanced settings", id="websearch-advanced-toggle")
                with Vertical(id="websearch-advanced"):
                    for name, label, help_text in ADVANCED_FIELDS:
                        with Vertical(id=f"websearch-field-{name}"):
                            yield NoMarkupStatic(label, classes="websearch-field-label")
                            yield VscodeCompatInput(id=f"websearch-input-{name}")
                            yield NoMarkupStatic(
                                help_text, classes="websearch-field-help"
                            )
            yield NoMarkupStatic("", id="websearch-message")
            with Horizontal(id="websearch-actions"):
                yield Button("Save search settings", id="websearch-save")
                yield Button(
                    "Back to presets" if self.mode == "onboarding" else "Back",
                    id="websearch-back",
                )
                if self.mode == "onboarding":
                    yield Button("Skip for now", id="websearch-skip")
                yield Button("Refresh (discard draft)", id="websearch-refresh")
                yield Button("Retry runtime reload", id="websearch-retry")
            with Vertical(id="websearch-confirmation"):
                yield NoMarkupStatic(
                    "Discard unsaved web search settings and any key still in the input? "
                    "Keys already saved to disk or this session remain saved.",
                    id="websearch-confirmation-text",
                )
                with Horizontal(classes="websearch-confirm-actions"):
                    yield Button("Keep editing", id="websearch-keep")
                    yield Button("Discard edits", id="websearch-discard")
            yield NoMarkupStatic(
                shortcut_hint(
                    f"{shortcut('Tab')} next  {shortcut('Space')} select  "
                    f"{shortcut('Enter')} accept/action  "
                    f"{shortcut('Esc')} back"
                ),
                id="websearch-help",
            )

    def on_mount(self) -> None:
        self._resize_surface()
        self._sync_form()
        self.query_one("#websearch-providers", OptionList).focus()

    def on_resize(self, _event: events.Resize) -> None:
        self._resize_surface()

    def _resize_surface(self) -> None:
        content = self.query_one("#websearch-content", Vertical)
        fullscreen = (
            self.size.width < MIN_MODAL_WIDTH or self.size.height < MIN_MODAL_HEIGHT
        )
        content.set_class(fullscreen, "fullscreen")
        content.border_title = "" if fullscreen else "Web search"

    def _saved_value(self, name: str) -> str:
        value = self._fields[name].effective_value
        return "" if value is None else str(value)

    def _selected_env(self) -> str | None:
        web = self.snapshot.web_search
        assert web is not None
        provider = self._draft["provider"]
        if self._visible_provider(provider) is None:
            return None
        if provider == "duckduckgo":
            return None
        return self._draft["api_key_env_var"] or web.default_credential_env_vars.get(
            provider
        )

    def _has_key(self, env_var: str) -> bool:
        try:
            return bool(self.credentials.resolve_key(env_var))
        except Exception:
            return False

    def _choice_prompt(
        self, value: str, label: str, selected: str | None, cursor: str
    ) -> str:
        marker = chrome_glyph("cursor") if value == cursor else " "
        radio = chrome_glyph("radio_selected" if value == selected else "radio_empty")
        return f"{marker} {radio}  {label}"

    def _provider_choices(self) -> tuple[tuple[str, str], ...]:
        return (
            ONBOARDING_PROVIDERS if self.mode == "onboarding" else STANDALONE_PROVIDERS
        )

    def _visible_provider(self, raw: str) -> str | None:
        if self.mode == "standalone" and raw == "auto":
            return "mistral"
        return raw if raw in dict(self._provider_choices()) else None

    def _sync_choices(
        self, widget_id: str, choices: tuple[tuple[str, str], ...], selected: str | None
    ) -> None:
        options = self.query_one(widget_id, OptionList)
        previous = (
            options.highlighted_option.id if options.highlighted_option else selected
        )
        values = [value for value, _ in choices]
        cursor = (
            str(previous)
            if previous in values
            else selected
            if selected in values
            else values[0]
        )
        options.clear_options()
        options.add_options(
            Option(self._choice_prompt(value, label, selected, cursor), id=value)
            for value, label in choices
        )
        options.highlighted = (
            values.index(previous)
            if previous in values
            else values.index(selected)
            if selected in values
            else 0
        )
        self._provider_cursor = cursor

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_list.id != "websearch-providers":
            return
        options = event.option_list
        cursor = str(event.option.id)
        for value in {self._provider_cursor, cursor}:
            if value is None:
                continue
            label = dict(self._provider_choices()).get(value)
            if label is not None:
                options.replace_option_prompt(
                    value,
                    self._choice_prompt(
                        value,
                        label,
                        self._visible_provider(self._draft["provider"]),
                        cursor,
                    ),
                )
        self._provider_cursor = cursor

    def _sync_form(self) -> None:  # noqa: PLR0915
        self._syncing = True
        try:
            self._sync_choices(
                "#websearch-providers",
                self._provider_choices(),
                self._visible_provider(self._draft["provider"]),
            )
            for name, _, _ in ADVANCED_FIELDS:
                widget = self.query_one(f"#websearch-input-{name}", Input)
                if widget.value != self._draft[name]:
                    widget.value = self._draft[name]
            provider = self._draft["provider"]
            choice_selected = self._visible_provider(provider) is not None
            keyed = choice_selected and provider != "duckduckgo"
            self.query_one("#websearch-credential").display = keyed
            self.query_one("#websearch-field-api_key_env_var").display = (
                self._advanced and keyed
            )
            self.query_one("#websearch-field-base_url").display = (
                self._advanced and keyed
            )
            self.query_one("#websearch-field-timeout").display = self._advanced
            self.query_one("#websearch-field-max_results").display = self._advanced
            self.query_one("#websearch-field-model").display = (
                self._advanced
                and self.mode == "standalone"
                and provider in {"auto", "mistral"}
            )
            self.query_one("#websearch-advanced").display = self._advanced
            self.query_one("#websearch-advanced-toggle", Button).label = (
                "Hide advanced settings" if self._advanced else "Show advanced settings"
            )
            env_var = self._selected_env()
            self.query_one("#websearch-selected-env", NoMarkupStatic).update(
                f"Selected variable: {env_var}"
                if env_var
                else "No API key required"
                if not keyed
                else "Credential variable unavailable; repair the Mistral provider config"
            )
            web = self.snapshot.web_search
            assert web is not None
            provider_field = self._fields["provider"]
            provider_labels = dict(STANDALONE_PROVIDERS) | {"auto": "Mistral"}
            effective_label = provider_labels.get(
                self._saved_value("provider"),
                f"Invalid ({self._saved_value('provider')})",
            )
            user_label = (
                str(provider_field.saved_value)
                if provider_field.saved_explicit
                else "no user override"
            )
            self.query_one("#websearch-current-provider", NoMarkupStatic).update(
                f"Effective: {effective_label} ({provider_field.origin}); saved user: {user_label}"
            )
            if self.mode == "onboarding" and not choice_selected:
                readiness = "Choose Exa, Brave or DuckDuckGo to set up web search, or Skip for now"
            elif self._dirty_settings():
                readiness = (
                    "Credential variable unavailable; repair provider config"
                    if keyed and env_var is None
                    else "Credential available for draft; validate by saving settings"
                    if env_var is None or self._has_key(env_var)
                    else f"Draft needs a key in {env_var}"
                )
            elif web.readiness == "ready":
                readiness = "Configured; connection not verified"
            elif web.readiness == "missing_key" and env_var and self._has_key(env_var):
                readiness = "Credential available; connection not verified"
            else:
                readiness = (
                    web.readiness_message or "Web search configuration needs attention"
                )
            self.query_one("#websearch-readiness", NoMarkupStatic).update(readiness)
            self.query_one("#websearch-provider-note", NoMarkupStatic).update(
                "Invalid saved fields: "
                + ", ".join(
                    sorted(path.removeprefix(PREFIX) for path in self._invalid_fields)
                )
                + ". Choose a valid provider or replace fields in Advanced; for permission use /open-config-file."
                if self._invalid_fields
                else "Provider switch resets custom credential variable and endpoint to the new provider defaults."
                if self._forced_reset
                else "Choose a fallback provider with Space; then Save and finish."
                if self.mode == "onboarding"
                else "Choose a provider; the radio choice is a draft until Save search settings."
            )
            self.query_one("#websearch-key-result", NoMarkupStatic).update(
                self._credential_message
            )
            message = self._message
            if (
                (self._runtime_failed or self._ui_refresh_failed)
                and not self._needs_refresh
                and self._dirty_settings()
            ):
                message += " Save or discard the draft before retry."
            self.query_one("#websearch-message", NoMarkupStatic).update(message)
            self.query_one("#websearch-refresh", Button).display = self._needs_refresh
            self.query_one("#websearch-retry", Button).display = (
                self._runtime_failed or self._ui_refresh_failed
            ) and not self._needs_refresh
            self.query_one("#websearch-retry", Button).label = (
                "Retry UI refresh"
                if self._ui_refresh_failed
                else "Retry settings reload"
                if self.mode == "onboarding"
                else "Retry runtime reload"
            )
            self.query_one("#websearch-confirmation").display = self._confirming
            self.query_one("#websearch-form").display = not self._confirming
            self.query_one("#websearch-actions").display = not self._confirming
            save_button = self.query_one("#websearch-save", Button)
            if self.mode == "onboarding":
                save_button.styles.min_width = 21
                save_button.label = (
                    "Save and finish" if self._dirty_settings() else "Finish setup"
                )
            save_button.disabled = (
                self._busy
                or self._needs_refresh
                or (self.mode == "onboarding" and not choice_selected)
                or (self.mode == "standalone" and not self._dirty_settings())
            )
            editing_disabled = self._busy or self._needs_refresh
            self.query_one(
                "#websearch-providers", OptionList
            ).disabled = editing_disabled
            self.query_one(
                "#websearch-advanced-toggle", Button
            ).disabled = editing_disabled
            self.query_one("#websearch-key", Input).disabled = editing_disabled
            for name, _, _ in ADVANCED_FIELDS:
                self.query_one(
                    f"#websearch-input-{name}", Input
                ).disabled = editing_disabled
            self.query_one("#websearch-save-key", Button).disabled = (
                editing_disabled or env_var is None
            )
            self.query_one("#websearch-back", Button).disabled = self._busy
            if self.mode == "onboarding":
                self.query_one("#websearch-skip", Button).disabled = self._busy
            self.query_one("#websearch-refresh", Button).disabled = self._busy
            self.query_one("#websearch-retry", Button).disabled = (
                self._busy or self._dirty_settings()
            )
        finally:
            self._syncing = False

    def _dirty_settings(self) -> bool:
        return bool(self._forced_reset) or any(
            self._draft[name] != self._saved_value(name) for name in self._touched
        )

    def _changes(self) -> dict[str, JsonValue]:
        unrepaired = self._invalid_fields - {
            PREFIX + name for name in self._touched | self._forced_reset
        }
        if unrepaired:
            names = ", ".join(sorted(path.removeprefix(PREFIX) for path in unrepaired))
            raise ValueError(
                f"Repair invalid saved fields before saving: {names}. "
                "Use /open-config-file for permission."
            )
        changed: dict[str, JsonValue] = {}
        for name in self._touched | self._forced_reset:
            raw = self._draft[name]
            if name not in self._forced_reset and raw == self._saved_value(name):
                continue
            if name in {"timeout", "max_results"}:
                try:
                    value: JsonValue = int(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{name.replace('_', ' ').capitalize()} must be a whole number."
                    ) from exc
                if name == "timeout" and value <= 0:
                    raise ValueError("Timeout must be greater than zero.")
                if name == "max_results" and not 1 <= value <= MAX_RESULTS:
                    raise ValueError("Result limit must be between 1 and 20.")
            else:
                value = raw
            if name == "model" and not raw.strip():
                raise ValueError("Mistral search model cannot be empty.")
            if (
                name == "api_key_env_var"
                and raw
                and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw)
            ):
                raise ValueError(
                    "Credential variable must be a valid environment variable name."
                )
            changed[PREFIX + name] = value
        return changed

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._syncing or self._busy or self._confirming or self._needs_refresh:
            return
        selected = str(event.option.id)
        if event.option_list.id == "websearch-providers":
            if selected == self._visible_provider(self._draft["provider"]):
                return
            self._draft["provider"] = selected
            self._draft["api_key_env_var"] = ""
            self._draft["base_url"] = ""
            self._touched.update({"provider", "api_key_env_var", "base_url"})
            self._forced_reset.update({"api_key_env_var", "base_url"})
            self.query_one("#websearch-key", Input).value = ""
            self._credential_message = (
                "Unsaved key cleared; saved credentials remain saved."
            )
        else:
            return
        self._message = "Draft changed. Save search settings to update user config."
        self._sync_form()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._syncing or self._busy or self._needs_refresh or not event.input.id:
            return
        if event.input.id.startswith("websearch-input-"):
            name = event.input.id.removeprefix("websearch-input-")
            if event.value == self._draft[name]:
                return
            self._draft[name] = event.value
            self._touched.add(name)
            self._message = "Draft changed. Save search settings to update user config."
            self._sync_form()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "websearch-key":
            self.query_one("#websearch-save-key", Button).focus()
        elif event.input.id and event.input.id.startswith("websearch-input-"):
            self.query_one("#websearch-save", Button).focus()

    def action_accept_provider_draft(self) -> None:
        """Advance focus without changing the selected radio value."""
        if self._busy or self._needs_refresh:
            return
        if self._visible_provider(self._draft["provider"]) is None:
            self._message = "Choose a valid provider with Space before continuing."
            self._sync_form()
            return
        target = (
            "#websearch-key"
            if self._selected_env() is not None
            else "#websearch-advanced-toggle"
        )
        self.query_one(target).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._busy:
            return
        match event.button.id:
            case "websearch-save":
                if self.mode == "onboarding" and not self._dirty_settings():
                    self.run_worker(
                        self._finish_setup(), group="web-search-settings-finish"
                    )
                else:
                    self.run_worker(
                        self._save_settings(), group="web-search-settings-save"
                    )
            case "websearch-save-key":
                self._save_key()
            case "websearch-back":
                self.action_back()
            case "websearch-skip":
                self._request_leave("skip")
            case "websearch-advanced-toggle":
                self._advanced = not self._advanced
                self._sync_form()
            case "websearch-refresh":
                self.run_worker(self._refresh(), group="web-search-settings-refresh")
            case "websearch-retry":
                self.run_worker(self._retry_runtime(), group="web-search-runtime-retry")
            case "websearch-keep":
                self._confirming = False
                self._sync_form()
                self.query_one("#websearch-back", Button).focus()
            case "websearch-discard":
                self.query_one("#websearch-key", Input).value = ""
                self.dismiss(self._leaving if self.mode == "onboarding" else None)

    def _save_key(self) -> None:
        env_var = self._selected_env()
        key_input = self.query_one("#websearch-key", Input)
        key = key_input.value
        if env_var is None:
            self._credential_message = "This provider does not use an API key."
        elif not key.strip():
            self._credential_message = "Enter an API key before choosing Save API key."
        else:
            try:
                result = self.credentials.save_key(env_var, key)
            except Exception:
                self._credential_message = (
                    f"Could not save API key for {env_var}; the key was not saved."
                )
            else:
                status = getattr(result, "status", "")
                if status == "saved":
                    self._credential_message = f"API key saved for {env_var}. Search settings draft is unchanged."
                elif status == "session_only":
                    self._credential_message = f"API key available for this session in {env_var}; disk save failed. Search settings draft is unchanged."
                else:
                    self._credential_message = (
                        getattr(result, "message", None)
                        or f"Could not save API key for {env_var}."
                    )
                if status in {"saved", "session_only"}:
                    key_input.value = ""
        self._sync_form()

    async def _save_settings(self) -> None:
        if self._busy or self._needs_refresh:
            return
        try:
            changes = self._changes()
        except ValueError as exc:
            self._message = f"Not saved: {exc}"
            self._sync_form()
            return
        if not changes:
            self._message = "No search settings changes to save."
            self._sync_form()
            return
        self._busy = True
        outcome: SettingsSaveOutcome | None = None
        self._message = "Saving search settings to user config…"
        self._sync_form()
        try:
            outcome = await self.service.save(changes, self.snapshot.user_revision)
        except Exception as exc:
            self._message = f"Not saved: {exc}"
        else:
            self._record_save_outcome(outcome)
        finally:
            self._busy = False
            self._sync_form()
            if not self._needs_refresh:
                self.query_one("#websearch-save", Button).focus()
            if (
                self.mode == "onboarding"
                and outcome is not None
                and not self._needs_refresh
            ):
                if (
                    outcome.persistence == "saved"
                    and outcome.application == "applied"
                    and not outcome.shadowed
                ):
                    await self._finish_setup()

    async def _finish_setup(self) -> None:
        if self.mode != "onboarding" or self._busy or self._needs_refresh:
            return
        if self._visible_provider(self._draft["provider"]) is None:
            self._message = (
                "Choose Exa, Brave or DuckDuckGo with Space, or Skip for now."
            )
        elif self.query_one("#websearch-key", Input).value:
            self._message = (
                "Save or clear the API key in the input before finishing setup."
            )
        elif self._runtime_failed or self._ui_refresh_failed:
            self._message = (
                "Saved settings need a successful reload before finishing setup."
            )
        elif self._dirty_settings():
            self._message = "Save search settings before finishing setup."
        else:
            self._busy = True
            self._sync_form()
            try:
                snapshot = await self.service.read()
            except Exception as error:
                self._needs_refresh = True
                self._message = f"Could not verify saved search settings: {error}. Refresh before finishing."
            else:
                self._adopt_snapshot(snapshot)
                if snapshot.view_only or (
                    snapshot.web_search is not None
                    and any(
                        field.origin == "live config"
                        for field in snapshot.web_search.fields
                    )
                ):
                    self._needs_refresh = True
                    self._message = "Saved search settings could not be verified. Refresh or Skip for now."
                elif self._visible_provider(self._draft["provider"]) is None:
                    self._message = "Saved provider changed. Choose Exa, Brave or DuckDuckGo with Space, or Skip for now."
                elif (
                    snapshot.web_search is None
                    or snapshot.web_search.readiness == "invalid"
                ):
                    self._message = (
                        "Repair invalid web search settings, or choose Skip for now."
                    )
                elif snapshot.web_search.readiness == "missing_key":
                    self._message = "Save a web search API key, choose a keyless provider, or Skip for now."
                else:
                    self.dismiss("finish")
                    return
            finally:
                self._busy = False
        self._sync_form()

    def _record_save_outcome(self, outcome: SettingsSaveOutcome) -> None:
        if outcome.persistence == "not_saved":
            self._message = f"Not saved: {outcome.error or 'write failed'}."
            if outcome.error == "conflict":
                self._needs_refresh = True
                self._message += " Refresh before editing again."
        elif outcome.snapshot is None:
            self._needs_refresh = True
            self._ui_refresh_failed = (
                outcome.error == "ui_update_failed_snapshot_unknown"
            )
            self._runtime_failed = (
                outcome.application != "applied" and not self._ui_refresh_failed
            )
            if outcome.persistence == "durability_uncertain":
                self._message = "Search settings write completed, but disk durability and current state are uncertain. Refresh before editing again."
            else:
                self._message = "Saved to user config; current settings could not be read. Refresh before editing again."
            if self._runtime_failed:
                self._message += (
                    " Startup settings need a reload after Refresh."
                    if self.mode == "onboarding"
                    else " Runtime not applied; retry after Refresh."
                )
            elif self._ui_refresh_failed:
                self._message += " Runtime applied; UI refresh failed. Retry UI refresh after Refresh."
        else:
            self._adopt_snapshot(outcome.snapshot)
            self._ui_refresh_failed = outcome.error == "ui_update_failed"
            self._runtime_failed = (
                outcome.application != "applied" and not self._ui_refresh_failed
            )
            if outcome.persistence == "durability_uncertain":
                self._message = (
                    "Search settings may have saved; disk durability is uncertain."
                )
            elif self._ui_refresh_failed:
                self._message = "Saved to user config and runtime; UI refresh failed. Retry UI refresh."
            elif outcome.application == "failed":
                self._message = (
                    "Saved to user config; startup settings reload failed. Retry settings reload."
                    if self.mode == "onboarding"
                    else "Saved to user config; runtime apply failed. Retry runtime reload."
                )
            elif outcome.application == "unchanged":
                self._message = (
                    "Saved to user config; reload settings before finishing setup."
                    if self.mode == "onboarding"
                    else "Saved to user config; runtime unchanged. Retry runtime reload."
                )
            else:
                self._message = (
                    "Search settings saved to user config for startup."
                    if self.mode == "onboarding"
                    else "Search settings saved to user config and applied to current runtime."
                )
            if outcome.shadowed:
                self._message += " A higher config layer still controls: " + ", ".join(
                    outcome.shadowed
                )

    def _adopt_snapshot(self, snapshot: SettingsReadResponse) -> None:
        if snapshot.web_search is None:
            self._needs_refresh = True
            return
        self.snapshot = snapshot
        self._invalid_fields = set(snapshot.web_search.invalid_fields)
        self._fields = {
            field.path.removeprefix(PREFIX): field
            for field in snapshot.web_search.fields
        }
        self._draft = {
            name: ""
            if field.path in self._invalid_fields or field.effective_value is None
            else str(field.effective_value)
            for name, field in self._fields.items()
        }
        self._touched.clear()
        self._forced_reset.clear()
        self._needs_refresh = False
        self._advanced = self._advanced or bool(
            self._invalid_fields - {PREFIX + "provider", PREFIX + "permission"}
        )

    async def _refresh(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._message = "Refreshing authoritative search settings…"
        self._sync_form()
        try:
            snapshot = await self.service.read()
            self._adopt_snapshot(snapshot)
            self._message = "Search settings refreshed; prior draft discarded. Review before editing."
        except Exception as exc:
            self._message = f"Refresh failed: {exc}. Saved state remains unknown."
        finally:
            self._busy = False
            self._sync_form()

    async def _retry_runtime(self) -> None:
        if (
            self._busy
            or self._needs_refresh
            or not (self._runtime_failed or self._ui_refresh_failed)
            or self._dirty_settings()
        ):
            return
        self._busy = True
        retry_ui = self._ui_refresh_failed
        self._message = (
            "Refreshing the UI from applied search settings…"
            if retry_ui
            else "Reloading saved search settings for startup…"
            if self.mode == "onboarding"
            else "Reloading saved search settings into runtime…"
        )
        self._sync_form()
        try:
            outcome = (
                await self.service.retry_ui()
                if retry_ui
                else await self.service.retry_runtime()
            )
            if not outcome.runtime_applied:
                self._message = (
                    "Saved to user config; settings reload failed: "
                    if self.mode == "onboarding"
                    else "Saved to user config; runtime reload failed: "
                ) + (
                    f"{outcome.error or 'unknown error'}. Retry settings reload."
                    if self.mode == "onboarding"
                    else f"{outcome.error or 'unknown error'}. Retry runtime reload."
                )
            else:
                self._runtime_failed = False
                self._ui_refresh_failed = not outcome.ui_applied
                if outcome.snapshot is not None:
                    self._adopt_snapshot(outcome.snapshot)
                else:
                    self._needs_refresh = True
                if self._ui_refresh_failed:
                    self._message = (
                        "Saved settings applied in runtime; UI refresh failed. "
                        "Refresh settings first if required, then Retry UI refresh."
                    )
                elif self._needs_refresh:
                    self._message = (
                        "Saved settings reloaded; settings read failed. Refresh before editing."
                        if self.mode == "onboarding"
                        else "Saved settings applied in runtime; settings read failed. Refresh before editing."
                    )
                else:
                    self._message = (
                        "Saved search settings reloaded for startup."
                        if self.mode == "onboarding"
                        else "Saved search settings applied to current runtime."
                    )
        except Exception as exc:
            self._message = (
                f"Retry failed before outcome was known: {exc}. Refresh settings."
            )
            self._needs_refresh = True
        finally:
            self._busy = False
            self._sync_form()

    def action_back(self) -> None:
        self._request_leave("back")

    def _request_leave(self, destination: WebSearchExit) -> None:
        if self._busy:
            return
        self._leaving = destination
        if self._confirming:
            self._confirming = False
            self._sync_form()
            self.query_one("#websearch-back", Button).focus()
            return
        if self._dirty_settings() or self.query_one("#websearch-key", Input).value:
            self._confirming = True
            self._sync_form()
            self.query_one("#websearch-keep", Button).focus()
            return
        self.dismiss(destination if self.mode == "onboarding" else None)
