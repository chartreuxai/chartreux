"""Connection controller for the provider workbench with an injected host."""

from __future__ import annotations

from collections.abc import Coroutine
from copy import deepcopy
from dataclasses import replace
from functools import partial
import re
from typing import Any, Literal, cast

from textual.widgets import Input, Label, SelectionList
from textual.widgets.option_list import Option

from chartreux.core.dispatch.session import bind_snapshot_policy
from chartreux.core.model_catalog.contracts import (
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
    ProviderDraft,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.matching import match_discovered_model
from chartreux.core.model_catalog.presets import (
    FULLY_CUSTOM,
    MISTRAL,
    PRESETS,
    ProviderPreset,
)
from chartreux.core.model_catalog.schema import ProviderDefinition, valid_provider_name
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.providers.management_state import (
    ConnectionDraft,
    ManagementState,
    usable_canonical_models,
)
from chartreux.ui.providers.workbench_types import WorkbenchView
from chartreux.ui.providers.workbench_widgets import ModelChecklist, WorkbenchList
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class WorkbenchConnection:  # noqa: PLR0904 - connection controller interface
    """Add/edit connection flow, credential, discovery, and models checklist.

    Plain object (no mixins): the screen constructs it in ``__init__`` with
    ``host=self`` and is its late-bound host — every collaboration is
    ``self._host._method(...)`` at call time, never a constructor-captured
    bound method. The add/edit connection state (``_add*``, ``_stage``,
    ``_editing``, ``_protocol_*``) stays on the screen; the controller
    mutates it through the host exactly where the screen's own code did.

    Pinned invariants, moved verbatim: ``_save_key`` mutates
    ``_saved_credential_envs``/``_session_only_credential_envs`` through the
    host by mutation, never rebinding (the screen shares the session-only
    set with the ChartreuxApp by reference); the ``_discover`` rejection
    window keeps its ``_feedback_token`` comparison; the discovery worker
    runs through ``host.run_worker`` with ``group="workbench-discovery"``;
    the ``_accept_editor`` ``detail:`` branch stays one method; and the
    TWP:78-79 ``_select_action`` instance-patch seam keeps screen-attribute
    resolution (``select_add_action`` calls ``self._host._select_action``).
    """

    def __init__(self, *, host: Any) -> None:
        self._host = host

    def start_add(self) -> None:
        self._host._add_opener = (
            self._host._frame() if self._host._view == WorkbenchView.PROVIDERS else None
        )
        self._host._pre_add_state = (
            deepcopy(self._host.state) if self._host.state is not None else None
        )
        self._host._provider_open = False
        self._host._stage = "choose"
        self._host._view = WorkbenchView.CHOOSE
        self._host._add = None
        self._host._add_checkpoint = None
        self._host._add_state = None
        if self._host.is_mounted:
            self._host.query_one("#wb-connection-key", Input).value = ""
        if self._host._feedback_kind not in {"error", "warning"}:
            self._host._message = ""
        choose = self._host.query_one("#wb-choose", NavigableOptionList)
        choose.clear_options()
        for name in sorted(self._host.snapshot.catalog.providers):
            choose.add_option(
                Option(
                    f"Existing: {self._host._friendly(name)} ({name})",
                    id=f"existing:{name}",
                )
            )
        for preset in PRESETS:
            choose.add_option(Option(f"{preset.name}", id=preset.id))
        choose.highlighted = 0
        self._host._activate_view(WorkbenchView.CHOOSE, focus_id="wb-choose")
        self._host._update_help()

    def choose_preset(self, preset: ProviderPreset) -> None:
        if self._host._view == WorkbenchView.PROVIDERS:
            self._host._add_opener = self._host._frame()
        if (
            self._host.state is not None
            and self._host._add is None
            and self._host._pre_add_state is None
        ):
            self._host._pre_add_state = deepcopy(self._host.state)
        name = (
            ""
            if preset is FULLY_CUSTOM
            else preset.id
            if preset is MISTRAL
            else preset.name
        )
        existing = (
            self._host.snapshot.catalog.providers.get("mistral")
            if preset is MISTRAL and "mistral" in self._host.snapshot.overlaid_providers
            else None
        )
        self._host._mistral_overwrite_approved = False
        self._host._draft_session += 1
        self._host._add = ProviderDraft(
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
        self._host._add_checkpoint = deepcopy(self._host._add)
        if self._host.is_mounted:
            self._host.query_one("#wb-connection-key", Input).value = ""
        self._host._stage = "connection"
        self._host._view = WorkbenchView.CONNECTION
        self.refresh_add_connection()
        self._host._activate_view(WorkbenchView.CONNECTION, focus_id="wb-actions")

    def refresh_add_connection(
        self, *, highlight: str | None = None, selected_id: str | None = None
    ) -> None:
        draft = self._host._add
        if draft is None:
            return
        actions = self._host.query_one("#wb-actions", NavigableOptionList)
        position = self._host._position_before_rebuild(
            WorkbenchView.CONNECTION, "wb-actions"
        )
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
            actions.add_option(Option(self._host._summary(f"{label}  {value}"), id=key))
        actions.add_option(
            Option(
                self._host._summary(
                    f"API key — {self._host._api_key_status(draft.api_key_env_var)}"
                ),
                id="key",
            )
        )
        ids = [str(option.id) for option in actions.options]
        actions.highlighted = ids.index(previous) if previous in ids else 0
        self.refresh_connection_form()
        if highlight is None and selected_id is None:
            self._host._restore_group_position(position)
        self._host._update_help()

    def refresh_connection_form(self) -> None:
        draft = self._host._add
        if draft is None or not self._host.is_mounted:
            return
        for widget_id, value in (
            ("#wb-connection-name", draft.name),
            ("#wb-connection-base", draft.api_base),
            ("#wb-connection-env", draft.api_key_env_var),
        ):
            self._host.query_one(widget_id, Input).value = value
        style = self._host.query_one("#wb-connection-style", WorkbenchList)
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
        self.refresh_connection_actions()

    def refresh_connection_actions(self) -> None:
        draft = self._host._add
        if draft is None:
            return
        actions = self._host.query_one("#wb-connection-actions", WorkbenchList)
        position = self._host._position_before_rebuild(
            WorkbenchView.CONNECTION, "wb-connection-actions"
        )
        current = actions.highlighted_option.id if actions.highlighted_option else None
        actions.clear_options()
        actions.add_option(Option("Save and configure models", id="continue"))
        ids = [option.id for option in actions.options]
        actions.highlighted = ids.index(current) if current in ids else len(ids) - 1
        self._host._restore_group_position(position)

    def connection_action(self, key: str) -> None:
        if self._host._busy:
            return
        draft = self._host._add
        if draft is None:
            return
        if key == "style":
            self.open_protocol_picker(draft.api_style)
            return
        if key == "continue":
            try:
                provider_id = (
                    draft.provider_id
                    if draft.preset == "existing"
                    else self.new_provider_id(draft.name)
                )
                definition = ProviderDefinition.model_validate({
                    **(
                        self._host.snapshot.catalog.providers[provider_id].model_dump()
                        if provider_id == "mistral"
                        and provider_id in self._host.snapshot.catalog.providers
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
                self._host._set_feedback("field", str(exc), "error")
                field = (
                    "name"
                    if not draft.name.strip() or "name" in str(exc).lower()
                    else "base"
                    if "base" in str(exc).lower() or "url" in str(exc).lower()
                    else "env"
                )
                self.refresh_add_connection(highlight=field)
                self._host._update_help()
                return
            if (
                self._host._add_state
                and self._host._add_state.provider_id != provider_id
            ):
                self._host._set_feedback(
                    "field",
                    "Keep the original name to preserve selected models, or create this provider first and add another.",
                    "error",
                )
                self.refresh_add_connection(highlight="name")
                self._host._update_help()
                return
            if self._host._pre_add_state is None and self._host.state is not None:
                self._host._pre_add_state = deepcopy(self._host.state)
            self._host._add = replace(draft, provider_id=provider_id, name=provider_id)
            if self._host._view == WorkbenchView.CONNECTION and (
                not self._host._navigation
                or self._host._navigation[-1].view != WorkbenchView.CONNECTION
            ):
                self._host._push_view(WorkbenchView.CONNECTION)
            if self._host.state is None:
                self._host.state = ManagementState.from_snapshot(
                    self._host.snapshot, provider_id
                )
            if provider_id in self._host.snapshot.catalog.providers:
                self._host.state.for_provider(provider_id)
                self._host.state.set_connection(
                    ConnectionDraft.from_definition(definition)
                )
            else:
                self._host.state.stage_provider(provider_id, definition)
            self._host._add_state = self._host.state
            existing_edit = draft.preset == "existing"
            self._host._connection_pending = (
                deepcopy(self._host.state.pending) if existing_edit else None
            )
            self._host._after_commit = "models"
            if existing_edit:
                self._host._connection_approved = True
            self._host._start_commit(self._host.state, create=not existing_edit)
            return
        if key in {"name", *self._host.FIELDS, "key"}:
            self.open_connection_editor(key, draft)

    def open_connection_editor(self, key: str, draft: ProviderDraft) -> None:
        if key == "key" and not draft.api_key_env_var:
            self._host._key_after_env = True
            key = "env"
        self._host._editing = key
        editor = self._host.query_one("#wb-input", Input)
        editor.password = key == "key"
        editor.value = (
            "" if key == "key" else str(getattr(draft, self._host.FIELDS.get(key, key)))
        )
        if key == "env" and self._host._key_after_env and not editor.value:
            normalized = re.sub(r"[^A-Z0-9]+", "_", draft.name.upper()).strip("_")
            editor.value = f"{normalized or 'PROVIDER'}_API_KEY"
        self._host.query_one("#wb-editor-label", Label).update(
            {
                "name": "Provider Name *",
                "base": "API Base *",
                "style": "API Style *",
                "env": "Credential Env Var (for API key)"
                if self._host._key_after_env
                else "Credential Env Var",
                "key": "API Key *",
            }[key]
        )
        self._host._sync_view()
        self._host._restore_view_focus("wb-input")
        self._host._update_help()

    def refresh_protocol(self) -> None:
        picker = self._host.query_one("#wb-protocol", NavigableOptionList)
        current = (
            str(picker.highlighted_option.id) if picker.highlighted_option else None
        )
        values = ("openai", "openai-responses", "anthropic")
        picker.clear_options()
        for value in values:
            picker.add_option(
                Option(
                    f"{chrome_glyph('radio_selected') if value == self._host._protocol_value else chrome_glyph('radio_empty')} {value}",
                    id=value,
                )
            )
        picker.highlighted = (
            values.index(current)
            if current in values
            else values.index(self._host._protocol_value or "openai")
        )

    def select_protocol(self, value: str) -> None:
        if self._host._can_activate("wb-protocol"):
            self._host._protocol_value = value
            self.refresh_protocol()

    def accept_protocol(self) -> None:
        if not self._host._can_activate("wb-protocol"):
            return
        value = self._host._protocol_value
        self._host._protocol_picker = False
        if value and self._host._stage == "connection" and self._host._add:
            self._host._add = replace(self._host._add, api_style=value)
            self.refresh_add_connection(highlight="style")
        elif value and self._host.state:
            self._host.state.set_connection(
                replace(self._host.state.connection, api_style=value)
            )
            self._host._refresh_actions(highlight="style")
        if not self._host._pop_view():
            self._host._activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
        self._host._update_help()

    def open_protocol_picker(self, current: str) -> None:
        if not self._host._protocol_picker:
            self._host._push_view(WorkbenchView.PROTOCOL)
        self._host._protocol_value = current
        self.refresh_protocol()
        self._host._protocol_picker = True
        self._host._activate_view(WorkbenchView.PROTOCOL, focus_id="wb-protocol")
        self._host._update_help()

    def new_provider_id(self, name: str) -> str:
        provider_id = valid_provider_name(name)
        if (
            self._host.state is not None
            and provider_id in self._host.state.staged_providers
            and (
                self._host._add_state is None
                or self._host._add_state.provider_id != provider_id
            )
        ):
            raise ValueError(
                f"Provider name {provider_id!r} already exists in the draft."
            )
        if provider_id in self._host.snapshot.catalog.providers and not (
            provider_id == "mistral"
            and self._host._add
            and self._host._add.preset == "mistral"
        ):
            raise ValueError(f"Provider name {provider_id!r} already exists.")
        return provider_id

    def refresh_add_models(self, *, highlight: str | None = None) -> None:
        state = self._host.state
        if state is None:
            return
        actions = self._host.query_one("#wb-actions", NavigableOptionList)
        position = self._host._position_before_rebuild(
            WorkbenchView.ACTIONS, "wb-actions"
        )
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
        if self._host._reload_failed:
            actions.add_option(Option("Retry Runtime Reload", id="retry-reload"))
        ids = [str(option.id) for option in actions.options]
        actions.highlighted = ids.index(previous) if previous in ids else 1
        if highlight is None:
            self._host._restore_group_position(position)
        self._host._update_help()

    def advance_models(self, action: str) -> None:
        if self._host.state is None or action not in {
            "add-another",
            "continue-presets",
        }:
            return
        destination = "another" if action == "add-another" else "presets"
        if not self._host.state.dirty:
            if destination == "another":
                self.start_add()
            else:
                self._host._open_presets()
            return
        self._host._after_commit = destination
        self._host._start_commit(self._host.state, create=False)

    def close_picker(self) -> None:
        self._host._picker = False
        if self._host._pop_view():
            self._host._update_help()
            return
        view = (
            WorkbenchView.CATALOG
            if self._host._models_view
            else WorkbenchView.ACTIONS
            if self._host.state
            else WorkbenchView.PROVIDERS
        )
        self._host._activate_view(view)
        self._host._update_help()

    def expand(self, provider_id: str, *, committed: bool = False) -> None:
        if self._host._busy and not committed:
            return
        if not committed and self._host._view == WorkbenchView.PROVIDERS:
            self._host._push_view(WorkbenchView.PROVIDERS)
        if self._host.state is not None:
            self._host.state.begin_discovery()
            if provider_id in self._host.state.catalog.providers:
                self._host.state.for_provider(provider_id)
            else:
                self._host.state = ManagementState.from_snapshot(
                    self._host.snapshot, provider_id
                )
        else:
            self._host.state = ManagementState.from_snapshot(
                self._host.snapshot, provider_id
            )
        self._host._confirm = None
        if self._host._feedback_kind not in {"error", "warning"}:
            self._host._message = ""
        self._host._connection_approved = False
        self._host._models_list_open = False
        self._host._provider_open = True
        self._host._refresh_actions()
        self._host._activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
        self._host._update_help()

    def select_add_action(self, key: str, state: ManagementState) -> bool:
        if key == "retry-reload":
            self._host._retry_reload()
            return True
        if key in {"create", "add-another", "continue-presets"}:
            self.advance_models("continue-presets" if key == "create" else key)
        elif key == "edit-connection":
            self.begin_saved_connection_edit(state)
        elif key == "edit-key":
            self._host._select_action("key")
        elif key == "manual":
            self._host._editing = "manual"
            editor = self._host.query_one("#wb-input", Input)
            editor.password = False
            editor.value = ""
            self._host.query_one("#wb-editor-label", Label).update("Model ID *")
            self._host._sync_view()
            self._host._restore_view_focus("wb-input")
            self._host._update_help()
        else:
            return key == "status"
        return True

    def begin_saved_connection_edit(self, state: ManagementState) -> None:
        connection = state.connection
        self._host._draft_session += 1
        self._host._add = ProviderDraft(
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
        self._host._add_checkpoint = deepcopy(self._host._add)
        self._host._add_state = state
        state.begin_discovery()
        self._host._stage = "connection"
        self._host._view = WorkbenchView.CONNECTION
        self._host._models_list_open = False
        self.refresh_add_connection()
        self._host._activate_view(WorkbenchView.CONNECTION, focus_id="wb-actions")

    def select_action(self, key: str) -> None:  # noqa: PLR0912, PLR0915
        state = self._host.state
        if state is None or self._host._busy:
            return
        if self._host._stage == "models" and self.select_add_action(key, state):
            return
        if key == "retry-reload":
            self._host._retry_reload()
            return
        if key == "presets":
            self._host._open_presets()
            return
        if key == "style":
            self.open_protocol_picker(state.connection.api_style)
            return
        if key in self._host.FIELDS or key == "key":
            if key == "key" and not state.connection.api_key_env_var:
                self._host._message = "Set a credential environment variable first."
                self._host._update_help()
                return
            if key == "key":
                self._host._credential_input_id = "wb-input"
            self._host._editing = key
            field = self._host.FIELDS.get(key)
            editor = self._host.query_one("#wb-input", Input)
            editor.password = key == "key"
            editor.value = str(getattr(state.connection, field)) if field else ""
            self._host.query_one("#wb-editor-label", Label).update(
                {
                    "base": "API Base",
                    "style": "API Style",
                    "env": "Credential Env Var",
                    "key": "API Key *",
                }[key]
            )
            self._host._sync_view()
            self._host._restore_view_focus("wb-input")
        elif key == "discover":
            self._host.run_worker(
                self.discovery_request(state),
                group="workbench-discovery",
                exclusive=True,
            )
        elif key == "models":
            if self._host._view != WorkbenchView.MODELS:
                self._host._push_view(WorkbenchView.MODELS)
            self._host._models_list_open = True
            self._host._view = WorkbenchView.MODELS
            self._host._refresh_models()
            self._host._activate_view(WorkbenchView.MODELS, focus_id="wb-models")
        elif key.startswith("collision:"):
            wire = key.partition(":")[2]
            item = state.pending[wire]
            outcome = match_discovered_model(state.catalog, state.provider_id, wire)
            if outcome.kind == "multiple_matches":
                self._host._message = f"{wire} already matches multiple deployments; cannot resolve safely."
            elif outcome.kind == "base_exists_other_provider" and item.decision is None:
                self._host._confirm = f"existing:{wire}"
            else:
                self._host._editing = f"canonical:{wire}"
                editor = self._host.query_one("#wb-input", Input)
                editor.password = False
                editor.value = item.canonical_name
                self._host.query_one("#wb-editor-label", Label).update(
                    "Separate Canonical Name"
                )
                self._host._sync_view()
                self._host._restore_view_focus("wb-input")
        elif key == "apply":
            self._host._apply(state)
        elif key == "discard":
            if state.dirty:
                self._host._confirm = "discard"
            else:
                self.expand(state.provider_id)
        self._host._update_help()

    def field_error(self, reason: str) -> None:
        self._host.query_one("#wb-input", Input).add_class("-invalid")
        error = self._host.query_one("#wb-field-error", NoMarkupStatic)
        error.add_class("has-error")
        error.update(f"Error: {reason}")
        self._host._set_feedback("field", reason, "error")

    def accept_connection_form_field(self, input_widget: Input) -> None:
        if self._host._add is None:
            return
        input_id = input_widget.id or ""
        field = {
            "wb-connection-name": "name",
            "wb-connection-base": "api_base",
            "wb-connection-env": "api_key_env_var",
        }.get(input_id)
        if field:
            self._host._add = replace(
                self._host._add, **{field: input_widget.value.strip()}
            )
            if field == "api_key_env_var":
                self.refresh_connection_form()
            next_widget: Input | WorkbenchList
            if input_id == "wb-connection-name":
                next_widget = self._host.query_one("#wb-connection-base", Input)
            elif input_id == "wb-connection-base":
                next_widget = self._host.query_one(
                    "#wb-connection-style", WorkbenchList
                )
            elif self._host._add.api_key_env_var:
                next_widget = self._host.query_one("#wb-connection-key", Input)
            else:
                next_widget = self._host.query_one(
                    "#wb-connection-actions", WorkbenchList
                )
            self._host.set_focus(next_widget)
        elif input_widget.id == "wb-connection-key":
            actions = self._host.query_one("#wb-connection-actions", WorkbenchList)
            if self._host._add.api_key_env_var:
                actions.highlighted = 0
            self._host.set_focus(actions)

    def accept_add_connection(self, key: str) -> None:
        assert self._host._add is not None
        editor = self._host.query_one("#wb-input", Input)
        value = editor.value.strip()
        try:
            draft = replace(self._host._add, **{self._host.FIELDS.get(key, key): value})
            if key == "name":
                self.new_provider_id(value)
            if key in self._host.FIELDS and key != "base":
                ProviderDefinition.model_validate({
                    "api_base": draft.api_base or "https://placeholder.invalid",
                    "api_style": draft.api_style,
                    "api_key_env_var": draft.api_key_env_var,
                })
            if key == "base" and value:
                ProviderDefinition.model_validate({"api_base": value})
        except ValueError as exc:
            self.field_error(f"Invalid {key}: {exc}")
            return
        self._host._add = draft
        if key == "env" and self._host._key_after_env:
            if not value:
                self.field_error("Enter an environment variable name for the API key.")
                return
            self._host._key_after_env = False
            self._host._editing = None
            self.refresh_add_connection(highlight="key")
            self.connection_action("key")
            return
        self._host._set_feedback("field", "", "info")
        self._host._editing = None
        editor.value = ""
        editor.remove_class("-invalid")
        self._host._sync_view()
        self.refresh_add_connection(highlight=key)
        self._host._return_from_editor()

    def accept_manual_model(self, state: ManagementState) -> None:
        editor = self._host.query_one("#wb-input", Input)
        wire = editor.value.strip()
        if not wire or "@" in wire or "/" in wire:
            self.field_error("Enter a model name without @ or /.")
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
        self._host._flow.remember_manual_model(state, wire)
        self._host._set_feedback("field", "", "info")
        self._host._editing = None
        editor.value = ""
        self._host._sync_view()
        self._host._refresh_models(highlight=wire)
        self.refresh_add_models(highlight="manual")
        self._host._return_from_editor()

    def accept_editor(self) -> None:  # noqa: PLR0911
        if self._host._busy:
            return
        state = self._host.state
        key = self._host._editing
        if key is None:
            return
        if key.startswith("detail:"):
            field = key.removeprefix("detail:")
            widget_id = (
                f"#price-{field}"
                if field in {"input", "output", "cached_input"}
                else f"#detail-{field}"
            )
            target = self._host.query_one(widget_id, Input)
            target.value = self._host.query_one("#wb-input", Input).value.strip()
            target.remove_class("-invalid")
            error_id = (
                f"#error-{field}"
                if field in {"input", "output", "cached_input"}
                else f"#detail-error-{field}"
            )
            error = self._host.query_one(error_id, NoMarkupStatic)
            error.update("")
            error.display = False
            self._host._editing = None
            self._host._detail_edit_row = None
            self._host._refresh_detail_fields(highlight=field)
            self._host._activate_view(
                WorkbenchView.DETAIL, focus_id="wb-detail-fields", selected_id=field
            )
            self._host._return_from_editor()
            return
        if (
            self._host._stage == "connection"
            and self._host._add is not None
            and key != "key"
        ):
            self.accept_add_connection(key)
            return
        if key == "manual" and state is not None:
            self.accept_manual_model(state)
            return
        editor = self._host.query_one("#wb-input", Input)
        value = editor.value.strip()
        if key == "key":
            self.accept_credential(value, state)
            return
        if state is None:
            return
        if key.startswith("canonical:"):
            wire = key.partition(":")[2]
            item = state.pending[wire]
            state.pending[wire] = replace(item, canonical_name=value)
            self._host._refresh_models(highlight=value)
        else:
            try:
                candidate = replace(state.connection, **{self._host.FIELDS[key]: value})
                ProviderDefinition.model_validate({
                    **self._host.snapshot.catalog.providers[
                        state.provider_id
                    ].model_dump(),
                    "api_base": candidate.api_base,
                    "api_style": candidate.api_style,
                    "api_key_env_var": candidate.api_key_env_var,
                })
            except ValueError as exc:
                self.field_error(f"Invalid {key}: {exc}")
                return
            state.set_connection(candidate)
        self._host._set_feedback("field", "", "info")
        self._host._editing = None
        editor.value = ""
        editor.remove_class("-invalid")
        self._host._sync_view()
        self._host._refresh_actions()
        self._host._refresh_browser()
        self._host._return_from_editor()

    def accept_credential(self, value: str, state: ManagementState | None) -> None:
        if not value:
            self.field_error("Enter a key before saving.")
            return
        env = (
            self._host._add.api_key_env_var
            if self._host._stage and self._host._add
            else state.connection.api_key_env_var
            if state
            else ""
        )
        if env and (
            env in self._host._saved_credential_envs
            or self._host.credential_resolver(env)
        ):
            self._host._confirm = "shared-key"
            self._host._update_help()
            return
        self.save_key(value)

    def persisted_credential_value(self, env_var: str) -> str | None:
        if env_var in self._host._session_only_credential_envs:
            return None
        return self._host._credential_value(env_var)

    def bound_roster_identities(self, snapshot: CatalogSnapshot) -> frozenset[str]:
        """Distinct canonical identities the snapshot's slots would bind."""
        return bind_snapshot_policy(
            snapshot, allowed_models=self._host.allowed_models
        ).bound_canonical_identities

    def notify_models_saved(
        self, previously_usable: frozenset[str], roster_before: frozenset[str]
    ) -> None:
        if self._host.on_models_saved is not None:
            usable = usable_canonical_models(
                self._host.snapshot,
                self.persisted_credential_value,
                allowed_models=self._host.allowed_models,
                thinking_overrides=self._host.thinking_overrides,
            )
            self._host.on_models_saved(
                roster_before,
                self.bound_roster_identities(self._host.snapshot),
                usable - previously_usable,
            )

    def save_key(self, value: str) -> None:
        previously_usable = usable_canonical_models(
            self._host.snapshot,
            self.persisted_credential_value,
            allowed_models=self._host.allowed_models,
            thinking_overrides=self._host.thinking_overrides,
        )
        roster_before = self.bound_roster_identities(self._host.snapshot)
        state = self._host.state
        env = (
            self._host._add.api_key_env_var
            if self._host._stage and self._host._add
            else state.connection.api_key_env_var
            if state
            else ""
        )
        if not env:
            return
        result = self._host.credentials.save_key(env, value)
        if result.status != "invalid_env_var":
            self._host._saved_credential_envs.add(env)
        if result.status == "session_only":
            self._host._session_only_credential_envs.add(env)
        if result.status == "saved":
            self._host._session_only_credential_envs.discard(env)
            self.notify_models_saved(previously_usable, roster_before)
        self._host._unresolved.pop("field", None)
        self._host._confirm = None
        self._host._editing = None
        editor = self._host.query_one("#wb-input", Input)
        credential_input = self._host.query_one(
            f"#{self._host._credential_input_id}", Input
        )
        editor.value = ""
        self._host._sync_view()
        if self._host._stage == "connection":
            self.refresh_add_connection(highlight="key")
            self._host.set_focus(self._host.query_one("#wb-actions", WorkbenchList))
        else:
            self._host.set_focus(
                self._host.query_one("#wb-actions", NavigableOptionList)
            )
        feedback_kind = cast(
            Literal["running", "success", "error", "warning", "info"],
            {"saved": "success", "session_only": "warning", "invalid_env_var": "error"}[
                result.status
            ],
        )
        self._host._set_feedback(
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
            self._host._changed = True
            if self._host._stage == "models" and state:
                state.begin_discovery()
        if self._host._stage == "connection":
            self.refresh_connection_actions()
        elif self._host._stage == "models":
            self.refresh_add_models(highlight="edit-key")
        self._host._refresh_browser()
        self._host._return_from_editor()
        self._host._update_help()

    @staticmethod
    def is_chat_model(item: DiscoveryItem) -> bool:
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

    def chat_discovery(
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
            if item.wire_id and self.is_chat_model(item):
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

    def discovery_request(
        self, state: ManagementState
    ) -> partial[Coroutine[None, None, None]]:
        provider_id, generation = state.begin_discovery()
        return partial(
            self.discover,
            state,
            request=(
                provider_id,
                generation,
                self._host._stage,
                deepcopy(state.connection),
                self._host._add.name
                if self._host._stage == "models" and self._host._add
                else self._host._friendly(provider_id),
            ),
        )

    async def discover(
        self,
        state: ManagementState,
        *,
        request: tuple[str, int, str | None, ConnectionDraft, str] | None = None,
    ) -> None:
        if (
            self._host._unmounted
            or self._host._dismissed
            or not self._host.is_mounted
            or self._host._busy
        ):
            return
        if request is None:
            if self._host.state is not state:
                return
            provider_id, generation = state.begin_discovery()
            stage, connection = self._host._stage, deepcopy(state.connection)
            display_name = (
                self._host._add.name
                if stage == "models" and self._host._add
                else self._host._friendly(provider_id)
            )
        else:
            provider_id, generation, stage, connection, display_name = request
        running_message = (
            f"Discovering models for {provider_id}{chrome_glyph('running')}"
        )
        self._host._set_feedback("discovery", running_message, "running")
        feedback_token = self._host._feedback_token
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
            result = await self._host.discovery_service(
                provider,
                self._host.credential_resolver(connection.api_key_env_var)
                if connection.api_key_env_var
                else None,
                self._host.tls,
            )
        except Exception:
            result = DiscoveryError(
                "connection",
                f"Discovery failed for {provider_id}; retry or check the connection.",
            )
        if isinstance(result, DiscoveryResult):
            result = self.chat_discovery(state, result, provider_id=provider_id)
        rejected = (
            self._host._stage != stage
            or self._host._unmounted
            or not self._host.is_mounted
        )
        if not rejected:
            rejected = (
                self._host._busy
                or self._host._dismissed
                or self._host.state is not state
                or state.provider_id != provider_id
                or not state.accept_discovery(provider_id, generation, result)
            )
        if rejected:
            if (
                self._host.is_mounted
                and not self._host._unmounted
                and self._host._feedback_token == feedback_token
                and self._host._message == running_message
            ):
                self._host._set_feedback("discovery", "", "info")
                self._host._update_help()
            return
        self._host._set_feedback(
            "discovery",
            f"Discovery Failed for {provider_id}: " + result.message
            if isinstance(result, DiscoveryError)
            else f"Discovered {len(result.models)} models for {provider_id}; inference not tested.",
            "error" if isinstance(result, DiscoveryError) else "info",
        )
        self._host._refresh_browser()
        self._host._refresh_actions()
        if self._host._models_list_open or self._host.query_one("#wb-models").display:
            self._host._refresh_models()
        self._host._update_help()

    def input_submitted_body(self, event: Input.Submitted) -> None:
        if not self._host._can_activate(event.input.id):
            return
        if event.input.id == "wb-input":
            self.accept_editor()
        elif event.input.id in {
            "wb-connection-name",
            "wb-connection-base",
            "wb-connection-env",
            "wb-connection-key",
        }:
            self.accept_connection_form_field(event.input)
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
            self._host.set_focus(
                self._host.query_one(f"#{fields[index + 1]}", Input)
                if index + 1 < len(fields)
                else self._host.query_one("#wb-detail-actions", WorkbenchList)
            )

    def input_changed_body(self, event: Input.Changed) -> None:
        if event.input.id == "wb-input":
            event.input.remove_class("-invalid")
            error = self._host.query_one("#wb-field-error", NoMarkupStatic)
            error.remove_class("has-error")
            error.update("")
        elif event.input.id in {
            "detail-thinking",
            "detail-temperature",
            "detail-auto_compact_threshold",
        }:
            event.input.remove_class("-invalid")
            field = event.input.id.removeprefix("detail-")
            error = self._host.query_one(f"#detail-error-{field}", NoMarkupStatic)
            error.update("")
            error.display = False
        elif (
            event.input.id
            in {"wb-connection-name", "wb-connection-base", "wb-connection-env"}
            and self._host._add
        ):
            field = {
                "wb-connection-name": "name",
                "wb-connection-base": "api_base",
                "wb-connection-env": "api_key_env_var",
            }[event.input.id]
            self._host._add = replace(self._host._add, **{field: event.input.value})
            if event.input.id == "wb-connection-env":
                self.refresh_connection_actions()
        elif event.input.id == "wb-connection-key" and self._host._add:
            self.refresh_connection_actions()

    def selection_highlighted_body(
        self, event: SelectionList.SelectionHighlighted
    ) -> None:
        self._host._update_help()

    def selection_changed_body(self, event: SelectionList.SelectedChanged) -> None:
        if self._host._busy or self._host._model_sync or self._host.state is None:
            return
        if event.selection_list.id == "wb-models" and (
            not self._host._can_activate("wb-models")
            or self._host._group_contents_context.get("wb-models")
            != (
                WorkbenchView.MODELS,
                self._host._group_context(WorkbenchView.MODELS, "wb-models"),
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
        state = self._host.state
        selected = set(event.selection_list.selected)
        rows = state.model_rows()
        if all((name in selected) == enabled for name, _wire, enabled, _found in rows):
            return  # SelectionList emits queued events for programmatic row refreshes.
        original_enabled = dict(state.enabled)
        original_pending = dict(state.pending)
        for name, wire, enabled, _found in rows:
            if (name in selected) == enabled:
                continue
            self._host._remember_onboarding_model_selection(name, selected)
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
            self._host._set_feedback("models", "; ".join(validation.errors), "error")
            state.enabled = original_enabled
            state.pending = original_pending
            models = self._host.query_one("#wb-models", ModelChecklist)
            self._host._model_sync = True
            for name, _wire, enabled, _found in rows:
                if enabled:
                    models.select(name)
                else:
                    models.deselect(name)
            self._host._model_sync = False
        else:
            self._host._set_feedback(
                "models",
                "; ".join(validation.warnings),
                "warning" if validation.warnings else "info",
            )
        self._host._refresh_actions()
        self._host._refresh_browser()
        self._host._update_help()

    def leave_add(self) -> None:
        if self._host.state:
            self._host.state.begin_discovery()
        self._host.state = self._host._pre_add_state
        self._host._pre_add_state = None
        self._host._add_state = None
        self._host._add = None
        self._host._add_checkpoint = None
        self._host._stage = None
        self._host._models_list_open = False
        self._host._provider_open = False
        self._host._navigation.clear()
        self._host._view = WorkbenchView.PROVIDERS
        self._host._activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
        self._host._restore_transient_opener(self._host._add_opener)
        self._host._add_opener = None
        self._host._update_help()

    def back_add(self) -> None:
        if self._host._stage == "models":
            if self._host.state:
                self.begin_saved_connection_edit(self._host.state)
        elif self._host._stage == "connection":
            self._host._view = WorkbenchView.CONNECTION
            draft = self._host._add
            checkpoint = self._host._add_checkpoint
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
                self._host._confirm = (
                    "add-discard"
                    if self._host._add_state is not None
                    else "add-discard-simple"
                )
            else:
                self.leave_add()
        elif self._host._stage == "choose":
            self.leave_add()
