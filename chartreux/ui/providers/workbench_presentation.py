"""Presentation helpers for the provider workbench with an injected host."""

from __future__ import annotations

from typing import Any, Literal, cast

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.color import Color
from textual.content import Content
from textual.widgets import OptionList
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.providers.workbench_types import WorkbenchView
from chartreux.ui.providers.workbench_widgets import ModelChecklist, WorkbenchList
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class WorkbenchPresentation:
    """Help, hint, and visibility rendering; all state stays on the host."""

    def __init__(self, *, host: Any) -> None:
        self._host = host

    def update_help(self) -> None:  # noqa: PLR0912, PLR0914, PLR0915
        if not self._host.is_mounted or self._host._unmounted:
            return
        primary = self.sync_view()
        state = self._host.state
        if self._host._busy:
            description = (
                "Saving; wait to close"
                if "Saving" in self._host._message
                else "Operation in progress; wait to close"
            )
        elif self._host._confirm:
            if self._host._confirm == "add-discard":
                description = "Choose Keep editing, Continue setup, or Discard edits."
            elif self._host._confirm == "detail-discard":
                description = "Choose Keep editing, Accept model edits into the catalog draft, or Discard model edits."
            else:
                description = "Choose Cancel to keep the current state."
        elif self._host._editing:
            description = self._host.DESCRIPTIONS.get(
                self._host._editing, "Enter accepts this field into the draft."
            )
        elif self._host._detail:
            save_destination = (
                "Save and continue to presets persists the draft."
                if self._host._stage == "models"
                else "Apply all catalog edits to disk later."
            )
            description = (
                f"{self._host._detail}: {chrome_glyph('vertical')} move; Enter edits a field; "
                "Space toggles images. Tab moves to acceptance actions. Accept model edits returns to the opener; "
                f"{save_destination}"
            )
        elif primary == "models" and state:
            if self._host.query_one("#wb-models-actions").has_focus:
                action = self._host.query_one(
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
                models = self._host.query_one("#wb-models", ModelChecklist)
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
                self._host._help_focus_id == "wb-catalog-filter"
                or self._host.query_one("#wb-catalog-filter").has_focus
            )
            if filter_focused:
                description = (
                    f"Provider filter. {chrome_glyph('vertical')} choose a provider; Tab moves to models; "
                    + (
                        "Esc clears this filter first."
                        if self._host._model_filter
                        else "Esc goes back."
                    )
                )
                option = None
            else:
                group = (
                    "#wb-catalog-actions"
                    if self._host._help_focus_id == "wb-catalog-actions"
                    or self._host.query_one("#wb-catalog-actions").has_focus
                    else "#wb-catalog"
                )
                option = self._host.query_one(
                    group, NavigableOptionList
                ).highlighted_option
            catalog = state.catalog if state else self._host.snapshot.catalog
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
                    else self._host.DESCRIPTIONS[str(option.id).removeprefix("\x00")]
                    if option and option.id in {"\x00apply", "\x00discard"}
                    else "Select a provider filter or a model to inspect."
                )
            )
        elif primary == "picker":
            picker = self._host.query_one("#wb-picker", NavigableOptionList)
            option = picker.highlighted_option
            description = (
                f"Enter selects {option.id} for {self._host._preset_role}; Esc returns to presets."
                if self._host._preset_field and option and option.id
                else "Enter opens the selected deployment's model details."
                if self._host._view == WorkbenchView.DEPLOYMENTS
                else "No models configured."
            )
        elif primary == "actions" and (state or self._host._add):
            group = self._host._help_focus_id
            if group not in {
                "wb-actions",
                "wb-provider-operations",
                "wb-connection-actions",
            }:
                group = "wb-actions"
            option = self._host.query_one(
                f"#{group}", NavigableOptionList
            ).highlighted_option
            key = str(option.id) if option and option.id else ""
            description = self._host.DESCRIPTIONS.get(
                key, "Enter accepts fields into the catalog draft; only Apply saves it."
            )
            if key in {"name", "base", "style", "env"}:
                connection = (
                    self._host._add
                    if self._host._stage == "connection" and self._host._add
                    else state.connection
                    if state
                    else None
                )
                value = (
                    self._host._add.name
                    if key == "name"
                    and self._host._stage == "connection"
                    and self._host._add
                    else state.provider_id
                    if key == "name" and state
                    else getattr(connection, self._host.FIELDS[key], None)
                )
                display_value = value or ("Not required" if key == "env" else "Not set")
                description = f"{display_value}. {description}"
        elif primary == "providers":
            option = self.root_option()
            catalog = (
                self._host.state.catalog
                if self._host.state
                else self._host.snapshot.catalog
            )
            if option and option.id in catalog.providers:
                action = (
                    "sets up connection and deployments"
                    if self._host._mistral_needs_setup(str(option.id))
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
        if self._host._help_open:
            description = f"Shortcuts: {chrome_glyph('vertical')} move; Space toggles a deployment; Enter opens or accepts; Esc returns; F1 closes help."
        feedback = self.feedback()
        help_text = Text()
        if feedback:
            help_text.append_text(feedback)
        if description and not self._host._busy:
            if feedback:
                help_text.append("  ")
            help_text.append(description)
        self._host.query_one("#wb-help", NoMarkupStatic).update(help_text)
        model_option = (
            self._host.query_one("#wb-models", ModelChecklist).highlighted_option
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
            if self._host.query_one("#wb-models-actions").has_focus
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
            self._host.query_one("#wb-picker", NavigableOptionList).highlighted_option
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
                ("Esc", "Clear filter" if self._host._model_filter else "Back"),
            ]
            if self._host._help_focus_id == "wb-catalog-filter"
            or self._host.query_one("#wb-catalog-filter").has_focus
            else [
                ("↑↓", "Move"),
                ("Tab", "Provider filter"),
                ("Shift+Tab", "Provider filter"),
                (
                    "Enter",
                    "Select"
                    if self._host.query_one("#wb-catalog-actions").has_focus
                    else "Edit",
                ),
                ("Esc", "Back"),
            ]
        )
        groups = self._host._focus_groups()
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
            current = self._host.focused.id if self._host.focused else None
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
            self._host.query_one("#wb-help", NoMarkupStatic).update(help_text)

        detail_bindings = [
            ("↑↓", "Move"),
            ("Space", "Toggle images"),
            (
                "Enter",
                "Accept edits"
                if self._host.query_one("#wb-detail-actions").has_focus
                else "Edit field",
            ),
            ("Esc", "Back"),
        ]
        bindings = (
            [("Esc", "Wait to close")]
            if self._host._busy
            else [("Enter", "Select"), ("Esc", "Cancel")]
            if self._host._confirm
            else [("Space", "Select"), ("Enter", "Accept field"), ("Esc", "Back")]
            if self._host._protocol_picker
            else [
                (
                    "Enter",
                    "Save API key" if self._host._editing == "key" else "Accept field",
                ),
                ("Esc", "Back"),
            ]
            if self._host._editing
            else detail_bindings
            if self._host._detail
            else model_bindings
            if primary == "models"
            else picker_bindings
            if primary == "picker"
            else catalog_bindings
            if primary == "catalog"
            else [
                ("↑↓", "Move"),
                ("Enter", self.root_enter_hint()),
                ("Esc", self.root_escape_hint()),
            ]
            if primary == "providers"
            else [("↑↓", "Move"), ("Enter", "Select"), ("Esc", "Back")]
        )
        if (
            primary
            in {"providers", "actions", "detail-fields", "presets", "preset-editor"}
            and groups
            and not self._host._confirm
            and not self._host._busy
            and not self._host._editing
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
            current = self._host.focused.id if self._host.focused else None
            index = groups.index(current) if current in groups else 0
            traversal = [
                ("Tab", labels[groups[(index + 1) % len(groups)]]),
                ("Shift+Tab", labels[groups[(index - 1) % len(groups)]]),
            ]
            bindings = traversal + bindings
        width = max(1, self._host.size.width - 2)
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
                and not self._host._confirm
                and not self._host._busy
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
        self._host._hint_targets = []
        offset = 0
        for key, label in shown:
            width = hint_content([(key, label)]).cell_length
            self._host._hint_targets.append((offset, offset + width, key))
            offset += width + 2
        self._host.query_one("#wb-hint", NoMarkupStatic).update(hint_content(shown))

    def sync_view(self) -> str:
        if self._host._editing and self._host._editor_opener is None:
            self._host._editor_opener = self._host._frame()
        primary, visible = self.visibility_projection()
        opening_confirmation = bool(
            self._host._confirm and not self._host.query_one("#wb-confirm").display
        )
        if self._host._confirm:
            confirm = self._host.query_one("#wb-confirm")
            confirm.border_title = "Confirm change"
            host = self._host.query_one("#wb-confirm-host")
            width = min(60, max(1, int(self._host.size.width * 0.8)))
            host.styles.offset = (
                max(0, (self._host.size.width - width) // 2),
                max(0, (self._host.size.height - 10) // 2),
            )
            if opening_confirmation:
                self._host._confirm_opener = (
                    self._host.focused.id if self._host.focused else None
                )
                self._host._confirm_position = self._host._capture_group_position()
                message, label = self.confirmation_content()
                self._host.query_one("#wb-confirm-text", NoMarkupStatic).update(message)
                actions = self._host.query_one(
                    "#wb-confirm-actions", NavigableOptionList
                )
                actions.clear_options()
                if self._host._confirm == "detail-discard":
                    actions.add_option(Option("Keep editing", id="cancel"))
                    actions.add_option(
                        Option(
                            "Accept model edits into catalog draft", id="save-detail"
                        )
                    )
                    actions.add_option(
                        Option("Discard model edits", id="discard-detail")
                    )
                elif self._host._confirm == "add-discard":
                    actions.add_option(Option("Keep editing", id="cancel"))
                    actions.add_option(Option("Continue setup", id="continue"))
                else:
                    actions.add_option(Option("Keep editing", id="cancel"))
                if self._host._confirm != "detail-discard":
                    actions.add_option(
                        Option(
                            f"! {label}"
                            if label.startswith(("Discard", "Replace"))
                            else label,
                            id="accept",
                        )
                    )
                actions.highlighted = 0
        self.apply_visibility(visible)
        if opening_confirmation:
            self._host._restore_view_focus("wb-confirm-actions")
        context = (
            "Unsaved provider draft"
            if self._host._confirm in {"add-discard", "add-discard-simple"}
            else "Unsaved model edits"
            if self._host._confirm == "detail-discard"
            else "Confirm change"
            if self._host._confirm
            else self.detail_context()
            if self._host._detail
            else f"Edit {self._host._editing} for {self._host.state.provider_id if self._host.state else self._host._add.name if self._host._add else 'provider'}"
            if self._host._editing
            else f"API style for {self._host.state.provider_id if self._host.state else self._host._add.name if self._host._add else 'provider'}"
            if self._host._protocol_picker
            else "Choose deployment to edit"
            if self._host._view == WorkbenchView.DEPLOYMENTS
            else f"Choose {self._host._preset_field} for {self._host._preset_role}"
            if self._host._preset_field and self._host._preset_role
            else f"Edit {self._host._preset_role.replace('main', 'Main assistant').title()} preset"
            if self._host._view == WorkbenchView.PRESET_EDITOR
            and self._host._preset_role
            else "Choose a default preset"
            if self._host._picker
            else "Choose default presets"
            if self._host._view == WorkbenchView.PRESETS
            else f"Models: {self._host._model_filter or 'all providers'}"
            if self._host._models_view
            else "Choose provider type"
            if self._host._stage == "choose"
            else f"Models for {self._host.state.provider_id} · click/Space toggles inclusion · Enter edits details"
            if self._host._models_list_open and self._host.state
            else f"Provider: {self._host.state.provider_id}"
            if self._host._provider_open and self._host.state
            else f"New provider: {self._host._add.name or 'unnamed'} — {self._host._stage}"
            if self._host._stage and self._host._add
            else (
                f"Filter: {self._host.filter_text or 'type to filter'}  ·  "
                f"{self._host._visible_provider_count()}"
            )
        )
        self._host.query_one("#wb-filter", NoMarkupStatic).update(context)
        self._host.query_one("#wb-count").display = False
        self._host._group_contents_context["wb-help"] = (
            self._host._view,
            self._host._group_context(self._host._view, "wb-help"),
        )
        return primary

    def visibility_projection(self) -> tuple[str, set[str]]:
        """Resolve body, replacement editor, and overlay without changing state."""
        groups = self._host.VIEW_GROUPS[self._host._view]
        if self._host._editing:
            groups = self._host.VIEW_GROUPS[WorkbenchView.EDITOR]
        elif self._host._detail:
            groups = self._host.VIEW_GROUPS[WorkbenchView.DETAIL]
        primary = groups[0]
        visible = set(groups)
        if primary == "models" and self._host._stage != "models":
            visible.discard("models-actions")
        if primary == "catalog" and not (self._host.state and self._host.state.dirty):
            visible.discard("catalog-actions")
        if self._host._confirm:
            visible.add("confirm")
        else:
            visible.update(("help", "hint"))
        return primary, visible

    def apply_visibility(self, visible: set[str]) -> None:
        """The only view-control display writer; pending-action is host-owned."""
        for name in (*self._host.VIEW_CONTROLS, "help", "hint"):
            self._host.query_one(f"#wb-{name}").display = name in visible
        self._host.query_one("#wb-actions").set_class(
            self._host._view == WorkbenchView.ACTIONS, "wb-management-fields"
        )
        self._host.query_one("#wb-confirm-host").display = "confirm" in visible
        for name in self._host.UNDERLAY_CONTROLS:
            widget = self._host.query_one(f"#wb-{name}")
            widget.set_class(
                bool(self._host._confirm and widget.display), "wb-confirm-underlay"
            )

    def set_feedback(
        self,
        operation: str,
        message: str,
        kind: Literal["running", "success", "error", "warning", "info"],
    ) -> None:
        self._host._feedback_token += 1
        if kind in {"error", "warning"}:
            self._host._unresolved[operation] = (message, kind)
        elif kind in {"success", "info"}:
            self._host._unresolved.pop(operation, None)
        if self._host._unresolved and kind not in {"error", "warning"}:
            self._host._message, self._host._feedback_kind = next(
                reversed(self._host._unresolved.values())
            )
        else:
            self._host._message, self._host._feedback_kind = message, kind
        self.update_help()

    def feedback(self) -> Text:
        message = self._host._message
        kind = self._host._feedback_kind
        if self._host._unresolved:
            message, kind = next(reversed(self._host._unresolved.values()))
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
                            self._host.app.theme_variables["primary"]
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
                            self._host.app.theme_variables["success"]
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
                        color=Color.parse(
                            self._host.app.theme_variables["error"]
                        ).rich_color
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
                            self._host.app.theme_variables["warning"]
                        ).rich_color
                    ),
                ),
                f"Warning: {message}",
            )
        return Text(message)

    def confirmation_content(self) -> tuple[str, str]:  # noqa: PLR0911
        decision = self._host._confirm
        state = self._host.state
        provider = (
            state.provider_id
            if state
            else self._host._add.name
            if self._host._add
            else "this provider"
        )
        if decision == "shared-key":
            env = (
                self._host._add.api_key_env_var
                if self._host._stage and self._host._add
                else state.connection.api_key_env_var
                if state
                else ""
            )
            affected = [
                name
                for name, definition in self._host.snapshot.catalog.providers.items()
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
                f"Replace customized connections for {', '.join(self._host._connection_scopes)} and apply catalog changes "
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

    def summary(self, text: str, *, reserve: int = 8) -> str:
        width = max(1, self._host.size.width - reserve)
        if cell_len(text) <= width:
            return text
        result = ""
        for char in text:
            if cell_len(result + char) > width - 1:
                break
            result += char
        return result + chrome_glyph("truncation")

    def detail_context(self) -> str:
        state, name = self._host.state, self._host._detail
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
        return self.summary(
            f"Model details: {name} · {state.provider_id}/{wire}", reserve=4
        )

    def root_option(self) -> Option | None:
        group = (
            "#wb-root-actions"
            if self._host.query_one("#wb-root-actions").has_focus
            or self._host._help_focus_id == "wb-root-actions"
            else "#wb-providers"
        )
        return self._host.query_one(group, OptionList).highlighted_option

    def root_enter_hint(self) -> str:  # noqa: PLR0911
        option = self.root_option()
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
        catalog = (
            self._host.state.catalog
            if self._host.state
            else self._host.snapshot.catalog
        )
        if option is not None and option.id in catalog.providers:
            return (
                "Set up"
                if self._host._mistral_needs_setup(str(option.id))
                else "Manage"
            )
        return "Select"

    def root_escape_hint(self) -> str:
        if self._host.filter_text:
            return "Clear filter"
        return self._host._flow.escape_hint()
