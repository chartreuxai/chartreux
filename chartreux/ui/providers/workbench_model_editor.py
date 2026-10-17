"""Model editor controller for the provider workbench with an injected host."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

from textual.widgets import Input, Label
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

from chartreux.core.model_catalog.contracts import (
    DiscoveryResult,
    ModelEdits,
    OptionalEdit,
)
from chartreux.core.model_catalog.matching import match_discovered_model
from chartreux.core.model_catalog.schema import (
    BaseModelDefinition,
    DeploymentDefinition,
)
from chartreux.ui.providers.management_state import ManagementState, PendingModel
from chartreux.ui.providers.workbench_types import WorkbenchView
from chartreux.ui.providers.workbench_widgets import ModelChecklist, WorkbenchList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class WorkbenchModelEditor:
    """Model detail editor for the provider workbench.

    Plain object (no mixins): the screen constructs it in ``__init__`` with
    ``host=self`` and is its late-bound host — every collaboration is
    ``self._host._method(...)`` at call time, never a constructor-captured
    bound method. The detail state (``_detail*``, ``_editing``,
    ``_detail_edit_row``) stays on the screen; the controller mutates it
    through the host exactly where the screen's own code did.

    Pinned invariants, moved verbatim: ``save_detail``'s six-field
    validation-and-OptionalEdit-diff block moves as one unit, never split
    per-field; ``open_detail`` keeps the discovered-deployment read-only
    preview (``_detail_preview_wire``) and its no-deployment bail-out
    ordering; ``close_detail`` restores the models/catalog cursor and view
    exactly as before. ``canonical_thinking_baseline`` keeps
    ``@staticmethod`` (the second and last screen staticmethod; the 18.2b
    ``friendly``/``provider_count_label`` and 18.2c ``is_chat_model``
    precedent). The TWP:120-132 ``_save_detail`` setattr dispatch spy stays
    interceptable: ``on_option_list_option_selected`` dispatches through the
    screen attribute into the delegate.
    """

    def __init__(self, *, host: Any) -> None:
        self._host = host

    @staticmethod
    def canonical_thinking_baseline(
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

    def open_detail(self, name: str, *, push: bool = True) -> None:  # noqa: PLR0914, PLR0915 - form setup
        state = self._host.state
        if state is None:
            return
        self._host._detail_preview_wire = None
        if (
            self._host._stage == "models"
            and name not in state.configured()
            and not any(item.canonical_name == name for item in state.pending.values())
            and isinstance(state.discovery, DiscoveryResult)
            and any(item.wire_id == name for item in state.discovery.models)
        ):
            # Opening a discovered deployment is a read-only preview until Save.
            self._host._detail_preview_wire = name
        if (
            self._host._detail_preview_wire is None
            and name not in state.configured()
            and not any(item.canonical_name == name for item in state.pending.values())
        ):
            self._host._message = "Enable this discovered model before editing details."
            self._host._update_help()
            return
        self._host._detail = name
        if push and self._host._view != WorkbenchView.DETAIL:
            self._host._push_view(WorkbenchView.DETAIL)
        self._host._view = WorkbenchView.DETAIL
        models = self._host.query_one("#wb-models", ModelChecklist)
        self._host._detail_cursor = name
        self._host._detail_scroll = models.scroll_offset.y
        item = next(
            (item for item in state.pending.values() if item.canonical_name == name),
            None,
        )
        if item is None and self._host._detail_preview_wire is not None:
            item = PendingModel(self._host._detail_preview_wire, name, False)
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
            self._host._message = (
                "This model has no deployment in the selected provider."
            )
            self._host._detail = None
            self._host._pop_view()
            self._host._update_help()
            return
        self._host.query_one("#wb-detail-model-id", NoMarkupStatic).update(
            f"Model: {name} · Deployment: {state.provider_id}/{deployment.name}"
        )
        self._host.query_one("#wb-detail-deployment-id").display = False
        self._host.query_one("#wb-image-support-label").display = False
        supported = deployment.supported_thinking_levels
        self._host.query_one("#wb-detail-supported-thinking", NoMarkupStatic).update(
            "Supported thinking levels: "
            + (", ".join(supported) if supported is not None else "provider default")
        )
        thinking_baseline = self.canonical_thinking_baseline(state, name, item)
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
            widget = self._host.query_one(f"#detail-{field}", Input)
            widget.remove_class("-invalid")
            widget.value = "" if value is None else str(value)
            error = self._host.query_one(f"#detail-error-{field}", NoMarkupStatic)
            error.update("")
            error.display = False
        images = self._host.query_one("#wb-image-support", ModelChecklist)
        self._host._model_sync = True
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
        self._host._model_sync = False
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
            self._host.query_one(f"#price-{field}", Input).remove_class("-invalid")
            error = self._host.query_one(f"#error-{field}", NoMarkupStatic)
            error.update("")
            error.display = False
            self._host.query_one(f"#price-{field}", Input).value = (
                "" if value is None else str(value)
            )
        self._host._detail_original = self.detail_form_snapshot()
        detail_actions = self._host.query_one("#wb-detail-actions", WorkbenchList)
        action_position = self._host._position_before_rebuild(
            WorkbenchView.DETAIL, "wb-detail-actions"
        )
        detail_actions.clear_options()
        detail_actions.add_option(
            Option("Accept model edits into catalog draft", id="save-detail")
        )
        detail_actions.highlighted = 0
        self._host._restore_group_position(action_position)
        self.refresh_detail_fields()
        self._host._activate_view(WorkbenchView.DETAIL, focus_id="wb-detail-fields")
        self._host._update_help()

    def refresh_detail_fields(self, *, highlight: str | None = None) -> None:
        rows = self._host.query_one("#wb-detail-fields", WorkbenchList)
        position = self._host._position_before_rebuild(
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
            widget = self._host.query_one(widget_id, Input)
            value = widget.value or "Not set"
            status = "Error: " if widget.has_class("-invalid") else ""
            rows.add_option(
                Option(self._host._summary(f"{status}{label}  {value}"), id=key)
            )
        images = self._host.query_one("#wb-image-support", ModelChecklist)
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
            self._host._restore_group_position(position)

    def toggle_detail_images(self) -> None:
        images = self._host.query_one("#wb-image-support", ModelChecklist)
        if "supports-images" in images.selected:
            images.deselect("supports-images")
        else:
            images.select("supports-images")
        self.refresh_detail_fields(highlight="images")
        self._host.query_one("#wb-detail-fields", WorkbenchList).focus()

    def select_detail_field(self, key: str) -> None:
        if key == "save-detail":
            self.save_detail()
            return
        if key == "images":
            self.toggle_detail_images()
            return
        widget_id = (
            f"#price-{key}"
            if key in {"input", "output", "cached_input"}
            else f"#detail-{key}"
        )
        widget = self._host.query_one(widget_id, Input)
        self._host._detail_edit_row = key
        self._host._editing = f"detail:{key}"
        editor = self._host.query_one("#wb-input", Input)
        editor.password = False
        editor.value = widget.value
        self._host.query_one("#wb-editor-label", Label).update(
            key.replace("_", " ").title()
        )
        self._host._update_help()
        editor.focus()

    def detail_form_snapshot(self) -> tuple[object, ...]:
        fields = (
            "thinking",
            "temperature",
            "auto_compact_threshold",
            "input",
            "output",
            "cached_input",
        )
        values = tuple(
            self._host.query_one(f"#detail-{field}", Input).value
            for field in fields[:3]
        )
        prices = tuple(
            self._host.query_one(f"#price-{field}", Input).value for field in fields[3:]
        )
        images = tuple(
            sorted(self._host.query_one("#wb-image-support", ModelChecklist).selected)
        )
        return (*values, *prices, images)

    def save_detail(self) -> None:  # noqa: PLR0911, PLR0912, PLR0914, PLR0915 - validation and draft save
        if self._host._busy:
            return
        state = self._host.state
        name = self._host._detail
        if state is None or name is None:
            return
        item = next(
            (item for item in state.pending.values() if item.canonical_name == name),
            None,
        )
        if item is None and self._host._detail_preview_wire is not None:
            item = PendingModel(self._host._detail_preview_wire, name, False)
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
            self.set_detail_error(
                "thinking", "This model has no deployment in the selected provider."
            )
            return

        thinking_widget = self._host.query_one("#detail-thinking", Input)
        thinking = thinking_widget.value.strip()
        if not thinking:
            self.set_detail_error("thinking", "Thinking level cannot be blank.")
            return
        temperature_widget = self._host.query_one("#detail-temperature", Input)
        temperature_raw = temperature_widget.value.strip()
        try:
            temperature = float(temperature_raw) if temperature_raw else None
            if temperature is not None and not isfinite(temperature):
                raise ValueError
        except ValueError:
            self.set_detail_error("temperature", "Use a finite number or leave blank.")
            return
        threshold_widget = self._host.query_one("#detail-auto_compact_threshold", Input)
        threshold_raw = threshold_widget.value.strip()
        try:
            threshold = float(threshold_raw) if threshold_raw else None
            if threshold is not None and (not isfinite(threshold) or threshold <= 0):
                raise ValueError
        except ValueError:
            self.set_detail_error(
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
            self.set_detail_error("thinking", str(exc))
            return
        for field in ("thinking", "temperature", "auto_compact_threshold"):
            widget = self._host.query_one(f"#detail-{field}", Input)
            widget.remove_class("-invalid")
            error = self._host.query_one(f"#detail-error-{field}", NoMarkupStatic)
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
            else self.canonical_thinking_baseline(state, name, item)
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
            in self._host.query_one("#wb-image-support", ModelChecklist).selected
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
            widget = self._host.query_one(f"#price-{field}", Input)
            raw = widget.value.strip()
            try:
                value = float(raw) if raw else None
                if value is not None and (not isfinite(value) or value < 0):
                    raise ValueError
            except ValueError:
                widget.add_class("-invalid")
                error = self._host.query_one(f"#error-{field}", NoMarkupStatic)
                error.update("Error: use a non-negative finite number.")
                error.display = True
                self.refresh_detail_fields(highlight=field)
                self._host.query_one("#wb-detail-fields", WorkbenchList).focus()
                self._host._set_feedback(
                    "detail", f"{field} price must be non-negative and finite.", "error"
                )
                self._host._update_help()
                return
            widget.remove_class("-invalid")
            error = self._host.query_one(f"#error-{field}", NoMarkupStatic)
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
        if self._host._feedback_kind not in {"error", "warning"}:
            self._host._message = ""
        self.close_detail()
        self._host._refresh_actions()
        self._host._refresh_browser()

    def set_detail_error(self, field: str, message: str) -> None:
        widget = self._host.query_one(f"#detail-{field}", Input)
        widget.add_class("-invalid")
        error = self._host.query_one(f"#detail-error-{field}", NoMarkupStatic)
        error.update(f"Error: {message}")
        error.display = True
        self.refresh_detail_fields(highlight=field)
        self._host.query_one("#wb-detail-fields", WorkbenchList).focus()
        self._host._set_feedback("detail", message, "error")
        self._host._update_help()

    def detail_is_modified(self) -> bool:
        if self._host._detail_original is None or not self._host._detail:
            return False
        return self.detail_form_snapshot() != self._host._detail_original

    def close_detail(self) -> None:
        self._host._detail = None
        self._host._detail_edit_row = None
        self._host._detail_preview_wire = None
        self._host._detail_original = None
        if self._host._pop_view():
            return
        self._host._refresh_models(highlight=self._host._detail_cursor)
        models = self._host.query_one("#wb-models", ModelChecklist)
        models.scroll_y = self._host._detail_scroll
        if self._host._models_view:
            self._host._refresh_catalog()
            self._host._activate_view(WorkbenchView.CATALOG, focus_id="wb-catalog")
        else:
            self._host._activate_view(
                WorkbenchView.MODELS
                if self._host._models_list_open
                else WorkbenchView.ACTIONS
            )
        self._host._update_help()
