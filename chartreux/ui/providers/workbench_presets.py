"""Presets controller for the provider workbench with an injected host."""

from __future__ import annotations

from typing import Any

from textual.widgets.option_list import Option

from chartreux.core.dispatch.schema import DispatchMode
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.providers.management_state import ManagementState
from chartreux.ui.providers.workbench_types import WorkbenchView
from chartreux.ui.providers.workbench_widgets import WorkbenchList


class WorkbenchPresets:
    """Role preset panes for the provider workbench.

    Plain object (no mixins): the screen constructs it in ``__init__`` with
    ``host=self`` and is its late-bound host — every collaboration is
    ``self._host._method(...)`` at call time, never a constructor-captured
    bound method. The presets state (``_preset_*``, ``_presets_opener``,
    ``_customize_presets``) stays on the screen; the controller mutates it
    through the host exactly where the screen's own code did.

    Pinned invariants, moved verbatim: the ``state.dispatch_mode``/
    ``set_role_preset``/``preset``/``validate*`` calls stay direct on the
    ManagementState argument (the state does not move). The mode decisions
    route through the flow policy (``host._flow``): ``open_presets`` reaches
    the seeding through ``host._seed_onboarding_presets`` (re-homed to the
    policy at 18.3a), and the summary layout, finish label, finish validation,
    and after-commit destination read the policy. The TWP:120-132
    ``_select_preset_action``/``_select_preset_editor`` setattr dispatch
    spies stay interceptable: ``on_option_list_option_selected`` dispatches
    through the screen attributes into the delegates.
    """

    def __init__(self, *, host: Any) -> None:
        self._host = host

    def open_presets(self) -> None:
        # Guided setup advances a stage; management presets are a nested view.
        self._host._presets_opener = (
            self._host._frame() if self._host._stage is None else None
        )
        if self._host.state is None:
            first = next(iter(self._host.snapshot.catalog.providers), None)
            if first is None:
                self._host._set_feedback(
                    "preset", "Connect a provider first.", "warning"
                )
                return
            self._host.state = ManagementState.from_snapshot(self._host.snapshot, first)
        if self._host._flow.is_onboarding:
            self._host._seed_onboarding_presets()
        self._host._stage = self._host._flow.presets_stage()
        self._host._add = None
        self._host._preset_field = None
        self._host._preset_role = None
        self._host._customize_presets = False
        self.refresh_presets()
        self._host._activate_view(WorkbenchView.PRESETS, focus_id="wb-presets")
        self._host._defer_focus(self._host.query_one("#wb-presets", WorkbenchList))

    def refresh_presets(self, *, highlight: str | None = None) -> None:
        state = self._host.state
        if state is None:
            return
        rows = self._host.query_one("#wb-presets", WorkbenchList)
        position = self._host._position_before_rebuild(
            WorkbenchView.PRESETS, "wb-presets"
        )
        previous = highlight or (
            str(rows.highlighted_option.id) if rows.highlighted_option else None
        )
        rows.clear_options()
        rows.add_option(Option("DEFAULT PRESETS", id="heading", disabled=True))
        summary = self._host._flow.presets_summary_layout(self._host._customize_presets)
        if summary:
            models = sorted({state.preset(role)[0] for role in state.catalog.roles})
            rows.add_option(
                Option(
                    self._host._summary(
                        f"Roles bound automatically to {', '.join(models)}"
                        if self._host._onboarding_presets_seeded
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
                    self._host._summary(f"{marker} {label}"),
                    id=f"mode:{dispatch_mode.value}",
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
                    self._host._summary(f"{title}  {model} · thinking {thinking}"),
                    id=f"preset:{role}",
                )
            )
        actions = self._host.query_one("#wb-presets-actions", WorkbenchList)
        action_position = self._host._position_before_rebuild(
            WorkbenchView.PRESETS, "wb-presets-actions"
        )
        actions.clear_options()
        actions.add_option(Option(self._host._flow.presets_finish_label(), id="finish"))
        actions.add_option(Option("Add another provider", id="add-another"))
        actions.highlighted = 0
        self._host._restore_group_position(action_position)
        ids = [str(option.id) for option in rows.options]
        rows.highlighted = (
            ids.index(previous) if previous in ids else (2 if summary else 1)
        )
        if highlight is None:
            self._host._restore_group_position(position)
        self._host._update_help()

    def select_preset_action(self, key: str) -> None:  # noqa: PLR0911, PLR0912
        state = self._host.state
        if state is None or self._host._busy:
            return
        if key == "customize":
            self._host._customize_presets = True
            self.refresh_presets(highlight="preset:orchestrator")
            return
        if key.startswith("mode:"):
            state.dispatch_mode = DispatchMode(key.partition(":")[2])
            self._host._set_feedback(
                "preset", "Mode staged; save to apply next session.", "info"
            )
            self.refresh_presets(highlight=key)
            return
        if key == "add-another":
            if state.dirty:
                self._host._after_commit = "another"
                self._host._start_commit(state, create=False)
            else:
                self._host._start_add()
            return
        if key == "finish":
            if self._host._flow.finish_validates():
                validation = state.validate(
                    mode="onboarding",
                    credential_resolver=self._host._credential_value,
                    allowed_models=self._host.allowed_models,
                    thinking_overrides=self._host.thinking_overrides,
                )
                if validation.errors:
                    self._host._customize_presets = True
                    self._host._set_feedback(
                        "preset", "; ".join(validation.errors), "error"
                    )
                    self.refresh_presets(
                        highlight=f"preset:{(validation.unresolved_roles or validation.unusable_roles or ('orchestrator',))[0]}"
                    )
                    self._host.query_one("#wb-presets").focus()
                    return
            else:
                for role, pair in state.role_presets.items():
                    definition = state.catalog.roles[role]
                    if pair == (definition.model, definition.thinking):
                        continue
                    error = state.validate_preset(
                        role,
                        self._host._credential_value,
                        allowed_models=self._host.allowed_models,
                        thinking_overrides=self._host.thinking_overrides,
                    )
                    if error:
                        self._host._set_feedback("preset", error, "error")
                        self.refresh_presets(highlight=f"preset:{role}")
                        self._host.query_one("#wb-presets").focus()
                        return
            if state.dirty:
                self._host._after_commit = self._host._flow.after_commit_destination(
                    self._host.mode
                )
                self._host._start_commit(state, create=False)
            elif self._host._flow.is_onboarding:
                self._host._setup_ready = True
                self._host._finish()
            else:
                self._host._set_feedback("preset", "Presets are already saved.", "info")
            return
        field, _, role = key.partition(":")
        if field != "preset" or role not in state.catalog.roles:
            return
        self._host._push_view(WorkbenchView.PRESET_EDITOR)
        self._host._preset_role = role
        self._host._preset_draft = state.preset(role)
        self._host._preset_field = None
        self.refresh_preset_editor()
        self._host._activate_view(
            WorkbenchView.PRESET_EDITOR, focus_id="wb-preset-editor"
        )
        self._host._defer_focus(
            self._host.query_one("#wb-preset-editor", WorkbenchList)
        )

    def refresh_preset_editor(self, *, highlight: str | None = None) -> None:
        if self._host._preset_role is None or self._host._preset_draft is None:
            return
        rows = self._host.query_one("#wb-preset-editor", WorkbenchList)
        position = self._host._position_before_rebuild(
            WorkbenchView.PRESET_EDITOR, "wb-preset-editor"
        )
        previous = highlight or (
            str(rows.highlighted_option.id) if rows.highlighted_option else "model"
        )
        model, thinking = self._host._preset_draft
        supported = (
            self._host.state.preset_thinking_levels(model) if self._host.state else ()
        )
        rows.clear_options()
        rows.add_option(Option(self._host._summary(f"Model  {model}"), id="model"))
        rows.add_option(
            Option(
                self._host._summary(
                    f"Thinking  {thinking}{' (unsupported)' if supported and thinking not in supported else ''}"
                ),
                id="thinking",
            )
        )
        actions = self._host.query_one("#wb-preset-editor-actions", WorkbenchList)
        action_position = self._host._position_before_rebuild(
            WorkbenchView.PRESET_EDITOR, "wb-preset-editor-actions"
        )
        actions.clear_options()
        actions.add_option(Option("Apply model and thinking", id="apply"))
        actions.add_option(Option("Cancel", id="cancel"))
        actions.highlighted = 0
        self._host._restore_group_position(action_position)
        ids = [str(option.id) for option in rows.options]
        rows.highlighted = ids.index(previous) if previous in ids else 0
        if highlight is None:
            self._host._restore_group_position(position)

    def select_preset_editor(self, key: str) -> None:
        if (
            self._host.state is None
            or self._host._preset_role is None
            or self._host._preset_draft is None
        ):
            return
        if key == "cancel":
            self.close_preset_editor()
            return
        if key == "apply":
            model, thinking = self._host._preset_draft
            supported = self._host.state.preset_thinking_levels(model)
            if thinking not in supported:
                self._host._set_feedback(
                    "preset",
                    f"@{self._host._preset_role}: {model} supports {', '.join(supported) or 'no thinking levels'}; choose a supported level before Apply.",
                    "error",
                )
                self.refresh_preset_editor(highlight="thinking")
                self._host.query_one("#wb-preset-editor").focus()
                return
            role = self._host._preset_role
            self._host.state.set_role_preset(role, model, thinking)
            self.close_preset_editor()
            return
        if key not in {"model", "thinking"}:
            return
        model, current_thinking = self._host._preset_draft
        supported = (
            self._host.state.preset_thinking_levels(model) if key == "thinking" else ()
        )
        if key == "thinking" and not supported:
            self._host._set_feedback(
                "preset",
                f"@{self._host._preset_role}: {model} has no enabled deployment. Choose a model first.",
                "warning",
            )
            return
        self._host._push_view(WorkbenchView.PICKER)
        self._host._preset_field = key
        position = self._host._position_before_rebuild(
            WorkbenchView.PICKER, "wb-picker"
        )
        picker = self._host.query_one("#wb-picker", WorkbenchList)
        picker.clear_options()
        if key == "model":
            names = set(self._host.state.catalog.models) | {
                item.canonical_name
                for pending in self._host.state.pending_by_provider.values()
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
        self._host._restore_group_position(position)
        self._host._activate_view(WorkbenchView.PICKER, focus_id="wb-picker")
        self._host._defer_focus(picker)

    def close_preset_editor(self) -> None:
        self._host._capture_group_position()
        role = self._host._preset_role
        self._host._preset_field = None
        self._host._preset_draft = None
        self._host._preset_role = None
        self.refresh_presets(highlight=f"preset:{role}")
        if not self._host._pop_view():
            self._host._activate_view(
                WorkbenchView.PRESETS,
                focus_id="wb-presets",
                selected_id=f"preset:{role}",
            )

    def choose_preset_value(self, value: str) -> None:
        if (
            self._host.state is None
            or self._host._preset_role is None
            or self._host._preset_field is None
        ):
            return
        role, field = self._host._preset_role, self._host._preset_field
        model, thinking = self._host._preset_draft or self._host.state.preset(role)
        candidate = value if field == "model" else model
        self._host._preset_draft = (
            candidate,
            value if field == "thinking" else thinking,
        )
        if field == "model":
            supported = self._host.state.preset_thinking_levels(candidate)
            if supported and thinking not in supported:
                self._host._set_feedback(
                    "preset",
                    f"@{role}: {candidate} supports {', '.join(supported)}; choose a supported thinking level before Apply.",
                    "warning",
                )
        self._host._capture_group_position()
        self._host._preset_field = None
        self.refresh_preset_editor(highlight=field)
        if not self._host._pop_view():
            self._host._activate_view(
                WorkbenchView.PRESET_EDITOR,
                focus_id="wb-preset-editor",
                selected_id=field,
            )
