"""Onboarding flow policy for the provider workbench with an injected host."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import project_catalog_changes
from chartreux.ui.providers.management_state import ManagementState


class WorkbenchFlowPolicy:
    """Onboarding-vs-management flow decisions; all state stays on the host.

    Plain object (no mixins): the screen constructs it in ``__init__`` with
    ``host=self`` and is its late-bound host — every collaboration is
    ``self._host._method(...)`` at call time, never a constructor-captured
    bound method. The policy owns the mode DECISIONS only: ``ManagementState``
    stays the screen-owned single catalog draft, and the mode itself stays on
    the screen — tests write ``screen.mode`` after construction, so the policy
    reads ``host.mode`` at decision time rather than capturing the
    construction-time value.

    Pinned invariants, moved verbatim: ``_seed_onboarding_presets`` re-homes
    here as ``seed`` (the seeding body reads ``state._build()``/
    ``project_catalog_changes``/``state._preset_readiness`` — ManagementState
    internals; the state type does not move — and calls
    ``host._credential_value``/``_set_feedback``/``snapshot``/
    ``selected_model``/``allowed_models``/``thinking_overrides`` late-bound),
    and ``_remember_onboarding_model_selection`` re-homes here as
    ``remember_selection``. The 16 ``self.mode`` decision sites convert to
    policy calls; each conversion site keeps its surrounding ordering and
    only the boolean source changes.
    """

    def __init__(self, *, host: Any) -> None:
        self._host = host

    @property
    def is_onboarding(self) -> bool:
        return self._host.mode == "onboarding"

    def preferred_root_cursor(self, visible: Sequence[str]) -> str | None:
        """The onboarding-preferred root cursor, or None."""
        if not self.is_onboarding or "mistral" not in visible:
            return None
        if self._host._mistral_needs_setup("mistral"):
            return "mistral"
        return None

    def remember_catalog_selection(self, name: str) -> None:
        if self.is_onboarding:
            self._host.selected_model = name

    def commit_destination(self) -> str:
        """Where a commit request lands: onboarding finishes through presets."""
        return "presets" if self.is_onboarding else "apply"

    def presets_stage(self) -> str | None:
        return "presets" if self.is_onboarding else None

    def presets_summary_layout(self, customize: bool) -> bool:
        return self.is_onboarding and not customize

    def presets_finish_label(self) -> str:
        return "Finish setup" if self.is_onboarding else "Save presets"

    def finish_validates(self) -> bool:
        return self.is_onboarding

    def after_commit_destination(self, mode: str) -> str:
        return "finish" if mode == "onboarding" else "presets"

    def remember_manual_model(self, state: ManagementState, name: str) -> None:
        if self.is_onboarding:
            item = state.pending.get(name)
            self._host.selected_model = item.canonical_name if item else name

    def remember_selection(self, name: str, selected: set[str]) -> None:
        if self.is_onboarding and name in selected:
            self._host.selected_model = name

    def commit_mode(
        self, mode: Literal["management", "onboarding"]
    ) -> Literal["management", "onboarding"]:
        return "management" if mode == "onboarding" else mode

    def escape_unwinds_customize(self) -> bool:
        return self.is_onboarding

    def finish_outcome(self, setup_ready: bool, changed: bool) -> bool:
        """The finish decision: onboarding completes on setup readiness."""
        return setup_ready if self.is_onboarding else changed

    def escape_hint(self) -> str:
        return "Cancel setup" if self.is_onboarding else "Close"

    def seed(self, state: ManagementState | None) -> None:
        """Seed usable roles; shipped dispatch slots resolve through these bindings.

        A single canonical model (even at different thinking levels) needs no
        explicit slot overlay. Keep the saved mode, including one-model orchestration.
        """
        self._host._onboarding_presets_seeded = False
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
            if provider_id in self._host.snapshot.overlaid_providers
            or provider_id not in SHIPPED_CATALOG.providers
            or bool(
                provider.api_key_env_var
                and self._host._credential_value(provider.api_key_env_var)
            )
        }
        candidates = [
            (name, thinking)
            for name, definition in sorted(state.catalog.models.items())
            if (self._host.selected_model is None or name == self._host.selected_model)
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
                self._host._credential_value,
                candidate=candidate,
                allowed_models=self._host.allowed_models,
                thinking_overrides=self._host.thinking_overrides,
            )
            is None
        ]
        if not candidates:
            reason = None
            if self._host.selected_model is not None:
                reason = state._preset_readiness(
                    self._host.selected_model,
                    state.preset_thinking_levels(self._host.selected_model)[0]
                    if self._host.selected_model in state.catalog.models
                    and state.preset_thinking_levels(self._host.selected_model)
                    else "medium",
                    self._host._credential_value,
                    candidate=candidate,
                    allowed_models=self._host.allowed_models,
                    thinking_overrides=self._host.thinking_overrides,
                )
            self._host._set_feedback(
                "preset",
                f"Selected model {self._host.selected_model!r} is not ready: {reason}"
                if reason
                else "No configured, enabled model is ready. Connect a provider, enable a model, and provide its credential.",
                "warning",
            )
            return
        self._host._onboarding_presets_seeded = True
        for role in state.catalog.roles:
            model, thinking = state.preset(role)
            if (
                state._preset_readiness(
                    model,
                    thinking,
                    self._host._credential_value,
                    role=role,
                    candidate=candidate,
                    allowed_models=self._host.allowed_models,
                    thinking_overrides=self._host.thinking_overrides,
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
