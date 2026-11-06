"""The ``/plan accept`` consent modal: the plan, its path, and the executor.

One explicit decision before the handoff replaces the conversation: start
execution with the snapshotted executor selection, or cancel and stay in plan
mode. Nothing here resolves or re-reads a selection — the app snapshots it at
modal open and the handoff applies exactly what this screen displays.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button

from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

CANCEL_BUTTON_ID = "plan-consent-cancel"
START_BUTTON_ID = "plan-consent-start"


class PlanConsentScreen(ModalScreen[bool]):
    """Confirm start of the executor session, or cancel and keep planning."""

    CSS_PATH = "plan_consent.tcss"
    # Default-No: the affirmative action replaces the conversation, so the
    # first focus is Cancel and Escape always cancels.
    AUTO_FOCUS = f"#{CANCEL_BUTTON_ID}"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True)
    ]

    def __init__(
        self,
        *,
        plan_path: str,
        summary: str,
        executor_model: str,
        executor_thinking: str,
        executor_identity: str,
        executor_display_name: str,
    ) -> None:
        super().__init__(id="plan-consent-screen")
        self._plan_path = plan_path
        self._summary = summary
        self._executor_model = executor_model
        self._executor_thinking = executor_thinking
        self._executor_identity = executor_identity
        self._executor_display_name = executor_display_name

    def compose(self) -> ComposeResult:
        with Vertical(id="plan-consent-dialog"):
            yield NoMarkupStatic("Start plan execution", id="plan-consent-title")
            with VerticalScroll(id="plan-consent-body"):
                yield NoMarkupStatic("Plan document", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._plan_path,
                    classes="plan-consent-value",
                    id="plan-consent-path",
                )
                yield NoMarkupStatic("Summary", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._summary,
                    classes="plan-consent-value",
                    id="plan-consent-summary",
                )
                yield NoMarkupStatic("Executor", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._executor_line(),
                    classes="plan-consent-value",
                    id="plan-consent-executor",
                )
            yield NoMarkupStatic(
                shortcut_hint(
                    f"{shortcut('Tab')} Focus  {shortcut('Enter')} Activate  "
                    f"{shortcut('Esc')} Cancel"
                ),
                id="plan-consent-hint",
            )
            with Horizontal(id="plan-consent-actions"):
                yield Button("Cancel", id=CANCEL_BUTTON_ID)
                yield Button("Start execution", id=START_BUTTON_ID)

    def _executor_line(self) -> str:
        model = (
            self._executor_model
            if self._executor_display_name == self._executor_model
            else f"{self._executor_display_name} ({self._executor_model})"
        )
        separator = chrome_glyph("metadata_separator")
        return (
            f"{model}  {separator}  thinking {self._executor_thinking}  "
            f"{separator}  identity {self._executor_identity}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == START_BUTTON_ID:
            self.action_start()
        else:
            self.action_cancel()

    def action_start(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
