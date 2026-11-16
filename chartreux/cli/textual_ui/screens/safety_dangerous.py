"""v0.8.0 A-P1: the DANGEROUS-mode ceremony screens and the persistent banner.

Entering SafetyLevel DANGEROUS is ceremony-only (unguarded-mode-design.md
:49-65, surfaced as ``/safety dangerous`` per the amendment Q1/Q2): a
non-dismissible warning screen gated on the typed phrase
``ENABLE DANGEROUS MODE``, then a SEPARATE explicit confirmation. The
non-dismissible banner rides outside the scrolling transcript while the
level is active; if it cannot be displayed, activation is refused.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

#: The exact typed phrase that unlocks the confirmation step.
DANGEROUS_PHRASE = "ENABLE DANGEROUS MODE"

WARNING_TEXT = (
    "DANGEROUS MODE suspends every Chartreux guard for this session AND its "
    "subagents: tools, filesystem, secret files, identity, shell, network, "
    "and dangerous-command approvals stop applying. The agent may read "
    "credentials, delete files, execute arbitrary commands, and publish "
    "without further prompts; hook vetoes will not block. OS and external "
    "restrictions still apply. Actions may be irreversible.\n\n"
    f'Type "{DANGEROUS_PHRASE}" exactly to continue.'
)


class DangerousWarningScreen(ModalScreen[str | None]):
    """Non-dismissible warning screen with the typed-phrase gate.

    Escape does not dismiss it (priority no-op binding); only submitting the
    exact phrase (result ``"phrase_ok"``) or the app cancelling the ceremony
    (result ``None``) closes it.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "noop", "", show=False, priority=True),
        Binding("tab", "focus_phrase", "", show=False, priority=True),
    ]
    AUTO_FOCUS = "#dangerous-phrase"

    def compose(self) -> ComposeResult:
        with Vertical(id="dangerous-warning"):
            yield Static(WARNING_TEXT, id="dangerous-warning-text")
            yield Input(id="dangerous-phrase", placeholder=DANGEROUS_PHRASE)
            yield Static("", id="dangerous-warning-error")

    def on_mount(self) -> None:
        self.call_after_refresh(self._focus_phrase)

    def _focus_phrase(self) -> None:
        self.query_one("#dangerous-phrase", Input).focus()

    def action_focus_phrase(self) -> None:
        self._focus_phrase()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.value.strip() == DANGEROUS_PHRASE:
            self.dismiss("phrase_ok")
        else:
            error = self.query_one("#dangerous-warning-error", Static)
            error.update("The phrase does not match. Type it exactly to continue.")

    def action_noop(self) -> None:
        """The warning screen is non-dismissible: Escape does nothing."""


class DangerousConfirmScreen(ModalScreen[bool]):
    """The separate, explicit confirmation after the typed phrase."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "", show=False, priority=True)
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="dangerous-confirm"):
            yield Static(
                "Enable DANGEROUS MODE for this session and its subagents? "
                "This cannot be undone by a bare enter; use /safety safe to revoke.",
                id="dangerous-confirm-text",
            )
            with Vertical(id="dangerous-confirm-buttons"):
                yield Button(
                    "Enable DANGEROUS MODE", id="dangerous-confirm-yes", variant="error"
                )
                yield Button("Cancel", id="dangerous-confirm-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dangerous-confirm-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class DangerousBanner(Static):
    """The NON-DISMISSIBLE DANGEROUS banner, outside the scrolling transcript."""

    def __init__(self) -> None:
        super().__init__(
            "DANGEROUS MODE ACTIVE — all guards suspended for this session and its subagents",
            id="dangerous-banner",
        )
