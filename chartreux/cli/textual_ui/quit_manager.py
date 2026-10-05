from __future__ import annotations

import time
from typing import ClassVar, Literal

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Static

from chartreux.cli.textual_ui.widgets.session_status_line import SessionStatusLine

QuitConfirmKey = Literal["Ctrl+C", "Ctrl+D", "/exit"]

QUIT_CONFIRM_DELAY = 30.0


class ExitConsequencesScreen(ModalScreen[bool]):
    """Reachable full consequence text, including from a picker or inspection."""

    AUTO_FOCUS = "#exit-cancel"
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "cancel", "Keep working")
    ]
    DEFAULT_CSS = """
    ExitConsequencesScreen { align: center middle; background: $background 85%; }
    ExitConsequencesScreen #exit-dialog {
        width: 90%; max-width: 76; height: 90%; max-height: 26;
        padding: 1 2; border: round $warning;
        background: $surface; color: $foreground;
    }
    ExitConsequencesScreen Static { height: auto; }
    ExitConsequencesScreen #exit-consequences { height: 1fr; }
    ExitConsequencesScreen #exit-actions { height: 3; align-horizontal: right; }
    ExitConsequencesScreen Button { margin-left: 1; }
    """

    def __init__(self, consequences: str) -> None:
        super().__init__()
        self._consequences = consequences

    def compose(self) -> ComposeResult:
        with Vertical(id="exit-dialog"):
            yield Static("Exit consequences")
            with VerticalScroll(id="exit-consequences"):
                yield Static(
                    self._consequences
                    + "\n\nCancel keeps working without stopping work, abandoning "
                    "queued input, or answering pending decisions."
                )
            yield Static("Tab/Shift+Tab: focus    Enter: activate    Esc: Cancel")
            with Horizontal(id="exit-actions"):
                yield Button("Cancel", id="exit-cancel")
                yield Button("Exit", variant="warning", id="exit-confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "exit-confirm":
            self.action_confirm()
        else:
            self.action_cancel()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class QuitManager:
    def __init__(self, app: App) -> None:
        self._confirm_time: float | None = None
        self._confirm_key: QuitConfirmKey | None = None
        self._confirm_timer: Timer | None = None
        self._app = app

    @property
    def confirm_key(self) -> QuitConfirmKey | None:
        return self._confirm_key

    def is_confirmed(self, key: QuitConfirmKey) -> bool:
        return (
            self._confirm_time is not None
            and self._confirm_key == key
            and (time.monotonic() - self._confirm_time) < QUIT_CONFIRM_DELAY
        )

    def request_confirmation(self, key: QuitConfirmKey, extra: str = "") -> None:
        if self._confirm_timer is not None:
            self._confirm_timer.stop()
            self._confirm_timer = None
        self._confirm_time = time.monotonic()
        self._confirm_key = key
        prompt = f"Press {key} again to quit"
        if extra:
            prompt = f"{prompt} ({extra})"
        self._app.query_one(SessionStatusLine).set_feedback(prompt)
        self._confirm_timer = self._app.set_timer(
            QUIT_CONFIRM_DELAY, self.cancel_confirmation
        )

    def cancel_confirmation(self) -> None:
        if self._confirm_time is None:
            return
        self._confirm_time = None
        self._confirm_key = None
        if self._confirm_timer:
            self._confirm_timer.stop()
            self._confirm_timer = None
        self._app.query_one(SessionStatusLine).clear_feedback()
