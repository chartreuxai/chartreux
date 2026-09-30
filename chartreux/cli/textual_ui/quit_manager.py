from __future__ import annotations

import time
from typing import ClassVar, Literal

from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Static

from chartreux.cli.textual_ui.widgets.path_display import PathDisplay
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint

QuitConfirmKey = Literal["Ctrl+C", "Ctrl+D", "/exit"]

QUIT_CONFIRM_DELAY = 30.0


class ExitConsequencesScreen(ModalScreen[bool]):
    """Reachable full consequence text, including from a picker or inspection."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("enter", "confirm", "Confirm exit"),
        ("escape", "cancel", "Keep working"),
    ]
    DEFAULT_CSS = """
    ExitConsequencesScreen { align: center middle; background: $background 85%; }
    ExitConsequencesScreen Static {
        width: 90%; max-width: 76; height: auto;
        padding: 1 2; border: round $warning;
        background: $surface; color: $foreground;
    }
    """

    def __init__(self, consequences: str) -> None:
        super().__init__()
        self._consequences = consequences

    def compose(self) -> ComposeResult:
        yield Static(
            "Exit consequences\n\n"
            + self._consequences
            + "\n\nEnter: confirm exit    Esc: keep working"
        )

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
        prompt = f"Press {shortcut(key) if key != '/exit' else '/exit'} again to quit"
        if extra:
            prompt = f"{prompt} ({extra})"
        try:
            path_display = self._app.query_one(PathDisplay)
            path_display.update(shortcut_hint(prompt))
        except Exception:
            pass
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
        try:
            path_display = self._app.query_one(PathDisplay)
            path_display.refresh_display()
        except Exception:
            pass
