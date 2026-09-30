from __future__ import annotations

from typing import Any

from textual.timer import Timer

from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.widgets.no_markup_static import NonSelectableStatic

DEFAULT_NOTICE_TIMEOUT = 4.0


class InlineNotice(NonSelectableStatic):
    """An inline status line; only informational feedback expires automatically."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.display = False
        self._hide_timer: Timer | None = None

    def show(
        self,
        message: str,
        *,
        timeout: float | None = DEFAULT_NOTICE_TIMEOUT,
        severity: str = "information",
    ) -> None:
        warning = severity == "warning" or message.startswith(("Warning:", "! "))
        error = severity == "error" or message.startswith((
            "Error:",
            "Failed:",
            "✗ ",
            "x ",
        ))
        if warning:
            wording = message.removeprefix("! ")
            if not wording.startswith("Warning:"):
                wording = f"Warning: {wording}"
            message = f"{chrome_glyph('warning')} {wording}"
        elif error:
            wording = message.removeprefix("✗ ").removeprefix("x ")
            wording = wording.removeprefix("Error: ")
            if not wording.startswith("Failed:"):
                wording = f"Failed: {wording}"
            message = f"{chrome_glyph('error')} {wording}"
        self.update(message)
        self.set_class(warning, "-warning")
        self.set_class(error, "-error")
        self.display = True
        if self._hide_timer is not None:
            self._hide_timer.stop()
            self._hide_timer = None
        if not warning and not error and timeout is not None:
            self._hide_timer = self.set_timer(timeout, self._hide)

    def hide(self) -> None:
        if self._hide_timer is not None:
            self._hide_timer.stop()
            self._hide_timer = None
        self.display = False

    def _hide(self) -> None:
        self.display = False
