from __future__ import annotations

import os

from rich.markup import escape
from textual.content import Content

from chartreux.ui.chrome_glyphs import chrome_glyph

SHORTCUT_STYLE = "b not dim $primary"
NO_COLOR_SHORTCUT_STYLE = "b not dim $foreground"


def rich_theme_style(value: str) -> str:
    """Translate a Textual theme color into a Rich style when one is available."""
    color = value.split()[0]
    if color in {"auto", "ansi_default"}:
        return ""
    return color.removeprefix("ansi_")


def shortcut(key: str) -> str:
    key = key.replace("↑↓", chrome_glyph("vertical"))
    key = key.replace("←→", chrome_glyph("horizontal"))
    style = (
        NO_COLOR_SHORTCUT_STYLE
        if os.environ.get("NO_COLOR") is not None
        else SHORTCUT_STYLE
    )
    return f"[{style}]{escape(key)}[/]"


def shortcut_hint(markup: str) -> Content:
    return Content.from_markup(markup)


def with_status(status: str | None, hint: Content) -> Content:
    if not status:
        return hint
    return Content(status + "  ") + hint
