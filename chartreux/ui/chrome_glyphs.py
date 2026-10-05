"""Application-wide chrome glyphs (never applied to user-supplied content)."""

from __future__ import annotations

from textual._context import active_app

_GLYPHS: dict[str, tuple[str, str]] = {
    "metadata_separator": ("·", "."),
    "information": ("i", "i"),
    "running": ("…", "..."),
    "success": ("✓", "v"),
    "warning": ("!", "!"),
    "error": ("✗", "x"),
    "muted": ("□", " "),
    "cursor": ("▸", ">"),
    "checked": ("■", "x"),
    "unchecked": (" ", " "),
    "radio_selected": ("(*)", "(*)"),
    "radio_empty": ("( )", "( )"),
    "disclosure_closed": ("+", "+"),
    "disclosure_open": ("-", "-"),
    "vertical": ("↑↓", "Up/Down"),
    "horizontal": ("←→", "Left/Right"),
    "forward": ("→", "->"),
    "truncation": ("…", "..."),
}


def ascii_chrome_enabled() -> bool:
    """Whether the active application requests ASCII chrome."""
    app = active_app.get(None)
    try:
        config = getattr(app, "config", None)
    except RuntimeError:
        # Chat UI exposes config through its server, which starts after compose.
        return False
    return bool(getattr(config, "ascii_chrome", False))


def chrome_glyph(name: str) -> str:
    """Resolve one chrome glyph against the active application's ASCII setting."""
    return _GLYPHS[name][ascii_chrome_enabled()]
