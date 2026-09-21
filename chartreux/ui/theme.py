from __future__ import annotations

from chartreux.config_values import AUTO_THEME, DARK_THEME, DEFAULT_THEME, LIGHT_THEME
from chartreux.observability.logging import logger
from chartreux.ui._theme_detection import (
    resolve_auto_theme as resolve_auto_theme,
    resolve_theme as resolve_theme,
)

_VALID_THEMES = frozenset({AUTO_THEME, LIGHT_THEME, DARK_THEME})


def resolve_theme_name(value: object) -> str:
    if value == AUTO_THEME:
        return AUTO_THEME
    if not isinstance(value, str) or value not in _VALID_THEMES:
        logger.warning("Unknown theme=%s; falling back to %s", value, DEFAULT_THEME)
        return DEFAULT_THEME
    return value
