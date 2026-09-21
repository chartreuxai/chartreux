from __future__ import annotations

from typing import Literal

AUTO_THEME = "auto"
DARK_THEME = "dark"
LIGHT_THEME = "light"
FALLBACK_THEME = DARK_THEME
DEFAULT_THEME = AUTO_THEME
DEFAULT_LOG_LEVEL = "WARNING"

type ThinkingLevel = Literal["off", "low", "medium", "high", "max"]
THINKING_LEVELS: tuple[ThinkingLevel, ...] = ("off", "low", "medium", "high", "max")
