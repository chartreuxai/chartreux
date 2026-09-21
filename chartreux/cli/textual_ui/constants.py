from __future__ import annotations

from enum import StrEnum

# Value persisted for the unpinned "default" model state. Mirrors the schema's
# ``UNPINNED_ACTIVE_MODEL`` sentinel; kept UI-side so the textual layer stays
# free of ``chartreux.core`` imports (see tests/cli/textual_ui/test_app_server_boundary).
UNPINNED_ACTIVE_MODEL = ""


class ChartreuxColors(StrEnum):
    BLUE_GREY = "#3A506B"
    BLUE_GREY_LIGHT = "#5A7287"
    COPPER = "#B87333"
    COPPER_LIGHT = "#D4924A"
    COPPER_DARK = "#8B5A2B"
