"""Stable visual baselines for the dedicated Web search editor."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest
from rich.console import Console

from chartreux.cli.textual_ui.screens.web_search import WebSearchScreen
from chartreux.cli.textual_ui.settings_service import SettingsService
from tests.cli.textual_ui.web_search_fixture import WebSearchHarness
from tests.snapshots.snap_compare import SnapCompare


class AdvancedWebSearchHarness(WebSearchHarness):
    def on_mount(self) -> None:
        screen = WebSearchScreen(
            cast(SettingsService, self.service),
            self.service.snapshot,
            credentials=self.credentials,
        )
        screen._advanced = True
        self.push_screen(screen)


def compare_with_palette(
    snap_compare: SnapCompare, app: WebSearchHarness, size: tuple[int, int]
) -> bool:
    """Export ANSI SVGs with the actual Textual theme palette."""
    palette = app.ansi_theme
    original = Console.export_svg

    def export_with_palette(console: Console, *args: Any, **kwargs: Any) -> str:
        kwargs["theme"] = palette
        return original(console, *args, **kwargs)

    with patch.object(Console, "export_svg", export_with_palette):
        return snap_compare(app, terminal_size=size)


@pytest.mark.parametrize("theme", ["ansi-dark", "ansi-light"])
def test_web_search_default(snap_compare: SnapCompare, theme: str) -> None:
    app = WebSearchHarness()
    app.theme = theme
    assert compare_with_palette(snap_compare, app, (80, 24))


@pytest.mark.parametrize("theme", ["ansi-dark", "ansi-light"])
def test_web_search_advanced(snap_compare: SnapCompare, theme: str) -> None:
    app = AdvancedWebSearchHarness()
    app.theme = theme
    assert compare_with_palette(snap_compare, app, (80, 48))


@pytest.mark.parametrize("theme", ["ansi-dark", "ansi-light"])
def test_web_search_onboarding_fallback(snap_compare: SnapCompare, theme: str) -> None:
    app = WebSearchHarness(mode="onboarding")
    app.theme = theme
    assert compare_with_palette(snap_compare, app, (80, 24))
