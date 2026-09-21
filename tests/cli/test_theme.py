from __future__ import annotations

import pytest

from chartreux.config_values import DEFAULT_THEME
from chartreux.ui.theme import resolve_theme_name


@pytest.mark.parametrize("theme", ["auto", "light", "dark"])
def test_resolve_theme_name_preserves_a_supported_theme(theme: str) -> None:
    assert resolve_theme_name(theme) == theme


@pytest.mark.parametrize("value", [None, "unknown-theme"])
def test_resolve_theme_name_falls_back_for_unsupported_values(value: object) -> None:
    assert resolve_theme_name(value) == DEFAULT_THEME
