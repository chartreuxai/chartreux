from __future__ import annotations

import ast
from pathlib import Path

import pytest

from chartreux.ui.cache_display import format_cache_hit


@pytest.mark.parametrize(
    ("cached", "prompt", "reported", "expected"),
    [
        (84_000, 200_000, True, "42%"),
        (0, 200_000, True, "0%"),
        (200_000, 200_000, True, "100%"),
        (1, 3, True, "33%"),
        (500, 1_000, True, "50%"),
        # No denominator to measure against reads as unknown, never 0%.
        (0, 0, True, "—"),
        (10, -5, True, "—"),
        (None, 100, True, "—"),
        (100, None, True, "—"),
        # Unreported cache fields are unknown, never a claimed ratio or 0%.
        (100, 100, False, "—"),
        (0, 100, False, "—"),
        (None, None, False, "—"),
    ],
)
def test_cache_hit_formatting(cached, prompt, reported, expected):
    assert format_cache_hit(cached, prompt, reported=reported) == expected


def test_cache_display_formatter_has_only_standard_library_dependencies():
    source_path = Path(__file__).parents[2] / "chartreux" / "ui" / "cache_display.py"
    tree = ast.parse(source_path.read_text())
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert set(imports) <= {"__future__"}
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
