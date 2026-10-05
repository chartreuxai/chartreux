from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

import pytest

from chartreux.ui.context_display import format_context


@pytest.mark.parametrize(
    ("usage", "threshold", "expected"),
    [
        (135_000, 400_000, "135k/400k (34%)"),
        (None, 400_000, "—/400k"),
        (135_000, None, "135k/—"),
        (None, None, "—/—"),
        (0, 400_000, "0/400k (0%)"),
        (0, None, "0/—"),
        (500_000, 400_000, "500k/400k (125%)"),
        (400_000, 400_000, "400k/400k (100%)"),
        (1_999, 3_001, "1k/3k (67%)"),
        (-1, 400_000, "—/400k"),
    ],
)
def test_context_display(usage, threshold, expected):
    assert format_context(usage, threshold) == expected


@pytest.mark.parametrize("style", ["tokens", "tokens-percent"])
@pytest.mark.parametrize("threshold", [0, -1, -400_000])
@pytest.mark.parametrize("usage", [135_000, 0, None])
def test_nonpositive_threshold_means_off(
    style: Literal["tokens", "tokens-percent"], threshold: int, usage: int | None
):
    usage_text = {135_000: "135k", 0: "0", None: "—"}[usage]
    assert format_context(usage, threshold, style=style) == (
        f"{usage_text} (auto-compact off)"
    )
    assert format_context(usage, threshold, style=style) != format_context(
        usage, None, style=style
    )


@pytest.mark.parametrize(
    ("usage", "threshold", "expected"),
    [
        (135_000, 400_000, "135k/400k"),
        (None, 400_000, "—/400k"),
        (135_000, None, "135k/—"),
        (0, 400_000, "0/400k"),
        (500_000, 400_000, "500k/400k"),
        (-1, 400_000, "—/400k"),
    ],
)
def test_tokens_style_only_omits_percentage(usage, threshold, expected):
    assert format_context(usage, threshold, style="tokens") == expected


@pytest.mark.parametrize("style", ["tokens", "tokens-percent"])
def test_compacting_and_post_compaction_stay_unknown_until_measured(
    style: Literal["tokens", "tokens-percent"],
):
    assert format_context(135_000, 400_000, style=style, compacting=True) == "—/400k"
    assert format_context(None, 400_000, style=style) == "—/400k"
    assert format_context(0, 400_000, style=style, compacting=True) == "—/400k"
    assert format_context(135_000, None, style=style, compacting=True) == "—/—"
    assert format_context(135_000, 0, style=style, compacting=True) == (
        "— (auto-compact off)"
    )


@pytest.mark.parametrize("style", ["tokens", "tokens-percent"])
@pytest.mark.parametrize(
    ("usage", "threshold"), [(135_000, 400_000), (None, 400_000), (135_000, 0)]
)
def test_last_recorded_uses_callers_snapshot(
    style: Literal["tokens", "tokens-percent"], usage: int | None, threshold: int
):
    current = format_context(usage, threshold, style=style)
    assert format_context(usage, threshold, style=style, last_recorded=True) == (
        f"{current} (last recorded)"
    )


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1k"),
        (999_999, "999k"),
        (1_000_000, "1.0M"),
        (1_250_000, "1.2M"),
        (2_999_999, "3.0M"),
    ],
)
def test_existing_token_abbreviation_convention(tokens, expected):
    assert format_context(tokens, None) == f"{expected}/—"


def test_formatter_has_only_standard_library_dependencies():
    source_path = Path(__file__).parents[2] / "chartreux" / "ui" / "context_display.py"
    tree = ast.parse(source_path.read_text())
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert set(imports) <= {"__future__", "typing"}
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
