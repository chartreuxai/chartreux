from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.geometry import Size
from textual.widgets import Button

from chartreux.app_server.models import ShellEffectOutput
from chartreux.cli.textual_ui.widgets.collapsible import HeaderCollapsibleSection
from chartreux.cli.textual_ui.widgets.tool_widgets import (
    BashResultWidget,
    clean_output,
    shell_output_body,
)
from chartreux.cli.textual_ui.widgets.virtual_output import (
    VirtualOutputBody,
    VirtualOutputText,
    _advance_column,
    _iter_clusters,
)
from chartreux.ui.clipboard import ClipboardCopyResult
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class _OutputApp(App[None]):
    CSS_PATH = Path(__file__).parents[3] / "chartreux/cli/textual_ui/app.tcss"

    def __init__(self, content: str, *, folded: bool = False) -> None:
        super().__init__()
        self.section = HeaderCollapsibleSection(
            lambda: shell_output_body(content), header_text="shell result"
        )
        self.folded = folded
        self.content = content

    def compose(self) -> ComposeResult:
        if self.folded:
            yield self.section
        else:
            yield Vertical(VirtualOutputBody(self.content), id="root")


@pytest.mark.parametrize(
    ("source", "expected_lines", "width", "first_line"),
    [
        ("", 1, 0, ""),
        ("a\n", 2, 1, "a"),
        ("a\x1b[31mb\x1b[0m\rfinal", 1, 5, "final"),
        ("界\tX\ne\u0301\tX", 2, 9, "界      X"),
    ],
)
def test_cleaned_index_and_rows(
    source: str, expected_lines: int, width: int, first_line: str
) -> None:
    widget = VirtualOutputText(clean_output(source))
    size = Size(80, 20)
    assert widget.line_count == expected_lines
    assert widget.max_cell_width == width
    assert widget.get_content_height(size, size, 80) == expected_lines
    assert widget.render_line(0).text == first_line


@pytest.mark.parametrize(
    ("content", "width", "rendered"),
    [
        ("👩‍💻", 2, "👩‍💻"),
        ("e\u0301", 1, "e\u0301"),
        ("✌️", 2, "✌️"),
        ("❤️", 2, "❤️"),
        ("⚠️", 2, "⚠️"),
        ("#️⃣", 2, "#️⃣"),
        ("👩‍💻\tX", 9, "👩‍💻      X"),
        ("e\u0301\tX", 9, "e\u0301       X"),
    ],
)
def test_grapheme_clusters_use_one_cell_width_unit(
    content: str, width: int, rendered: str
) -> None:
    widget = VirtualOutputText(content)
    line = widget.render_line(0)

    assert widget.max_cell_width == width
    assert line.text == rendered
    assert line.cell_length == width
    if "\t" not in content:
        assert widget.max_cell_width == cell_len(content)


def test_ascii_width_fast_path_matches_cluster_path() -> None:
    content = "ASCII output with 123 and punctuation!"
    cluster_width = 0
    for cluster in _iter_clusters(content):
        cluster_width = _advance_column(cluster_width, cluster)

    with patch(
        "chartreux.cli.textual_ui.widgets.virtual_output._iter_clusters",
        side_effect=AssertionError("ASCII fast path should skip clustering"),
    ):
        widget = VirtualOutputText(content)

    assert widget.max_cell_width == cluster_width


def test_content_reassignment_rebuilds_index() -> None:
    widget = VirtualOutputText("old")

    widget.content = "界\nnew\n"

    assert widget.line_count == 3
    assert widget.max_cell_width == 3
    assert widget.get_content_height(Size(10, 10), Size(10, 10), 10) == 3
    assert widget.render_line(0).text == "界"
    assert widget.render_line(1).text == "new"
    assert widget.render_line(2).text == ""


@pytest.mark.parametrize(
    ("source", "virtual"),
    [
        ("x\n" * 1998 + "x", False),
        ("x\n" * 1999 + "x", True),
        ("a" * (128 * 1024 - 1), False),
        ("a" * (128 * 1024), True),
        ("界" * 43690, False),
        ("界" * 43691, True),
    ],
    ids=[
        "1999-lines",
        "2000-lines",
        "under-128k",
        "at-128k",
        "utf8-under",
        "utf8-over",
    ],
)
def test_thresholds(source: str, virtual: bool) -> None:
    body = shell_output_body(source)
    assert isinstance(body, VirtualOutputBody if virtual else NoMarkupStatic)
    if not virtual:
        assert body.render() == source


def test_measurement_uses_cached_index() -> None:
    widget = VirtualOutputText("界\tword\n" * 2000)
    original_offsets = widget._offsets
    with patch(
        "chartreux.cli.textual_ui.widgets.virtual_output.cell_len",
        side_effect=AssertionError("rescan"),
    ):
        for _ in range(100):
            assert widget.get_content_width(Size(10, 10), Size(10, 10)) == 12
            assert widget.get_content_height(Size(10, 10), Size(10, 10), 10) == 2001
    assert widget._offsets is original_offsets
    assert VirtualOutputText.ALLOW_SELECT is False


@pytest.mark.asyncio
async def test_horizontal_scroll_reaches_rightmost_cells() -> None:
    app = _OutputApp("x" * 100 + "\n" + "界" * 60)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        body = app.query_one(VirtualOutputText)
        scroller = app.query_one(".virtual-output-scroll", Vertical)
        assert body.size.width >= 120
        assert scroller.max_scroll_x > 0
        scroller.scroll_to(x=scroller.max_scroll_x, animate=False)
        await pilot.pause()
        assert scroller.scroll_x == scroller.max_scroll_x
        assert body.render_line(1).text.endswith("界" * 60)


@pytest.mark.asyncio
async def test_lazy_copy_and_click_does_not_fold() -> None:
    content = "text\x1b[31m\rfinal\n" + "x\n" * 2000
    cleaned = clean_output(content)
    app = _OutputApp(cleaned, folded=True)
    original_init = VirtualOutputText.__init__
    with patch.object(
        VirtualOutputText, "__init__", autospec=True, side_effect=original_init
    ) as constructor:
        async with app.run_test(size=(55, 12)) as pilot:
            await pilot.pause()
            constructor.assert_not_called()
            assert app.section.is_collapsed
            app.section.set_collapsed(False)
            await pilot.pause()
            constructor.assert_called_once()
            assert app.query_one(VirtualOutputText).content == cleaned
            button = app.query_one(Button)
            with patch(
                "chartreux.cli.textual_ui.widgets.virtual_output.copy_text_to_clipboard",
                return_value=ClipboardCopyResult(cleaned, False),
            ) as copy:
                await pilot.click(button)
                await pilot.pause()
                copy.assert_called_once_with(
                    app, cleaned, success_message="Full output copied to clipboard"
                )
                button.focus()
                await pilot.press("enter")
                await pilot.pause()
                assert copy.call_count == 2
            assert not app.section.is_collapsed


def test_small_bash_output_remains_identical() -> None:
    widget = BashResultWidget(
        result=ShellEffectOutput(
            stdout="", stderr="", output="\n\x1b[31mhello\x1b[0m\n"
        ),
        success=True,
        message="",
    )
    children = list(widget.compose())
    assert len(children) == 1
    assert isinstance(children[0], NoMarkupStatic)
    assert children[0].render() == "hello"


@pytest.mark.asyncio
async def test_large_bash_copy_keeps_sanitized_trailing_newline() -> None:
    source = "row\x1b[31m\rfinal\n" * 2000
    widget = BashResultWidget(
        result=ShellEffectOutput(stdout="", stderr="", output=source),
        success=True,
        message="",
    )
    children = list(widget.compose())
    assert len(children) == 1
    assert isinstance(children[0], VirtualOutputBody)
    app = _OutputApp(clean_output(source))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(VirtualOutputText).content == "final\n" * 2000
