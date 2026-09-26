from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal, HorizontalScroll, VerticalScroll
from textual.content import Content
from textual.geometry import Size
from textual.visual import Visual
from textual.widgets import Static

from chartreux.cli.textual_ui.widgets.messages import (
    ExpandingBorder,
    ExpandingSeparator,
)


class _BorderApp(App[None]):
    CSS = """
    .expanding-border {
        height: 1fr;
        width: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield ExpandingBorder(classes="expanding-border")


class _ScrolledBorderApp(App[None]):
    CSS = """
    HorizontalScroll { width: 8; height: 4; }
    Horizontal { width: 30; height: 12; }
    ExpandingBorder { width: 1; height: 1fr; color: blue; text-style: dim; }
    """

    def compose(self) -> ComposeResult:
        with HorizontalScroll():
            with Horizontal():
                yield ExpandingBorder()
                yield Static("small output")


class _TallBorderApp(App[None]):
    CSS = """
    VerticalScroll { height: 4; }
    ExpandingBorder { width: 1; height: 10000; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield ExpandingBorder()


class _SeparatorApp(App[None]):
    CSS = """
    #expanding-separator {
        width: 1fr;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield ExpandingSeparator(id="expanding-separator")


def test_get_content_width_is_constant_without_rendering() -> None:
    border = ExpandingBorder()

    with patch.object(ExpandingBorder, "render") as mock_render:
        width = border.get_content_width(Size(80, 24), Size(80, 24))

    assert width == 1
    mock_render.assert_not_called()


@pytest.mark.asyncio
async def test_resize_renders_only_visible_lines() -> None:
    app = _BorderApp()

    async with app.run_test(size=(80, 24)) as pilot:
        border = app.query_one(ExpandingBorder)
        with (
            patch.object(border, "render", wraps=border.render) as mock_render,
            patch.object(border, "render_line", wraps=border.render_line) as mock_line,
        ):
            await pilot.resize_terminal(80, 25)
            await pilot.pause()

    mock_render.assert_not_called()
    assert 0 < mock_line.call_count <= 25


@pytest.mark.asyncio
@pytest.mark.parametrize("height", [1, 2, 6])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_border_glyphs_colors_and_inherited_style(
    height: int, theme: str
) -> None:
    app = _BorderApp()
    app.theme = theme
    async with app.run_test(size=(12, height)):
        border = app.query_one(ExpandingBorder)
        border.styles.color = "blue"
        border.styles.text_style = "dim"
        border.set_row_colors({0: "red", height - 1: "green"})
        rendered = Content("")
        for y in range(height):
            if y:
                rendered += Content("\n")
            glyph = "⎣" if y == height - 1 else "⎢"
            color = border._row_colors.get(y)
            rendered += Content.styled(glyph, color) if color else Content(glyph)
        expected = Visual.to_strips(border, rendered, 1, height, border.visual_style)
        for y, row in enumerate(expected):
            glyph = "⎣" if y == height - 1 else "⎢"
            assert list(border.render_line(y)) == list(row)
            assert border.render_line(y).text == glyph
        style = list(border.render_line(height - 1))[0].style
        assert style is not None and style.dim


@pytest.mark.asyncio
async def test_nested_horizontal_scroller_keeps_small_output_and_border() -> None:
    app = _ScrolledBorderApp()
    async with app.run_test(size=(20, 6)) as pilot:
        border = app.query_one(ExpandingBorder)
        scroller = app.query_one(HorizontalScroll)
        assert border.size.width == 1
        assert border.render_line(0).text == "⎢"
        assert border.render_line(border.size.height - 1).text == "⎣"
        assert app.query(Static).last().content == "small output"
        scroller.scroll_end(animate=False)
        await pilot.pause()
        assert border.render_line(0).text == "⎢"
        assert border.render_line(border.size.height - 1).text == "⎣"


@pytest.mark.asyncio
async def test_tall_border_only_renders_visible_rows() -> None:
    app = _TallBorderApp()
    async with app.run_test(size=(20, 6)) as pilot:
        border = app.query_one(ExpandingBorder)
        assert border.size.height == 10000
        with patch.object(border, "render_line", wraps=border.render_line) as mock_line:
            border.refresh()
            await pilot.pause()
        viewport_height = app.query_one(VerticalScroll).size.height
        assert mock_line.call_count == viewport_height
        assert {call.args[0] for call in mock_line.call_args_list} == set(
            range(viewport_height)
        )
        assert border.render_line(9999).text == "⎣"


@pytest.mark.asyncio
async def test_separator_resize_does_not_add_redundant_refresh() -> None:
    app = _SeparatorApp()

    async with app.run_test(size=(80, 24)) as pilot:
        separator = app.query_one(ExpandingSeparator)
        with (
            patch.object(separator, "refresh", wraps=separator.refresh) as mock_refresh,
            patch.object(separator, "render", wraps=separator.render) as mock_render,
        ):
            await pilot.resize_terminal(81, 24)
            await pilot.pause()

    assert mock_refresh.call_count == 1
    assert mock_render.call_count == 1
