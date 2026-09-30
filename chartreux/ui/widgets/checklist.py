"""Consistent outlined checkboxes for SelectionList-based editors."""

from __future__ import annotations

from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip
from textual.widgets import SelectionList
from textual.widgets.option_list import OptionDoesNotExist

from chartreux.ui.chrome_glyphs import chrome_glyph

CHECKBOX_SEGMENTS = 4


class Checklist(SelectionList[str]):
    """Render checked and unchecked rows without losing mouse option metadata."""

    def render_line(self, y: int) -> Strip:
        line = super().render_line(y)
        index = self.scroll_offset.y + y
        try:
            selected = self.get_option_at_index(index).value in self.selected
        except OptionDoesNotExist:
            return line
        segments = list(line)
        if len(segments) < CHECKBOX_SEGMENTS:
            return line
        style = (segments[3].style or self.rich_style) + Style(meta={"option": index})
        segments[:3] = [
            Segment("  ", style),
            Segment("[", style),
            Segment(chrome_glyph("checked" if selected else "unchecked"), style),
            Segment("]", style),
        ]
        return Strip(segments)
