from __future__ import annotations

from collections.abc import Iterator
import unicodedata

from rich.cells import cell_len
from rich.segment import Segment
from textual import events
from textual.containers import Vertical
from textual.geometry import Size
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Button

from chartreux.ui.clipboard import copy_text_to_clipboard

_ZERO_WIDTH_JOINER = "\u200d"
_EMOJI_MODIFIER_START = 0x1F3FB
_EMOJI_MODIFIER_END = 0x1F3FF


def _is_cluster_extension(char: str) -> bool:
    return unicodedata.category(char).startswith("M") or (
        _EMOJI_MODIFIER_START <= ord(char) <= _EMOJI_MODIFIER_END
    )


def _iter_clusters(text: str) -> Iterator[str]:
    """Yield pragmatic grapheme clusters, keeping tabs as standalone units."""
    cluster = ""
    join_next = False
    for char in text:
        if char == "\t":
            if cluster:
                yield cluster
                cluster = ""
            yield char
            join_next = False
        elif not cluster:
            cluster = char
            join_next = char == _ZERO_WIDTH_JOINER
        elif join_next or char == _ZERO_WIDTH_JOINER or _is_cluster_extension(char):
            cluster += char
            join_next = char == _ZERO_WIDTH_JOINER
        else:
            yield cluster
            cluster = char
            join_next = char == _ZERO_WIDTH_JOINER
    if cluster:
        yield cluster


def _advance_column(column: int, cluster: str) -> int:
    if cluster == "\t":
        return column + 8 - column % 8
    return column + cell_len(cluster)


def _line_cell_width(line: str) -> int:
    if line.isascii() and "\t" not in line:
        return len(line)
    column = 0
    for cluster in _iter_clusters(line):
        column = _advance_column(column, cluster)
    return column


class VirtualOutputText(Widget):
    ALLOW_SELECT = False

    def __init__(self, content: str) -> None:
        super().__init__(classes="virtual-output-text tool-result-detail")
        self._content = content
        self._rebuild_index()

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, content: str) -> None:
        self._content = content
        self._rebuild_index()
        self.refresh(layout=True)

    def _rebuild_index(self) -> None:
        self._offsets = [0]
        self.max_cell_width = 0
        start = 0
        while True:
            end = self.content.find("\n", start)
            line = self.content[start:] if end < 0 else self.content[start:end]
            self.max_cell_width = max(self.max_cell_width, _line_cell_width(line))
            if end < 0:
                break
            start = end + 1
            self._offsets.append(start)
        self.line_count = len(self._offsets)

    def get_content_width(self, container: Size, viewport: Size) -> int:
        return self.max_cell_width

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self.line_count

    def render_line(self, y: int) -> Strip:
        if y < 0 or y >= self.line_count:
            return Strip.blank(self.size.width, self.visual_style.rich_style)
        start = self._offsets[y]
        end = self._offsets[y + 1] - 1 if y + 1 < self.line_count else len(self.content)
        line = self.content[start:end]
        column = 0
        parts: list[str] = []
        for cluster in _iter_clusters(line):
            if cluster == "\t":
                spaces = 8 - column % 8
                parts.append(" " * spaces)
            else:
                parts.append(cluster)
            column = _advance_column(column, cluster)
        return Strip(
            [Segment("".join(parts), self.visual_style.rich_style)], cell_length=column
        )


class CopyFullOutput(Button):
    def __init__(self, content: str) -> None:
        super().__init__("Copy full output", classes="copy-full-output")
        self._full_output = content

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self._copy()

    def action_press(self) -> None:
        self._copy()
        super().action_press()

    def _copy(self) -> None:
        copy_text_to_clipboard(
            self.app,
            self._full_output,
            success_message="Full output copied to clipboard",
        )


class VirtualOutputBody(Vertical):
    def __init__(self, content: str) -> None:
        super().__init__(
            CopyFullOutput(content),
            Vertical(VirtualOutputText(content), classes="virtual-output-scroll"),
            classes="virtual-output-body",
        )
