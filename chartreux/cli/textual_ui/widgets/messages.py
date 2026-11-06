from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
from time import monotonic
from typing import TYPE_CHECKING, ClassVar, cast

from chartreux.app_server.models import (
    FileImageSource,
    HookSeverity,
    ImageAttachment,
    InlineImageSource,
)

if TYPE_CHECKING:
    from rich.style import Style as RichStyle
    from textual.css.styles import RulesMap
    from textual.timer import Timer
    from textual.widgets import Markdown
    from textual.widgets._markdown import (
        MarkdownBlock,
        MarkdownBullet,
        MarkdownBulletList,
        MarkdownListItem,
        MarkdownParagraph,
        MarkdownStream,
    )

    from chartreux.cli.textual_ui.app import ChatScroll


from rich.cells import cell_len
from rich.segment import Segment
from textual import events
from textual.app import ComposeResult
from textual.color import Color
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.geometry import Size
from textual.message import Message
from textual.strip import Strip
from textual.style import Style
from textual.visual import RenderOptions, Visual
from textual.widget import Widget
from textual.widgets import Button, Link, Static

from chartreux.cli.textual_ui.widgets.collapsible import (
    ClickWithoutDragMixin,
    DisclosureHeader,
    OverflowCollapsibleSection,
    lines_label,
)
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.message_header import MessageHeader
from chartreux.cli.textual_ui.widgets.spinner import SpinnerMixin, SpinnerType
from chartreux.cli.textual_ui.widgets.tool_widgets import clean_output
from chartreux.ui.chrome_glyphs import ascii_chrome_enabled, chrome_glyph
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic, NonSelectableStatic

# Full-content MarkdownStream reparsing per delta dominated CPU; 50ms frames cut it ~40%.
# The delay is a target cadence, not a fixed wait: the pump can hand a frame its
# turn ~50ms late while a dense stream keeps it busy, which doubles the visible
# frame period. Each frame therefore measures the period it actually achieved and
# corrects the delay toward STREAM_WRITE_FRAME_SECONDS; the delay is clamped to
# [STREAM_WRITE_FRAME_SECONDS_MIN, STREAM_WRITE_FRAME_SECONDS_MAX] so
# back-pressure cannot stretch the transcript past the hard cap.
STREAM_WRITE_FRAME_SECONDS = 0.05
STREAM_WRITE_FRAME_SECONDS_MIN = 0.025
STREAM_WRITE_FRAME_SECONDS_MAX = 0.15
STREAM_WRITE_FRAME_CORRECTION_GAIN = 0.5
# Corrections smaller than this only add jitter to the cadence.
STREAM_WRITE_FRAME_DEADBAND_SECONDS = 0.005
# A pause longer than this is the stream idling, not render back-pressure: the
# next frame restarts cadence tracking instead of over-correcting for the gap.
STREAM_WRITE_FRAME_GAP_RESET_SECONDS = 0.25


@dataclass(slots=True)
class StreamWriteCounters:
    """Process-local byte and flush accounting for the streaming write path.

    Measurement surface only: the perf suite reads these to attribute a
    streaming turn's cost to its buffered flushes, and no code branches on them.
    Counters are cumulative, so a reader resets them before measuring a turn.
    """

    flush_count: int = 0
    flush_bytes: int = 0
    initial_write_count: int = 0
    initial_write_bytes: int = 0
    final_flush_count: int = 0
    final_flush_bytes: int = 0

    def reset(self) -> None:
        self.flush_count = 0
        self.flush_bytes = 0
        self.initial_write_count = 0
        self.initial_write_bytes = 0
        self.final_flush_count = 0
        self.final_flush_bytes = 0


STREAM_WRITE_COUNTERS = StreamWriteCounters()

# ---------------------------------------------------------------------------
# P5b — simple-list streaming fast path.
#
# Textual's Markdown.append re-tokenizes the entire unparsed tail on every
# streaming frame while a list is open (_last_parsed_line stays pinned at the
# list's first line), so a growing list costs O(document) parse per frame — the
# reparse channel that dominated the P5a freeze (78% of the 400-item wall).
# For documents the parser confirms are SIMPLE — a paragraph prefix plus one
# tight bullet list of single-line items whose inline content carries no
# reference-style brackets — SimpleListStreamEngine renders each item exactly
# once: completed items are mounted as list-item widgets and are never
# reparsed or reflowed again, and only the active (last) item is reparsed per
# frame. Anything else (loose lists, code fences, nested containers, late
# reference definitions) takes the whole-document path — Markdown.update per
# frame — which may reparse earlier items for correct rendering.
# ---------------------------------------------------------------------------

# A simple-list item marker: a bullet at column 0 ("- ", "* ", "+ ").
_SIMPLE_ITEM_MARKER_RE = re.compile(r"^[-*+] ")
_LIST_START_RE = re.compile(r"(?:^|\n)[-*+] ")

# Inline tokens a simple item may contain. Any "[" anywhere in the document
# forces the whole-document path, so a reference definition arriving late can
# never change an already-finalized item.
_SIMPLE_INLINE_TOKEN_TYPES = frozenset({
    "text",
    "code_inline",
    "em_open",
    "em_close",
    "strong_open",
    "strong_close",
    "s_open",
    "s_close",
})


class _CompletedListItem:
    """One finalized simple-list item: its parsed inline content plus the
    strips it was last rendered to.

    Completed items live here rather than as widgets. While the layout key
    holds they are never re-parsed or re-wrapped — the canvas serves their
    cached strips — so a streaming frame's layout pass stays O(1) in list
    length instead of O(items x widgets-per-item).
    """

    __slots__ = ("content", "bullet", "strips")

    def __init__(self, content: Content, bullet: str) -> None:
        self.content = content
        self.bullet = bullet
        self.strips: list[Strip] = []


class _SimpleListCanvas(Widget):
    """Renders the completed prefix of a simple list from cached strips.

    A single widget carries every completed item, so appending an item never
    grows the widget tree and a relayout never re-measures old items.
    """

    DEFAULT_CSS = """
    _SimpleListCanvas {
        width: 1fr;
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._lines: list[Strip] = []

    def set_lines(self, lines: list[Strip]) -> None:
        self._lines = lines
        self.refresh(layout=True)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return len(self._lines)

    def render_line(self, y: int) -> Strip:
        try:
            return self._lines[y]
        except IndexError:
            return Strip.blank(self.size.width, self.visual_style.rich_style)


class SimpleListStreamEngine:
    """Incremental render engine for parser-confirmed simple bullet lists.

    ``write`` returns True when it consumed the chunk itself (fast path or
    whole-document fallback); False means the caller should use the plain
    MarkdownStream path (document not list-shaped yet). The counters
    (``parse_bytes_log``, ``layout_cache``, ``cache_invalidations``,
    ``fallback_count``) are a measurement surface read by tests; no rendering
    decision branches on them.

    In fast mode the list block holds exactly two children: a
    ``_SimpleListCanvas`` carrying every completed item as cached strips, and
    the active item's ``Horizontal`` subtree, whose paragraph is updated in
    place per frame. Each frame parses only item-sized source ranges.
    """

    def __init__(self, owner: StreamingMessageBase) -> None:
        self._owner = owner
        self.mode = "undecided"  # "undecided" | "fast" | "fallback"
        self._consumed = 0  # char offset up to which complete lines are processed
        self._scanned = 0  # char offset already scanned for a list start
        self._active_offset: int | None = None  # start of the active item's line
        self._list_block: MarkdownBulletList | None = None
        self._canvas: _SimpleListCanvas | None = None
        self._completed: list[_CompletedListItem] = []
        self._active_h: Horizontal | None = None  # active item's Horizontal widget
        self._active_bullet: MarkdownBullet | None = None
        self._active_para: MarkdownParagraph | None = None  # active item's paragraph
        self._render_options: RenderOptions | None = None
        self._bullet_style: RichStyle | None = None
        self._style_generation = 0
        self._last_theme: str | None = None
        self._cache_key: tuple[int, int] | None = None
        self.parse_bytes_log: list[int] = []
        self.layout_cache: dict[tuple[int, int], list[_CompletedListItem]] = {}
        self.cache_invalidations = 0
        self.fallback_count = 0

    @property
    def _item_count(self) -> int:
        """Items rendered so far: completed cache entries plus the active item."""
        return len(self._completed) + (1 if self._active_h is not None else 0)

    def reset(self) -> None:
        """Return to undecided (ReasoningMessage expansion replays content)."""
        self.mode = "undecided"
        self._consumed = 0
        self._scanned = 0
        self._active_offset = None
        self._list_block = None
        self._canvas = None
        self._completed.clear()
        self._active_h = None
        self._active_bullet = None
        self._active_para = None
        self._render_options = None
        self._bullet_style = None
        self._cache_key = None
        self.parse_bytes_log.clear()
        self.layout_cache.clear()
        self.cache_invalidations = 0
        self.fallback_count = 0

    async def write(self, chunk: str, full: str) -> bool:
        """Handle one buffered write; True when consumed here."""
        if self.mode == "undecided":
            window = full[max(0, self._scanned - 2) :]
            self._scanned = len(full)
            if _LIST_START_RE.search(window):
                if self._is_simple_document(full):
                    await self._engage(full)
                    return True
                await self._fallback(full)
                return True
            return False
        if self.mode == "fast":
            await self._fast_write(full)
            return True
        await self._fallback(full)
        return True

    # -- shape confirmation (parser-confirmed, via the widget's own parser) --

    def _is_simple_document(self, text: str) -> bool:
        if "[" in text:
            return False
        from textual.widgets._markdown import MarkdownBulletList, MarkdownParagraph

        md = self._owner._get_markdown()
        blocks = md._build_from_source(text)
        list_blocks = [b for b in blocks if isinstance(b, MarkdownBulletList)]
        if len(list_blocks) > 1 or (list_blocks and blocks[-1] is not list_blocks[0]):
            return False
        prefix = blocks[: len(blocks) - 1] if list_blocks else blocks
        if any(not isinstance(b, MarkdownParagraph) for b in prefix):
            return False
        if not list_blocks:
            return True
        if not self._list_region_lines_ok(text):
            return False
        return all(
            self._single_item_paragraph(item) is not None
            for item in list_blocks[0]._blocks
        )

    @staticmethod
    def _list_region_lines_ok(text: str) -> bool:
        """Every complete line from the first marker on is a marker line."""
        # split("\n") leaves a trailing element: the split artifact after a final
        # newline, or the still-partial last line — either way it is not complete.
        lines = text.split("\n")[:-1]
        for index, line in enumerate(lines):
            if _SIMPLE_ITEM_MARKER_RE.match(line):
                return all(_SIMPLE_ITEM_MARKER_RE.match(l) for l in lines[index:])
        return False

    def _single_item_paragraph(self, item: MarkdownBlock) -> MarkdownParagraph | None:
        """The item's single tight paragraph, or None when the item is not simple."""
        from textual.widgets._markdown import (
            MarkdownParagraph,
            MarkdownUnorderedListItem,
        )

        if not isinstance(item, MarkdownUnorderedListItem):
            return None
        if len(item._blocks) != 1 or not isinstance(item._blocks[0], MarkdownParagraph):
            return None
        para = item._blocks[0]
        if not para._token.hidden:  # tight list only; loose lists fall back
            return None
        inline = para._inline_token
        if inline is not None and inline.children is not None:
            if any(c.type not in _SIMPLE_INLINE_TOKEN_TYPES for c in inline.children):
                return None
        return para

    def _parse_single_item(self, src: str) -> MarkdownListItem | None:
        """Parse one item's source; return its list-item block or None."""
        from textual.widgets._markdown import MarkdownBulletList, MarkdownListItem

        md = self._owner._get_markdown()
        blocks = md._build_from_source(src if src.endswith("\n") else src + "\n")
        if len(blocks) != 1 or not isinstance(blocks[0], MarkdownBulletList):
            return None
        items = blocks[0]._blocks
        if len(items) != 1 or not isinstance(items[0], MarkdownListItem):
            return None
        item = items[0]
        if self._single_item_paragraph(item) is None:
            return None
        return item

    # -- fast path --

    @staticmethod
    def _line_offsets(full: str) -> tuple[int, int | None, int | None]:
        """(end of complete lines, first marker-line start, active-line start).

        The active line is the last marker line — complete or the still
        partial tail.
        """
        parts = full.split("\n")
        partial = "" if full.endswith("\n") else parts[-1]
        pos = 0
        list_start: int | None = None
        active_offset: int | None = None
        for line in parts[:-1]:
            if _SIMPLE_ITEM_MARKER_RE.match(line):
                if list_start is None:
                    list_start = pos
                active_offset = pos
            pos += len(line) + 1
        if partial and _SIMPLE_ITEM_MARKER_RE.match(partial):
            if list_start is None:
                list_start = pos
            active_offset = pos
        return pos, list_start, active_offset

    async def _engage(self, full: str) -> None:
        await self._stop_owner_stream()
        md = self._owner._get_markdown()
        self.mode = "fast"
        self.parse_bytes_log.append(len(full.encode("utf-8")))
        from textual.widgets._markdown import MarkdownBulletList, MarkdownListItem

        consumed, list_start, active_offset = self._line_offsets(full)
        if list_start is None:
            await self._fallback(full)
            return
        self._consumed = consumed
        self._active_offset = active_offset
        prefix, list_src = full[:list_start], full[list_start:]
        # Render only the prefix through Markdown.update (in the common case the
        # plain stream already wrote exactly this), and build the list from a
        # region parse — never a whole-document reparse or remount.
        await md.update(prefix)
        parsed = md._build_from_source(
            list_src if list_src.endswith("\n") else list_src + "\n"
        )
        list_block = next(
            (b for b in parsed if isinstance(b, MarkdownBulletList)), None
        )
        items = list(list_block._blocks) if list_block is not None else []
        if (
            list_block is None
            or not items
            or any(not isinstance(item, MarkdownListItem) for item in items)
            or any(self._single_item_paragraph(item) is None for item in items)
        ):
            await self._fallback(full)
            return
        list_items = [cast("MarkdownListItem", item) for item in items]
        await self._mount_fast_list(md, list_block, list_items)
        md._markdown = full
        self._sync_layout_cache()

    async def _mount_fast_list(
        self,
        md: Markdown,
        list_block: MarkdownBulletList,
        list_items: list[MarkdownListItem],
    ) -> None:
        """Mount the list block empty, then the canvas and the active subtree.

        Every item's parsed paragraph is harvested into the completed-item
        cache without mounting the whole-document item widgets; the last item
        stays live as the active one.
        """
        from textual.widgets._markdown import MarkdownBullet

        for item in list_items[:-1]:
            para = cast("MarkdownParagraph", item._blocks[0])
            self._completed.append(_CompletedListItem(para._content, item.bullet))
        active_item = list_items[-1]
        active_para = cast("MarkdownParagraph", active_item._blocks[0])
        list_block._blocks.clear()
        await md.mount(list_block)
        self._list_block = list_block
        canvas = _SimpleListCanvas()
        await list_block.mount(canvas)
        self._canvas = canvas
        bullet = MarkdownBullet()
        bullet.symbol = active_item.bullet
        horizontal = Horizontal(bullet, Vertical(active_para))
        await list_block.mount(horizontal)
        self._active_h = horizontal
        self._active_bullet = bullet
        self._active_para = active_para

    def _capture_render_context(self) -> None:
        """Borrow wrap rules and bullet style from the live item widgets so
        canvas-rendered rows match a whole-document render exactly.
        """
        para = self._active_para
        if para is not None:
            self._render_options = RenderOptions(
                para._get_style, cast("RulesMap", para.styles), None, None
            )
        bullet = self._active_bullet
        if bullet is not None:
            self._bullet_style = bullet.visual_style.rich_style

    async def _fast_write(self, full: str) -> None:
        md = self._owner._get_markdown()
        list_block = self._list_block
        if list_block is None:
            await self._fallback(full)
            return
        parsed_bytes = 0
        parts = full[self._consumed :].split("\n")
        complete, tail = parts[:-1], parts[-1]
        consumed = self._consumed
        # (line, ends_with_newline): every complete line, plus the still
        # partial tail line when present. All but the last become completed
        # cache entries; the last is the active item.
        lines: list[tuple[str, bool]] = [(line, True) for line in complete]
        if tail:
            lines.append((tail, False))
        with md.app.batch_update():
            if (
                complete
                and self._active_offset is not None
                and self._active_offset < consumed
            ):
                # The previously active line is complete now: finalize it.
                src = full[self._active_offset : consumed]
                parsed_bytes += len(src.encode("utf-8"))
                if not self._complete_item(src):
                    await self._fallback(full)
                    return
            for line, _is_complete in lines[:-1]:
                consumed += len(line) + 1
                if not _SIMPLE_ITEM_MARKER_RE.match(line):
                    await self._fallback(full)
                    return
                parsed_bytes += len(line.encode("utf-8")) + 1
                if not self._complete_item(line + "\n"):
                    await self._fallback(full)
                    return
            if lines:
                line, is_complete = lines[-1]
                line_start = consumed
                if is_complete:
                    consumed += len(line) + 1
                if not _SIMPLE_ITEM_MARKER_RE.match(line):
                    await self._fallback(full)
                    return
                parsed_bytes += len(line.encode("utf-8")) + 1
                src = full[line_start:]
                item = self._parse_single_item(
                    src if src.endswith("\n") else src + "\n"
                )
                if item is None or not await self._activate_item(item, line_start):
                    await self._fallback(full)
                    return
        self._consumed = consumed
        md._markdown = full
        self.parse_bytes_log.append(parsed_bytes)
        self._push_canvas_lines()
        self._sync_layout_cache()

    def _complete_item(self, src: str) -> bool:
        """Parse one finalized item line into the completed-item cache."""
        item = self._parse_single_item(src)
        if item is None:
            return False
        para = cast("MarkdownParagraph", item._blocks[0])
        entry = _CompletedListItem(para._content, item.bullet)
        if self._cache_key is not None:
            entry.strips = self._item_strips(
                entry.content, entry.bullet, self._cache_key[0], self._render_options
            )
        self._completed.append(entry)
        return True

    async def _activate_item(self, item: MarkdownListItem, line_start: int) -> bool:
        """Make ``item`` the active item: reuse the live subtree in place, or
        mount one when the list was empty at engagement.
        """
        para = cast("MarkdownParagraph", item._blocks[0])
        self._active_offset = line_start
        if self._active_h is None or self._active_para is None:
            list_block = self._list_block
            if list_block is None:
                return False
            from textual.widgets._markdown import MarkdownBullet

            bullet = MarkdownBullet()
            bullet.symbol = item.bullet
            horizontal = Horizontal(bullet, Vertical(para))
            await list_block.mount(horizontal)
            self._active_h = horizontal
            self._active_para = para
            self._active_bullet = bullet
            self._capture_render_context()
        else:
            if self._active_bullet is not None:
                self._active_bullet.symbol = item.bullet
            await self._active_para._update_from_block(para)
        return True

    # -- completed-item strip cache --

    def _item_strips(
        self, content: Content, bullet: str, width: int, options: RenderOptions | None
    ) -> list[Strip]:
        """Render one item to rows exactly as a whole-document render would:
        the bullet on the first row, the paragraph wrapped to the remaining
        width (borrowed wrap rules), blank bullet cells on later rows.
        """
        if width <= 0 or options is None:
            return []
        bullet_width = cell_len(bullet)
        para_width = max(1, width - bullet_width)
        para_strips = content.render_strips(para_width, None, Style.null(), options)
        rows: list[Strip] = []
        for index, para_strip in enumerate(para_strips):
            prefix = (
                Strip([Segment(bullet, self._bullet_style)])
                if index == 0
                else Strip.blank(bullet_width)
            )
            rows.append(prefix + para_strip)
        return rows

    def _render_completed_strips(self, key: tuple[int, int]) -> None:
        """Re-wrap every completed item at ``key`` from its cached parsed
        content — a re-render on invalidation, never a re-parse.
        """
        width = key[0]
        options = self._render_options
        for entry in self._completed:
            entry.strips = self._item_strips(
                entry.content, entry.bullet, width, options
            )

    def _push_canvas_lines(self) -> None:
        canvas = self._canvas
        if canvas is None:
            return
        lines: list[Strip] = []
        for entry in self._completed:
            lines.extend(entry.strips)
        canvas.set_lines(lines)

    # -- whole-document fallback --

    async def _fallback(self, full: str) -> None:
        self.mode = "fallback"
        self.fallback_count += 1
        await self._stop_owner_stream()
        md = self._owner._get_markdown()
        # Markdown.update removes the old blocks — including the canvas and
        # the active subtree — and remounts whole-document blocks.
        await md.update(full)
        self._list_block = None
        self._canvas = None
        self._completed.clear()
        self._active_h = None
        self._active_bullet = None
        self._active_para = None
        self._render_options = None
        self._active_offset = None
        self._cache_key = None
        self.layout_cache.clear()

    async def _stop_owner_stream(self) -> None:
        stream = self._owner._stream
        if stream is not None:
            self._owner._stream = None
            await stream.stop()

    # -- completed-item layout cache --

    def current_layout_key(self) -> tuple[int, int]:
        """(effective width, theme/style generation) the items are rendered at.

        The style generation is derived lazily from the app theme — no watcher
        is registered, so a removed message widget stays collectable.
        """
        md = self._owner._get_markdown()
        theme = self._owner.app.theme
        if theme != self._last_theme:
            self._last_theme = theme
            self._style_generation += 1
        return (md.content_region.width, self._style_generation)

    def _sync_layout_cache(self) -> None:
        """Key the completed-item layout cache by width and style generation.

        While the key holds, completed items are served from their cached
        strips and are never re-parsed or re-wrapped; a width or theme change
        invalidates the old key and re-wraps the completed items — from their
        cached parsed content, never re-parsed — under the new one.
        """
        list_block = self._list_block
        if self.mode != "fast" or list_block is None:
            return
        key = self.current_layout_key()
        if key == self._cache_key:
            return
        if self._cache_key is not None:
            self.layout_cache.pop(self._cache_key, None)
        self._cache_key = key
        self.cache_invalidations += 1
        if self._active_para is not None or self._active_bullet is not None:
            self._capture_render_context()
        self._render_completed_strips(key)
        self._push_canvas_lines()
        self.layout_cache[key] = list(self._completed)


# A fatal provider error can carry a multi-kilobyte single-line excerpt that
# fold-wraps into dozens of rows and obscures the transcript. Errors estimated
# taller than this render as a one-line summary plus an "+N lines" disclosure;
# the expanded body is capped at the same height (keep in sync with the
# `.error-overflow-body` max-height in app.tcss).
_ERROR_MAX_ROWS = 12
_ERROR_ESTIMATE_WIDTH = 80


def _estimated_row_count(text: str, width: int = _ERROR_ESTIMATE_WIDTH) -> int:
    """Upper-bound the wrapped row count at a typical transcript width.

    The fold-wrap of one unbreakable token is what blows up the layout, so
    estimating per logical line (never trusting newlines alone) is enough to
    tell a short error from an oversized payload; the disclosure body's
    max-height is the hard cap either way.
    """
    return sum(max(1, -(-len(line) // width)) for line in text.split("\n"))


class ExpandingBorder(NonSelectableStatic):
    def __init__(self, *, classes: str | None = None) -> None:
        super().__init__(classes=classes)
        self._row_colors: dict[int, str] = {}

    def set_row_colors(self, colors: dict[int, str]) -> None:
        self._row_colors = colors
        self.refresh()

    # The border is always a single glyph column. Returning a constant avoids the
    # default measurement, which renders the widget (reading self.size) during
    # arrange and forces an O(N) compositor map rebuild on every layout pass.
    def get_content_width(self, container: Size, viewport: Size) -> int:
        return 1

    def render_line(self, y: int) -> Strip:
        if not self.size.width or y < 0 or y >= self.size.height:
            return Strip.blank(self.size.width, self.visual_style.rich_style)
        char = "⎣" if y == self.size.height - 1 else "⎢"
        color = self._row_colors.get(y)
        content = Content.styled(char, color) if color else Content(char)
        strip = Visual.to_strips(self, content, 1, 1, self.visual_style)[0]
        return strip.apply_offsets(0, y)


# Mimic a border bottom with this component in order to have dimmed colors in ANSI themes
# Move back to border when Textual supports dimmed borders or foreground-muted in ANSI themes
class ExpandingSeparator(NonSelectableStatic):
    def render(self) -> str:
        return "─" * max(self.size.width, 1)


def _attachment_label(attachment: ImageAttachment) -> str:
    alias_path = Path(attachment.alias).expanduser()
    if not alias_path.is_absolute():
        return attachment.alias
    return _format_display_path(alias_path)


def _format_display_path(path: Path) -> str:
    home = Path.home()
    try:
        relative = path.relative_to(home)
    except ValueError:
        return str(path)
    if str(relative) == ".":
        return "~"
    return str(Path("~") / relative)


class UserMessageAttachment(Horizontal):
    def __init__(self, attachment: ImageAttachment) -> None:
        super().__init__(classes="user-message-attachment-line")
        self._attachment = attachment

    def compose(self) -> ComposeResult:
        yield NoMarkupStatic(
            "└ attached image: ", classes="user-message-attachment-label"
        )
        match self._attachment.source:
            case FileImageSource(path=path):
                image_path = Path(path)
                yield Link(
                    _attachment_label(self._attachment),
                    url=image_path.as_uri(),
                    classes="user-message-attachment-link",
                )
            case InlineImageSource():
                # Inline images have no file on disk, so there's nothing to link.
                yield NoMarkupStatic(
                    _attachment_label(self._attachment),
                    classes="user-message-attachment-link",
                )


class UserMessage(Static):
    PROMPT_CHAR: ClassVar[str] = "You"
    SHOW_SEPARATOR: ClassVar[bool] = False

    def __init__(
        self,
        content: str,
        pending: bool = False,
        history_entry_id: str | None = None,
        images: list[ImageAttachment] | None = None,
        *,
        posted_at: datetime | None = None,
        show_message_timestamps: bool = True,
    ) -> None:
        super().__init__()
        self.header = MessageHeader(
            self.PROMPT_CHAR, show_message_timestamps=show_message_timestamps
        )
        self.posted_at = posted_at
        self.add_class("user-message")
        self._content = content
        self._pending = pending
        self.header.set_timestamp(None if pending else posted_at)
        self._images = images or []
        self.history_entry_id = history_entry_id

    def get_content(self) -> str:
        return self._content

    def update_content(self, content: str) -> None:
        self._content = content
        try:
            content_widget = self.query_one(".user-message-content", NoMarkupStatic)
            content_widget.update(content)
        except Exception:
            pass

    @property
    def pending(self) -> bool:
        return self._pending

    def compose(self) -> ComposeResult:
        self.set_class(
            ascii_chrome_enabled()
            or os.environ.get("NO_COLOR") is not None
            or bool(self.app.console.no_color),
            "plain-user-accent",
        )
        with Vertical(classes="user-message-wrapper"):
            self.header.set_timestamp(None if self._pending else self.posted_at)
            yield self.header
            with Horizontal(classes="user-message-container"):
                yield NonSelectableStatic(">", classes="user-message-prompt")
                yield NoMarkupStatic(self._content, classes="user-message-content")
            if self._images:
                with Vertical(classes="user-message-attachments"):
                    for image in self._images:
                        yield UserMessageAttachment(image)
            if self.SHOW_SEPARATOR:
                yield ExpandingSeparator(classes="user-message-separator")
            if self._pending:
                self.add_class("pending")

    def on_mount(self) -> None:
        self._apply_accent_tint()
        self.watch(self.app, "theme", self._refresh_accent_tint, init=False)

    def _refresh_accent_tint(self) -> None:
        # App updates its CSS variables after notifying theme watchers.
        self.call_after_refresh(self._apply_accent_tint)

    def _apply_accent_tint(self) -> None:
        wrapper = self.query_one(".user-message-wrapper")
        if self.has_class("plain-user-accent"):
            wrapper.styles.background = None
            return
        accent = Color.parse(self.app.theme_variables["accent"])
        # Textual treats ANSI destinations as opaque even with TCSS alpha.
        # Resolve their theme-provided RGB channels before applying the tint.
        if accent.ansi is not None:
            base = Color.parse(
                "ansi_black" if self.app.current_theme.dark else "ansi_white"
            )
            wrapper.styles.background = Color(base.r, base.g, base.b).blend(
                Color(accent.r, accent.g, accent.b), 0.08
            )
        else:
            wrapper.styles.background = None

    @staticmethod
    def _attachment_label(attachment: ImageAttachment) -> str:
        return _attachment_label(attachment)

    @staticmethod
    def _format_display_path(path: Path) -> str:
        return _format_display_path(path)

    async def set_pending(self, pending: bool) -> None:
        if pending == self._pending:
            return

        self._pending = pending
        self.header.set_timestamp(None if pending else self.posted_at)

        if pending:
            self.add_class("pending")
            return

        self.remove_class("pending")

    def reconcile_timestamp(self, posted_at: datetime | None) -> None:
        """Apply history's authority without stamping optimistic or queued input."""
        self.posted_at = posted_at
        self.header.set_timestamp(None if self._pending else posted_at)

    def set_show_message_timestamps(self, show: bool) -> None:
        self.header.set_timestamp(
            None if self._pending else self.posted_at, show_message_timestamps=show
        )

    def set_show_separator(self, show: bool) -> None:
        self.set_class(not show, "no-separator")

    def on_resize(self) -> None:
        self.header._refresh_time()

    def set_follows_previous(self, follows: bool) -> None:
        self.set_class(follows, "follows-user")


class QueueHeaderMessage(Static):
    DEFAULT_LABEL = "» Queued"
    PAUSED_LABEL = f"» Queued — press {shortcut('Enter')} to send, type to add"

    def __init__(self, *, paused: bool = False) -> None:
        super().__init__()
        self.add_class("queue-header-message")
        self._paused = paused
        self._label_widget: NoMarkupStatic | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="queue-header-container"):
            self._label_widget = NoMarkupStatic(
                shortcut_hint(self._current_label()), classes="queue-header-content"
            )
            yield self._label_widget
            yield ExpandingSeparator(classes="queue-header-separator")

    def set_paused(self, paused: bool) -> None:
        if paused == self._paused:
            return
        self._paused = paused
        if self._label_widget is not None:
            self._label_widget.update(shortcut_hint(self._current_label()))

    def _current_label(self) -> str:
        return self.PAUSED_LABEL if self._paused else self.DEFAULT_LABEL


class SlashCommandMessage(UserMessage):
    PROMPT_CHAR = "You"
    SHOW_SEPARATOR = False

    def __init__(self, content: str, pending: bool = False) -> None:
        super().__init__(
            content if content.startswith("/") else f"/{content}", pending=pending
        )
        self.add_class("slash-command-message")


class StreamingMessageBase(Static):
    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content
        self._markdown: Markdown | None = None
        self._stream: MarkdownStream | None = None
        self._write_timer: Timer | None = None
        self._content_initialized = False
        self._to_write_buffer = ""
        self._frame_delay = STREAM_WRITE_FRAME_SECONDS
        self._last_frame_started: float | None = None
        self._list_engine: SimpleListStreamEngine | None = None

    def _get_markdown(self) -> Markdown:
        if self._markdown is None:
            raise RuntimeError(
                "Markdown widget not initialized. compose() must be called first."
            )
        return self._markdown

    def _ensure_stream(self) -> MarkdownStream:
        if self._stream is None:
            from textual.widgets import Markdown

            self._stream = Markdown.get_stream(self._get_markdown())
        return self._stream

    def _ensure_list_engine(self) -> SimpleListStreamEngine | None:
        """The fast path needs a live Markdown widget; doubles without one keep
        the plain stream path.
        """
        if self._markdown is None:
            return None
        engine = getattr(self, "_list_engine", None)
        if engine is None:
            engine = SimpleListStreamEngine(self)
            self._list_engine = engine
        return engine

    async def _write_chunk(self, text: str) -> None:
        """Route one buffered write to the simple-list fast path or the stream."""
        engine = self._ensure_list_engine()
        if engine is not None and await engine.write(text, self._content):
            return
        stream = self._ensure_stream()
        await stream.write(text)

    def _is_chat_at_bottom(self) -> bool:
        try:
            chat = cast("ChatScroll", self.app.query_one("#chat"))
            return chat.is_at_bottom
        except Exception:
            return True

    async def append_content(self, content: str) -> None:
        if not content:
            return

        self._content += content

        if not self._should_write_content():
            return

        self._to_write_buffer += content
        if self._is_chat_at_bottom() and self._write_timer is None:
            self._write_timer = self.set_timer(
                self._frame_delay, self._flush_write_buffer
            )

    async def _flush_write_buffer(self) -> None:
        self._write_timer = None
        if (
            not self._to_write_buffer
            or not self._should_write_content()
            or not self._is_chat_at_bottom()
        ):
            return

        self._adapt_frame_delay(monotonic())
        to_write = self._to_write_buffer
        self._to_write_buffer = ""
        STREAM_WRITE_COUNTERS.flush_count += 1
        STREAM_WRITE_COUNTERS.flush_bytes += len(to_write.encode("utf-8"))
        await self._write_chunk(to_write)

    def _adapt_frame_delay(self, started: float) -> None:
        """Steer the scheduled delay so frames land every STREAM_WRITE_FRAME_SECONDS.

        A late frame means the pump was busy elsewhere; shortening the delay lets
        the next frame take its turn sooner, which is what keeps the *achieved*
        period near the target. The clamp stops back-pressure from stretching the
        transcript: the scheduled interval never exceeds the hard cap.
        """
        previous = self._last_frame_started
        self._last_frame_started = started
        if previous is None:
            return

        achieved = started - previous
        if achieved > STREAM_WRITE_FRAME_GAP_RESET_SECONDS:
            return

        correction = (STREAM_WRITE_FRAME_SECONDS - achieved) * (
            STREAM_WRITE_FRAME_CORRECTION_GAIN
        )
        if abs(correction) < STREAM_WRITE_FRAME_DEADBAND_SECONDS:
            return

        self._frame_delay = min(
            STREAM_WRITE_FRAME_SECONDS_MAX,
            max(STREAM_WRITE_FRAME_SECONDS_MIN, self._frame_delay + correction),
        )

    def _cancel_write_timer(self) -> None:
        if self._write_timer is not None:
            self._write_timer.stop()
            self._write_timer = None

    async def write_initial_content(self) -> None:
        if self._content_initialized:
            return
        self._content_initialized = True
        if self._content and self._should_write_content():
            self._cancel_write_timer()
            STREAM_WRITE_COUNTERS.initial_write_count += 1
            STREAM_WRITE_COUNTERS.initial_write_bytes += len(
                self._content.encode("utf-8")
            )
            await self._write_chunk(self._content)
            self._to_write_buffer = ""

    async def stop_stream(self) -> None:
        self._cancel_write_timer()
        if self._to_write_buffer and self._should_write_content():
            STREAM_WRITE_COUNTERS.final_flush_count += 1
            STREAM_WRITE_COUNTERS.final_flush_bytes += len(
                self._to_write_buffer.encode("utf-8")
            )
            await self._write_chunk(self._to_write_buffer)
        self._to_write_buffer = ""

        if self._stream is None:
            return

        await self._stream.stop()
        self._stream = None

    def _on_unmount(self) -> None:
        # A widget's MarkdownStream._run task pends forever until stop() and pins
        # the whole widget tree; unmount is the one hook every removal path
        # routes through (Textual dispatches it from the pump's message-loop exit).
        # Cancel without awaiting: stop() is async and the Widget._on_unmount
        # signature it overrides is sync — the cancelled task ends on its own.
        self._cancel_write_timer()
        self._to_write_buffer = ""
        self._list_engine = None
        if self._stream is not None:
            stream, self._stream = self._stream, None
            if stream._task is not None:
                stream._task.cancel()

    def _should_write_content(self) -> bool:
        return True

    def get_content(self) -> str:
        return self._content

    def is_stripped_content_empty(self) -> bool:
        return self._content.strip() == ""


class AssistantMessage(StreamingMessageBase):
    def __init__(
        self,
        content: str,
        *,
        posted_at: datetime | None = None,
        turn_duration_ms: float | None = None,
        show_message_timestamps: bool = True,
    ) -> None:
        super().__init__(content)
        self.header = MessageHeader(
            "Assistant",
            posted_at=posted_at,
            turn_duration_ms=turn_duration_ms,
            show_message_timestamps=show_message_timestamps,
        )
        self.add_class("assistant-message")

    def reconcile_timing(self, duration_ms: float | None) -> None:
        self.header.set_turn_duration(duration_ms)

    def on_resize(self) -> None:
        self.header._refresh_time()

    def reconcile_timestamp(self, posted_at: datetime | None) -> None:
        self.header.set_timestamp(posted_at)

    def set_show_message_timestamps(self, show: bool) -> None:
        self.header.set_timestamp(self.header.posted_at, show_message_timestamps=show)

    def compose(self) -> ComposeResult:
        from textual.widgets import Markdown

        markdown = Markdown("")
        self._markdown = markdown
        yield self.header
        yield markdown


class ReasoningMessage(ClickWithoutDragMixin, SpinnerMixin, StreamingMessageBase):
    SPINNER_TYPE = SpinnerType.PULSE
    SPINNING_TEXT = "Thinking"
    COMPLETED_TEXT = "Thought"

    def __init__(
        self,
        content: str,
        collapsed: bool = True,
        *,
        completed: bool = False,
        entry_id: str | None = None,
        expansion_state: EntryExpansionState | None = None,
    ) -> None:
        super().__init__(content)
        self.add_class("reasoning-message")
        self.entry_id = entry_id
        self._expansion_state = expansion_state
        self.collapsed = (
            expansion_state.register(entry_id)
            if entry_id is not None and expansion_state is not None
            else collapsed
        )
        self._indicator_widget: Static | None = None
        self._header_widget: Horizontal | None = None
        self.init_spinner()
        self._is_spinning = not completed

    def compose(self) -> ComposeResult:
        with Vertical(classes="reasoning-message-wrapper"):
            self._header_widget = DisclosureHeader(
                activate=self._toggle_collapsed, classes="reasoning-message-header"
            )
            with self._header_widget:
                self._indicator_widget = NonSelectableStatic(
                    self._spinner.current_frame()
                    if self._is_spinning
                    else chrome_glyph("checked"),
                    classes="reasoning-indicator",
                )
                yield self._indicator_widget
                self._status_text_widget = NoMarkupStatic(
                    self.SPINNING_TEXT if self._is_spinning else self.COMPLETED_TEXT,
                    classes="reasoning-collapsed-text",
                )
                yield self._status_text_widget
            from textual.widgets import Markdown

            markdown = Markdown("", classes="reasoning-message-content")
            markdown.display = not self.collapsed
            self._markdown = markdown
            yield markdown

    def on_mount(self) -> None:
        if self._is_spinning:
            self.start_spinner_timer()

    def on_resize(self) -> None:
        self.refresh_spinner()

    def stop_spinning(self, success: bool = True) -> None:
        super().stop_spinning(success)
        if self._indicator_widget:
            self._indicator_widget.remove_class("success", "error")
            self._indicator_widget.update(
                chrome_glyph(
                    "disclosure_closed" if self.collapsed else "disclosure_open"
                )
            )

    def _is_click_on_toggle(self, event: events.Click) -> bool:
        return self._is_click_within(event, self._header_widget)

    async def on_click(self, event: events.Click) -> None:
        if not self._is_click_on_toggle(event) or self._click_is_passive(event):
            return
        await self._toggle_collapsed()

    async def _toggle_collapsed(self) -> None:
        await self.set_collapsed(not self.collapsed)

    def _should_write_content(self) -> bool:
        return not self.collapsed

    async def set_collapsed(self, collapsed: bool) -> None:
        if self.collapsed == collapsed:
            return

        focused = self.screen.focused
        if (
            collapsed
            and self._markdown is not None
            and self._markdown.is_mounted
            and focused is not None
            and self._markdown in focused.ancestors_with_self
        ):
            assert self._header_widget is not None
            self._header_widget.focus()
        self.collapsed = collapsed
        if self.entry_id is not None and self._expansion_state is not None:
            self._expansion_state.set_collapsed(self.entry_id, collapsed)
        if self._indicator_widget and not self._is_spinning:
            self._indicator_widget.update(
                chrome_glyph("disclosure_closed" if collapsed else "disclosure_open")
            )
        if self._markdown:
            self._markdown.display = not collapsed
            if not collapsed and self._content:
                self._cancel_write_timer()
                if self._stream is not None:
                    await self._stream.stop()
                    self._stream = None
                await self._markdown.update("")
                engine = getattr(self, "_list_engine", None)
                if engine is not None:
                    engine.reset()
                await self._write_chunk(self._content)
                self._to_write_buffer = ""


class UserCommandMessage(Static):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.add_class("user-command-message")
        self._content = content

    def compose(self) -> ComposeResult:
        from textual.widgets import Markdown

        with Horizontal(classes="user-command-container"):
            yield ExpandingBorder(classes="user-command-border")
            with Vertical(classes="user-command-content"):
                yield Markdown(self._content)


class WhatsNewMessage(Static):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.add_class("whats-new-message")
        self._content = content

    def compose(self) -> ComposeResult:
        from textual.widgets import Markdown

        yield Markdown(self._content)


class CustomToolsDeprecationMessage(Static):
    def __init__(self, tool_names: list[str]) -> None:
        super().__init__()
        self.add_class("custom-tools-deprecation-message")
        names = ", ".join(f"`{name}`" for name in sorted(tool_names))
        replacement = "a skill" if len(tool_names) == 1 else "skills"
        self._content = (
            "**Support for custom tools will be deprecated soon.** "
            f"Ask Chartreux to help replace yours ({names}) with {replacement}."
        )

    def compose(self) -> ComposeResult:
        from textual.widgets import Markdown

        yield Markdown(self._content)


class InterruptMessage(Static):
    def __init__(self) -> None:
        super().__init__()
        self.add_class("interrupt-message")

    def compose(self) -> ComposeResult:
        with Horizontal(classes="interrupt-container"):
            yield ExpandingBorder(classes="interrupt-border")
            yield NoMarkupStatic(
                "Interrupted — completed tool results are kept; in-progress effects may be partial. What next?",
                classes="interrupt-content",
            )


class ErrorRetryRequested(Message):
    """The retry affordance was activated (pointer click or keyboard).

    R1 (plan v0.5.6-recovery): posted by the "Retry — new turn" button on a
    terminal retryable error widget. The app routes it to the EXISTING retry
    operation — one injected new turn with the user-initiated cooldown reset —
    never resume_agent, never an automatic continuation.
    """


class ErrorMessage(Static):
    def __init__(
        self,
        error: str | Content,
        collapsed: bool = False,
        show_border: bool = True,
        retryable: bool = False,
    ) -> None:
        super().__init__()
        self.add_class("error-message")
        self._error = error
        self.collapsed = collapsed
        self._show_border = show_border
        self._retryable = retryable
        self._retry_activated = False
        self._content_widget: Static | None = None
        self._overflow_section: OverflowCollapsibleSection | None = None
        self._retry_button: Button | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="error-container"):
            if self._show_border:
                yield ExpandingBorder(classes="error-border")
            error = (
                self._error
                if isinstance(self._error, Content)
                else Content(clean_output(self._error))
            )
            text = Content("Error: ") + error if self._show_border else error
            plain = text.plain
            if _estimated_row_count(plain) > _ERROR_MAX_ROWS:
                # Oversized payload: keep a one-line summary plus the house
                # "+N lines" disclosure. Even when the caller asks for an
                # expanded start (collapsed=False), the body is capped by the
                # `.error-overflow-body` max-height, so the error can never
                # reclaim the viewport.
                head, _, _ = plain.partition("\n")
                body = plain
                self._content_widget = NoMarkupStatic(
                    head, classes="error-content error-head"
                )
                self._overflow_section = OverflowCollapsibleSection(
                    lambda: NoMarkupStatic(body, classes="error-overflow-body"),
                    collapsed_label=lines_label(_estimated_row_count(body) - 1),
                )
                if self._retryable:
                    # R1: the keyboard-accessible "Retry — new turn" affordance.
                    # A Button is focusable and activates with Enter/Space; the
                    # warning text above it (kept by the turn supervisor's hint)
                    # states that completed actions may repeat.
                    with Vertical(classes="error-body"):
                        yield self._content_widget
                        yield self._overflow_section
                        self._retry_button = Button(
                            "Retry — new turn",
                            classes="error-retry-button",
                            variant="primary",
                        )
                        yield self._retry_button
                else:
                    yield self._content_widget
                    yield self._overflow_section
            else:
                self._content_widget = NoMarkupStatic(text, classes="error-content")
                if self._retryable:
                    # R1: the keyboard-accessible "Retry — new turn" affordance.
                    # A Button is focusable and activates with Enter/Space; the
                    # warning text above it (kept by the turn supervisor's hint)
                    # states that completed actions may repeat.
                    with Vertical(classes="error-body"):
                        yield self._content_widget
                        self._retry_button = Button(
                            "Retry — new turn",
                            classes="error-retry-button",
                            variant="primary",
                        )
                        yield self._retry_button
                else:
                    yield self._content_widget

    def on_mount(self) -> None:
        # OverflowCollapsibleSection always starts collapsed; honor the caller's
        # collapsed=False here, after the children are mounted and the lazy body
        # can be built safely.
        if not self.collapsed and self._overflow_section is not None:
            self._overflow_section.set_collapsed(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button is not self._retry_button:
            return
        event.stop()
        if self._retry_activated:
            # Double activation starts at most one turn (R1).
            return
        self._retry_activated = True
        event.button.disabled = True
        self.post_message(ErrorRetryRequested())

    def set_collapsed(self, collapsed: bool) -> None:
        if self._overflow_section is not None:
            self._overflow_section.set_collapsed(collapsed)


class HookRunContainer(Vertical):
    def __init__(self) -> None:
        super().__init__(classes="hook-run-container")
        self.display = False

    async def add_message(self, widget: HookSystemMessageLine) -> None:
        await self.mount(widget)
        self.display = True


_HOOK_SEVERITY_ICONS: dict[HookSeverity, str] = {
    HookSeverity.OK: "success",
    HookSeverity.WARNING: "warning",
    HookSeverity.ERROR: "error",
}


class HookSystemMessageLine(Static):
    def __init__(
        self,
        hook_name: str,
        content: str,
        severity: HookSeverity = HookSeverity.WARNING,
    ) -> None:
        super().__init__()
        self.add_class("hook-system-message")
        self.add_class(f"hook-severity-{severity}")
        self._hook_name = hook_name
        self._content = content
        self._severity = severity

    def compose(self) -> ComposeResult:
        icon = chrome_glyph(
            _HOOK_SEVERITY_ICONS.get(
                self._severity, _HOOK_SEVERITY_ICONS[HookSeverity.WARNING]
            )
        )
        wording = {
            HookSeverity.OK: "Saved:",
            HookSeverity.WARNING: "Warning:",
            HookSeverity.ERROR: "Failed:",
        }.get(self._severity, "Warning:")
        with Horizontal(classes="hook-system-container"):
            yield NonSelectableStatic(icon, classes="hook-system-icon")
            yield NoMarkupStatic(
                f"[{self._hook_name}] {wording} {self._content}",
                classes="hook-system-content",
            )


class WarningMessage(Static):
    def __init__(self, message: str, show_border: bool = True) -> None:
        super().__init__()
        self.add_class("warning-message")
        self._message = message
        self._show_border = show_border

    def compose(self) -> ComposeResult:
        with Horizontal(classes="warning-container"):
            if self._show_border:
                yield ExpandingBorder(classes="warning-border")
            yield NoMarkupStatic(
                f"! Warning: {self._message}", classes="warning-content"
            )
