from __future__ import annotations

from datetime import datetime

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget

from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.duration_display import format_duration
from chartreux.ui.widgets.no_markup_static import NonSelectableStatic


def local_now() -> datetime:
    """Renderer clock, independently pinnable from the canonical posting clock."""
    return datetime.now().astimezone()


def format_message_timestamp(
    posted_at: datetime | None, *, now: datetime | None = None
) -> str:
    """Format canonical posting time in the local timezone; never backfill history."""
    if posted_at is None:
        return ""
    local = posted_at.astimezone()
    today = (now if now is not None else local_now()).astimezone().date()
    return local.strftime("%H:%M" if local.date() == today else "%Y-%m-%d %H:%M")


class MessageHeader(Horizontal):
    """Metadata only; elapsed total survives width pressure before posting time."""

    DEFAULT_CSS = """
    MessageHeader {
        width: 100%;
        height: 1;
        overflow: hidden hidden;
    }
    MessageHeader .message-header-time {
        width: auto;
        height: 1;
        color: $text-muted;
        text-style: none;
        text-wrap: nowrap;
    }
    """

    def __init__(
        self,
        role: str,
        *,
        posted_at: datetime | None = None,
        turn_duration_ms: float | None = None,
        show_message_timestamps: bool = True,
    ) -> None:
        super().__init__()
        self.role = role
        self.posted_at = posted_at
        self.turn_duration_ms = turn_duration_ms
        self.show_message_timestamps = show_message_timestamps
        self._time = NonSelectableStatic("", classes="message-header-time")
        self.display = bool(self.metadata_for_width(10000))

    def compose(self) -> ComposeResult:
        yield self._time

    def set_timestamp(
        self, posted_at: datetime | None, *, show_message_timestamps: bool | None = None
    ) -> None:
        self.posted_at = posted_at
        if show_message_timestamps is not None:
            self.show_message_timestamps = show_message_timestamps
        self._refresh_time()

    def set_turn_duration(self, duration_ms: float | None) -> None:
        self.turn_duration_ms = duration_ms
        self._refresh_time()

    def metadata_for_width(self, width: int, *, now: datetime | None = None) -> str:
        if not self.show_message_timestamps:
            return ""
        time = format_message_timestamp(self.posted_at, now=now)
        total = (
            format_duration(self.turn_duration_ms) if self.role == "Assistant" else ""
        )
        metadata = (
            f"{time} {chrome_glyph('metadata_separator')} {total}"
            if time and total
            else time or total
        )
        if cell_len(metadata) <= width:
            return metadata
        return total if total and cell_len(total) <= width else ""

    def timestamp_for_width(self, width: int, *, now: datetime | None = None) -> str:
        return self.metadata_for_width(width, now=now)

    def _refresh_time(self) -> None:
        # Hidden rows have zero size; use the parent's available width so metadata
        # can reappear on preference changes and after a narrow-to-wide resize.
        width = (
            self.parent.content_size.width if isinstance(self.parent, Widget) else 10000
        )
        if self.styles.width is not None and self.styles.width.is_cells:
            width = min(width, int(self.styles.width.value))
        if not width:
            return
        metadata = self.metadata_for_width(width)
        self._time.update(metadata)
        self.display = bool(metadata)

    def on_mount(self) -> None:
        self._refresh_time()

    def on_resize(self) -> None:
        self._refresh_time()
