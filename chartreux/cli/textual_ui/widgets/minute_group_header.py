from __future__ import annotations

from datetime import datetime

from chartreux.cli.textual_ui.widgets.message_header import (
    MinuteKey,
    format_group_header_timestamp,
)
from chartreux.ui.widgets.no_markup_static import NonSelectableStatic


class MinuteGroupHeader(NonSelectableStatic):
    """The wall-clock minute stamp above the transcript records it groups.

    A record that renders its own header (a user/assistant message) opens the
    minute instead, so this widget marks only the minutes whose records would
    otherwise carry no time of their own.
    """

    DEFAULT_CSS = """
    MinuteGroupHeader {
        width: 100%;
        height: 1;
        color: $text-muted;
        text-style: none;
    }
    """

    def __init__(self, minute: MinuteKey, *, now: datetime | None = None) -> None:
        label = format_group_header_timestamp(minute, now=now)
        super().__init__(label)
        self.minute = minute
        self.label = label

    def set_minute(self, minute: MinuteKey, *, now: datetime | None = None) -> None:
        self.minute = minute
        self.label = format_group_header_timestamp(minute, now=now)
        self.update(self.label)

    def refresh_label(self) -> None:
        """Re-apply the day-dependent format; the minute itself did not move."""
        label = format_group_header_timestamp(self.minute)
        if label != self.label:
            self.label = label
            self.update(label)
