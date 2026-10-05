"""Static elapsed-duration text shared by transcript metadata."""

from __future__ import annotations

import math

_MIN_DISPLAY_SECONDS = 0.1
_TENTHS_PER_MINUTE = 600
_MINUTES_PER_HOUR = 60


def format_duration(duration_ms: float | None) -> str:
    """Format known milliseconds, rounding before splitting minutes and hours."""
    if duration_ms is None or not math.isfinite(duration_ms) or duration_ms < 0:
        return ""
    seconds = duration_ms / 1000
    if seconds < _MIN_DISPLAY_SECONDS:
        return "<0.1s"
    tenths = math.floor(seconds * 10 + 0.5)
    if tenths < _TENTHS_PER_MINUTE:
        return f"{tenths / 10:.1f}s"
    whole = math.floor(seconds + 0.5)
    minutes, seconds = divmod(whole, 60)
    if minutes < _MINUTES_PER_HOUR:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"
