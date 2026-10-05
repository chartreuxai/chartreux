"""Plain context text shared by agent details and the session status line."""

from __future__ import annotations

from typing import Literal

_THOUSAND = 1_000
_MILLION = 1_000_000


def _format_token_count(tokens: int) -> str:
    # Truncate thousands, but show millions to one decimal place.
    if tokens >= _MILLION:
        return f"{tokens / _MILLION:.1f}M"
    if tokens >= _THOUSAND:
        return f"{tokens // _THOUSAND}k"
    return str(tokens)


def format_context(
    context_tokens: int | None,
    auto_compact_threshold: int | None,
    *,
    style: Literal["tokens", "tokens-percent"] = "tokens-percent",
    compacting: bool = False,
    last_recorded: bool = False,
) -> str:
    """Format usage against the automatic-compaction trigger, not a hard cap.

    None (or the legacy negative usage sentinel) means unknown usage. While
    compacting, even a retained measurement is hidden; after compaction the
    caller must keep supplying unknown usage until conversation stats arrive.
    Zero is displayed as zero, including for fresh stats: it does not establish
    that a request was measured. This function never infers measurement from
    cumulative usage or other counters.

    A missing threshold is unknown; a nonpositive threshold explicitly means
    automatic compaction is off. Percentages use unabridged counts, round to a
    whole percent, and may exceed 100. The tokens style only omits percentages.
    For tombstones, callers supply the retained snapshot and set last_recorded;
    this function does not keep state or retrieve historical measurements.
    """
    if style not in {"tokens", "tokens-percent"}:
        raise ValueError(f"Unknown context style: {style}")

    usage_known = not compacting and context_tokens is not None and context_tokens >= 0
    usage = (
        _format_token_count(context_tokens)
        if usage_known and context_tokens is not None
        else "—"
    )
    if auto_compact_threshold is None:
        text = f"{usage}/—"
    elif auto_compact_threshold <= 0:
        text = f"{usage} (auto-compact off)"
    else:
        text = f"{usage}/{_format_token_count(auto_compact_threshold)}"
        if style == "tokens-percent" and usage_known and context_tokens is not None:
            percent = context_tokens * 100 / auto_compact_threshold
            text += f" ({percent:.0f}%)"

    if last_recorded:
        text += " (last recorded)"
    return text
