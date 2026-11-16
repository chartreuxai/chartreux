"""Plain per-call cache-hit text shared by the status line and agent rows."""

from __future__ import annotations


def format_cache_hit(
    cached_tokens: int | None, prompt_tokens: int | None, *, reported: bool
) -> str:
    """Format one LLM call's cache-hit share of its prompt tokens.

    ``reported`` False means the provider did not report cache fields, so the
    cached count is unknown rather than a reported zero; a missing or
    nonpositive prompt leaves no denominator to measure against. Both read as
    ``—`` (unknown, ``-`` in ASCII chrome), never as a claimed ``0%``. Zero
    cached tokens on a reported call is a real 0% and renders as such.
    """
    if (
        not reported
        or cached_tokens is None
        or prompt_tokens is None
        or prompt_tokens <= 0
    ):
        return "—"
    return f"{cached_tokens * 100 / prompt_tokens:.0f}%"
