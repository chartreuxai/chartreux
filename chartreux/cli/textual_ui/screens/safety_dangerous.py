"""v0.8.0 A-P1: the persistent DANGEROUS banner.

Entering SafetyLevel DANGEROUS is interactive-TUI-only (unguarded-mode-design.md
:49-65, surfaced as ``/safety dangerous`` per the amendment Q1/Q2). v0.8.2 W2
removed the typed-phrase/confirmation ceremony screens, so the transition is
requested directly. The non-dismissible banner rides outside the scrolling
transcript while the level is active; if it cannot be displayed, activation is
refused.
"""

from __future__ import annotations

from textual.widgets import Static


class DangerousBanner(Static):
    """The NON-DISMISSIBLE DANGEROUS banner, outside the scrolling transcript."""

    def __init__(self) -> None:
        super().__init__(
            "DANGEROUS MODE ACTIVE — all guards suspended for this session and its subagents",
            id="dangerous-banner",
        )
