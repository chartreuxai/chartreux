from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from textual.screen import Screen


@dataclass(frozen=True, slots=True)
class OnboardingHost:
    """Small host adapter used by onboarding-only screens."""

    show_theme: Callable[[], None]
    confirm_theme: Callable[[str], None]
    cancel: Callable[[], None]


class OnboardingScreen(Screen[None]):
    """Onboarding screen that delegates navigation and exit to its host."""

    def __init__(self, host: OnboardingHost) -> None:
        super().__init__()
        self.host = host

    def action_cancel(self) -> None:
        self.host.cancel()
