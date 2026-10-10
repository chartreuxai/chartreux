from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from chartreux.core.config.types import ConfigSaveResult


class RootGrantPort(Protocol):
    """Runtime-owned callback that applies an approved session root grant."""

    async def grant_root(self, session_id: str, root: Path) -> None: ...

    async def save_root(
        self, session_id: str, root: Path, expected_revision: str
    ) -> ConfigSaveResult:
        """Persist one approved grant for the calling session's project.

        The runtime routes the save to the root orchestrator keyed by the
        registered caller session's cwd; child orchestrators never write. The
        result reports persistence honestly, including conflicts, write
        failures, and durability uncertainty.
        """
        ...


RootGrantCallback = Callable[[str, Path], object]
