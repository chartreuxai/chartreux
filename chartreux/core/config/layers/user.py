from __future__ import annotations

from pathlib import Path

from chartreux.core.config.layers._base import BaseTomlConfigLayer
from chartreux.core.paths._vibe_home import CHARTREUX_HOME


class UserConfigLayer(BaseTomlConfigLayer):
    """Reads the user-level TOML config file. Always trusted.

    Defaults to ``~/.chartreux/config.toml`` (via CHARTREUX_HOME).
    Pass an explicit ``path`` for testing.
    """

    def __init__(self, *, path: Path | None = None, name: str = "user-toml") -> None:
        super().__init__(name=name)
        self._path = path or (CHARTREUX_HOME.path / "config.toml")

    @property
    def _target_path(self) -> Path:
        return self._path

    async def _check_trust(self) -> bool:
        return True
