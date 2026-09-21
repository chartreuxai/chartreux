from __future__ import annotations

import copy
from typing import Any

from chartreux.core.config.fingerprint import create_dict_fingerprint
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.types import LayerConfigSnapshot


class LaunchOverridesLayer(ConfigLayer[RawConfig]):
    """Child-only, nonpersistent launch settings with highest merge precedence."""

    NAME = "launch-overrides"

    def __init__(self, *, data: dict[str, Any] | None = None, name: str = NAME) -> None:
        super().__init__(name=name)
        self._data = copy.deepcopy(data or {})

    async def _check_trust(self) -> bool:
        return True

    async def _build_config_snapshot(self) -> LayerConfigSnapshot:
        data = copy.deepcopy(self._data)
        return LayerConfigSnapshot(data=data, fingerprint=create_dict_fingerprint(data))

    def _accept_loaded_state(self, staged: ConfigLayer[RawConfig]) -> None:
        super()._accept_loaded_state(staged)
        if isinstance(staged, LaunchOverridesLayer) and self._data != staged._data:
            self._data = copy.deepcopy(staged._data)

    async def _save_to_store(self, next_config: RawConfig) -> str:
        self._data = copy.deepcopy(next_config.model_dump())
        return create_dict_fingerprint(self._data)
