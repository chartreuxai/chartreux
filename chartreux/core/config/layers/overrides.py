from __future__ import annotations

import copy
from typing import Any

from chartreux.core.config.fingerprint import create_dict_fingerprint
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.types import LayerConfigSnapshot


class OverridesLayer(ConfigLayer[RawConfig]):
    """Highest-priority layer wrapping an arbitrary dict passed at construction.

    Always trusted and read-only.
    Used by CLI and ACP entry points to inject runtime overrides.
    """

    NAME = "overrides"

    def __init__(self, *, data: dict[str, Any], name: str = NAME) -> None:
        super().__init__(name=name)
        self._data = data

    async def _check_trust(self) -> bool:
        return True

    async def _build_config_snapshot(self) -> LayerConfigSnapshot:
        data = copy.deepcopy(self._data)
        fingerprint = create_dict_fingerprint(data)
        return LayerConfigSnapshot(data=data, fingerprint=fingerprint)

    def _accept_loaded_state(self, staged: ConfigLayer[RawConfig]) -> None:
        super()._accept_loaded_state(staged)
        if isinstance(staged, OverridesLayer) and self._data != staged._data:
            # For an in-memory layer the backing store is part of acceptance.
            # Adopting only its cache resurrects old overrides on explicit reload.
            self._data = copy.deepcopy(staged._data)

    async def _save_to_store(self, next_config: RawConfig) -> str:
        data = copy.deepcopy(next_config.model_dump())
        self._data = data
        return create_dict_fingerprint(data)
