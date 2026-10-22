from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chartreux.core.agents.manager import AgentManager

__all__ = ["AgentManager"]
_MAPPING: dict[str, tuple[str, str]] = {
    "AgentManager": ("chartreux.core.agents.manager", "AgentManager")
}


def __getattr__(name: str) -> object:
    if name in _MAPPING:
        import importlib

        module_name, attr_name = _MAPPING[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
