from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chartreux.core.agents.manager import AgentManager
    from chartreux.core.agents.models import (
        BUILTIN_SUBAGENTS,
        WORKER,
        AgentProfile,
        AgentSafety,
        AgentType,
    )

__all__ = [
    "BUILTIN_SUBAGENTS",
    "WORKER",
    "AgentManager",
    "AgentProfile",
    "AgentSafety",
    "AgentType",
]
_MAPPING: dict[str, tuple[str, str]] = {
    "AgentManager": ("chartreux.core.agents.manager", "AgentManager"),
    "AgentProfile": ("chartreux.core.agents.models", "AgentProfile"),
    "AgentSafety": ("chartreux.core.agents.models", "AgentSafety"),
    "AgentType": ("chartreux.core.agents.models", "AgentType"),
    "BUILTIN_SUBAGENTS": ("chartreux.core.agents.models", "BUILTIN_SUBAGENTS"),
    "WORKER": ("chartreux.core.agents.models", "WORKER"),
}


def __getattr__(name: str) -> object:
    if name in _MAPPING:
        import importlib

        module_name, attr_name = _MAPPING[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
