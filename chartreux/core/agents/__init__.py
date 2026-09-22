from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chartreux.core.agents.manager import AgentManager
    from chartreux.core.agents.models import (
        ADVISOR,
        BUILTIN_SUBAGENTS,
        REVIEWER,
        WORKER,
        AgentProfile,
        AgentSafety,
        AgentType,
    )

__all__ = [
    "ADVISOR",
    "BUILTIN_SUBAGENTS",
    "REVIEWER",
    "WORKER",
    "AgentManager",
    "AgentProfile",
    "AgentSafety",
    "AgentType",
]
_MAPPING: dict[str, tuple[str, str]] = {
    "ADVISOR": ("chartreux.core.agents.models", "ADVISOR"),
    "AgentManager": ("chartreux.core.agents.manager", "AgentManager"),
    "AgentProfile": ("chartreux.core.agents.models", "AgentProfile"),
    "AgentSafety": ("chartreux.core.agents.models", "AgentSafety"),
    "AgentType": ("chartreux.core.agents.models", "AgentType"),
    "BUILTIN_SUBAGENTS": ("chartreux.core.agents.models", "BUILTIN_SUBAGENTS"),
    "REVIEWER": ("chartreux.core.agents.models", "REVIEWER"),
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
