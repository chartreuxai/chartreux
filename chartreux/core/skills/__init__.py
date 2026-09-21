from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chartreux.core.skills.manager import SkillManager
    from chartreux.core.skills.models import SkillConfigIssue, SkillInfo, SkillMetadata
    from chartreux.core.skills.parser import SkillParseError

__all__ = [
    "SkillConfigIssue",
    "SkillInfo",
    "SkillManager",
    "SkillMetadata",
    "SkillParseError",
]

_MAPPING: dict[str, tuple[str, str]] = {
    "SkillManager": ("chartreux.core.skills.manager", "SkillManager"),
    "SkillConfigIssue": ("chartreux.core.skills.models", "SkillConfigIssue"),
    "SkillInfo": ("chartreux.core.skills.models", "SkillInfo"),
    "SkillMetadata": ("chartreux.core.skills.models", "SkillMetadata"),
    "SkillParseError": ("chartreux.core.skills.parser", "SkillParseError"),
}


def __getattr__(name: str) -> object:
    if name in _MAPPING:
        import importlib

        module_name, attr_name = _MAPPING[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
