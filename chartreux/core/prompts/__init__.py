from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum, auto
from pathlib import Path

from chartreux import CHARTREUX_ROOT
from chartreux.core.config.harness_files import get_harness_files_manager
from chartreux.core.policy.models import ResourceKind
from chartreux.core.policy.resources import ResourceLoader

PROMPTS_DIR = CHARTREUX_ROOT / "core" / "prompts"


class Prompt(StrEnum):
    @property
    def path(self) -> Path:
        return (PROMPTS_DIR / self.value).with_suffix(".md")

    def read(self) -> str:
        content = (
            ResourceLoader({ResourceKind.IDENTITY: PROMPTS_DIR})
            .resolve(ResourceKind.IDENTITY, self.value)
            .content
        )
        assert content is not None
        return content.strip()


class UtilityPrompt(Prompt):
    AGENTS_DOC = auto()
    COMPACT = auto()
    COMPACT_SUMMARY_PREFIX = auto()
    COMPACT_SYSTEM = auto()
    DANGEROUS_DIRECTORY = auto()
    PROJECT_CONTEXT = auto()
    SESSION_TITLE = auto()
    WORKTREE_NAME = auto()


class MissingPromptFileError(ValueError):
    def __init__(
        self,
        setting_name: str,
        prompt_id: str,
        builtin_ids: Iterable[str],
        custom_dirs: Iterable[Path],
        custom_ids: Iterable[str],
    ) -> None:
        builtin_hint = ", ".join('"' + i + '"' for i in builtin_ids)
        dirs_hint = " or ".join(str(d) for d in custom_dirs) or "<no prompt dirs>"
        custom_hint = ", ".join('"' + i + '"' for i in custom_ids) or "<none>"
        super().__init__(
            f"Invalid {setting_name} value: '{prompt_id}'. "
            f"Must be one of the available prompts ({builtin_hint}), "
            f"or correspond to a .md file in {dirs_hint} (available: {custom_hint})"
        )
        self.setting_name = setting_name
        self.prompt_id = prompt_id


def _validate_prompt_id(prompt_id: str, setting_name: str) -> None:
    if (
        not prompt_id
        or prompt_id in {".", ".."}
        or "/" in prompt_id
        or "\\" in prompt_id
    ):
        raise ValueError(
            f"Invalid {setting_name} value: '{prompt_id}' must be a bare filename "
            "without path separators"
        )


def custom_prompt_ids() -> set[str]:
    """Enumerate the same custom prompt directories used by load_prompt."""
    mgr = get_harness_files_manager()
    return {
        p.stem
        for d in (*mgr.project_prompts_dirs, *mgr.user_prompts_dirs)
        for p in d.glob("*.md")
        if p.is_file()
    }


def load_prompt(
    prompt_id: str, *, setting_name: str, builtins: Mapping[str, Path]
) -> str:
    """Load a utility prompt through the strict resource loader.

    Package 16b retired the enum-based full-skeleton prompt axis: the
    ``SystemPrompt`` enum, ``load_system_prompt``, and the legacy unqualified
    branch (project/user custom directories shadowing shipped prompts) are
    gone. What survives is the utility/compaction surface: a built-in id
    resolves from the shipped ``builtins`` map, a custom id resolves from the
    user prompts directory, and the reserved built-in names cannot be
    shadowed.
    """
    _validate_prompt_id(prompt_id.split(":", 1)[-1], setting_name)
    mgr = get_harness_files_manager()
    builtin_map = {name.lower(): path for name, path in builtins.items()}
    user_dirs = mgr.user_prompts_dirs
    loader = ResourceLoader(
        {ResourceKind.IDENTITY: user_dirs[0] if user_dirs else Path("/nonexistent")},
        {ResourceKind.IDENTITY: builtin_map},
    )
    resource = loader.resolve(ResourceKind.IDENTITY, prompt_id)
    if resource.content is not None:
        return resource.content
    raise MissingPromptFileError(
        setting_name,
        prompt_id,
        tuple(builtin_map),
        user_dirs,
        tuple(sorted(custom_prompt_ids())),
    )


__all__ = [
    "PROMPTS_DIR",
    "MissingPromptFileError",
    "Prompt",
    "UtilityPrompt",
    "custom_prompt_ids",
    "load_prompt",
]
