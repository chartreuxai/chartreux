from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from chartreux.core.config.models import ThinkingLevel
from chartreux.core.tools.models import ToolPermission


class LaunchToolOverride(BaseModel):
    """The only tool settings accepted for a subagent launch."""

    model_config = ConfigDict(extra="forbid")

    permission: ToolPermission | None = None
    allowlist: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _omit_explicit_nulls(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if item is not None}
        return value


class LaunchConfig(BaseModel):
    """Semantic launch choices, preserving supplied fields for accumulated state."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    instructions: str | None = None
    system_prompt_id: str | None = None
    thinking: ThinkingLevel | None = None
    enabled_tools: list[str] | None = None
    disabled_tools: list[str] | None = None
    tools: dict[str, LaunchToolOverride] | None = None

    @model_validator(mode="before")
    @classmethod
    def _omit_explicit_nulls(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if item is not None}
        return value
