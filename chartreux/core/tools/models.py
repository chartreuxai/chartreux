from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel

from chartreux.permissions import PermissionScope as PermissionScope


class ToolPermissionError(Exception):
    pass


class ToolPermission(StrEnum):
    ALWAYS = auto()
    NEVER = auto()

    @classmethod
    def by_name(cls, name: str) -> ToolPermission:
        try:
            return ToolPermission(name.upper())
        except ValueError:
            raise ToolPermissionError(
                f"Invalid tool permission: {name}. Must be one of {list(cls)}"
            )


class PermissionContext(BaseModel):
    permission: ToolPermission
    reason: str | None = None
    denial_kind: (
        Literal[
            "out_of_root",
            "sensitive",
            "path_rule",
            "plan_scope",
            "parent_ceiling",
            "tool_policy",
        ]
        | None
    ) = None
