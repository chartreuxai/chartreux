from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, model_validator

from chartreux.core.dangerous_commands import DangerousCommandProposal
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
            "dangerous_command",
        ]
        | None
    ) = None
    # A grantable dangerous-command denial carries the typed whole-call
    # proposal; every other denial kind never carries one.
    dangerous_command_proposal: DangerousCommandProposal | None = None

    @model_validator(mode="after")
    def _check_dangerous_command_proposal(self) -> PermissionContext:
        if self.dangerous_command_proposal is not None:
            if (
                self.denial_kind != "dangerous_command"
                or self.permission != ToolPermission.NEVER
            ):
                raise ValueError(
                    "dangerous_command_proposal requires a NEVER denial of kind "
                    "'dangerous_command'"
                )
        elif self.denial_kind == "dangerous_command":
            raise ValueError(
                "a 'dangerous_command' denial requires a grantable proposal"
            )
        return self
