from __future__ import annotations

from enum import StrEnum, auto

__all__ = ["PermissionScope"]


class PermissionScope(StrEnum):
    COMMAND_PATTERN = auto()
    OUTSIDE_DIRECTORY = auto()
    FILE_PATTERN = auto()
