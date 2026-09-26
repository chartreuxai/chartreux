from __future__ import annotations

import fnmatch

from chartreux.core.tools.models import (
    PermissionContext as PermissionContext,
    PermissionScope as PermissionScope,
)


def wildcard_match(text: str, pattern: str) -> bool:
    """If pattern ends with " *", trailing args are optional (match with or without)."""
    if fnmatch.fnmatch(text, pattern):
        return True
    if pattern.endswith(" *") and fnmatch.fnmatch(text, pattern[:-2]):
        return True
    return False
