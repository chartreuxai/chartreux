from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import fnmatch

from chartreux.core.tools.models import (
    PermissionContext as PermissionContext,
    PermissionScope as PermissionScope,
)

#: Live getter for the session's DANGEROUS authority flag (v0.8.0 A-P1). The
#: agent loop binds it around every public entry point to its authority
#: state's level, so mid-session transitions stay visible; a context without
#: a bound getter fails closed to guarded (not dangerous).
_dangerous_mode_getter: ContextVar[Callable[[], bool] | None] = ContextVar(
    "chartreux_dangerous_mode_getter", default=None
)


def dangerous_mode_active() -> bool:
    """Whether the current session runs at SafetyLevel DANGEROUS.

    v0.8.0 A-P1 enforcement seam: every guard layer (shell policy, file
    authority, tool availability, grant decisions, hooks, completion and
    submission) keys its bypass on this read. UNREACHABLE-BY-STATE at this
    package's chunk-1 landing: no path can produce DANGEROUS before chunk 2's
    ceremony exists, and an absent or broken getter fails closed.
    """
    getter = _dangerous_mode_getter.get()
    if getter is None:
        return False
    try:
        return bool(getter())
    except Exception:
        return False


@contextmanager
def bind_dangerous_mode(getter: Callable[[], bool]) -> Iterator[None]:
    """Bind the live DANGEROUS getter for the wrapped execution context."""
    token = _dangerous_mode_getter.set(getter)
    try:
        yield
    finally:
        _dangerous_mode_getter.reset(token)


def wildcard_match(text: str, pattern: str) -> bool:
    """If pattern ends with " *", trailing args are optional (match with or without)."""
    if fnmatch.fnmatch(text, pattern):
        return True
    if pattern.endswith(" *") and fnmatch.fnmatch(text, pattern[:-2]):
        return True
    return False
