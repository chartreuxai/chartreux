"""Harness-owned tool authority classifications.

Classification is descriptive input to identity posture enforcement, never a grant.
Unknown implementations (including custom replacements and MCP proxies) fail closed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    from chartreux.core.tools.base import BaseTool


class ToolClass(StrEnum):
    MUTATING = "mutating"
    NON_MUTATING = "non-mutating"
    CONTROL_PLANE = "control-plane"


_CLASSIFICATIONS: dict[str, ToolClass] = {
    "bash": ToolClass.MUTATING,
    "bash_start": ToolClass.MUTATING,
    "bash_read": ToolClass.NON_MUTATING,
    "bash_stop": ToolClass.CONTROL_PLANE,
    "bash_list": ToolClass.NON_MUTATING,
    "read_file": ToolClass.NON_MUTATING,
    "read_image": ToolClass.NON_MUTATING,
    "grep": ToolClass.NON_MUTATING,
    "web_search": ToolClass.NON_MUTATING,
    "web_fetch": ToolClass.NON_MUTATING,
    "edit": ToolClass.MUTATING,
    "write_file": ToolClass.MUTATING,
    "skill": ToolClass.MUTATING,
    "todo": ToolClass.MUTATING,
    "ask_user_question": ToolClass.CONTROL_PLANE,
    "task": ToolClass.CONTROL_PLANE,
    "check_agents": ToolClass.CONTROL_PLANE,
    "wait_for_agent": ToolClass.CONTROL_PLANE,
    "get_agent_result": ToolClass.CONTROL_PLANE,
    "cancel_agent": ToolClass.CONTROL_PLANE,
    "release_agent": ToolClass.CONTROL_PLANE,
    "resume_agent": ToolClass.CONTROL_PLANE,
    "reconcile_result": ToolClass.CONTROL_PLANE,
}
TOOL_CLASSIFICATIONS: Mapping[str, ToolClass] = MappingProxyType(_CLASSIFICATIONS)


def reclassify_tool(name: str, classification: ToolClass) -> None:
    """Update a reviewed builtin classification; unknown names are rejected."""
    if name not in _CLASSIFICATIONS:
        raise KeyError(f"Unknown builtin tool: {name}")
    _CLASSIFICATIONS[name] = classification


def classify_tool(name: str, *, builtin: bool = False) -> ToolClass:
    """Classify an implementation; names alone never classify custom tools."""
    if builtin:
        return TOOL_CLASSIFICATIONS.get(name, ToolClass.MUTATING)
    return ToolClass.MUTATING


def classification_revision() -> str:
    """Return a content-derived revision for authority-cache invalidation."""
    payload = "\n".join(
        f"{name}:{classification.value}"
        for name, classification in sorted(_CLASSIFICATIONS.items())
    )
    return sha256(payload.encode()).hexdigest()


# Harness-side registry provenance (S2 round 2: forgeable provenance stamp;
# S2 round 3: weak-key aliasing). The registry-conferred builtin stamp lives
# in THIS side table, never in the class namespace: a custom tool class can
# declare ``_registry_builtin_name = "read_file"`` in its own body, and a
# ``__dict__`` lookup cannot distinguish an author-declared attribute from a
# registry-conferred one, so the forged declaration would confer the reviewed
# builtin classification. Only the harness's gated stamping paths write here:
# builtin discovery in ``ToolManager._register_discovered_tool_variant``
# confers the stamp, and the reuse-path copy re-asserts an existing one.
#
# The table is keyed by ``id(class)`` and stores ``(weakref(class), name)``.
# A WeakKeyDictionary would dispatch the lookup through the class's
# ``__hash__``/``__eq__``, which a custom METACLASS can alias to a stamped
# builtin class's: the unstamped custom would then read the builtin's stamp
# and ride a read-only posture (S2 round 3: weak-key aliasing). The identity
# check ``reference() is tool_class`` makes the lookup alias-proof — a stale
# id whose weak reference died (or points elsewhere) returns None, which
# classifies MUTATING.
_REGISTRY_BUILTIN_NAMES: dict[int, tuple[weakref.ref[type[BaseTool]], str]] = {}


def _stamp_registry_builtin(tool_class: type[BaseTool], name: str) -> None:
    """Record the registry-conferred builtin name for one exact class.

    This is the ONLY writer of the provenance side table, and the name is
    private: the harness calls it only from its gated stamping paths (builtin
    discovery in ``ToolManager._register_discovered_tool_variant`` and the
    reuse-path copy in ``copy_registry_builtin_stamps``), so an author-declared
    class attribute never reaches the table — declarative/data-level forgery
    is gated (S2 round 2: forgeable provenance stamp). A determined
    in-process caller could still reach the private name, but custom-tool
    module code already executes arbitrary code at import — an imperative call
    from such code sits INSIDE the arbitrary-code-execution trust boundary the
    extension model grants, not outside it: the privacy of this name is
    defense-in-depth, not a security boundary the side table can enforce.
    """
    _REGISTRY_BUILTIN_NAMES[id(tool_class)] = (weakref.ref(tool_class), name)


def registry_builtin_name(tool_class: type[BaseTool]) -> str | None:
    """The registry-conferred builtin name of one exact class, if any.

    Absent for anything the registry never stamped: custom replacements, MCP
    proxies, and subclasses all fail closed as MUTATING. The lookup is
    identity-safe — keyed on ``id(class)`` and verified through a weak
    reference — so a metaclass aliasing a stamped class's hash/equality (or a
    recycled id) cannot read another class's stamp (S2 round 3: weak-key
    aliasing).
    """
    entry = _REGISTRY_BUILTIN_NAMES.get(id(tool_class))
    if entry is None:
        return None
    reference, name = entry
    if reference() is not tool_class:
        # A stale id whose weak reference died (or points elsewhere) carries
        # no provenance: fail closed.
        return None
    return name


def copy_registry_builtin_stamps(tool_classes: Iterable[type[BaseTool]]) -> None:
    """Re-assert provenance stamps for reused discovery classes.

    The ``discovery_source`` reuse path (policy-only staging) copies the
    accepted variant registries without re-running discovery. The side table
    is keyed by the exact class, so the reused classes already carry their
    stamps; the copy re-asserts them alongside ``_tool_variants_by_name`` so
    the reuse path never depends on when the source manager's discovery ran.
    """
    for tool_class in tool_classes:
        stamped = registry_builtin_name(tool_class)
        if stamped is not None:
            _stamp_registry_builtin(tool_class, stamped)
