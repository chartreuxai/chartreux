"""Harness-owned tool authority classifications.

Classification is descriptive input to identity posture enforcement, never a grant.
Unknown implementations (including custom replacements and MCP proxies) fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    "skill": ToolClass.NON_MUTATING,
    "todo": ToolClass.CONTROL_PLANE,
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


# Shipped-builtin identity (P4-2). A class is a shipped builtin exactly when
# a shipped builtin module DEFINES it: the registry discovers precisely those
# classes from the builtin directory (always first in the search paths, with
# re-exports filtered there), so the defining module is the ground truth of
# builtin identity and no side table is needed. The reviewed, name-keyed
# table above then classifies it; a builtin-module class publishing an
# unknown name still fails closed as MUTATING.
#
# This replaces the provenance-stamp side table removed in P4-2, which
# defended impersonation of a builtin by a foreign class — a custom class
# declaring ``_registry_builtin_name`` in its own body, or a metaclass
# aliasing a builtin's hash/equality to read its stamp (S2 rounds 2-3). The
# charter excludes that threat: tool modules execute arbitrary code at
# import, inside the trust boundary the extension model grants, so an
# in-process forger can already reach any private writer — the removed
# table's own docstring conceded "defense-in-depth, not a security boundary
# the side table can enforce". Foreign classes (custom replacements,
# subclasses, MCP proxies, project-discovered tools) are defined outside the
# shipped builtin modules, so they still fail closed as MUTATING.
_BUILTIN_MODULE_PREFIX = "chartreux.core.tools.builtins."


def shipped_builtin_name(tool_class: type[BaseTool]) -> str | None:
    """The reviewed table name of a shipped builtin class, or None.

    A class defined by a shipped builtin module is that builtin, and its
    reviewed name is its tool name. Anything defined elsewhere — a custom
    replacement, a subclass, an MCP proxy, a project-discovered tool — is
    not a shipped builtin and fails closed. The check reads the class's
    defining module, never its namespace: an author-declared attribute
    confers nothing (see the module comment for the excluded threat).
    """
    if tool_class.__module__.startswith(_BUILTIN_MODULE_PREFIX):
        return tool_class.get_name()
    return None


# File-location provenance for local user-authored tools (P4-1, charter
# decision 4). A tool authored under the user's OWN tools directory
# (``~/.chartreux/tools/`` — the user's home config, not project files) is
# the user's own code, so the harness admits it to the non-mutating posture
# by WHERE it lives: a deliberate, separate trust contract keyed on the
# tools directory, distinct from the shipped-builtin identity check above
# (which guards BUILTIN identity, not user authorship) and distinct from the
# fail-closed defaults kept by project-discovered tools and MCP proxies.
#
# The table is keyed by ``id(class)`` and verified through a weak reference
# for the same alias-proofing reasons the removed provenance-stamp table used
# (S2 round 3: weak-key aliasing): a custom metaclass aliasing a trusted
# class's hash/equality must not read its trust, and a recycled id whose weak
# reference died (or points elsewhere) must not inherit it. Only the
# harness's gated discovery path writes here — user-tools-directory
# discovery in ``ToolManager._register_discovered_tool_variant`` — so an
# author-declared class attribute never reaches the table, and neither do
# project-discovered tools, config ``tool_paths`` entries, or MCP proxies.
_LOCATION_TRUSTED_USER_TOOLS: dict[int, weakref.ref[type[BaseTool]]] = {}


def _stamp_location_trusted_user_tool(tool_class: type[BaseTool]) -> None:
    """Record file-location trust for one exact class.

    This is the ONLY writer of the location-trust side table, and the name is
    private: the harness calls it only from its gated discovery path (a
    class discovered from a file under the user's own tools directory in
    ``ToolManager._register_discovered_tool_variant``), so an
    author-declared class attribute never reaches the table. A determined
    in-process caller could still reach the private name, but custom-tool
    module code already executes arbitrary code at import — inside the
    arbitrary-code-execution trust boundary the extension model grants, not
    outside it: the privacy of this name is defense-in-depth, not a
    security boundary the side table can enforce.
    """
    _LOCATION_TRUSTED_USER_TOOLS[id(tool_class)] = weakref.ref(tool_class)


def location_trusted_user_tool(tool_class: type[BaseTool]) -> bool:
    """Whether the exact class was discovered from the user's tools directory.

    Absent for anything the gated discovery path never stamped: custom
    replacements, project-discovered tools, config ``tool_paths`` entries,
    MCP proxies, and subclasses all fail closed. The lookup is identity-safe
    — keyed on ``id(class)`` and verified through a weak reference — so a
    metaclass aliasing a trusted class's hash/equality (or a recycled id)
    cannot read another class's trust (S2 round 3: weak-key aliasing).
    """
    reference = _LOCATION_TRUSTED_USER_TOOLS.get(id(tool_class))
    if reference is None:
        return False
    # A stale id whose weak reference died (or points elsewhere) carries no
    # trust: fail closed.
    return reference() is tool_class
