"""Execution-admission gate: the final permission check before a tool runs.

The admission pre-check (``AgentLoop._should_execute_tool``) decides whether a
call may execute, but the loop then awaits the tool's file snapshot before
invoking it. That await is a world-change window: a dangerous-command grant
revoked while the snapshot runs — or a denylist entry added, a tool disabled,
a manager retired — used to leave the already-decided call executing anyway,
because ``BaseTool.invoke`` validates arguments and manager lifetime but never
re-checks permission.

The gate closes the window. Immediately after the snapshot await and before
``invoke``, the full permission resolution runs again — config-level
disablement plus the tool's own ``resolve_permission`` — with no prompt and
no grant mutation, so a grant already given is never re-asked and a revoked
one fails closed. Because the gate sits before ``invoke``, it covers every
execution path of the tool, including the shell tool's local execution and
its client-terminal delegation.

A once permit stays invocation-bound. The approval flow consumes the permit
before the call resumes — a concurrent or later identical call must not
inherit it — and the decision's ``ExecutionAdmission`` carries its
requirements privately. The gate re-arms a permit in the grant store for the
duration of one synchronous revalidation and consumes it again before
execution, so the re-armed authority is observable by no other coroutine: the
gate never awaits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chartreux.core.agent_loop._dangerous_command_grants import (
    DangerousCommandGrantStore,
)
from chartreux.core.dangerous_commands import RepositoryScope
from chartreux.core.tools.base import BaseTool, ToolPermission
from chartreux.core.tools.manager import ToolManager
from chartreux.observability.logging import logger


@dataclass(frozen=True, slots=True)
class ExecutionAdmission:
    """Invocation-bound authority carried from the pre-check to the gate.

    ``once_requirements`` holds the (operation, repository scope) pairs an
    "Allow once" approval authorized for exactly one invocation. The permit
    itself is consumed before the call resumes, so no concurrent or later
    identical call can inherit it; the gate re-arms it from these
    requirements for its own revalidation only.
    """

    once_requirements: tuple[tuple[str, RepositoryScope], ...] | None = None


def revalidate_execution_admission(
    tool: BaseTool,
    args: Any,
    tool_manager: ToolManager,
    grant_store: DangerousCommandGrantStore | None,
    admission: ExecutionAdmission | None,
) -> str | None:
    """Re-run the full permission resolution at the execution handoff.

    Returns the denial feedback when the call may not execute, and None when
    it may. Every pre-check dimension is re-checked — disablement, denylist,
    sensitivity, paths, parent ceilings, repository identity, retired
    manager, and the dangerous-command authority itself. The check is
    synchronous and never prompts: a grant already given is not re-asked, and
    a revoked one fails closed. A revalidation that itself fails fails closed
    as a denial, so a denied admission is always accounted as a rejected
    call, never an unhandled error.
    """
    tool_name = tool.get_name()
    # The snapshot await is a world-change window: drop cached authority so
    # the revalidation reads the live world — an in-place config change, a
    # store revocation, a retirement — exactly as the grant flows invalidate
    # before their final recheck.
    tool_manager._invalidate_dangerous_command_authority()
    try:
        config = tool_manager.get_tool_config(tool_name)
    except Exception:
        # A retired manager or an authority change mid-revalidation: the
        # resolution below fails closed with the honest reason.
        logger.warning(
            "Execution admission could not read the tool config tool=%s",
            tool_name,
            exc_info=True,
        )
        config = None
    if config is not None and config.permission == ToolPermission.NEVER:
        return f"Tool '{tool_name}' is disabled by policy"
    once_requirements = admission.once_requirements if admission is not None else None
    store = grant_store if once_requirements else None
    permit = None
    if store is not None and once_requirements:
        # Re-arm the invocation-bound permit for this revalidation only.
        permit = store.install_once(frozenset(once_requirements))
        tool_manager._invalidate_dangerous_command_authority()
    try:
        resolved = tool.resolve_permission(args)
    except Exception:
        logger.warning(
            "Execution admission revalidation failed tool=%s", tool_name, exc_info=True
        )
        return f"Tool '{tool_name}' is disabled by policy"
    finally:
        if permit is not None and store is not None:
            # Consume the re-armed permit whether the revalidation passed or
            # failed: it authorizes this invocation and nothing else.
            store.destroy_once(permit)
            tool_manager._invalidate_dangerous_command_authority()
    if resolved is not None and resolved.permission == ToolPermission.NEVER:
        return resolved.reason or f"Tool '{tool_name}' is disabled by policy"
    return None
