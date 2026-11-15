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

G5b adds one asynchronous leg for a PROCESS child: its ceiling lives in the
process that spawned it, so the gate asks the owner again AFTER the snapshot
await and immediately before the synchronous revalidation. That ask is the last
thing that can stop the call: nothing awaits between it and ``invoke``, so a
revocation ordered before the answer is not bypassed by the call, while a
revocation ordered after it is the next call's problem — the per-decision
contract the owner's own stability loop guarantees.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from chartreux.core.agent_loop._dangerous_command_grants import (
    DangerousCommandGrantStore,
    RemoteExecGrantRequirement,
)
from chartreux.core.dangerous_commands import RepositoryScope
from chartreux.core.tools.base import BaseTool, ToolPermission
from chartreux.core.tools.manager import ToolManager
from chartreux.observability.logging import logger


@dataclass(frozen=True, slots=True)
class ExecutionAdmission:
    """Invocation-bound authority carried from the pre-check to the gate.

    ``once_requirements`` holds the (operation, repository scope) pairs an
    "Allow once" approval authorized for exactly one invocation, and
    ``remote_exec_requirements`` holds the destination-bound pairs for a
    remote-exec "Allow once". The permits themselves are consumed before the
    call resumes, so no concurrent or later identical call can inherit them;
    the gate re-arms them from these requirements for its own revalidation
    only.
    """

    policy_revision: int
    once_requirements: tuple[tuple[str, RepositoryScope], ...] | None = None
    remote_exec_requirements: tuple[RemoteExecGrantRequirement, ...] | None = None


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
    if admission is not None and (
        grant_store is None
        or admission.policy_revision != grant_store.current_policy_revision
    ):
        return f"Tool '{tool_name}' denied: policy revision is stale"
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
    remote_exec_requirements = (
        admission.remote_exec_requirements if admission is not None else None
    )
    store = grant_store if (once_requirements or remote_exec_requirements) else None
    permit = None
    remote_exec_permit = None
    if store is not None and once_requirements:
        # Re-arm the invocation-bound permit for this revalidation only.
        permit = store.install_once(frozenset(once_requirements))
        tool_manager._invalidate_dangerous_command_authority()
    if store is not None and remote_exec_requirements:
        # Same re-arming rule for a destination-bound remote-exec permit.
        remote_exec_permit = store.install_remote_exec_once(
            frozenset(remote_exec_requirements)
        )
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
        if remote_exec_permit is not None and store is not None:
            store.destroy_remote_exec_once(remote_exec_permit)
            tool_manager._invalidate_dangerous_command_authority()
    if resolved is not None and resolved.permission == ToolPermission.NEVER:
        return resolved.reason or f"Tool '{tool_name}' is disabled by policy"
    return None


def canonical_live_authority_arguments(args: Any) -> Mapping[str, Any]:
    """The canonical arguments one live-authority question carries.

    Both ends must agree on what "these arguments" means, so the wire form is
    the validated model's JSON projection: exactly the values the invocation
    will run with, in a shape the owner can re-validate against its own
    implementation. A model that cannot project fails closed at the caller.
    """
    dump = getattr(args, "model_dump", None)
    if dump is None:
        return dict(args) if isinstance(args, Mapping) else {}
    return dump(mode="json")


async def live_authority_admission_denial(
    tool: BaseTool, args: Any, tool_manager: ToolManager
) -> str | None:
    """Ask the live owner again at the execution handoff; ``None`` when allowed.

    This is the ASYNC leg of the gate (G5b) and it is deliberately placed after
    the file-snapshot await: that await is the world-change window in which an
    owner can tighten or revoke, and a decision taken before it would authorize
    the call on policy that no longer exists. Every failure of the question —
    no live client, a dead pipe, a timeout, an unreadable or foreign answer —
    is a denial: only the owner's own allow lets the call through.
    """
    ask = getattr(tool_manager, "live_authority_denial", None)
    if ask is None:
        return None
    try:
        arguments = canonical_live_authority_arguments(args)
    except Exception:
        logger.warning(
            "Execution admission could not canonicalize the arguments tool=%s",
            tool.get_name(),
            exc_info=True,
        )
        return f"Tool '{tool.get_name()}' denied: its arguments cannot be presented to live authority"
    try:
        return await ask(tool.get_name(), type(tool), arguments)
    except asyncio.CancelledError:
        raise
    except BaseException:
        # The manager already maps a failed ask to a refusal; this is the
        # belt-and-braces path for an ask that itself raised.
        logger.warning(
            "Execution admission live-authority ask failed tool=%s",
            tool.get_name(),
            exc_info=True,
        )
        return f"Tool '{tool.get_name()}' denied: live authority is unavailable"
