"""Tool decision collaborators for AgentLoop (ALL cluster: decisions/grants).

Owns the denial formatting, the dangerous-command approval workflow
(prompt, resume, persistence), the out-of-root approval workflow (resolve,
resume, persistence), and the shared root-prompt implementation with its
asker lifecycle, all extracted from _loop.py. The coordinator borrows the
grant stores and the request broker, and reads live loop values
(user-input capability, subagent flag, persistence ports, tool manager,
session identity, config orchestrator, cwd) through getters so
post-binding and manager replacement stay visible. The patched
module-level ``read_saved_dangerous_command_grants`` and
``read_saved_roots`` surfaces are resolved at invocation time through
facade callbacks, so test patches of the _loop globals keep intercepting.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel

from chartreux.core.agent_loop._dangerous_command_grants import (
    DANGEROUS_COMMAND_DECLINED_FEEDBACK,
    DangerousCommandCallIdentity,
    DangerousCommandGrantStore,
    dangerous_command_grant_save_note,
    dangerous_command_request,
    derive_call_requirements,
    parse_dangerous_command_result,
)
from chartreux.core.agent_loop._execution_admission import ExecutionAdmission
from chartreux.core.agent_loop._loop_types import ToolDecision, ToolExecutionResponse
from chartreux.core.agent_loop._request_broker import InteractionRequestBroker
from chartreux.core.agent_loop._root_grants import (
    ROOT_GRANT_DECLINED_FEEDBACK,
    ROOT_GRANT_DENY,
    ROOT_GRANT_PROJECT,
    ROOT_GRANT_TOOLS,
    SessionRootGrants,
    parse_root_grant_result,
    propose_root_grant,
    root_grant_request,
    root_grant_save_note,
    root_grant_target,
)
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._dangerous_command_persistence import SavedGrantsRead
from chartreux.core.config._root_persistence import SavedRootsRead
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.dangerous_command_grants import DangerousCommandGrantPersistencePort
from chartreux.core.dangerous_commands import (
    OPERATION_REGISTRY,
    DangerousCommandProposal,
    GrantChoice,
    RepositoryScope,
)
from chartreux.core.events import UserInputOrigin
from chartreux.core.root_grants import RootGrantPort
from chartreux.core.tools.base import BaseTool, ToolPermission
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.permissions import PermissionContext
from chartreux.observability.logging import logger


def _out_of_root_denial(ctx: PermissionContext, tool_name: str) -> ToolDecision:
    """Denial decision that names the persistent config knob for the path."""
    reason = ctx.reason or f"Tool '{tool_name}' is disabled by policy"
    return ToolDecision(
        verdict=ToolExecutionResponse.SKIP,
        approval_type=ToolPermission.NEVER,
        feedback=(
            f"{reason} Add the path to authorized_roots_by_project for this "
            "project in your user config.toml to authorize it."
        ),
    )


def _declined_root_decision() -> ToolDecision:
    return ToolDecision(
        verdict=ToolExecutionResponse.SKIP,
        approval_type=ToolPermission.NEVER,
        feedback=ROOT_GRANT_DECLINED_FEEDBACK,
    )


def _dangerous_command_denial(
    ctx: PermissionContext, tool_name: str, proposal: DangerousCommandProposal
) -> ToolDecision:
    """Denial decision naming the persistent config knob for the operation."""
    reason = ctx.reason or f"Tool '{tool_name}' is disabled by policy"
    persistable = tuple(
        operation_id
        for operation_id in proposal.operations
        if (operation := OPERATION_REGISTRY.get(operation_id)) is not None
        and operation.persistable
    )
    blocked = tuple(
        operation_id
        for operation_id in proposal.operations
        if (operation := OPERATION_REGISTRY.get(operation_id)) is None
        or not operation.persistable
    )
    if persistable and not blocked:
        remediation = (
            f"Add {', '.join(persistable)} to "
            "dangerous_command_grants_by_project for this project in your "
            "user config.toml to authorize it."
        )
    elif persistable:
        # A mixed call: the saved grant covers only the persistable subset,
        # so the message must not claim the config edit authorizes the call.
        remediation = (
            f"Add {', '.join(persistable)} to "
            "dangerous_command_grants_by_project for this project in your "
            "user config.toml to pre-authorize only part of this call; the "
            f"remaining operations ({', '.join(blocked)}) cannot be "
            "pre-authorized and require approval in an interactive session."
        )
    else:
        remediation = (
            "This operation cannot be pre-authorized; it requires approval in "
            "an interactive session."
        )
    return ToolDecision(
        verdict=ToolExecutionResponse.SKIP,
        approval_type=ToolPermission.NEVER,
        feedback=f"{reason} {remediation}",
    )


def _declined_dangerous_command_decision() -> ToolDecision:
    return ToolDecision(
        verdict=ToolExecutionResponse.SKIP,
        approval_type=ToolPermission.NEVER,
        feedback=DANGEROUS_COMMAND_DECLINED_FEEDBACK,
    )


def _retrieve_future_exception(future: asyncio.Future[str]) -> None:
    if not future.cancelled():
        future.exception()


class ToolDecisionCoordinator:
    """Own the approval workflows and the root-prompt asker lifecycle.

    The dangerous-command approval/resume/persistence block moves as one
    unit: the resume method's finally still destroys an unconsumed once
    permit and invalidates the cached authority it fed on every exit path,
    so no reusable authority outlives an approval. A session grant is
    intentional and persistent: it is never rolled back here. The root flow
    keeps its own semantics: concurrent waiters coalesce on one shared
    prompt decision, and batch teardown abandons lingering askers without
    joining them.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        grants_store: DangerousCommandGrantStore,
        request_broker: InteractionRequestBroker,
        user_input_capability_getter: Callable[[], bool],
        is_subagent_getter: Callable[[], bool],
        grant_port_getter: Callable[[], DangerousCommandGrantPersistencePort | None],
        tool_manager_getter: Callable[[], ToolManager],
        session_id_getter: Callable[[], str],
        config_orchestrator_getter: Callable[
            [], ConfigOrchestrator[ChartreuxConfigSchema]
        ],
        read_saved_grants: Callable[
            [ConfigOrchestrator[ChartreuxConfigSchema], Path],
            Awaitable[SavedGrantsRead],
        ],
        root_grants_store: SessionRootGrants,
        root_grant_port_getter: Callable[[], RootGrantPort | None],
        cwd_getter: Callable[[], Path],
        read_saved_roots: Callable[
            [ConfigOrchestrator[ChartreuxConfigSchema], Path], Awaitable[SavedRootsRead]
        ],
        apply_approved_root_grant: Callable[[Path], Awaitable[None]],
    ) -> None:
        # Borrowed shared ownership: the grant stores and the request broker
        # are the loop's own objects, never cloned or replaced.
        self._grants_store = grants_store
        self._root_grants_store = root_grants_store
        self._request_broker = request_broker
        # Live getters: the persistence ports are bound after construction
        # (bind_dangerous_command_grant_port / bind_root_grant_port), and the
        # tool manager is replaced by configuration transactions; all are
        # resolved at decision time, never captured.
        self._user_input_capability_getter = user_input_capability_getter
        self._is_subagent_getter = is_subagent_getter
        self._grant_port_getter = grant_port_getter
        self._root_grant_port_getter = root_grant_port_getter
        self._tool_manager_getter = tool_manager_getter
        self._session_id_getter = session_id_getter
        self._config_orchestrator_getter = config_orchestrator_getter
        self._cwd_getter = cwd_getter
        # Facade callbacks: resolve the patched _loop module globals at
        # invocation time; the approved-grant application stays on the loop
        # so runtime port routing (registered session, cwd-keyed save) and
        # any instance patch of it keep intercepting.
        self._read_saved_grants = read_saved_grants
        self._read_saved_roots = read_saved_roots
        self._apply_approved_root_grant = apply_approved_root_grant
        # Root-prompt asker lifecycle: the live asker set is owned here so
        # batch teardown can abandon it synchronously.
        self._root_grant_askers: set[asyncio.Task[None]] = set()

    async def resolve_dangerous_command_grant(
        self,
        tool: BaseTool,
        args: BaseModel,
        ctx: PermissionContext,
        tool_name: str,
        tool_call_id: str | None,
    ) -> ToolDecision | None:
        """Resolve a grantable dangerous-command denial, prompting when allowed.

        Returns None when the denial must keep the existing behavior: no typed
        proposal, or a call that can no longer be re-derived, or a bare loop
        without user input keeps the plain resolver denial. A runtime-managed
        session that cannot prompt denies promptly with the remediation
        instead of waiting. A decided call identity is never re-prompted: it
        returns the honest refusal.
        """
        proposal = ctx.dangerous_command_proposal
        command = getattr(args, "command", None)
        # Derive the approval's coverage in the same synchronous window as the
        # denial, so a world change before the answer fails the recheck instead
        # of silently re-scoping the approval to a different repository.
        requirements = (
            derive_call_requirements(command, proposal.project, proposal.operations)
            if proposal is not None and isinstance(command, str)
            else None
        )
        if proposal is None or requirements is None:
            # No typed proposal, or the call can no longer be re-derived:
            # keep the plain denial.
            return None
        if self._is_subagent_getter() or not self._user_input_capability_getter():
            # Blanket child denial: a child session never prompts for
            # ask-class authority, whatever its user-input capability — the
            # denial with remediation flows back to the parent as a blocker,
            # and dangerous_command_grants_by_project stays the only child
            # pre-authorization path. A runtime-managed session that cannot
            # prompt names the persistent config knob (publication only); a
            # bare loop keeps the existing denial text unchanged.
            return (
                _dangerous_command_denial(ctx, tool_name, proposal)
                if self._grant_port_getter() is not None
                else None
            )
        store = self._grants_store
        identity = DangerousCommandCallIdentity(
            scope=proposal.scope, command=proposal.command
        )
        if store.is_refused(identity):
            return _declined_dangerous_command_decision()
        choice = await self.prompt_dangerous_command_grant(proposal, tool_call_id)
        if choice is None:
            return _dangerous_command_denial(ctx, tool_name, proposal)
        if choice == "deny":
            store.record_refusal(identity)
            return _declined_dangerous_command_decision()
        return await self.resume_after_dangerous_command_grant(
            tool, args, tool_name, proposal, choice, requirements
        )

    async def prompt_dangerous_command_grant(
        self, proposal: DangerousCommandProposal, tool_call_id: str | None
    ) -> GrantChoice | None:
        """Ask the user to approve *proposal*; None when the prompt fails.

        A failed prompt is never a user decision: it denies without recording
        a refusal, so a retry can prompt again.
        """
        try:
            answer = await self._request_broker.request_user_input(
                dangerous_command_request(proposal),
                tool_call_id or "",
                # Runtime approval, never a model question: permanently
                # ineligible for unattended expiry (plan U/OD1).
                origin=UserInputOrigin.RUNTIME_APPROVAL,
            )
            return parse_dangerous_command_result(answer, proposal.operations)
        except Exception:
            logger.warning(
                "Dangerous-command prompt failed; keeping the denial", exc_info=True
            )
            return None

    async def resume_after_dangerous_command_grant(
        self,
        tool: BaseTool,
        args: BaseModel,
        tool_name: str,
        proposal: DangerousCommandProposal,
        choice: GrantChoice,
        requirements: frozenset[tuple[str, RepositoryScope]],
    ) -> ToolDecision:
        """Apply an approved grant, fully re-resolve, and resume the call.

        The approval installs runtime authority first — a once permit bound to
        exactly this invocation, or a session grant keyed by operation and
        repository scope — then the original call is FULLY re-resolved
        (disablement, denylist, sensitivity, paths, parent ceilings, repository
        identity, retired manager). A failed recheck executes nothing and
        destroys any unused once authority; a passed recheck consumes the
        permit before the call resumes, leaving no reusable authority. A
        project grant then awaits persistence, and that await is itself a
        world-change window: the full resolution runs again after it, so the
        last recheck is the final admission gate exactly as the roots flow
        rechecks last.

        An installed once permit never outlives this function unless the call
        resumes: the recheck runs while the permit is live, so an exception
        from it — or a cancellation delivered at any later await — would
        otherwise leave reusable authority behind. The finally block destroys
        an unconsumed permit and invalidates the cached authority it fed on
        every exit path, so a failed or interrupted recheck leaks no
        authority. A session grant is intentional and persistent: it is never
        rolled back here.
        """
        store = self._grants_store
        permit = None
        try:
            if choice == "allow_once":
                if requirements:
                    permit = store.install_once(requirements)
                    self._tool_manager_getter()._invalidate_dangerous_command_authority()
            elif store.grant_session(requirements):
                self._tool_manager_getter()._invalidate_dangerous_command_authority()
            # Full recheck of the original call under the approved authority. A
            # failed recheck executes nothing, destroys any unused once
            # authority, and never persists.
            resolved = tool.resolve_permission(args)
            if resolved is not None and resolved.permission == ToolPermission.NEVER:
                return ToolDecision(
                    verdict=ToolExecutionResponse.SKIP,
                    approval_type=ToolPermission.NEVER,
                    feedback=(
                        resolved.reason or f"Tool '{tool_name}' is disabled by policy"
                    ),
                )
            if permit is not None:
                # The permit authorized exactly this invocation; it is consumed
                # before the call resumes, so no authority outlives the approval.
                store.destroy_once(permit)
                self._tool_manager_getter()._invalidate_dangerous_command_authority()
                permit = None
            result_note = None
            if choice == "allow_project":
                result_note = await self.persist_project_dangerous_command_grant(
                    proposal
                )
                # The persist await sits between the recheck above and
                # execution, so that recheck is not yet the final admission
                # gate: the world may have changed while it ran. Invalidate
                # cached authority — an in-place config change does not move
                # the manager's authority token — and re-run the full
                # resolution, exactly as the roots flow rechecks last. A
                # mid-await change (denylist, disablement, a retired manager)
                # skips the call cleanly instead of executing it or failing at
                # invoke time.
                self._tool_manager_getter()._invalidate_dangerous_command_authority()
                resolved = tool.resolve_permission(args)
                if resolved is not None and resolved.permission == ToolPermission.NEVER:
                    return ToolDecision(
                        verdict=ToolExecutionResponse.SKIP,
                        approval_type=ToolPermission.NEVER,
                        feedback=(
                            resolved.reason
                            or f"Tool '{tool_name}' is disabled by policy"
                        ),
                    )
            return ToolDecision(
                verdict=ToolExecutionResponse.EXECUTE,
                approval_type=ToolPermission.ALWAYS,
                result_note=result_note,
                # The once permit was consumed above, before the call resumed;
                # the admission carries its requirements so the execution
                # gate can revalidate exactly this invocation without
                # authorizing any other call.
                admission=(
                    ExecutionAdmission(once_requirements=tuple(requirements))
                    if choice == "allow_once"
                    else None
                ),
            )
        finally:
            if permit is not None:
                # The permit was installed but never consumed: an exception or
                # a cancellation left this function before the call resumed.
                # Destroy it so no reusable authority outlives the approval,
                # and invalidate the cached authority it fed.
                store.destroy_once(permit)
                self._tool_manager_getter()._invalidate_dangerous_command_authority()

    async def persist_project_dangerous_command_grant(
        self, proposal: DangerousCommandProposal
    ) -> str:
        """Attempt to persist an approved grant; report grant and save apart.

        The user revision is fetched from the saved-grants read surface at
        decision time and never cached across turns. A failed or uncertain
        save never blocks the already-applied session grant; cancellation
        propagates so the tool cannot resume without the session grant.
        """
        repository = proposal.scope.work_tree or proposal.scope.git_dir
        operations = ", ".join(proposal.operations)
        port = self._grant_port_getter()
        if port is None:
            return (
                f"Dangerous-command grant for {operations} in {repository} "
                "applies to this session; this session cannot save "
                "dangerous-command grants to your user config."
            )
        try:
            saved = await self._read_saved_grants(
                self._config_orchestrator_getter(), proposal.project
            )
        except Exception:
            logger.warning(
                "Saved dangerous-command grants read failed for %s",
                proposal.project,
                exc_info=True,
            )
            saved = None
        if saved is None or saved.unavailable is not None or not saved.user_revision:
            return (
                f"Dangerous-command grant for {operations} in {repository} "
                "applies to this session only; your saved grants could not be "
                "read, so it was not saved to your config."
            )
        outcomes: list[tuple[str, ConfigSaveResult | None]] = []
        for operation_id in proposal.operations:
            operation = OPERATION_REGISTRY.get(operation_id)
            if operation is None or not operation.persistable:
                continue  # Only publication grants are persistable.
            try:
                result = await port.save_dangerous_command_grant(
                    self._session_id_getter(),
                    proposal.project,
                    operation_id,
                    saved.user_revision,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Project dangerous-command grant save failed for %s",
                    operation_id,
                    exc_info=True,
                )
                result = None
            outcomes.append((operation_id, result))
        if not outcomes:
            return (
                f"Dangerous-command grant for {operations} in {repository} "
                "applies to this session; no persistable operation was saved "
                "to your user config."
            )
        return dangerous_command_grant_save_note(
            proposal.project, proposal.scope, tuple(outcomes)
        )

    async def resolve_out_of_root_grant(
        self,
        tool: BaseTool,
        args: BaseModel,
        ctx: PermissionContext,
        tool_name: str,
        tool_call_id: str | None,
    ) -> ToolDecision | None:
        """Resolve an out-of-root file denial, prompting when the runtime allows.

        Returns None when the denial must keep the existing behavior: the tool
        is not a promptable file tool, or no promptable target can be proposed.
        A child session never prompts, whatever its user-input capability: it
        denies with remediation and the blocker flows back to the parent.
        """
        if tool_name not in ROOT_GRANT_TOOLS:
            return None
        if self._is_subagent_getter() or not self._user_input_capability_getter():
            # Blanket child denial: a child session never prompts for
            # out-of-root scope, whatever its user-input capability — the
            # denial with remediation flows back to the parent as a blocker,
            # and already-issued child-local grants stay effective because
            # this gate is only the interactive ask for new scope.
            # authorized_roots_by_project and parent-directed scope changes
            # stay the child's only paths to new roots. A runtime-managed
            # session that cannot prompt names the persistent config knob; a
            # bare loop keeps the existing denial text unchanged.
            return (
                _out_of_root_denial(ctx, tool_name)
                if self._root_grant_port_getter() is not None
                else None
            )
        target = root_grant_target(tool_name, args, self._cwd_getter())
        root = propose_root_grant(target) if target is not None else None
        if root is None:
            return None
        store = self._root_grants_store
        try:
            decision = (
                ROOT_GRANT_DENY
                if root in store.denied_roots
                else await self.prompt_root(root, tool_call_id)
            )
        except Exception:
            logger.warning(
                "Session root grant prompt failed; keeping the denial", exc_info=True
            )
            return _out_of_root_denial(ctx, tool_name)
        if decision == ROOT_GRANT_DENY:
            store.deny(root)
            return _declined_root_decision()
        return await self.resume_after_root_grant(tool, args, tool_name, root, decision)

    async def resume_after_root_grant(
        self, tool: BaseTool, args: BaseModel, tool_name: str, root: Path, decision: str
    ) -> ToolDecision:
        """Apply an approved grant, then re-resolve and resume the original call."""
        # The session grant is applied first: the tool call resumes only with
        # the grant in force, whatever the optional project save reports.
        await self._apply_approved_root_grant(root)
        result_note = None
        if decision == ROOT_GRANT_PROJECT:
            result_note = await self.persist_project_root_grant(root)
        resolved = tool.resolve_permission(args)
        if resolved is not None and resolved.permission == ToolPermission.NEVER:
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                approval_type=ToolPermission.NEVER,
                feedback=(
                    resolved.reason or f"Tool '{tool_name}' is disabled by policy"
                ),
            )
        return ToolDecision(
            verdict=ToolExecutionResponse.EXECUTE,
            approval_type=ToolPermission.ALWAYS,
            result_note=result_note,
        )

    async def persist_project_root_grant(self, root: Path) -> str:
        """Attempt to persist an approved grant; report grant and save apart.

        The user revision is fetched from the saved-roots read surface at
        decision time and never cached across turns. A failed or uncertain save
        never blocks the already-applied session grant; cancellation propagates
        so the tool cannot resume without the session grant.
        """
        port = self._root_grant_port_getter()
        if port is None:
            return (
                f"Root grant for {root} applies to this session; this session "
                "cannot save roots to your user config."
            )
        try:
            saved = await self._read_saved_roots(
                self._config_orchestrator_getter(), self._cwd_getter()
            )
        except Exception:
            logger.warning(
                "Saved roots read failed for %s", self._cwd_getter(), exc_info=True
            )
            saved = None
        if saved is None or saved.unavailable is not None or not saved.user_revision:
            return (
                f"Root grant for {root} applies to this session only; your saved "
                "roots could not be read, so it was not saved to your config."
            )
        try:
            result = await port.save_root(
                self._session_id_getter(), root, saved.user_revision
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Project root grant save failed for %s", root, exc_info=True)
            return (
                f"Root grant for {root} applies to this session only; saving it "
                "to your user config failed."
            )
        return root_grant_save_note(root, self._cwd_getter(), result)

    async def prompt_root(self, root: Path, tool_call_id: str | None) -> str:
        """Ask about *root*, coalescing concurrent waiters on one shared decision."""
        store = self._root_grants_store
        pending = store.pending.get(root)
        if pending is None:
            pending = asyncio.get_running_loop().create_future()
            store.pending[root] = pending
            asker = asyncio.create_task(
                self._ask_root_grant(root, pending, tool_call_id)
            )
            self._root_grant_askers.add(asker)
            asker.add_done_callback(self._root_grant_askers.discard)
        # One waiter's cancellation must not cancel the shared decision.
        return await asyncio.shield(pending)

    async def _ask_root_grant(
        self, root: Path, pending: asyncio.Future[str], tool_call_id: str | None
    ) -> None:
        """Own the single user-input request behind a shared root decision."""
        try:
            answer = await self._request_broker.request_user_input(
                root_grant_request(root),
                tool_call_id or "",
                # Runtime approval, never a model question: permanently
                # ineligible for unattended expiry (plan U/OD1).
                origin=UserInputOrigin.RUNTIME_APPROVAL,
            )
            decision = parse_root_grant_result(answer)
        except asyncio.CancelledError:
            if not pending.done():
                pending.cancel()
            raise
        except Exception as exc:
            if not pending.done():
                pending.set_exception(exc)
                pending.add_done_callback(_retrieve_future_exception)
            logger.warning("Session root grant prompt failed for %s: %s", root, exc)
        else:
            if not pending.done():
                pending.set_result(decision)
        finally:
            self._root_grants_store.pending.pop(root, None)

    def abandon_root_requests(self) -> None:
        """Cancel lingering root-prompt askers; a tool batch join is NOT added.

        Every waiter is settled when a tool batch ends, so no prompt can
        still have one; a lingering asker only holds an abandoned request.
        """
        for asker in tuple(self._root_grant_askers):
            asker.cancel()
