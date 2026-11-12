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
    RemoteExecGrantRequirement,
    dangerous_command_grant_save_note,
    dangerous_command_request,
    derive_call_requirements,
    parse_dangerous_command_result,
    remote_exec_grant_save_note,
)
from chartreux.core.agent_loop._execution_admission import ExecutionAdmission
from chartreux.core.agent_loop._loop_types import ToolDecision, ToolExecutionResponse
from chartreux.core.agent_loop._request_broker import (
    CHILD_QUESTION_REFUSAL,
    MIN_UNATTENDED_TIMEOUT_SECONDS,
    InteractionRequestBroker,
    UserInputAdmission,
)
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
from chartreux.core.deferred_questions import DeferredQuestionStore
from chartreux.core.events import UserInputOrigin, UserInputRequestEvent
from chartreux.core.root_grants import RootGrantPort
from chartreux.core.tools.base import BaseTool, ToolPermission
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.permissions import PermissionContext
from chartreux.observability.logging import logger
from chartreux.questions import (
    QuestionKind,
    UnansweredQuestion,
    UnansweredQuestions,
    UnattendedExpiryPolicy,
    UserQuestionRequest,
)


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
    if proposal.destination is not None:
        # A remote-exec grant is a destination grant: the persistent knob
        # names the exact destination, never a bare operation ID.
        return ToolDecision(
            verdict=ToolExecutionResponse.SKIP,
            approval_type=ToolPermission.NEVER,
            feedback=(
                f"{reason} Add the destination {proposal.destination} to "
                "remote_exec_grants_by_project for this project in your user "
                "config.toml to authorize it."
            ),
        )
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


def _prompt_outcome_decision(
    store: DangerousCommandGrantStore,
    identity: DangerousCommandCallIdentity,
    proposal: DangerousCommandProposal,
    choice: GrantChoice | None,
    ctx: PermissionContext,
    tool_name: str,
) -> ToolDecision:
    """Turn a failed or declined prompt into the matching denial.

    ``None`` means the prompt itself failed, which denies without recording a
    refusal so a retry can prompt again; ``deny`` is a decision and is
    remembered.
    """
    if choice is None:
        return _dangerous_command_denial(ctx, tool_name, proposal)
    store.record_refusal(identity)
    return _declined_dangerous_command_decision()


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

    def _live_scratchpad_roots(self) -> frozenset[Path]:
        """The live scratch roots both derivation call sites thread.

        An unavailable authority chain — a retired manager, a broken parent
        link — must not turn a grant decision into an exception, and the
        re-derivation's requirements never read roots, so an unavailable
        chain degrades to no roots.
        """
        try:
            return self._tool_manager_getter().scratchpad_roots
        except Exception:
            return frozenset()

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
        if proposal is not None and proposal.destination is not None:
            # A remote-exec denial is a destination grant, a separate
            # authority class from the scope-keyed operations below.
            return await self.resolve_remote_exec_grant(
                tool, args, ctx, tool_name, tool_call_id
            )
        command = getattr(args, "command", None)
        # Derive the approval's coverage in the same synchronous window as the
        # denial, so a world change before the answer fails the recheck instead
        # of silently re-scoping the approval to a different repository.
        requirements = (
            derive_call_requirements(
                command,
                proposal.project,
                proposal.operations,
                scratchpad_roots=self._live_scratchpad_roots(),
            )
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
            # and the persisted grant fields
            # (dangerous_command_grants_by_project, and
            # remote_exec_grants_by_project for a destination) stay the only
            # child pre-authorization paths. A runtime-managed session that cannot
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
        if choice is None or choice == "deny":
            return _prompt_outcome_decision(
                store, identity, proposal, choice, ctx, tool_name
            )
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
                dangerous_command_request(
                    proposal, scratchpad_roots=self._live_scratchpad_roots()
                ),
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
        policy_revision = store.current_policy_revision
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
                admission=ExecutionAdmission(
                    policy_revision=policy_revision,
                    once_requirements=(
                        tuple(requirements) if choice == "allow_once" else None
                    ),
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

    @staticmethod
    def _remote_exec_requirements(
        proposal: DangerousCommandProposal,
    ) -> frozenset[RemoteExecGrantRequirement] | None:
        if proposal.destination is None:
            return None
        return frozenset({
            RemoteExecGrantRequirement(
                scope=proposal.scope, destination=proposal.destination
            )
        })

    async def resolve_remote_exec_grant(
        self,
        tool: BaseTool,
        args: BaseModel,
        ctx: PermissionContext,
        tool_name: str,
        tool_call_id: str | None,
    ) -> ToolDecision | None:
        """Resolve a grantable remote-exec denial, prompting when allowed.

        The destination comes from the proposal (the later SSH admission
        classifies one validated destination per call). Returns None when the
        denial must keep the existing behavior: no typed proposal, no
        destination, or a bare loop without user input. A child never
        prompts, whatever its user-input capability; it denies with the
        persisted destination knob named. A decided call identity is never
        re-prompted.
        """
        proposal = ctx.dangerous_command_proposal
        if proposal is None:
            return None
        requirements = self._remote_exec_requirements(proposal)
        if requirements is None:
            return None
        if self._is_subagent_getter() or not self._user_input_capability_getter():
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
        if choice is None or choice == "deny":
            return _prompt_outcome_decision(
                store, identity, proposal, choice, ctx, tool_name
            )
        return await self.resume_after_remote_exec_grant(
            tool, args, tool_name, proposal, choice, requirements
        )

    async def resume_after_remote_exec_grant(
        self,
        tool: BaseTool,
        args: BaseModel,
        tool_name: str,
        proposal: DangerousCommandProposal,
        choice: GrantChoice,
        requirements: frozenset[RemoteExecGrantRequirement],
    ) -> ToolDecision:
        """Apply a remote-exec destination grant, recheck, and resume the call.

        Same discipline as the scope-keyed flow: install runtime authority
        first, fully re-resolve, destroy an unconsumed once permit on every
        exit path, and recheck once more after a persistence await.
        """
        store = self._grants_store
        policy_revision = store.current_policy_revision
        permit = None
        try:
            if choice == "allow_once":
                if requirements:
                    permit = store.install_remote_exec_once(requirements)
                    self._tool_manager_getter()._invalidate_dangerous_command_authority()
            elif store.grant_remote_exec_session(requirements):
                self._tool_manager_getter()._invalidate_dangerous_command_authority()
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
                store.destroy_remote_exec_once(permit)
                self._tool_manager_getter()._invalidate_dangerous_command_authority()
                permit = None
            result_note = None
            if choice == "allow_project":
                result_note = await self.persist_project_remote_exec_grant(proposal)
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
                admission=ExecutionAdmission(
                    policy_revision=policy_revision,
                    remote_exec_requirements=(
                        tuple(requirements) if choice == "allow_once" else None
                    ),
                ),
            )
        finally:
            if permit is not None:
                store.destroy_remote_exec_once(permit)
                self._tool_manager_getter()._invalidate_dangerous_command_authority()

    async def persist_project_remote_exec_grant(
        self, proposal: DangerousCommandProposal
    ) -> str:
        """Attempt to persist an approved destination grant; report it apart.

        The live destination grant is already applied; this only reports the
        persistence outcome. A failed or uncertain save never blocks the
        session grant; cancellation propagates.
        """
        destination = proposal.destination
        if destination is None:
            return "Remote-exec grant applies to this session only."
        port = self._grant_port_getter()
        if port is None:
            return (
                f"Remote-exec grant for {destination} applies to this session; "
                "this session cannot save remote-exec grants to your user config."
            )
        try:
            saved = await self._read_saved_grants(
                self._config_orchestrator_getter(), proposal.project
            )
        except Exception:
            logger.warning(
                "Saved remote-exec grants read failed for %s",
                proposal.project,
                exc_info=True,
            )
            saved = None
        if saved is None or saved.unavailable is not None or not saved.user_revision:
            return (
                f"Remote-exec grant for {destination} applies to this session "
                "only; your saved grants could not be read, so it was not saved "
                "to your config."
            )
        try:
            result = await port.save_remote_exec_grant(
                self._session_id_getter(),
                proposal.project,
                destination.key,
                saved.user_revision,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Project remote-exec grant save failed for %s",
                destination,
                exc_info=True,
            )
            result = None
        return remote_exec_grant_save_note(
            proposal.project, proposal.scope, destination, result
        )

    async def resolve_out_of_root_grant(
        self,
        tool: BaseTool,
        args: BaseModel,
        ctx: PermissionContext,
        tool_name: str,
        tool_call_id: str | None,
    ) -> ToolDecision | None:
        """Resolve an out-of-root denial, prompting when the runtime allows.

        Returns None when the denial must keep the existing behavior: the tool
        has no grantable target, or no promptable target can be proposed.
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
        target = root_grant_target(
            tool_name, args, self._cwd_getter(), ctx.out_of_root_target
        )
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


class LoopUserInputAdmission(UserInputAdmission):
    """The loop's user-input admission policy (plan U/OD1, plan D2).

    Implements the broker's :class:`UserInputAdmission` protocol from live
    loop state, so the decision runs in the executing agent process: child
    model questions are refused at admission, and the unattended-expiry
    policy is computed from provenance- and policy-based gates only — never
    from model-controlled values beyond the validated timeout override.
    """

    def __init__(
        self,
        *,
        is_subagent_getter: Callable[[], bool],
        user_input_capability_getter: Callable[[], bool],
        config_getter: Callable[[], ChartreuxConfigSchema],
        session_id_getter: Callable[[], str],
        session_generation_getter: Callable[[], int],
        active_turn_id_getter: Callable[[], str | None],
        deferred_questions: DeferredQuestionStore,
    ) -> None:
        self._is_subagent_getter = is_subagent_getter
        self._user_input_capability_getter = user_input_capability_getter
        self._config_getter = config_getter
        self._session_id_getter = session_id_getter
        self._session_generation_getter = session_generation_getter
        self._active_turn_id_getter = active_turn_id_getter
        self._deferred_questions = deferred_questions

    def child_refusal(self, event: UserInputRequestEvent) -> str | None:
        """Child-origin admission refusal (plan D2).

        A subagent's model question never opens a user-input callback. The
        settling of the broker waiter is the whole refusal — the asking tool
        call resolves with the typed denial, nothing is granted, and no
        authority state is touched.
        """
        if (
            event.origin is UserInputOrigin.MODEL_QUESTION
            and self._is_subagent_getter()
        ):
            return CHILD_QUESTION_REFUSAL
        return None

    def expiry_policy(
        self, event: UserInputRequestEvent
    ) -> UnattendedExpiryPolicy | None:
        """Admission for unattended expiry (plan U/OD1).

        Returns the expiry policy snapshot to attach to the request, or None
        when the request is permanently ineligible. Raises ValueError when a
        model-supplied timeout override does not fit the configured range
        (rejected, never clamped).

        Every gate is provenance- or policy-based, never model-controlled:
        runtime approvals (dangerous-command and root-access grants) share
        the broker but are permanently ineligible, and a requires_user
        question (or a missing default) blocks the batch.
        """
        args = event.args
        if not isinstance(args, UserQuestionRequest):
            return None
        eligible = (
            # Runtime approvals are permanently ineligible (see docstring).
            event.origin is UserInputOrigin.MODEL_QUESTION
            # Root agent in a supported interactive session only.
            and not self._is_subagent_getter()
            and self._user_input_capability_getter()
            and self._config_getter().user_questions.unattended_timeout_enabled
        )
        if not eligible:
            return None
        questions = self._echoed_questions(event.request_id, args)
        if questions is None:
            return None
        config = self._config_getter().user_questions
        # The only actions a default may declare are defer_affected_work and
        # preserve_existing_choice: neither authorizes anything new, and no
        # option label can become an answer through expiry.
        timeout = config.default_timeout_seconds
        # The timeout override is tool-owned (AskUserQuestionArgs); the
        # generic runtime request stays free of it.
        override = getattr(args, "timeout_seconds", None)
        if override is not None:
            if (
                override < MIN_UNATTENDED_TIMEOUT_SECONDS
                or override > config.max_timeout_seconds
            ):
                raise ValueError(
                    "ask_user_question timeout_seconds must fit the configured "
                    "[user_questions] range "
                    f"[{MIN_UNATTENDED_TIMEOUT_SECONDS:g}, "
                    f"{config.max_timeout_seconds:g}] seconds; got "
                    f"{override:g} (rejected, not clamped)"
                )
            timeout = override
        return UnattendedExpiryPolicy(timeout_seconds=timeout, questions=questions)

    @staticmethod
    def _echoed_questions(
        request_id: str, args: UserQuestionRequest
    ) -> list[UnansweredQuestion] | None:
        """Echoed questions with stable IDs, or None when not defaultable.

        Every question must be explicitly defaultable (plan U/OD1): a
        requires_user question (or a missing default) blocks the whole batch,
        so a mixed or protected batch never expires.
        """
        echoed: list[UnansweredQuestion] = []
        for index, question in enumerate(args.questions):
            default = question.unattended_default
            if question.kind is QuestionKind.REQUIRES_USER or default is None:
                return None
            echoed.append(
                UnansweredQuestion(
                    question_id=f"{request_id}:{index}",
                    question=question.question,
                    header=question.header,
                    kind=question.kind,
                    unattended_default=default,
                )
            )
        return echoed or None

    def session_id(self) -> str:
        return self._session_id_getter()

    def session_generation(self) -> int:
        return self._session_generation_getter()

    def active_turn_id(self) -> str | None:
        return self._active_turn_id_getter()

    def record_unanswered(
        self, request_id: str, unanswered: UnansweredQuestions
    ) -> None:
        """Record exactly one deferred-question record for an expiry."""
        self._deferred_questions.record_unanswered(
            session_id=self._session_id_getter(),
            request_id=request_id,
            unanswered=unanswered,
        )
