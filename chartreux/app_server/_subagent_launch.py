from __future__ import annotations

from collections.abc import Awaitable, Iterator
from contextlib import AbstractAsyncContextManager, contextmanager, suppress
from dataclasses import dataclass
import fnmatch
from typing import TYPE_CHECKING, Protocol

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agents.launch import FrozenPersona, LaunchCandidate, resolve_launch
from chartreux.core.agents.models import AgentProfile
from chartreux.core.session.saved_sessions import delete_saved_session
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.models import ToolPermission, ToolPermissionError

if TYPE_CHECKING:
    from chartreux.app_server._runtime import AgentRuntimeFactory
    from chartreux.app_server._sessions import SessionRuntime

_AGENT_ID_HIGH_WATER_MARK_KEY = "_app_server_agent_id_high_water_mark"


class LaunchStateView(Protocol):
    """Narrow typed view over the registry-owned launch state.

    Dynamic access: every member is read through the view at call time, so
    replaceable scalars (root identity, generation, admission flags, the
    runtime factory) and the sequence counter are never snapshotted at
    construction — registry state mutated or rebound after the
    collaborator is built stays visible, and instance patches of the
    registry delegates still shadow.
    """

    _root: SessionRuntime | None
    _generation_identity: tuple[str, int] | None
    _admission_closed: bool
    _draining_children: bool
    _policy_reserved: bool
    _creating_children: int
    _children: dict[str, SessionRuntime]
    _child_links: dict[str, tuple[SessionRuntime, str]]
    _next_agent_seq: int
    _runtime_factory: AgentRuntimeFactory

    def _require_policy_unreserved(self) -> None: ...

    def _resolve_launch_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        profile: str | None = None,
    ) -> LaunchCandidate: ...

    def _build_child_runtime(self, child: AgentLoop) -> SessionRuntime: ...

    def _discard_child(self, child: AgentLoop) -> Awaitable[None]: ...

    def _child_publication(
        self, runtime: SessionRuntime
    ) -> AbstractAsyncContextManager[None]: ...


@dataclass
class ChildPublicationAttempt:
    """Per-invocation state for one child publication (ASC area 1).

    Holds the parent runtime and captured identity, the resolved launch
    candidate and tool-call id, the constructed child/runtime once they
    exist, and the two transaction flags the rollback region reads. The
    registry-owned child maps stay canonical; the attempt carries only
    the publication transaction's local state.
    """

    parent: SessionRuntime
    identity: tuple[str, int]
    candidate: LaunchCandidate
    tool_call_id: str
    child: AgentLoop | None = None
    runtime: SessionRuntime | None = None
    link_recorded: bool = False
    projection_started: bool = False

    async def rollback(self, view: LaunchStateView) -> None:
        """Remove every durable/runtime trace of an unaccepted publication.

        Replaces the local transaction flags of the former registry method
        without widening the cleanup region or changing exception
        suppression.
        """
        child = self.child
        runtime = self.runtime
        assert child is not None and runtime is not None
        session_id = child.session_id
        view._children.pop(session_id, None)
        view._child_links.pop(session_id, None)
        if self.projection_started:
            with suppress(Exception):
                await self.parent.turns.unlink_subagent(self.tool_call_id, session_id)
        if self.link_recorded:
            with suppress(Exception):
                await self.parent.agent_loop.forget_child_session(
                    session_id, self.tool_call_id
                )
        with suppress(Exception):
            await runtime.close()
        with suppress(Exception):
            await delete_saved_session(session_id, child.config.session_logging)


class ChildLaunchAdmission:
    """Launch admission, publication, and rollback for child sessions.

    Narrow collaborator of the session runtime registry (ASC area 1).
    Profile validation and candidate resolution (17.1a) are pure functions
    of the parent runtime and the launch arguments: they read no registry
    state, take no locks, and hold no state of their own. Sequence
    allocation and restoration, admission validation, the creation fence,
    registered-child publication, and created-child rollback (17.1b) run
    over the registry-owned typed view, which also supplies the runtime
    construction, child-publication context, and discard dependencies with
    call-time dynamic access. Canonical child maps, admission flags, and
    the sequence counter stay registry-owned (§6.7); only the existing
    registry locks are used; no private collaborator locks, tasks, queues,
    or relay layers. The registry keeps thin same-name delegates so
    existing instance patches and direct calls keep working.
    """

    def __init__(self, view: LaunchStateView) -> None:
        self._view = view

    def require_profile_allowed(self, parent: SessionRuntime, profile: str) -> None:
        """Apply effective Task admission before constructing child resources."""
        manager = parent.agent_loop.tool_manager
        if isinstance(manager, ToolManager) and "task" not in manager.available_tools:
            raise ToolPermissionError("Task tool is disabled")
        config = manager.get_tool_config("task")
        if (
            isinstance(manager, ToolManager)
            and config.permission == ToolPermission.NEVER
        ):
            raise ToolPermissionError("Task tool permission is never")
        if any(fnmatch.fnmatch(profile, pattern) for pattern in config.denylist):
            raise ToolPermissionError(f"Task denied for agent profile: {profile}")

    def resolve_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        profile: str | None = None,
    ) -> LaunchCandidate:
        """Resolve the launch candidate for a fresh or reused child."""
        child = runtime.agent_loop if runtime is not None else None
        retained_profile = (
            parent.agent_loop.agent_manager.get_agent(profile)
            if profile is not None
            else None
        )
        return resolve_launch(
            profile_name=(
                args.agent_type if "agent_type" in args.model_fields_set else profile
            ),
            config=args.config,
            parent_orchestrator=(
                child.config_orchestrator
                if child is not None
                else parent.agent_loop.config_orchestrator
            ),
            tool_inventory={
                name: object()
                for name in parent.agent_loop.tool_manager.registered_tools
            },
            authorized_tool_names=frozenset(
                parent.agent_loop.tool_manager.available_tools
            ),
            history=tuple(child.messages) if child is not None else (),
            agent_manager=parent.agent_loop.agent_manager,
            retained_profile=retained_profile,
            accumulated_overrides=(
                child.launch_overrides if child is not None else None
            ),
            committed_model=(
                child.committed_model
                if child is not None
                else parent.agent_loop.committed_model
            ),
            frozen_persona=(
                FrozenPersona(
                    system_prompt_id=child.frozen_system_prompt_id,
                    instructions=child.frozen_instructions,
                )
                if child is not None and child.frozen_system_prompt_id is not None
                else None
            ),
        )

    def validate_admission(
        self, parent: SessionRuntime, identity: tuple[str, int]
    ) -> None:
        """Reject a launch whose admission window closed or identity changed."""
        live_identity = (
            parent.agent_loop.session_id,
            parent.agent_loop._session_generation,
        )
        admission_blocked = (
            self._view._admission_closed or self._view._draining_children
        )
        identity_changed = (
            self._view._root is not parent
            or self._view._generation_identity != identity
            or live_identity != identity
        )
        if admission_blocked or identity_changed or parent._closed is True:
            message = (
                "Agent admission is closed"
                if self._view._admission_closed
                else "Agent admission changed during child creation"
            )
            raise RuntimeError(message)

    @contextmanager
    def creation_fence(self) -> Iterator[None]:
        """Hold the child-creation fence (policy reservation + counter)."""
        self._view._require_policy_unreserved()
        self._view._creating_children += 1
        try:
            yield
        finally:
            self._view._creating_children -= 1

    def restore_sequence(self, root: SessionRuntime) -> None:
        """Restore the agent-id high-water mark from persisted metadata."""
        metadata = root.agent_loop.session_logger.session_metadata
        config = metadata.config if metadata is not None else None
        high_water_mark = (
            config.get(_AGENT_ID_HIGH_WATER_MARK_KEY) if config is not None else None
        )
        if isinstance(high_water_mark, int) and not isinstance(high_water_mark, bool):
            self._view._next_agent_seq = max(
                self._view._next_agent_seq, high_water_mark
            )

    async def issue_agent_id(self, parent: SessionRuntime) -> str:
        """Allocate the next agent id, persisting the high-water mark."""
        self._view._next_agent_seq += 1
        reserved_seq = self._view._next_agent_seq
        metadata = parent.agent_loop.session_logger.session_metadata
        if metadata is not None:
            config = dict(metadata.config or {})
            config[_AGENT_ID_HIGH_WATER_MARK_KEY] = reserved_seq
            metadata.config = config
            await parent.agent_loop.session_logger._persist_metadata_field(
                "config", config
            )
        return f"agent-{reserved_seq}"

    async def create_registered(
        self, parent: SessionRuntime, args: TaskArgs, ctx: InvokeContext
    ) -> tuple[SessionRuntime, AgentProfile]:
        """Construct, register, and project one child, rolling back on failure."""
        candidate = self._view._resolve_launch_candidate(parent, args)
        attempt = ChildPublicationAttempt(
            parent=parent,
            identity=(
                parent.agent_loop.session_id,
                parent.agent_loop._session_generation,
            ),
            candidate=candidate,
            tool_call_id=ctx.tool_call_id,
        )
        self.validate_admission(parent, attempt.identity)
        with self.creation_fence():
            child = await self._view._runtime_factory.create_child(
                parent.agent_loop, candidate
            )
            # Spawn handoff, inside the creation fence: the child's first
            # message lists the working-set files the dispatcher hands it
            # (paths + last-seen revisions + current outlines), ahead of the
            # task brief. None when the parent has no working set.
            handoff = child.file_state_handoff_message()
            if handoff is not None:
                child.messages.append(handoff)
            attempt.child = child
        try:
            self.validate_admission(parent, attempt.identity)
        except BaseException:
            await self._view._discard_child(child)
            raise
        runtime = self._view._build_child_runtime(child)
        attempt.runtime = runtime
        async with self._view._child_publication(runtime):
            try:
                self.validate_admission(parent, attempt.identity)
            except BaseException:
                await self._view._discard_child(child)
                raise
            self._view._children[child.session_id] = runtime
            self._view._child_links[child.session_id] = (parent, ctx.tool_call_id)
            try:
                await child.persist_empty_session()
                self.validate_admission(parent, attempt.identity)
                await parent.agent_loop.record_child_session(
                    child, ctx.tool_call_id, candidate.profile.name
                )
                attempt.link_recorded = True
                self.validate_admission(parent, attempt.identity)
                attempt.projection_started = True
                if parent.turns._projector is not None:
                    await parent.turns.link_subagent(ctx.tool_call_id, child.session_id)
                self.validate_admission(parent, attempt.identity)
            except BaseException:
                await attempt.rollback(self._view)
                raise
            return runtime, candidate.profile

    async def rollback_created(
        self, parent: SessionRuntime, runtime: SessionRuntime, tool_call_id: str
    ) -> None:
        """Remove every durable/runtime trace of a child rejected before acceptance."""
        session_id = runtime.agent_loop.session_id
        self._view._children.pop(session_id, None)
        self._view._child_links.pop(session_id, None)
        with suppress(Exception):
            await parent.turns.unlink_subagent(tool_call_id, session_id)
        with suppress(Exception):
            await parent.agent_loop.forget_child_session(session_id, tool_call_id)
        with suppress(Exception):
            await runtime.close()
        with suppress(Exception):
            await delete_saved_session(
                session_id, runtime.agent_loop.config.session_logging
            )
