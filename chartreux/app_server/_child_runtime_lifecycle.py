from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from chartreux.app_server._execution import (
    SessionExecution,
    SessionExecutionConflict,
    SessionExecutionKind,
)
from chartreux.app_server._model import ProtocolModel
from chartreux.app_server._root_session import SessionCoordinator
from chartreux.app_server._runtime import (
    SessionDangerousCommandGrantPort,
    SessionRootGrantPort,
)
from chartreux.app_server._session_history import SessionHistory
from chartreux.app_server._subagent_records import (
    AgentRecord,
    RunRecord,
    StoredChildSession,
    StoredRunResult,
    _AgentState,
    _AgentTombstone,
)
from chartreux.app_server._turns import DeliverCallback, TurnController
from chartreux.app_server.models import (
    OpenCallbackState,
    PublicCheckpointEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicSessionState,
    PublicTurn,
)
from chartreux.app_server.protocol import (
    HistoryEntryAddedParams,
    HistoryEntryUpdatedParams,
    SessionCompactedParams,
    StatsUpdatedParams,
    TurnCompletedParams,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.events import BaseEvent
from chartreux.core.subagents import (
    AgentEviction,
    RunStatus,
    RunStopReason,
    SubagentRunnerPort,
)
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._runtime import AgentRuntimeFactory
    from chartreux.app_server._sessions import SessionRuntime


class ChildRuntimeAssemblyView(SubagentRunnerPort, SessionCoordinator, Protocol):
    """Narrow typed view over the registry-owned child-runtime assembly state.

    The two one-method ports the assembly forwards to the TurnController
    (the runner port and the session coordinator) are inherited, so the view
    stays narrow while remaining assignable where those ports are expected.
    Dynamic access: every member is read through the view at call time, so
    the root/child maps, the notification sink, the callback ports, the
    tool IO port, and the runtime factory are never snapshotted at
    construction — registry state mutated or rebound after the assembler is
    built stays visible, and instance patches of the registry delegates
    (including the notification capture) still shadow.
    """

    _root: SessionRuntime | None
    _children: dict[str, SessionRuntime]
    _readable_children: dict[str, SessionRuntime]
    _agent_records: dict[str, AgentRecord]
    _registry_lock: asyncio.Lock
    _draining_children: bool
    _suppressed_notifications: set[str]
    _notify_child: Callable[[str, ProtocolModel], Awaitable[None]]
    _deliver_callback: DeliverCallback
    _tool_io: ToolIOPort | None
    _runtime_factory: AgentRuntimeFactory

    def _public_state(
        self,
        runtime: SessionRuntime,
        history_limit: int,
        *,
        turns_limit: int | None = None,
        include_history: bool = True,
        include_turns: bool = True,
    ) -> PublicSessionState: ...

    def _resolve_session_loop(self, session_id: str) -> AgentLoop | None: ...

    def _root_grant_save(
        self, session_id: str, root: Path, expected_revision: str
    ) -> Awaitable[ConfigSaveResult]: ...

    def _dangerous_command_grant_save(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> Awaitable[ConfigSaveResult]: ...

    def _capture_child_notification(
        self,
        runtime: SessionRuntime,
        record: AgentRecord | None,
        run: RunRecord | None,
        params: ProtocolModel,
    ) -> Coroutine[Any, Any, None]: ...

    def _generation_is_current(self, generation: int) -> bool: ...

    def _resolve_run_outcome(
        self, run: RunRecord, turn: PublicTurn | None = None
    ) -> tuple[RunStatus, RunStopReason | None]: ...

    def _emit_agents_update(
        self, evictions: list[AgentEviction] | None = None
    ) -> Coroutine[Any, Any, None]: ...


class ChildRuntimeBinding:
    """Late-bound child runtime context (ASC area 7, migration step 1).

    Replaces the assembly's late-bound runtime closure: initially unbound,
    bound exactly once after TurnController construction. Exposes the
    snapshot and notification methods handed to the TurnController; the
    notification method snapshots record, run, and child generation BEFORE
    transport delivery and validates those exact identities afterward. The
    registry-owned child maps, notification sink, and capture delegate stay
    canonical (§6.7); the binding holds only the assembled runtime reference
    and reads every dependency through the view at call time.
    """

    def __init__(self, child: AgentLoop, view: ChildRuntimeAssemblyView) -> None:
        self.child = child
        self.runtime: SessionRuntime | None = None
        self._view = view

    def bind(self, runtime: SessionRuntime) -> None:
        """Bind the assembled runtime exactly once, after TurnController construction."""
        self.runtime = runtime

    def snapshot_state(self) -> PublicSessionState:
        """Project the bound runtime; an unbound assembly cannot be projected."""
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("Child session runtime is not bound")
        return self._view._public_state(runtime, 200)

    async def notify_child(self, method: str, params: ProtocolModel) -> None:
        """Deliver one child notification and capture it under the exact claim."""
        bound = self.runtime
        record = next(
            (
                record
                for record in self._view._agent_records.values()
                if record.runtime is bound
            ),
            None,
        )
        run = record.current_run if record is not None else None
        generation = self.child._session_generation
        await self._view._notify_child(method, params)
        if bound is not None and self.child._session_generation == generation:
            await self._view._capture_child_notification(bound, record, run, params)


class ChildRuntimeAssembler:
    """Child runtime assembly, lookup, and notification capture for the registry.

    Narrow collaborator of the session runtime registry (ASC area 7,
    migration step 1): the late-bound runtime binding replaces the
    assembly's closure pair, the assembler builds the child runtime over the
    registry-owned typed view (grant callbacks, state projection,
    notification capture, accounting retirement), the runtime lookups route
    the root/child maps, and the notification capture validates the exact
    record/run/child identities around transport delivery. Canonical child
    maps, admission flags, and the shared locks stay registry-owned (§6.7);
    no private collaborator locks, tasks, queues, or relay layers. The
    registry keeps thin same-name delegates so existing instance patches and
    direct calls keep working.
    """

    def __init__(self, view: ChildRuntimeAssemblyView) -> None:
        self._view = view

    def build(
        self,
        child: AgentLoop,
        runtime_type: type[SessionRuntime],
        *,
        base_history: list[PublicHistoryEntry] | None = None,
        event_sink: Callable[[BaseEvent], Awaitable[None]] | None = None,
    ) -> SessionRuntime:
        """Assemble one child runtime and bind it exactly once.

        The SessionRuntime type is supplied by the registry facade, resolved
        at call time so the facade keeps object identity; the grant ports,
        callback ports, state projection, notification capture, and
        accounting retirement arrive through the registry-owned view.
        """
        # The runtime-owned port resolves the calling session, so a child's
        # approved grant lands in that child's store only, never the parent's.
        # Its save routes through the root orchestrator keyed by the registered
        # caller's cwd; child orchestrators never write.
        child.bind_root_grant_port(
            SessionRootGrantPort(
                self._view._resolve_session_loop, self._view._root_grant_save
            )
        )
        child.bind_dangerous_command_grant_port(
            SessionDangerousCommandGrantPort(
                self._view._resolve_session_loop,
                self._view._dangerous_command_grant_save,
            )
        )
        execution = SessionExecution()
        history = SessionHistory(base_history or [])
        binding = ChildRuntimeBinding(child, self._view)
        turns = TurnController(
            child,
            binding.notify_child,
            self._view._deliver_callback,
            execution,
            self._view,
            snapshot_state=binding.snapshot_state,
            tool_io=self._view._tool_io,
            event_sink=event_sink,
            session_coordinator=self._view,
        )
        runtime = runtime_type(
            child,
            turns,
            execution,
            history,
            retire_accounting=lambda: (
                self._view._runtime_factory.retire_child_accounting(child)
            ),
        )
        binding.bind(runtime)
        return runtime

    def runtime(self, session_id: str | None) -> SessionRuntime:
        if session_id is None:
            raise RuntimeError("Subagent parent session is missing")
        root = self._view._root
        if root is not None and root.agent_loop.session_id == session_id:
            return root
        child = self._view._children.get(session_id)
        if child is None:
            raise RuntimeError(f"Subagent parent session not found: {session_id}")
        return child

    def require_child(self, session_id: str) -> SessionRuntime:
        child = self._view._children.get(
            session_id
        ) or self._view._readable_children.get(session_id)
        if child is None:
            raise KeyError(session_id)
        return child

    async def capture_child_notification(  # noqa: PLR0912
        self,
        runtime: SessionRuntime,
        record: AgentRecord | None,
        run: RunRecord | None,
        params: ProtocolModel,
    ) -> None:
        # The sink may await arbitrary transport work. Recheck the exact claim,
        # not just the session id, so reuse and teardown cannot revive old updates.
        async with self._view._registry_lock:
            if record is None or run is None:
                return
            if (
                self._view._agent_records.get(record.agent_id) is not record
                or record.runtime is not runtime
                or self._view._children.get(record.session_id) is not runtime
                or record.current_run is not run
            ):
                return
            if (
                record.state is not _AgentState.RUNNING
                or runtime._closed
                or self._view._draining_children
                or record.agent_id in self._view._suppressed_notifications
                or not self._view._generation_is_current(record.root_generation)
            ):
                return
            if getattr(params, "session_id", None) != record.session_id:
                return
            turn_id = getattr(params, "turn_id", None)
            active_turn = runtime.turns.active_turn
            if turn_id is not None and (
                active_turn is None or turn_id != active_turn.id
            ):
                return
            before = (
                record.context_tokens,
                record.context_window,
                record.compacting,
                record.stop_reason,
            )
            if isinstance(params, StatsUpdatedParams):
                record.context_tokens = (
                    params.stats.context_tokens
                    if params.stats.context_tokens >= 0
                    else None
                )
                record.context_window = params.context_window
            elif isinstance(params, HistoryEntryAddedParams):
                entry = params.entry
                if (
                    isinstance(entry, PublicCheckpointEntry)
                    and entry.kind == "compaction"
                    and entry.generation_status
                    is PublicEntryGenerationStatus.IN_PROGRESS
                ):
                    record.compacting = True
                    record.compaction_entry_id = entry.id
            elif isinstance(params, SessionCompactedParams):
                # This precedes stats and the completed checkpoint patch. Never
                # expose the old numerator with activity already cleared.
                record.context_tokens = None
                record.compacting = False
                record.compaction_entry_id = None
            elif isinstance(params, HistoryEntryUpdatedParams):
                if params.entry_id == record.compaction_entry_id and any(
                    patch.path == "/generationStatus" and patch.value == "completed"
                    for patch in params.patch
                ):
                    # Finalization also completes checkpoints on error. Only
                    # CompactEndEvent supplies structured summary details; use
                    # that proof when logging-off compaction has no ID handoff.
                    succeeded = any(
                        patch.path == "/details"
                        and isinstance(patch.value, dict)
                        and "summaryLength" in patch.value
                        and "oldSessionId" in patch.value
                        and "newSessionId" in patch.value
                        for patch in params.patch
                    )
                    if succeeded:
                        record.context_tokens = None
                        record.compacting = False
                    record.compaction_entry_id = None
            elif isinstance(params, TurnCompletedParams):
                if active_turn is not None and params.turn.id != active_turn.id:
                    return
                record.compacting = False
                record.compaction_entry_id = None
                _, run.stop_reason = self._view._resolve_run_outcome(run, params.turn)
                record.stop_reason = run.stop_reason
            after = (
                record.context_tokens,
                record.context_window,
                record.compacting,
                record.stop_reason,
            )
        if before != after:
            try:
                await self._view._emit_agents_update()
            except Exception as exc:
                logger.warning("Failed to publish child context update", exc_info=exc)


class ChildDrainView(Protocol):
    """Narrow typed view over the registry-owned resume/close/drain state.

    Dynamic access: every member is read through the view at call time, so
    the admission and drain flags, the reaper fields, the producer/task
    inventory, the result collections, the child maps, and the shared locks
    are never snapshotted at construction — registry state mutated or
    rebound after the collaborator is built stays visible, and instance
    patches of the registry delegates (including the drain itself, patched
    at the class by test_session.py:4797) still shadow.
    """

    _ensure_child_lock: asyncio.Lock
    _creating_children: int
    _agent_records: dict[str, AgentRecord]
    _admission_closed: bool
    _draining_children: bool
    _registry_lock: asyncio.Lock
    _reaper_task: asyncio.Task[None] | None
    _reaper_deadline: float | None
    _drain_suppressed_inventory: bool
    _detached_reaper_tasks: set[asyncio.Task[None]]
    _monitor_tasks: set[asyncio.Task[None]]
    _eviction_tasks: set[asyncio.Task[bool]]
    _evicted_agents: dict[str, _AgentTombstone]
    _pending_evictions: dict[str, object]
    _result_store: dict[tuple[str, str], StoredRunResult]
    _expired_results: set[tuple[str, str]]
    _latest_run_ids: dict[str, str]
    _wait_leases: dict[tuple[str, str], int]
    _pending_notifications: set[tuple[str, str]]
    _result_write_tokens: dict[tuple[str, str], object]
    _children: dict[str, SessionRuntime]
    _child_links: dict[str, tuple[SessionRuntime, str]]
    _stored_children: dict[str, StoredChildSession]
    _readable_children: dict[str, SessionRuntime]
    _detached_child_ids: set[str]
    _pending_child_closes: dict[str, tuple[SessionRuntime, asyncio.Task[None]]]
    _teardown_tasks: set[asyncio.Task[None]]
    _suppression_owners: dict[str, int]
    _suppressed_notifications: set[str]

    def _require_policy_unreserved(self) -> None: ...

    def _clear_replacement_locked(self, record: AgentRecord) -> None: ...

    @staticmethod
    def _retire_teardown_task(task: asyncio.Task[None]) -> None: ...

    def drain_children(self) -> Coroutine[Any, Any, None]: ...

    def _emit_agents_update(
        self, evictions: list[AgentEviction] | None = None
    ) -> Coroutine[Any, Any, None]: ...


class DrainContext:
    """Per-invocation drain state (ASC area 7, migration step 3 — 17.7c).

    Temporary state for ONE drain_children invocation: the accumulated
    errors, the change intent, and the phase-specific runtime snapshots
    that cross a phase boundary. Producer task sets are NOT snapshotted
    at entry — earlier phases create later teardown work, so each phase
    method snapshots its own producers when it runs. The suppressed-
    inventory half of the publication intent stays registry-owned on the
    view (_drain_suppressed_inventory); only the change intent rides the
    context. No state moves out of the registry (§6.7).
    """

    def __init__(self) -> None:
        self.errors: list[BaseException] = []
        self.agents_changed: bool = False
        self.children: list[SessionRuntime] = []
        self.pending_close_runtimes: list[SessionRuntime] = []


class ChildDrain:
    """Resume admission, child close, and the final drain for the registry.

    Narrow collaborator of the session runtime registry (ASC area 7,
    migration step 2 — the resume/close facade): reserve_resume closes child
    admission atomically with the live-work check (the check and the
    admission closure share the publication/materialization lock);
    close_children closes admission and delegates to the drain — plain
    drain_children has different flag behavior, so the two are NOT
    collapsed; drain_children runs the seven named phase methods in
    producer order (live/displaced reapers, monitors, evictions, remaining
    runs, tracked teardowns, retryable runtimes, final inventory — 17.7c)
    over one per-invocation DrainContext; discard_child closes one child
    loop through the late-resolving facade callback, preserving the dynamic
    _sessions.close_agent_loop lookup for the existing patches. Canonical
    state stays registry-owned (§6.7); the collaborator creates no
    independent task collections and holds no private locks. The registry
    keeps thin same-name delegates so existing instance patches, class
    patches, and direct calls keep working.
    """

    def __init__(self, view: ChildDrainView) -> None:
        self._view = view

    async def reserve_resume(self, root: SessionRuntime) -> None:
        """Close child admission atomically with the live-work check."""
        async with self._view._ensure_child_lock:
            reasons = []
            if (
                root.execution.active is not None
                and root.execution.active.kind is not SessionExecutionKind.LIFECYCLE
            ):
                reasons.append("a turn or operation is running")
            if root.turns.has_queued_turns:
                reasons.append("input is queued")
            if any(
                isinstance(callback.state, OpenCallbackState)
                for callback in root.turns.callbacks
            ):
                reasons.append("a decision is pending")
            if self._view._creating_children or any(
                (
                    record.current_run is not None
                    and record.current_run.status is RunStatus.RUNNING
                )
                or record.reserved_for_replacement
                for record in self._view._agent_records.values()
            ):
                reasons.append("background agents are running")
            if reasons:
                raise SessionExecutionConflict(
                    "Cannot switch sessions while "
                    + ", ".join(reasons)
                    + ". Wait for work to finish, answer pending decisions, remove queued input, "
                    "or stop work first; then retry. Nothing was switched."
                )
            self._view._admission_closed = True

    async def close_children(self) -> None:
        self._view._require_policy_unreserved()
        self._view._admission_closed = True
        await self._view.drain_children()

    async def drain_children(self) -> None:
        """Run the drain phases in producer order over one DrainContext."""
        self._view._require_policy_unreserved()
        self._view._draining_children = True
        async with self._view._registry_lock:
            for record in self._view._agent_records.values():
                if record.reserved_for_replacement:
                    self._view._clear_replacement_locked(record)
        context = DrainContext()
        try:
            await self._drain_reapers(context)
            await self._drain_monitors(context)
            await self._drain_evictions(context)
            await self._drain_remaining_runs(context)
            await self._drain_tracked_teardowns(context)
            await self._drain_retryable_runtimes(context)
            await self._drain_final_inventory(context)
        finally:
            self._view._draining_children = False
            self._view._drain_suppressed_inventory = False

    async def _drain_reapers(self, context: DrainContext) -> None:
        """Cancel and join the live reaper and the displaced reapers."""
        reaper = self._view._reaper_task
        self._view._reaper_task = None
        self._view._reaper_deadline = None
        if reaper is not None and reaper is not asyncio.current_task():
            # Cancelling the reaper may abort an in-flight publication:
            # record the lost intent so the final empty inventory below
            # still surfaces the disappearance.
            self._view._drain_suppressed_inventory = True
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
        # A reaper that handed its schedule to a fresh timer (a
        # displaced reaper) is still an inventory producer until it
        # settles: drain owns it too, so drain cannot return — or
        # publish its final empty inventory — while a displaced reaper
        # can still deliver a payload. The registered reaper is never
        # in the detached set (the handoff unregisters before arming
        # the fresh timer); the guard keeps the join single.
        detached_reapers = [
            task
            for task in self._view._detached_reaper_tasks
            if task is not asyncio.current_task() and task is not reaper
        ]
        for task in detached_reapers:
            task.cancel()
        if detached_reapers:
            # Same for displaced reapers: a cancelled reaper may have
            # been about to publish its reduced inventory.
            self._view._drain_suppressed_inventory = True
            await asyncio.gather(*detached_reapers, return_exceptions=True)

    async def _drain_monitors(self, context: DrainContext) -> None:
        """Cancel and join the run monitors."""
        monitors = [
            task
            for task in self._view._monitor_tasks
            if task is not asyncio.current_task()
        ]
        for task in monitors:
            task.cancel()
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)

    async def _drain_evictions(self, context: DrainContext) -> None:
        """Join the eviction cleanup tasks."""
        evictions = [
            task
            for task in self._view._eviction_tasks
            if task is not asyncio.current_task()
        ]
        if evictions:
            await asyncio.gather(*evictions, return_exceptions=True)

    async def _drain_remaining_runs(self, context: DrainContext) -> None:
        """Cancel the remaining runs and clear the registry-owned inventory."""
        async with self._view._ensure_child_lock:
            self._view._require_policy_unreserved()
            background_tasks = [
                record.current_run.completion_task
                for record in self._view._agent_records.values()
                if record.current_run is not None
                and record.current_run.status is RunStatus.RUNNING
            ]
            for task in background_tasks:
                task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            context.agents_changed = bool(
                self._view._agent_records or self._view._evicted_agents
            )
            self._view._agent_records.clear()
            self._view._evicted_agents.clear()
            self._view._pending_evictions.clear()
            self._view._result_store.clear()
            self._view._expired_results.clear()
            self._view._latest_run_ids.clear()
            self._view._wait_leases.clear()
            self._view._pending_notifications.clear()
            self._view._result_write_tokens.clear()
            context.children = list(self._view._children.values())
            context.pending_close_runtimes = list(
                {
                    id(runtime): runtime
                    for runtime, _ in self._view._pending_child_closes.values()
                }.values()
            )
            self._view._children.clear()
            self._view._child_links.clear()
            self._view._stored_children.clear()
            self._view._readable_children.clear()
            self._view._detached_child_ids.clear()

    async def _drain_tracked_teardowns(self, context: DrainContext) -> None:
        """Join the tracked teardown tasks."""
        teardown = [
            task
            for task in self._view._teardown_tasks
            if task is not asyncio.current_task()
        ]
        if teardown:
            await asyncio.gather(*teardown, return_exceptions=True)

    async def _drain_retryable_runtimes(self, context: DrainContext) -> None:
        """Retry the pending child closes and retire their failed wrappers."""
        for runtime in context.pending_close_runtimes:
            if not runtime._closed:
                try:
                    await runtime.close()
                except BaseException as exc:
                    context.errors.append(exc)
                    continue
            for session_id, pending in list(self._view._pending_child_closes.items()):
                if pending[0] is runtime:
                    self._view._pending_child_closes.pop(session_id, None)
                    failed_wrapper = pending[1]
                    self._view._teardown_tasks.discard(failed_wrapper)
                    self._view._retire_teardown_task(failed_wrapper)
            close_task = runtime._close_task
            if close_task is not None:
                self._view._teardown_tasks.discard(close_task)

    async def _drain_final_inventory(self, context: DrainContext) -> None:
        """Close the remaining children and publish the final empty inventory."""
        self._view._suppression_owners.clear()
        self._view._suppressed_notifications.clear()
        for runtime in context.children:
            try:
                await runtime.close()
            except BaseException as exc:
                context.errors.append(exc)
        # A disappearance publication suppressed or cancelled at the
        # drain boundary (a reaper pruning the final tombstone across
        # the boundary, or an in-flight publication aborted by
        # cancelling its reaper) must still surface: the final empty
        # inventory is emitted so the delivered client inventory never
        # keeps a ghost row for the drained registry.
        if context.agents_changed or self._view._drain_suppressed_inventory:
            try:
                await self._view._emit_agents_update()
            except BaseException as exc:
                context.errors.append(exc)
        if len(context.errors) == 1:
            raise context.errors[0]
        if context.errors:
            raise BaseExceptionGroup(
                "Failed to drain child session runtimes", context.errors
            )

    @staticmethod
    async def discard_child(
        child: AgentLoop, close_loop: Callable[[AgentLoop], Awaitable[None]]
    ) -> None:
        with suppress(Exception):
            await close_loop(child)
