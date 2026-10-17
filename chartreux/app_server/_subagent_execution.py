from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from chartreux.app_server._subagent_records import (
    AgentRecord,
    RunRecord,
    StoredRunResult,
    _ActiveWorkSlot,
    _AgentState,
    _AgentTombstone,
    _ReplacementClaim,
)
from chartreux.app_server.models import PublicTurn, PublicTurnStatus, TextContentBlock
from chartreux.app_server.protocol import TurnInterruptParams, TurnStartParams
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.events import BaseEvent, ToolStreamEvent
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.subagents import (
    AgentBusyError,
    AgentClosedError,
    AgentEviction,
    AgentProfileMismatchError,
    LaunchConfig,
    LaunchOutcome,
    RetentionPolicy,
    RunStatus,
    RunStopReason,
    SubagentRunAccumulator,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    normalize_task_summary,
)
from chartreux.model_display import format_model_display_name
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime
    from chartreux.app_server._streaming import BoundedEventQueue
    from chartreux.app_server._subagent_results import PublicationDecision
    from chartreux.app_server._turns import TurnStartAction
    from chartreux.core.agent_loop._config_transactions import (
        _PreparedLaunchReconfiguration,
    )
    from chartreux.core.agent_loop.backend_lifetime import BackendPublish
    from chartreux.core.agents.launch import LaunchCandidate
    from chartreux.core.agents.models import AgentProfile
    from chartreux.core.tools.base import InvokeContext

_BACKGROUND_PROGRESS_LIMIT = 32
_MAX_RUN_HISTORY = 32


class ExecutionStateView(Protocol):
    """Narrow typed view over the registry-owned execution state.

    Dynamic access: every member is read through the view at call time, so
    the active-work slot set, the monitor-task set, the record/child maps,
    the latest-run index, the shared registry lock, the result store, the
    pending notifications, the drain flag, the notification suppression,
    the retention policy, and the clock are never snapshotted at
    construction — registry state mutated after the collaborator is built
    stays visible,
    and instance patches of the registry delegates (including the background
    admission validation, the launch-candidate resolution, the profile
    admission, the child creation + rollback pair, the run-id allocation,
    the clock, the retention policy, the run-outcome resolution, the
    generation check, the result expiry, the write-token registration, the
    publication decision, the agents-update emission, the record naming,
    and the close-on-completion transition) still shadow.
    """

    _active_work_slots: set[_ActiveWorkSlot]
    _agent_records: dict[str, AgentRecord]
    _children: dict[str, SessionRuntime]
    _latest_run_ids: dict[str, str]
    _registry_lock: asyncio.Lock
    _result_store: dict[tuple[str, str], StoredRunResult]
    _pending_notifications: set[tuple[str, str]]
    _monitor_tasks: set[asyncio.Task[None]]
    _draining_children: bool
    _suppressed_notifications: set[str]
    _retention_policy: RetentionPolicy
    _clock: Callable[[], float]

    def _validate_background_admission(
        self, parent: SessionRuntime, identity: tuple[str, int]
    ) -> None: ...

    def _live_tombstone_locked(
        self, agent_id: str, now: float
    ) -> _AgentTombstone | None: ...

    @staticmethod
    def _tombstone_reuse_error(
        agent_id: str, tombstone: _AgentTombstone
    ) -> AgentClosedError: ...

    def _validate_replacement_locked(self, claim: _ReplacementClaim) -> None: ...

    def _require_task_profile_allowed(
        self, parent: SessionRuntime, profile: str
    ) -> None: ...

    def _rearm_reaper_locked(self, exclude: frozenset[str] = frozenset()) -> None: ...

    def _resolve_launch_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        profile: str | None = None,
    ) -> LaunchCandidate: ...

    def _issue_agent_id(self, parent: SessionRuntime) -> Coroutine[Any, Any, str]: ...

    def _create_registered_child(
        self, parent: SessionRuntime, args: TaskArgs, ctx: InvokeContext
    ) -> Coroutine[Any, Any, tuple[SessionRuntime, AgentProfile]]: ...

    def _rollback_created_child(
        self, parent: SessionRuntime, runtime: SessionRuntime, tool_call_id: str
    ) -> Coroutine[Any, Any, None]: ...

    def _rollback_reused_claim(
        self,
        record: AgentRecord,
        previous_idle_since: float | None,
        *,
        apply_policy: bool = False,
    ) -> Coroutine[Any, Any, None]: ...

    @staticmethod
    def _new_run_id(agent_id: str) -> str: ...

    def _resolve_run_outcome(
        self, run: RunRecord, turn: PublicTurn | None = None
    ) -> tuple[RunStatus, RunStopReason | None]: ...

    def _generation_is_current(self, generation: int) -> bool: ...

    def _expire_results_locked(self) -> None: ...

    def _register_writer_locked(
        self, agent_id: str, run_id: str, token: object
    ) -> None: ...

    def _publish_locked(
        self, context: BackgroundExecutionContext
    ) -> PublicationDecision: ...

    def _emit_agents_update(
        self, evictions: list[AgentEviction] | None = None
    ) -> Coroutine[Any, Any, None]: ...

    @staticmethod
    def _record_name(record: AgentRecord) -> str: ...

    def _close_unretained_locked(
        self, record: AgentRecord, run: RunRecord | None, *, name: str
    ) -> bool: ...


@dataclass
class ForegroundExecutionContext:
    """Per-invocation state for one foreground execution (ASC area 3).

    Carries the child runtime and loop, the bounded progress queue, the run
    accumulator, and the completion metadata mark captured at admission; the
    previous event sink, the turn id, the completion task, and the work slot;
    and the terminal result once the stream settles. The registry-owned slot
    set stays canonical; the context holds only the foreground run's
    per-invocation state. Contexts hold REFERENCES, not copies.
    """

    runtime: SessionRuntime
    child: AgentLoop
    progress: BoundedEventQueue[ToolStreamEvent]
    accumulator: SubagentRunAccumulator
    completion_metadata_mark: tuple[int, int]
    tool_call_id: str
    previous_event_sink: Callable[[BaseEvent], Awaitable[None]] | None
    slot: _ActiveWorkSlot
    turn_id: str | None = None
    completion: asyncio.Task[PublicTurn] | None = None
    result: TaskResult | None = None

    async def consume_event(self, event: BaseEvent) -> None:
        """Feed one child event into the bounded progress queue.

        The former foreground consume_event closure of the registry's
        admitted run (the 17.6a de-captured consumer), converted to a
        context method: the accumulator, the progress queue, and the
        tool-call id are context fields. The queue-feeding body is
        unchanged.
        """
        update = self.accumulator.observe(event, tool_call_id=self.tool_call_id)
        if update is None:
            return
        try:
            self.progress.put_nowait(update)
        except asyncio.QueueFull:
            with suppress(asyncio.QueueEmpty):
                self.progress.get_nowait()
            self.progress.put_nowait(update)

    def discard_work_slot(
        self, view: ExecutionStateView, _task: asyncio.Task[PublicTurn]
    ) -> None:
        """Release the capacity slot when the completion task settles.

        The former slot discard callback of the registry's admitted run,
        converted to a named context method (the context-method-takes-the
        view convention): the registry-owned active-work slot set stays
        canonical and is read through the view at call time.
        """
        view._active_work_slots.discard(self.slot)


@dataclass
class BackgroundExecutionContext:
    """Per-invocation state for one background execution (ASC area 3).

    Carries the exact child runtime, record, and run record with their IDs;
    the created/reused status and launch profile; the previous idle
    timestamp and the generation/authority reservations; the retained
    candidate and prepared reconfiguration; the prior record/retention/
    latest-run/index snapshot the rollback restores; the previous event
    sink; the accumulator and completion metadata mark; the prepared
    completion future and the installed monitor; the publication token,
    turn id, start action, and backend publication; and the mutable
    committed flag. Contexts hold REFERENCES, not copies: the monitor reads
    the same committed flag the acceptance writes, and the registry-owned
    state stays canonical.
    """

    runtime: SessionRuntime
    record: AgentRecord
    run_record: RunRecord
    agent_id: str
    run_id: str
    tool_call_id: str
    created_record: bool
    launch_profile: AgentProfile | None
    previous_idle_since: float | None
    reserved_child_generation: int | None
    reserved_parent_authority_revision: int | None
    retained_candidate: LaunchCandidate | None
    prepared_reconfiguration: _PreparedLaunchReconfiguration | None
    prior_current_run: RunRecord | None
    prior_retained: bool
    prior_latest_run_id: str | None
    prior_registry_latest_present: bool
    prior_registry_latest: str | None
    prior_idle_since: float | None
    previous_event_sink: Callable[[BaseEvent], Awaitable[None]] | None
    accumulator: SubagentRunAccumulator
    completion_metadata_mark: tuple[int, int]
    completion_placeholder: asyncio.Future[None]
    publication_token: object
    monitor: asyncio.Task[None] | None = None
    turn_id: str | None = None
    action: TurnStartAction | None = None
    backend_publication: BackendPublish | None = None
    committed: bool = False

    async def consume_event(self, event: BaseEvent) -> None:
        """Feed one child event into the run record's progress summaries.

        The former background consume_event closure of the registry's
        admitted run (the 17.6a de-captured consumer), converted to a
        context method: the accumulator, the run record, and the tool-call
        id are context fields. The observation body is unchanged.
        """
        update = self.accumulator.observe(event, tool_call_id=self.tool_call_id)
        if update is None:
            return
        if len(self.run_record.progress_summaries) < _BACKGROUND_PROGRESS_LIMIT:
            self.run_record.progress_summaries.append(update.message)
        else:
            self.run_record.progress_overflow = True

    def settle_prepared_waiters(self, _task: asyncio.Task[None]) -> None:
        """Settle the admission-time completion future exactly once.

        The former settle_prepared_waiters done-callback of the registry's
        admitted run (the 17.6a de-captured callback), converted to a
        context method: the prepared completion future is a context field.
        A waiter can snapshot the prepared run before its monitor is
        installed; keep that admission-time future linked to the same
        completion, including a prepared-start stop.
        """
        if not self.completion_placeholder.done():
            self.completion_placeholder.set_result(None)


class SubagentExecution:
    """Invocation orchestration for subagent execution (ASC area 3).

    Narrow collaborator of the session runtime registry: the execution
    per-invocation contexts and their converted closures — the event
    consumers and the prepared-waiter callback as context methods, the
    foreground teardown as the executor's teardown_foreground, the
    background preparation + rollback as the executor's prepare_background
    + rollback_background, the background monitor + finalization as the
    executor's monitor, and the background acceptance + acknowledgment as
    the executor's accept_background + acknowledge_background — live here,
    while the run envelope and the foreground streaming stay on the
    registry and operate on the contexts.
    Canonical state stays registry-owned (§6.7); no private collaborator
    locks; no new tasks/queues/relay layers. The registry wires the
    collaborator in its constructor; teardown_foreground reads no registry
    state, so it stays a @staticmethod and the admitted-run path calls it
    through the class, keeping the registry's patch seams. The background
    path is not driven through spec'd mock registries, so its moved regions
    read registry state through the view — instance patches and the manual
    clock still shadow.
    """

    def __init__(self, view: ExecutionStateView) -> None:
        self._view = view

    @staticmethod
    async def teardown_foreground(context: ForegroundExecutionContext) -> None:
        """Tear down one foreground execution after its stream settles.

        Replaces the former teardown_foreground closure of the registry's
        admitted run: the completion task and the runtime are context
        fields (narrowed with an assert — the ReleaseContext.cleanup alias
        convention). The cleanup region and the exception suppression are
        unchanged.
        """
        errors: list[BaseException] = []
        completion = context.completion
        runtime = context.runtime
        assert completion is not None
        try:
            if not completion.done():
                active_turn = runtime.turns.active_turn
                if active_turn is not None:
                    runtime.turns.interrupt(
                        TurnInterruptParams(
                            session_id=active_turn.session_id,
                            expected_turn_id=active_turn.id,
                        )
                    )
                else:
                    await runtime.turns.close()
            await asyncio.gather(completion, return_exceptions=True)
        except BaseException as exc:
            errors.append(exc)
        try:
            await runtime.close()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Failed to tear down foreground agent", errors)

    async def prepare_background(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        parent: SessionRuntime,
        parent_identity: tuple[str, int],
        args: TaskArgs,
        ctx: InvokeContext,
        slot: _ActiveWorkSlot,
        *,
        replacement: _ReplacementClaim | None,
        summary_task: str | None,
    ) -> BackgroundExecutionContext:
        """Prepare one background execution and return its context.

        The former background preparation region of the registry's admitted
        run (ASC area 3, step 3): the reuse reservation, the candidate
        resolution and reconfiguration preparation, the created-child path,
        and the run-record setup, moved intact. Registry state is read
        through the view at call time; the returned context carries the
        preparation + acceptance-tail state as references.
        """
        self._view._validate_background_admission(parent, parent_identity)
        retained_candidate: LaunchCandidate | None = None
        previous_idle_since: float | None = None
        reserved_child_generation: int | None = None
        reserved_parent_authority_revision: int | None = None
        if args.agent_id is not None:
            async with self._view._registry_lock:
                self._view._validate_background_admission(parent, parent_identity)
                record = self._view._agent_records.get(args.agent_id)
                if record is None:
                    tombstone = self._view._live_tombstone_locked(
                        args.agent_id, self._view._clock()
                    )
                    if tombstone is not None:
                        raise self._view._tombstone_reuse_error(
                            args.agent_id, tombstone
                        )
                    raise UnknownAgentError(
                        f"Unknown agent_id instance handle: {args.agent_id}. Check the agent_id with check_agents"
                    )
                if (
                    "agent_type" in args.model_fields_set
                    and record.profile != args.agent_type
                ):
                    raise AgentProfileMismatchError(
                        f"Agent profile mismatch: expected {record.profile}, got {args.agent_type}"
                    )
                if replacement is not None:
                    self._view._validate_replacement_locked(replacement)
                    if record is not replacement.record:
                        raise RuntimeError("Agent replacement residency changed")
                elif (
                    record.reserved_for_replacement
                    or record.state is not _AgentState.IDLE
                ):
                    raise AgentBusyError("Agent is already running a task")
                if (
                    record.runtime._closed is True
                    or self._view._children.get(record.session_id) is not record.runtime
                ):
                    raise UnknownAgentError(
                        f"Unknown agent_id instance handle: {args.agent_id}. Check the agent_id with check_agents"
                    )
                self._view._require_task_profile_allowed(parent, record.profile)
                previous_idle_since = record.idle_since
                reserved_child_generation = (
                    record.runtime.agent_loop._session_generation
                )
                reserved_parent_authority_revision = (
                    parent.agent_loop._authority_revision
                )
                record.reserved_for_replacement = False
                # Keep ownership through the launch acknowledgment: this
                # generator's finalizer may run after another retask reserves.
                record.state = _AgentState.RUNNING
                record.idle_since = None
                self._view._rearm_reaper_locked()
        else:
            record = None
        created_record = False
        launch_profile: AgentProfile | None = None
        effective_task = summary_task if summary_task is not None else args.task
        prepared_reconfiguration = None
        if record is not None:
            if args.config is not None or isinstance(
                record.runtime.agent_loop.launch_overrides, LaunchConfig
            ):
                try:
                    retained_candidate = self._view._resolve_launch_candidate(
                        parent, args, runtime=record.runtime, profile=record.profile
                    )
                except BaseException:
                    await self._view._rollback_reused_claim(
                        record,
                        previous_idle_since,
                        apply_policy=replacement is not None,
                    )
                    raise
            if retained_candidate is not None:
                try:
                    assert reserved_child_generation is not None
                    assert reserved_parent_authority_revision is not None
                    prepared_reconfiguration = await record.runtime.agent_loop.prepare_launch_reconfiguration(
                        retained_candidate,
                        expected_session_generation=reserved_child_generation,
                        expected_parent_authority_revision=reserved_parent_authority_revision,
                    )
                except BaseException:
                    await self._view._rollback_reused_claim(
                        record,
                        previous_idle_since,
                        apply_policy=replacement is not None,
                    )
                    raise
            if parent.turns._projector is not None:
                try:
                    await parent.turns.link_subagent(
                        ctx.tool_call_id, record.session_id
                    )
                    async with self._view._registry_lock:
                        self._view._validate_background_admission(
                            parent, parent_identity
                        )
                        if (
                            self._view._agent_records.get(record.agent_id) is not record
                            or self._view._children.get(record.session_id)
                            is not record.runtime
                            or record.runtime._closed is True
                        ):
                            raise RuntimeError(
                                "Agent admission changed during child reuse"
                            )
                except BaseException:
                    if prepared_reconfiguration is not None:
                        await record.runtime.agent_loop.abort_launch_reconfiguration(
                            prepared_reconfiguration
                        )
                    await self._view._rollback_reused_claim(
                        record,
                        previous_idle_since,
                        apply_policy=replacement is not None,
                    )
                    raise
            runtime = record.runtime
        else:
            self._view._require_task_profile_allowed(parent, args.agent_type)
            runtime, profile = await self._view._create_registered_child(
                parent, args, ctx
            )
            launch_profile = profile
            agent_id = await self._view._issue_agent_id(parent)
            record = AgentRecord(
                agent_id=agent_id,
                profile=runtime.agent_loop.launch_profile or args.agent_type,
                session_id=runtime.agent_loop.session_id,
                runtime=runtime,
                root_generation=parent.agent_loop._session_generation,
                idle_ttl_seconds=profile.idle_ttl_seconds,
                parent_session_id=parent.agent_loop.session_id,
                initial_task_summary=normalize_task_summary(
                    args.task_summary, fallback=effective_task
                ),
                state=_AgentState.RUNNING,
            )
            try:
                async with self._view._registry_lock:
                    self._view._validate_background_admission(parent, parent_identity)
                    if self._view._children.get(record.session_id) is not runtime:
                        raise RuntimeError(
                            "Agent admission changed during child creation"
                        )
                    self._view._agent_records[agent_id] = record
            except BaseException:
                await self._view._rollback_created_child(
                    parent, runtime, ctx.tool_call_id
                )
                raise
            created_record = True

        record.work_slot = slot
        agent_id = record.agent_id
        run_id = self._view._new_run_id(agent_id)
        prior_current_run = record.current_run
        prior_retained = record.retained
        prior_latest_run_id = record.latest_run_id
        prior_registry_latest_present = agent_id in self._view._latest_run_ids
        prior_registry_latest = self._view._latest_run_ids.get(agent_id)
        prior_idle_since = record.idle_since
        previous_event_sink = runtime.turns._event_sink
        publication_token = object()
        accumulator = SubagentRunAccumulator()
        completion_placeholder = asyncio.get_running_loop().create_future()
        run_record = RunRecord(
            run_id=run_id,
            agent_id=agent_id,
            profile=record.profile,
            status=RunStatus.RUNNING,
            completion_task=completion_placeholder,
            task_summary=normalize_task_summary(
                args.task_summary, fallback=effective_task
            ),
        )
        record.current_run = run_record
        record.latest_run_id = run_id
        self._view._latest_run_ids[agent_id] = run_id
        # Commit the retention policy for this admission. A fresh launch
        # resolves explicit task arg -> profile setting -> generation
        # snapshot default; a reuse inherits the committed policy unless
        # the caller explicitly overrides it (retain=False is a final
        # continuation that closes after this run).
        if created_record:
            assert launch_profile is not None
            # Stable first-admission timestamp for deterministic Active
            # ordering; preserved across reuse and run-history pruning.
            record.first_admitted_at = self._view._clock()
            if args.retain is not None:
                record.retained = args.retain
            elif launch_profile.retain is not None:
                record.retained = launch_profile.retain
            else:
                record.retained = self._view._retention_policy.retain_agents
            # Capture the profile display label only when the profile's
            # display_name was explicitly provided (design §5): a TOML
            # profile without display_name synthesizes a title-case
            # picker label, which must not become the name segment —
            # the derivation helper falls back to the profile name.
            record.profile_display_label = (
                launch_profile.display_name
                if launch_profile.display_name_explicit
                else None
            )
        elif args.retain is not None:
            record.retained = args.retain
        run_record.start_count = sum(
            message.role is Role.assistant for message in runtime.agent_loop.messages
        )
        completion_metadata_mark = runtime.agent_loop.completion_metadata_mark()
        return BackgroundExecutionContext(
            runtime=runtime,
            record=record,
            run_record=run_record,
            agent_id=agent_id,
            run_id=run_id,
            tool_call_id=ctx.tool_call_id,
            created_record=created_record,
            launch_profile=launch_profile,
            previous_idle_since=previous_idle_since,
            reserved_child_generation=reserved_child_generation,
            reserved_parent_authority_revision=reserved_parent_authority_revision,
            retained_candidate=retained_candidate,
            prepared_reconfiguration=prepared_reconfiguration,
            prior_current_run=prior_current_run,
            prior_retained=prior_retained,
            prior_latest_run_id=prior_latest_run_id,
            prior_registry_latest_present=prior_registry_latest_present,
            prior_registry_latest=prior_registry_latest,
            prior_idle_since=prior_idle_since,
            previous_event_sink=previous_event_sink,
            accumulator=accumulator,
            completion_metadata_mark=completion_metadata_mark,
            completion_placeholder=completion_placeholder,
            publication_token=publication_token,
        )

    async def rollback_background(
        self,
        context: BackgroundExecutionContext,
        parent: SessionRuntime,
        *,
        replacement: _ReplacementClaim | None,
    ) -> None:
        """Roll back a failed background admission.

        The former rollback region of the registry's admitted run (ASC area
        3, step 3): the prepared-start abort, the reconfiguration rollback,
        the event-sink restoration, the completion cancellation, the write
        token consumption (the results publication decision — the decision
        itself is unused on rollback), the prior-state restore, and the
        created/reused rollback, moved intact. The caller's except block
        keeps the re-raise tail.
        """
        if context.action is not None and not context.committed:
            context.action.abort()
        if context.prepared_reconfiguration is not None and not context.committed:
            if context.backend_publication is not None:
                context.runtime.agent_loop.rollback_launch_reconfiguration(
                    context.prepared_reconfiguration, context.backend_publication
                )
            else:
                await context.runtime.agent_loop.abort_launch_reconfiguration(
                    context.prepared_reconfiguration
                )
        context.runtime.turns._event_sink = context.previous_event_sink
        attempted = context.run_record.completion_task
        if not attempted.done():
            attempted.cancel()
        with suppress(asyncio.CancelledError):
            await attempted
        async with self._view._registry_lock:
            self._view._publish_locked(context)
            owns_attempt = (
                self._view._agent_records.get(context.agent_id) is context.record
                and context.record.current_run is context.run_record
            )
            if owns_attempt:
                context.record.current_run = context.prior_current_run
                context.record.retained = context.prior_retained
                context.record.latest_run_id = context.prior_latest_run_id
                context.record.idle_since = context.prior_idle_since
                if self._view._latest_run_ids.get(context.agent_id) == context.run_id:
                    if context.prior_registry_latest_present:
                        assert context.prior_registry_latest is not None
                        self._view._latest_run_ids[context.agent_id] = (
                            context.prior_registry_latest
                        )
                    else:
                        self._view._latest_run_ids.pop(context.agent_id, None)
        if context.created_record:
            async with self._view._registry_lock:
                if self._view._agent_records.get(context.agent_id) is context.record:
                    self._view._agent_records.pop(context.agent_id)
            await self._view._rollback_created_child(
                parent, context.runtime, context.tool_call_id
            )
        else:
            await self._view._rollback_reused_claim(
                context.record,
                context.previous_idle_since,
                apply_policy=replacement is not None,
            )

    async def monitor(  # noqa: PLR0912, PLR0915
        self,
        context: BackgroundExecutionContext,
        parent: SessionRuntime,
        slot: _ActiveWorkSlot,
    ) -> None:
        """Monitor one background execution and finalize its result.

        The former monitor closure of the registry's admitted run (ASC area
        3, step 4): the run observation, the result build, the residency and
        publication transaction, the completion notification, and the
        close-on-completion transition, moved intact with the scheduling
        unchanged (the monitor task creation stays with the acceptance
        flow). The turn id is read from the context field and narrowed with
        an assert (the teardown_foreground precedent); the parent runtime
        and the capacity slot are parameters. The token-based publication
        authority (the residency check, the notification suppression, and
        the exact write-token check/pop) is decided by the results
        collaborator's publish_locked through the view; the result store
        write and the pending-notification add stay in this locked block,
        gated by the returned PublicationDecision. Registry state is read
        through the view at call time; the event-sink restore and the
        committed gate are unchanged.
        """
        turn_id = context.turn_id
        assert turn_id is not None
        try:
            turn = await context.runtime.turns.wait_for_operation(turn_id)
            if turn.error is not None:
                context.accumulator.record_error(turn.error.message)
            end_count = sum(
                message.role is Role.assistant
                for message in context.runtime.agent_loop.messages
            )
            context.run_record.result = context.accumulator.build_result(
                turns_used=end_count - context.run_record.start_count,
                completed=turn.status is PublicTurnStatus.COMPLETED,
            ).model_copy(
                update={
                    "agent_id": context.agent_id,
                    "run_id": context.run_id,
                    "metadata": context.runtime.agent_loop.completion_metadata_since(
                        context.completion_metadata_mark
                    ),
                }
            )
            context.run_record.status, context.run_record.stop_reason = (
                self._view._resolve_run_outcome(context.run_record, turn)
            )
        except asyncio.CancelledError:
            context.run_record.status, context.run_record.stop_reason = (
                self._view._resolve_run_outcome(
                    context.run_record,
                    context.runtime.turns.operation_terminal_turn(turn_id),
                )
            )
            context.run_record.result = context.accumulator.build_result(
                turns_used=sum(
                    message.role is Role.assistant
                    for message in context.runtime.agent_loop.messages
                )
                - context.run_record.start_count,
                completed=context.run_record.status is RunStatus.COMPLETED,
            ).model_copy(
                update={
                    "agent_id": context.agent_id,
                    "run_id": context.run_id,
                    "metadata": context.runtime.agent_loop.completion_metadata_since(
                        context.completion_metadata_mark
                    ),
                }
            )
            raise
        except Exception as exc:
            context.accumulator.record_error(str(exc))
            context.run_record.status = RunStatus.FAILED
            context.run_record.stop_reason = RunStopReason.ERROR
            context.run_record.result = context.accumulator.build_result(
                turns_used=0, completed=False
            ).model_copy(
                update={
                    "agent_id": context.agent_id,
                    "run_id": context.run_id,
                    "metadata": context.runtime.agent_loop.completion_metadata_since(
                        context.completion_metadata_mark
                    ),
                }
            )
        finally:
            if context.run_record.result is not None:
                context.run_record.result = context.run_record.result.model_copy(
                    update={"stop_reason": context.run_record.stop_reason}
                )
            active_task = context.runtime.turns._active_task
            if context.record.reserved_for_replacement:
                # The old execution is joined, but its allocation now
                # belongs to the handoff, including notification awaits.
                pass
            elif active_task is not None and not active_task.done():
                # Release/drain may cancel the monitor before provider
                # cancellation finishes; keep counting the actual work.
                active_task.add_done_callback(
                    lambda _: self._view._active_work_slots.discard(slot)
                )
            else:
                self._view._active_work_slots.discard(slot)
            context.runtime.turns._event_sink = context.previous_event_sink
            if not context.committed:
                return  # noqa: B012
            context.run_record.completed_at = self._view._clock()
            committed_model = context.runtime.agent_loop.committed_model
            if committed_model is not None:
                context.run_record.terminal_identity = (
                    committed_model.base_model,
                    committed_model.provider,
                    committed_model.wire_name,
                )
            key = (context.agent_id, context.run_id)
            async with self._view._registry_lock:
                decision = self._view._publish_locked(context)
                if decision.resident:
                    context.record.compacting = False
                    context.record.compaction_entry_id = None
                    context.record.stop_reason = context.run_record.stop_reason
                    committed_model = context.runtime.agent_loop.committed_model
                    context.record.base_model = (
                        committed_model.base_model
                        if committed_model is not None
                        else None
                    )
                    if committed_model is not None:
                        context.record.effective_model = format_model_display_name(
                            committed_model.provider, committed_model.wire_name
                        )
                    context.record.active_provider = (
                        committed_model.provider
                        if committed_model is not None
                        else None
                    )
                    context.record.state = (
                        _AgentState.RUNNING
                        if context.record.reserved_for_replacement
                        else _AgentState.FINALIZING
                    )
                    context.record.idle_since = context.run_record.completed_at
                    context.record.last_task_summary = context.run_record.task_summary
                    context.record.last_run_status = context.run_record.status
                    context.record.current_run = None
                    if context.run_record not in context.record.run_history:
                        context.record.run_history.append(context.run_record)
                        if len(context.record.run_history) > _MAX_RUN_HISTORY:
                            context.record.run_history = context.record.run_history[
                                -_MAX_RUN_HISTORY:
                            ]
                if decision.write_permitted and context.run_record.result is not None:
                    self._view._result_store[key] = StoredRunResult(
                        agent_id=context.agent_id,
                        run_id=context.run_id,
                        result=context.run_record.result,
                        completed_at=context.run_record.completed_at,
                        root_generation=context.record.root_generation,
                        terminal_identity=context.run_record.terminal_identity,
                    )
                if decision.notify_permitted:
                    self._view._pending_notifications.add(key)
                self._view._expire_results_locked()
            if not decision.notify_permitted:
                return  # noqa: B012
            try:
                await self._view._emit_agents_update()
            except Exception as exc:
                logger.warning(
                    "Failed to publish background agent update", exc_info=exc
                )
            try:
                async with self._view._registry_lock:
                    notification_permitted = (
                        self._view._generation_is_current(
                            context.record.root_generation
                        )
                        and self._view._agent_records.get(context.agent_id)
                        is context.record
                        and self._view._children.get(context.record.session_id)
                        is context.runtime
                        and not self._view._draining_children
                        and context.agent_id not in self._view._suppressed_notifications
                        and key in self._view._result_store
                        and key in self._view._pending_notifications
                    )
                if not notification_permitted:
                    return  # noqa: B012
                # Notify the parent that a background run completed.
                status_word = (
                    "budget-stopped (budget exceeded)"
                    if context.run_record.stop_reason is RunStopReason.BUDGET_EXCEEDED
                    else "budget-stopped (budget unverifiable)"
                    if context.run_record.stop_reason
                    is RunStopReason.BUDGET_UNVERIFIABLE
                    else "completed"
                    if context.run_record.status is RunStatus.COMPLETED
                    else "failed"
                    if context.run_record.status is RunStatus.FAILED
                    else "cancelled (by user)"
                    if context.run_record.stop_reason is RunStopReason.USER_CANCELLED
                    else "retasked"
                    if context.run_record.stop_reason is RunStopReason.RETASKED
                    else "cancelled"
                )
                # The notification derives from this run's own
                # record state at completion (§5): name alongside
                # handle, run, and outcome.
                notification = (
                    f"Background agent {self._view._record_name(context.record)} "
                    f"({context.agent_id}, run {context.run_id})"
                    f" {status_word}."
                    " Use get_agent_result to retrieve the result."
                )
                if (
                    context.run_record.stop_reason is not RunStopReason.USER_CANCELLED
                    and parent.turns.active_turn is None
                    and parent.turns._active_task is None
                    and not parent.turns.queue_state.items
                ):
                    try:
                        _response, _start = parent.turns.start(
                            TurnStartParams(
                                session_id=parent.agent_loop.session_id,
                                message=[TextContentBlock(text=notification)],
                            )
                        )
                        _start()
                    except Exception:
                        parent.agent_loop._pending_injected_messages.append(
                            LLMMessage(
                                role=Role.user, content=notification, injected=True
                            )
                        )
                else:
                    parent.agent_loop._pending_injected_messages.append(
                        LLMMessage(role=Role.user, content=notification, injected=True)
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to finalize background agent notification", exc_info=exc
                )
            finally:
                async with self._view._registry_lock:
                    self._view._pending_notifications.discard(key)
                    if (
                        self._view._agent_records.get(context.agent_id)
                        is context.record
                        and self._view._children.get(context.record.session_id)
                        is context.runtime
                        and not context.record.reserved_for_replacement
                        and context.record.state is _AgentState.FINALIZING
                        and self._view._generation_is_current(
                            context.record.root_generation
                        )
                    ):
                        # Completion transition: a retained instance
                        # returns to idle; an unretained instance closes
                        # (detach + `closed` tombstone + tracked runtime
                        # close) after the result was published above.
                        if not self._view._close_unretained_locked(
                            context.record,
                            context.run_record,
                            name=(
                                f"vibe-subagent-close-on-completion:{context.agent_id}"
                            ),
                        ):
                            context.record.state = _AgentState.IDLE
                    self._view._expire_results_locked()
                    if not self._view._draining_children:
                        self._view._rearm_reaper_locked()
                if not self._view._draining_children:
                    try:
                        await self._view._emit_agents_update()
                    except Exception as exc:
                        logger.warning(
                            "Failed to publish agent completion transition",
                            exc_info=exc,
                        )

    async def accept_background(
        self,
        context: BackgroundExecutionContext,
        parent: SessionRuntime,
        parent_identity: tuple[str, int],
        slot: _ActiveWorkSlot,
    ) -> None:
        """Accept one prepared background execution and commit the run.

        The former acceptance region of the registry's admitted run (ASC
        area 3, step 5): the admission revalidation under the registry
        lock, the monitor task creation (the scheduling is unchanged — the
        task is created here exactly as the acceptance flow created it), the
        prepared-waiter settle install, the monitor-task tracking, and the
        acceptance writes (the write-token registration routed to the
        results collaborator, the committed flag, the slot transfer, the
        launch metadata install + finalize), moved intact. The start action
        is read from the context field (the existing assert narrows it); the
        parent runtime, the parent identity, and the capacity slot are
        parameters. Registry state is read through the view at call time.
        """
        action = context.action
        async with self._view._registry_lock:
            self._view._validate_background_admission(parent, parent_identity)
            if (
                self._view._agent_records.get(context.agent_id) is not context.record
                or self._view._children.get(context.record.session_id)
                is not context.runtime
                or context.record.current_run is not context.run_record
                or context.runtime._closed is True
            ):
                raise RuntimeError("Agent admission changed before launch")
            if context.reserved_child_generation is not None and (
                context.runtime.agent_loop._session_generation
                != context.reserved_child_generation
                or parent.agent_loop._authority_revision
                != context.reserved_parent_authority_revision
            ):
                raise RuntimeError("Agent authority changed before launch")
            self._view._require_task_profile_allowed(parent, context.record.profile)
            self._view._register_writer_locked(
                context.agent_id, context.run_id, context.publication_token
            )
        monitor_coro = self.monitor(context, parent, slot)
        try:
            context.monitor = asyncio.create_task(
                monitor_coro, name=f"vibe-subagent-background:{context.run_id}"
            )
        except BaseException:
            monitor_coro.close()
            raise
        context.run_record.completion_task = context.monitor

        context.monitor.add_done_callback(context.settle_prepared_waiters)
        self._view._monitor_tasks.add(context.monitor)
        context.monitor.add_done_callback(self._view._monitor_tasks.discard)
        assert action is not None
        if context.prepared_reconfiguration is not None:
            context.backend_publication = (
                context.runtime.agent_loop.publish_launch_reconfiguration(
                    context.prepared_reconfiguration
                )
            )
        action()
        context.run_record.started_at = self._view._clock()
        context.record.stop_reason = None
        context.record.compacting = False
        context.record.compaction_entry_id = None
        context.record.context_window = (
            context.runtime.agent_loop.config.get_active_model().auto_compact_threshold
        )
        if context.prepared_reconfiguration is not None:
            # Acceptance makes the new envelope authoritative in memory
            # before persistence can block on the logger save lock.
            context.runtime.agent_loop.install_launch_metadata()
        if context.retained_candidate is not None:
            context.record.effective_model = (
                context.retained_candidate.effective_model.alias
            )
            context.record.effective_thinking = (
                context.retained_candidate.effective_thinking
            )
        else:
            context.record.effective_model = (
                context.runtime.agent_loop.config.active_model
            )
            context.record.effective_thinking = (
                context.runtime.agent_loop.config.get_active_model().thinking
            )
        committed_model = context.runtime.agent_loop.committed_model
        if committed_model is not None:
            context.record.effective_model = format_model_display_name(
                committed_model.provider, committed_model.wire_name
            )
        context.record.base_model = (
            committed_model.base_model if committed_model is not None else None
        )
        context.record.active_provider = (
            committed_model.provider if committed_model is not None else None
        )
        context.committed = True
        slot.transferred = True
        if context.prepared_reconfiguration is not None:
            assert context.backend_publication is not None
            context.runtime.agent_loop.finalize_launch_reconfiguration(
                context.prepared_reconfiguration, context.backend_publication
            )

    async def acknowledge_background(
        self,
        context: BackgroundExecutionContext,
        *,
        replacement: _ReplacementClaim | None,
        defer_launch_agents_update: bool,
    ) -> TaskResult:
        """Acknowledge one accepted background execution.

        The former acknowledgment region of the registry's admitted run
        (ASC area 3, step 5): the launch-metadata persistence, the initial
        agents update, and the TaskResult construction, moved intact; the
        caller yields the returned acknowledgment. The yield -> return
        restructure keeps the admitted-run generator single-layered: no
        unowned forwarding generator is introduced, and a cancellation
        during the metadata persistence (after acceptance) does not roll
        the accepted run back — accepted execution is NOT acknowledgment
        delivery.
        """
        if context.prepared_reconfiguration is not None:
            # Acceptance has happened: persistence failure is reported without
            # rolling the live configuration or accepted run back.
            await context.runtime.agent_loop.persist_launch_metadata()
        if not defer_launch_agents_update:
            try:
                await self._view._emit_agents_update()
            except Exception as exc:
                logger.warning(
                    "Failed to publish initial background agent update", exc_info=exc
                )
        # An explicit status distinguishes this successful launch acknowledgment
        # from the terminal result that get_agent_result later returns. The
        # guidance is policy-aware: retained agents stay resident and can be
        # continued; unretained agents close on completion with the result
        # still retrievable.
        if context.record.retained:
            policy_guidance = (
                " This agent is retained and stays resident after the run "
                "completes; to continue this instance, use "
                f'task(agent_id="{context.agent_id}", background=true, task="...").'
            )
        else:
            policy_guidance = (
                " This agent is not retained and closes when the run "
                "completes; the result stays retrievable via "
                "get_agent_result. Launch with retain=true to keep a "
                "resident agent."
            )
        # The name is derived from the record's committed admission state;
        # refusals and failed admissions never relabel (§5).
        agent_name = self._view._record_name(context.record)
        return TaskResult(
            response=(
                f"Background agent {agent_name} ({context.agent_id}, run {context.run_id}) "
                "launched and running; use check_agents to monitor, "
                "get_agent_result or wait_for_agent to retrieve the "
                "result." + policy_guidance
            ),
            turns_used=0,
            completed=True,
            status="launched",
            launch_outcome=LaunchOutcome.LAUNCHED,
            agent_id=context.agent_id,
            run_id=context.run_id,
            name=agent_name,
            metadata={
                "base_model": context.record.base_model,
                "active_provider": context.record.active_provider,
                "effective_model": context.record.effective_model,
                **(
                    {
                        "replaced_run_id": replacement.run.run_id,
                        "replacement_run_id": context.run_id,
                    }
                    if replacement is not None
                    else {}
                ),
            },
        )
