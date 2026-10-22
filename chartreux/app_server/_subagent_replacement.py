from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Protocol

from chartreux.app_server._subagent_records import (
    AgentRecord,
    RunRecord,
    _ActiveWorkSlot,
    _AgentState,
    _AgentTombstone,
    _ReplacementClaim,
)
from chartreux.core.agents.manager import requested_identity_reference
from chartreux.core.subagents import (
    AgentClosedError,
    AgentEvictedError,
    AgentIdentityMismatchError,
    LaunchConfigError,
    LaunchOutcome,
    RunStatus,
    RunStopReason,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermissionError

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime


class ReplacementStateView(Protocol):
    """Narrow typed view over the registry-owned replacement state.

    Dynamic access: every member is read through the view at call time, so
    the agent-record map, the resident child map, the capacity-slot set,
    the tracked teardown set, the shared registry lock, and the clock are
    never snapshotted at construction — registry state mutated after the
    collaborator is built stays visible, and instance patches of the
    registry delegates (including the identity-admission and reaper
    delegates) still shadow.
    """

    _agent_records: dict[str, AgentRecord]
    _children: dict[str, SessionRuntime]
    _active_work_slots: set[_ActiveWorkSlot]
    _teardown_tasks: set[asyncio.Task[None]]
    _registry_lock: asyncio.Lock
    _clock: Callable[[], float]

    def _runtime(self, session_id: str | None) -> SessionRuntime: ...

    def _validate_background_admission(
        self, parent: SessionRuntime, identity: tuple[str, int]
    ) -> None: ...

    def _live_tombstone_locked(
        self, agent_id: str, now: float
    ) -> _AgentTombstone | None: ...

    def _require_task_identity_allowed(
        self, parent: SessionRuntime, identity: str
    ) -> None: ...

    def _close_unretained_locked(
        self, record: AgentRecord, run: RunRecord | None, *, name: str
    ) -> bool: ...

    def _schedule_agents_update_locked(self) -> None: ...

    def _rearm_reaper_locked(self, exclude: frozenset[str] = frozenset()) -> None: ...

    @staticmethod
    def _interrupt_run(record: AgentRecord, run: RunRecord) -> bool: ...

    def _generation_is_current(self, generation: int) -> bool: ...

    def _invalidate_recovery_locked(self, record: AgentRecord | None) -> None: ...


class ReplacementCoordinator:
    """Replacement arbitration and claim lifecycle for the registry.

    Narrow collaborator of the session runtime registry (ASC area 2,
    migration steps 1–3): refusal construction and locked claim validation
    (17.4a), then reservation, the owner-fenced handoff context, settlement
    clearing, the completion-pending/settled unwind, and the reused-claim
    rollback (17.4b) operate on the registry-owned agent records, resident
    child maps, capacity slots, tracked teardowns, and the shared lock
    through the registry-owned typed view. The claim stays the
    exact-identity object attached to the canonical record;
    ``reserved_for_replacement``, ``replacement_owner``, and
    ``slot.transferred`` stay distinct phase markers. Canonical state stays
    registry-owned (§6.7); no private collaborator locks; no new
    tasks/queues/relay layers (the unwind task is still created into the
    registry-owned tracked teardown set). The registry keeps thin same-name
    delegates so existing instance patches and direct calls keep working.
    """

    def __init__(self, view: ReplacementStateView) -> None:
        self._view = view

    @staticmethod
    def replacement_refusal(record: AgentRecord, outcome: LaunchOutcome) -> TaskResult:
        """Build the typed refusal result for a rejected busy replacement."""
        run = record.current_run
        return TaskResult(
            response=f"Replacement not launched: {outcome.value}",
            turns_used=0,
            completed=False,
            launch_outcome=outcome,
            agent_id=record.agent_id,
            run_id=run.run_id if run is not None else record.latest_run_id,
            stop_reason=run.requested_stop_reason
            if run is not None
            else record.stop_reason,
        )

    @staticmethod
    def tombstone_reuse_error(
        agent_id: str, tombstone: _AgentTombstone
    ) -> AgentClosedError:
        """Select the typed reuse error by tombstone reason (§3.1 matrix).

        A ``closed`` tombstone rejects reuse with :class:`AgentClosedError` and
        continuation guidance; an ``evicted`` tombstone rejects it with the
        :class:`AgentEvictedError` subtype. Both admission sites look the
        tombstone up through :meth:`_live_tombstone_locked`, so a tombstone
        whose metadata deadline passed (before the reaper prunes it) behaves
        as unknown for reuse, per the §3.1 matrix.
        """
        if tombstone.reason == "closed":
            return AgentClosedError(
                "This instance is closed; launch a new task without "
                "agent_id. Use retain=true when a resident continuation "
                "is required."
            )
        return AgentEvictedError(f"Agent evicted: {agent_id}; start a new agent")

    def validate_locked(self, claim: _ReplacementClaim) -> None:
        """Re-validate claim residency, authority, and ownership (locked)."""
        record = claim.record
        self._view._validate_background_admission(claim.parent, claim.parent_identity)
        resident = (
            self._view._agent_records.get(record.agent_id) is record
            and self._view._children.get(record.session_id) is record.runtime
            and record.runtime._closed is not True
        )
        authorized = (
            record.runtime.agent_loop._session_generation == claim.child_generation
            and claim.parent.agent_loop._authority_revision == claim.authority_revision
            and record.parent_session_id == claim.parent_identity[0]
            and self._view._generation_is_current(record.root_generation)
        )
        if (
            not resident
            or not authorized
            or not record.reserved_for_replacement
            or record.replacement_owner is not claim
            or claim.slot not in self._view._active_work_slots
        ):
            raise RuntimeError("Agent replacement admission changed during handoff")
        self._view._require_task_identity_allowed(claim.parent, record.identity)

    async def reserve(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> _ReplacementClaim | TaskResult | None:
        """Reserve the target record for a busy replacement (public async port)."""
        if args.agent_id is None or not args.background:
            raise ValueError("replace_run requires agent_id and background mode")
        if ctx.is_subagent:
            raise RuntimeError("Agent depth limit of 1 reached")
        parent = self._view._runtime(ctx.session_id)
        identity = (parent.agent_loop.session_id, parent.agent_loop._session_generation)
        async with self._view._registry_lock:
            self._view._validate_background_admission(parent, identity)
            record = self._view._agent_records.get(args.agent_id)
            if record is None:
                tombstone = self._view._live_tombstone_locked(
                    args.agent_id, self._view._clock()
                )
                if tombstone is not None:
                    raise self.tombstone_reuse_error(args.agent_id, tombstone)
                raise UnknownAgentError(
                    f"Unknown agent_id instance handle: {args.agent_id}. Check the agent_id with check_agents"
                )
            if ctx.session_id != record.parent_session_id:
                raise ToolPermissionError(
                    "Only the owning parent may replace this agent's run"
                )
            # A busy replacement reuses the SAME instance: the requested
            # identity — structured or shorthand — must match the record's
            # resolved identity on both sides (11c).
            requested_identity = requested_identity_reference(args)
            retained_identity = record.identity
            if requested_identity is not None and requested_identity != (
                retained_identity
            ):
                raise AgentIdentityMismatchError(
                    f"Requested identity {requested_identity} does not match "
                    f"retained agent identity {retained_identity}"
                )
            self._view._require_task_identity_allowed(parent, record.identity)
            if record.reserved_for_replacement:
                return self.replacement_refusal(
                    record, LaunchOutcome.REJECTED_RESERVATION
                )
            if record.state is _AgentState.IDLE:
                return None
            if record.state is _AgentState.SUSPENDED:
                return self._reserve_suspended_locked(record, args, parent, identity)
            run = record.current_run
            if (
                record.state is _AgentState.FINALIZING
                or run is None
                or run.status is not RunStatus.RUNNING
                or run.initial_turn_id is None
                or record.runtime.turns.operation_pending_turn_id(run.initial_turn_id)
                is None
            ):
                return self.replacement_refusal(record, LaunchOutcome.ALREADY_FINISHING)
            if run.requested_stop_reason is not None:
                return self.replacement_refusal(record, LaunchOutcome.ALREADY_STOPPING)
            if args.config is not None:
                raise LaunchConfigError(
                    "Busy replacement cannot change launch configuration"
                )
            if args.model is not None:
                # The retained launch path resolves the canonical top-level
                # model; a busy replacement reuses the SAME instance, so a
                # model selection would change its committed deployment.
                raise LaunchConfigError(
                    "Busy replacement cannot change launch configuration"
                )
            if args.retain is not None and args.retain is not record.retained:
                raise LaunchConfigError(
                    "Busy replacement cannot change retention policy"
                )
            if (
                record.runtime._closed is True
                or self._view._children.get(record.session_id) is not record.runtime
                or not self._view._generation_is_current(record.root_generation)
                or record.work_slot not in self._view._active_work_slots
            ):
                raise RuntimeError("Agent is no longer eligible for replacement")
            assert record.work_slot is not None
            claim = _ReplacementClaim(
                record,
                run,
                parent,
                identity,
                record.runtime.agent_loop._session_generation,
                parent.agent_loop._authority_revision,
                record.work_slot,
            )
            run.requested_stop_reason = RunStopReason.RETASKED
            record.reserved_for_replacement = True
            record.replacement_owner = claim
            return claim

    def _reserve_suspended_locked(
        self,
        record: AgentRecord,
        args: TaskArgs,
        parent: SessionRuntime,
        identity: tuple[str, int],
    ) -> _ReplacementClaim:
        """Reserve a suspended record for replacement (handoff §1.8).

        WP2: replacement remains permitted at any point — including while
        a suspended barrier is pending or prohibited. The failed run is
        already terminal, so the handoff only joins it; a fresh capacity
        slot is allocated for the replacement launch (the suspended
        record's slot was released when its run finalized), and pending
        recovery is superseded.
        """
        if args.config is not None:
            raise LaunchConfigError(
                "Busy replacement cannot change launch configuration"
            )
        if args.model is not None:
            # The retained launch path resolves the canonical top-level
            # model; a busy replacement reuses the SAME instance, so a
            # model selection would change its committed deployment.
            raise LaunchConfigError(
                "Busy replacement cannot change launch configuration"
            )
        if args.retain is not None and args.retain is not record.retained:
            raise LaunchConfigError("Busy replacement cannot change retention policy")
        failed_run = record.run_history[-1] if record.run_history else None
        if (
            failed_run is None
            or failed_run.run_id != record.latest_run_id
            or record.runtime._closed is True
            or self._view._children.get(record.session_id) is not record.runtime
            or not self._view._generation_is_current(record.root_generation)
        ):
            raise RuntimeError("Agent is no longer eligible for replacement")
        slot = _ActiveWorkSlot()
        self._view._active_work_slots.add(slot)
        record.work_slot = slot
        claim = _ReplacementClaim(
            record,
            failed_run,
            parent,
            identity,
            record.runtime.agent_loop._session_generation,
            parent.agent_loop._authority_revision,
            slot,
        )
        self._view._invalidate_recovery_locked(record)
        record.reserved_for_replacement = True
        record.replacement_owner = claim
        return claim

    @asynccontextmanager
    async def handoff(self, claim: _ReplacementClaim) -> AsyncIterator[None]:
        """Interrupt the replaced run and transfer its capacity slot (async context).

        This is the WP2 SAME-AGENT replacement path (``replace_run``): one
        identity, a new run. The WP5 failure-time CROSS-MODEL replacement is
        a distinct path: the orchestrator claims a replacement through
        ``config.predecessor_agent_id`` on a fresh launch, and
        ``ChildLaunchAdmission.validate_replacement_claim`` disables the old
        recovery (``_invalidate_recovery_locked``) BEFORE the new run's write
        token is registered at acceptance — the write-ownership transfer
        happens only after reconciliation state is settled and the explicit
        handoff is admitted. Both paths preserve lineage, counters, and
        artifacts, and neither auto-restarts the predecessor.
        """
        try:
            self._view._interrupt_run(claim.record, claim.run)
            await asyncio.shield(claim.run.completion_task)
            claim.slot.transferred = False
            yield
        finally:
            # Retry lock acquisition on every cancellation: unwinding must finish
            # before caller cancellation propagates, even under repeated interrupts.
            cancelled = False
            while True:
                try:
                    async with self._view._registry_lock:
                        owner = claim.record.replacement_owner
                        if owner is claim and claim.record.reserved_for_replacement:
                            self.clear_locked(claim.record)
                        elif owner is claim:
                            claim.record.replacement_owner = None
                        if (
                            owner is None or owner is claim
                        ) and not claim.slot.transferred:
                            self._view._active_work_slots.discard(claim.slot)
                    break
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError

    def clear_locked(self, record: AgentRecord) -> None:
        """Settle the claim and schedule the completion-pending unwind (locked)."""
        record.reserved_for_replacement = False
        record.replacement_owner = None
        if record.current_run is None:
            if record.work_slot is not None:
                self._view._active_work_slots.discard(record.work_slot)
            if self._view._agent_records.get(record.agent_id) is record:
                latest = record.run_history[-1] if record.run_history else None
                if latest is not None and not latest.completion_task.done():
                    record.state = _AgentState.FINALIZING
                    record.idle_since = self._view._clock()
                    task = asyncio.create_task(
                        self.finish_unwind(record, latest),
                        name=f"vibe-subagent-replacement-unwind:{record.agent_id}",
                    )
                    self._view._teardown_tasks.add(task)
                    task.add_done_callback(self._view._teardown_tasks.discard)
                # Re-apply the committed policy: an unretained instance whose run
                # settled closes instead of returning to idle.
                elif self._view._close_unretained_locked(
                    record,
                    latest,
                    name=f"vibe-subagent-replacement-unwind:{record.agent_id}",
                ):
                    self._view._schedule_agents_update_locked()
                else:
                    record.state = _AgentState.IDLE
                    record.idle_since = self._view._clock()
        self._view._rearm_reaper_locked()

    async def finish_unwind(self, record: AgentRecord, run: RunRecord) -> None:
        """Apply the committed policy once the replaced run has settled."""
        # Cancellation may arrive after the monitor's idle/eviction decision,
        # while its final agents update is still awaiting delivery.
        with suppress(asyncio.CancelledError):
            await asyncio.shield(run.completion_task)
        async with self._view._registry_lock:
            resident = (
                self._view._agent_records.get(record.agent_id) is record
                and self._view._children.get(record.session_id) is record.runtime
                and self._view._generation_is_current(record.root_generation)
            )
            if (
                resident
                and record.latest_run_id == run.run_id
                and record.current_run is None
                and record.state is _AgentState.FINALIZING
                and not record.reserved_for_replacement
            ):
                # Re-apply the committed policy after the awaited run settles.
                if self._view._close_unretained_locked(
                    record,
                    run,
                    name=f"vibe-subagent-replacement-unwind:{record.agent_id}",
                ):
                    self._view._schedule_agents_update_locked()
                else:
                    record.state = _AgentState.IDLE
                    record.idle_since = self._view._clock()
                self._view._rearm_reaper_locked()

    async def rollback_reuse(
        self,
        record: AgentRecord,
        previous_idle_since: float | None,
        *,
        apply_policy: bool = False,
    ) -> None:
        """Restore idle residency after a failed reuse or replacement admission."""
        async with self._view._registry_lock:
            if (
                self._view._agent_records.get(record.agent_id) is not record
                or record.state is not _AgentState.RUNNING
                or record.reserved_for_replacement
                or record.runtime._closed is True
                or self._view._children.get(record.session_id) is not record.runtime
            ):
                return
            record.state = _AgentState.IDLE
            record.idle_since = (
                previous_idle_since
                if previous_idle_since is not None
                else self._view._clock()
            )
            # Re-apply the committed policy only for a busy-replacement unwind:
            # a failed admission that followed a stopped run closes an unretained
            # instance instead of leaving it resident. A plain reuse failure
            # restores the prior idle residency. No-op for retained records.
            if apply_policy:
                closed = self._view._close_unretained_locked(
                    record,
                    record.run_history[-1] if record.run_history else None,
                    name=f"vibe-subagent-replacement-unwind:{record.agent_id}",
                )
                if closed:
                    # Publish the terminal transition: the client inventory
                    # must reflect the closure.
                    self._view._schedule_agents_update_locked()
            self._view._rearm_reaper_locked()
