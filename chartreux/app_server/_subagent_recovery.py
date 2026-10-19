"""Explicit warm resume and effect reconciliation for subagents (WP2).

Implements the v0.5.6 recovery plan's WP2 package (handoff §1.2/§1.4/§1.8):
a recoverable failure SUSPENDS the agent — warm, resident, explicitly
resumable, an explicit bounded exception BEFORE closure — and the
explicitly invoked ``resume_agent`` admission starts exactly one successor
run on the SAME agent identity with a NEW run_id. There is NO
auto-scheduling in WP2: an admitted successor starts immediately, and the
later automatic controller (WP4) must call the same admission operation.

Safety invariants (code, never config escape hatches):

- Single-flight, run-pinned admission: concurrent same-key and
  different-key calls admit exactly one successor.
- The old failed result and every wait pinned to it stay immutable.
- Prior execution/tool tasks are joined before admission.
- Unknown tool outcomes block replay (``reconcile_required``) until an
  externally verified ``reconcile_result`` clears the barrier — pinned to
  the failed run and its transcript boundary, so stale or duplicate
  evidence cannot clear a newer barrier — or flips it to prohibited when
  verification shows unrepeatable damage. The barrier never auto-clears.
- Only the failed backend is retired; sibling agents own separate
  backend lifetimes and are never touched.
- The injected resume continuation NEVER sets ``user_initiated_retry``
  (F6: injected turns never clear shared availability cooldowns).
- Minimal warm bounds: the per-chain resume budget
  (``recovery.subagents.same_agent_resumes_max``), the suspended-count cap
  (``recovery.subagents.max_suspended_agents``), the suspension TTL
  (``recovery.subagents.suspension_ttl_seconds`` — WP3), and the separate
  suspension byte cap (``recovery.subagents.max_suspended_bytes`` — WP3).
  The ``[recovery.subagents]`` section moved under the ``[recovery]``
  container in WP4; suspension itself is warm preservation and stays
  deliberately NOT gated on ``recovery.subagents.enabled`` — the enable
  flag gates only the WP4 automatic controller.
- Warm means the conversation stays resident while the provider
  connection is released: the failed backend is retired at suspension
  through its ownership-aware lifetime (WP3, handoff §1.5).
- Closed/evicted agents remain non-reusable; replacement remains
  permitted at any point.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
import time
from typing import TYPE_CHECKING, Any, Protocol

from chartreux.app_server._subagent_execution import _outstanding_tool_call_ids
from chartreux.app_server._subagent_records import (
    AgentRecord,
    RunRecord,
    _ActiveWorkSlot,
    _AgentState,
    _AgentTombstone,
    _ResumeAdmission,
    _ResumeOperation,
)
from chartreux.app_server.models import TextContentBlock
from chartreux.app_server.protocol import TurnStartParams
from chartreux.core.config.models import RecoveryConfig
from chartreux.core.llm_models import LLMMessage
from chartreux.core.recovery_models import FailureEnvelope, RecoveryEligibility
from chartreux.core.subagents import (
    EffectVerification,
    ReconcileOutcome,
    ReconcileResult,
    ResumeAgentResult,
    ResumeOutcome,
    RunStatus,
    TaskArgs,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.observability.logging import logger
from chartreux.utils.retry_prompt import build_resume_prompt

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime
    from chartreux.app_server._subagent_execution import SubagentExecution


class RecoveryConfigView(Protocol):
    """The minimal view the policy accessor reads: just the bound root.

    WP4's automatic recovery controller passes its own narrower view; the
    accessor only ever reads ``_root``, so this structural protocol keeps
    both the WP2 view and the WP4 controller view acceptable.
    """

    _root: SessionRuntime | None


class RecoveryStateView(Protocol):
    """Narrow typed view over the registry-owned recovery state.

    Dynamic access: every member is read through the view at call time, so
    the record/child/tombstone maps, the slot set, the latest-run index,
    the shared registry lock, the clock, the root, the admission/drain
    scalars, and the execution collaborator are never snapshotted at
    construction — registry state mutated after the collaborator is built
    stays visible.
    """

    _agent_records: dict[str, AgentRecord]
    _children: dict[str, SessionRuntime]
    _evicted_agents: dict[str, _AgentTombstone]
    _active_work_slots: set[_ActiveWorkSlot]
    _latest_run_ids: dict[str, str]
    _registry_lock: asyncio.Lock
    _clock: Callable[[], float]
    _root: SessionRuntime | None
    _admission_closed: bool
    _draining_children: bool
    _max_running_subagents: int | None
    _execution: SubagentExecution

    def _generation_is_current(self, generation: int) -> bool: ...

    def _new_run_id(self, agent_id: str) -> str: ...

    def _validate_background_admission(
        self, parent: SessionRuntime, identity: tuple[str, int]
    ) -> None: ...

    def _live_tombstone_locked(
        self, agent_id: str, now: float
    ) -> _AgentTombstone | None: ...

    def _suspension_admit_locked(
        self, record: AgentRecord, config: RecoveryConfig
    ) -> bool: ...

    def _suspension_capture_locked(self, record: AgentRecord) -> None: ...

    def _emit_agents_update(self) -> Coroutine[Any, Any, None]: ...


def _effective_eligibility(
    record: AgentRecord, envelope: FailureEnvelope
) -> tuple[RecoveryEligibility, list[str]]:
    """The barrier's current effective eligibility and reason codes.

    The published envelope is immutable; a reconcile_result submission
    flips the EFFECTIVE view tracked on the record (cleared -> eligible,
    verified unrepeatable damage -> prohibited).
    """
    if record.reconcile_outcome == ReconcileOutcome.CLEARED.value:
        return RecoveryEligibility.ELIGIBLE, [
            "reconcile_cleared_effects_settled",
            *record.reconcile_reason_codes,
        ]
    if record.reconcile_outcome == ReconcileOutcome.PROHIBITED.value:
        return RecoveryEligibility.PROHIBITED, [
            "reconcile_verified_unrepeatable_damage",
            *record.reconcile_reason_codes,
        ]
    return envelope.recovery.eligibility, list(envelope.recovery.reason_codes)


def _recovery_config(view: RecoveryConfigView) -> RecoveryConfig:
    """The child controller's policy: the ``[recovery.subagents]`` section (WP4).

    WP4 moved the child recovery policy into the ``[recovery.subagents]``
    subsection of the ``[recovery]`` container (round-3 symmetry amendment);
    this accessor keeps returning the POLICY object so every WP1–WP3
    consumer (suspension caps, resume budget, reaper TTL) is unchanged.
    """
    root = view._root
    if root is None:
        return RecoveryConfig()
    return root.agent_loop.config.recovery.subagents


def _submission_defect(
    messages: Sequence[LLMMessage], effects: Sequence[EffectVerification]
) -> tuple[list[str], str] | None:
    """The STALE accounting defect of a clearing submission (review R5).

    A clearing submission must account for exactly the tool calls the
    failed run left outstanding — an unrelated id, an incomplete subset, a
    repeated id, or (with calls outstanding) an empty submission verifies
    nothing about the unknown effects and never clears the barrier. A
    verified damage report always prohibits: the prohibit direction is
    fail-safe and is never refused over a technicality, so the defect
    check is skipped when damage is present. Returns (reason_codes,
    reason) for a STALE submission, else None.
    """
    if any(effect.disposition == "unrepeatable_damage" for effect in effects):
        return None
    outstanding = _outstanding_tool_call_ids(messages)
    seen: set[str] = set()
    repeated: set[str] = set()
    for effect in effects:
        if effect.tool_call_id in seen:
            repeated.add(effect.tool_call_id)
        seen.add(effect.tool_call_id)
    if repeated:
        return (
            ["duplicate_tool_call_id", *sorted(repeated)],
            "The submission repeats a tool call id",
        )
    unknown = sorted(seen - outstanding)
    missing = sorted(outstanding - seen)
    if unknown or missing:
        mismatch_codes: list[str] = []
        if missing:
            mismatch_codes.append("unverified_outstanding_tool_calls")
            mismatch_codes.extend(missing)
        if unknown:
            mismatch_codes.append("unknown_tool_call_ids")
            mismatch_codes.extend(unknown)
        return (
            mismatch_codes,
            "The submission does not account for every "
            "outstanding tool call of the failed run",
        )
    return None


class SubagentRecovery:
    """Suspension, explicit resume admission, and reconciliation (WP2).

    Narrow collaborator of the session runtime registry: the
    suspension decision at the monitor's completion transition, the
    single-flight run-pinned resume admission (locked check + joined
    successor execution), the run/boundary-pinned reconcile_result
    barrier, and the recovery invalidation on release/replacement. The
    registry-owned record/child maps, slot set, latest-run index, lock,
    and clock stay canonical; the registry wires the collaborator in its
    constructor and keeps thin same-name delegates.
    """

    def __init__(self, view: RecoveryStateView) -> None:
        self._view = view

    # ------------------------------------------------------------------
    # Suspension hook (monitor completion transition; caller holds the lock)
    # ------------------------------------------------------------------

    def suspend_after_failure_locked(  # noqa: PLR0911 - guard-clause decision matrix
        self, record: AgentRecord, run: RunRecord
    ) -> bool:
        """Move a recoverably failed record to SUSPENDED instead of closing.

        The decision reads the run's published typed failure envelope:
        only ELIGIBLE or RECONCILE_REQUIRED eligibility suspends, bounded
        by the per-chain resume budget, the suspended-count cap, the
        suspension TTL (WP3: a zero TTL disables warm suspension entirely),
        and the separate suspension byte cap (WP3). A new barrier starts
        from its own envelope — the previous barrier's reconciliation
        flip never carries into a newer barrier.
        """
        view = self._view
        if run.status is not RunStatus.FAILED:
            return False
        envelope = run.failure
        if envelope is None:
            return False
        if envelope.recovery.eligibility not in {
            RecoveryEligibility.ELIGIBLE,
            RecoveryEligibility.RECONCILE_REQUIRED,
        }:
            return False
        if record.state is not _AgentState.FINALIZING:
            return False
        if record.reserved_for_replacement or view._draining_children:
            return False
        if not view._generation_is_current(record.root_generation):
            return False
        if view._children.get(record.session_id) is not record.runtime:
            return False
        if record.runtime._closed is True:
            return False
        if record.latest_run_id != run.run_id:
            return False
        config = _recovery_config(view)
        if record.same_agent_resumes_used >= config.same_agent_resumes_max:
            return False
        suspended_count = sum(
            1
            for other in view._agent_records.values()
            if other.state is _AgentState.SUSPENDED
        )
        if suspended_count >= config.max_suspended_agents:
            return False
        # WP3: the warm window is configured — a zero TTL means no warm
        # suspension at all, and the deterministic byte accounting must fit
        # the separate suspension byte cap (the oldest suspended records
        # expire first to make room; an oversized record never suspends).
        if config.suspension_ttl_seconds <= 0:
            return False
        if not view._suspension_admit_locked(record, config):
            return False
        record.state = _AgentState.SUSPENDED
        record.suspended_at = view._clock()
        if record.recovery_chain_id is None:
            record.recovery_chain_id = envelope.recovery_chain_id
        record.reconcile_outcome = None
        record.reconcile_reason_codes = []
        view._suspension_capture_locked(record)
        # WP3 (handoff §1.5): warm suspension keeps the live conversation
        # resident but RELEASES the provider connection — the failed
        # backend is retired through its ownership-aware lifetime and a
        # fresh one is installed. Sibling agents own separate backend
        # lifetimes and are never touched; the resume admission refreshes
        # again exactly as in WP2. Best-effort: a turn somehow still active
        # must not break the completion transition (the close paths still
        # bound the connection).
        try:
            record.runtime.agent_loop.refresh_backend_for_resume()
        except Exception as exc:
            logger.warning(
                "Failed to release the provider connection at suspension "
                "of agent_id=%s",
                record.agent_id,
                exc_info=exc,
            )
        return True

    # ------------------------------------------------------------------
    # Recovery invalidation (release/replacement; caller holds the lock)
    # ------------------------------------------------------------------

    def invalidate_locked(self, record: AgentRecord | None) -> None:
        """Invalidate pending recovery for a detached or superseded record.

        Release pops the record entirely; replacement supersedes resume.
        An in-flight admission still fails its residency revalidation and
        rolls back through the normal ownership paths.
        """
        if record is not None:
            record.pending_resume = None
            record.recovery_superseded = True

    # ------------------------------------------------------------------
    # resume_agent (explicit admission; one operation, no auto-scheduling)
    # ------------------------------------------------------------------

    async def resume_agent(  # noqa: PLR0911, PLR0912, PLR0914, PLR0915 - disposition matrix
        self,
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
    ) -> ResumeAgentResult:
        """Admit at most one successor for a suspended failed run (WP2)."""
        view = self._view
        admission: _ResumeAdmission | None = None
        slot: _ActiveWorkSlot | None = None
        eligibility: RecoveryEligibility | None = None
        reason_codes: list[str] = []
        raise_after_lock: RuntimeError | None = None
        async with view._registry_lock:
            if view._admission_closed or view._draining_children:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.FORBIDDEN,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    reason="Agent admission is closed",
                    reason_codes=["admission_closed"],
                )
            record = view._agent_records.get(agent_id)
            if record is None:
                tombstone = view._live_tombstone_locked(agent_id, view._clock())
                reason = (
                    f"Agent {agent_id} is {tombstone.summary.availability.value} "
                    "terminal metadata and cannot be resumed"
                    if tombstone is not None
                    else f"Unknown agent {agent_id}"
                )
                return ResumeAgentResult(
                    outcome=ResumeOutcome.NOT_RESUMABLE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    reason=reason,
                    reason_codes=["not_suspended"],
                )
            if record.parent_session_id != requester_session_id:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.FORBIDDEN,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    reason="Only the owning parent may resume this agent",
                    reason_codes=["forbidden_requester"],
                )
            # Single-flight: a pending admission deduplicates same-key calls
            # and refuses different-key calls — exactly one successor.
            pending = record.pending_resume
            if pending is not None:
                if (
                    pending.idempotency_key == idempotency_key
                    and pending.expected_run_id == expected_run_id
                ):
                    return ResumeAgentResult(
                        outcome=ResumeOutcome.ALREADY_SCHEDULED,
                        agent_id=agent_id,
                        run_id=expected_run_id,
                        successor_run_id=pending.successor_run_id,
                        recovery_operation_id=pending.operation_id,
                        retry_at=pending.admitted_at,
                        same_agent_resumes_used=record.same_agent_resumes_used,
                        reason="A resume with this idempotency key is already scheduled",
                    )
                return ResumeAgentResult(
                    outcome=ResumeOutcome.ALREADY_RESUMED,
                    agent_id=agent_id,
                    run_id=pending.expected_run_id,
                    successor_run_id=pending.successor_run_id,
                    recovery_operation_id=pending.operation_id,
                    same_agent_resumes_used=record.same_agent_resumes_used,
                    reason=("A resume admission for this agent is already in flight"),
                )
            # Run-pinned already-resumed detection.
            successor = next(
                (
                    item
                    for item in record.run_history
                    if item.resumed_from_run_id == expected_run_id
                ),
                None,
            )
            if successor is not None:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.ALREADY_RESUMED,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    successor_run_id=successor.run_id,
                    same_agent_resumes_used=record.same_agent_resumes_used,
                    reason="The pinned failed run already has an admitted successor",
                )
            if record.recovery_superseded or record.reserved_for_replacement:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.NOT_RESUMABLE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    reason="Recovery for this agent was superseded by replacement",
                    reason_codes=["recovery_superseded"],
                )
            if record.state is not _AgentState.SUSPENDED:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.NOT_RESUMABLE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    reason=(f"Agent is {record.availability.value}, not suspended"),
                    reason_codes=["not_suspended"],
                )
            failed_run = next(
                (
                    item
                    for item in [record.current_run, *record.run_history]
                    if item is not None and item.run_id == expected_run_id
                ),
                None,
            )
            if failed_run is None or record.latest_run_id != expected_run_id:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.NOT_RESUMABLE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    reason=("expected_run_id does not match the suspended failed run"),
                    reason_codes=["stale_expected_run_id"],
                )
            envelope = failed_run.failure
            if envelope is None:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.NOT_RESUMABLE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    reason="The suspended run has no failure envelope",
                    reason_codes=["no_failure_envelope"],
                )
            eligibility, reason_codes = _effective_eligibility(record, envelope)
            if eligibility is RecoveryEligibility.RECONCILE_REQUIRED:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.RECONCILE_REQUIRED,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    eligibility=eligibility,
                    reason_codes=reason_codes,
                    reason=(
                        "Unknown tool outcomes block replay; verify them "
                        "externally and submit reconcile_result first"
                    ),
                )
            if eligibility is RecoveryEligibility.PROHIBITED:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.NOT_RESUMABLE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    eligibility=eligibility,
                    reason_codes=reason_codes,
                    reason=(
                        "Recovery eligibility is prohibited; replacement "
                        "remains permitted"
                    ),
                )
            # Expiry is an admission stop, never an in-flight kill (Q2): the
            # envelope's epoch fields compare against the wall clock.
            now_epoch = time.time()
            if (
                envelope.recovery.expires_at is not None
                and now_epoch >= envelope.recovery.expires_at
            ):
                return ResumeAgentResult(
                    outcome=ResumeOutcome.EXPIRED,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    eligibility=eligibility,
                    reason="The recovery window expired",
                    reason_codes=["recovery_window_expired"],
                )
            config = _recovery_config(view)
            if now_epoch >= envelope.observed_at + config.episode_deadline_seconds:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.EXPIRED,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    eligibility=eligibility,
                    reason=(
                        "The advisory episode deadline passed; no new "
                        "admission after the deadline"
                    ),
                    reason_codes=["episode_deadline_passed"],
                )
            if record.same_agent_resumes_used >= config.same_agent_resumes_max:
                return ResumeAgentResult(
                    outcome=ResumeOutcome.NOT_RESUMABLE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    eligibility=eligibility,
                    reason="The same-agent resume budget is exhausted",
                    reason_codes=["resume_budget_exhausted"],
                )
            if (
                view._children.get(record.session_id) is not record.runtime
                or record.runtime._closed is True
            ):
                return ResumeAgentResult(
                    outcome=ResumeOutcome.NOT_RESUMABLE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    eligibility=eligibility,
                    reason="The suspended runtime is no longer resident",
                    reason_codes=["runtime_unavailable"],
                )
            # Capacity: the successor consumes one active-work slot, exactly
            # like a fresh launch (the failed run's slot was released when
            # its monitor finalized).
            cap = view._max_running_subagents
            if cap is None:
                root = view._root
                cap = (
                    root.agent_loop.config.subagents.max_running_subagents
                    if root is not None
                    else 16
                )
            # Review R8: ``recovery_concurrency`` bounds CONCURRENT recovery
            # admissions. The count includes RESTORING records — an admitted
            # resume whose successor has not installed its run record yet —
            # so concurrent resumes can never all pass the same check.
            active_recoveries = sum(
                1
                for item in view._agent_records.values()
                if item.state is _AgentState.RESTORING
                or (
                    item.current_run is not None
                    and item.current_run.resumed_from_run_id is not None
                )
            )
            if active_recoveries >= config.recovery_concurrency:
                raise_after_lock = RuntimeError(
                    f"Subagent resume rejected: {active_recoveries} recovery "
                    f"runs already active (cap {config.recovery_concurrency}); "
                    "wait for completion or release agents before resuming"
                )
            elif len(view._active_work_slots) >= cap:
                raise_after_lock = RuntimeError(
                    f"Subagent resume rejected: {len(view._active_work_slots)} "
                    f"agents already running (cap {cap}); wait for completion "
                    "or release agents before resuming"
                )
            else:
                root = view._root
                assert root is not None
                successor_run_id = view._new_run_id(agent_id)
                operation = _ResumeOperation(
                    operation_id=f"resume-op:{agent_id}:{successor_run_id}",
                    agent_id=agent_id,
                    expected_run_id=expected_run_id,
                    idempotency_key=idempotency_key,
                    successor_run_id=successor_run_id,
                    admitted_at=now_epoch,
                )
                record.pending_resume = operation
                record.state = _AgentState.RESTORING
                record.same_agent_resumes_used += 1
                # Review R8: the successor's slot is RESERVED here, under the
                # lock — the cap checks above and the reservation are one
                # atomic step, so concurrent resumes can never all pass the
                # same check (the launch path reserves the same way).
                slot = _ActiveWorkSlot()
                view._active_work_slots.add(slot)
                admission = _ResumeAdmission(
                    record=record,
                    failed_run=failed_run,
                    operation=operation,
                    parent=root,
                    parent_identity=(
                        root.agent_loop.session_id,
                        root.agent_loop._session_generation,
                    ),
                    successor_run_id=successor_run_id,
                    resumes_used_before=record.same_agent_resumes_used - 1,
                )
        if raise_after_lock is not None:
            raise raise_after_lock
        assert admission is not None
        assert slot is not None
        try:
            await self._admit_successor(admission, slot)
        except BaseException:
            async with view._registry_lock:
                record = admission.record
                if (
                    view._agent_records.get(agent_id) is record
                    and record.pending_resume is admission.operation
                ):
                    record.pending_resume = None
                    record.state = _AgentState.SUSPENDED
                    record.same_agent_resumes_used = admission.resumes_used_before
                if not slot.transferred:
                    view._active_work_slots.discard(slot)
            raise
        assert eligibility is not None
        return ResumeAgentResult(
            outcome=ResumeOutcome.SCHEDULED,
            agent_id=agent_id,
            run_id=expected_run_id,
            successor_run_id=admission.successor_run_id,
            recovery_operation_id=admission.operation.operation_id,
            retry_at=admission.operation.admitted_at,
            eligibility=eligibility,
            reason_codes=reason_codes,
            same_agent_resumes_used=admission.record.same_agent_resumes_used,
            reason="Resume admitted; the successor run started immediately",
        )

    async def _admit_successor(
        self, admission: _ResumeAdmission, slot: _ActiveWorkSlot
    ) -> None:
        """Drive one admitted resume to a running successor (no scheduling).

        Restoration contract (handoff §1.4): prior execution/tool tasks
        joined; the failed backend retired through its ownership-aware
        lifetime (siblings untouched); ONE resume marker + injected
        continuation (never user-initiated); the successor run id
        allocated at admission is published atomically with the run.
        ``slot`` is the capacity slot RESERVED at admission, under the
        registry lock (review R8) — the cap checks and the reservation are
        one atomic step.
        """
        view = self._view
        record = admission.record
        failed_run = admission.failed_run
        # 1. Join prior execution and tool tasks before admission.
        joinables: list[Any] = [failed_run.completion_task]
        active_task = record.runtime.turns._active_task
        if active_task is not None:
            joinables.append(active_task)
        if joinables:
            await asyncio.gather(*joinables, return_exceptions=True)
        # 2. Retire only the failed backend; install a fresh one. Sibling
        #    agents own separate backend lifetimes and are never touched.
        record.runtime.agent_loop.refresh_backend_for_resume()
        # 3. The capacity slot was reserved at admission (under the lock).
        try:
            args = TaskArgs(
                task=(record.initial_task_summary or record.last_task_summary or ""),
                agent_id=record.agent_id,
                background=True,
            )
            ctx = InvokeContext(
                tool_call_id=f"resume:{admission.operation.operation_id}",
                session_id=admission.parent.agent_loop.session_id,
            )
            context = await view._execution.prepare_background(
                admission.parent,
                admission.parent_identity,
                args,
                ctx,
                slot,
                replacement=None,
                summary_task=None,
                resume=admission,
            )
            try:
                context.runtime.turns._event_sink = context.consume_event
                # 4. ONE resume marker + injected continuation, mirroring
                #    /retry. Injected and NEVER user_initiated_retry (F6).
                response, action = context.runtime.turns.start(
                    TurnStartParams(
                        session_id=context.runtime.agent_loop.session_id,
                        message=[
                            TextContentBlock(
                                text=build_resume_prompt(failed_run.run_id)
                            )
                        ],
                        injected=True,
                    )
                )
                context.turn_id = response.turn.id
                context.action = action
                context.run_record.initial_turn_id = response.turn.id
                context.run_record.resumed_from_run_id = failed_run.run_id
                context.run_record.recovery_chain_id = record.recovery_chain_id
                await view._execution.accept_background(
                    context, admission.parent, admission.parent_identity, slot
                )
            except BaseException:
                await view._execution.rollback_background(
                    context, admission.parent, replacement=None
                )
                raise
        except BaseException:
            view._active_work_slots.discard(slot)
            raise
        # 5. Publish the admitted successor.
        async with view._registry_lock:
            if view._agent_records.get(record.agent_id) is record:
                record.pending_resume = None
        try:
            await view._emit_agents_update()
        except Exception as exc:
            logger.warning("Failed to publish resume admission update", exc_info=exc)

    # ------------------------------------------------------------------
    # reconcile_result (run + transcript-boundary pinned barrier)
    # ------------------------------------------------------------------

    async def reconcile_result(  # noqa: PLR0911
        self,
        agent_id: str,
        expected_run_id: str,
        transcript_boundary: str,
        verified_effects: Sequence[EffectVerification],
        *,
        requester_session_id: str,
    ) -> ReconcileResult:
        """Submit externally verified tool effects for a reconcile barrier."""
        view = self._view
        effects = list(verified_effects)
        async with view._registry_lock:
            if view._admission_closed or view._draining_children:
                return ReconcileResult(
                    outcome=ReconcileOutcome.FORBIDDEN,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    reason="Agent admission is closed",
                )
            record = view._agent_records.get(agent_id)
            if record is None:
                return ReconcileResult(
                    outcome=ReconcileOutcome.NOT_PENDING,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    reason="Unknown agent; no reconcile barrier is pending",
                )
            if record.parent_session_id != requester_session_id:
                return ReconcileResult(
                    outcome=ReconcileOutcome.FORBIDDEN,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    reason="Only the owning parent may reconcile this agent",
                )
            if record.state is not _AgentState.SUSPENDED:
                return ReconcileResult(
                    outcome=ReconcileOutcome.NOT_PENDING,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    reason=(
                        f"Agent is {record.availability.value}; no reconcile "
                        "barrier is pending"
                    ),
                )
            barrier_run = next(
                (
                    item
                    for item in [record.current_run, *record.run_history]
                    if item is not None and item.run_id == expected_run_id
                ),
                None,
            )
            if barrier_run is None or record.latest_run_id != expected_run_id:
                return ReconcileResult(
                    outcome=ReconcileOutcome.STALE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    reason=(
                        "The submission is pinned to a run that is not the "
                        "current barrier; a newer barrier exists"
                    ),
                    reason_codes=["stale_expected_run_id"],
                )
            envelope = barrier_run.failure
            if (
                envelope is None
                or envelope.recovery.eligibility
                is not RecoveryEligibility.RECONCILE_REQUIRED
            ):
                return ReconcileResult(
                    outcome=ReconcileOutcome.NOT_PENDING,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    reason="The barrier is not reconcile_required",
                )
            boundary = envelope.progress.transcript_boundary
            if boundary is None or transcript_boundary != boundary:
                return ReconcileResult(
                    outcome=ReconcileOutcome.STALE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    reason=(
                        "The transcript boundary does not match the barrier's boundary"
                    ),
                    reason_codes=["stale_transcript_boundary"],
                )
            if record.last_reconciled == (expected_run_id, transcript_boundary):
                eligibility, reason_codes = _effective_eligibility(record, envelope)
                return ReconcileResult(
                    outcome=ReconcileOutcome.DUPLICATE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    eligibility=eligibility,
                    reason_codes=reason_codes,
                    reason="This barrier was already reconciled",
                )
            # Review R5: a clearing submission must account for exactly
            # the tool calls the failed run left outstanding — an
            # unrelated id, an incomplete subset, a repeated id, or (with
            # calls outstanding) an empty submission verifies nothing
            # about the unknown effects and never clears the barrier. A
            # verified damage report always prohibits: the prohibit
            # direction is fail-safe and is never refused over a
            # technicality. Against a barrier with no transcript-visible
            # calls the empty submission is the complete zero-entry
            # accounting ("one entry per unknown tool call") and clears.
            defect = _submission_defect(record.runtime.agent_loop.messages, effects)
            if defect is not None:
                mismatch_codes, mismatch_reason = defect
                return ReconcileResult(
                    outcome=ReconcileOutcome.STALE,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    reason=mismatch_reason,
                    reason_codes=mismatch_codes,
                )
            damage = [
                effect
                for effect in effects
                if effect.disposition == "unrepeatable_damage"
            ]
            record.last_reconciled = (expected_run_id, transcript_boundary)
            if damage:
                record.reconcile_outcome = ReconcileOutcome.PROHIBITED.value
                record.reconcile_reason_codes = [
                    "verified_unrepeatable_damage",
                    *(effect.tool_call_id for effect in damage),
                ]
                result = ReconcileResult(
                    outcome=ReconcileOutcome.PROHIBITED,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    eligibility=RecoveryEligibility.PROHIBITED,
                    reason_codes=list(record.reconcile_reason_codes),
                    reason=(
                        "Verification showed unrepeatable damage; recovery "
                        "is prohibited. Replacement remains permitted."
                    ),
                )
            else:
                record.reconcile_outcome = ReconcileOutcome.CLEARED.value
                record.reconcile_reason_codes = [
                    "effects_verified_settled",
                    *(effect.tool_call_id for effect in effects),
                ]
                result = ReconcileResult(
                    outcome=ReconcileOutcome.CLEARED,
                    agent_id=agent_id,
                    run_id=expected_run_id,
                    transcript_boundary=transcript_boundary,
                    eligibility=RecoveryEligibility.ELIGIBLE,
                    reason_codes=list(record.reconcile_reason_codes),
                    reason=(
                        "All verified effects settled; the barrier is "
                        "cleared and the agent may be resumed."
                    ),
                )
        try:
            await view._emit_agents_update()
        except Exception as exc:
            logger.warning("Failed to publish reconciliation update", exc_info=exc)
        return result
