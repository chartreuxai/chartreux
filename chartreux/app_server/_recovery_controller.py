"""Bounded automatic recovery controller for child subagents (WP4).

Implements the v0.5.6 recovery plan's WP4 package (handoff §2.1/§2.3): the
harness's AUTOMATIC half for a recoverably failed child — bounded same-agent
resume with cancellable backoff, driven INDEPENDENTLY of parent turns.

Design invariants (code, never config escape hatches):

- ONE admission operation: automatic and explicit requests both go through
  the WP2 ``resume_agent`` admission (single-flight, run-pinned) — the
  controller never builds a second scheduler.
- The controller is INERT unless ``[recovery.subagents] enabled`` is true.
  Suspension (WP2/WP3) and the explicit resume/reconcile ports are untouched
  either way, and automatic recovery IS ENABLED by default since v0.5.6
  (promoted at WP6 per OD5 once WP4's acceptance matrix passed; ``enabled``
  defaults to True) — set it to False for manual-resume-only recovery.
- Eligibility: the failed run's typed envelope must be ELIGIBLE (protected
  classes — cancellation, budget stop, authorization, preflight, task
  blocker, permanent config, unknown/internal — are never auto-resumed) AND
  its class must be in the configured ``eligible_classes``.
- Budgets: ONE episode budget per recovery chain (``auto_retries_max``
  automatic admissions, anchored at the episode's first failure), PLUS a
  package-lineage recurrence counter that SPANS replacements (a replacement
  agent re-dispatched with the same brief cannot reset it). Episode policy is
  SNAPSHOTTED at episode creation — a config reload never extends an active
  episode or replenishes its count. Spend/turn/permission budgets are never
  touched (the WP2 resume continuation preserves them).
- launch_failure policy rows (handoff §2.3): a confirmed transient
  launch_failure gets ONE bounded-backoff startup retry preserving identity
  and brief; a REPEATED launch failure STOPS the relaunch loop (the
  instant-relaunch pattern L1/L2, R2/R3 is the production behavior this
  prevents).
- Cooldown/jitter: the first/second stall use the configured
  ``cooldowns_seconds`` entries with ``jitter_fraction``; a longer Retry-After
  and any deployment/endpoint cooldown postpone admission. Waiting for
  availability NEVER consumes budget (read-only re-checks).
- Endpoint-correlated throttling blocks recovery onto ALL affected-endpoint
  deployments regardless of model/provider naming: the pre-admission
  eligibility view is READ-ONLY (``claim_probes=False``) — single-flight
  probes stay token-owned by the callers that actually attempt, and cheap
  in-call retries remain telemetry only.
- Q2-negative branch: the aggregate ``episode_deadline_seconds`` is an
  ADVISORY scheduling/admission bound — "no new admission after deadline" —
  never a hard in-flight kill (no hard timeout exists; the retry budget is
  consulted only between attempts).
- Events: failure / scheduled / successor-start / success / exhaustion
  milestone lines with deterministic ids, each emitted EXACTLY ONCE per
  episode (deduplicated by the episode's emitted-event set).
- Invalidation: cancellation, release, replacement, generation change,
  session switch, and shutdown cancel pending schedules (rollback: cancel
  timers, invalidate tokens, retain counters/results — never clear cooldowns
  globally to look healthy).

The controller runs on the registry's own clock/wakeup: its timers are
asyncio tasks on the shared event loop, so a scheduled resume fires on a fake
overnight idle with NO parent model call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import random
import time
from typing import TYPE_CHECKING, Any, Protocol

from chartreux.app_server._subagent_records import AgentRecord, RunRecord, _AgentState
from chartreux.app_server._subagent_recovery import (
    _effective_eligibility,
    _recovery_config,
)
from chartreux.core.compaction import reorder_for_tool_adjacency
from chartreux.core.compaction.context import select_model_context
from chartreux.core.config.models import RecoveryConfig, RecoveryEligibleClass
from chartreux.core.llm.failures import FailureCategory
from chartreux.core.model_catalog.availability import (
    AllDeploymentsUnavailableError,
    ExclusionReason,
    eligible_deployments,
)
from chartreux.core.recovery_models import (
    FailureClass,
    FailureEnvelope,
    RecoveryEligibility,
    canonical_endpoint_identity,
    measured_timeout_reason_of,
)
from chartreux.core.subagents import ResumeAgentResult, ResumeOutcome, RunStatus
from chartreux.core.utils.retry import jittered_backoff
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime

#: Milestone events (handoff §2.1), emitted exactly once each per episode.
EVENT_FAILURE = "failure"
EVENT_SCHEDULED = "scheduled"
EVENT_SUCCESSOR_START = "successor_start"
EVENT_SUCCESS = "success"
EVENT_EXHAUSTION = "exhaustion"

#: handoff §2.3: launch_failure #1 gets ONE startup retry; a repeated launch
#: failure stops the relaunch loop. A code invariant, not a config knob.
_LAUNCH_FAILURE_RETRIES_MAX = 1

#: Exclusion reasons that are TEMPORARY (time-based) — recovery waits for
#: them instead of stopping. Everything else is permanent incompatibility.
_TEMPORARY_EXCLUSIONS = frozenset({
    ExclusionReason.COOLDOWN,
    ExclusionReason.ENDPOINT_COOLDOWN,
})

#: The envelope's cause category → the recovery-layer eligible class.
_CAUSE_TO_ELIGIBLE_CLASS: dict[FailureCategory, RecoveryEligibleClass] = {
    FailureCategory.CONNECTION: RecoveryEligibleClass.CONNECTION,
    FailureCategory.TIMEOUT: RecoveryEligibleClass.TIMEOUT,
    FailureCategory.SERVER_ERROR: RecoveryEligibleClass.SERVER_ERROR,
    FailureCategory.OVERLOAD: RecoveryEligibleClass.OVERLOAD,
    FailureCategory.RATE_LIMIT: RecoveryEligibleClass.RATE_LIMIT,
    FailureCategory.INCOMPLETE_STREAM: RecoveryEligibleClass.INCOMPLETE_STREAM,
    FailureCategory.EMPTY_RESPONSE: RecoveryEligibleClass.EMPTY_RESPONSE,
}


@dataclass(slots=True)
class _Episode:
    """One automatic-recovery episode: a recovery chain's bounded budget.

    ``started_at`` anchors the ADVISORY aggregate deadline (epoch seconds of
    the episode's first observed failure). The policy fields are snapshotted
    at creation: a config reload never extends an active episode or
    replenishes its count. ``pending_*`` snapshot the failed run's immutable
    envelope fields for the pending schedule.
    """

    agent_id: str
    chain_id: str
    started_at: float
    auto_retries_max: int
    cooldowns: list[float]
    jitter_fraction: float
    episode_deadline_seconds: float
    auto_retries_used: int = 0
    launch_retries_used: int = 0
    scheduled_failed_run_id: str | None = None
    pending_retry_after: float = 0.0
    pending_launch_failure: bool = False
    pending_lineage_key: str = ""
    timer: asyncio.Task[None] | None = None
    emitted: set[str] = field(default_factory=set)
    active: bool = True

    @property
    def driven(self) -> bool:
        """Whether the controller ever armed a schedule or admitted a retry."""
        return (
            self.auto_retries_used > 0
            or self.launch_retries_used > 0
            or self.scheduled_failed_run_id is not None
        )


@dataclass(frozen=True, slots=True)
class _Availability:
    """The read-only pre-admission eligibility snapshot (WP4).

    ``available``: at least one deployment of the committed base is eligible
    right now. ``permanent``: every deployment is excluded for a PERMANENT
    reason (disabled/allowlist/images/thinking/reasoning/compaction) —
    recovery must not be scheduled at all. ``next_check_delay``: seconds
    until the earliest temporary exclusion (deployment cooldown, endpoint
    cooldown, or probe occupancy) may change. ``endpoint_blocked``: every
    deployment is excluded by endpoint-wide correlation.
    """

    available: bool
    permanent: bool
    next_check_delay: float
    endpoint_blocked: bool


class RecoveryControllerStateView(Protocol):
    """Narrow typed view over the registry-owned recovery state.

    Dynamic access: every member is read through the view at call time, so
    the record map, the lock, the clock/wakeup, the root, the admission/drain
    scalars, and the resume admission are never snapshotted at construction.
    """

    _agent_records: dict[str, AgentRecord]
    _registry_lock: asyncio.Lock
    _clock: Callable[[], float]
    _wakeup: Callable[[float], Awaitable[None]]
    _root: SessionRuntime | None
    _draining_children: bool
    _admission_closed: bool

    def _generation_is_current(self, generation: int) -> bool: ...

    def resume_agent(
        self,
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
    ) -> Awaitable[ResumeAgentResult]: ...


def _lineage_key(record: AgentRecord, envelope: FailureEnvelope) -> str:
    """The package-lineage identity the recurrence counter spans (WP4).

    The authoritative key is the envelope's ``work_package_id`` (WP5
    populates it through the contract-surface lineage fields). Until then
    the launch identity — identity + initial task summary — is the package
    identity a replacement is re-dispatched with, so the counter spans the
    replacement. Without either, recurrence degrades honestly to the single
    agent (documented; never a silent reset).
    """
    if envelope.work_package_id:
        return f"pkg:{envelope.work_package_id}"
    summary = record.initial_task_summary or record.last_task_summary
    if summary:
        return f"brief:{record.identity}:{summary}"
    return f"agent:{record.agent_id}"


def _eligible_class_of(envelope: FailureEnvelope) -> RecoveryEligibleClass | None:
    """Map the envelope to its recovery-layer eligible class (or None)."""
    failure = envelope.failure
    if failure.failure_class is FailureClass.LAUNCH_FAILURE:
        return RecoveryEligibleClass.LAUNCH_FAILURE
    category = failure.cause_category
    if category is None:
        return None
    return _CAUSE_TO_ELIGIBLE_CLASS.get(category)


class RecoveryController:
    """The bounded automatic recovery controller for child subagents (WP4).

    Narrow collaborator of the session runtime registry: it observes every
    settled run at the monitor's completion transition (the caller holds the
    registry lock), decides from the typed failure envelope + the configured
    ``[recovery.subagents]`` policy, schedules a cancellable backoff timer on
    the registry's clock/wakeup, and admits through the SAME ``resume_agent``
    operation an explicit request uses. Episode state and the package-lineage
    recurrence counters are controller-owned and live exactly as long as the
    registry (a parent restart forfeits them with the warm suspension — Q6).
    """

    def __init__(self, view: RecoveryControllerStateView) -> None:
        self._view = view
        self._episodes: dict[str, _Episode] = {}
        self._lineage: dict[str, int] = {}
        # Round-2 V5: per-(agent, key) attribution of the lineage counts.
        # A brief key (``brief:{identity}:{summary}``) can be shared by an
        # UNRELATED same-brief lineage, so the R7 key migration may move
        # only a predecessor's OWN contribution, never the shared total.
        self._lineage_agent_counts: dict[tuple[str, str], int] = {}
        # Injectable determinism seams for tests (production defaults below).
        self._epoch: Callable[[], float] = time.time
        self._jitter_draw: Callable[[], float] = random.random

    # ------------------------------------------------------------------
    # Observation hook (the monitor's completion transition; locked)
    # ------------------------------------------------------------------

    def observe_run_locked(self, record: AgentRecord, run: RunRecord) -> None:
        """Observe one settled run; the caller holds the registry lock.

        INERT unless ``[recovery.subagents] enabled`` — and inert while
        draining, while child admission is closed (session switch), and for
        records of a retired generation (a retired generation forfeits
        recovery along with its records).
        """
        view = self._view
        if view._draining_children or view._admission_closed:
            return
        if not view._generation_is_current(record.root_generation):
            return
        config = _recovery_config(view)
        if not config.enabled:
            return
        if run.status is RunStatus.FAILED:
            self._on_failure_locked(record, run, config)
        elif run.status is RunStatus.COMPLETED:
            self._on_success_locked(record, run)

    # ------------------------------------------------------------------
    # Invalidation hooks (registry lifecycle paths)
    # ------------------------------------------------------------------

    def invalidate(self, record: AgentRecord | None) -> None:
        """Invalidate pending recovery for a superseded record (locked).

        Release pops the record; replacement supersedes resume. Called from
        the registry's recovery-invalidation delegate.
        """
        if record is None:
            return
        episode = self._episodes.get(record.agent_id)
        if episode is not None:
            self._stop_episode_locked(episode, reason="invalidated")

    def cancel_pending(self, agent_id: str) -> None:
        """Cancel one agent's pending schedule (cancellation/release; locked)."""
        episode = self._episodes.get(agent_id)
        if episode is not None:
            self._stop_episode_locked(episode, reason="invalidated")

    def cancel_all(self) -> None:
        """Cancel every pending schedule (generation change, drain, shutdown).

        Safe outside the registry lock: stopping an episode only cancels its
        timer task; the timer's own re-validation is the backstop for any
        schedule armed in a race with the lifecycle transition.
        """
        for episode in list(self._episodes.values()):
            self._stop_episode_locked(episode, reason="invalidated")

    def stand_down_schedule_locked(self, agent_id: str) -> None:
        """Cancel one agent's armed backoff timer, keeping the episode (C6a).

        Called at a COMMITTED resume admission (the WP2 operation, explicit or
        automatic): the timer must never fire on a RESTORING record. The
        episode itself stays active — the successor's completion transition
        closes it (success/exhaustion accounting), and the automatic route's
        own post-admission counting still sees an active episode. A REJECTED
        admission never reaches this hook, so a rejected admission preserves
        the pending timer (read-only waits never consume anything).
        """
        episode = self._episodes.get(agent_id)
        if episode is not None and episode.active and episode.timer is not None:
            episode.timer.cancel()
            episode.timer = None
            episode.scheduled_failed_run_id = None

    def rearm_schedule_locked(self, agent_id: str) -> None:
        """Re-arm a stood-down backoff timer after a failed post-commit admission.

        C6a: a committed resume admission stands the pending automatic
        schedule down (the timer must never fire on a RESTORING record);
        when the successor admission then fails, the rollback restores the
        SUSPENDED record — and must restore the schedule too. A suspended
        agent never produces the new failure envelope that would re-arm it,
        so without this the automatic recovery route is dead for that agent.
        Read-only for the counters: the re-arm consumes no budget, and the
        scheduled milestone keeps its exactly-once emission.
        """
        episode = self._episodes.get(agent_id)
        if episode is None or not episode.active or episode.timer is not None:
            return
        record = self._view._agent_records.get(agent_id)
        # A settled run leaves ``current_run`` cleared (it lives in the
        # history); the suspended record's failed run is the one its
        # ``latest_run_id`` still names.
        run = (
            next(
                (
                    item
                    for item in (record.current_run, *record.run_history)
                    if item is not None and item.run_id == record.latest_run_id
                ),
                None,
            )
            if record is not None
            else None
        )
        if record is None or run is None or run.status is not RunStatus.FAILED:
            return
        if (
            record.state is not _AgentState.SUSPENDED
            or record.latest_run_id != run.run_id
            or record.recovery_superseded
            or record.reserved_for_replacement
        ):
            return
        availability = self._availability(record)
        if availability.permanent:
            self._stop_episode_locked(
                episode, reason="permanent_incompatibility", emit_exhaustion=True
            )
            return
        delay = self._compute_delay(episode, availability)
        # The admission would happen after the advisory deadline: stop now
        # instead of arming a timer that can only be refused.
        remaining = (
            episode.started_at + episode.episode_deadline_seconds - self._epoch()
        )
        if delay > remaining:
            self._stop_episode_locked(
                episode, reason="episode_deadline_passed", emit_exhaustion=True
            )
            return
        self._arm_timer_locked(record, run, episode, delay)

    # ------------------------------------------------------------------
    # Failure path (locked)
    # ------------------------------------------------------------------

    def _on_failure_locked(  # noqa: PLR0911 - the gate matrix
        self, record: AgentRecord, run: RunRecord, config: RecoveryConfig
    ) -> None:
        envelope = run.failure
        chain_id = record.recovery_chain_id or (
            envelope.recovery_chain_id
            if envelope is not None
            else f"recovery-chain:{record.agent_id}:{run.run_id}"
        )
        episode = self._episodes.get(record.agent_id)
        if episode is None or not episode.active or episode.chain_id != chain_id:
            # A new episode snapshots the policy: a config reload never
            # extends an ACTIVE episode or replenishes its count.
            episode = _Episode(
                agent_id=record.agent_id,
                chain_id=chain_id,
                started_at=(
                    envelope.observed_at if envelope is not None else self._epoch()
                ),
                auto_retries_max=config.auto_retries_max,
                cooldowns=list(config.cooldowns_seconds),
                jitter_fraction=config.jitter_fraction,
                episode_deadline_seconds=config.episode_deadline_seconds,
            )
            self._episodes[record.agent_id] = episode
        # The failure milestone — once per deduplicatable failure event.
        event_id = (
            envelope.event_id
            if envelope is not None
            else f"failure-event:{record.agent_id}:{run.run_id}"
        )
        self._emit(
            episode,
            EVENT_FAILURE,
            event_id=event_id,
            run_id=run.run_id,
            detail=(
                f"class={envelope.failure.failure_class.value}"
                if envelope is not None
                else "class=unavailable"
            ),
        )
        # No typed envelope = no measured transient cause — never schedule.
        if envelope is None:
            self._stop_episode_locked(
                episode, reason="no_failure_envelope", emit_exhaustion=episode.driven
            )
            return
        # C5a: a harness-measured progress deadline NEVER schedules the
        # automatic same-payload continuation — the specific measured timeout
        # reason stops the episode regardless of the configured eligible
        # classes. The envelope-level classification keeps the explicit routes
        # (resume admission, replacement claim) available; only the AUTOMATIC
        # continuation is prohibited here. This gate names the specific
        # reason in the exhaustion milestone instead of a generic refusal.
        if measured_timeout_reason_of(envelope) is not None:
            self._stop_episode_locked(
                episode,
                reason="resume_progress_deadline_no_auto_recovery",
                emit_exhaustion=episode.driven,
            )
            return
        # Gate 1: the envelope's effective eligibility. RECONCILE_REQUIRED is
        # a barrier that never auto-clears; PROHIBITED covers every protected
        # class (cancellation/budget/authorization/preflight/task/permanent
        # config) plus unknown/internal without a transient cause.
        eligibility, _reason_codes = _effective_eligibility(record, envelope)
        if eligibility is not RecoveryEligibility.ELIGIBLE:
            self._stop_episode_locked(
                episode,
                reason=f"not_eligible:{envelope.failure.failure_class.value}",
                emit_exhaustion=episode.driven,
            )
            return
        # Gate 2: the configured eligible_classes allowlist.
        eligible_class = _eligible_class_of(envelope)
        if eligible_class is None or eligible_class not in config.eligible_classes:
            self._stop_episode_locked(
                episode, reason="class_not_eligible", emit_exhaustion=True
            )
            return
        # Gate 3: the ONE episode budget (counted on admission, never on
        # scheduling — a read-only wait never consumes a retry).
        if episode.auto_retries_used >= episode.auto_retries_max:
            self._stop_episode_locked(
                episode, reason="auto_retries_exhausted", emit_exhaustion=True
            )
            return
        # Gate 4 (handoff §2.3): a confirmed transient launch_failure gets
        # ONE bounded startup retry; a repeated launch failure STOPS the
        # relaunch loop.
        is_launch_failure = (
            envelope.failure.failure_class is FailureClass.LAUNCH_FAILURE
        )
        if (
            is_launch_failure
            and episode.launch_retries_used >= _LAUNCH_FAILURE_RETRIES_MAX
        ):
            self._stop_episode_locked(
                episode, reason="repeated_launch_failure", emit_exhaustion=True
            )
            return
        # Gate 5: the package-lineage recurrence counter spans replacements.
        lineage_key = self._resolve_lineage_key_locked(record, envelope)
        if self._lineage.get(lineage_key, 0) >= episode.auto_retries_max:
            self._stop_episode_locked(
                episode, reason="package_recurrence_exhausted", emit_exhaustion=True
            )
            return
        # Gate 6 (Q2-negative): the aggregate deadline is ADVISORY — no new
        # admission after it, never an in-flight kill.
        if self._epoch() >= episode.started_at + episode.episode_deadline_seconds:
            self._stop_episode_locked(
                episode, reason="episode_deadline_passed", emit_exhaustion=True
            )
            return
        # Gate 7: the record must be SUSPENDED on exactly this failed run.
        if (
            record.state is not _AgentState.SUSPENDED
            or record.latest_run_id != run.run_id
        ):
            self._stop_episode_locked(
                episode, reason="not_suspended", emit_exhaustion=episode.driven
            )
            return
        # Gate 8: the read-only availability snapshot. Endpoint-correlated
        # throttling blocks recovery onto ALL affected-endpoint deployments;
        # permanent incompatibility stops the episode; temporary exclusions
        # only postpone admission (without consuming budget).
        availability = self._availability(record)
        if availability.permanent:
            self._stop_episode_locked(
                episode, reason="permanent_incompatibility", emit_exhaustion=True
            )
            return
        episode.pending_retry_after = envelope.request.retry_after_seconds or 0.0
        episode.pending_launch_failure = is_launch_failure
        episode.pending_lineage_key = lineage_key
        delay = self._compute_delay(episode, availability)
        # The admission would happen after the advisory deadline: stop now
        # instead of arming a timer that can only be refused.
        remaining = (
            episode.started_at + episode.episode_deadline_seconds - self._epoch()
        )
        if delay > remaining:
            self._stop_episode_locked(
                episode, reason="episode_deadline_passed", emit_exhaustion=True
            )
            return
        self._arm_timer_locked(record, run, episode, delay)

    def _resolve_lineage_key_locked(
        self, record: AgentRecord, envelope: FailureEnvelope
    ) -> str:
        """The recurrence key, migrating predecessor-keyed counts (locked).

        Review R7: an initially-unclaimed package's key changes
        ``brief:...`` → ``pkg:...`` when a replacement claim establishes
        the package identity — the earlier automatic-admission count
        under the predecessor's key must migrate with it, never reset.
        The predecessor chain is walked (bounded, cycle-safe); each
        ancestor's count under a DIFFERENT key folds into this key.
        Idempotent: migrated keys are popped, so a re-walk folds nothing
        twice.

        Round-2 V5: only the count the ancestor's OWN lineage contributed
        migrates. A brief key can be shared by an unrelated same-brief
        lineage; the per-(agent, key) attribution map scopes the fold, and
        the unrelated lineage's count stays under the shared key.
        """
        key = _lineage_key(record, envelope)
        if not key.startswith("pkg:"):
            return key
        migrated = 0
        seen = {record.agent_id}
        ancestor_id = record.predecessor_agent_id
        while ancestor_id is not None and ancestor_id not in seen:
            seen.add(ancestor_id)
            ancestor = self._view._agent_records.get(ancestor_id)
            if ancestor is None:
                break
            ancestor_run = next(
                (
                    item
                    for item in (ancestor.current_run, *reversed(ancestor.run_history))
                    if item is not None
                    and item.status is RunStatus.FAILED
                    and item.failure is not None
                ),
                None,
            )
            ancestor_envelope = ancestor_run.failure if ancestor_run else None
            if ancestor_envelope is not None:
                ancestor_key = _lineage_key(ancestor, ancestor_envelope)
                if ancestor_key != key:
                    # Round-2 V5: move only THIS ancestor's attributed
                    # contribution; an unrelated same-brief lineage's count
                    # under the shared key is never stolen.
                    attributed = self._lineage_agent_counts.pop(
                        (ancestor.agent_id, ancestor_key), 0
                    )
                    if attributed:
                        migrated += attributed
                        remaining = self._lineage.get(ancestor_key, 0) - attributed
                        if remaining > 0:
                            self._lineage[ancestor_key] = remaining
                        else:
                            self._lineage.pop(ancestor_key, None)
            ancestor_id = ancestor.predecessor_agent_id
        if migrated:
            self._lineage[key] = self._lineage.get(key, 0) + migrated
        return key

    def _arm_timer_locked(
        self, record: AgentRecord, run: RunRecord, episode: _Episode, delay: float
    ) -> None:
        """Arm the cancellable backoff timer (the caller holds the lock)."""
        if episode.timer is not None:
            # A newer failure supersedes an older pending schedule.
            episode.timer.cancel()
        episode.scheduled_failed_run_id = run.run_id
        episode.timer = asyncio.create_task(
            self._timer(record.agent_id, episode, run.run_id, delay),
            name=f"vibe-subagent-recovery:{record.agent_id}",
        )
        self._emit(
            episode,
            EVENT_SCHEDULED,
            event_id=(
                f"scheduled:{episode.agent_id}:{episode.chain_id}:"
                f"{episode.auto_retries_used}:{run.run_id}"
            ),
            run_id=run.run_id,
            detail=(
                f"delay={delay:.3f} retry_index={episode.auto_retries_used} "
                "deadline=advisory"
            ),
        )

    # ------------------------------------------------------------------
    # Success path (locked)
    # ------------------------------------------------------------------

    def _on_success_locked(self, record: AgentRecord, run: RunRecord) -> None:
        """Close the episode when a chain-continuation run completes."""
        episode = self._episodes.get(record.agent_id)
        if episode is None or not episode.active:
            return
        if episode.timer is not None:
            # An explicit resume raced the pending schedule: stand the timer
            # down; the successor's completion closes the episode.
            episode.timer.cancel()
            episode.timer = None
        if (
            run.resumed_from_run_id is None
            and run.recovery_chain_id != episode.chain_id
        ):
            # The agent moved on to NEW explicit work: the automatic episode
            # ended without its own successor completing.
            self._stop_episode_locked(
                episode, reason="superseded_by_explicit_run", emit_exhaustion=True
            )
            return
        self._emit(
            episode,
            EVENT_SUCCESS,
            event_id=f"success:{episode.agent_id}:{episode.chain_id}",
            run_id=run.run_id,
            detail=f"auto_retries_used={episode.auto_retries_used}",
        )
        episode.active = False
        episode.scheduled_failed_run_id = None
        self._episodes.pop(episode.agent_id, None)

    # ------------------------------------------------------------------
    # The timer: re-validate, then admit through the SAME operation
    # ------------------------------------------------------------------

    async def _timer(  # noqa: PLR0911, PLR0912, PLR0915 - re-validation ladder
        self, agent_id: str, episode: _Episode, failed_run_id: str, delay: float
    ) -> None:
        """Wait out the backoff, re-validate, and admit the successor.

        Runs independently of parent turns: the wakeup is the registry's
        (possibly gated) sleep, and the admission is the WP2 operation.
        """
        view = self._view
        try:
            await view._wakeup(delay)
        except asyncio.CancelledError:
            return  # invalidation owns the episode state
        except Exception as exc:
            logger.warning("Recovery timer wakeup failed", exc_info=exc)
            return
        async with view._registry_lock:
            episode.timer = None
            if not episode.active or episode.scheduled_failed_run_id != failed_run_id:
                # Superseded by a newer schedule or a closed episode.
                return
            record = view._agent_records.get(agent_id)
            if record is None:
                self._stop_episode_locked(episode, reason="invalidated")
                return
            barrier_intact = (
                record.state is _AgentState.SUSPENDED
                and record.latest_run_id == failed_run_id
                and not record.recovery_superseded
                and not record.reserved_for_replacement
            )
            generation_current = view._generation_is_current(record.root_generation)
            if (
                not barrier_intact
                or not generation_current
                or view._draining_children
                or view._admission_closed
            ):
                self._stop_episode_locked(episode, reason="invalidated")
                return
            config = _recovery_config(view)
            if not config.enabled:
                self._stop_episode_locked(episode, reason="disabled")
                return
            if self._epoch() >= episode.started_at + episode.episode_deadline_seconds:
                # Q2-negative: the deadline stops NEW admissions only.
                self._stop_episode_locked(
                    episode, reason="episode_deadline_passed", emit_exhaustion=True
                )
                return
            availability = self._availability(record)
            if availability.permanent:
                self._stop_episode_locked(
                    episode, reason="permanent_incompatibility", emit_exhaustion=True
                )
                return
            if not availability.available:
                # Wait for the temporary exclusion WITHOUT consuming budget:
                # re-arm for the next read-only check.
                rearm_delay = self._compute_delay(episode, availability)
                remaining = (
                    episode.started_at
                    + episode.episode_deadline_seconds
                    - self._epoch()
                )
                if rearm_delay > remaining:
                    self._stop_episode_locked(
                        episode, reason="episode_deadline_passed", emit_exhaustion=True
                    )
                    return
                episode.timer = asyncio.create_task(
                    self._timer(agent_id, episode, failed_run_id, rearm_delay),
                    name=f"vibe-subagent-recovery:{agent_id}",
                )
                return
        # Admit through the SAME operation an explicit resume uses. The
        # admission takes the registry lock itself, so it runs unlocked here.
        root = view._root
        if root is None:
            async with view._registry_lock:
                self._stop_episode_locked(episode, reason="no_parent")
            return
        idempotency_key = (
            f"auto-recovery:{episode.chain_id}:{episode.auto_retries_used}"
        )
        # Review R8: capacity/concurrency admission rejections raise (the
        # explicit route surfaces the same error); the automatic episode
        # stops with an exhaustion milestone instead of dying mid-timer.
        try:
            result = await view.resume_agent(
                agent_id,
                failed_run_id,
                idempotency_key,
                requester_session_id=root.agent_loop.session_id,
            )
        except RuntimeError as exc:
            logger.warning(
                "Automatic recovery admission rejected for agent %s: %s", agent_id, exc
            )
            async with view._registry_lock:
                self._stop_episode_locked(
                    episode, reason="admission_capacity_rejected", emit_exhaustion=True
                )
            return
        async with view._registry_lock:
            if not episode.active:
                # Invalidation raced the admission: the admission is the
                # authority for the successor run (if any); this closed
                # episode no longer counts or emits.
                return
            if result.outcome is ResumeOutcome.SCHEDULED:
                # Count ADMITTED continuations, never request attempts.
                episode.auto_retries_used += 1
                if episode.pending_launch_failure:
                    episode.launch_retries_used += 1
                self._lineage[episode.pending_lineage_key] = (
                    self._lineage.get(episode.pending_lineage_key, 0) + 1
                )
                # Round-2 V5: attribute the count to the admitted agent so
                # a later key migration moves only this lineage's own
                # contribution (a shared brief key is not all ours).
                attribution = (episode.agent_id, episode.pending_lineage_key)
                self._lineage_agent_counts[attribution] = (
                    self._lineage_agent_counts.get(attribution, 0) + 1
                )
                episode.scheduled_failed_run_id = None
                self._emit(
                    episode,
                    EVENT_SUCCESSOR_START,
                    event_id=(
                        f"successor-start:{episode.agent_id}:{result.successor_run_id}"
                    ),
                    run_id=result.successor_run_id,
                    detail=(
                        f"operation={result.recovery_operation_id} "
                        f"auto_retries_used={episode.auto_retries_used}"
                    ),
                )
            elif result.outcome in {
                ResumeOutcome.ALREADY_SCHEDULED,
                ResumeOutcome.ALREADY_RESUMED,
            }:
                # An explicit resume won the single-flight race: stand down;
                # the successor's completion closes this episode.
                episode.scheduled_failed_run_id = None
            else:
                self._stop_episode_locked(
                    episode,
                    reason=f"admission_{result.outcome.value}",
                    emit_exhaustion=True,
                )

    # ------------------------------------------------------------------
    # Episode termination + event emission (locked)
    # ------------------------------------------------------------------

    def _stop_episode_locked(
        self, episode: _Episode, *, reason: str, emit_exhaustion: bool | None = None
    ) -> None:
        """End an episode: cancel its timer, emit exhaustion once (locked)."""
        if not episode.active:
            return
        # Capture BEFORE clearing the schedule state: an episode that armed a
        # schedule but never admitted a retry is still driven and must
        # report its stop (cancellation/release/drain invalidation).
        was_driven = episode.driven
        episode.active = False
        if episode.timer is not None:
            episode.timer.cancel()
            episode.timer = None
        episode.scheduled_failed_run_id = None
        if emit_exhaustion is None:
            emit_exhaustion = was_driven
        if emit_exhaustion:
            self._emit(
                episode,
                EVENT_EXHAUSTION,
                event_id=f"exhaustion:{episode.agent_id}:{episode.chain_id}",
                detail=reason,
            )
        self._episodes.pop(episode.agent_id, None)

    def _emit(
        self,
        episode: _Episode,
        event: str,
        *,
        event_id: str,
        run_id: str | None = None,
        detail: str = "",
    ) -> None:
        """Emit one milestone line, exactly once per event id (locked)."""
        if event_id in episode.emitted:
            return
        episode.emitted.add(event_id)
        logger.info(
            "recovery-controller event=%s event_id=%s agent_id=%s run_id=%s "
            "chain=%s detail=%s",
            event,
            event_id,
            episode.agent_id,
            run_id or "",
            episode.chain_id,
            detail,
        )

    # ------------------------------------------------------------------
    # Read-only availability + the delay schedule
    # ------------------------------------------------------------------

    def _availability(self, record: AgentRecord) -> _Availability:
        """The non-mutating pre-admission eligibility snapshot (WP4).

        Never claims or releases a single-flight probe (``claim_probes=False``):
        scheduling checks stay read-only, and the probe machinery remains
        token-owned by the callers that actually attempt.
        """
        loop = record.runtime.agent_loop
        config = loop.config
        snapshot = config.catalog_snapshot
        committed = loop.committed_model
        if snapshot is None or committed is None:
            # Missing evidence never blocks scheduling on its own — the
            # admission itself is the authority.
            return _Availability(
                available=True,
                permanent=False,
                next_check_delay=0.0,
                endpoint_blocked=False,
            )
        registry = loop.config_orchestrator.availability_registry
        try:
            eligible_deployments(
                snapshot=snapshot,
                committed=committed,
                registry=registry,
                config=config,
                history=reorder_for_tool_adjacency(select_model_context(loop.messages)),
                thinking=config.get_active_model().thinking,
                compaction_base=config.compaction_model or None,
                allowed_models=config.allowed_models,
                claim_probes=False,
            )
        except AllDeploymentsUnavailableError as exc:
            return self._availability_from_exclusions(
                committed, registry, config, exc.exclusions
            )
        return _Availability(
            available=True,
            permanent=False,
            next_check_delay=0.0,
            endpoint_blocked=False,
        )

    def _availability_from_exclusions(
        self, committed: Any, registry: Any, config: Any, exclusions: Any
    ) -> _Availability:
        """Derive the wait/permanent/endpoint-blocked snapshot from exclusions."""
        now = self._view._clock()
        gaps: list[float] = []
        any_temporary = False
        all_endpoint = bool(exclusions)
        for exclusion in exclusions:
            if exclusion.reason not in _TEMPORARY_EXCLUSIONS:
                all_endpoint = False
                continue
            any_temporary = True
            if exclusion.reason is ExclusionReason.ENDPOINT_COOLDOWN:
                provider = None
                if config.catalog_snapshot is not None:
                    provider = config.catalog_snapshot.catalog.providers.get(
                        exclusion.provider
                    )
                endpoint = canonical_endpoint_identity(
                    getattr(provider, "api_base", None)
                )
                until = registry.endpoint_cooldown_until(endpoint)
            else:
                all_endpoint = False
                until = registry.cooldown_until(
                    committed.base_model, exclusion.provider
                )
            gaps.append(max(0.0, until - now) if until is not None else 0.0)
        if exclusions and not any_temporary:
            # Every deployment is excluded for a permanent reason: recovery
            # must not be scheduled (all-deployments-unavailable rejects
            # permanent incompatibility).
            return _Availability(
                available=False,
                permanent=True,
                next_check_delay=0.0,
                endpoint_blocked=False,
            )
        return _Availability(
            available=False,
            permanent=False,
            next_check_delay=min(gaps) if gaps else 0.0,
            endpoint_blocked=all_endpoint,
        )

    def _compute_delay(self, episode: _Episode, availability: _Availability) -> float:
        """The backoff for the pending schedule: configured cooldown per retry
        index, a longer Retry-After, and any temporary exclusion gap — with
        the configured symmetric jitter.
        """
        cooldowns = episode.cooldowns or [0.0]
        backoff = cooldowns[min(episode.auto_retries_used, len(cooldowns) - 1)]
        base = max(backoff, episode.pending_retry_after)
        if not availability.available:
            base = max(base, availability.next_check_delay)
        return jittered_backoff(base, episode.jitter_fraction, draw=self._jitter_draw)
