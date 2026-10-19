"""R2 — the app-server root-turn recovery gate (plan state.md R2, OD3).

The app-server half of the bounded automatic root-turn recovery pair. The
TUI controller (``chartreux/cli/textual_ui/root_recovery.py``) owns the
in-process timer and local cancellation via an input-generation token; THIS
gate owns the authoritative decisions:

- ELIGIBILITY from the typed terminal projection (:class:`RootTurnFailure`)
  captured BEFORE ``public_error()`` destroys the exception context and
  BEFORE ``_finalize_turn`` cancels callbacks (absence of callbacks
  afterwards proves nothing). Only the typed transient allowlist mirrored
  by ``[recovery.root] eligible_classes`` is eligible, and only with
  MEASURED transient provenance: a tool's network exception carries no
  call evidence and is never a root model-call failure.
- EPISODE ACCOUNTING: ONE shared automatic-turn budget per user-turn
  episode, MERGING the legacy incomplete-stream continuation (limit 2)
  into this controller — alternating failure kinds share one count and one
  monotonic deadline anchored to the initial failed logical call's start.
  Count ADMITTED continuations, not request attempts. A manual user retry
  starts a NEW user-directed episode and never replenishes this one.
- SINGLE-FLIGHT ADMISSION: the idempotency key
  ``root-auto-recovery:{episode_id}:{retry_index}`` admits at most one
  successor; duplicate admits reconcile to the original successor (or
  report the consumed admission) and never create a second successor.
- COOLDOWN-AWARE SCHEDULING: a NON-MUTATING availability snapshot — never
  ``reset_base``, never ``user_initiated_retry``, never a probe claim for
  a read-only check. ``due = max(configured backoff, Retry-After, candidate
  cooldown gap) + bounded jitter``; at admission the live availability is
  re-read (a probe in flight defers within the episode without consuming
  budget). All-deployments-unavailable waits only when exclusions are
  transient/cooldown-backed; permanent incompatibility is not
  auto-retryable.

Code invariants (never config escape hatches): preflight, permission/
authorization denial, budget stop, cancellation, task blocker, permanent
config/auth/billing/quota, context-size, and unknown-internal without
measured transient provenance NEVER schedule. When
``[recovery.root] enabled`` is false NO root automatic continuation runs
at all — including the migrated incomplete-stream path (OD4a); the manual
R1 retry affordance remains the fallback. Meta-recovery: a
controller/notice/admission failure is not another provider failure —
ambiguous starts reconcile via idempotency identity, and anything
unreconcilable parks to the manual fallback, never a self-retry loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
import random
import time
from typing import Literal

from chartreux.app_server._projector import ProjectedUpdate
from chartreux.app_server.models import PublicTurnStatus, RootRecoveryNoticeDetail
from chartreux.app_server.protocol import (
    RootRecoveryAdmitParams,
    RootRecoveryAdmitResponse,
    RootRecoveryCancelParams,
    RootRecoveryCancelResponse,
    RootRecoveryScheduleParams,
    RootRecoveryScheduleResponse,
)
from chartreux.core.agent_loop._transcript_integrity import validate_transcript
from chartreux.core.config.models import RecoveryEligibleClass, RootRecoveryConfig
from chartreux.core.llm_models import LLMMessage
from chartreux.core.model_catalog.availability import AvailabilitySnapshot
from chartreux.core.recovery_models import (
    FailureCategory,
    FailureClass,
    RootTurnFailure,
)
from chartreux.core.utils.retry import jittered_backoff
from chartreux.observability.logging import logger

__all__ = ["RootRecoveryGate", "RootRecoverySessionView"]

#: Bounded FIFO capacity of the typed terminal projections (by turn id).
_PROJECTION_CAPACITY = 64

#: Bounded capacity of the tracked episodes (active + recently ended).
_EPISODE_CAPACITY = 16

#: Bounded FIFO capacity of the terminal-turn → episode index.
_TERMINAL_TURN_CAPACITY = 64

#: The recovery-layer eligible classes (the ONE authoritative WP1b
#: allowlist) mapped from the projection's failure class / cause category.
_CAUSE_TO_ELIGIBLE_CLASS: dict[FailureCategory, RecoveryEligibleClass] = {
    FailureCategory.CONNECTION: RecoveryEligibleClass.CONNECTION,
    FailureCategory.TIMEOUT: RecoveryEligibleClass.TIMEOUT,
    FailureCategory.SERVER_ERROR: RecoveryEligibleClass.SERVER_ERROR,
    FailureCategory.OVERLOAD: RecoveryEligibleClass.OVERLOAD,
    FailureCategory.RATE_LIMIT: RecoveryEligibleClass.RATE_LIMIT,
    FailureCategory.INCOMPLETE_STREAM: RecoveryEligibleClass.INCOMPLETE_STREAM,
    FailureCategory.EMPTY_RESPONSE: RecoveryEligibleClass.EMPTY_RESPONSE,
}

#: The ONLY failure classes that may EVER schedule (a CODE invariant, never a
#: config escape hatch): the measured transient classes. A protected class
#: (preflight, cancellation — including a stop-reason-driven one whose cause
#: category is transient — authorization, permanent config, budget stop, task
#: blocker, internal, unknown) is never eligible, whatever its cause category
#: or provenance flags say.
_TRANSIENT_FAILURE_CLASSES = frozenset({
    FailureClass.PROVIDER_STALL,
    FailureClass.PROVIDER_OVERLOADED,
    FailureClass.PROVIDER_RATE_LIMIT,
    FailureClass.LAUNCH_FAILURE,
    FailureClass.INCOMPLETE_STREAM,
    FailureClass.EMPTY_RESPONSE,
})

#: The milestone vocabulary (deterministic notice ids, exactly once each).
Milestone = Literal["scheduled", "started", "recovered", "exhausted", "cancelled"]


def _eligible_class_of(failure: RootTurnFailure) -> RecoveryEligibleClass | None:
    """Map the typed terminal projection to its recovery-layer eligible class."""
    if failure.failure_class is FailureClass.LAUNCH_FAILURE:
        return RecoveryEligibleClass.LAUNCH_FAILURE
    category = failure.cause_category
    if category is None:
        return None
    return _CAUSE_TO_ELIGIBLE_CLASS.get(category)


def _affected_deployment(failure: RootTurnFailure) -> str | None:
    """The attempted deployment, frozen at the completion boundary."""
    if failure.attempted_provider is not None:
        if failure.attempted_wire_name is not None:
            return f"{failure.attempted_provider}/{failure.attempted_wire_name}"
        return failure.attempted_provider
    return failure.attempted_base_model


@dataclass(frozen=True, slots=True)
class RootRecoverySessionView:
    """Live server-side interaction state, read at schedule/admission time.

    The interaction gates (plan R2): no schedule or admission while any
    runtime approval, user-question callback, queued user work, unsettled
    predecessor cleanup, or lifecycle reservation exists.
    """

    session_id: str
    active_turn_id: str | None
    queued_work: bool
    open_callbacks: bool
    stop_requested: bool
    handoff_pending: bool
    lifecycle_reserved: bool
    closing: bool


@dataclass(slots=True)
class _RootEpisode:
    """One automatic-recovery episode: a continuation chain's bounded budget.

    ``started_at`` anchors the ADVISORY aggregate deadline at the initial
    failed logical call's start (capture time minus the measured call
    elapsed); it is never re-anchored — no reset on partial output or
    category change. The policy fields are snapshotted at creation: a
    config reload never extends an active episode or replenishes its
    count. The deadline is a scheduling/admission bound, NEVER an in-flight
    kill (Q2 negative honored).
    """

    episode_id: str
    initial_turn_id: str
    started_at: float
    deadline_at: float
    auto_retries_max: int
    cooldowns: list[float]
    jitter_fraction: float
    availability_recheck_seconds: float
    # ONE shared count of ADMITTED continuations (the merged legacy
    # incomplete-stream budget counts here too).
    admitted_count: int = 0
    pending_key: str | None = None
    pending_turn_id: str | None = None
    pending_attempt: int = 0
    pending_delay: float = 0.0
    pending_input_generation: int = 0
    successor_turn_id: str | None = None
    successor_in_flight: bool = False
    awaiting_successor: bool = False
    # Single-flight admission map: idempotency key → attempt index.
    admitted_keys: dict[str, int] = field(default_factory=dict)
    emitted: set[str] = field(default_factory=set)
    ended: bool = False

    @property
    def driven(self) -> bool:
        """Whether the episode ever armed a schedule or admitted a retry."""
        return (
            self.admitted_count > 0
            or self.pending_key is not None
            or self.successor_turn_id is not None
        )


class RootRecoveryGate:
    """The authoritative app-server gate for automatic root-turn recovery.

    Narrow collaborator of the turn controller (constructed inside
    ``TurnController.__init__``): the controller feeds it the typed
    terminal projections (captured before ``public_error()`` and before
    callback cancellation), the turn start/finish transitions, and the live
    session view; the TUI controller drives the wire surface
    (``recovery/root/schedule|admit|cancel``). Every decision re-reads the
    live config, session view, and availability — nothing is cached across
    a decision.
    """

    def __init__(
        self,
        *,
        config: Callable[[], RootRecoveryConfig],
        session_view: Callable[[], RootRecoverySessionView],
        availability: Callable[[], AvailabilitySnapshot],
        stage_notice: Callable[[RootRecoveryNoticeDetail, str], ProjectedUpdate | None],
        emit_projected: Callable[[ProjectedUpdate], Awaitable[None]],
        transcript_integrity: Callable[[], Sequence[LLMMessage]] | None = None,
    ) -> None:
        self._config = config
        self._session_view = session_view
        self._availability = availability
        self._stage_notice_callback = stage_notice
        self._emit_projected = emit_projected
        # PI1/T (S6): OPTIONAL collaborator — the live transcript view the
        # schedule/admission pre-check validates (fail closed on a poisoned
        # transcript: park to the manual fallback, never a doomed retry).
        # Defaulted so existing constructions (and the gate unit tests) are
        # unaffected.
        self._transcript_integrity = transcript_integrity
        # Typed terminal projections by turn id (bounded FIFO).
        self._projections: dict[str, RootTurnFailure] = {}
        # Active + recently ended episodes by episode id.
        self._episodes: dict[str, _RootEpisode] = {}
        # Terminal turn id → episode id (bounded FIFO; survives episode end
        # so duplicate observations of one failed turn never re-schedule).
        self._by_terminal_turn: dict[str, str] = {}
        # Injectable determinism seams for tests (production defaults below).
        self._clock: Callable[[], float] = time.time
        self._jitter_draw: Callable[[], float] = random.random

    # ------------------------------------------------------------------
    # Observation hooks (the turn controller's terminal/start transitions)
    # ------------------------------------------------------------------

    def capture_terminal(self, failure: RootTurnFailure) -> None:
        """Stash one typed terminal projection, keyed by turn id (bounded).

        Called at the terminalization boundary — at ``_run_turn_phase``'s
        except (BEFORE ``public_error``) and at the top of
        ``_terminalize_failed_turn`` (BEFORE ``_cancel_callbacks``) — so
        the eligibility decision never depends on post-cancellation state.
        Total by contract: a capture failure must never mask the primary
        failure, so the caller wraps this in a suppressing try.
        """
        self._projections[failure.turn_id] = failure
        while len(self._projections) > _PROJECTION_CAPACITY:
            self._projections.pop(next(iter(self._projections)))

    def observe_turn_started(
        self, turn_id: str, idempotency_key: str | None
    ) -> list[ProjectedUpdate]:
        """Observe one turn start; link or refuse the pending continuation.

        A start carrying an admitted continuation's idempotency key links
        the successor to its episode (exactly one successor — duplicate
        start acks are no-ops) and stages the ``started`` notice. Any OTHER
        turn start while a schedule is pending or an admission awaits its
        successor is a user/harness takeover: the schedule is cancelled and
        the reason shown — never silently rearmed. Returns the staged
        notice updates for the caller to deliver (this hook runs in a
        synchronous context).
        """
        updates: list[ProjectedUpdate] = []
        if idempotency_key is not None:
            for episode in list(self._episodes.values()):
                if episode.ended:
                    continue
                if idempotency_key in episode.admitted_keys:
                    if episode.awaiting_successor:
                        attempt = episode.admitted_keys[idempotency_key]
                        episode.awaiting_successor = False
                        episode.successor_turn_id = turn_id
                        episode.successor_in_flight = True
                        update = self._stage_notice(
                            episode,
                            "started",
                            attempt,
                            message=(
                                f"Automatic retry {attempt} of "
                                f"{episode.auto_retries_max} started as turn "
                                f"{turn_id}. The previous turn failed. "
                                "Completed tool results are kept; recent "
                                "actions may repeat."
                            ),
                            successor_turn_id=turn_id,
                            predecessor_turn_id=episode.pending_turn_id,
                        )
                        if update is not None:
                            updates.append(update)
                    # A duplicate start ack for the linked successor: no
                    # second successor, no second notice.
                    return updates
        # A non-continuation turn: user takeover / harness work cancels any
        # pending schedule or unlinked admission.
        for episode in list(self._episodes.values()):
            if episode.ended or episode.successor_in_flight:
                continue
            update = self._stage_notice(
                episode,
                "cancelled",
                max(episode.admitted_count, episode.pending_attempt),
                message=(
                    "Automatic root recovery cancelled: a new turn started "
                    f"(turn {turn_id}) before the continuation was admitted."
                ),
                reason="superseded_by_turn",
            )
            if update is not None:
                updates.append(update)
            self._end(episode)
        return updates

    async def observe_turn_finished(
        self, turn_id: str, status: PublicTurnStatus
    ) -> None:
        """Observe one turn's terminal transition.

        A COMPLETED successor recovers the episode (``recovered`` notice,
        episode ends). An INTERRUPTED successor is a user takeover
        (``cancelled`` notice, episode ends). A FAILED successor keeps the
        episode and its linkage: the failure's own observation drives the
        next schedule within the SAME budget and deadline.
        """
        for episode in list(self._episodes.values()):
            if episode.ended or episode.successor_turn_id != turn_id:
                continue
            if status is PublicTurnStatus.COMPLETED:
                await self._emit_notice(
                    episode,
                    "recovered",
                    max(episode.admitted_count, 1),
                    message=(
                        "Automatic root recovery succeeded: turn "
                        f"{turn_id} completed after "
                        f"{episode.admitted_count} automatic "
                        "continuation(s)."
                    ),
                    successor_turn_id=turn_id,
                    predecessor_turn_id=episode.pending_turn_id,
                )
                self._end(episode)
            elif status is PublicTurnStatus.INTERRUPTED:
                await self._emit_notice(
                    episode,
                    "cancelled",
                    max(episode.admitted_count, 1),
                    message=(
                        "Automatic root recovery cancelled by the user: "
                        f"turn {turn_id} was interrupted."
                    ),
                    reason="interrupted_by_user",
                    successor_turn_id=turn_id,
                )
                self._end(episode)
            else:
                # FAILED: the successor's own terminal observation continues
                # the episode; the linkage stays for budget/deadline sharing.
                episode.successor_in_flight = False
            return

    def invalidate_all(self) -> None:
        """Drop every episode, projection, and admission (session change /
        shutdown). References are invalidated, never silently rearmed; the
        failed turns, notices, and history entries are retained by the
        session itself. No notices: the session is going away or resetting.
        """
        self._projections.clear()
        self._episodes.clear()
        self._by_terminal_turn.clear()

    # ------------------------------------------------------------------
    # The wire surface (the TUI controller's only production caller)
    # ------------------------------------------------------------------

    async def schedule(  # noqa: PLR0911, PLR0912 — one branch per eligibility decision
        self, params: RootRecoveryScheduleParams
    ) -> RootRecoveryScheduleResponse:
        """Decide whether to schedule the automatic continuation of one
        terminally failed root turn (bounded, cooldown-aware).
        """
        config = self._config()
        view = self._session_view()
        if view.session_id != params.session_id:
            return self._declined_schedule("session_changed")
        if not config.enabled:
            # OD4a removal-by-default: NO root automatic continuation runs
            # at all while disabled — including the migrated incomplete-
            # stream path. The manual R1 retry affordance is the fallback.
            return self._declined_schedule("disabled")
        existing = self._by_terminal_turn.get(params.terminal_turn_id)
        if existing is not None:
            episode = self._episodes.get(existing)
            if (
                episode is not None
                and not episode.ended
                and episode.pending_key is not None
            ):
                # One schedule per failed turn: solicited and unsolicited
                # observations of the same terminal turn share it.
                return RootRecoveryScheduleResponse(
                    decision="already_scheduled",
                    episode_id=episode.episode_id,
                    idempotency_key=episode.pending_key,
                    attempt=episode.pending_attempt,
                    limit=episode.auto_retries_max,
                    delay_seconds=episode.pending_delay,
                )
            return self._declined_schedule("episode_ended")
        failure = self._projections.get(params.terminal_turn_id)
        if failure is None:
            # Never schedule without the typed terminal projection.
            return self._declined_schedule("no_typed_projection")
        integrity = self._transcript_integrity_reason()
        if integrity is not None:
            # PI1/T (S6): a poisoned transcript never schedules another
            # doomed continuation — end the linked episode and park to the
            # manual fallback with the repair offer (session reload per S4).
            await self._end_linked_episode(failure.turn_id, integrity)
            return self._declined_schedule(integrity)
        eligible = _eligible_class_of(failure)
        if (
            eligible is None
            or failure.failure_class not in _TRANSIENT_FAILURE_CLASSES
            or not failure.transient_cause_confirmed
        ):
            # NEVER auto-retry preflight/permission/authorization denial/
            # budget stop/cancellation/task blocker/permanent config-auth-
            # billing-quota/context-size/unknown-internal without measured
            # transient provenance. The protected failure CLASSES are an
            # absolute exclusion — a stop-reason-driven cancellation whose
            # cause category is transient is still a cancellation.
            await self._end_linked_episode(failure.turn_id, "not_eligible")
            return self._declined_schedule(
                "not_eligible", eligible_class=_class_hint(failure)
            )
        if eligible not in config.eligible_classes:
            await self._end_linked_episode(failure.turn_id, "class_not_enabled")
            return self._declined_schedule(
                "class_not_enabled", eligible_class=eligible.value
            )
        if failure.interaction_barrier:
            await self._end_linked_episode(failure.turn_id, "interaction_barrier")
            return self._declined_schedule(
                "interaction_barrier",
                eligible_class=eligible.value,
                reason_detail=failure.barrier_detail,
            )
        if not failure.effects_settled:
            await self._end_linked_episode(failure.turn_id, "effects_unsettled")
            return self._declined_schedule(
                "effects_unsettled", eligible_class=eligible.value
            )
        barrier = self._barrier_reason(view)
        if barrier is not None:
            # No schedule while any runtime approval, user-question
            # callback, queued user work, unsettled predecessor cleanup,
            # or lifecycle reservation exists — cancel and show why.
            await self._end_linked_episode(failure.turn_id, barrier)
            return self._declined_schedule(barrier, eligible_class=eligible.value)
        episode = self._episode_for(failure, config)
        if episode.admitted_count >= episode.auto_retries_max:
            await self._exhaust(episode, "budget_exhausted")
            return self._declined_schedule(
                "budget_exhausted", episode=episode, eligible_class=eligible.value
            )
        now = self._clock()
        if now >= episode.deadline_at:
            await self._exhaust(episode, "deadline_exceeded")
            return self._declined_schedule(
                "deadline_exceeded", episode=episode, eligible_class=eligible.value
            )
        snapshot = self._live_availability()
        if snapshot.permanent:
            # Permanent incompatibility (disabled/allowlist/images/thinking/
            # reasoning/compaction) is not auto-retryable.
            await self._exhaust(episode, "permanent_incompatibility")
            return self._declined_schedule(
                "permanent_incompatibility",
                episode=episode,
                eligible_class=eligible.value,
            )
        delay = self._compute_delay(episode, failure, snapshot)
        if delay > episode.deadline_at - now:
            await self._exhaust(episode, "deadline_exceeded")
            return self._declined_schedule(
                "deadline_exceeded", episode=episode, eligible_class=eligible.value
            )
        attempt = episode.admitted_count + 1
        key = f"root-auto-recovery:{episode.episode_id}:{attempt}"
        episode.pending_key = key
        episode.pending_turn_id = params.terminal_turn_id
        episode.pending_attempt = attempt
        episode.pending_delay = delay
        episode.pending_input_generation = params.input_generation
        self._by_terminal_turn[params.terminal_turn_id] = episode.episode_id
        while len(self._by_terminal_turn) > _TERMINAL_TURN_CAPACITY:
            self._by_terminal_turn.pop(next(iter(self._by_terminal_turn)))
        deployment = _affected_deployment(failure)
        await self._emit_notice(
            episode,
            "scheduled",
            attempt,
            message=(
                f"Automatic retry {attempt} of {episode.auto_retries_max} "
                f"scheduled in {delay:.1f}s after a terminal root turn "
                f"failure ({eligible.value}"
                + (f" on {deployment}" if deployment else "")
                + "). No user action is required; typing, submitting, or "
                "retrying manually cancels it."
            ),
            eligible_class=eligible.value,
            affected_deployment=deployment,
            delay_seconds=delay,
        )
        return RootRecoveryScheduleResponse(
            decision="scheduled",
            episode_id=episode.episode_id,
            idempotency_key=key,
            attempt=attempt,
            limit=episode.auto_retries_max,
            delay_seconds=delay,
            eligible_class=eligible.value,
            affected_deployment=deployment,
        )

    async def admit(  # noqa: PLR0911, PLR0912 — one branch per admission recheck
        self, params: RootRecoveryAdmitParams
    ) -> RootRecoveryAdmitResponse:
        """Single-flight admission of one scheduled continuation.

        Rechecks session/turn/input generation, queued work, callbacks, stop
        state, budget, deadline, and live availability AT admission. A
        ``deferred`` decision is a read-only availability wait — it never
        consumes a retry.
        """
        config = self._config()
        view = self._session_view()
        if view.session_id != params.session_id:
            return self._declined_admit("session_changed")
        episode = self._episodes.get(params.episode_id)
        if episode is None or episode.ended:
            return self._declined_admit("no_episode")
        if not config.enabled:
            # Config disable cancels the pending schedule; an admitted
            # successor in flight settles via normal interruption rules
            # (never cancel an already-admitted call as rollback).
            if episode.successor_in_flight:
                return self._declined_admit("disabled")
            await self._cancel_episode(episode, "disabled")
            return self._declined_admit("disabled")
        integrity = self._transcript_integrity_reason()
        if integrity is not None:
            # PI1/T (S6): a poisoned transcript never admits another doomed
            # continuation — cancel the pending schedule and park to the
            # manual fallback with the repair offer (session reload per S4).
            await self._cancel_episode(episode, integrity)
            return self._declined_admit(integrity)
        if params.idempotency_key in episode.admitted_keys:
            # Single-flight: an ambiguous redelivery reconciles to the
            # original successor — never a second admission, never a loop.
            attempt = episode.admitted_keys[params.idempotency_key]
            return RootRecoveryAdmitResponse(
                decision="already_admitted",
                attempt=attempt,
                limit=episode.auto_retries_max,
                turn_id=(
                    episode.successor_turn_id
                    if not episode.awaiting_successor
                    else None
                ),
                predecessor_turn_id=episode.pending_turn_id,
            )
        if params.idempotency_key != episode.pending_key:
            return self._declined_admit("key_mismatch")
        if params.terminal_turn_id != episode.pending_turn_id:
            return self._declined_admit("key_mismatch")
        if params.input_generation != episode.pending_input_generation:
            # User input arrived after the schedule: cancel, never admit.
            await self._cancel_episode(episode, "user_input")
            return self._declined_admit("user_input")
        barrier = self._barrier_reason(view)
        if barrier is not None:
            await self._cancel_episode(episode, barrier)
            return self._declined_admit(barrier)
        if view.active_turn_id is not None:
            await self._cancel_episode(episode, "turn_active")
            return self._declined_admit("turn_active")
        if episode.admitted_count >= episode.auto_retries_max:
            await self._exhaust(episode, "budget_exhausted")
            return self._declined_admit("budget_exhausted")
        now = self._clock()
        if now >= episode.deadline_at:
            # The deadline is an ADVISORY admission bound — it never kills
            # an in-flight request, and none is in flight here.
            await self._exhaust(episode, "deadline_exceeded")
            return self._declined_admit("deadline_exceeded")
        snapshot = self._live_availability()
        if snapshot.permanent:
            await self._exhaust(episode, "permanent_incompatibility")
            return self._declined_admit("permanent_incompatibility")
        if not snapshot.available:
            # A probe in flight or a cooldown gap: wait/recheck within the
            # episode — a read-only check NEVER consumes a retry.
            delay = snapshot.next_check_delay
            if snapshot.probe_in_flight or delay <= 0:
                delay = max(delay, episode.availability_recheck_seconds)
            if now + delay >= episode.deadline_at:
                await self._exhaust(episode, "deadline_exceeded")
                return self._declined_admit("deadline_exceeded")
            return RootRecoveryAdmitResponse(
                decision="deferred",
                attempt=episode.pending_attempt,
                limit=episode.auto_retries_max,
                delay_seconds=delay,
            )
        # ADMIT: count the continuation (not the request attempt).
        attempt = episode.pending_attempt
        episode.admitted_count += 1
        episode.admitted_keys[params.idempotency_key] = attempt
        episode.awaiting_successor = True
        return RootRecoveryAdmitResponse(
            decision="admitted",
            attempt=attempt,
            limit=episode.auto_retries_max,
            predecessor_turn_id=params.terminal_turn_id,
        )

    async def cancel(
        self, params: RootRecoveryCancelParams
    ) -> RootRecoveryCancelResponse:
        """Cancel a pending schedule (user takeover / local cancellation)."""
        view = self._session_view()
        if view.session_id != params.session_id:
            return RootRecoveryCancelResponse(cancelled=False)
        episode = self._episodes.get(params.episode_id)
        if episode is None or episode.ended:
            return RootRecoveryCancelResponse(cancelled=False)
        if episode.successor_in_flight:
            # The admission was already consumed; the successor settles via
            # normal interruption rules — never cancelled as rollback.
            return RootRecoveryCancelResponse(cancelled=False)
        await self._cancel_episode(episode, params.reason or "cancelled_by_user")
        return RootRecoveryCancelResponse(cancelled=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _episode_for(
        self, failure: RootTurnFailure, config: RootRecoveryConfig
    ) -> _RootEpisode:
        """The episode a failure belongs to: the active episode whose
        admitted successor failed (ONE shared budget/deadline across
        alternating failure kinds), else a NEW episode anchored to the
        initial failed logical call's start.
        """
        for episode in self._episodes.values():
            if not episode.ended and episode.successor_turn_id == failure.turn_id:
                return episode
        episode_id = f"root-auto-{failure.turn_id}"
        started_at = failure.observed_at - (failure.call_elapsed_seconds or 0.0)
        episode = _RootEpisode(
            episode_id=episode_id,
            initial_turn_id=failure.turn_id,
            started_at=started_at,
            deadline_at=started_at + config.deadline_seconds,
            auto_retries_max=config.auto_retries_max,
            cooldowns=list(config.cooldowns_seconds),
            jitter_fraction=config.jitter_fraction,
            availability_recheck_seconds=config.availability_recheck_seconds,
        )
        self._episodes[episode_id] = episode
        while len(self._episodes) > _EPISODE_CAPACITY:
            self._episodes.pop(next(iter(self._episodes)))
        return episode

    def _barrier_reason(  # noqa: PLR0911 — one return per barrier
        self, view: RootRecoverySessionView
    ) -> str | None:
        """The first live interaction barrier, or None."""
        if view.closing:
            return "closing"
        if view.open_callbacks:
            return "open_callbacks"
        if view.queued_work:
            return "queued_work"
        if view.stop_requested:
            return "stop_requested"
        if view.handoff_pending:
            return "handoff_pending"
        if view.lifecycle_reserved:
            return "lifecycle_reserved"
        return None

    def _transcript_integrity_reason(self) -> str | None:
        """The transcript-integrity decline reason, or None (PI1/T S6).

        Read-only pre-check over the live transcript (live tail exempt: the
        designed backfill repairs unresolved calls before any request). The
        collaborator is OPTIONAL — when absent the pre-check is a no-op. A
        check failure degrades to no-reason (meta-recovery: the gate never
        masks a decision on a pre-check error), and a poisoned transcript
        parks to the manual fallback — never another doomed continuation.
        """
        if self._transcript_integrity is None:
            return None
        try:
            violations = validate_transcript(
                self._transcript_integrity(), live_tail_exempt=True
            )
        except Exception:
            logger.exception("root recovery transcript integrity pre-check failed")
            return None
        if violations:
            logger.warning(
                "Root recovery declined: transcript integrity violations: %s",
                "; ".join(str(violation) for violation in violations),
            )
            return "transcript_integrity"
        return None

    def _live_availability(self) -> AvailabilitySnapshot:
        """The non-mutating eligibility snapshot; a read failure degrades to
        available (missing evidence never blocks on its own — the admission
        itself is the authority).
        """
        try:
            return self._availability()
        except Exception:
            logger.exception("root recovery availability snapshot failed")
            return AvailabilitySnapshot(
                available=True,
                permanent=False,
                endpoint_blocked=False,
                probe_in_flight=False,
                next_check_delay=0.0,
            )

    def _compute_delay(
        self,
        episode: _RootEpisode,
        failure: RootTurnFailure,
        snapshot: AvailabilitySnapshot,
    ) -> float:
        """due = max(configured backoff, applicable Retry-After, candidate
        cooldown gap) + bounded jitter.
        """
        cooldowns = episode.cooldowns or [0.0]
        backoff = cooldowns[min(episode.admitted_count, len(cooldowns) - 1)]
        retry_after = 0.0
        if failure.retry_after_seconds is not None:
            elapsed = self._clock() - failure.observed_at
            retry_after = max(0.0, failure.retry_after_seconds - elapsed)
        base = max(backoff, retry_after)
        if not snapshot.available:
            base = max(base, snapshot.next_check_delay)
        return jittered_backoff(base, episode.jitter_fraction, draw=self._jitter_draw)

    def _end(self, episode: _RootEpisode) -> None:
        episode.ended = True
        self._episodes.pop(episode.episode_id, None)

    async def _end_linked_episode(self, terminal_turn_id: str, reason: str) -> None:
        """End the episode linked to this failed turn with an honest notice
        (a nonrecoverable outcome or a live barrier ends the episode; an
        undriven episode emits nothing).
        """
        for episode in list(self._episodes.values()):
            if episode.ended:
                continue
            if terminal_turn_id in {
                episode.successor_turn_id,
                episode.pending_turn_id,
                episode.initial_turn_id,
            }:
                if episode.driven:
                    await self._emit_notice(
                        episode,
                        "exhausted",
                        max(episode.admitted_count, episode.pending_attempt),
                        message=(
                            "Automatic root recovery stopped: "
                            + _blocked_message(reason, episode)
                            + " The failed turn is kept; the manual Retry "
                            "affordance (where offered) or a new message is "
                            "the fallback."
                        ),
                        reason=reason,
                    )
                self._end(episode)
                return

    async def _exhaust(self, episode: _RootEpisode, reason: str) -> None:
        """End an episode at its budget/deadline/permanent bound."""
        if episode.driven:
            await self._emit_notice(
                episode,
                "exhausted",
                max(episode.admitted_count, episode.pending_attempt),
                message=(
                    "Automatic root recovery exhausted: "
                    + _blocked_message(reason, episode)
                    + " The failed turn is kept; the manual Retry affordance "
                    "(where offered) or a new message is the fallback."
                ),
                reason=reason,
            )
        self._end(episode)

    async def _cancel_episode(self, episode: _RootEpisode, reason: str) -> None:
        """Cancel a pending schedule and show why (never silently rearm)."""
        if episode.driven:
            await self._emit_notice(
                episode,
                "cancelled",
                max(episode.admitted_count, episode.pending_attempt),
                message=(
                    "Automatic root recovery cancelled: "
                    + _blocked_message(reason, episode)
                    + "."
                ),
                reason=reason,
            )
        self._end(episode)

    def _stage_notice(
        self,
        episode: _RootEpisode,
        milestone: Milestone,
        attempt: int,
        *,
        message: str,
        reason: str = "",
        successor_turn_id: str | None = None,
        predecessor_turn_id: str | None = None,
        eligible_class: str | None = None,
        affected_deployment: str | None = None,
        delay_seconds: float = 0.0,
    ) -> ProjectedUpdate | None:
        """Stage one milestone notice, exactly once per (milestone, attempt).

        The deterministic entry id makes the emission idempotent at both the
        gate (the episode's emitted set) and the projection layer; the entry
        rides the normal history, so it stays readable across a reconnect
        and is never injected as a fresh instruction.
        """
        notice_id = f"{milestone}:{attempt}"
        if notice_id in episode.emitted:
            return None
        episode.emitted.add(notice_id)
        detail = RootRecoveryNoticeDetail(
            milestone=milestone,
            episode_id=episode.episode_id,
            attempt=attempt,
            limit=episode.auto_retries_max,
            reason=reason,
            delay_seconds=delay_seconds,
            successor_turn_id=successor_turn_id,
            predecessor_turn_id=predecessor_turn_id,
            eligible_class=eligible_class,
            affected_deployment=affected_deployment,
        )
        try:
            return self._stage_notice_callback(detail, message)
        except Exception:
            # Meta-recovery: a notice failure is not a provider failure —
            # the decision stands; the milestone stays emitted-once.
            logger.exception("root recovery notice staging failed")
            return None

    async def _emit_notice(
        self,
        episode: _RootEpisode,
        milestone: Milestone,
        attempt: int,
        *,
        message: str,
        reason: str = "",
        successor_turn_id: str | None = None,
        predecessor_turn_id: str | None = None,
        eligible_class: str | None = None,
        affected_deployment: str | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        """Stage + deliver one milestone notice (async context)."""
        update = self._stage_notice(
            episode,
            milestone,
            attempt,
            message=message,
            reason=reason,
            successor_turn_id=successor_turn_id,
            predecessor_turn_id=predecessor_turn_id,
            eligible_class=eligible_class,
            affected_deployment=affected_deployment,
            delay_seconds=delay_seconds,
        )
        if update is None:
            return
        try:
            await self._emit_projected(update)
        except Exception:
            # The entry is already staged in the session history (durable
            # across reconnect); a lost notification is not a lost notice.
            logger.exception("root recovery notice delivery failed")

    def _declined_schedule(
        self,
        reason: str,
        *,
        episode: _RootEpisode | None = None,
        eligible_class: str | None = None,
        reason_detail: str = "",
    ) -> RootRecoveryScheduleResponse:
        return RootRecoveryScheduleResponse(
            decision="declined",
            reason=reason + (f": {reason_detail}" if reason_detail else ""),
            episode_id=episode.episode_id if episode is not None else None,
            limit=episode.auto_retries_max if episode is not None else 0,
            eligible_class=eligible_class,
        )

    def _declined_admit(self, reason: str) -> RootRecoveryAdmitResponse:
        return RootRecoveryAdmitResponse(decision="declined", reason=reason)


def _class_hint(failure: RootTurnFailure) -> str | None:
    return failure.failure_class.value if failure.failure_class else None


def _blocked_message(reason: str, episode: _RootEpisode) -> str:  # noqa: PLR0911
    """The honest blocked/exhausted reason line (one arm per reason)."""
    match reason:
        case "budget_exhausted":
            return (
                f"the automatic continuation budget is exhausted "
                f"({episode.admitted_count} of {episode.auto_retries_max} "
                "used)"
            )
        case "deadline_exceeded":
            return "the episode deadline passed before a continuation could start"
        case "permanent_incompatibility":
            return "every deployment of the model is permanently incompatible"
        case "not_eligible":
            return "the failure is not an eligible transient model-call failure"
        case "class_not_enabled":
            return "the failure class is not in the configured eligible classes"
        case "interaction_barrier":
            return "an approval or user question was still open"
        case "effects_unsettled":
            return "the failed turn's tool effects were not settled"
        case "transcript_integrity":
            return (
                "the session transcript failed message-order integrity "
                "checks; reload (resume) the session to apply the "
                "automatic transcript repair, or rewind to before the "
                "damage — retrying would fail the same way"
            )
        case _:
            return reason.replace("_", " ")
