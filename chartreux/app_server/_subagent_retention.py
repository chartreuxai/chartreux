from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
import itertools
from typing import TYPE_CHECKING, Any, Protocol, cast

from chartreux.app_server._subagent_records import (
    AgentRecord,
    RunRecord,
    _AgentState,
    _AgentTombstone,
)
from chartreux.app_server.models import TextContentBlock
from chartreux.app_server.protocol import TurnStartParams
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.subagents import (
    AgentAvailability,
    AgentEviction,
    AgentSummary,
    RetentionPolicy,
    RunStatus,
)
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime


class RetentionStateView(Protocol):
    """Narrow typed view over the registry-owned retention state.

    Dynamic access: every member is read through the view at call time, so
    the replaceable retention policy, the drain flag, and the tombstone,
    eviction-authority, record, and child maps are never snapshotted at
    construction — registry state mutated after the collaborator is built
    stays visible, and instance patches of the registry delegates (including
    the tombstone build/insert pair) still shadow.
    """

    _agent_records: dict[str, AgentRecord]
    _evicted_agents: dict[str, _AgentTombstone]
    _pending_evictions: dict[str, object]
    _children: dict[str, SessionRuntime]
    _child_links: dict[str, tuple[SessionRuntime, str]]
    _expired_results: set[tuple[str, str]]
    _draining_children: bool
    _retention_policy: RetentionPolicy
    _clock: Callable[[], float]
    _latest_run_ids: dict[str, str]
    _reaper_task: asyncio.Task[None] | None
    _reaper_deadline: float | None
    _eviction_tasks: set[asyncio.Task[bool]]
    _root: SessionRuntime | None
    _suppressed_notifications: set[str]
    _registry_lock: asyncio.Lock
    _detached_reaper_tasks: set[asyncio.Task[None]]
    _drain_suppressed_inventory: bool
    _wakeup: Callable[[float], Awaitable[None]]

    def _agent_turns_used(self, record: AgentRecord) -> int | None: ...

    def _run_elapsed_seconds(
        self, run: RunRecord | None, now: float
    ) -> float | None: ...

    @staticmethod
    def _record_name(record: AgentRecord) -> str: ...

    @staticmethod
    def _agent_id_sort_key(agent_id: str) -> int: ...

    def _track_child_close(
        self,
        runtime: SessionRuntime,
        cleanup: Coroutine[Any, Any, None] | None = None,
        *,
        name: str,
    ) -> asyncio.Task[None]: ...

    def _build_terminal_tombstone(
        self,
        record: AgentRecord,
        run: RunRecord | None,
        availability: AgentAvailability,
        idle_seconds: float,
    ) -> _AgentTombstone | None: ...

    def _insert_tombstone_locked(
        self, agent_id: str, tombstone: _AgentTombstone
    ) -> None: ...

    def _eligible_idle_locked(self) -> list[AgentRecord]: ...

    def _generation_is_current(self, generation: int) -> bool: ...

    def _rearm_reaper_locked(self, exclude: frozenset[str] = frozenset()) -> None: ...

    def _emit_agents_update(
        self, evictions: list[AgentEviction] | None = None
    ) -> Coroutine[Any, Any, None]: ...

    def _evict_agent(
        self,
        agent_id: str,
        reason: str,
        notifications: list[EvictionNotice] | None = None,
    ) -> Coroutine[Any, Any, bool]: ...

    def _run_reaper(self, delay: float) -> Coroutine[Any, Any, None]: ...

    def _prune_expired_tombstones_locked(self, now: float) -> list[str]: ...

    def _suspension_next_expiry_locked(self) -> float | None: ...

    def _suspension_expire_elapsed_locked(self, now: float) -> list[AgentRecord]: ...


@dataclass(frozen=True)
class EvictionNotice:
    """Immutable eviction-notification context (ASC area 5).

    Replaces the historical four-element
    ``(record, eviction, tombstone, authority)`` tuple: the exact record,
    the eviction description, the optional terminal tombstone, and the exact
    release-invalidatable authority token travel as named fields.
    """

    record: AgentRecord
    eviction: AgentEviction
    tombstone: _AgentTombstone | None
    authority: object


class AgentRetention:
    """Tombstone policy, eviction, and reaper scheduling for the registry.

    Narrow collaborator of the session runtime registry (ASC area 5,
    migration steps 1–4): tombstone construction/insertion/lookup/pruning,
    the close-on-completion decision, eviction with typed
    ``EvictionNotice`` contexts, eviction-notification authorization, and
    idle/metadata reaper scheduling (rearm/detach/run) operate on the
    registry-owned tombstone map, pending-eviction authority, resident
    records, reaper fields, detached reaper tasks, eviction tasks, and the
    replaceable retention policy through the registry-owned typed view.
    Every locked operation requires the caller to hold the registry's own
    lock at the same points; the public eviction and the reaper acquire it
    at the same points. Canonical state stays registry-owned (§6.7); no
    private collaborator locks; no new tasks/queues/relay layers (the
    reaper creates its timer and eviction tasks through the registry
    delegates). The registry keeps thin same-name delegates so existing
    instance patches and direct calls keep working.
    """

    def __init__(self, view: RetentionStateView) -> None:
        self._view = view

    def build_terminal(
        self,
        record: AgentRecord,
        run: RunRecord | None,
        availability: AgentAvailability,
        idle_seconds: float,
    ) -> _AgentTombstone | None:
        """Construct terminal metadata shared by close-on-completion and eviction.

        The summary freezes the record/run fields at insertion time.
        ``expires_at`` is ``closed_at + tombstone_ttl_seconds``. Returns
        ``None`` when the tombstone TTL is zero: no terminal metadata is
        retained at all, and the handle behaves as metadata-expired/unknown
        once the record detaches (the result stays published and
        retrievable).

        WP1 E4 #2 (carried into tombstones): the summary carries the
        latest failed run's public typed failure envelope — the same
        latest-first search the live summary uses — so a closed (or
        expired-suspension) agent's terminal metadata still shows WHY the
        run failed. The data is on the record at build time and the wire
        (``AgentSummary.last_failure``) already supports it.
        """
        agent_id = record.agent_id
        latest_run_id = record.latest_run_id or (
            run.run_id if run is not None else None
        )
        now = self._view._clock()
        ttl = self._view._retention_policy.tombstone_ttl_seconds
        if ttl <= 0:
            return None
        # The latest failed run's envelope, searched latest-first (the
        # terminal run, then the current run, then run history) — mirroring
        # the live summary projection in the registry.
        last_failure = None
        for candidate in (run, record.current_run, *reversed(record.run_history)):
            if candidate is not None and candidate.failure is not None:
                last_failure = candidate.failure
                break
        summary = AgentSummary(
            agent_id=agent_id,
            profile=record.profile,
            availability=availability,
            current_run_id=run.run_id if run is not None else None,
            current_run_status=run.status if run is not None else None,
            turns_used=self._view._agent_turns_used(record),
            last_run_status=record.last_run_status,
            initial_task_summary=record.initial_task_summary,
            current_task_summary=run.task_summary if run is not None else None,
            idle_seconds=idle_seconds,
            latest_run_id=latest_run_id,
            run_elapsed_seconds=self._view._run_elapsed_seconds(run, now),
            effective_model=record.effective_model,
            base_model=record.base_model,
            active_provider=record.active_provider,
            effective_thinking=record.effective_thinking,
            context_tokens=record.context_tokens,
            context_window=record.context_window,
            compacting=record.compacting,
            stop_reason=record.stop_reason,
            last_failure=last_failure,
            result_expired=(
                (agent_id, latest_run_id) in self._view._expired_results
                if latest_run_id is not None
                else False
            ),
            name=self._view._record_name(record),
        )
        return _AgentTombstone(
            summary=summary,
            child_session_id=record.session_id,
            parent_identity=(record.parent_session_id, record.root_generation),
            reason="closed" if availability is AgentAvailability.CLOSED else "evicted",
            closed_at=now,
            expires_at=now + ttl,
            profile_display_label=record.profile_display_label,
        )

    def insert_locked(self, agent_id: str, tombstone: _AgentTombstone) -> None:
        """Store terminal metadata and enforce the oldest-first count cap."""
        self._view._evicted_agents[agent_id] = tombstone
        cap = self._view._retention_policy.max_tombstones
        if cap <= 0:
            return
        while len(self._view._evicted_agents) > cap:
            oldest_id = min(
                self._view._evicted_agents,
                key=lambda aid: (
                    self._view._evicted_agents[aid].closed_at,
                    self._view._agent_id_sort_key(aid),
                ),
            )
            self._view._evicted_agents.pop(oldest_id, None)
            # The capped tombstone's pending-eviction authority settles with
            # it: the eviction it authorized can no longer notify once its
            # metadata is gone, so the entry must not outlive the tombstone.
            self._view._pending_evictions.pop(oldest_id, None)

    @staticmethod
    def tombstone_expired(tombstone: _AgentTombstone, now: float) -> bool:
        return tombstone.expires_at is not None and now >= tombstone.expires_at

    def live_tombstone_locked(
        self, agent_id: str, now: float
    ) -> _AgentTombstone | None:
        """Return the tombstone for ``agent_id`` unless its metadata has expired."""
        tombstone = self._view._evicted_agents.get(agent_id)
        if tombstone is not None and not self.tombstone_expired(tombstone, now):
            return tombstone
        return None

    def prune_locked(self, now: float) -> list[str]:
        """Drop expired terminal metadata only (results, markers, leases survive).

        Returns the pruned agent ids so callers can publish the reduced
        inventory.
        """
        expired = [
            agent_id
            for agent_id, tombstone in self._view._evicted_agents.items()
            if self.tombstone_expired(tombstone, now)
        ]
        for agent_id in expired:
            self._view._evicted_agents.pop(agent_id, None)
            self._view._pending_evictions.pop(agent_id, None)
        return expired

    def close_unretained_locked(
        self, record: AgentRecord, run: RunRecord | None, *, name: str
    ) -> bool:
        """Detach and close a settled resident record whose policy is unretained.

        Stores a ``closed`` tombstone (none at all when the tombstone TTL is
        zero) and joins the runtime close through the tracked teardown set.
        Returns ``False`` without acting when the record is not a settled
        resident (draining, detached, reserved, retained, or still running).
        """
        agent_id = record.agent_id
        if self._view._draining_children:
            return False
        if self._view._agent_records.get(agent_id) is not record:
            return False
        if record.reserved_for_replacement or record.retained:
            return False
        active = record.current_run
        if active is not None and active.status is RunStatus.RUNNING:
            return False
        record.state = _AgentState.EVICTING
        self._view._agent_records.pop(agent_id, None)
        self._view._children.pop(record.session_id, None)
        self._view._child_links.pop(record.session_id, None)
        tombstone = self._view._build_terminal_tombstone(
            record, run, AgentAvailability.CLOSED, idle_seconds=0.0
        )
        if tombstone is not None:
            self._view._insert_tombstone_locked(agent_id, tombstone)
        self._view._track_child_close(record.runtime, name=name)
        return True

    async def evict(
        self,
        agent_id: str,
        reason: str,
        notifications: list[EvictionNotice] | None = None,
    ) -> bool:
        """Evict a settled idle record and authorize its notification.

        Revalidates eligibility under the registry lock, stores the
        release-invalidatable pending-eviction authority, builds and
        inserts the tombstone, and rearms the reaper when required. The
        awaited runtime cleanup happens outside the lock, and the schedule
        handoff (rearm) occurs BEFORE waiting for eviction cleanup. Returns
        ``True`` when the record was evicted.
        """
        if reason not in {"ttl", "idle_cap"}:
            raise ValueError(f"Unknown eviction reason: {reason}")
        async with self._view._registry_lock:
            record = self._view._agent_records.get(agent_id)
            if (
                record is None
                or record.state is not _AgentState.IDLE
                or record.reserved_for_replacement
                or not self._view._generation_is_current(record.root_generation)
            ):
                return False
            cap = self._view._retention_policy.max_idle_agents
            if reason == "idle_cap" and (
                not cap or len(self._view._eligible_idle_locked()) <= cap
            ):
                return False
            record.state = _AgentState.EVICTING
            self._view._agent_records.pop(agent_id)
            self._view._children.pop(record.session_id, None)
            self._view._child_links.pop(record.session_id, None)
            # Release-invalidatable authority for this pending eviction. A
            # TTL>0 eviction is authorized by the stored tombstone; a zero-TTL
            # eviction stores no metadata, so this entry is what authorizes
            # (and release revokes) its notification.
            authority = object()
            self._view._pending_evictions[agent_id] = authority
            run = record.current_run or (
                record.run_history[-1] if record.run_history else None
            )
            latest_run_id = record.latest_run_id or (
                run.run_id if run is not None else None
            )
            if latest_run_id is not None:
                self._view._latest_run_ids[agent_id] = latest_run_id
            idle_duration = max(
                0.0,
                self._view._clock()
                - (
                    record.idle_since
                    if record.idle_since is not None
                    else self._view._clock()
                ),
            )
            tombstone = self._view._build_terminal_tombstone(
                record, run, AgentAvailability.EVICTED, idle_duration
            )
            if tombstone is not None:
                self._view._insert_tombstone_locked(agent_id, tombstone)
                # Rearm promptly so tombstone expiry is scheduled even when
                # the awaited runtime cleanup below stalls. A reaper-driven
                # eviction skips this while its coordinating reaper is still
                # registered (that reaper rearms when it finishes; cancelling
                # it here would abort the in-flight run) — but after a
                # schedule handoff (no reaper registered), or when this
                # tombstone expires before the armed timer, the rearm is
                # safe and required.
                if (
                    asyncio.current_task() not in self._view._eviction_tasks
                    or self._view._reaper_task is None
                    or (
                        tombstone.expires_at is not None
                        and (
                            self._view._reaper_deadline is None
                            or tombstone.expires_at < self._view._reaper_deadline
                        )
                    )
                ):
                    self._view._rearm_reaper_locked()
        teardown = self._view._track_child_close(
            record.runtime, name=f"vibe-subagent-teardown:{agent_id}"
        )
        teardown_result = await asyncio.shield(
            asyncio.gather(teardown, return_exceptions=True)
        )
        if teardown_result and isinstance(teardown_result[0], BaseException):
            logger.warning(
                "Failed to close evicted background agent agent_id=%s",
                agent_id,
                exc_info=teardown_result[0],
            )
        eviction = AgentEviction(
            agent_id=agent_id,
            run_id=run.run_id if run is not None else "",
            reason=reason,  # type: ignore[arg-type]
            idle_duration_seconds=idle_duration,
            root_generation=record.root_generation,
        )
        notice = EvictionNotice(
            record=record, eviction=eviction, tombstone=tombstone, authority=authority
        )
        if notifications is not None:
            notifications.append(notice)
        else:
            await self.notify_evictions([notice])
        return True

    @staticmethod
    def eviction_run_segments(
        record: AgentRecord, eviction: AgentEviction
    ) -> tuple[str, str]:
        """The latest-run identity and outcome segments for an eviction notice.

        Design §5: eviction notifications carry the name alongside the handle,
        run, and outcome. Both segments stay empty when the evicted record has
        no run matching the eviction's run id, so a run-less eviction keeps its
        handle-only wording.
        """
        run = next(
            (
                candidate
                for candidate in (record.current_run, *record.run_history)
                if candidate is not None and candidate.run_id == eviction.run_id
            ),
            None,
        )
        if run is None:
            return "", ""
        outcome = (
            "completed"
            if run.status is RunStatus.COMPLETED
            else "failed"
            if run.status is RunStatus.FAILED
            else "cancelled"
            if run.status is RunStatus.CANCELLED
            else "unknown"
        )
        return f", run {run.run_id}", f", last run {outcome}"

    async def notify_evictions(self, notifications: list[EvictionNotice]) -> bool:
        """Authorize and deliver eviction notifications.

        Returns ``True`` when an inventory update was published. A TTL>0
        eviction is authorized only while the tombstone it built is still
        stored; a zero-TTL eviction stores no metadata, so it is authorized
        only while its release-invalidatable authority entry survives. The
        authority entry is consumed by this check on every settlement path
        (both TTL variants): a settled eviction notification never leaves
        the entry behind.
        """
        permitted: list[tuple[AgentRecord, AgentEviction]] = []
        async with self._view._registry_lock:
            for notice in sorted(
                notifications, key=lambda notice: notice.eviction.agent_id
            ):
                # Consume the release-invalidatable authority on every
                # settlement: this notification attempt settles the pending
                # entry for both TTL variants, so a delivered (or fenced)
                # eviction never leaves its authority behind.
                stored_authority = self._view._pending_evictions.pop(
                    notice.record.agent_id, None
                )
                if notice.tombstone is not None:
                    authorized = (
                        self._view._evicted_agents.get(notice.record.agent_id)
                        is notice.tombstone
                    )
                else:
                    authorized = stored_authority is notice.authority
                if (
                    self._view._generation_is_current(notice.record.root_generation)
                    and not self._view._draining_children
                    and notice.record.agent_id
                    not in self._view._suppressed_notifications
                    and authorized
                ):
                    permitted.append((notice.record, notice.eviction))

            for parent_session_id, parent_notifications in itertools.groupby(
                sorted(
                    permitted,
                    key=lambda notification: notification[0].parent_session_id,
                ),
                key=lambda notification: notification[0].parent_session_id,
            ):
                root = self._view._root
                parent = (
                    root
                    if root is not None
                    and root.agent_loop.session_id == parent_session_id
                    else self._view._children.get(parent_session_id)
                )
                if parent is None:
                    continue
                parent_notifications = list(parent_notifications)
                detailed = [
                    (record, eviction, *self.eviction_run_segments(record, eviction))
                    for record, eviction in parent_notifications
                ]
                details = "; ".join(
                    f"{self._view._record_name(record)} "
                    f"({eviction.agent_id}{run_segment}, "
                    f"{eviction.reason}, idle {eviction.idle_duration_seconds:g}s"
                    f"{outcome_segment})"
                    for record, eviction, run_segment, outcome_segment in detailed
                )
                if len(detailed) == 1:
                    record, eviction, run_segment, outcome_segment = detailed[0]
                    notification = (
                        f"Background agent {self._view._record_name(record)} "
                        f"({eviction.agent_id}{run_segment}) evicted "
                        f"({eviction.reason}, "
                        f"idle {eviction.idle_duration_seconds:g}s"
                        f"{outcome_segment}); "
                        "launch a new agent for further work."
                    )
                else:
                    notification = (
                        "Background agents evicted: "
                        f"{details}; launch a new agent for further work."
                    )
                try:
                    if (
                        parent.turns.active_turn is None
                        and parent.turns._active_task is None
                        and not parent.turns.queue_state.items
                    ):
                        try:
                            _response, start = parent.turns.start(
                                TurnStartParams(
                                    session_id=parent.agent_loop.session_id,
                                    message=[TextContentBlock(text=notification)],
                                )
                            )
                            start()
                        except Exception:
                            parent.agent_loop._pending_injected_messages.append(
                                LLMMessage(
                                    role=Role.user, content=notification, injected=True
                                )
                            )
                    else:
                        parent.agent_loop._pending_injected_messages.append(
                            LLMMessage(
                                role=Role.user, content=notification, injected=True
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to notify parent of background agent eviction",
                        exc_info=exc,
                    )
        if permitted:
            try:
                await self._view._emit_agents_update([
                    eviction for _record, eviction in permitted
                ])
            except Exception as exc:
                logger.warning(
                    "Failed to publish background agent eviction", exc_info=exc
                )
                return False
            return True
        return False

    def rearm_locked(self, exclude: frozenset[str] = frozenset()) -> None:
        """(Re)arm the idle/metadata reaper timer; the caller holds the lock.

        A live reaper is displaced — registered drain-owned and cancelled —
        so every displaced reaper stays drain-owned until settlement,
        INCLUDING ordinary rearm displacement. No rearm happens while
        draining or when the caller is the reaper task itself.
        """
        if self._view._draining_children:
            return
        current = asyncio.current_task()
        if self._view._reaper_task is current:
            return
        # Replacing a live reaper displaces it: register the displaced task
        # as drain-owned (its finalizer discards it when it settles) so
        # drain_children cannot return while it can still publish.
        displaced = self.detach_reaper_locked()
        if displaced is not None:
            displaced.cancel()
        self._view._reaper_deadline = None
        policy = self._view._retention_policy
        now = self._view._clock()
        delays: list[float] = []
        eligible = [
            record
            for record in self._view._eligible_idle_locked()
            if record.agent_id not in exclude
        ]
        ttl_eligible = [
            record
            for record in eligible
            if record.effective_idle_ttl(policy.idle_ttl_seconds) > 0
        ]
        if policy.max_idle_agents and len(eligible) > policy.max_idle_agents:
            delays.append(0.0)
        elif ttl_eligible:
            delays.append(
                max(
                    0.0,
                    min(
                        cast(float, record.idle_since)
                        + record.effective_idle_ttl(policy.idle_ttl_seconds)
                        for record in ttl_eligible
                    )
                    - now,
                )
            )
        if policy.tombstone_ttl_seconds > 0:
            expiries = [
                tombstone.expires_at
                for tombstone in self._view._evicted_agents.values()
                if tombstone.expires_at is not None
            ]
            if expiries:
                delays.append(max(0.0, min(expiries) - now))
        # WP3: the warm-suspension TTL is a SEPARATE reaper deadline — the
        # earliest suspension expiry arms the shared timer too, so suspension
        # expiry runs on the reaper without any parent model turn.
        suspension_deadline = self._view._suspension_next_expiry_locked()
        if suspension_deadline is not None:
            delays.append(max(0.0, suspension_deadline - now))
        if not delays:
            return
        delay = min(delays)
        self._view._reaper_task = asyncio.create_task(
            self._view._run_reaper(delay), name="vibe-subagent-idle-reaper"
        )
        self._view._reaper_deadline = now + delay

    async def publish_inventory_change(self) -> None:
        """Publish the current (possibly reduced) inventory after reaping."""
        if self._view._draining_children:
            # Drain owns the final inventory publication: a reaper settling
            # mid-drain must not publish a transient pre-drain snapshot on
            # top of the empty payload drain is about to emit. The
            # suppression is recorded so drain still emits the final empty
            # inventory — the disappearance must not be lost at the boundary.
            self._view._drain_suppressed_inventory = True
            return
        try:
            await self._view._emit_agents_update()
        except Exception as exc:
            logger.warning("Failed to publish tombstone expiry update", exc_info=exc)

    def detach_reaper_locked(self) -> asyncio.Task[None] | None:
        """Unregister the live reaper, keeping drain ownership of its task.

        The caller holds the registry lock. The displaced task is
        registered in ``_detached_reaper_tasks`` so drain_children
        joins/cancels it like any other tracked task instead of returning
        while it can still publish; the displaced task's finalizer discards
        it when it settles (cancel, normal completion, or exception).
        Returns the displaced task (``None`` when no reaper is registered).
        """
        displaced = self._view._reaper_task
        self._view._reaper_task = None
        if displaced is not None:
            self._view._detached_reaper_tasks.add(displaced)
        return displaced

    async def run_reaper(self, delay: float) -> None:  # noqa: PLR0915
        """Run one reaper pass: prune expired metadata, evict victims, publish.

        The schedule handoff (detach + rearm) occurs BEFORE awaiting
        eviction cleanup; the reaper invocation state (victims, pruned ids,
        notices, publication disposition) never replaces the authoritative
        shared timer fields. Victim ordering uses the existing record
        ordering (idle_since, agent_id). WP3: the pass also expires
        suspensions whose warm window elapsed (handoff §1.5).
        """
        pruned_tombstones: list[str] = []
        expired_suspensions: list[AgentRecord] = []
        try:
            await self._view._wakeup(delay)
            async with self._view._registry_lock:
                if asyncio.current_task() is not self._view._reaper_task:
                    return
                now = self._view._clock()
                policy = self._view._retention_policy
                eligible = sorted(
                    self._view._eligible_idle_locked(),
                    key=lambda record: (record.idle_since, record.agent_id),
                )
                victims: list[tuple[str, str]] = []
                victims.extend(
                    (record.agent_id, "ttl")
                    for record in eligible
                    if (
                        now - cast(float, record.idle_since)
                        >= record.effective_idle_ttl(policy.idle_ttl_seconds)
                        > 0
                    )
                )
                selected = {agent_id for agent_id, _ in victims}
                remaining = [r for r in eligible if r.agent_id not in selected]
                if policy.max_idle_agents and len(remaining) > policy.max_idle_agents:
                    victims.extend(
                        (record.agent_id, "idle_cap")
                        for record in remaining[
                            : len(remaining) - policy.max_idle_agents
                        ]
                    )
                # Expired terminal metadata is pruned here; results, expiry
                # markers, latest-run indexes, and wait leases are untouched.
                pruned_tombstones = self._view._prune_expired_tombstones_locked(now)
                # WP3: suspension records whose warm window elapsed convert
                # to closed terminal tombstones here — eligibility removed,
                # transcripts/artifacts/stored results untouched, runtime
                # closed through the normal ownership paths. This runs on
                # the reaper's timer, without any parent model turn.
                expired_suspensions = self._view._suspension_expire_elapsed_locked(now)
                if victims:
                    # Hand the schedule to a fresh timer BEFORE awaiting
                    # eviction cleanup: stalled runtime cleanup must not hold
                    # idle/metadata deadline scheduling hostage. This task
                    # unregisters itself (no self-cancellation), so its
                    # finalizer will not rearm; the fresh timer owns the
                    # schedule from here. The victims are excluded because
                    # their idle deadlines are moot once evicted. The
                    # displaced task is registered as drain-owned so
                    # drain_children joins/cancels it like any other tracked
                    # task instead of returning while it can still publish.
                    self.detach_reaper_locked()
                    self._view._rearm_reaper_locked(
                        exclude=frozenset(agent_id for agent_id, _ in victims)
                    )
            eviction_notifications: list[EvictionNotice] = []
            eviction_tasks = [
                asyncio.create_task(
                    self._view._evict_agent(agent_id, reason, eviction_notifications),
                    name=f"vibe-subagent-eviction:{agent_id}",
                )
                for agent_id, reason in victims
            ]
            self._view._eviction_tasks.update(eviction_tasks)
            for task in eviction_tasks:
                task.add_done_callback(self._view._eviction_tasks.discard)
            published = False
            if eviction_tasks:
                eviction_results = asyncio.gather(
                    *eviction_tasks, return_exceptions=True
                )
                try:
                    await asyncio.shield(eviction_results)
                except asyncio.CancelledError:
                    await asyncio.shield(eviction_results)
                    published = await self.notify_evictions(eviction_notifications)
                    if not published and (
                        pruned_tombstones
                        or eviction_notifications
                        or expired_suspensions
                    ):
                        await self.publish_inventory_change()
                    raise
                published = await self.notify_evictions(eviction_notifications)
            # Publish whenever the inventory actually changed — pruned
            # terminal metadata, actual evictions, and/or expired suspensions
            # — regardless of whether eviction tasks existed or what they
            # did. Permitted eviction notifications already published this
            # same inventory.
            if not published and (
                pruned_tombstones or eviction_notifications or expired_suspensions
            ):
                await self.publish_inventory_change()
        finally:
            # A pending reaper can be destroyed while no event loop is
            # running (teardown or interpreter shutdown); the rearm
            # bookkeeping is meaningless there and must stay silent. With a
            # running loop, rearm failures propagate normally.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop: nothing to rearm and nowhere to report.
                pass
            else:
                # This reaper is settling — handoff-displaced or
                # rearm-displaced: drain ownership ends with the task, on
                # every settle path (cancel, normal completion, exception).
                current = asyncio.current_task()
                if current is not None:
                    self._view._detached_reaper_tasks.discard(current)
                async with self._view._registry_lock:
                    if asyncio.current_task() is self._view._reaper_task:
                        self._view._reaper_task = None
                        self._view._rearm_reaper_locked()
