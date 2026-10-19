"""Bounded warm suspension lifecycle for subagents (WP3; handoff §1.5/§1.8).

v1 is WARM-ONLY (resolved Q6): a recoverably failed agent SUSPENDS — the
live conversation stays resident (warm) while its provider connection is
released — for a separately bounded window, and a parent-process restart
forfeits suspended agents (they convert to closed/non-resumable; scratch
artifacts remain the salvage path). This module owns the suspension half
of the bounded lifecycle:

- The suspension TTL (``recovery.suspension_ttl_seconds``): the per-record
  warm window. The shared idle/metadata reaper arms for the earliest
  suspension deadline and expires elapsed records — the expiry runs on the
  reaper's timer, WITHOUT any parent model turn, and publishes the reduced
  inventory through the ``agents/update`` notification only.
- The suspension byte cap (``recovery.max_suspended_bytes``): a separate
  total bound on the deterministic warm-conversation byte accounting across
  suspended records — independent of the idle-retention and tombstone caps.
  Deterministic cap selection: the OLDEST suspended record (suspended_at,
  then the numeric agent-id tie-break) expires first to make room, and a
  record whose conversation alone exceeds the cap is never suspended.
- Expiry converts a suspended record to a ``closed`` terminal tombstone
  through the normal ownership paths (tracked runtime close): it removes
  recovery eligibility WITHOUT deleting transcripts, artifacts, or stored
  results.
- The internal storage seam: ``SuspensionStorage`` is the interface a
  post-v1 cold path would implement. The v1 default
  (``WarmOnlySuspensionStorage``) persists NOTHING — there is no checkpoint
  writer, durable scheduler, restoration lease, or cold path in v1.

Safety invariants (code, never config escape hacks):

- Expiry only selects records in the SUSPENDED state — a RESTORING record
  is mid-admission and is never expired underneath its own admission.
- Every conversion invalidates the record's pending recovery first; the
  failed run's stored result, the on-disk transcript, and the workspace
  artifacts survive.
- The byte accounting is deterministic (same conversation -> same number),
  computed once per suspension, and bounded by the configured total cap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol, cast

from chartreux.app_server._subagent_records import (
    AgentRecord,
    _AgentState,
    _AgentTombstone,
)
from chartreux.core.config.models import RecoveryConfig
from chartreux.core.llm_models import LLMMessage
from chartreux.core.subagents import AgentAvailability
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime

# Fixed per-message accounting constant covering envelope/metadata overhead
# not itemized per field below (role plumbing, ids, timestamps).
_MESSAGE_OVERHEAD_BYTES = 64


class SuspensionStorage(Protocol):
    """The internal suspension storage seam (WP3; handoff §1.5 cold note).

    This protocol is the ONLY seam a post-v1 cold suspension path would
    implement: capture a suspended record's warm state, restore it, and
    discard it. v1 ships ``WarmOnlySuspensionStorage`` — nothing is
    persisted, so a parent-process restart forfeits suspended agents. The
    seam is deliberately NOT a checkpoint writer, durable scheduler, or
    restoration lease, and nothing in v1 schedules through it.
    """

    def capture(self, record: AgentRecord) -> None:
        """Persist one suspended record's warm state (v1: no-op)."""
        ...

    def restore(self, agent_id: str) -> AgentRecord | None:
        """Return a previously captured record (v1: always None)."""
        ...

    def discard(self, agent_id: str) -> None:
        """Drop any captured state for one agent (v1: no-op)."""
        ...


class WarmOnlySuspensionStorage:
    """The v1 default storage: warm-only — nothing survives a restart.

    ``capture``/``discard`` are deliberate no-ops and ``restore`` always
    reports absent, which is what converts a suspended agent to
    closed/non-resumable across a parent-process restart (Q6).
    """

    def capture(self, record: AgentRecord) -> None:
        return None

    def restore(self, agent_id: str) -> AgentRecord | None:
        return None

    def discard(self, agent_id: str) -> None:
        return None


def _warm_conversation_bytes(messages: Iterable[LLMMessage]) -> int:
    """Deterministic estimated bytes of one live conversation.

    The accounting sums a fixed per-message overhead plus the itemized
    payload fields (role, text content, reasoning content, tool-call ids,
    names, and arguments, tool result ids). It is an ESTIMATE for cap
    accounting — deterministic (same conversation, same number), computed
    once per suspension, and bounded by the configured total cap — not a
    byte-exact serialization of the transcript.
    """
    total = 0
    for message in messages:
        total += _MESSAGE_OVERHEAD_BYTES
        total += len(str(message.role))
        content = message.content
        if isinstance(content, str):
            total += len(content)
        reasoning = message.reasoning_content
        if isinstance(reasoning, str):
            total += len(reasoning)
        for tool_call in message.tool_calls or []:
            if tool_call.id is not None:
                total += len(tool_call.id)
            function = tool_call.function
            if function.name is not None:
                total += len(function.name)
            if function.arguments is not None:
                total += len(function.arguments)
        if message.tool_call_id is not None:
            total += len(message.tool_call_id)
        if message.name is not None:
            total += len(message.name)
    return total


class SuspensionStateView(Protocol):
    """Narrow typed view over the registry-owned suspension state.

    Dynamic access: every member is read through the view at call time, so
    the record/child maps, the tombstone and latest-run maps, the shared
    registry lock, the clock, the drain flag, and the tombstone/close
    delegates are never snapshotted at construction — registry state
    mutated after the collaborator is built stays visible, and instance
    patches of the registry delegates still shadow.
    """

    _agent_records: dict[str, AgentRecord]
    _children: dict[str, SessionRuntime]
    _child_links: dict[str, tuple[SessionRuntime, str]]
    _evicted_agents: dict[str, _AgentTombstone]
    _latest_run_ids: dict[str, str]
    _draining_children: bool

    def _build_terminal_tombstone(
        self,
        record: AgentRecord,
        run: Any,
        availability: AgentAvailability,
        idle_seconds: float,
    ) -> _AgentTombstone | None: ...

    def _insert_tombstone_locked(
        self, agent_id: str, tombstone: _AgentTombstone
    ) -> None: ...

    def _track_child_close(
        self, runtime: SessionRuntime, cleanup: Any = None, *, name: str
    ) -> asyncio.Task[None]: ...

    @staticmethod
    def _agent_id_sort_key(agent_id: str) -> int: ...


class SuspensionLifecycle:
    """Warm-suspension caps, byte accounting, and expiry for the registry.

    Narrow collaborator of the session runtime registry (WP3): the
    suspension byte-cap admission check (deterministic oldest-first victim
    selection), the TTL-expiry selection driving the shared reaper pass,
    and the expiry conversion (suspended -> closed terminal tombstone
    through the normal ownership paths) operate on the registry-owned
    record/child/tombstone maps, latest-run index, and clock through the
    registry-owned typed view. Every locked operation requires the caller
    to hold the registry's own lock at the same points. Canonical state
    stays registry-owned; the collaborator holds no private locks and
    creates no independent task collections (the conversion's runtime
    close goes through the registry-owned tracked-close delegate).
    """

    def __init__(
        self, view: SuspensionStateView, store: SuspensionStorage | None = None
    ) -> None:
        self._view = view
        self._store: SuspensionStorage = store or WarmOnlySuspensionStorage()

    # ------------------------------------------------------------------
    # Byte accounting (deterministic, bounded)
    # ------------------------------------------------------------------

    @staticmethod
    def warm_bytes(record: AgentRecord) -> int:
        """The deterministic warm-conversation byte estimate of a record."""
        return _warm_conversation_bytes(record.runtime.agent_loop.messages)

    def suspended_total_bytes_locked(self) -> int:
        """Total warm bytes held by suspended/restoring records (locked)."""
        return sum(
            record.suspended_warm_bytes or 0
            for record in self._view._agent_records.values()
            if record.state in {_AgentState.SUSPENDED, _AgentState.RESTORING}
        )

    def admit_locked(self, record: AgentRecord, config: RecoveryConfig) -> bool:
        """Byte-cap admission check for one would-be suspension (locked).

        Captures the record's deterministic warm-conversation bytes and
        enforces the separate suspension byte cap: a record whose
        conversation alone exceeds the cap is refused (never suspended),
        and otherwise the OLDEST suspended records expire — deterministic
        (suspended_at, then numeric agent-id tie-break) — until the new
        record fits. Returns True (with ``suspended_warm_bytes`` captured)
        when the record may suspend. A zero cap disables byte accounting.
        """
        needed = self.warm_bytes(record)
        cap = config.max_suspended_bytes
        if cap > 0:
            if needed > cap:
                return False
            while self.suspended_total_bytes_locked() + needed > cap:
                victim = self._oldest_suspended_locked(
                    exclude=frozenset({record.agent_id})
                )
                if victim is None:
                    return False
                self.expire_locked(victim)
        record.suspended_warm_bytes = needed
        return True

    def capture_locked(self, record: AgentRecord) -> None:
        """Hand one suspended record to the storage seam (locked; v1 no-op)."""
        try:
            self._store.capture(record)
        except Exception as exc:
            # The seam must never break the suspension transition; v1's
            # default store cannot fail, and a future cold path owns its
            # own error handling.
            logger.warning(
                "Suspension storage capture failed for agent_id=%s",
                record.agent_id,
                exc_info=exc,
            )

    def discard_locked(self, agent_id: str) -> None:
        """Drop the storage-seam state for one agent (locked; v1 no-op)."""
        try:
            self._store.discard(agent_id)
        except Exception as exc:
            logger.warning(
                "Suspension storage discard failed for agent_id=%s",
                agent_id,
                exc_info=exc,
            )

    # ------------------------------------------------------------------
    # TTL expiry (driven by the shared reaper; no parent model turn)
    # ------------------------------------------------------------------

    def next_expiry_locked(self, config: RecoveryConfig) -> float | None:
        """The earliest suspension-TTL deadline, or None (locked).

        Only SUSPENDED records with a captured suspension timestamp count;
        a nonpositive TTL disables TTL expiry (and suspension entirely, at
        the suspension hook).
        """
        suspended = [
            record
            for record in self._view._agent_records.values()
            if record.state is _AgentState.SUSPENDED and record.suspended_at is not None
        ]
        if not suspended:
            return None
        ttl = config.suspension_ttl_seconds
        if ttl <= 0:
            return None
        return min(cast(float, record.suspended_at) + ttl for record in suspended)

    def expire_elapsed_locked(
        self, now: float, config: RecoveryConfig
    ) -> list[AgentRecord]:
        """Expire every suspension whose warm window elapsed (locked).

        Returns the expired records so the caller can publish the reduced
        inventory. Selection is deterministic: all SUSPENDED records whose
        ``suspended_at + suspension_ttl_seconds`` has passed, in record
        order. The config is read only when suspended records exist, so a
        registry without suspensions never dereferences it.
        """
        suspended = [
            record
            for record in self._view._agent_records.values()
            if record.state is _AgentState.SUSPENDED and record.suspended_at is not None
        ]
        if not suspended:
            return []
        ttl = config.suspension_ttl_seconds
        if ttl <= 0:
            return []
        elapsed = [
            record
            for record in suspended
            if now >= cast(float, record.suspended_at) + ttl
        ]
        for record in elapsed:
            self.expire_locked(record)
        return elapsed

    # ------------------------------------------------------------------
    # Expiry conversion (shared by TTL expiry, byte-cap victims, and
    # cancellation invalidation)
    # ------------------------------------------------------------------

    def expire_locked(self, record: AgentRecord) -> _AgentTombstone | None:
        """Convert one suspended record to a closed terminal tombstone (locked).

        The conversion invalidates the record's pending recovery, detaches
        the record and its child maps, stores a ``closed`` tombstone (which
        carries the failed run's public failure envelope), and closes the
        warm runtime through the registry-owned tracked-close delegate —
        the normal ownership path, exactly like close-on-completion. The
        failed run's stored result, the on-disk transcript, and workspace
        artifacts are untouched: expiry removes recovery eligibility only.
        """
        view = self._view
        agent_id = record.agent_id
        record.pending_resume = None
        record.recovery_superseded = True
        record.state = _AgentState.EVICTING
        record.suspended_warm_bytes = None
        view._agent_records.pop(agent_id, None)
        view._children.pop(record.session_id, None)
        view._child_links.pop(record.session_id, None)
        run = record.current_run or (
            record.run_history[-1] if record.run_history else None
        )
        latest_run_id = record.latest_run_id or (
            run.run_id if run is not None else None
        )
        if latest_run_id is not None:
            view._latest_run_ids[agent_id] = latest_run_id
        tombstone = view._build_terminal_tombstone(
            record, run, AgentAvailability.CLOSED, idle_seconds=0.0
        )
        if tombstone is not None:
            view._insert_tombstone_locked(agent_id, tombstone)
        view._track_child_close(
            record.runtime, name=f"vibe-subagent-suspension-expiry:{agent_id}"
        )
        self.discard_locked(agent_id)
        return tombstone

    def _oldest_suspended_locked(
        self, *, exclude: frozenset[str] = frozenset()
    ) -> AgentRecord | None:
        """The deterministic byte-cap victim: oldest suspended record.

        Ordering is ``(suspended_at, numeric agent-id tie-break)`` — the
        same convention as the retention caps' oldest-first selection.
        """
        candidates = [
            record
            for record in self._view._agent_records.values()
            if record.state is _AgentState.SUSPENDED
            and record.suspended_at is not None
            and record.agent_id not in exclude
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda record: (
                cast(float, record.suspended_at),
                self._view._agent_id_sort_key(record.agent_id),
            ),
        )
