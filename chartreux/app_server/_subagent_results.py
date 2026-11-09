from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import math
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

from chartreux.app_server._subagent_records import (
    AgentRecord,
    RunRecord,
    StoredRunResult,
    _AgentState,
    _AgentTombstone,
)
from chartreux.core.session.session_logger import DeliveryStateScan
from chartreux.core.subagents import (
    AgentResultExpiredError,
    CancelOutcome,
    CancelResult,
    RunStatus,
    RunStopReason,
    TaskResult,
    UnknownAgentError,
    WaitForAgentResult,
    WaitStatus,
    still_running_guidance,
)

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime
    from chartreux.app_server._subagent_execution import BackgroundExecutionContext

logger = logging.getLogger(__name__)

_MAX_STORED_RESULTS = 32

#: P2d fetch routes named in the delivery telemetry (per-route reporting).
_DELIVERY_ROUTE_GET = "get_agent_result"
_DELIVERY_ROUTE_WAIT = "wait_for_agent"

#: The fetch representations a mode-bearing delivery records.
DeliveryMode = Literal["digest", "full"]


@dataclass(frozen=True)
class ResultDelivery:
    """Delivery-state outcome of a mode-bearing fetch (P2d, design b).

    ``status`` is one of:

    - ``"delivered"``: the result was delivered; ``marking`` is the envelope
      payload the fetch tool embeds in its result so the marking commits
      atomically with the transcript write; ``post_lift`` marks a re-delivery
      after a compaction boundary lifted the refusal (the fresh marking
      re-arms it).
    - ``"refused"``: mechanical refusal — the run was already delivered in
      this representation with no compaction boundary since, and the fetch
      was not forced; no payload is re-delivered. ``message`` carries the
      force affordance.
    - ``"nothing_left"``: COEXISTENCE — the retained result expired but a
      persisted marking authorizes "nothing left to deliver"; a forced
      refetch surfaces ``AgentResultExpiredError`` instead.
    - ``"pending"``: the run is still running (get path); nothing was marked.
    - ``"passthrough"``: a wait's ``still_running`` disposition passes through
      unchanged; no delivery happened.
    """

    status: str
    representation: str | None = None
    result: TaskResult | None = None
    disposition: WaitForAgentResult | None = None
    marking: dict[str, str] | None = None
    message: str = ""
    forced: bool = False
    post_lift: bool = False


class ResultsStateView(Protocol):
    """Narrow typed view over the registry-owned result state.

    Dynamic access: every member is read through the view at call time, so
    the clock, the replaceable admission/drain scalars, the result maps,
    the write tokens, and the notification suppression are never snapshotted
    at construction — registry state mutated after the collaborator is built
    stays visible.
    """

    _agent_records: dict[str, AgentRecord]
    _result_store: dict[tuple[str, str], StoredRunResult]
    _expired_results: set[tuple[str, str]]
    _collected_results: set[tuple[str, str]]
    # P2d: in-memory delivery markings, (agent_id, run_id) -> {representation:
    # position}. Installed lazily by RunResults on first use under the
    # registry lock (the registry's __getattr__ raises for unknown names, so
    # the store is created on demand). Like _collected_results the in-memory
    # marker dies with eviction; the persisted envelope marking outlives it
    # (COEXISTENCE).
    _delivery_markings: dict[tuple[str, str], dict[str, float]]
    _latest_run_ids: dict[str, str]
    _wait_leases: dict[tuple[str, str], int]
    _pending_notifications: set[tuple[str, str]]
    _result_write_tokens: dict[tuple[str, str], object]
    _evicted_agents: dict[str, _AgentTombstone]
    _children: dict[str, SessionRuntime]
    _admission_closed: bool
    _draining_children: bool
    _suppressed_notifications: set[str]
    _registry_lock: asyncio.Lock
    _clock: Callable[[], float]

    def _agent_turns_used(self, record: AgentRecord) -> int | None: ...

    def _generation_is_current(self, generation: int) -> bool: ...

    def _note_uncollected_change_locked(self, agent_id: str) -> None: ...


@dataclass
class PinnedWait:
    """The run a wait is pinned to (ASC area 4 context).

    Timeout progress is tied to this exact target — never the newest run.
    """

    key: tuple[str, str]
    target: RunRecord | StoredRunResult
    run_id: str | None
    started_at: float


@dataclass(frozen=True)
class PublicationDecision:
    """Publication decision for one finished run (ASC area 4).

    Replaces the historical inline ``publication_permitted`` /
    ``suppress_notification`` pair of the monitor's locked publication
    block: ``write_permitted`` is the result-write permission — the exact
    write-token match, checked separately from residency, so stale
    residency never forbids storing a result — and ``notify_permitted``
    is the notification/residency permission (the write permission plus
    the residency, drain, and suppression state). ``resident`` carries the
    residency check for the caller's record finalization.
    """

    resident: bool
    write_permitted: bool
    notify_permitted: bool


class RunResults:
    """Result expiry, run lookup, waits, and cancellation for the registry.

    Narrow collaborator of the session runtime registry (ASC area 4,
    migration steps 1–4): count-based expiry, run resolution, pinned waits
    with timeout disposition, cancellation/interruption, and the
    token-based publication protocol (writer registration, the publication
    decision, and the per-agent purge) operate on the registry-owned
    result store, expiry markers, latest-run index, wait leases, pending
    notifications, write tokens, tombstones, notification suppression, and
    the admission/drain scalars through the registry-owned typed view,
    under the registry's own lock at the same points. Canonical state
    stays registry-owned (§6.7); no private collaborator locks; no new
    tasks/queues/relay layers. The registry keeps thin same-name
    delegates so existing instance patches and direct calls keep working.
    """

    def __init__(self, view: ResultsStateView) -> None:
        self._view = view

    def expire_locked(self) -> None:
        """Evict unreferenced results past the per-generation cap (locked)."""
        unreferenced_by_generation: dict[int, list[StoredRunResult]] = {}
        for key, stored in self._view._result_store.items():
            if (
                key not in self._view._pending_notifications
                and not self._view._wait_leases.get(key)
            ):
                unreferenced_by_generation.setdefault(
                    stored.root_generation, []
                ).append(stored)
        for unreferenced in unreferenced_by_generation.values():
            excess = len(unreferenced) - _MAX_STORED_RESULTS
            if excess <= 0:
                continue
            for stored in sorted(
                unreferenced, key=lambda item: (item.completed_at, item.agent_id)
            )[:excess]:
                key = (stored.agent_id, stored.run_id)
                # U3b: an uncollected key leaves a visible agent's list here.
                listed = key not in self._view._collected_results
                self._view._result_store.pop(key, None)
                self._view._expired_results.add(key)
                # Marker lifetime never exceeds result lifetime: the run is
                # no longer retrievable, so its collection marker dies here.
                self._view._collected_results.discard(key)
                # P2d COEXISTENCE: the in-memory delivery marker dies with the
                # result exactly like the collection marker; the PERSISTED
                # envelope marking outlives it and still authorizes "nothing
                # left to deliver".
                self._delivery_markings_locked().pop(key, None)
                if listed:
                    self._view._note_uncollected_change_locked(stored.agent_id)
                tombstone = self._view._evicted_agents.get(stored.agent_id)
                if (
                    tombstone is not None
                    and tombstone.summary.current_run_id == stored.run_id
                ):
                    tombstone.summary.result_expired = True
                record = self._view._agent_records.get(stored.agent_id)
                if record is not None:
                    for run in record.run_history:
                        if run.run_id == stored.run_id:
                            run.result = None

    def resolve_locked(
        self, agent_id: str, run_id: str | None
    ) -> tuple[tuple[str, str], RunRecord | StoredRunResult] | None:
        """Resolve the exact run record or stored result for a request."""
        record = self._view._agent_records.get(agent_id)
        if run_id is None:
            run_id = self._view._latest_run_ids.get(agent_id)
            if (
                run_id is None
                and record is not None
                and (record.latest_run_id is not None or record.current_run is not None)
            ):
                run_id = (
                    record.latest_run_id
                    if record.latest_run_id is not None
                    else cast(RunRecord, record.current_run).run_id
                )
            if run_id is None:
                candidates = [
                    stored
                    for stored in self._view._result_store.values()
                    if stored.agent_id == agent_id
                ]
                if candidates:
                    latest = max(candidates, key=lambda item: item.completed_at)
                    return (agent_id, latest.run_id), latest
                return None
        key = (agent_id, run_id)
        if key in self._view._expired_results:
            raise AgentResultExpiredError(
                f"Result expired for run {run_id} on agent {agent_id}"
            )
        if record is not None:
            if record.current_run is not None and record.current_run.run_id == run_id:
                return key, record.current_run
            past = next(
                (item for item in record.run_history if item.run_id == run_id), None
            )
            if past is not None and past.result is not None:
                return key, past
        stored = self._view._result_store.get(key)
        if stored is not None:
            return key, stored
        return None

    def uncollected_run_ids_locked(self, agent_id: str) -> list[str]:
        """The agent's retrievable, uncollected run IDs, newest first (locked).

        The universe is the SAME resolution space the collection paths read
        (design B1): result-store keys UNION the resident record's current
        and historical runs that still carry a result, minus expired keys,
        minus collected markers. A result published only in the resident
        record (assigned before the locked store write, or retained when the
        store write is revoked) is therefore retrievable AND visible.

        Ordering is descending ``completed_at`` — the finalization timestamp
        stamped before publication, so newest publication first — with the
        run ID as the deterministic tie-break for equal or missing
        timestamps. The caller must hold the registry lock.
        """
        completed_at_by_run: dict[str, float] = {}
        record = self._view._agent_records.get(agent_id)
        if record is not None:
            for run in [record.current_run, *record.run_history]:
                if run is None or run.result is None:
                    continue
                timestamp = run.completed_at
                if timestamp is None:
                    timestamp = run.started_at
                completed_at_by_run[run.run_id] = (
                    float("-inf") if timestamp is None else timestamp
                )
        for (stored_agent_id, run_id), stored in self._view._result_store.items():
            if stored_agent_id == agent_id:
                completed_at_by_run[run_id] = stored.completed_at
        return sorted(
            (
                run_id
                for run_id in completed_at_by_run
                if (agent_id, run_id) not in self._view._expired_results
                and (agent_id, run_id) not in self._view._collected_results
            ),
            key=lambda run_id: (completed_at_by_run[run_id], run_id),
            reverse=True,
        )

    def register_writer_locked(self, agent_id: str, run_id: str, token: object) -> None:
        """Register the exact write token for one accepted run (locked).

        The former acceptance registration of the registry's admitted run
        (ASC area 4, migration step 4): the write token is installed for
        the exact (agent, run) key at acceptance; publication later accepts
        this exact token — authority is never reconstructed from handle
        strings. The caller must hold the registry lock.

        Review R6: writer tokens are PACKAGE-exclusive, not agent-keyed.
        Registering a replacement's token revokes every OTHER live token
        whose record carries the same work package, so a superseded
        predecessor (or an earlier replacement of the same package) still
        in flight can never publish over the package's current owner.
        """
        self._view._result_write_tokens[(agent_id, run_id)] = token
        record = self._view._agent_records.get(agent_id)
        package_id = record.work_package_id if record is not None else None
        if package_id is None:
            return
        for key in list(self._view._result_write_tokens):
            if key[0] == agent_id:
                continue
            other = self._view._agent_records.get(key[0])
            if other is not None and other.work_package_id == package_id:
                self._view._result_write_tokens.pop(key, None)

    def publish_locked(
        self, context: BackgroundExecutionContext
    ) -> PublicationDecision:
        """Decide and consume one finished run's publication authority (locked).

        The former monitor publication check/pop of the registry's admitted
        run (ASC area 4, migration step 4), moved intact into the results
        collaborator: the residency check, the notification suppression,
        and the exact write-token check/pop run in the caller's locked
        block, and the returned PublicationDecision separates the
        result-write permission from the notification/residency permission.
        Publication accepts the execution context's exact write token and
        record/run references; stale residency never forbids storing a
        result. The rollback consumes the same authority (the decision is
        unused there). A collection marker recorded by a read that raced
        ahead of the result-store write is reconciled here: a permitted
        write or a retained resident copy of the result keeps it, while a
        revoked publication with no retrievable resident result drops it, so
        no marker outlives its result. The caller must hold the registry
        lock.
        """
        key = (context.agent_id, context.run_id)
        resident = (
            self._view._agent_records.get(context.agent_id) is context.record
            and self._view._children.get(context.record.session_id) is context.runtime
            and context.record.current_run is context.run_record
            and self._view._generation_is_current(context.record.root_generation)
        )
        suppress_notification = (
            not resident
            or self._view._draining_children
            or context.agent_id in self._view._suppressed_notifications
        )
        write_permitted = (
            self._view._result_write_tokens.get(key) is context.publication_token
        )
        if write_permitted:
            self._view._result_write_tokens.pop(key, None)
        self._reconcile_collection_locked(context, write_permitted=write_permitted)
        return PublicationDecision(
            resident=resident,
            write_permitted=write_permitted,
            notify_permitted=write_permitted and not suppress_notification,
        )

    def _reconcile_collection_locked(
        self, context: BackgroundExecutionContext, *, write_permitted: bool
    ) -> None:
        """Drop a collection marker whose result this publication revokes (locked).

        A read can observe a resident run's result before the result-store
        write (``RunRecord.result`` is assigned ahead of the locked write),
        so its marker must survive exactly as long as a retrievable result
        does: a permitted write or a resident record still holding the
        result keeps it; a revoked publication with neither drops it.
        """
        key = (context.agent_id, context.run_id)
        if key not in self._view._collected_results or write_permitted:
            return
        if key in self._view._result_store:
            return
        record = self._view._agent_records.get(context.agent_id)
        if record is not None and any(
            run is not None and run.run_id == context.run_id and run.result is not None
            for run in [record.current_run, *record.run_history]
        ):
            return
        self._view._collected_results.discard(key)

    def purge_agent_locked(self, agent_id: str) -> set[tuple[str, str]]:
        """Purge every result collection entry for one agent (locked).

        The former release-transaction purge (ASC area 4, migration step 4;
        ASC area 6 calls this synchronous locked result purge from inside
        the release's one critical section): the result store, expiry
        markers, collection markers, pending notifications, wait leases, and
        write tokens are purged for the exact agent key, the latest-run index
        entry is removed, and the purged keys are returned so the release
        keeps its unknown-handle check. The caller must hold the registry
        lock.
        """
        keys = {key for key in self._view._result_store if key[0] == agent_id}
        keys.update(key for key in self._view._expired_results if key[0] == agent_id)
        keys.update(key for key in self._view._collected_results if key[0] == agent_id)
        keys.update(
            key for key in self._view._pending_notifications if key[0] == agent_id
        )
        keys.update(key for key in self._view._wait_leases if key[0] == agent_id)
        keys.update(
            key for key in self._view._result_write_tokens if key[0] == agent_id
        )
        keys.update(
            key for key in self._delivery_markings_locked() if key[0] == agent_id
        )
        for key in keys:
            self._view._result_store.pop(key, None)
            self._view._expired_results.discard(key)
            self._view._collected_results.discard(key)
            self._view._pending_notifications.discard(key)
            self._view._wait_leases.pop(key, None)
            self._view._result_write_tokens.pop(key, None)
            self._delivery_markings_locked().pop(key, None)
        self._view._latest_run_ids.pop(agent_id, None)
        return keys

    # ------------------------------------------------------------------
    # P2d delivery states: the marking, the refusal predicate, telemetry.
    # ------------------------------------------------------------------

    def _delivery_markings_locked(self) -> dict[tuple[str, str], dict[str, float]]:
        """The registry-owned in-memory delivery marking store (locked).

        Installed lazily on the view: the registry constructs RunResults before
        any fetch, and its __getattr__ raises for unknown names, so the first
        locked access creates the store.
        """
        markings = getattr(self._view, "_delivery_markings", None)
        if markings is None:
            markings = {}
            self._view._delivery_markings = markings
        return markings

    def _conversation_id_locked(self) -> str | None:
        """The persisted conversation id of the root session, when bound."""
        root = getattr(self._view, "_root", None)
        loop = getattr(root, "agent_loop", None) if root is not None else None
        session_id = getattr(loop, "session_id", None) if loop is not None else None
        return session_id if isinstance(session_id, str) else None

    def _delivery_scan_locked(self) -> DeliveryStateScan | None:
        """The transcript delivery-state scan via the bound root, if any.

        Reads the persisted envelope markings and compaction boundaries
        (surviving restart via transcript replay) plus the live tail. Returns
        None when no real logger is bound — the refusal predicate then only
        sees the in-memory markings.
        """
        root = getattr(self._view, "_root", None)
        loop = getattr(root, "agent_loop", None) if root is not None else None
        session_logger = (
            getattr(loop, "session_logger", None) if loop is not None else None
        )
        scan_fn = getattr(session_logger, "delivery_state_scan", None)
        if not callable(scan_fn):
            return None
        messages = getattr(loop, "messages", None)
        try:
            scan = scan_fn(messages)
        except Exception:
            return None
        return scan if isinstance(scan, DeliveryStateScan) else None

    def _refusal_state_locked(
        self, key: tuple[str, str], mode: str, scan: DeliveryStateScan | None
    ) -> tuple[float | None, bool]:
        """The refusal predicate state for one (run, representation) fetch.

        Returns ``(position, lifted)``: ``position`` is the latest marking
        position for (conversation, agent, run, representation) — None when
        the run was never delivered in this representation — and ``lifted``
        is whether a compaction boundary landed AFTER that marking (the
        mechanical refusal lift: "delivered AND no compaction boundary
        between that delivery and this fetch" is false, so the fetch may
        re-deliver). The in-memory marking carries the position the
        delivery's tool message lands at (the scan's end), so a boundary
        appended after it lifts even in the same live session.
        """
        conversation_id = self._conversation_id_locked()
        persisted_position: float | None = None
        if scan is not None:
            for marking in scan.markings:
                if (
                    marking.agent_id == key[0]
                    and marking.run_id == key[1]
                    and marking.representation == mode
                    and (
                        conversation_id is None
                        or marking.conversation_id == conversation_id
                    )
                ):
                    if (
                        persisted_position is None
                        or marking.position > persisted_position
                    ):
                        persisted_position = marking.position
        in_memory_position = self._delivery_markings_locked().get(key, {}).get(mode)
        if persisted_position is None and in_memory_position is None:
            return None, False
        position = max(
            persisted_position if persisted_position is not None else -math.inf,
            in_memory_position if in_memory_position is not None else -math.inf,
        )
        lifted = scan is not None and any(
            boundary > position for boundary in scan.compaction_boundaries
        )
        return position, lifted

    def _persisted_marking_exists_locked(
        self,
        agent_id: str,
        run_id: str | None,
        mode: str,
        conversation_id: str | None,
        scan: DeliveryStateScan | None,
    ) -> bool:
        """Whether a persisted envelope marking authorizes "nothing left to
        deliver" for the agent's run (COEXISTENCE; locked).
        """
        if scan is None:
            return False
        return any(
            marking.agent_id == agent_id
            and marking.representation == mode
            and (run_id is None or marking.run_id == run_id)
            and (conversation_id is None or marking.conversation_id == conversation_id)
            for marking in scan.markings
        )

    def _note_delivery_locked(
        self,
        key: tuple[str, str],
        mode: str,
        status: str,
        payload_bytes: int,
        forced: bool,
        post_lift: bool,
        route: str,
    ) -> None:
        """Content-free delivery telemetry (P3 measurement conventions).

        Digest deliveries, full deliveries, post-lift re-deliveries, and
        refusals are reported per route with the delivered payload's
        serialized bytes (the base P3's replay weighting multiplies).
        """
        logger.info(
            "result_delivery event=result_delivery route=%s status=%s "
            "representation=%s agent_id=%s run_id=%s payload_bytes=%d "
            "forced=%s post_lift=%s",
            route,
            status,
            mode,
            key[0],
            key[1],
            payload_bytes,
            forced,
            post_lift,
        )

    def _settle_delivery_locked(
        self,
        key: tuple[str, str],
        mode: DeliveryMode,
        force: bool,
        *,
        route: str,
        result: TaskResult | None = None,
        disposition: WaitForAgentResult | None = None,
    ) -> ResultDelivery:
        """Evaluate the refusal and register the delivery marking (locked).

        Pinned semantics: the marking PERSISTS; the refusal predicate is
        "delivered AND no compaction boundary between that delivery and this
        fetch". A delivered fetch registers a FRESH in-memory marking (and the
        tool embeds the envelope payload so the persisted marking commits
        with the transcript write), re-arming the refusal; a post-lift
        re-delivery is exactly that fresh marking after a boundary.
        """
        scan = self._delivery_scan_locked()
        prior, lifted = self._refusal_state_locked(key, mode, scan)
        if prior is not None and not lifted and not force:
            self._note_delivery_locked(key, mode, "refused", 0, force, False, route)
            return ResultDelivery(
                status="refused",
                representation=mode,
                message=(
                    f"Run {key[1]} on agent {key[0]} was already delivered as "
                    f"{mode} in this conversation and no compaction boundary "
                    "has landed since that delivery, so the delivered payload "
                    "is still in context. Pass force=True to re-deliver it "
                    "anyway; refetching after a compaction boundary is normal "
                    "and writes a fresh marking that re-arms this refusal."
                ),
            )
        # The fresh in-memory marking sits where this delivery's tool message
        # lands: the scan's end when a transcript is bound, else +inf (no
        # boundaries are knowable without one).
        delivery_position = float(scan.position_count) if scan is not None else math.inf
        self._delivery_markings_locked().setdefault(key, {})[mode] = delivery_position
        marking = {
            "conversation_id": self._conversation_id_locked() or "",
            "agent_id": key[0],
            "run_id": key[1],
            "representation": mode,
        }
        carried = (
            result
            if result is not None
            else (disposition.result if disposition is not None else None)
        )
        payload_bytes = (
            len(
                json.dumps(carried.model_dump(mode="json"), default=str).encode("utf-8")
            )
            if carried is not None
            else 0
        )
        self._note_delivery_locked(
            key, mode, "delivered", payload_bytes, force, lifted, route
        )
        return ResultDelivery(
            status="delivered",
            representation=mode,
            result=result,
            disposition=disposition,
            marking=marking,
            forced=force,
            post_lift=lifted,
        )

    @overload
    async def get_result(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        mode: None = None,
        force: bool = False,
    ) -> TaskResult | None: ...

    @overload
    async def get_result(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        mode: DeliveryMode,
        force: bool = False,
    ) -> ResultDelivery: ...

    async def get_result(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        mode: DeliveryMode | None = None,
        force: bool = False,
    ) -> TaskResult | None | ResultDelivery:
        """Return the stored result for a run, expiring first (public read).

        A resolved ``TaskResult`` is marked collected inside the same locked
        block that resolves it (design B3): no suspension point separates the
        mark from the return, so a cancellation cannot mark a run whose
        result never reached the caller. The RUNNING -> ``None`` disposition
        and the typed expiry/unknown errors never mark.

        With ``mode`` set (the fetch tools' delivery path; keyword-only, so
        the registry's legacy delegate and the perf call sites keep the raw
        ``TaskResult | None`` contract unchanged) the call evaluates the P2d
        delivery states instead and returns a ``ResultDelivery``: delivered /
        refused / nothing_left / pending. COEXISTENCE: a non-forced fetch of
        an evicted result whose persisted marking covers (conversation,
        agent, run, representation) answers "nothing left to deliver"; a
        forced refetch surfaces ``AgentResultExpiredError``.
        """
        async with self._view._registry_lock:
            self.expire_locked()
            try:
                resolved = self.resolve_locked(agent_id, run_id)
            except AgentResultExpiredError:
                if mode is not None and not force:
                    conversation_id = self._conversation_id_locked()
                    if self._persisted_marking_exists_locked(
                        agent_id,
                        run_id,
                        mode,
                        conversation_id,
                        self._delivery_scan_locked(),
                    ):
                        key = (agent_id, run_id or "latest")
                        self._note_delivery_locked(
                            key,
                            mode,
                            "nothing_left",
                            0,
                            force,
                            False,
                            _DELIVERY_ROUTE_GET,
                        )
                        return ResultDelivery(
                            status="nothing_left",
                            representation=mode,
                            message=(
                                f"Nothing left to deliver for run {run_id or 'latest'} "
                                f"on agent {agent_id}: it was already delivered as "
                                f"{mode} in this conversation (persisted marking) "
                                "and the retained result has since expired. A "
                                "forced refetch surfaces AgentResultExpiredError."
                            ),
                        )
                raise
            if resolved is None:
                raise UnknownAgentError(f"Unknown run {run_id} for agent {agent_id}")
            key, target = resolved
            if isinstance(target, RunRecord):
                if target.status is RunStatus.RUNNING:
                    if mode is None:
                        return None
                    return ResultDelivery(status="pending", representation=mode)
                if target.result is None:
                    raise AgentResultExpiredError(
                        f"Result expired for run {target.run_id} on agent {agent_id}"
                    )
                result = target.result
            else:
                result = target.result
            # U3b: a first collection removes the run from a visible agent's
            # list; a repeated collection changes nothing.
            if key not in self._view._collected_results:
                self._view._collected_results.add(key)
                self._view._note_uncollected_change_locked(agent_id)
            if mode is None:
                return result
            return self._settle_delivery_locked(
                key, mode, force, route=_DELIVERY_ROUTE_GET, result=result
            )

    @overload
    async def wait(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        timeout: float | None = None,
        mode: None = None,
        force: bool = False,
    ) -> WaitForAgentResult: ...

    @overload
    async def wait(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        timeout: float | None = None,
        mode: DeliveryMode,
        force: bool = False,
    ) -> ResultDelivery: ...

    async def wait(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        timeout: float | None = None,
        mode: DeliveryMode | None = None,
        force: bool = False,
    ) -> WaitForAgentResult | ResultDelivery:
        """Wait for a run and return the typed disposition (design §6).

        ``completed`` carries the finished run's TaskResult; ``still_running``
        is a normal outcome when the timeout expires first. The timeout never
        cancels the run (the completion task is shielded) and the wait lease
        keeps the result protected from count expiry. Race rule: every
        disposition reports the run the caller pinned at wait start, never a
        newer run's progress.

        Lease cleanup completes before the locked settle-and-mark commit
        (design B3): a wait cancelled while that cleanup would have been
        pending never marks, and no suspension point separates the
        collection marker from the registry's return.

        With ``mode`` set (the wait tool's delivery path; keyword-only, so the
        registry delegate and the perf call sites keep the raw
        ``WaitForAgentResult`` contract unchanged — the marking is SHARED
        with the get route) a ``completed`` disposition evaluates the P2d
        delivery states in the same locked settle block and returns a
        ``ResultDelivery``; ``still_running`` passes through unchanged.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")
        wait_started_at = self._view._clock()
        async with self._view._registry_lock:
            resolved = self.resolve_locked(agent_id, run_id)
            if resolved is None:
                raise UnknownAgentError(f"Unknown run {run_id} for agent {agent_id}")
            key, target = resolved
            self._view._wait_leases[key] = self._view._wait_leases.get(key, 0) + 1
        pinned = PinnedWait(
            key=key, target=target, run_id=run_id, started_at=wait_started_at
        )
        settled = False
        try:
            if (
                isinstance(pinned.target, RunRecord)
                and pinned.target.status is RunStatus.RUNNING
            ):
                try:
                    await asyncio.wait_for(
                        asyncio.shield(pinned.target.completion_task), timeout=timeout
                    )
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        raise
                    async with self._view._registry_lock:
                        released = pinned.key not in self._view._wait_leases
                    if not released:
                        raise
                except TimeoutError:
                    disposition = await self.still_running_disposition(
                        agent_id,
                        pinned.key,
                        pinned.target,
                        run_id=pinned.run_id,
                        wait_started_at=pinned.started_at,
                        mode=mode,
                        force=force,
                    )
                    settled = True
                    return disposition
            delivery: ResultDelivery | None = None
            async with self._view._registry_lock:
                disposition = self.settle_wait_locked(agent_id, pinned)
                if mode is not None and disposition.status is WaitStatus.COMPLETED:
                    # P2d: the delivery marking settles in the SAME locked
                    # block as the collection marker — no suspension point
                    # between the refusal check, the marking, and the return.
                    delivery = self._settle_delivery_locked(
                        pinned.key,
                        mode,
                        force,
                        route=_DELIVERY_ROUTE_WAIT,
                        disposition=disposition,
                    )
            settled = True
            return delivery if delivery is not None else disposition
        finally:
            if not settled:
                async with self._view._registry_lock:
                    self.release_wait_lease_locked(pinned.key)
                    self.expire_locked()

    def settle_wait_locked(
        self, agent_id: str, pinned: PinnedWait
    ) -> WaitForAgentResult:
        """Resolve, mark, and commit one completed wait (locked, never awaited).

        Release fencing comes first (a released or expired pinned run keeps
        its typed error), then the pinned result resolves, the wait lease is
        released, the collection marker is recorded, and the disposition is
        built — one locked block with no suspension point, so the mark and
        the return commit together (design B3).
        """
        if pinned.key not in self._view._wait_leases:
            resolved = self.resolve_locked(agent_id, pinned.run_id)
            if resolved is None:
                raise UnknownAgentError(
                    f"Unknown run {pinned.run_id} for agent {agent_id}"
                )
            raise AgentResultExpiredError(
                f"Result expired for run {pinned.key[1]} on agent {agent_id}"
            )
        stored = self._view._result_store.get(pinned.key)
        result = stored.result if stored is not None else pinned.target.result
        if result is None:
            raise RuntimeError(f"Run {pinned.key[1]} completed without a result")
        self.release_wait_lease_locked(pinned.key)
        if pinned.key not in self._view._collected_results:
            self._view._collected_results.add(pinned.key)
            self._view._note_uncollected_change_locked(agent_id)
        self.expire_locked()
        return WaitForAgentResult(status=WaitStatus.COMPLETED, result=result)

    def release_wait_lease_locked(self, key: tuple[str, str]) -> None:
        """Release one wait lease (locked); other holders keep theirs."""
        leases = self._view._wait_leases.get(key, 0)
        if leases > 1:
            self._view._wait_leases[key] = leases - 1
        else:
            self._view._wait_leases.pop(key, None)

    @overload
    async def still_running_disposition(
        self,
        agent_id: str,
        key: tuple[str, str],
        target: RunRecord,
        *,
        run_id: str | None,
        wait_started_at: float,
        mode: None = None,
        force: bool = False,
    ) -> WaitForAgentResult: ...

    @overload
    async def still_running_disposition(
        self,
        agent_id: str,
        key: tuple[str, str],
        target: RunRecord,
        *,
        run_id: str | None,
        wait_started_at: float,
        mode: DeliveryMode,
        force: bool = False,
    ) -> ResultDelivery: ...

    async def still_running_disposition(
        self,
        agent_id: str,
        key: tuple[str, str],
        target: RunRecord,
        *,
        run_id: str | None,
        wait_started_at: float,
        mode: DeliveryMode | None = None,
        force: bool = False,
    ) -> WaitForAgentResult | ResultDelivery:
        """Build the ``still_running`` disposition for an expired wait (§6).

        The pinned run identity is preserved across timeout/release/replacement
        races: the disposition reports the run this wait targeted. When the
        pinned run actually settled at the timeout boundary, the completed
        disposition is returned instead. A wait fenced by release keeps the
        existing typed errors. The wait lease is released inside the same
        locked block that builds the disposition, so a completed boundary
        disposition is marked and returned with no suspension point between
        (design B3). With ``mode`` set, a completed boundary disposition
        additionally evaluates the P2d delivery states in that same locked
        block and returns a ``ResultDelivery``.
        """
        async with self._view._registry_lock:
            if key not in self._view._wait_leases:
                # Released while waiting: keep the release fencing semantics.
                resolved = self.resolve_locked(agent_id, run_id)
                if resolved is None:
                    raise UnknownAgentError(
                        f"Unknown run {run_id} for agent {agent_id}"
                    )
                raise AgentResultExpiredError(
                    f"Result expired for run {key[1]} on agent {agent_id}"
                )
            stored = self._view._result_store.get(key)
            result = stored.result if stored is not None else target.result
            if result is not None:
                self.release_wait_lease_locked(key)
                if key not in self._view._collected_results:
                    self._view._collected_results.add(key)
                    self._view._note_uncollected_change_locked(agent_id)
                self.expire_locked()
                disposition = WaitForAgentResult(
                    status=WaitStatus.COMPLETED, result=result
                )
                if mode is None:
                    return disposition
                return self._settle_delivery_locked(
                    key,
                    mode,
                    force,
                    route=_DELIVERY_ROUTE_WAIT,
                    disposition=disposition,
                )
            if target.status is not RunStatus.RUNNING:
                raise RuntimeError(f"Run {key[1]} completed without a result")
            elapsed = max(0.0, self._view._clock() - wait_started_at)
            record = self._view._agent_records.get(agent_id)
            # Pinned-run progress: only the record's live current run has a
            # meaningful in-flight turn count; a displaced run reports zero.
            if record is not None and record.current_run is target:
                turns_used = self._view._agent_turns_used(record) or 0
            else:
                turns_used = 0
            disposition = WaitForAgentResult(
                status=WaitStatus.STILL_RUNNING,
                agent_id=agent_id,
                run_id=key[1],
                elapsed_seconds=elapsed,
                run_status=target.status,
                turns_used=turns_used,
                guidance=still_running_guidance(
                    agent_id=agent_id,
                    run_id=key[1],
                    elapsed_seconds=elapsed,
                    run_status=target.status,
                    turns_used=turns_used,
                ),
            )
            self.release_wait_lease_locked(key)
            self.expire_locked()
            if mode is None:
                return disposition
            # P2d: a still_running wait delivered nothing — pass through.
            return ResultDelivery(
                status="passthrough", representation=mode, disposition=disposition
            )

    async def cancel(  # noqa: PLR0911
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        reason: RunStopReason,
        requester_session_id: str,
    ) -> CancelResult:
        async with self._view._registry_lock:
            # Only the server's user route supplies USER_CANCELLED. Tool stops
            # retain parent ownership; users may stop any run in this registry.
            user_cancel = reason is RunStopReason.USER_CANCELLED
            record = self._view._agent_records.get(agent_id)
            tombstone = self._view._evicted_agents.get(agent_id)
            if record is None and tombstone is None:
                if (
                    not user_cancel
                    and agent_id in self._view._children
                    and not any(
                        item.session_id == agent_id
                        for item in self._view._agent_records.values()
                    )
                ):
                    raise ValueError(
                        "Foreground agents cannot be stopped with cancel_agent; "
                        "only background runs are supported"
                    )
                return CancelResult(outcome=CancelOutcome.UNKNOWN_RUN, run_id=run_id)
            if self._view._admission_closed or self._view._draining_children:
                return CancelResult(outcome=CancelOutcome.FORBIDDEN, run_id=run_id)
            if record is None:
                assert tombstone is not None
                summary = tombstone.summary
                if (
                    not user_cancel
                    and requester_session_id != tombstone.parent_identity[0]
                ):
                    return CancelResult(outcome=CancelOutcome.FORBIDDEN, run_id=run_id)
                return CancelResult(
                    outcome=CancelOutcome.NOT_RUNNING,
                    run_id=summary.latest_run_id or summary.current_run_id,
                    stop_reason=summary.stop_reason,
                )
            if not user_cancel and requester_session_id != record.parent_session_id:
                return CancelResult(outcome=CancelOutcome.FORBIDDEN, run_id=run_id)
            if run_id is None:
                run_id = (
                    record.current_run.run_id
                    if record.current_run is not None
                    else record.latest_run_id
                )
            run = next(
                (
                    item
                    for item in [record.current_run, *record.run_history]
                    if item is not None and item.run_id == run_id
                ),
                None,
            )
            key = (agent_id, run_id)
            if run is None and run_id != record.latest_run_id:
                if (
                    key not in self._view._result_store
                    and key not in self._view._expired_results
                ):
                    return CancelResult(
                        outcome=CancelOutcome.UNKNOWN_RUN, run_id=run_id
                    )
            if (
                record.state is _AgentState.FINALIZING
                and run_id == record.latest_run_id
            ):
                return CancelResult(
                    outcome=CancelOutcome.ALREADY_FINISHING,
                    run_id=run_id,
                    stop_reason=record.stop_reason,
                )
            if (
                run is not None
                and record.current_run is run
                and record.state is _AgentState.RUNNING
                and (
                    run.initial_turn_id is None
                    or record.runtime.turns.operation_pending_turn_id(
                        run.initial_turn_id
                    )
                    is None
                )
            ):
                return CancelResult(
                    outcome=CancelOutcome.ALREADY_FINISHING,
                    run_id=run_id,
                    stop_reason=run.stop_reason,
                )
            if (
                run is None
                or record.current_run is not run
                or record.state is not _AgentState.RUNNING
                or run.status is not RunStatus.RUNNING
            ):
                return CancelResult(
                    outcome=CancelOutcome.NOT_RUNNING,
                    run_id=run_id,
                    stop_reason=run.stop_reason if run is not None else None,
                )
            if run.requested_stop_reason is not None:
                return CancelResult(
                    outcome=CancelOutcome.ALREADY_STOPPING,
                    run_id=run_id,
                    stop_reason=run.requested_stop_reason,
                )
            run.requested_stop_reason = reason
        # This helper is synchronous: no scheduling gap between arbitration and
        # interruption, no lock across cleanup awaits, and no monitor cancellation.
        if not self.interrupt_operation(record, run):
            # No await separates the latch from delivery; a failed delivery did
            # not accept a stop and must not leave a misleading stopping latch.
            run.requested_stop_reason = None
            return CancelResult(
                outcome=CancelOutcome.ALREADY_FINISHING,
                run_id=run.run_id,
                stop_reason=run.stop_reason,
            )
        return CancelResult(
            outcome=CancelOutcome.STOP_REQUESTED,
            run_id=run.run_id,
            stop_reason=run.requested_stop_reason,
        )

    @staticmethod
    def interrupt_operation(record: AgentRecord, run: RunRecord) -> bool:
        """Non-destructive and operation-pinned; release owns fallback teardown."""
        if run.initial_turn_id is None:
            return False
        return record.runtime.turns.interrupt_operation(run.initial_turn_id)
