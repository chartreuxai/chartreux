from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from chartreux.app_server._subagent_records import (
    AgentRecord,
    RunRecord,
    StoredRunResult,
    _AgentState,
    _AgentTombstone,
)
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

_MAX_STORED_RESULTS = 32


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
                self._view._result_store.pop(key, None)
                self._view._expired_results.add(key)
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
        unused there). The caller must hold the registry lock.
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
        return PublicationDecision(
            resident=resident,
            write_permitted=write_permitted,
            notify_permitted=write_permitted and not suppress_notification,
        )

    def purge_agent_locked(self, agent_id: str) -> set[tuple[str, str]]:
        """Purge every result collection entry for one agent (locked).

        The former release-transaction purge (ASC area 4, migration step 4;
        ASC area 6 calls this synchronous locked result purge from inside
        the release's one critical section): the result store, expiry
        markers, pending notifications, wait leases, and write tokens are
        purged for the exact agent key, the latest-run index entry is
        removed, and the purged keys are returned so the release keeps its
        unknown-handle check. The caller must hold the registry lock.
        """
        keys = {key for key in self._view._result_store if key[0] == agent_id}
        keys.update(key for key in self._view._expired_results if key[0] == agent_id)
        keys.update(
            key for key in self._view._pending_notifications if key[0] == agent_id
        )
        keys.update(key for key in self._view._wait_leases if key[0] == agent_id)
        keys.update(
            key for key in self._view._result_write_tokens if key[0] == agent_id
        )
        for key in keys:
            self._view._result_store.pop(key, None)
            self._view._expired_results.discard(key)
            self._view._pending_notifications.discard(key)
            self._view._wait_leases.pop(key, None)
            self._view._result_write_tokens.pop(key, None)
        self._view._latest_run_ids.pop(agent_id, None)
        return keys

    async def get_result(
        self, agent_id: str, run_id: str | None = None
    ) -> TaskResult | None:
        """Return the stored result for a run, expiring first (public read)."""
        async with self._view._registry_lock:
            self.expire_locked()
            resolved = self.resolve_locked(agent_id, run_id)
        if resolved is None:
            raise UnknownAgentError(f"Unknown run {run_id} for agent {agent_id}")
        target = resolved[1]
        if isinstance(target, RunRecord):
            if target.status is RunStatus.RUNNING:
                return None
            if target.result is None:
                raise AgentResultExpiredError(
                    f"Result expired for run {target.run_id} on agent {agent_id}"
                )
            return target.result
        return target.result

    async def wait(
        self, agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> WaitForAgentResult:
        """Wait for a run and return the typed disposition (design §6).

        ``completed`` carries the finished run's TaskResult; ``still_running``
        is a normal outcome when the timeout expires first. The timeout never
        cancels the run (the completion task is shielded) and the wait lease
        keeps the result protected from count expiry. Race rule: every
        disposition reports the run the caller pinned at wait start, never a
        newer run's progress.
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
                    return await self.still_running_disposition(
                        agent_id,
                        pinned.key,
                        pinned.target,
                        run_id=pinned.run_id,
                        wait_started_at=pinned.started_at,
                    )
            async with self._view._registry_lock:
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
                    raise RuntimeError(
                        f"Run {pinned.key[1]} completed without a result"
                    )
                return WaitForAgentResult(status=WaitStatus.COMPLETED, result=result)
        finally:
            async with self._view._registry_lock:
                leases = self._view._wait_leases.get(pinned.key, 0)
                if leases > 1:
                    self._view._wait_leases[pinned.key] = leases - 1
                else:
                    self._view._wait_leases.pop(pinned.key, None)
                self.expire_locked()

    async def still_running_disposition(
        self,
        agent_id: str,
        key: tuple[str, str],
        target: RunRecord,
        *,
        run_id: str | None,
        wait_started_at: float,
    ) -> WaitForAgentResult:
        """Build the ``still_running`` disposition for an expired wait (§6).

        The pinned run identity is preserved across timeout/release/replacement
        races: the disposition reports the run this wait targeted. When the
        pinned run actually settled at the timeout boundary, the completed
        disposition is returned instead. A wait fenced by release keeps the
        existing typed errors.
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
                return WaitForAgentResult(status=WaitStatus.COMPLETED, result=result)
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
            return WaitForAgentResult(
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
