from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from chartreux.app_server._subagent_records import (
    AgentRecord,
    RunRecord,
    _AgentState,
    _AgentTombstone,
)
from chartreux.core.subagents import (
    AgentEviction,
    ReleaseAgentOutcome,
    RunStatus,
    UnknownAgentError,
)
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime


class ReleaseStateView(Protocol):
    """Narrow typed view over the registry-owned release state.

    Dynamic access: every member is read through the view at call time, so
    the monitor-task set, the child maps, the shared registry lock, the
    generation identity, the drain flag, the reference-counted suppression
    ownership, the record/tombstone maps, the root, the tracked teardown
    set, and the clock are never snapshotted at
    construction — registry state mutated after the collaborator is built
    stays visible, and instance patches of the registry delegates (including
    the agents-update emission, the run interruption, the result purge, and
    the release prepare/join pair) still shadow.
    """

    _monitor_tasks: set[asyncio.Task[None]]
    _children: dict[str, SessionRuntime]
    _child_links: dict[str, tuple[SessionRuntime, str]]
    _registry_lock: asyncio.Lock
    _generation_identity: tuple[str, int] | None
    _draining_children: bool
    _suppression_owners: dict[str, int]
    _agent_records: dict[str, AgentRecord]
    _evicted_agents: dict[str, _AgentTombstone]
    _pending_evictions: dict[str, object]
    _root: SessionRuntime | None
    _teardown_tasks: set[asyncio.Task[None]]
    _clock: Callable[[], float]

    @staticmethod
    def _interrupt_run(record: AgentRecord, run: RunRecord) -> bool: ...

    def _emit_agents_update(
        self, evictions: list[AgentEviction] | None = None
    ) -> Coroutine[Any, Any, None]: ...

    def _generation_is_current(self, generation: int) -> bool: ...

    def _release_notification_suppression(self, agent_id: str) -> None: ...

    def _acquire_notification_suppression(self, agent_id: str) -> None: ...

    def _clear_replacement_locked(self, record: AgentRecord) -> None: ...

    def _rearm_reaper_locked(self, exclude: frozenset[str] = frozenset()) -> None: ...

    def _purge_agent_locked(self, agent_id: str) -> set[tuple[str, str]]: ...

    def _track_child_close(
        self,
        runtime: SessionRuntime,
        cleanup: Coroutine[Any, Any, None] | None = None,
        *,
        name: str,
    ) -> asyncio.Task[None]: ...

    def _live_tombstone_locked(
        self, agent_id: str, now: float
    ) -> _AgentTombstone | None: ...

    def _suspension_discard_locked(self, agent_id: str) -> None: ...

    def _prepare_release_locked(
        self, agent_id: str
    ) -> tuple[asyncio.Task[None], ReleaseAgentOutcome]: ...

    def _join_release_teardown(
        self, teardown: asyncio.Task[None], agent_id: str, outcome: ReleaseAgentOutcome
    ) -> Coroutine[Any, Any, ReleaseAgentOutcome]: ...


@dataclass
class ReleaseContext:
    """Per-release invocation state (ASC area 6).

    Carries the agent id, the exact detached record, the release
    generation, and the notification intent captured by the release
    transaction, plus the cleanup state the tracked teardown accumulates.
    The registry-owned suppression ownership, child maps, monitor tasks,
    and lock stay canonical; the context holds only the release
    transaction's local state.
    """

    agent_id: str
    record: AgentRecord | None
    release_generation: int | None
    notify_release: bool
    errors: list[BaseException] = field(default_factory=list)

    async def cleanup(self, view: ReleaseStateView) -> None:
        """Run the tracked release teardown for one release transaction.

        Replaces the former nested cleanup closure of the registry's
        release transaction (the ``ChildPublicationAttempt.rollback``
        convention — the context method takes the view). The cleanup
        region and the exception suppression are unchanged; suppression
        stays registry reference-counted, and the sole-owner publication
        condition is preserved verbatim.
        """
        errors = self.errors
        try:
            if self.record is not None:
                record = self.record
                monitor_tasks = {
                    run.completion_task
                    for run in [*record.run_history, record.current_run]
                    if run is not None
                    and isinstance(run.completion_task, asyncio.Task)
                    and run.completion_task in view._monitor_tasks
                    and not run.completion_task.done()
                }
                current_run = record.current_run
                for task in monitor_tasks:
                    task.cancel()
                try:
                    if current_run is not None:
                        interrupted = view._interrupt_run(record, current_run)
                        if not interrupted and current_run.status is RunStatus.RUNNING:
                            await record.runtime.turns.close()
                    if monitor_tasks:
                        await asyncio.gather(*monitor_tasks, return_exceptions=True)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    view._children.pop(record.session_id, None)
                    view._child_links.pop(record.session_id, None)
                    try:
                        await record.runtime.close()
                    except BaseException as exc:
                        errors.append(exc)
            if self.notify_release:
                async with view._registry_lock:
                    notification_permitted = (
                        (
                            self.release_generation is None
                            and view._generation_identity is None
                        )
                        or (
                            self.release_generation is not None
                            and view._generation_is_current(self.release_generation)
                        )
                    ) and (
                        not view._draining_children
                        and view._suppression_owners.get(self.agent_id) == 1
                    )
                if notification_permitted:
                    try:
                        await view._emit_agents_update()
                    except BaseException as exc:
                        errors.append(exc)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup("Failed to release background agent", errors)
        finally:
            view._release_notification_suppression(self.agent_id)


class AgentRelease:
    """Destructive release transaction and teardown join for the registry.

    Narrow collaborator of the session runtime registry (ASC area 6,
    migration steps 1–3): the tracked teardown join (17.5a), then the
    synchronous purge transaction and the guarded/unguarded entrypoints
    (17.5b) operate on the registry-owned records, result collections,
    suppression ownership, child maps, and the shared lock through the
    registry-owned typed view. UI validation and destructive preparation
    stay in ONE registry-lock critical section; suppression stays registry
    reference-counted, never a context-local boolean; the sole-owner
    publication condition is preserved verbatim. Canonical state stays
    registry-owned (§6.7); no private collaborator locks; no new
    tasks/queues/relay layers (the teardown task is still created into the
    registry-owned tracked sets). The registry keeps thin same-name
    delegates so existing instance patches and direct calls keep working.
    """

    def __init__(self, view: ReleaseStateView) -> None:
        self._view = view

    def prepare_locked(
        self, agent_id: str
    ) -> tuple[asyncio.Task[None], ReleaseAgentOutcome]:
        """Destructive release critical section; the caller must hold the lock.

        Purges the agent's results, expiry markers, pending notifications,
        wait leases, and write tokens (the results collaborator's
        synchronous locked per-agent purge, which returns the purged keys
        for the unknown-handle check below); detaches the record; clears a
        replacement reservation; removes terminal metadata and its
        pending-eviction authority; rearms the reaper; and returns the
        tracked teardown task together with the release outcome. Raises
        UnknownAgentError when nothing about the handle is known.
        """
        self._view._acquire_notification_suppression(agent_id)
        record = self._view._agent_records.pop(agent_id, None)
        # WP3: release forfeits the record's suspension storage-seam state
        # (a released agent's pending recovery — including a suspended
        # barrier or an in-flight admission — is invalidated with the record).
        self._view._suspension_discard_locked(agent_id)
        if record is not None and record.reserved_for_replacement:
            self._view._clear_replacement_locked(record)
        known_evicted = self._view._evicted_agents.pop(agent_id, None) is not None
        # Purge the pending-eviction authority alongside everything else:
        # a zero-TTL eviction's notification is authorized only while
        # this entry survives.
        pending_eviction = self._view._pending_evictions.pop(agent_id, None) is not None
        release_generation = (
            record.root_generation
            if record is not None
            else self._view._root.agent_loop._session_generation
            if self._view._root is not None
            else None
        )
        notify_release = (
            record is not None
            and self._view._generation_is_current(record.root_generation)
        ) or known_evicted
        keys = self._view._purge_agent_locked(agent_id)
        if record is None and not known_evicted and not pending_eviction and not keys:
            self._view._release_notification_suppression(agent_id)
            raise UnknownAgentError(f"Unknown agent: {agent_id}")
        self._view._rearm_reaper_locked()
        context = ReleaseContext(
            agent_id=agent_id,
            record=record,
            release_generation=release_generation,
            notify_release=notify_release,
        )
        if record is not None:
            teardown = self._view._track_child_close(
                record.runtime,
                context.cleanup(self._view),
                name=f"vibe-subagent-release:{agent_id}",
            )
        else:
            teardown = asyncio.create_task(
                context.cleanup(self._view), name=f"vibe-subagent-release:{agent_id}"
            )
            self._view._teardown_tasks.add(teardown)
            teardown.add_done_callback(self._view._teardown_tasks.discard)
        return (
            teardown,
            ReleaseAgentOutcome.EVICTED
            if known_evicted
            else ReleaseAgentOutcome.RELEASED,
        )

    async def release(self, agent_id: str) -> ReleaseAgentOutcome:
        """Close and remove a retained agent or evicted tombstone.

        Unguarded by design: the model tool may release active instances
        (design §3.1). Returns RELEASED for a live agent and EVICTED when
        removing its tombstone. Raises ValueError for unknown agent_id.
        """
        async with self._view._registry_lock:
            teardown, outcome = self._view._prepare_release_locked(agent_id)
        return await self._view._join_release_teardown(teardown, agent_id, outcome)

    async def release_guarded(  # noqa: PLR0911
        self, agent_id: str, expected_run_id: str | None = None
    ) -> ReleaseAgentOutcome:
        """Guarded release for UI callers (design §3.1, §8).

        ATOMIC under the registry lock: attached-root/current-generation
        authority, the expected latest-run identity, a non-active state, and
        no replacement reservation are all verified BEFORE any destructive
        work — never a check-then-release sequence across lock releases. A
        running/finalizing or replacement-reserved record, or a stale
        ``expected_run_id``, returns CONFLICT without stopping or purging
        anything (a stale idle→running race never stops the new run).
        Released, metadata-expired, or unknown handles return UNKNOWN.
        Attached-root authority is verified exactly like the transcript
        authorization sites: the handle must belong to the currently attached
        root session and current generation, so a foreign-parent or
        stale-generation record or tombstone behaves as unknown — never a
        purge. On success the existing release cleanup semantics run:
        results, expiry markers, leases, tokens, and pending-eviction
        authorities are purged, the runtime close is tracked, the reaper is
        rearmed, and the reduced inventory is published.
        """
        async with self._view._registry_lock:
            record = self._view._agent_records.get(agent_id)
            tombstone = self._view._live_tombstone_locked(agent_id, self._view._clock())
            if record is None and tombstone is None:
                # Released, metadata-expired, or never seen: nothing to guard.
                return ReleaseAgentOutcome.UNKNOWN
            root = self._view._root
            identity = (
                (root.agent_loop.session_id, root.agent_loop._session_generation)
                if root is not None
                else None
            )
            if identity is None or self._view._generation_identity != identity:
                # No attached root (or a stale generation snapshot): there is
                # no authority to release under.
                return ReleaseAgentOutcome.UNKNOWN
            if record is not None:
                if (
                    record.parent_session_id != identity[0]
                    or record.root_generation != identity[1]
                ):
                    # Foreign-parent or stale-generation resident record.
                    return ReleaseAgentOutcome.UNKNOWN
                if record.reserved_for_replacement:
                    return ReleaseAgentOutcome.CONFLICT
                if record.state in {_AgentState.RUNNING, _AgentState.FINALIZING}:
                    return ReleaseAgentOutcome.CONFLICT
                latest_run_id = record.latest_run_id or (
                    record.run_history[-1].run_id if record.run_history else None
                )
            else:
                assert tombstone is not None
                if tombstone.parent_identity != identity:
                    # Foreign-parent or stale-generation terminal metadata.
                    return ReleaseAgentOutcome.UNKNOWN
                latest_run_id = tombstone.summary.latest_run_id
            if expected_run_id is not None and latest_run_id != expected_run_id:
                return ReleaseAgentOutcome.CONFLICT
            teardown, outcome = self._view._prepare_release_locked(agent_id)
        return await self._view._join_release_teardown(teardown, agent_id, outcome)

    @staticmethod
    async def join_teardown(
        teardown: asyncio.Task[None], agent_id: str, outcome: ReleaseAgentOutcome
    ) -> ReleaseAgentOutcome:
        """Join a release teardown task, preserving cancellation semantics."""
        cancellation: asyncio.CancelledError | None = None
        while not teardown.done():
            try:
                await asyncio.shield(teardown)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
            except BaseException:
                break
        try:
            await teardown
        except BaseException as exc:
            if cancellation is None:
                raise
            logger.warning(
                "Failed to release background agent agent_id=%s", agent_id, exc_info=exc
            )
        if cancellation is not None:
            raise cancellation
        return outcome
