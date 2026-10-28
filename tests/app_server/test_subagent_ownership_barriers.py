"""Stage-17 (W8b) ownership/access-matrix barrier tests.

Deterministic barriers for the SessionRuntimeRegistry ownership/access
invariants that the W8b area extractions must preserve (plan 17.0a,
design §6.7): result publication/revocation, replacement identity,
release validation, displaced-reaper drain ownership, and execution
commitment. Each test pins one previously uncovered window of its
invariant against the unsplit registry. The invariant -> existing-test
mapping and the covered-window enumeration live in the stage-17 report
and the 17.0a commit message; the displaced-reaper drain-ownership
invariant is already fully covered by the drain/reaper batteries in
test_subagent_lifecycle.py and test_subagents.py and needs no barrier
here.

These are characterization barriers, not bug hunts: they assert current
behavior exactly, must stay green across every extraction move, and must
never motivate a fix — existing scheduling points are preserved.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._sessions import (
    _MAX_STORED_RESULTS,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server._turns import TurnStartAction
from chartreux.core.subagents import ReleaseAgentOutcome, TaskArgs, UnknownAgentError
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    RetaskBackend,
    _background_result,
    _ManualClock,
    _real_reused_background,
    _retention_registry,
    _stored_result,
    _zero_retention_registry,
)
from tests.app_server.test_subagent_management_surface import (
    _attached_registry,
    _resident_record,
)

# ---------------------------------------------------------------------------
# Publication/revocation: expiry is bounded per generation and excludes
# results carrying a pending notification or a wait lease.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_expiry_excludes_pending_notifications_and_wait_leases() -> None:
    """Count-based expiry evicts only unreferenced results past the cap.

    The wait-lease exclusion is pinned by
    test_subagent_management_surface.py::test_wait_lease_protects_result_from_count_expiry;
    this barrier pins the pending-notification exclusion of the same rule:
    a result whose completion notification is still pending is referenced,
    so the oldest evicted result is the oldest unprotected one.
    """
    registry = _retention_registry()
    agent_id = "agent-1"
    protected_notification = (agent_id, "run-00")
    protected_lease = (agent_id, "run-01")
    for index in range(_MAX_STORED_RESULTS + 3):
        run_id = f"run-{index:02}"
        registry._result_store[(agent_id, run_id)] = _stored_result(
            agent_id, run_id, completed_at=float(index)
        )
    registry._pending_notifications.add(protected_notification)
    registry._wait_leases[protected_lease] = 1

    async with registry._registry_lock:
        registry._expire_results_locked()

    evicted = (agent_id, "run-02")
    assert evicted in registry._expired_results
    assert evicted not in registry._result_store
    assert protected_notification in registry._result_store
    assert protected_lease in registry._result_store
    assert len(registry._result_store) == _MAX_STORED_RESULTS + 2
    await registry.drain_children()


# ---------------------------------------------------------------------------
# Replacement identity: the claim is the exact-identity object attached to
# the canonical record, and the capacity slot's transferred flag marks the
# handoff phases (reserved_for_replacement, replacement_owner, and
# slot.transferred stay distinct).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replacement_claim_slot_phases_pin_capacity_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A busy retask's claim carries the predecessor's exact record, run, and
    slot; across the handoff the three identity fields stay distinct: the
    admission clears reserved_for_replacement while keeping the claim as
    owner, the handoff resets slot.transferred, and the new acceptance
    re-marks it.
    """
    backend = RetaskBackend()
    registry, parent, _notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    ctx = InvokeContext(tool_call_id="retask", session_id=parent.session_id)
    handoff_window = asyncio.Event()
    handoff_release = asyncio.Event()
    retask: asyncio.Task[Any] | None = None
    try:
        old = await _background_result(registry, TaskArgs(task="old"), ctx)
        assert old.agent_id is not None
        record = registry._agent_records[old.agent_id]
        await backend.started.wait()
        old_run = record.current_run
        assert old_run is not None
        slot = record.work_slot
        assert slot is not None
        # Phase 1 — the accepted old run owns its capacity slot.
        assert slot.transferred is True
        assert slot in registry._active_work_slots

        # Gate the reuse launch between the handoff's slot reset and the new
        # admission's acceptance: the old run has settled, the admission has
        # cleared the reservation flag, and the slot is unmarked.
        root = registry._root
        assert root is not None
        original_link = root.turns.link_subagent

        async def gated_link(tool_call_id: str, session_id: str) -> None:
            handoff_window.set()
            await handoff_release.wait()
            await original_link(tool_call_id, session_id)

        monkeypatch.setattr(root.turns, "link_subagent", gated_link)
        backend.cleanup_release.set()
        retask = asyncio.create_task(
            _background_result(
                registry,
                TaskArgs(task="new", agent_id=old.agent_id, replace_run=True),
                ctx,
            )
        )
        await asyncio.wait_for(handoff_window.wait(), 5)
        # Phase 2 — handoff reset, pre-acceptance: the claim stays attached by
        # exact identity and carries the predecessor's exact slot; the
        # admission cleared reserved_for_replacement while keeping the claim
        # as owner through the launch acknowledgment, and the handoff reset
        # the slot's transferred flag.
        claim = record.replacement_owner
        assert claim is not None
        assert claim.record is record
        assert claim.run is old_run
        assert claim.slot is slot
        assert claim.slot.transferred is False
        assert record.reserved_for_replacement is False
        assert record.current_run is None
        assert slot in registry._active_work_slots

        handoff_release.set()
        result = await asyncio.wait_for(retask, 5)
        # Phase 3 — acceptance: the new run re-marked the slot transferred and
        # owns it; the reservation fields are cleared by exact identity.
        assert claim.slot.transferred is True
        assert claim.slot is record.work_slot
        assert record.reserved_for_replacement is False
        assert record.replacement_owner is None
        assert slot in registry._active_work_slots
        new_run = record.current_run
        assert new_run is not None and new_run.run_id == result.run_id
        assert record.state is _AgentState.RUNNING

        # Phase 4 — settlement: the accepted execution retires the slot and
        # its result stays retrievable.
        backend.release.set()
        await registry.wait_for_agent(record.agent_id, result.run_id)
        assert slot not in registry._active_work_slots
        stored = await registry.get_agent_result(record.agent_id, result.run_id)
        assert stored is not None and stored.run_id == result.run_id
    finally:
        handoff_release.set()
        backend.release.set()
        backend.cleanup_release.set()
        if retask is not None and not retask.done():
            retask.cancel()
            await asyncio.gather(retask, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# Release validation: UI validation and destructive preparation share one
# registry-lock critical section, the cleanup publishes only while it holds
# the sole notification suppression for the handle.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guarded_release_validates_and_prepares_under_one_lock_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guarded conflict validates without any destructive preparation, and
    a successful release prepares while the same registry lock is held.
    """
    clock = _ManualClock(10)
    registry = _attached_registry(clock)
    record = _resident_record(registry, "agent-1", clock=clock)
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")
    lock_states: list[bool] = []
    original_prepare = registry._prepare_release_locked

    def recording_prepare(
        agent_id: str,
    ) -> tuple[asyncio.Task[None], ReleaseAgentOutcome]:
        lock_states.append(registry._registry_lock.locked())
        return original_prepare(agent_id)

    monkeypatch.setattr(registry, "_prepare_release_locked", recording_prepare)
    try:
        # A conflict validates without any destructive preparation.
        record.state = _AgentState.RUNNING
        assert (
            await registry.release_agent_guarded("agent-1", "run-1")
            is ReleaseAgentOutcome.CONFLICT
        )
        assert lock_states == []
        assert registry._agent_records["agent-1"] is record
        assert ("agent-1", "run-1") in registry._result_store

        # A successful release prepares under the held registry lock.
        record.state = _AgentState.IDLE
        assert (
            await registry.release_agent_guarded("agent-1", "run-1")
            is ReleaseAgentOutcome.RELEASED
        )
        assert lock_states == [True]
        assert "agent-1" not in registry._agent_records
        assert not [key for key in registry._result_store if key[0] == "agent-1"]
    finally:
        await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_publication_requires_the_sole_suppression_owner() -> (
    None
):
    """The release cleanup publishes the reduced inventory only while it
    holds the sole notification suppression for the handle.
    """
    clock = _ManualClock(10)
    registry = _attached_registry(clock)
    _resident_record(registry, "agent-1", clock=clock)
    notify = AsyncMock()
    registry._notify_agents = notify

    # A second suppression owner (as a concurrent retirement or eviction
    # would hold) withholds the release's inventory publication.
    registry._acquire_notification_suppression("agent-1")
    try:
        assert (
            await registry.release_agent_guarded("agent-1", "run-1")
            is ReleaseAgentOutcome.RELEASED
        )
        assert "agent-1" not in registry._agent_records
        notify.assert_not_awaited()
    finally:
        registry._release_notification_suppression("agent-1")
        await registry.drain_children()

    # With the sole owner, the same release publishes the empty inventory.
    registry = _attached_registry(clock)
    _resident_record(registry, "agent-1", clock=clock)
    notify = AsyncMock()
    registry._notify_agents = notify
    assert (
        await registry.release_agent_guarded("agent-1", "run-1")
        is ReleaseAgentOutcome.RELEASED
    )
    await_args = notify.await_args
    assert await_args is not None
    assert await_args.args[0] == []
    await registry.drain_children()


# ---------------------------------------------------------------------------
# Execution commitment: the monitor's finalization is gated on the shared
# committed flag — a monitor cancelled before acceptance stores no result,
# runs no completion transition, and leaves no publication authority.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_commit_monitor_finalization_stores_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch that fails between monitor creation and acceptance aborts the
    prepared turn, and its cancelled monitor finalizes nothing.
    """
    registry, parent, record, args, context = await _real_reused_background()
    generated_run_ids: list[str] = []
    original_new_run_id = SessionRuntimeRegistry._new_run_id

    def recording_new_run_id(agent_id: str) -> str:
        run_id = original_new_run_id(agent_id)
        generated_run_ids.append(run_id)
        return run_id

    monkeypatch.setattr(
        SessionRuntimeRegistry, "_new_run_id", staticmethod(recording_new_run_id)
    )
    original_start = SessionRuntimeRegistry._start_child_turn
    aborted: list[bool] = []

    class _FailingStartAction:
        """A prepared-start action whose acceptance raises."""

        def __init__(self, action: TurnStartAction) -> None:
            self._action = action

        def __call__(self) -> None:
            raise RuntimeError("injected acceptance-tail failure")

        def abort(self) -> None:
            aborted.append(True)
            self._action.abort()

    def failing_start(
        runtime: Any,
        task_args: TaskArgs,
        ctx: InvokeContext,
        session_id: str,
        **_kwargs: Any,
    ) -> tuple[str, Any]:
        turn_id, action = original_start(runtime, task_args, ctx, session_id)
        return turn_id, _FailingStartAction(action)

    monkeypatch.setattr(SessionRuntimeRegistry, "_start_child_turn", failing_start)
    try:
        with pytest.raises(RuntimeError, match="acceptance-tail failure"):
            await _background_result(registry, args, context)
        # The rollback aborted the prepared turn.
        assert aborted == [True]
        # The rollback restored the prior admission state exactly.
        assert len(generated_run_ids) == 1
        attempted_run_id = generated_run_ids[0]
        assert record.current_run is None
        assert record.latest_run_id is None
        assert record.state is _AgentState.IDLE
        assert record.idle_since == 0
        # The monitor finalized nothing: no stored result, no expiry marker,
        # no pending notification, no write token, no latest-run index.
        assert (record.agent_id, attempted_run_id) not in registry._result_store
        assert (record.agent_id, attempted_run_id) not in registry._expired_results
        assert (
            record.agent_id,
            attempted_run_id,
        ) not in registry._pending_notifications
        assert (record.agent_id, attempted_run_id) not in registry._result_write_tokens
        assert record.agent_id not in registry._latest_run_ids
        # No completion transition: the retained instance stays resident and
        # the attempted run is unknown, not expired.
        assert record.agent_id not in registry._evicted_agents
        assert registry._children[record.session_id] is record.runtime
        with pytest.raises(UnknownAgentError):
            await registry.get_agent_result(record.agent_id, attempted_run_id)
    finally:
        await registry.drain_children()
        await parent.aclose()
